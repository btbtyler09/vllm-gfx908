# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 small-M top-k softmax router (Triton).

ops.topk_softmax (topkGating<8, 512, ...>) costs ~24 us in-graph on MI100 for
1..16 tokens over 512 experts. One program per token: fp32 softmax, k passes
of max + lowest-index tie-break, optional renormalization over the selected k.
Same semantics as ops.topk_softmax(renormalize=...).
"""

import torch

from vllm.triton_utils import tl, triton

TOPK_MAX_TOKENS = 16


@triton.jit
def _topk_softmax_kernel(
    logits_ptr, w_ptr, ids_ptr, pad_ptr, E, stride_lm,
    TOPK: tl.constexpr, RENORM: tl.constexpr, BLOCK_E: tl.constexpr, HAS_PAD: tl.constexpr,
):
    m = tl.program_id(0)
    if HAS_PAD:
        is_pad = tl.load(pad_ptr + m).to(tl.int32)
    else:
        is_pad = 0
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    x = tl.load(logits_ptr + m * stride_lm + offs, mask=mask, other=-float("inf")).to(tl.float32)
    mx = tl.max(x, axis=0)
    ex = tl.where(mask, tl.exp(x - mx), 0.0)
    p = ex / tl.sum(ex, axis=0)
    cand = p
    sel_sum = 0.0
    for k in range(TOPK):
        v = tl.max(cand, axis=0)
        idx = tl.min(tl.where(cand == v, offs, BLOCK_E), axis=0)
        tl.store(w_ptr + m * TOPK + k, v)
        # stock kernel writes -1 for cudagraph-padded rows
        tl.store(ids_ptr + m * TOPK + k, tl.where(is_pad != 0, -1, idx).to(tl.int32))
        sel_sum += v
        cand = tl.where(offs == idx, -1.0, cand)
    if RENORM:
        tl.debug_barrier()
        for k in range(TOPK):
            w = tl.load(w_ptr + m * TOPK + k)
            tl.store(w_ptr + m * TOPK + k, w / sel_sum)


def gfx908_topk_softmax(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    gating_output: torch.Tensor, renormalize: bool,
    is_padding: torch.Tensor | None = None,
) -> None:
    M, E = gating_output.shape
    _topk_softmax_kernel[(M,)](
        gating_output, topk_weights, topk_ids,
        is_padding if is_padding is not None else topk_ids, E, gating_output.stride(0),
        TOPK=topk_ids.shape[1], RENORM=renormalize, BLOCK_E=triton.next_power_of_2(E),
        HAS_PAD=is_padding is not None,
    )
