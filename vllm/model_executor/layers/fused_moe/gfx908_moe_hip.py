# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 small-M W4A16 MoE: HIP GEMV partials + fused Triton reduces.

For decode-sized batches (M <= 8) the MFMA MoE kernel (fused_moe_kernel_gptq_awq)
runs ~6 MB of expert weight per layer in ~82 us (73 GB/s) at M=1. This path
launches one HIP thread-per-column GEMV per (token, expert) pair with a wide K
split (csrc/gfx908_w4gemv.hip, JIT-built with torch.utils.cpp_extension), then
fuses the split-K sums with silu*mul (w13) and with the routed-weight multiply +
top-k sum (w2). Graph-timed pipeline vs stock: 57 vs 96 us at M=1, 91 vs 137 at
M=2, 157 vs 187 at M=4, parity at M=8, slower at M=16.

Weight layout: the TRITON WNA16 backend's [E, N, K // 2] uint8 (even k in the
low nibble, symmetric zero point 8), viewed as [E, N, K // 8] int32 for the HIP
kernel; scales [E, N, K // group_size] bf16.
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.gfx908_w4a8 import (
    moe_w4a8,
    moe_w4a8_applies,
    shared_arm,
    shared_expert_from_pack,
    shared_take,
    w4a8_enabled,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

MOE_HIP_MAX_TOKENS = 8
_LAYOUT_LOGGED = False
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_w4gemv.hip")


@functools.cache
def _ext():
    """JIT-build (or load from the cache dir) the HIP GEMV extension."""
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP W4 GEMV extension in %s", build_dir)
    return load(
        name="gfx908_w4gemv_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def hip_gemv_available() -> bool:
    try:
        return _ext() is not None
    except Exception as exc:  # hipcc missing etc. -> stock path
        logger.warning_once("gfx908: HIP W4 GEMV unavailable (%s); using stock MoE", exc)
        return False


@triton.jit
def _moe_reduce_silu_mul_kernel(
    part_ptr, out_ptr, N,               # N = full w13 width (gate | up); out width N // 2
    stride_pk, stride_pp, stride_om,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pair = tl.program_id(1)
    half = N // 2
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half
    g = tl.zeros((BLOCK,), dtype=tl.float32)
    u = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        base = part_ptr + s * stride_pk + pair * stride_pp
        g += tl.load(base + offs, mask=mask, other=0.0)
        u += tl.load(base + half + offs, mask=mask, other=0.0)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + pair * stride_om + offs, y.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def _moe_reduce_weighted_sum_kernel(
    part_ptr, w_ptr, out_ptr, N,        # part [SPLIT_K, M*TOPK, N]; w [M*TOPK] routed weights
    stride_pk, stride_pp, stride_om,
    TOPK: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr, MUL_W: tl.constexpr,
):
    pid = tl.program_id(0)
    token = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in range(TOPK):
        pair = token * TOPK + t
        v = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in range(SPLIT_K):
            v += tl.load(part_ptr + s * stride_pk + pair * stride_pp + offs, mask=mask, other=0.0)
        if MUL_W:
            v = v * tl.load(w_ptr + pair).to(tl.float32)
        acc += v
    tl.store(out_ptr + token * stride_om + offs, acc.to(out_ptr.type.element_ty), mask=mask)



# (SK1, BN1, SK2, BN2) per token count, graph-timed (bench_hip.py --moe-only).
_CFG = {1: (8, 64, 1, 128), 2: (8, 64, 1, 128), 4: (40, 128, 5, 256), 8: (40, 128, 5, 256)}


def _cfg_for(M: int):
    for m in (1, 2, 4, 8):
        if M <= m:
            return _CFG[m]
    return _CFG[8]


@functools.cache
def _row_index(M: int, topk: int, device: torch.device):
    """Cached (row -> token, row -> self) int32 index tensors for M*topk pairs."""
    P = M * topk
    row_self = torch.arange(P, device=device, dtype=torch.int32)
    return row_self // topk, row_self


def gfx908_moe_hip(
    output: torch.Tensor,        # [M, K] bf16 (written)
    hidden_states: torch.Tensor,  # [M, K] bf16
    w1: torch.Tensor,             # [E, N1, K//2] uint8
    w2: torch.Tensor,             # [E, K, (N1//2)//2] uint8
    w1_scale: torch.Tensor,       # [E, N1, K//gs]
    w2_scale: torch.Tensor,       # [E, K, (N1//2)//gs]
    topk_weights: torch.Tensor,   # [M, topk] fp32
    topk_ids: torch.Tensor,       # [M, topk]
    group_size: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    ext = _ext()
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    P = M * topk
    N1 = w1.shape[1]
    K2 = N1 // 2
    dev = hidden_states.device
    global _LAYOUT_LOGGED
    if not _LAYOUT_LOGGED:
        _LAYOUT_LOGGED = True
        logger.info(
            "gfx908 MoE HIP layout: x %s %s | w1 %s %s stride %s | w1_scale %s %s stride %s | "
            "w2 %s %s stride %s | w2_scale %s %s | topk_ids %s %s | topk_w %s %s | gs=%d",
            tuple(hidden_states.shape), hidden_states.dtype, tuple(w1.shape), w1.dtype, w1.stride(),
            tuple(w1_scale.shape), w1_scale.dtype, w1_scale.stride(), tuple(w2.shape), w2.dtype,
            w2.stride(), tuple(w2_scale.shape), w2_scale.dtype, tuple(topk_ids.shape), topk_ids.dtype,
            tuple(topk_weights.shape), topk_weights.dtype, group_size,
        )
    assert hidden_states.is_contiguous() and w1.is_contiguous() and w2.is_contiguous()
    assert w1_scale.is_contiguous() and w2_scale.is_contiguous()
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8, (w1.dtype, w2.dtype)
    assert w1.shape[2] * 2 == K, (w1.shape, K)
    assert w2.shape[1] == K and w2.shape[2] * 2 == K2, (w2.shape, K, K2)
    assert w1_scale.shape == (w1.shape[0], N1, K // group_size), (w1_scale.shape, K, group_size)
    assert w2_scale.shape == (w2.shape[0], K, K2 // group_size), (w2_scale.shape, K2, group_size)
    assert w1_scale.dtype == hidden_states.dtype == torch.bfloat16, (w1_scale.dtype, hidden_states.dtype)
    assert topk_ids.shape == (M, topk) and topk_weights.shape == (M, topk)
    sk1, bn1, sk2, bn2 = _cfg_for(M)
    sk1 = max(1, min(sk1, K // 32))
    sk2 = max(1, min(sk2, K2 // 32))
    row_expert = topk_ids.reshape(-1)
    if row_expert.dtype != torch.int32:
        row_expert = row_expert.to(torch.int32)
    row_token, row_self = _row_index(M, topk, dev)
    w1_i = w1.view(torch.int32)
    w2_i = w2.view(torch.int32)

    # VLLM_GFX908_SHARED_AS_EXPERT=1: the shared expert ran just before us and handed over its
    # (input, repacked weights) instead of computing itself.  Pop it unconditionally so a
    # hand-off can never be dropped: either it is folded in as expert #E below, or it is computed
    # separately here and added to our output (same result, the pre-fusion launch count).
    pending = shared_take()
    shared = None
    if pending is not None and pending[0].data_ptr() == hidden_states.data_ptr() and (
        pending[0].shape == hidden_states.shape
    ):
        shared = pending[1]

    if w4a8_enabled() and moe_w4a8_applies(K, N1, group_size, hidden_states.dtype):
        # VLLM_GFX908_W4A8=1: int8-activation dot4 GEMVs, no split-K (gfx908_w4a8.py)
        out = moe_w4a8(
            output, hidden_states, w1_i, w2_i, w1_scale, w2_scale, topk_weights,
            row_token, row_self, row_expert, mul_routed_weight, shared=shared,
        )
        if out is not None:
            if shared is None:
                shared_arm(w1.shape[0], N1, K, K2, mul_routed_weight)
                if pending is not None:
                    out += shared_expert_from_pack(pending[0], pending[1])
            return out
        # the fused hand-off could not be taken this call: routed only, shared added separately
        out = moe_w4a8(
            output, hidden_states, w1_i, w2_i, w1_scale, w2_scale, topk_weights,
            row_token, row_self, row_expert, mul_routed_weight,
        )
        if pending is not None:
            out += shared_expert_from_pack(pending[0], pending[1])
        return out

    part1 = torch.empty((sk1, P, N1), dtype=torch.float32, device=dev)
    ext.w4gemv(hidden_states, w1_i, w1_scale, row_token, row_expert, part1, group_size, 1, bn1)
    inter = torch.empty((P, K2), dtype=hidden_states.dtype, device=dev)
    rb = 256
    _moe_reduce_silu_mul_kernel[(triton.cdiv(K2, rb), P)](
        part1, inter, N1, part1.stride(0), part1.stride(1), inter.stride(0),
        SPLIT_K=sk1, BLOCK=rb,
    )
    part2 = torch.empty((sk2, P, K), dtype=torch.float32, device=dev)
    ext.w4gemv(inter, w2_i, w2_scale, row_self, row_expert, part2, group_size, 1, bn2)
    rb2 = 1024
    _moe_reduce_weighted_sum_kernel[(triton.cdiv(K, rb2), M)](
        part2, topk_weights.reshape(-1).to(torch.float32), output, K,
        part2.stride(0), part2.stride(1), output.stride(0),
        TOPK=topk, SPLIT_K=sk2, BLOCK=rb2, MUL_W=mul_routed_weight,
    )
    if pending is not None:
        # stock split-K routed path: the deferred shared expert still has to be honoured
        output += shared_expert_from_pack(pending[0], pending[1])
    return output
