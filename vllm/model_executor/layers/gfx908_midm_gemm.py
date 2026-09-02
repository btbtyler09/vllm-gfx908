# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 mid-M bf16/fp16 GEMM (5 <= M <= 64): Triton split-K tl.dot kernel.

rocBLAS on MI100 picks pathological kernels for the small-N / long-K shapes
that dominate a batched decode step of Qwen4Exp (graph-timed, mb_midm.py,
2026-09-02): hyper-connection mix_down (N=320, K=10240) runs at 117 us
(112 GB/s) for any M, the MoE router (N=512, K=2560) at 69 us (38 GB/s), the
QSA indexer (N=640) at 69 us, GDN in_proj_qkvz (N=4096) at ~70 us. The skinny
kernels (LLMM1 / wvSplitK) only cover M <= 4. This kernel writes fp32 partials
over a K split and reduces them in a second launch; measured 16-34 us for
mix_down, 7-15 us for the router, 33-37 us for in_proj_qkvz at M <= 16.

Weight layout is the stock [N, K] row-major nn.Linear layout (K contiguous).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gfx908_midm_splitk_kernel(
    a_ptr,
    w_ptr,
    part_ptr,
    M,
    N,
    K,
    stride_am,
    stride_wn,
    stride_pk,
    stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_k = tl.cdiv(K, BLOCK_K)
    per = tl.cdiv(n_k, SPLIT_K)
    k0 = pid_k * per
    for kt in range(k0, tl.minimum(k0 + per, n_k)):
        offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mk[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=(offs_n[:, None] < N) & mk[None, :],
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(w))
    p = part_ptr + pid_k * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :]
    tl.store(p, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _gfx908_splitk_reduce_kernel(
    part_ptr,
    c_ptr,
    M,
    N,
    stride_pk,
    stride_pm,
    stride_cm,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_pk + m * stride_pm + offs, mask=mask, other=0.0
        )
    tl.store(c_ptr + m * stride_cm + offs, acc.to(c_ptr.type.element_ty), mask=mask)


MIDM_MIN_M = 5
MIDM_MAX_M = 64


def midm_gemm_applies(n_rows: int, out_features: int, in_features: int) -> bool:
    """Shapes where the split-K kernel beat rocBLAS on MI100 (mb_midm.py)."""
    if not (MIDM_MIN_M <= n_rows <= MIDM_MAX_M):
        return False
    if in_features % 8 != 0 or out_features < 64:  # N=24 in_proj_ba: rocBLAS wins
        return False
    # Small-N or long-K: rocBLAS is 2-7x slower at every M <= 64.
    if out_features <= 1024:
        return True
    # Wide-N, short-K (in_proj_qkvz, hc_mix_up): ~2x at M <= 16, parity after.
    return n_rows <= 16 and out_features >= 4096


def _config(n_rows: int, N: int, K: int) -> tuple[int, int, int, int]:
    block_m = max(16, triton.next_power_of_2(n_rows))
    if N <= 1024:
        block_n = 64 if n_rows <= 16 else 16
        block_k = 64 if K >= 8192 else 32
    elif K <= 512:
        block_n, block_k = 64, 32
    else:
        block_n, block_k = 64, 32
    n_tiles = triton.cdiv(N, block_n)
    k_tiles = triton.cdiv(K, block_k)
    split_k = max(1, min(k_tiles, 32, round(256 / n_tiles)))
    return block_m, block_n, block_k, split_k


def gfx908_midm_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """C = x @ weight.T (+ bias) for [M, K] x [N, K] bf16/fp16, 5 <= M <= 64."""
    x2 = x.reshape(-1, x.size(-1))
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M, K = x2.shape
    N = weight.shape[0]
    block_m, block_n, block_k, split_k = _config(M, N, K)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=x2.device)
    _gfx908_midm_splitk_kernel[(triton.cdiv(N, block_n), split_k)](
        x2,
        weight,
        part,
        M,
        N,
        K,
        x2.stride(0),
        weight.stride(0),
        part.stride(0),
        part.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
    )
    out = torch.empty((M, N), dtype=x2.dtype, device=x2.device)
    rb = 1024
    _gfx908_splitk_reduce_kernel[(triton.cdiv(N, rb), M)](
        part,
        out,
        M,
        N,
        part.stride(0),
        part.stride(1),
        out.stride(0),
        SPLIT_K=split_k,
        BLOCK=rb,
    )
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], N)
