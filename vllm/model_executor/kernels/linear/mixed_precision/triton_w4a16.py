# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W4A16 GEMM kernel for ROCm MI300.

Implements fused int4-weight dequantization + fp16 GEMM in a single kernel,
using GPTQ sequential packing (8 int4 values per int32, shifts [0,4,...,28]).
Plugs into the MPLinearKernel selection system and is preferred over
MarlinLinearKernel/ExllamaLinearKernel on ROCm.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K, N//8]  int32  — rows=K (input), cols=N//8 (N is packed)
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//8]  int32  (optional; None for symmetric uint4b8)

Checkpoint layout from compressed_tensors_wNa16 create_weights:
  weight_packed:     [N, K//8]  int32  (output_dim=0, input_dim=1, packed_dim=1)
  weight_scale:      [N, K//G]  fp16   (output_dim=0, input_dim=1)
  weight_zero_point: [N//8, K//G]  int32 (output_dim=0, packed_dim=0)
"""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
    scalar_types.uint4,  # asymmetric with explicit zeros
]


@triton.jit
def triton_w4a16_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [K, N//8]  int32 packed 4-bit weights (N is the packed dim)
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,  # [K//G, N//8]  int32 packed zeros (unused when HAS_ZP=False)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,  # stride in b along the packed N//8 dim
    stride_cm,
    stride_cn,
    # Quantization parameters
    group_size,
    # Whether explicit zero points are provided
    HAS_ZP: tl.constexpr,
    # Zero bias used when HAS_ZP is False (e.g. 8 for uint4b8)
    ZP_BIAS: tl.constexpr,
    # Block sizes (tuned for MI300 wavefront=64)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM: C[M,N] = A[M,K] @ dequant(B)[K,N]

    B is stored as [K, N//8] int32 using GPTQ sequential packing:
      each int32 packs 8 consecutive N-values at bit offsets [0,4,8,12,16,20,24,28].

    Dequant: w_fp = (w_int4 - zero) * scale
      HAS_ZP=True:  zero is loaded from zeros_ptr and unpacked
      HAS_ZP=False: zero = ZP_BIAS constant (e.g. 8 for uint4b8 symmetric)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # b/zeros are stored with N packed: N//8 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    # GPTQ sequential shifts tiled across BLOCK_N:
    #   [0,4,8,...,28] repeating for every group of 8 N-values.
    # Build 1D shifts_1d of length BLOCK_N: column j gets shift (j % 8) * 4.
    shifts_row = tl.arange(0, 8) * 4  # [8]
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))  # [BLOCK_N]
    # Broadcast to [BLOCK_K, BLOCK_N] for weight unpacking
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    # Scales column offsets: full N-width (one scale per output neuron)
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_K, BLOCK_N//8] int32 ----
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack int4 weights → [BLOCK_K, BLOCK_N] ----
        # tl.interleave(x, x) doubles the last dim by interleaving.
        # Starting from [BLOCK_K, BLOCK_N//8], three interleaves give
        # [BLOCK_K, BLOCK_N], where each int32 is replicated 8 times.
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        # Extract the correct 4-bit nibble for each output column
        b = (b >> shifts) & 0xF

        # ---- Compute scale/zero group row index ----
        g_idx = (k_start * BLOCK_K) // group_size

        # ---- Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N] ----
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        # ---- Load / compute zeros ----
        if HAS_ZP:
            # Load packed zeros row: [BLOCK_N//8] int32
            zero_offset = g_idx * (N // 8) + offs_bn
            zero_mask = offs_bn < N // 8
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            # Unpack to [BLOCK_N] using same interleave+shift pattern
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        # ---- Dequantize: (w - zero) * scale ----
        b_fp = (b - z).to(a.dtype) * scales

        # ---- Accumulate ----
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


# --------------------------------------------------------------------------- #
# gfx908 small-M path: no-MFMA GEMV with K-split (fp32 partials + reduce).
# The MFMA kernel above runs BLOCK_M=16 tiles at M=1 with no split-K, so a
# K=2560 -> N=320 projection launches 5 programs on 120 CUs and takes ~75 us;
# this path launches N/BLOCK_N * SPLIT_K programs and reads each weight byte
# once. Symmetric (uint4b8) only; asymmetric falls through to the MFMA kernel.
# --------------------------------------------------------------------------- #
@triton.jit
def triton_w4a16_gemv_partial_kernel(
    a_ptr, b_ptr, scales_ptr, part_ptr, M, N, K,
    stride_am, stride_bk, stride_pk, stride_pm,
    group_size,
    ZP_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    offs_m = tl.arange(0, BLOCK_M)
    shifts_row = tl.arange(0, 8) * 4
    shifts_1d = tl.reshape(
        tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8)), (BLOCK_N,)
    )
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_split = tl.cdiv(n_k_tiles, SPLIT_K)
    k_tile_start = pid_k * tiles_per_split
    k_tile_end = tl.minimum(k_tile_start + tiles_per_split, n_k_tiles)
    for kt in range(k_tile_start, k_tile_end):
        offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_packed = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :],
            mask=mask_k[:, None] & (offs_bn[None, :] < N // 8),
            other=0,
        )
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = ((b >> shifts) & 0xF) - ZP_BIAS
        g_idx = (kt * BLOCK_K) // group_size
        scales = tl.load(
            scales_ptr + g_idx * N + offs_n, mask=offs_n < N, other=0.0
        ).to(tl.float32)
        prod = a[:, :, None] * b.to(tl.float32)[None, :, :]
        acc += tl.sum(prod, axis=1) * scales[None, :]
    p_ptrs = part_ptr + pid_k * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :]
    tl.store(p_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def triton_w4a16_splitk_reduce_kernel(
    part_ptr, c_ptr, M, N, stride_pk, stride_pm, stride_cm,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * stride_pk + m * stride_pm + offs, mask=mask, other=0.0)
    tl.store(c_ptr + m * stride_cm + offs, acc.to(c_ptr.type.element_ty), mask=mask)


_GFX908_GEMV_FLAG: bool | None = None


def _gfx908_gemv_enabled() -> bool:
    global _GFX908_GEMV_FLAG
    if _GFX908_GEMV_FLAG is None:
        import os

        from vllm.platforms.rocm import on_gfx908

        _GFX908_GEMV_FLAG = on_gfx908() and os.environ.get(
            "VLLM_GFX908_W4_GEMV", "1"
        ) == "1"
    return _GFX908_GEMV_FLAG


_GFX908_GEMV_MAX_M = 16
# (K, N) -> ((BLOCK_N, SPLIT_K) for M == 1, (BLOCK_N, SPLIT_K) for 1 < M <= 16).
# Graph-timed on MI100 (mb_w4a16_gemv.py / mb_gemv_bm16.py, 2026-09-02) for the
# Qwen3.8-Flash-Next TP4 dense projections; BLOCK_M is 1 at M == 1 and 16
# otherwise (the 2/4/8-row tiles compile much worse than the 16-row tile).
_GFX908_GEMV_TABLE = {
    (2560, 320): ((64, 20), (64, 20)),  # shared_expert gate_up
    (160, 2560): ((128, 5), (64, 5)),  # shared_expert down
    (2560, 3584): ((256, 16), (64, 10)),  # QSA qkv(+gate)
    (1536, 2560): ((256, 32), (64, 10)),  # QSA o_proj
}


def _gfx908_gemv_config(M, K, N, k_tiles):
    entry = _GFX908_GEMV_TABLE.get((K, N))
    if entry is not None:
        block_n, split_k = entry[0] if M == 1 else entry[1]
    else:
        block_n = 64
        n_tiles = triton.cdiv(N, block_n)
        split_k = max(1, min(k_tiles, round(128 / n_tiles)))
    return block_n, min(split_k, k_tiles)


def _gfx908_w4a16_gemv(a, b_q, scales, group_size, zp_bias):
    M, K = a.shape
    N = b_q.shape[1] * 8
    if M <= 8 and group_size == 32 and zp_bias == 8:
        # VLLM_GFX908_W4A8=1: int8-activation dot4 GEMV for K in {2560, 1536}
        # (fused_moe/gfx908_w4a8.py); None -> not applicable / weight not yet repacked
        from vllm.model_executor.layers.fused_moe.gfx908_w4a8 import (
            dense_w4a8_gemv,
            w4a8_enabled,
        )

        if w4a8_enabled():
            c = dense_w4a8_gemv(a, b_q, scales)
            if c is not None:
                return c
    BLOCK_K = min(32, group_size)
    BLOCK_M = 1 if M == 1 else 16
    k_tiles = triton.cdiv(K, BLOCK_K)
    block_n, split_k = _gfx908_gemv_config(M, K, N, k_tiles)
    n_tiles = triton.cdiv(N, block_n)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    triton_w4a16_gemv_partial_kernel[(n_tiles, split_k)](
        a, b_q, scales, part, M, N, K,
        a.stride(0), b_q.stride(0), part.stride(0), part.stride(1),
        group_size, ZP_BIAS=zp_bias, BLOCK_M=BLOCK_M, BLOCK_N=block_n,
        BLOCK_K=BLOCK_K, SPLIT_K=split_k,
    )
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    RB = 1024
    triton_w4a16_splitk_reduce_kernel[(triton.cdiv(N, RB), M)](
        part, c, M, N, part.stride(0), part.stride(1), c.stride(0),
        SPLIT_K=split_k, BLOCK=RB,
    )
    return c


def triton_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [K, N//8] int32
    scales: torch.Tensor,  # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,  # [K//G, N//8] int32, or None
    group_size: int,
    zp_bias: int = 8,  # bias for uint4b8 when qzeros is None
) -> torch.Tensor:
    """
    Fused W4A16 GEMM using GPTQ-packed int4 weights.

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [K, N//8], int32 (GPTQ sequential).
        scales:     Per-group scales [K//G, N], same dtype as a.
        qzeros:     Per-group packed zero points [K//G, N//8] int32, or None
                    for symmetric quantization (uses zp_bias instead).
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero used when qzeros is None (default 8 for uint4b8).

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1] * 8

    assert b_q.shape == (K, N // 8), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // 8})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    if (
        qzeros is None
        and M <= _GFX908_GEMV_MAX_M
        and current_platform.is_rocm()
        and _gfx908_gemv_enabled()
    ):
        return _gfx908_w4a16_gemv(a, b_q, scales, group_size, zp_bias)

    # Large-M escape: dequantise once into a reusable scratch and let the stock
    # bf16 dispatch (rocBLAS) do the GEMM.  Off by default; see
    # _w4a16_dequant_escape.  Placed after the GEMV gate so the decode path is
    # untouched, and it returns None whenever it does not apply.
    if current_platform.is_rocm():
        escaped = _w4a16_dequant_escape(
            a, b_q, scales, qzeros, group_size, zp_bias
        )
        if escaped is not None:
            return escaped

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    # Provide a dummy pointer when HAS_ZP=False (Triton requires a valid ptr)
    zeros_ptr = qzeros if has_zp else b_q

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        try:
            from vllm.platforms.rocm import on_gfx908
            is_gfx908 = on_gfx908()
        except Exception:
            is_gfx908 = False

        if on_gfx1x():
            # Tuned for RDNA 3.5 (gfx1151, 40 CUs, 32-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
        elif is_gfx908:
            # gfx908: 120 CUs, 64-wide wavefronts, ~1.2 TB/s HBM2. Mirrors
            # the W8A16 gfx908 tiles: favor more N-tiles to saturate CUs at
            # low M (used by the GPTQ4 dual dispatch for prefill M).
            if M <= 16:
                BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32
            elif M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        else:
            # Tuned for MI300 (gfx942, 304 CUs, 64-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # The kernel loads scales/zeros for a single group per BLOCK_K tile
    # (one g_idx per iteration). If BLOCK_K > group_size, rows at the tail
    # of the tile dequantize with the wrong group's scales, silently
    # corrupting the output. Clamp BLOCK_K to group_size to keep one
    # scale group per tile.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w4a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def _gfx908_w4a16_gemm_op(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    return triton_w4a16_gemm(a, b_q, scales, None, group_size, zp_bias)


def _gfx908_w4a16_gemm_op_fake(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    return a.new_empty((a.shape[0], b_q.shape[1] * 8))


# Opaque to torch.compile: the small-M GEMV / tile selection in
# triton_w4a16_gemm branches on M, and under dynamo M is symbolic, so the
# branch was specialized once at trace time (large M) and frozen for every
# cudagraph size — the QSA projections ran the MFMA kernel with a large-M
# tile config at M=1 (162 us) while the eager MoE region took the GEMV.
direct_register_custom_op(
    op_name="gfx908_w4a16_gemm",
    op_func=_gfx908_w4a16_gemm_op,
    fake_impl=_gfx908_w4a16_gemm_op_fake,
)


@triton.jit
def triton_w4a16_dequant_kernel(
    b_ptr,           # [K, N//8] int32 packed weights (8 nibbles per int32)
    scales_ptr,      # [K//G, N] fp16/bf16
    zeros_ptr,       # [K//G, N//8] int32 packed zeros (unused when HAS_ZP=False)
    out_ptr,         # [K, N] (TRANSPOSE_OUT=False) or [N, K] (True)
    N, K,
    stride_bk, stride_bn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N8: tl.constexpr,
    TRANSPOSE_OUT: tl.constexpr,
):
    """Dequantize a [K, N//8]-packed GPTQ4 weight to dense [K, N] (or [N, K]).

    Dequant math is identical to triton_w4a16_gemm_kernel:
        w = (nibble - zero).to(scales.dtype) * scale
    with `zero` either the unpacked qzeros nibble (HAS_ZP) or the constant
    ZP_BIAS (symmetric uint4b8), so the dequant+hgemm route sees bit-identical
    weight values to the fused MFMA route.
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n8 = pid_n * BLOCK_N8 + tl.arange(0, BLOCK_N8)
    mask_k = offs_k < K
    mask_n8 = offs_n8 < (N // 8)

    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n8[None, :] * stride_bn
    b_packed = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n8[None, :], other=0)

    # Unpack nibbles: 3 interleaves expand the last dim by 8
    b = tl.interleave(b_packed, b_packed)
    b = tl.interleave(b, b)
    b = tl.interleave(b, b)
    shifts_row = tl.arange(0, 8) * 4
    shifts_1d = tl.reshape(
        tl.broadcast_to(shifts_row[None, :], (BLOCK_N8, 8)), (BLOCK_N8 * 8,)
    )
    b = (b >> shifts_1d[None, :]) & 0xF  # [BLOCK_K, BLOCK_N]

    offs_n = pid_n * BLOCK_N8 * 8 + tl.arange(0, BLOCK_N8 * 8)
    mask_n = offs_n < N
    g_idx = offs_k // group_size
    scales = tl.load(
        scales_ptr + g_idx[:, None] * N + offs_n[None, :],
        mask=mask_k[:, None] & mask_n[None, :],
        other=1.0,
    )

    if HAS_ZP:
        # zeros are packed exactly like the weights: [K//G, N//8] int32
        z_packed = tl.load(
            zeros_ptr + g_idx[:, None] * (N // 8) + offs_n8[None, :],
            mask=mask_k[:, None] & mask_n8[None, :],
            other=0,
        )
        z = tl.interleave(z_packed, z_packed)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = (z >> shifts_1d[None, :]) & 0xF
    else:
        z = tl.full((BLOCK_K, BLOCK_N8 * 8), ZP_BIAS, dtype=tl.int32)

    w = (b - z).to(scales.dtype) * scales
    if TRANSPOSE_OUT:
        out_ptrs = out_ptr + offs_n[:, None] * K + offs_k[None, :]
        tl.store(
            out_ptrs, tl.trans(w), mask=mask_n[:, None] & mask_k[None, :]
        )
    else:
        out_ptrs = out_ptr + offs_k[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, w, mask=mask_k[:, None] & mask_n[None, :])


def triton_w4a16_dequant(
    b_q: torch.Tensor,       # [K, N//8] int32
    scales: torch.Tensor,    # [K//G, N] fp16/bf16
    group_size: int,
    zp_bias: int = 8,
    qzeros: torch.Tensor | None = None,  # [K//G, N//8] int32 (asymmetric)
    out: torch.Tensor | None = None,     # optional pre-allocated [K, N] scratch
    transpose_out: bool = False,         # write [N, K] instead of [K, N]
) -> torch.Tensor:
    """Dequantize the repacked GPTQ4 weight to dense [K, N] in scales.dtype.

    Used by the gfx908 GPTQ4 dual dispatch and by the large-M dequant escape in
    TritonW4A16LinearKernel: at high M, dequant-once + rocBLAS hgemm beats the
    fused MFMA kernel whose BLOCK_K is clamped to group_size (=32 for GS32
    checkpoints).

    `out` lets the caller pass a reusable scratch buffer (shape [K, N], or
    [N, K] when transpose_out) so the escape does not allocate per call.
    """
    assert b_q.is_contiguous() and scales.is_contiguous()
    K = b_q.shape[0]
    N = b_q.shape[1] * 8
    shape = (N, K) if transpose_out else (K, N)
    if out is None:
        out = torch.empty(shape, dtype=scales.dtype, device=b_q.device)
    else:
        assert out.shape == shape and out.dtype == scales.dtype
        assert out.is_contiguous()
    has_zp = qzeros is not None
    BLOCK_K, BLOCK_N8 = 32, 16
    grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N // 8, BLOCK_N8))
    triton_w4a16_dequant_kernel[grid](
        b_q, scales, qzeros if has_zp else b_q, out,
        N, K,
        b_q.stride(0), b_q.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_K=BLOCK_K,
        BLOCK_N8=BLOCK_N8,
        TRANSPOSE_OUT=transpose_out,
    )
    return out


# --------------------------------------------------------------------------- #
# gfx908 large-M dequant escape (VLLM_GFX908_W4_DEQUANT_LARGE_M=1, default off)
#
# The fused W4A16 MFMA kernel is capped at BLOCK_K = group_size (32 for the
# GS32 checkpoints) and has no L2 swizzle, so at prefill M it runs at 17-36
# TFLOP/s against a 92.3 bf16-MFMA peak.  Above a threshold it is cheaper to
# materialise the weight once into a bf16 scratch and hand the GEMM to rocBLAS,
# which reaches 40-56 TFLOP/s on the same shapes.  Mirrors exllama.py's
# `dequant_mthresh` escape for the GPTQ8 dual dispatch.
#
# The scratch holds the weight **transposed**, `[N, K]`, so the GEMM is
# `F.linear` (the "NT" layout every other vLLM linear uses) rather than
# `torch.mm` on a `[K, N]` weight.  Graph-timed on MI100 (agents/w4_dequant,
# 2026-09-04) that choice is what makes the escape win: on the narrow shared
# expert shapes rocBLAS picks a much worse kernel for the `[K, N]` "NN" form
# (shared gate_up 2560x320 at M=1024: 145 us NN vs 49 us NT), and NN loses to
# the fused kernel outright at several M.
#
# The transposed store costs 6-25% more in the dequant kernel than the plain
# `[K, N]` store; the dequant is 1-5% of the call either way.
#
# At M >= 256 `rocm_unquantized_gemm_gfx908` provably terminates in `F.linear`
# for every dense W4 shape in this model (wvSplitK needs n<=4, LLMM1 n==1,
# the Triton mid-M path 5<=M<=64, the AITER whitelist does not contain these
# shapes, and the einsum path needs weight.shape[0] <= 8), so calling F.linear
# here is the same rocBLAS kernel with one fewer opaque hop.  It is also the
# safe choice: the scratch is a *mutable* buffer whose contents change every
# call, and the gfx908 W8A16 registry inside that dispatch caches int8 copies
# keyed on `weight.data_ptr()` for a shape set that includes (2560, 1536) --
# exactly the QSA o_proj weight in `[N, K]` form.  Today its M gates
# (W8A16_MAX_M=4, MFMA_MAX_M=64) make that unreachable at M >= 256, but
# handing it a scratch buffer at all is a trap worth not setting.
#
# The scratch is a single flat buffer per (device, dtype), grown on demand to
# the largest K*N seen, and is allocated on the first *eager* call only: the
# escape is skipped entirely while a CUDA graph is being captured, so no
# allocation and no cross-replay buffer aliasing can happen inside a graph.
# Prefill runs eager on gfx908, which is the only regime the escape targets.
# --------------------------------------------------------------------------- #
_W4_DEQUANT_LARGE_M_FLAG: bool | None = None
_W4_DEQUANT_MIN_M: int | None = None
_W4_DEQUANT_MIN_K: int | None = None
_W4_DEQUANT_LAYOUT: str | None = None
_W4_DEQUANT_SCRATCH: dict = {}


def _w4_dequant_large_m_enabled() -> bool:
    global _W4_DEQUANT_LARGE_M_FLAG
    if _W4_DEQUANT_LARGE_M_FLAG is None:
        import os

        try:
            from vllm.platforms.rocm import on_gfx908

            is_gfx908 = on_gfx908()
        except Exception:
            is_gfx908 = False
        _W4_DEQUANT_LARGE_M_FLAG = is_gfx908 and (
            os.environ.get("VLLM_GFX908_W4_DEQUANT_LARGE_M", "0") == "1"
        )
    return _W4_DEQUANT_LARGE_M_FLAG


def _w4_dequant_min_m() -> int:
    global _W4_DEQUANT_MIN_M
    if _W4_DEQUANT_MIN_M is None:
        import os

        try:
            _W4_DEQUANT_MIN_M = int(
                os.environ.get("VLLM_GFX908_W4_DEQUANT_MIN_M", "256")
            )
        except ValueError:
            _W4_DEQUANT_MIN_M = 256
    return _W4_DEQUANT_MIN_M


def _w4_dequant_min_k() -> int:
    """Smallest K the escape will fire for.  0 (default) = no K gate.

    Exists for the shared-expert down projection, K=160: its packed weight is
    0.2 MB, so it is L2-resident and the fused kernel already runs it at
    23-26 TFLOP/s -- the best of the four dense shapes here -- which leaves the
    escape little to win.  Graph-timed on MI100 it still wins at M=256 (11.9 vs
    16.8 us), M=1024 (wash) and M=8192 (135.0 vs 251.9), and loses only at
    M=2048 (71.8 vs 65.7 = -6 us/call, -0.3 ms across the 48 layers), so it is
    included by default: at the 7840-token chunk this model actually prefills
    with, that shape is worth ~5.6 ms/pass.  Set
    VLLM_GFX908_W4_DEQUANT_MIN_K=256 to exclude it for a short-prompt-dominated
    workload, where its per-call extra launch is a larger share of the cost.
    """
    global _W4_DEQUANT_MIN_K
    if _W4_DEQUANT_MIN_K is None:
        import os

        try:
            _W4_DEQUANT_MIN_K = int(
                os.environ.get("VLLM_GFX908_W4_DEQUANT_MIN_K", "0")
            )
        except ValueError:
            _W4_DEQUANT_MIN_K = 0
    return _W4_DEQUANT_MIN_K


def _w4_dequant_layout() -> str:
    """Scratch weight layout: "nt" -> [N, K] + F.linear, "nn" -> [K, N] + mm.

    "nt" is the default and wins overall (and on every shape at M=8192), but
    which layout rocBLAS prefers actually flips per (shape, M) -- most sharply
    on the narrow shared-expert gate_up (2560x320) at M=256, where "nn" is
    20 us and "nt" is 75 us against a 151 us fused kernel.  See
    agents/w4_dequant/INTEGRATION.md for the full table; a per-shape layout
    table is a follow-up worth ~2-3 ms/pass at the small M.
    """
    global _W4_DEQUANT_LAYOUT
    if _W4_DEQUANT_LAYOUT is None:
        import os

        v = os.environ.get("VLLM_GFX908_W4_DEQUANT_LAYOUT", "nt").lower()
        _W4_DEQUANT_LAYOUT = v if v in ("nt", "nn") else "nt"
    return _W4_DEQUANT_LAYOUT


def _w4_dequant_scratch(
    numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """One flat scratch buffer per (device, dtype), grown to the largest shape."""
    key = (device.type, -1 if device.index is None else device.index, dtype)
    buf = _W4_DEQUANT_SCRATCH.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=device)
        _W4_DEQUANT_SCRATCH[key] = buf
    return buf[:numel]


def _w4a16_dequant_escape(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor | None:
    """Return A @ dequant(B) via the bf16 dispatch, or None if not applicable."""
    if not _w4_dequant_large_m_enabled():
        return None
    if a.shape[0] < _w4_dequant_min_m():
        return None
    K = b_q.shape[0]
    N = b_q.shape[1] * 8
    if K < _w4_dequant_min_k():
        return None
    if scales.dtype != a.dtype:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    nt = _w4_dequant_layout() == "nt"
    shape = (N, K) if nt else (K, N)
    w = _w4_dequant_scratch(K * N, scales.dtype, b_q.device).view(shape)
    triton_w4a16_dequant(
        b_q=b_q,
        scales=scales,
        group_size=group_size,
        zp_bias=zp_bias,
        qzeros=qzeros,
        out=w,
        transpose_out=nt,
    )
    return torch.nn.functional.linear(a, w) if nt else torch.mm(a, w)


class TritonW4A16LinearKernel(MPLinearKernel):
    """
    Triton-based W4A16 GEMM kernel for ROCm (MI300 and newer).

    Supports GPTQ-format int4 weights (uint4b8 symmetric, uint4 asymmetric)
    with grouped quantization. Weight tensors are transposed from the
    compressed-tensors checkpoint layout to the kernel's [K, N//8] layout.
    """

    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton handles capability checks itself
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not (current_platform.is_rocm() or current_platform.is_cuda()):
            return False, "TritonW4A16LinearKernel requires CUDA or ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8 "
                "(8 int4 values packed per int32)",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "TritonW4A16LinearKernel",
            )

        gs = c.group_size
        if (
            gs not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({c.full_weight_shape[0]})",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Convert compressed-tensors checkpoint layout to kernel layout.

        Checkpoint (from compressed_tensors_wNa16.create_weights):
          weight_packed:     [N, K//8]  int32   input_dim=1, output_dim=0, packed_dim=1
          weight_scale:      [N, K//G]  fp16    input_dim=1, output_dim=0
          weight_zero_point: [N//8, K//G] int32  output_dim=0, packed_dim=0

        Kernel needs:
          qweight: [K, N//8]  int32   (transpose weight_packed)
          scales:  [K//G, N]  fp16    (transpose weight_scale)
          qzeros:  [K//G, N//8] int32 (transpose weight_zero_point)
        """

        # ---- Transform qweight: [N, K//8] → [K//8, N] → back to [K, N//8] ----
        # permute_param_layout_(x, input_dim=0, output_dim=1) rearranges so that
        # the input(K) dimension is at physical dim 0 and output(N) at dim 1.
        # Checkpoint has input_dim=1, output_dim=0, packed_dim=1 (K is packed).
        # After permute we get [K//8, N] (K packed at dim 0, N at dim 1).
        # The kernel wants [K, N//8] (K at dim 0, N packed at dim 1), so we
        # then transpose: [K//8, N].T = [N, K//8] — that's not right.
        #
        # Actually we need to change WHAT is packed:
        #   Original packing: K packed into K//8 (8 K-values per int32)
        #   Kernel packing:   N packed into N//8 (8 N-values per int32)
        # These require a full repack, not just a transpose.
        #
        # Simple approach: unpack → transpose the full [N, K] → repack as [K, N//8].
        # This is done CPU-side at load time (one-time cost).
        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            # x.data is [N, K//8] int32, K packed (GPTQ checkpoint format)
            # Step 1: bring to [N, K//8] with output(N) at dim 0
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32

            N_dim, K8 = w.shape
            K_dim = K8 * 8
            # Step 2: unpack to [N, K] int32 (vectorized)
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
            # Step 3: transpose to [K, N] int32
            w_KN = w_unpacked.t().contiguous()
            # Step 4: repack N into N//8 int32 values → [K, N//8] (vectorized)
            N8 = N_dim // 8
            w_repacked = torch.sum(
                (w_KN.view(K_dim, N8, 8) & 0xF) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # x.data is [N, K//G] fp16, bring to [K//G, N]
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # Kernel needs [K//G, N//8]:
                #        input(K) at dim 0, output(N) packed at dim 1.
                # AutoGPTQ:
                #        output_dim=1 -> already [K//G, N//8], no transpose.
                # compressed-tensors:
                #        output_dim=0 -> [N//8, K//G], needs transpose.
                # None (unknown):
                #        infer from shape; if square (ambiguous), default to transpose.
                zp_output_dim = getattr(zp, "output_dim", None)
                if zp_output_dim is not None:
                    needs_transpose = zp_output_dim != 1
                else:
                    # in case output_dim is None
                    c = self.config
                    K, N = c.partition_weight_shape
                    group_size = c.group_size if c.group_size != -1 else K
                    expected_shape = (K // group_size, N // 8)
                    transposed_shape = (N // 8, K // group_size)
                    if (
                        tuple(zp.data.shape) == expected_shape
                        and expected_shape != transposed_shape
                    ):
                        needs_transpose = False
                    else:
                        needs_transpose = True
                zp_data = (
                    zp.data.t().contiguous()
                    if needs_transpose
                    else zp.data.contiguous()
                )
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp_data, requires_grad=False),
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        # For symmetric types (uint4b8), use the scalar bias; no zeros tensor
        if c.weight_type.has_bias():
            zp_bias = c.weight_type.bias
            w_zp = None  # symmetric: ignore qzeros, use scalar bias instead
        else:
            zp_bias = 0

        if w_zp is None and _gfx908_gemv_enabled():
            output = torch.ops.vllm.gfx908_w4a16_gemm(
                x_2d, w_q, w_s, group_size, zp_bias
            )
        else:
            output = triton_w4a16_gemm(
                a=x_2d,
                b_q=w_q,
                scales=w_s,
                qzeros=w_zp,
                group_size=group_size,
                zp_bias=zp_bias,
            )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
