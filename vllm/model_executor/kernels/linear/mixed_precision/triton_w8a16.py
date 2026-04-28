# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W8A16 GEMM kernel for ROCm gfx908 (MI100).

Forked from triton_w4a16.py for W8 (uint8b128). Key differences vs W4:
  - 4 weights per int32 (vs 8); shifts [0,8,16,24] (vs [0,4,...,28])
  - mask 0xFF (vs 0xF); 2 interleaves (vs 3)
  - qweight column dim is N//4 (vs N//8); zeros also N//4

Plugs into the MPLinearKernel selection system. Required for dense
GPTQ-8bit on gfx908 — the default ops.gptq_gemm path runs at 2-7× off
HBM bandwidth for typical decode shapes; this kernel uses tl.dot
(MFMA) and is bandwidth-limited.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K, N//4]  int32  — rows=K (input), cols=N//4 (N is packed)
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//4]  int32  (optional; None for symmetric uint8b128)

Checkpoint layout from GPTQ create_weights (PackedvLLMParameter):
  qweight: [K//4, N]  int32  (input_dim=0, output_dim=1, packed_dim=0; K packed)
  scales:  [K//G, N]  fp16   (output_dim=1, input_dim=0)
  qzeros:  [K//G, N//4]  int32 (output_dim=1, packed_dim=1)
"""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W8A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W8A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint8b128,  # symmetric GPTQ-8bit (bias=128)
]


@triton.jit
def triton_w8a16_decode_kernel(
    # M=1 decode-specialized W8A16 kernel for gfx908.
    # No MFMA (tl.dot wastes 15/16 lanes for M=1); pure scalar reduction.
    # Split-K via atomicAdd to maximize CU utilization.
    a_ptr,           # [K] fp16 (single row of activations)
    b_ptr,           # [K, N//4] int32 packed weights (N-packed, 4 int8 per int32)
    scales_ptr,      # [K//G, N] fp16
    zeros_ptr,       # [K//G, N//4] int32 (unused when HAS_ZP=False)
    c_ptr,           # [N] fp16 output (initialized to 0; we atomicAdd)
    N, K,
    stride_bk, stride_bn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode-only W8A16 GEMM with split-K + atomicAdd reduction.

    Grid: (cdiv(N, BLOCK_N), SPLIT_K)
    Each program reduces one (BLOCK_N) × (K_chunk = K // SPLIT_K) tile.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # b/zeros: N//4 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)
    mask_bn = offs_bn < (N // 4)

    # 8-bit shifts tiled across BLOCK_N: [0,8,16,24] repeating
    shifts_row = tl.arange(0, 4) * 8  # [4]
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 4, 4))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))

    # Determine this program's K range
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_offset = pid_k * k_per_split
    k_end = tl.minimum(k_start_offset + k_per_split, K)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k = k_start_offset
    while k < k_end:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end

        # Load activation row segment: [BLOCK_K] fp16
        a = tl.load(a_ptr + offs_k, mask=mask_k, other=0.0)
        a_f32 = a.to(tl.float32)

        # Load packed weight tile: [BLOCK_K, BLOCK_N//4] int32
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_packed = tl.load(b_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0)

        # Unpack int8: 2 interleaves expand last dim by 4
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = (b >> shifts_1d[None, :]) & 0xFF  # [BLOCK_K, BLOCK_N]

        # Load scales (one group per BLOCK_K iter; BLOCK_K <= group_size)
        g_idx = k // group_size
        scales = tl.load(scales_ptr + g_idx * N + offs_n, mask=mask_n, other=1.0)

        # Dequant + accumulate
        if HAS_ZP:
            zero_offset = g_idx * (N // 4) + offs_bn
            z_packed = tl.load(zeros_ptr + zero_offset, mask=mask_bn, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xFF  # [BLOCK_N]
            b_centered_f32 = (b.to(tl.float32) - z[None, :].to(tl.float32))
        else:
            b_centered_f32 = (b.to(tl.float32) - ZP_BIAS)

        # Multiply by scale, then reduce K and accumulate
        # b_centered_f32: [BLOCK_K, BLOCK_N]; scales[None, :]: [1, BLOCK_N]
        b_fp = b_centered_f32 * scales[None, :].to(tl.float32)
        # a_f32[:, None] * b_fp → [BLOCK_K, BLOCK_N]; sum(0) → [BLOCK_N]
        accumulator += tl.sum(a_f32[:, None] * b_fp, axis=0)

        k += BLOCK_K

    # Atomic reduce into output (split-K)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_n, accumulator.to(c_ptr.type.element_ty), mask=mask_n)
    else:
        tl.atomic_add(c_ptr + offs_n, accumulator.to(c_ptr.type.element_ty), mask=mask_n)


@triton.jit
def triton_w8a16_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused W8A16 GEMM: C[M,N] = A[M,K] @ dequant(B)[K,N]

    B is stored as [K, N//4] int32 using GPTQ sequential 8-bit packing:
      each int32 packs 4 consecutive N-values at bit offsets [0,8,16,24].

    Dequant: w_fp = (w_int8 - zero) * scale
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # b/zeros: N//4 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)

    # GPTQ sequential 8-bit shifts tiled across BLOCK_N:
    #   [0,8,16,24] repeating for every group of 4 N-values.
    shifts_row = tl.arange(0, 4) * 8  # [4]
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 4, 4))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))  # [BLOCK_N]
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    # Scales column offsets: full N-width
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load activations A: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load packed weights B: [BLOCK_K, BLOCK_N//4] int32
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 4)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # Unpack int8 weights → [BLOCK_K, BLOCK_N]
        # Two interleaves multiply last dim by 4 (vs 3 for W4 / by 8).
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xFF

        # Group row index for this K tile
        g_idx = (k_start * BLOCK_K) // group_size

        # Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N]
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            # Load packed zeros row: [BLOCK_N//4] int32
            zero_offset = g_idx * (N // 4) + offs_bn
            zero_mask = offs_bn < N // 4
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xFF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        b_fp = (b - z).to(a.dtype) * scales

        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def _pick_block_sizes(M: int, N: int, K: int, group_size: int):
    """Per-arch block-size heuristics for W8A16 decode.

    On gfx908 the GPTQ-8 dense path is HBM-bandwidth bound at decode M=1,
    not arithmetic-bound. Tile choice optimizes for:
      - Enough N-tiles to keep the 120 CUs busy (avoid <60 tiles)
      - BLOCK_K = 32 (forced by group_size=32 in our checkpoint anyway)
      - BLOCK_M = 16 (smallest MFMA tile; pads M=1 with 15/16 wasted compute,
        which is fine when memory-bound)
    """
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        if on_gfx1x():
            # gfx1x branch from W4 kernel (RDNA, 32-wide wavefronts).
            if M <= 32:
                return 32, 32, 64
            if M <= 64:
                return 64, 64, 32
            return 128, 32, 64

        # Detect gfx908 via on_gfx908() (set in vllm/platforms/rocm.py)
        try:
            from vllm.platforms.rocm import on_gfx908
            is_gfx908 = on_gfx908()
        except Exception:
            is_gfx908 = False

        if is_gfx908:
            # gfx908: 120 CUs, 64-wide wavefronts, ~1.2 TB/s HBM2.
            # For M=1 decode (padded to BLOCK_M=16 for MFMA), favor more
            # N-tiles to saturate CUs.
            if M <= 16:
                return 16, 64, 32   # decode hot path
            if M <= 32:
                return 32, 64, 32
            if M <= 64:
                return 64, 64, 32
            return 128, 64, 32

        # MI300/gfx942 default
        if M <= 32:
            return 32, 64, 32
        if M <= 64:
            return 64, 64, 32
        return 128, 128, 32

    # Non-ROCm fallback
    if M <= 32:
        return 32, 64, 32
    if M <= 64:
        return 64, 64, 32
    return 128, 128, 32


def triton_w8a16_decode(
    a: torch.Tensor,           # [1, K] fp16/bf16 (M=1 only)
    b_q: torch.Tensor,         # [K, N//4] int32
    scales: torch.Tensor,      # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 128,
    split_k: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Decode-specialized W8A16 (M=1) using split-K + atomicAdd reduction.

    Optimized for gfx908 + decode hot path. Avoids tl.dot (no MFMA waste at
    M=1) and uses split-K to keep all 120 CUs busy.
    """
    assert a.shape[0] == 1, f"decode kernel requires M=1, got M={a.shape[0]}"
    M, K = a.shape
    N = b_q.shape[1] * 4

    # Pick split-K to target ~120 programs total (gfx908 CU count)
    if block_n is None:
        block_n = 64
    if block_k is None:
        block_k = min(group_size, 32)
    if split_k is None:
        n_tiles = (N + block_n - 1) // block_n
        target_programs = 120
        split_k = max(1, min(8, target_programs // max(1, n_tiles)))
        # Round split_k to a divisor of K/block_k for clean partitioning
        while split_k > 1 and (K // split_k) % block_k != 0:
            split_k -= 1

    # Output initialized to 0 (atomicAdd accumulates)
    c = torch.zeros((1, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q  # dummy ptr when unused

    grid = (triton.cdiv(N, block_n), split_k)
    triton_w8a16_decode_kernel[grid](
        a, b_q, scales, zeros_ptr, c,
        N, K,
        b_q.stride(0), b_q.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        SPLIT_K=split_k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def triton_w8a16_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 128,
) -> torch.Tensor:
    """Fused W8A16 GEMM using GPTQ-packed int8 weights.

    Args:
        a:          [M, K] fp16/bf16 activations.
        b_q:        [K, N//4] int32 packed weights (4 int8 per int32).
        scales:     [K//G, N] fp16/bf16 per-group scales.
        qzeros:     [K//G, N//4] int32 packed zero points, or None for
                    symmetric (uses zp_bias=128 for uint8b128).
        group_size: Group size (resolve -1 → K before calling).
        zp_bias:    Constant zero used when qzeros is None (default 128).
    """
    assert a.is_contiguous(), "a must be contiguous"
    assert b_q.is_contiguous(), "b_q must be contiguous"
    assert scales.is_contiguous(), "scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1] * 4

    assert b_q.shape == (K, N // 4), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // 4})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 4), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q  # dummy ptr when unused

    BLOCK_M, BLOCK_N, BLOCK_K = _pick_block_sizes(M, N, K, group_size)

    # The kernel loads scales/zeros for one group per BLOCK_K tile.
    # Clamp BLOCK_K to group_size so each tile sees one scale group.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w8a16_gemm_kernel[grid](
        a, b_q, scales, zeros_ptr, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


class TritonW8A16LinearKernel(MPLinearKernel):
    """Triton W8A16 GEMM kernel for ROCm (gfx908 / gfx942)."""

    SUPPORTED_QUANT_TYPES = TRITON_W8A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "TritonW8A16LinearKernel only targets ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 4 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 4 "
                "(4 int8 values packed per int32)",
            )

        if c.has_g_idx:
            return False, "Activation reordering (g_idx) not supported"

        gs = c.group_size
        if (
            gs not in TRITON_W8A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W8A16_SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return False, f"Input features {K} not divisible by group_size {eff_gs}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ checkpoint layout to kernel layout.

        Checkpoint qweight (PackedvLLMParameter):
          [K//4, N]  int32  input_dim=0, output_dim=1, packed_dim=0
        Kernel needs:
          [K, N//4]  int32  K at dim 0, N packed at dim 1
        """

        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            # Bring to [N, K//4] (output at dim 0, K packed at dim 1)
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//4] int32

            N_dim, K4 = w.shape
            K_dim = K4 * 4
            # Unpack to [N, K] int32
            shifts = torch.arange(4, device=w.device, dtype=torch.int32) * 8
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xFF).reshape(N_dim, K_dim)
            # Transpose to [K, N]
            w_KN = w_unpacked.t().contiguous()
            # Repack N into N//4 int32 → [K, N//4]
            N4 = N_dim // 4
            w_repacked = torch.sum(
                (w_KN.view(K_dim, N4, 4) & 0xFF) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # [N, K//G] → [K//G, N]
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # Checkpoint zeros: [N//4, K//G] int32 (N packed at dim 0)
                # Kernel needs: [K//G, N//4] — transpose
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
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

        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        output = triton_w8a16_gemm(
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
