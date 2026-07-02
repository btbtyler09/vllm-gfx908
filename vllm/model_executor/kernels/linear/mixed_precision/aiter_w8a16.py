# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER Triton W8A16/W8A8 dynamic blockscale kernel for ROCm gfx908."""

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# AITER Triton kernels. Keep at module level to avoid repeated import overhead
# inside the hot apply_weights path.
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
    gemm_a16w8_blockscale,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
)

_AITER_W8A16_SUPPORTED_QUANT_TYPES = [scalar_types.uint8b128]
_AITER_W8A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]

# Dynamic dispatch threshold: microbenchmarks show AITER W8A8 is ~2.5x faster
# than W8A16 for large-M prefill, while W8A16 is better for small-M decode.
_W8A8_DISPATCH_MIN_M = 256


def _get_aiter_w8a8_config(M: int, N: int, K: int, group_size: int):
    """Select a gfx908-suitable AITER a8w8_blockscale config.

    For W8A8 the activation and weight scales are block-wise along K with
    BLOCK_SIZE_K=128, matching the GPTQ group_size. Microbench shows the same
    large-M config as W8A16 works well.
    """
    if M <= 16:
        block_m, block_n, num_warps, num_stages = 16, 64, 4, 2
    elif M <= 32:
        block_m, block_n, num_warps, num_stages = 32, 64, 4, 2
    elif M <= 64:
        block_m, block_n, num_warps, num_stages = 64, 64, 4, 2
    else:
        block_m, block_n, num_warps, num_stages = 128, 128, 8, 1

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "cache_modifier": ".cg",
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 2048,
    }


def _quantize_activation_per_block(x: torch.Tensor, block_k: int = 128):
    """Quantize FP16/BF16 activation to int8 with per-block K scales.

    Returns (x_q, x_scale) where x_scale shape is [M, K//block_k].
    """
    M, K = x.shape
    assert K % block_k == 0, f"K={K} not divisible by block_k={block_k}"
    x_blocks = x.reshape(M, K // block_k, block_k)
    absmax = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax)).to(
        x.dtype)
    x_q = (x_blocks / scale).clamp(-128, 127).round().to(torch.int8)
    return x_q.reshape(M, K), scale.squeeze(-1)


def _get_aiter_w8a16_config(M: int, N: int, K: int, group_size: int):
    """Select a gfx908-suitable AITER a16w8_blockscale config.

    The heuristic mirrors the one used for AITER a16w16 on MI100: small M
    decode uses BLOCK_SIZE_M=16, prefill / larger M scales up to keep the
    120 CUs busy. BLOCK_SIZE_K is fixed at 128 because all model shapes are
    divisible by 128 and the GPTQ group_size is 128.
    """
    if M <= 16:
        block_m, block_n, num_warps, num_stages, waves_per_eu = 16, 64, 4, 2, 2
    elif M <= 32:
        block_m, block_n, num_warps, num_stages, waves_per_eu = 32, 64, 4, 2, 2
    elif M <= 64:
        block_m, block_n, num_warps, num_stages, waves_per_eu = 64, 64, 4, 2, 2
    else:
        # Large-M prefill: microbench shows BLOCK_SIZE_N=128 and waves_per_eu=1
        # wins substantially on the Qwen3.6-27B TP4 projection shapes.
        block_m, block_n, num_warps, num_stages, waves_per_eu = 128, 128, 8, 1, 1

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 2048,
    }


class AiterW8A16LinearKernel(MPLinearKernel):
    """AITER Triton a16w8_blockscale W8A16 kernel for ROCm (gfx908)."""

    SUPPORTED_QUANT_TYPES = _AITER_W8A16_SUPPORTED_QUANT_TYPES

    # Per-process cache so we only JIT-compile each (kernel, M, N, K, gs) once.
    _WARMUP_CACHE: set[tuple[str, int, int, int, int]] = set()

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "AiterW8A16LinearKernel only targets ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        if c.zero_points:
            return False, "Zero points are not supported by AITER a16w8_blockscale"

        if c.has_g_idx:
            return False, "Activation reordering (g_idx) not supported"

        gs = c.group_size
        if (
            gs not in _AITER_W8A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {_AITER_W8A16_SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return False, f"Input features {K} not divisible by group_size {eff_gs}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ packed weights to AITER a16w8_blockscale layout.

        Checkpoint layout from create_weights:
          qweight: [K//4, N] int32  (input_dim=0, output_dim=1, packed_dim=0)
          scales:  [K//G, N] fp16   (output_dim=1)

        AITER expects:
          qweight: [N, K] int8   (signed, zero-point bias of 128 removed)
          scales:  [N, K//G] fp16
        """

        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            # Checkpoint layout: [K//4, N] int32 (4 uint8 weights per int32).
            w = x.data
            K4, N_dim = w.shape
            K_dim = K4 * 4

            # Unpack along the packed K dimension -> [K, N] uint8.
            shifts = torch.arange(4, device=w.device, dtype=torch.int32) * 8
            w_u8 = ((w.unsqueeze(1) >> shifts[None, :, None]) & 0xFF).reshape(
                K_dim, N_dim
            )

            # Convert uint8b128 -> signed int8 (subtract the zero-point bias).
            wt = self.config.weight_type
            bias = wt.bias if wt.has_bias() else 128
            w_i8 = (w_u8.to(torch.int16) - bias).to(torch.int8)

            # AITER expects [N, K] int8.
            x.data = w_i8.t().contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # scales are [K//G, N]; transpose to [N, K//G].
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        # Pre-compile the Triton kernels for the shapes this layer will see so
        # the first real inference does not pay JIT compilation latency.
        self._warmup(layer)

    def _warmup(self, layer: torch.nn.Module) -> None:
        """JIT-compile AITER a16w8_blockscale and a8w8_blockscale configs."""

        w_q, w_s, _, _ = self._get_weight_params(layer)
        N, K = w_q.shape
        gs = self.config.group_size
        device = w_q.device
        dtype = self.config.act_type

        # Decode / small-M path: a16w8_blockscale.
        for M in (1, 17, 33, 65):
            key = ("a16w8", M, N, K, gs)
            if key in self._WARMUP_CACHE:
                continue
            self._WARMUP_CACHE.add(key)

            x = torch.empty((M, K), dtype=dtype, device=device)
            cfg = _get_aiter_w8a16_config(M, N, K, gs)

            try:
                gemm_a16w8_blockscale(
                    x,
                    w_q,
                    w_s,
                    dtype=dtype,
                    config=cfg,
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"AITER W8A16 warmup failed for (M={M}, N={N}, K={K}, gs={gs}): {e}"
                )

        # Large-M prefill path: a8w8_blockscale (only valid for 128-block scales).
        if gs != 128:
            return

        for M in (256, 1024, 4096):
            key = ("a8w8", M, N, K, gs)
            if key in self._WARMUP_CACHE:
                continue
            self._WARMUP_CACHE.add(key)

            x = torch.empty((M, K), dtype=dtype, device=device)
            cfg = _get_aiter_w8a8_config(M, N, K, gs)

            try:
                x_q, x_s = _quantize_activation_per_block(x, block_k=128)
                gemm_a8w8_blockscale(
                    x_q,
                    w_q,
                    x_s,
                    w_s,
                    dtype=dtype,
                    config=cfg,
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"AITER W8A8 warmup failed for (M={M}, N={N}, K={K}, gs={gs}): {e}"
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        M, K = x_2d.shape
        N = w_q.shape[0]

        # Large-M prefill uses true INT8 compute (A8W8); decode stays on A16W8
        # because microbenchmarks show W8A8 is no faster and sometimes slower
        # for small M, while W8A16 avoids the activation quantization overhead.
        if M >= _W8A8_DISPATCH_MIN_M and c.group_size == 128:
            x_q, x_s = _quantize_activation_per_block(x_2d, block_k=128)
            cfg = _get_aiter_w8a8_config(M, N, K, c.group_size)
            output = gemm_a8w8_blockscale(
                x_q,
                w_q,
                x_s,
                w_s,
                dtype=x_2d.dtype,
                config=cfg,
            )
        else:
            cfg = _get_aiter_w8a16_config(M, N, K, c.group_size)
            output = gemm_a16w8_blockscale(
                x_2d,
                w_q,
                w_s,
                dtype=x_2d.dtype,
                config=cfg,
            )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
