# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .triton_w8a16 import triton_w8a16_gemm

logger = init_logger(__name__)


# gfx908 (MI100) GPTQ8 "dual" graph-safe M-dispatch.
# The dispatch (native exllama for small M, Triton W8A16 MFMA for large M) MUST
# live inside an opaque custom op: a Python `if M > thresh` in apply() is a
# torch.compile graph break that drops the GPTQ8 layers out of the gfx908
# FULL_AND_PIECEWISE cudagraph (~3.6x decode regression). Registered as a custom
# op, dynamo sees one atomic node and cudagraphs capture it per-batch-size
# deterministically (M is constant within each captured graph). The native
# K-packed exllama kernel handles M<=MTHRESH (decode, zero regression); the
# repacked [K, N//4] Triton MFMA kernel handles M>MTHRESH (prefill/high-batch).
# Originally prototyped by curvedinf; adapted here to be cudagraph-safe.
# See docs/mi100_decode_opt/mtp_depth_sweep_gfx908.md.
def _gptq_dual_gemm_gfx908_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_repacked: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    mthresh: int,
    group_size: int,
    zero_offset: int,
    exllama_ready: bool,
    use_v2: bool,
    bit: int,
) -> torch.Tensor:
    if x.shape[0] > mthresh:
        return triton_w8a16_gemm(
            a=x,
            b_q=qweight_repacked,
            scales=scales,
            qzeros=qzeros,
            group_size=group_size,
            zp_bias=0,
            zero_offset=zero_offset,
        )
    return ops.gptq_gemm(x, qweight, qzeros, scales, g_idx, exllama_ready, use_v2, bit)


def _gptq_dual_gemm_gfx908_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_repacked: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    mthresh: int,
    group_size: int,
    zero_offset: int,
    exllama_ready: bool,
    use_v2: bool,
    bit: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], scales.shape[-1]))


direct_register_custom_op(
    op_name="gptq_dual_gemm_gfx908",
    op_func=_gptq_dual_gemm_gfx908_impl,
    mutates_args=[],
    fake_impl=_gptq_dual_gemm_gfx908_fake,
)


def _repack_gptq8_qweight_for_triton_w8a16(qweight: torch.Tensor) -> torch.Tensor:
    """Convert a GPTQ8 K-packed qweight [K//4, N] to the [K, N//4] layout the
    Triton W8A16 MFMA kernel consumes. Requires N divisible by 4; callers must
    fall back to the native path when it is not."""
    k4, output_size = qweight.shape
    if output_size % 4 != 0:
        raise ValueError(
            "Triton W8A16 GPTQ path requires output features divisible by 4, "
            f"got {output_size}."
        )
    shifts = torch.arange(4, device=qweight.device, dtype=torch.int32) * 8
    qweight_by_output = qweight.t().contiguous()
    unpacked = ((qweight_by_output.unsqueeze(-1) >> shifts) & 0xFF).reshape(
        output_size, k4 * 4
    )
    unpacked = unpacked.t().contiguous()
    repacked = torch.sum(
        (unpacked.view(k4 * 4, output_size // 4, 4) & 0xFF) << shifts,
        dim=2,
        dtype=torch.int32,
    )
    return repacked.contiguous()


def _gfx908_gptq8_dual_enabled(c: MPLinearLayerConfig) -> bool:
    """gfx908 GPTQ8 dual-layout dispatch (default ON for W8, no act reorder).

    VLLM_GFX908_GPTQ8 = "dual" (default) keeps BOTH weight layouts resident:
    native K-packed exllama for M<=MTHRESH (decode, zero regression,
    cudagraph'd) + repacked [K, N//4] Triton W8A16 MFMA for M>MTHRESH
    (prefill/high-batch, ~halves TTFT). Net: +12..84% concurrency, neutral
    c=1. Costs one extra qweight copy in VRAM. "native" opts out -> baseline
    exllama for all M (use for max-density long-context serving).
    """
    if c.weight_type != scalar_types.uint8b128 or c.has_g_idx:
        return False
    try:
        from vllm.platforms.rocm import on_gfx908

        if not (current_platform.is_rocm() and on_gfx908()):
            return False
    except Exception:
        return False
    return os.environ.get("VLLM_GFX908_GPTQ8", "dual").strip().lower() != "native"


def _gfx908_gptq8_mthresh() -> int:
    try:
        return int(os.environ.get("VLLM_GFX908_GPTQ8_MTHRESH", "16"))
    except ValueError:
        return 16


class ExllamaLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
    # In theory supports `scalar_types.uint2b2, scalar_types.uint3b4` too but
    # currently untested so not added to the list

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm",
            )

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act reordering currently not supported by Exllama, "
                "when the input features are partitioned across "
                "devices",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        if c.act_type != torch.float16:
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config

        # For Exllama, we need to set a zero-point tensor if there is not one
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            device = getattr(layer, self.w_q_name).device
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # if the type has a bias we have to create a zeros tensor that
                # contains the bias values repeated for each group (-1 due to
                # a bug in the original GPTQ checkpoint format leading to
                # exllama kernel adding 1 to the zero points during inference)
                # Documentation of the bug can be found here:
                #  https://garden.danieldk.eu/GPTQ-Checkpoint-Format
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "A 0 zero-point is not supported by Exllama due to "
                    "a bug in the original GPTQ checkpoint format leading to "
                    "exllama kernel adding 1 to the zero points during "
                    "inference"
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        if c.has_g_idx:

            def transform_w_g_idx(x):
                # Exllama wants the permutation array instead of the group
                # indices
                return torch.argsort(x).to(torch.int)

            self._transform_param(layer, self.w_gidx_name, transform_w_g_idx)  # type: ignore
        else:
            self.w_gidx_name = "g_idx"
            empty_g_idx = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int, device=device), requires_grad=False
            )
            setattr(layer, self.w_gidx_name, empty_g_idx)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            if _gfx908_gptq8_dual_enabled(c):
                # dual: materialize a SEPARATE repacked qweight copy for the
                # high-M Triton MFMA path BEFORE gptq_shuffle reorders the
                # K-packed layout the repack expects. scales/qzeros are shared.
                # Any layer whose repack is unsupported (e.g. N % 4 != 0) or
                # OOMs degrades to the native path rather than crashing load.
                try:
                    layer.gptq_qweight_repacked = torch.nn.Parameter(
                        _repack_gptq8_qweight_for_triton_w8a16(x_cont),
                        requires_grad=False,
                    )
                    layer.gptq_dual_ready = True
                except Exception as e:
                    if not getattr(ExllamaLinearKernel, "_dual_fallback_warned", False):
                        logger.warning(
                            "gfx908 GPTQ8 dual repack unavailable (%s); falling "
                            "back to native exllama for affected layers.",
                            e,
                        )
                        ExllamaLinearKernel._dual_fallback_warned = True
            ops.gptq_shuffle(x_cont, g_idx, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        # gptq_gemm supports GPTQv2 format by passing use_v2_format=True.
        # However, the MPLinearLayerConfig doesn't contain format info.
        # So hardcode GPTQv1 format here, to keep its behavior unchanged.
        use_v2_format = False

        assert w_zp is not None, "Zero points are required by Exllama"
        assert w_g_idx is not None, "Group index is required by Exllama"

        if getattr(layer, "gptq_dual_ready", False):
            # gfx908 dual: graph-safe M-dispatch via opaque custom op (keeps
            # cudagraphs). Inside the op: M>thresh -> Triton W8A16 MFMA on the
            # repacked copy; M<=thresh -> native exllama gptq_gemm.
            zero_offset = 0 if use_v2_format else 1
            output = torch.ops.vllm.gptq_dual_gemm_gfx908(
                x_2d.contiguous(),
                w_q,
                layer.gptq_qweight_repacked,
                w_zp,
                w_s,
                w_g_idx,
                _gfx908_gptq8_mthresh(),
                c.group_size,
                zero_offset,
                True,
                use_v2_format,
                c.weight_type.size_bits,
            )
            if bias is not None:
                output.add_(bias)
            return output.reshape(out_shape)

        output = ops.gptq_gemm(
            x_2d, w_q, w_zp, w_s, w_g_idx, True, use_v2_format, c.weight_type.size_bits
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
