# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"modules_to_not_convert={self.modules_to_not_convert})")

    def get_name(self) -> QuantizationMethods:
        return "awq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None)
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # Lazy import to avoid circular import.
            from .awq_marlin import AWQMarlinConfig, AWQMoEMethod
            from .moe_wna16 import MoeWNA16Config
            from .utils.marlin_utils import check_moe_marlin_supports_layer
            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels.")
                config = {
                    "quant_method": "awq",
                    "bits": self.weight_bits,
                    "group_size": self.group_size,
                    "zero_point": self.zero_point,
                    "lm_head": False,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix)
            marlin_compatible_config_dict = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
                "modules_to_not_convert": self.modules_to_not_convert,
            }
            awq_marlin_config = AWQMarlinConfig.from_config(
                marlin_compatible_config_dict)
            return AWQMoEMethod(awq_marlin_config)
        return None


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


# AWQ to GPTQ conversion functions for ROCm (based on nlzy's implementation)
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def pack_exllama(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=iweights.device)

    # packing rowwise
    iweights = iweights.view(iweights.shape[0] // (32 // bits), 32 // bits, -1)
    qweight = (
        torch.bitwise_left_shift(iweights, shifts[None, :, None])
        .sum(dim=1)
        .to(torch.int32)
    )

    # packing columnwise
    izeros = izeros.view(-1, izeros.shape[1] // (32 // bits), 32 // bits)
    qzeros = (
        torch.bitwise_left_shift(izeros, shifts[None, None, :])
        .sum(dim=-1)
        .to(torch.int32)
    )

    return qweight, qzeros


def unpack_reorder_pack(qweight, qzeros, bits):
    # Unpack the qweight and qzeros tensors
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    # Pack the qweight and qzeros tensors
    qweight, qzeros = pack_exllama(iweight, izeros, bits)

    return qweight, qzeros


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            num_groups,
            output_size_per_partition,
            dtype=params_dtype,
        ),
                                          input_dim=0,
                                          output_dim=1,
                                          weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_rocm():
            # Convert AWQ weights to GPTQ format for ROCm
            logger.info("Converting AWQ weights to GPTQ format for ROCm")
            
            # Check for known ROCm issue with asymmetric quantization + group_size 128
            if self.quant_config.group_size == 128 and self.quant_config.zero_point:
                logger.warning(
                    "ROCm has a known issue with asymmetric quantization + group_size 128. "
                    "This may produce incorrect results. Consider using models with "
                    "group_size 32 or symmetric quantization. "
                    "See: https://github.com/vllm-project/vllm/issues/[TBD]"
                )
            
            # Convert weights and zeros
            qweight, qzeros = unpack_reorder_pack(
                layer.qweight.data, 
                layer.qzeros.data,
                self.quant_config.weight_bits
            )
            
            # Update layer parameters with GPTQ format
            layer.qweight = Parameter(qweight, requires_grad=False)
            layer.qzeros = Parameter(qzeros, requires_grad=False)
            layer.scales = Parameter(layer.scales.data, requires_grad=False)
            
            # Apply GPTQ shuffle (with empty g_idx like nlzy)
            ops.gptq_shuffle(layer.qweight,
                           torch.empty(0, device=layer.qweight.device),
                           self.quant_config.weight_bits)
        else:
            # Standard AWQ processing for non-ROCm platforms
            layer.qweight = torch.nn.Parameter(layer.qweight.data,
                                               requires_grad=False)
            layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
                                              requires_grad=False)
            layer.scales = torch.nn.Parameter(layer.scales.data,
                                              requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if current_platform.is_rocm():
            # Use GPTQ kernels on ROCm with converted weights
            out_shape = x.shape[:-1] + (layer.qweight.shape[-1], )
            reshaped_x = x.reshape(-1, x.shape[-1])
            
            # Call GPTQ gemm with converted weights (empty g_idx like nlzy)
            output = ops.gptq_gemm(
                reshaped_x,
                layer.qweight,
                layer.qzeros,
                layer.scales,
                torch.empty(0, device=layer.qweight.device),  # empty g_idx
                True,  # use_exllama (optimized kernel)
                self.quant_config.weight_bits
            )
            
            if bias is not None:
                output.add_(bias)
            return output.reshape(out_shape)
        else:
            # Standard AWQ implementation for non-ROCm platforms
            qweight = layer.qweight
            scales = layer.scales
            qzeros = layer.qzeros
            pack_factor = self.quant_config.pack_factor
            out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
            reshaped_x = x.reshape(-1, x.shape[-1])

            # num_tokens >= threshold
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
                out = torch.matmul(reshaped_x, out)
            else:
                out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                                   pack_factor)
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)
