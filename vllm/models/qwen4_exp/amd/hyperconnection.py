# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities for the AMD model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This AMD variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows for
        # alignment and efficient backend dispatch.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )
        self._gfx908_hc_fused = False
        # TP sharding of the mix over tokens (VLLM_GFX908_HC_SHARD=1, default
        # off). For use_combine=True the hook lives inside the fused custom op
        # (gfx908_hc_fused._hc_mix_impl); the final mixer has no fused path, so
        # it gets its own opaque op below.
        from .gfx908_hc_fused import hc_fused_enabled, hc_shard_enabled

        self._gfx908_hc_shard = hc_shard_enabled()
        if use_combine:
            from .gfx908_hc_fused import (
                hc_w8_enabled,
                install_hc_w8_prepare,
            )

            self._gfx908_hc_fused = hc_fused_enabled()
            if self._gfx908_hc_shard and not self._gfx908_hc_fused:
                from vllm.logger import init_logger

                init_logger(__name__).warning_once(
                    "gfx908: VLLM_GFX908_HC_SHARD=1 has no effect on the "
                    "combining HC modules while the fused HC path is off"
                )
            if self._gfx908_hc_fused and hc_w8_enabled():
                # Wraps the two mix Linears' process_weights_after_loading so the
                # int8 copies are built (and the bf16 masters released) at load
                # time, before torch.compile and before any cudagraph capture.
                install_hc_w8_prepare(self)

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine and self._gfx908_hc_fused_applies(xn):
            block_input, injection = self._gfx908_hc_fused_mix(xn)
            return hidden_states, block_input, injection

        if not self.use_combine and self._gfx908_hc_shard_applies(xn):
            return hidden_states, self._gfx908_hc_shard_final_mix(xn), None

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine and self._gfx908_hc_fused_applies(xn):
            block_input, injection = self._gfx908_hc_fused_mix(xn)
            return hidden_states, block_input, injection

        if not self.use_combine and self._gfx908_hc_shard_applies(xn):
            return hidden_states, self._gfx908_hc_shard_final_mix(xn), None

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    # -- gfx908 fused projections (wvSplitK copy with silu / gate-mix epilogues) --
    def _gfx908_hc_fused_applies(self, xn: torch.Tensor) -> bool:
        # Plain attribute checks only: this runs under torch.compile tracing
        # (M is symbolic, and the extension loader is a dynamo-skipped call).
        # The op dispatches on the real M at capture/run time.
        return xn.dtype == torch.bfloat16 and self._gfx908_hc_fused

    def _gfx908_hc_shard_applies(self, xn: torch.Tensor) -> bool:
        # Plain attribute check only (same rule as the fused gate): the real M
        # dispatch happens inside the custom op.
        return xn.dtype == torch.bfloat16 and self._gfx908_hc_shard

    def _gfx908_hc_shard_final_mix(self, xn: torch.Tensor) -> torch.Tensor:
        from .gfx908_hc_fused import hc_final_mix

        return hc_final_mix(
            xn,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.hc_count,
            self.lora_rank,
            self.hidden_size,
        )

    def _gfx908_hc_fused_mix(self, xn: torch.Tensor):
        from .gfx908_hc_fused import hc_fused_mix

        return hc_fused_mix(
            xn,
            self.input_mix_weight_down_block_inject.weight,
            self.input_mix_weight_up.weight,
            self.hc_count,
            self.lora_rank,
            self.hidden_size,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
