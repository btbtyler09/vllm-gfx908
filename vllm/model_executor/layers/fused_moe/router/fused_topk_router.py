# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import os

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


_GFX908_TOPK: bool | None = None


def _gfx908_small_m_topk(gating_output: torch.Tensor, topk_indices: torch.Tensor) -> bool:
    global _GFX908_TOPK
    if _GFX908_TOPK is None:
        from vllm.platforms.rocm import on_gfx908

        _GFX908_TOPK = (
            current_platform_is_rocm()
            and on_gfx908()
            and os.environ.get("VLLM_GFX908_TOPK", "0") == "1"
        )
    from vllm.model_executor.layers.fused_moe.gfx908_topk import TOPK_MAX_TOKENS

    return (
        _GFX908_TOPK
        and gating_output.shape[0] <= TOPK_MAX_TOKENS
        and gating_output.is_cuda
        and topk_indices.dtype == torch.int32
    )


_GFX908_ROUTER_FUSED: bool | None = None


def _gfx908_router_fused_enabled() -> bool:
    """VLLM_GFX908_ROUTER_FUSED=1: fuse the gate GEMV + softmax + top-k (MI100).

    See vllm/model_executor/layers/fused_moe/gfx908_router_topk.py.
    """
    global _GFX908_ROUTER_FUSED
    if _GFX908_ROUTER_FUSED is None:
        from vllm.platforms.rocm import on_gfx908

        _GFX908_ROUTER_FUSED = (
            current_platform_is_rocm()
            and on_gfx908()
            and os.environ.get("VLLM_GFX908_ROUTER_FUSED", "0") == "1"
        )
    return _GFX908_ROUTER_FUSED


def current_platform_is_rocm() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm()


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    if _gfx908_small_m_topk(gating_output, topk_indices):
        from vllm.model_executor.layers.fused_moe.gfx908_topk import gfx908_topk_softmax

        gfx908_topk_softmax(
            topk_weights, topk_indices, gating_output, renormalize,
            _get_padding_mask(topk_indices.shape[0]),
        )
        return topk_weights, topk_indices
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        # gfx908 (MI100): one HIP kernel replaces the gate GEMV, the bf16->fp32
        # cast and topkGating for decode-sized batches. `router_logits` has
        # already been produced by MoERunner._forward_impl; the fused op
        # recomputes them from `hidden_states` and the gate weight (and the gate
        # module's GEMM is bypassed after the first call, see gfx908_router_topk).
        if self.scoring_func == "softmax" and _gfx908_router_fused_enabled():
            from vllm.model_executor.layers.fused_moe.gfx908_router_topk import (
                maybe_fused_router,
            )

            fused = maybe_fused_router(
                self,
                hidden_states,
                self.top_k,
                self.renormalize,
                _get_padding_mask(hidden_states.shape[0]),
            )
            if isinstance(fused, tuple):
                return fused
            if fused is not None:
                # MoERunner skipped the gate GEMM but this call cannot be
                # fused: maybe_fused_router recomputed the real logits.
                router_logits = fused

        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
