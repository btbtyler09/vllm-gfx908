# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 fused PLE decode glue (Qwen4Exp / Qwen3.8-Flash-Next).

The PLE layer runs once per step (``ple_layer_ids = [2]``) but its body is the
largest un-fused block outside the layer loop: at one decode token the
compiled graph launches 23 kernels for it (4 inductor kernels for the
norm/gate/norm_conv chain, 18 eager kernels inside the opaque short-conv op,
and the residual add), ~66 us warm / ~96 us cold on gfx908.

``VLLM_GFX908_PLE_GLUE=1`` replaces all of that with one HIP kernel that
reproduces the compiled numerics exactly (see
``docs/mi100_decode_opt/research`` / the agent report): 12.4 us warm,
15.6 us cold, 1 launch.  The op keeps an in-op fallback to the shipping python
for every batch shape it does not handle (prefill, mixed, spec decode,
non-uniform decode), so it is safe to leave on.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc", "gfx908_ple_glue.hip")
_NT = 512          # threads per workgroup in the kernel (fixed by its reduction layout)
_MAX_STATE = 16    # MAXS in the kernel


def ple_glue_enabled() -> bool:
    return os.environ.get("VLLM_GFX908_PLE_GLUE", "1") == "1"


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    from vllm.platforms.gfx908_ext import hashed_build_dir

    build_dir = hashed_build_dir(build_dir, "ple_glue", [_CSRC])
    logger.info_once("gfx908: building/loading the fused PLE glue extension in %s", build_dir)
    return load(
        name="gfx908_ple_glue_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def _try_fused(layer, hidden_states, key, value, output) -> bool:
    """Uniform-decode fast path; returns False for every other batch shape."""
    if layer.conv_state_len < 1 or layer.conv_state_len > _MAX_STATE:
        return False
    if layer.conv_kernel_size != 4 or layer.short_conv_dilation not in (1, 3):
        return False
    if layer.hidden_size % _NT:
        return False
    ctx = get_forward_context()
    md = ctx.attn_metadata
    if not isinstance(md, dict):
        return False
    m = md.get(layer.prefix)
    if not isinstance(m, PleShortConvAttentionMetadata):
        return False
    if m.spec_sequence_masks is not None or m.num_prefills:
        return False
    tokens = hidden_states.shape[0]
    if m.num_actual_tokens != tokens or m.num_decode_tokens != tokens:
        return False
    idx = m.state_indices_tensor
    if idx is None or idx.numel() < tokens or idx.dtype not in (torch.int32, torch.int64):
        return False
    has_init = m.has_initial_states_d
    if has_init is not None and has_init.numel() < tokens:
        return False

    conv_state = layer.kv_cache[0]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    capacity = layer.conv_state_len + layer.num_spec_tokens
    if conv_state.size(-1) < capacity:
        return False
    conv_state = conv_state[..., -capacity:]
    conv_w = layer.conv1d.weight.squeeze(1).to(dtype=hidden_states.dtype)

    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    _ext().ple_glue(
        hidden_states, key, value,
        layer.norm_query.weight, layer.norm_key.weight, layer.norm_conv.weight,
        conv_w, conv_state, idx[:tokens],
        has_init[:tokens] if has_init is not None else None,
        output, None, None,
        layer.hc_count, layer.short_conv_dilation, NULL_BLOCK_ID,
        layer.norm_conv.eps, True,
    )
    return True


def _ple_glue_body(
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    if _try_fused(layer, hidden_states, key, value, output):
        return
    output.copy_(layer.ple_body_eager(hidden_states, key, value))


def _ple_glue_body_fake(
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="gfx908_ple_glue_body",
    op_func=_ple_glue_body,
    mutates_args=["output"],
    fake_impl=_ple_glue_body_fake,
)
