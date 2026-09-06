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
    return os.environ.get("VLLM_GFX908_PLE_GLUE", "0") == "1"


def _envint(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _force_fallback() -> bool:
    """VLLM_GFX908_PLE_GLUE_FORCE_FALLBACK=1 keeps the op (and its graph split)
    but never launches the HIP kernel: every call runs ``ple_body_eager`` inside
    the opaque op.  Bisects "the fused kernel is wrong" from "running the PLE
    body opaquely instead of through inductor is wrong"."""
    return os.environ.get("VLLM_GFX908_PLE_GLUE_FORCE_FALLBACK", "0") == "1"


def _check_mode() -> int:
    """VLLM_GFX908_PLE_GLUE_CHECK: 0 off, 1 = run fused AND eager on the same
    inputs / the same starting conv state each call, log the deltas and serve
    the EAGER result, 2 = same but serve the FUSED result."""
    return _envint("VLLM_GFX908_PLE_GLUE_CHECK", 0)


_check_calls = 0
_check_bad = 0
_reasons: set[str] = set()


def _note(reason: str) -> bool:
    """Log a routing decision once per distinct reason."""
    if reason not in _reasons:
        _reasons.add(reason)
        logger.info("gfx908 PLE glue: %s", reason)
    return False


def _byte_range(t: torch.Tensor) -> tuple[int, int]:
    st = t.untyped_storage()
    base = st.data_ptr()
    return base, base + st.nbytes()


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False
    a0, a1 = _byte_range(a)
    b0, b1 = _byte_range(b)
    return a0 < b1 and b0 < a1


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
    # The kernel reads state_idx[t] linearly.  Under MTP `state_indices_tensor`
    # is `state_indices_tensor_d[:, 0]`, a column view with stride
    # 1 + num_spec_tokens, which would silently select the wrong cache slots.
    if idx.dim() != 1 or idx.stride(0) != 1:
        return False
    has_init = m.has_initial_states_d
    if has_init is not None and (
        has_init.numel() < tokens or has_init.dim() != 1 or has_init.stride(0) != 1
    ):
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

    # The kernel's per-(token, hc-stream) reduction reads EVERY channel of the
    # group while other workgroups of the same group are already storing their
    # slice of `output`.  If inductor ever hands us an `output` that shares
    # storage with `hidden_states` (the residual + the query of the same op),
    # that is a read/write race across workgroups, and the visible symptom is
    # the PLE delta being applied more than once on part of the row.  Stage
    # through a private buffer in that case (and say so, loudly, once).
    dst = output
    aliased = _overlaps(output, hidden_states) or _overlaps(output, key) or _overlaps(
        output, value
    )
    if aliased:
        logger.warning_once(
            "gfx908 PLE glue: `output` aliases an input buffer; staging the "
            "fused result through a private buffer"
        )
        dst = torch.empty_like(output)

    if _envint("VLLM_GFX908_PLE_GLUE_LOG", 0):
        _note(
            f"fused T={tokens} C={hidden_states.shape[1]} hc={layer.hc_count} "
            f"slen={layer.conv_state_len} dil={layer.short_conv_dilation} "
            f"cap={capacity} state{tuple(conv_state.shape)}"
            f"/{tuple(conv_state.stride())}/{conv_state.dtype} "
            f"idx{tuple(idx.shape)}/{idx.dtype}/stride{idx.stride(0)} "
            f"has_init={'None' if has_init is None else tuple(has_init.shape)} "
            f"aliased={aliased} contig_h={hidden_states.is_contiguous()}"
        )

    _ext().ple_glue(
        hidden_states, key, value,
        layer.norm_query.weight, layer.norm_key.weight, layer.norm_conv.weight,
        conv_w, conv_state, idx[:tokens],
        has_init[:tokens] if has_init is not None else None,
        dst, None, None,
        layer.hc_count, layer.short_conv_dilation, NULL_BLOCK_ID,
        layer.norm_conv.eps, True,
    )
    if dst is not output:
        output.copy_(dst)
    return True


def _check_body(layer, hidden_states, key, value, output, mode: int) -> bool:
    """Run the fused kernel and the eager body on the same inputs and the same
    starting conv state, log both deltas, and serve one of them.

    Returns False when the fused route is not eligible for this call, so the
    caller falls through to the normal path.
    """
    global _check_calls, _check_bad
    state = layer.kv_cache[0]
    if not isinstance(state, torch.Tensor) or state.numel() == 0:
        return False

    state0 = state.clone()
    out_f = torch.empty_like(output)
    if not _try_fused(layer, hidden_states, key, value, out_f):
        state.copy_(state0)
        return False
    state_f = state.clone()

    state.copy_(state0)
    out_e = layer.ple_body_eager(hidden_states, key, value)

    d_out = (out_f.float() - out_e.float()).abs().max().item()
    d_state = (state_f.float() - state.float()).abs().max().item()
    n_out = int((out_f != out_e).sum().item())
    n_state = int((state_f != state).sum().item())
    bad = d_out > 0.05 or d_state > 0.05
    _check_calls += 1
    _check_bad += int(bad)
    if _check_calls <= _envint("VLLM_GFX908_PLE_GLUE_CHECK_N", 24) or bad:
        ctx = get_forward_context()
        md = ctx.attn_metadata
        m = md.get(layer.prefix) if isinstance(md, dict) else None
        idx = getattr(m, "state_indices_tensor", None)
        hi = getattr(m, "has_initial_states_d", None)
        logger.info(
            "gfx908 PLE CHECK #%d %s T=%d na=%s nd=%s ndt=%s np=%s | "
            "out max|d|=%.3e mism %d/%d | state max|d|=%.3e mism %d/%d | "
            "idx[:8]=%s hi[:8]=%s alias_out_hidden=%s",
            _check_calls,
            "BAD" if bad else "ok",
            hidden_states.shape[0],
            getattr(m, "num_actual_tokens", None),
            getattr(m, "num_decodes", None),
            getattr(m, "num_decode_tokens", None),
            getattr(m, "num_prefills", None),
            d_out, n_out, out_f.numel(),
            d_state, n_state, state.numel(),
            None if idx is None else idx.flatten()[:8].tolist(),
            None if hi is None else hi.flatten()[:8].tolist(),
            _overlaps(output, hidden_states),
        )
    if mode >= 2:
        # serve the fused result AND leave the fused conv state behind, so the
        # session evolves exactly as it would without the check.
        state.copy_(state_f)
        output.copy_(out_f)
    else:
        output.copy_(out_e)
    return True


_LAST_LAYER = None  # VLLM_GFX908_PLE_GLUE_CMP: identity check from the layer


def _ple_glue_body(
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    global _LAST_LAYER
    _LAST_LAYER = layer
    mode = _check_mode()
    if mode and not torch.cuda.is_current_stream_capturing():
        # A/B every call; never do this while capturing (the clones would be
        # baked into the graph).
        if _check_body(layer, hidden_states, key, value, output, mode):
            return
    if not _force_fallback() and _try_fused(layer, hidden_states, key, value, output):
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
