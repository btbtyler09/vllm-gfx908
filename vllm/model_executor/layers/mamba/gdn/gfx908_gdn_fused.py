# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 fused GDN decode step (env `VLLM_GFX908_GDN_FUSED=1`, default off).

For non-spec decode with M <= 8 tokens the five launches between in_proj and
out_proj of a GDN layer (core_attn_out.zero_, z copy, causal_conv1d_update,
fused_recurrent_gated_delta_rule_packed_decode, RMSNormGated) become one HIP
kernel (csrc/gfx908_gdn_fused.hip): 13.4 -> 7.8 us at M=1, 25.7 -> 8.9 at M=4,
31.5 -> 12.4 at M=8 (graph-timed, MI100).  Conv state bit-exact, SSM state
within 2 fp32 ulps, output within 1 bf16 ulp of the stock sequence.

Fixed per-rank shapes: 4 K-heads, 12 V-heads, K = V = 128, conv width 4,
non-interleaved [q | k | v | z] / [b | a] projections, bf16 conv state, fp32
SSM state.  Everything else (prefill, spec decode, M > 8, other shapes) keeps
the stock path.
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

GDN_FUSED_MAX_TOKENS = 8
_H, _HV, _K, _V, _W = 4, 12, 128, 128, 4
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_gdn_fused.hip")
_FLAG: bool | None = None
_CACHE: dict[int, dict] = {}   # id(layer) -> fp32 params + scratch/counters
STATS = {"fused_calls": 0, "fallback_calls": 0}


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading fused GDN decode extension in %s", build_dir)
    return load(
        name="gfx908_gdn_fused_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def gdn_fused_enabled() -> bool:
    """Env flag (default off) and a successful extension build."""
    global _FLAG
    if _FLAG is None:
        from vllm.platforms.rocm import on_gfx908

        _FLAG = on_gfx908() and os.environ.get("VLLM_GFX908_GDN_FUSED", "0") == "1"
        if _FLAG:
            try:
                _ext()
            except Exception as exc:
                if os.environ.get("VLLM_GFX908_STRICT_EXT", "1") == "1":
                    raise RuntimeError("gfx908: fused GDN extension unavailable under its flag; set VLLM_GFX908_STRICT_EXT=0 to fall back") from exc
                logger.warning_once("gfx908: fused GDN extension unavailable (%s)", exc)
                _FLAG = False
    return _FLAG


def gdn_fused_layer_supported(layer, vllm_config) -> bool:
    """Static (init-time) eligibility of one GatedDeltaNetAttention layer."""
    if not gdn_fused_enabled():
        return False
    tp = layer.tp_size
    try:
        conv_dtype, ssm_dtype = layer.get_state_dtype()
    except Exception:
        return False
    ok = (
        not layer.gqa_interleaved_layout
        and layer.num_k_heads // tp == _H
        and layer.num_v_heads // tp == _HV
        and layer.head_k_dim == _K
        and layer.head_v_dim == _V
        and layer.conv_kernel_size == _W
        and layer.conv1d.bias is None
        and layer.activation in ("silu", "swish")
        and layer.norm.activation in ("silu", "swish", "sigmoid")
        and layer.norm.group_size is None
        and layer.norm.norm_before_gate
        and vllm_config.model_config.dtype == torch.bfloat16
        and conv_dtype == torch.bfloat16
        and ssm_dtype == torch.float32
    )
    logger.info_once("gfx908: fused GDN decode %s for %s", "ENABLED" if ok else "not applicable", layer.prefix)
    return ok


def _layer_cache(layer) -> dict | None:
    """fp32 copies of A_log / dt_bias / norm.weight plus scratch + self-resetting
    counters, built at the first eager call (never under graph capture)."""
    c = _CACHE.get(id(layer))
    if c is not None:
        return c
    if torch.cuda.is_current_stream_capturing():
        return None
    dev = layer.A_log.device
    c = {
        "A_log": layer.A_log.detach().float().contiguous(),
        "dt_bias": layer.dt_bias.detach().float().contiguous(),
        "norm_w": layer.norm.weight.detach().float().contiguous(),
        "conv_w": layer.conv1d.weight.detach().view(layer.conv1d.weight.size(0), -1).contiguous(),
        "scratch": torch.empty(GDN_FUSED_MAX_TOKENS, _HV, _V, dtype=torch.float32, device=dev),
        "cnt": torch.zeros(GDN_FUSED_MAX_TOKENS * (_HV + _H), dtype=torch.int32, device=dev),
    }
    _CACHE[id(layer)] = c
    return c


def _gfx908_gdn_fused_decode(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_w: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    idx: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor,
    cnt: torch.Tensor,
    eps: float,
    scale: float,
    rows: int,
    gate_sigmoid: bool,
) -> None:
    _ext().gdn_fused_decode(
        qkvz, ba, conv_w, None, conv_state, ssm_state, idx, A_log, dt_bias, norm_w,
        eps, scale, out, scratch, cnt, rows, 1, gate_sigmoid, 0,
    )


def _gfx908_gdn_fused_decode_fake(
    qkvz, ba, conv_w, conv_state, ssm_state, idx, A_log, dt_bias, norm_w, out, scratch, cnt,
    eps: float, scale: float, rows: int, gate_sigmoid: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="gfx908_gdn_fused_decode",
    op_func=_gfx908_gdn_fused_decode,
    mutates_args=["conv_state", "ssm_state", "out", "scratch", "cnt"],
    fake_impl=_gfx908_gdn_fused_decode_fake,
)


def maybe_fused_gdn_decode(layer, qkvz, ba, core_attn_out, attn_metadata) -> bool:
    """Run the fused kernel if this call is a plain decode of <= 8 tokens.
    Returns False (nothing touched) when the caller must take the stock path.
    `core_attn_out` receives the normed + gated bf16 result for every row."""
    if (
        attn_metadata.spec_sequence_masks is not None
        or attn_metadata.num_prefills != 0
        or attn_metadata.num_decodes <= 0
        or attn_metadata.num_actual_tokens > GDN_FUSED_MAX_TOKENS
        or qkvz.dtype != torch.bfloat16
    ):
        STATS["fallback_calls"] += 1
        return False
    c = _layer_cache(layer)
    if c is None:   # capture started before any eager call: stay on the stock path
        STATS["fallback_calls"] += 1
        return False
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    M = attn_metadata.num_actual_tokens
    kv = layer.kv_cache
    conv_state = kv[0] if is_conv_state_dim_first() else kv[0].transpose(-1, -2)
    idx = attn_metadata.non_spec_state_indices_tensor[:M]
    if not idx.is_contiguous():
        idx = idx.contiguous()
    torch.ops.vllm.gfx908_gdn_fused_decode(
        qkvz[:M], ba[:M], c["conv_w"], conv_state, kv[1], idx,
        c["A_log"], c["dt_bias"], c["norm_w"], core_attn_out[:M], c["scratch"], c["cnt"],
        float(layer.layer_norm_epsilon), float(layer.head_k_dim) ** -0.5,
        64 if M <= 4 else 128, layer.norm.activation == "sigmoid",
    )
    if core_attn_out.shape[0] > M:   # padded rows the kernel did not cover
        core_attn_out[M:].zero_()
    STATS["fused_calls"] += 1
    return True
