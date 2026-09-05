# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908: push-AR consumer fused into the hyper-connection combine (+ RMSNorm).

``VLLM_GFX908_HC_AR_FUSED=1`` (default OFF; needs the sentinel push all-reduce,
``VLLM_GFX908_PUSH_AR``, to be active -- otherwise every call takes the stock path).

Every TP all-reduce of a Qwen4Exp decoder layer (attention o_proj / out_proj and the
MoE down projection) is consumed by exactly one hyper-connection combine: the AR of
the attention block by ``mlp_hyper_connection.combine_and_mix``, the AR of the MoE by
the *next* layer's ``attn_hyper_connection.combine_and_mix`` (or ``.combine`` before a
PLE layer, or the final mixer).  With the push AR the all-reduce is ``push_k`` +
``consume_k`` and the reduced tensor is then re-read by the Triton combine/norm
kernel.  This module splits the all-reduce into two opaque ops so the consumer half
runs *inside* the combine kernel (``csrc/gfx908_hc_ar_fused.hip``):

  ``gfx908_ar_push_deferred(x)``   push this rank's partial to the four slots of the
                                    next site and return a placeholder (no consume);
                                    falls back to the stock all-reduce whenever the
                                    message is not push-eligible (T > 48 rows, warm-up,
                                    push AR off) and returns the reduced tensor.
  ``gfx908_hc_combine_norm_ar``    / ``gfx908_hc_combine_ar``: if the block-output
                                    argument is the placeholder of a pending push, poll
                                    the four slots, reduce, combine (+norm) in one
                                    kernel; else the stock Triton kernel.

The pairing is a Python-side ``_PENDING`` record (site, shape, placeholder pointer)
written by the push op and consumed by the very next combine op.  Both ops execute in
program order on one stream, at capture time exactly once per graph (so the site baked
into the consume kernel is the site baked into the push kernel), and the model
structure guarantees no other deferred push sits between a push and its combine.
The site sequence itself is unchanged (one site per all-reduce, reset per capture), so
the >= 2-sites-per-cycle invariant of ``gfx908_push_ar`` is untouched.

Numerics: the residual stream ``out`` is bit-identical to consume_k + Triton combine
(same fp32 rank-order sum, one RNE, same fma); the norm output ``y`` uses a different
sum-of-squares tree and differs by one bf16 ulp on ~3e-6 of the elements
(agents/hc_gdn_glue/REPORT.md).
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_hc_ar_fused.hip")
_FLAG: bool | None = None
# (site, T, N, placeholder data_ptr) of the push whose consume is still outstanding.
_PENDING: tuple[int, int, int, int] | None = None
STATS = {"fused": 0, "fused_split": 0, "consume_stock": 0, "stock": 0,
         "push_deferred": 0, "push_stock": 0}
# Kernel arithmetic modes: bit0 = exp2-based sigmoid (what Triton's AMD backend emits for
# tl.sigmoid; with expf 7 of 2.9M combine outputs differ by 1 ulp on catastrophic-cancellation
# elements), bit1 = fma in the combine.  Measured bit-exact for `out` (agents/hc_gdn_glue).
_MODES = 3
# grid shape of the fused kernel.  ``split`` = one workgroup per (row, HC stream), like the
# Triton kernel's own decomposition, so the four streams' combine/norm chains run on four CUs
# instead of serially on one; the slot is re-armed by the last stream WG of a row through a
# self-resetting per-(site, row) counter, so no workgroup ever waits on another.  Measured on
# GPU2 (agents/hc_gdn_glue/bench_hc_ar_split.json): T=1 push+kernel 8.64 us split vs 14.19 us
# single-workgroup vs 9.00 us for the 3-launch stock chain.  Off -> the single-WG kernel.
_SPLIT: bool | None = None


_FUSED_TMAX: int | None = None


def _fused_tmax() -> int:
    """Rows above which the fused consumer loses to the 3-launch chain.  The split kernel polls
    the row's four source rows once per HC stream, so its uncached read traffic is 4x the stock
    consumer's; at T = 48 that costs more than the launch it saves (measured +0.6 us at T <= 8,
    +0.1 at T = 16, -0.9 at T = 48).  Above the threshold this module still owns the consume, it
    just runs the stock consume_k + Triton combine (still one launch fewer than an unfused AR)."""
    global _FUSED_TMAX
    if _FUSED_TMAX is None:
        try:
            _FUSED_TMAX = int(os.environ.get("VLLM_GFX908_HC_AR_FUSED_TMAX", "16"))
        except ValueError:
            _FUSED_TMAX = 16
    return _FUSED_TMAX


def _split_enabled() -> bool:
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = os.environ.get("VLLM_GFX908_HC_AR_FUSED_SPLIT", "1") == "1"
    return _SPLIT


def _cnt(par, n: int) -> torch.Tensor:
    """int32 [sites, rows_max] arrival counters for the split kernel (zeroed; self-resetting)."""
    cache = getattr(par, "_gfx908_hc_ar_cnt", None)
    if cache is None:
        cache = {}
        par._gfx908_hc_ar_cnt = cache
    t = cache.get(n)
    if t is None:
        rows = max(1, par.slot_elems // n)
        t = torch.zeros(par.sites, rows, dtype=torch.int32, device=par.device)
        cache[n] = t
    return t


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading fused HC/AR extension in %s", build_dir)
    return load(
        name="gfx908_hc_ar_fused_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def hc_ar_fused_enabled() -> bool:
    global _FLAG
    if _FLAG is None:
        from vllm.platforms.rocm import on_gfx908

        _FLAG = on_gfx908() and os.environ.get("VLLM_GFX908_HC_AR_FUSED", "0") == "1"
        if _FLAG:
            try:
                _ext()
            except Exception as exc:
                if os.environ.get("VLLM_GFX908_STRICT_EXT", "1") == "1":
                    raise RuntimeError(
                        "gfx908: fused HC/AR extension unavailable under its flag; "
                        "set VLLM_GFX908_STRICT_EXT=0 to fall back"
                    ) from exc
                logger.warning_once("gfx908: fused HC/AR extension unavailable (%s)", exc)
                _FLAG = False
        if _FLAG:
            logger.info_once("gfx908: push-AR consumer fused into the HC combine/RMSNorm (VLLM_GFX908_HC_AR_FUSED=1)")
    return _FLAG


def _push_ar():
    """The PushAllreduce of the TP custom-AR communicator, or None."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        comm = get_tp_group().device_communicator
        ca = getattr(comm, "ca_comm", None)
        if ca is None or ca.disabled:
            return None, None
        return ca, getattr(ca, "_push_ar", None)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- push
def _ar_push_deferred_impl(x: torch.Tensor) -> torch.Tensor:
    global _PENDING
    from vllm.distributed.parallel_state import get_tp_group

    _PENDING = None
    ca, par = _push_ar()
    if par is not None and ca.should_custom_ar(x):
        if ca._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
            # cudagraph warm-up pass: the stock path communicates nothing and returns an
            # uninitialised tensor; do the same (the consume op then takes the stock kernel).
            return torch.empty_like(x)
        if par.eligible(x):
            site = par._next_site()
            if site is not None:
                n = x.shape[-1]
                t = x.numel() // n
                xv = x.view(t, n)
                from vllm.distributed.device_communicators.gfx908_push_ar import _ext as _push_ext

                _push_ext().push(xv, par.ptrs, (site * par.world_size + par.rank) * par.slot_elems)
                out = torch.empty_like(x)
                _PENDING = (site, t, n, out.data_ptr())
                par.calls += 1
                STATS["push_deferred"] += 1
                return out
        par.fallbacks += 1
    STATS["push_stock"] += 1
    return get_tp_group().all_reduce(x)


def _ar_push_deferred_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="gfx908_ar_push_deferred",
    op_func=_ar_push_deferred_impl,
    fake_impl=_ar_push_deferred_fake,
)


def ar_push_deferred(x: torch.Tensor) -> torch.Tensor:
    """TP all-reduce whose consume half is deferred to the next HC combine."""
    return torch.ops.vllm.gfx908_ar_push_deferred(x)


# --------------------------------------------------------------------------- consume
def _take_pending(block: torch.Tensor):
    global _PENDING
    p = _PENDING
    if p is None:
        return None
    T = block.shape[0] if block.dim() == 2 else block.numel() // block.shape[-1]
    if p[3] != block.data_ptr() or p[1] != T or p[2] != block.shape[-1]:
        # Not our placeholder: leave the record alone (a different consumer will not match
        # either, and the stock kernel on a real reduced tensor is always correct).
        return None
    _PENDING = None
    return p


def _hc_combine_norm_ar_impl(
    residual: torch.Tensor, block: torch.Tensor, inj: torch.Tensor, w: torch.Tensor,
    eps: float, hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .ops.hc import _hc_combine_norm

    p = _take_pending(block)
    ca, par = (None, None) if p is None else _push_ar()
    if p is None or par is None or hc != 4 or residual.stride(1) != 1 or inj.stride(1) != 1:
        STATS["stock"] += 1
        return _hc_combine_norm(residual, block, inj, w, eps, hc)
    site, T, N, _ = p
    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    base = par.ptrs[par.rank] + site * par.world_size * par.slot_elems * 2
    if T > _fused_tmax():
        _ext().consume(block.view(T, N), base, par.slot_elems, par.stats, par.max_spin,
                       site, par.spin_stats)
        STATS["consume_stock"] += 1
        return _hc_combine_norm(residual, block, inj, w, eps, hc)
    if _split_enabled() and T <= _cnt(par, N).shape[1]:
        _ext().hc_ar_combine_norm_split(
            residual, inj, w, out, y, float(eps), hc, base, par.slot_elems,
            par.stats, par.max_spin, site, par.spin_stats, _MODES, _cnt(par, N),
        )
        STATS["fused_split"] += 1
    else:
        _ext().hc_ar_combine_norm(
            residual, inj, w, out, y, float(eps), hc, base, par.slot_elems,
            par.stats, par.max_spin, site, par.spin_stats, _MODES,
        )
    STATS["fused"] += 1
    return out, y


def _hc_combine_ar_impl(
    residual: torch.Tensor, block: torch.Tensor, inj: torch.Tensor, hc: int
) -> torch.Tensor:
    from .ops.hc import _hc_combine

    p = _take_pending(block)
    ca, par = (None, None) if p is None else _push_ar()
    if p is None or par is None or hc != 4 or residual.stride(1) != 1 or inj.stride(1) != 1:
        STATS["stock"] += 1
        return _hc_combine(residual, block, inj, hc)
    site, T, N, _ = p
    out = residual.new_empty(residual.shape)
    base = par.ptrs[par.rank] + site * par.world_size * par.slot_elems * 2
    if T > _fused_tmax():
        _ext().consume(block.view(T, N), base, par.slot_elems, par.stats, par.max_spin,
                       site, par.spin_stats)
        STATS["consume_stock"] += 1
        return _hc_combine(residual, block, inj, hc)
    if _split_enabled() and T <= _cnt(par, N).shape[1]:
        _ext().hc_ar_combine_split(
            residual, inj, out, hc, base, par.slot_elems,
            par.stats, par.max_spin, site, par.spin_stats, _MODES, _cnt(par, N),
        )
        STATS["fused_split"] += 1
    else:
        _ext().hc_ar_combine(
            residual, inj, out, hc, base, par.slot_elems,
            par.stats, par.max_spin, site, par.spin_stats, _MODES,
        )
    STATS["fused"] += 1
    return out


def _hc_combine_norm_ar_fake(
    residual: torch.Tensor, block: torch.Tensor, inj: torch.Tensor, w: torch.Tensor,
    eps: float, hc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


def _hc_combine_ar_fake(
    residual: torch.Tensor, block: torch.Tensor, inj: torch.Tensor, hc: int
) -> torch.Tensor:
    return residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="gfx908_hc_combine_norm_ar",
    op_func=_hc_combine_norm_ar_impl,
    fake_impl=_hc_combine_norm_ar_fake,
)
direct_register_custom_op(
    op_name="gfx908_hc_combine_ar",
    op_func=_hc_combine_ar_impl,
    fake_impl=_hc_combine_ar_fake,
)


def hc_combine_norm_ar(residual, block, inj, w, eps: float, hc: int):
    return torch.ops.vllm.gfx908_hc_combine_norm_ar(residual, block, inj, w, eps, hc)


def hc_combine_ar(residual, block, inj, hc: int):
    return torch.ops.vllm.gfx908_hc_combine_ar(residual, block, inj, hc)


# --------------------------------------------------------------------------- layer wiring
def defer_layer_all_reduces(layer) -> bool:
    """Turn off the in-module all-reduce of the attention block and the MoE of one
    ``Qwen4ExpDecoderLayer`` so the layer forward can issue the deferred push instead.
    Returns True when both were re-wired (else nothing is changed)."""
    attn = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
    proj = getattr(attn, "out_proj", None) or getattr(attn, "o_proj", None)
    if proj is None or not getattr(proj, "reduce_results", False) or getattr(proj, "tp_size", 1) <= 1:
        return False
    mlp = layer.mlp
    experts = getattr(mlp, "experts", None)
    if experts is not None:
        cfg = getattr(experts, "moe_config", None)
        if cfg is None or getattr(mlp, "is_sequence_parallel", False) or getattr(mlp, "replicate_shared_expert", False):
            return False
        if getattr(cfg, "is_sequence_parallel", False) or getattr(cfg, "use_all2all_kernels", False):
            return False
        mk = getattr(getattr(experts, "quant_method", None), "moe_kernel", None)
        if mk is not None and mk.output_is_reduced():
            return False
        cfg.skip_final_all_reduce = True
    else:
        down = getattr(mlp, "down_proj", None)
        if down is None or not getattr(down, "reduce_results", False):
            return False
        down.reduce_results = False
    proj.reduce_results = False
    return True
