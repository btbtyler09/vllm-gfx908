# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 fused MoE router: bf16 gate GEMV + fp32 softmax + top-10 in one launch.

The stock decode-sized router on MI100 is three launches:

    gate GEMV (LLMM1 at M=1 / wvSplitK at M<=4 / rocBLAS at M=16)  ->  bf16 logits
    bf16 -> fp32 cast                                              ->  fp32 logits
    ops.topk_softmax (topkGating<8, 512>)                          ->  weights + ids

which is ~25 us in-graph at M=1..4 and ~88 us at M=16 for the Qwen3.8-Flash-Next
router shape (x [M, 2560] bf16, gate weight [512, 2560] bf16, top-10,
renormalize=True).  ``csrc/gfx908_router_topk.hip`` does all three in one kernel
(v_mfma_f32_16x16x8bf16 GEMV over 32 column tiles x KSO K-slices, LDS reduce,
atomic arrival counter, the last ceil(M/WPG) work groups finalize with a
wave-level softmax + 10 argmax passes): 9.2 / 9.7 / 10.7 us at M = 1 / 4 / 16.

Enable with ``VLLM_GFX908_ROUTER_FUSED=1`` (default off).

Numerics
--------
The kernel's GEMV accumulates in fp32 and is exact to ~5e-6 vs an fp32 reference.
The *stock* pipeline routes on **bf16-rounded** logits, and at M=1 it is worse
than that: the M=1 GEMV is LLMM1, whose output differs from a correctly rounded
bf16 in ~54% of the 512 logits (max err 1.2e-2 vs 7.7e-3), so the stock M=1
router disagrees with an fp32-reference top-10 on a large fraction of rows.

Two modes are therefore offered:

* default (``VLLM_GFX908_ROUTER_FUSED_BF16`` unset): route on the fp32 logits.
  More accurate than anything the stock path produces, but the selected experts
  differ from today's output on rows where two logits are within one bf16 ulp.
* ``VLLM_GFX908_ROUTER_FUSED_BF16=1``: round the logits to bf16 before the
  softmax, reproducing the bf16-GEMV pipeline.  This is the closest match to
  stock (0/100 and 0/400 rows differ at M=4 / M=16 vs the stock bf16 path; at
  M=1 3/25 rows differ, and those are rows where LLMM1 itself is wrong).

Other env knobs: ``VLLM_GFX908_ROUTER_FUSED_EXACT=1`` (xor-butterfly softmax sum,
weights bit-identical to topkGating), ``VLLM_GFX908_ROUTER_FUSED_FASTEXP=1``
(__expf, -0.3 us), ``VLLM_GFX908_ROUTER_FUSED_CFG=<0..6>`` (kernel shape, 4 is
best at every M), ``VLLM_GFX908_ROUTER_FUSED_SKIP_GATE=0`` (keep the stock gate
GEMM running even though its result is discarded; MoERunner skips it via gfx908_router_will_fuse).
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Compile-time constants of the kernel (csrc/gfx908_router_topk.hip).
ROUTER_NUM_EXPERTS = 512
ROUTER_TOPK = 10
ROUTER_HIDDEN = 2560
ROUTER_MAX_TOKENS = 16

# flag bits
_F_RENORM = 1
_F_BF16 = 2
_F_EXACT = 4
_F_FASTEXP = 8

# partials workspace: KSO(max 8) * 16 rows * 512 experts fp32
_PARTIALS_NUMEL = 8 * ROUTER_MAX_TOKENS * ROUTER_NUM_EXPERTS

_CSRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_router_topk.hip"
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


@functools.cache
def _ext():
    """JIT-build (or load from the cache dir) the fused router HIP extension."""
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm")
    )
    # VLLM_GFX908_HIP_BUILD_DIR is shared with the other gfx908 HIP extensions,
    # so give this one its own subdirectory (ninja keys off the directory).
    build_dir = os.path.join(build_dir, "gfx908_router_topk")
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading fused router extension in %s", build_dir)
    return load(
        name="gfx908_router_topk_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


@functools.cache
def fused_router_available() -> bool:
    try:
        return _ext() is not None
    except Exception as exc:  # hipcc missing / build failure -> stock path
        logger.warning_once(
            "gfx908: fused router unavailable (%s); using the stock router", exc
        )
        return False


@functools.cache
def _flags() -> int:
    """Flag bits that do not depend on the call (renormalize is added per call)."""
    f = 0
    if _env_flag("VLLM_GFX908_ROUTER_FUSED_BF16"):
        f |= _F_BF16
    if _env_flag("VLLM_GFX908_ROUTER_FUSED_EXACT"):
        f |= _F_EXACT
    if _env_flag("VLLM_GFX908_ROUTER_FUSED_FASTEXP"):
        f |= _F_FASTEXP
    return f


@functools.cache
def _cfg() -> int:
    return int(os.environ.get("VLLM_GFX908_ROUTER_FUSED_CFG", "4"))


# ---------------------------------------------------------------------------
# Persistent workspace.  The kernel needs an fp32 partials buffer and an int32[2]
# arrival counter that starts at zero; the last finalizing work group resets both
# counters in-kernel, so the same buffers are reused for every launch and every
# cudagraph replay.  Allocated once per device on the first call (which happens
# during the eager profiling run, before any capture).
# ---------------------------------------------------------------------------
_WORKSPACE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ws = _WORKSPACE.get(device)
    if ws is None:
        partials = torch.zeros(_PARTIALS_NUMEL, dtype=torch.float32, device=device)
        counter = torch.zeros(2, dtype=torch.int32, device=device)
        ws = (partials, counter)
        _WORKSPACE[device] = ws
    return ws


# ---------------------------------------------------------------------------
# The op.  Opaque to torch.compile (direct_register_custom_op) so inductor never
# traces into the persistent-buffer bookkeeping or the M-dependent dispatch.
# ---------------------------------------------------------------------------
def _router_topk_fused_impl(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    is_padding: torch.Tensor | None,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = hidden_states.shape[0]
    device = hidden_states.device
    partials, counter = _workspace(device)
    topk_weights = torch.empty(
        m, ROUTER_TOPK, dtype=torch.float32, device=device
    )
    topk_ids = torch.empty(m, ROUTER_TOPK, dtype=torch.int32, device=device)
    token_expert_indices = torch.empty(
        m, ROUTER_TOPK, dtype=torch.int32, device=device
    )
    flags = _flags() | (_F_RENORM if renormalize else 0)
    _ext().router_topk_fused(
        hidden_states,
        gate_weight,
        topk_weights,
        topk_ids,
        token_expert_indices,
        is_padding,
        partials,
        counter,
        flags,
        None,  # logits_out: not needed, nothing downstream reads them
        _cfg(),
        True,  # do_topk
    )
    return topk_weights, topk_ids, token_expert_indices


def _router_topk_fused_fake(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    is_padding: torch.Tensor | None,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = hidden_states.shape[0]
    return (
        hidden_states.new_empty((m, ROUTER_TOPK), dtype=torch.float32),
        hidden_states.new_empty((m, ROUTER_TOPK), dtype=torch.int32),
        hidden_states.new_empty((m, ROUTER_TOPK), dtype=torch.int32),
    )


direct_register_custom_op(
    op_name="gfx908_router_topk_fused",
    op_func=_router_topk_fused_impl,
    mutates_args=[],
    fake_impl=_router_topk_fused_fake,
)


def router_topk_fused(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    renormalize: bool,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(topk_weights [M,10] fp32, topk_ids [M,10] int32, token_expert_indices)."""
    return torch.ops.vllm.gfx908_router_topk_fused(
        hidden_states, gate_weight, is_padding, renormalize
    )


# ---------------------------------------------------------------------------
# Shape / dtype gate.
# ---------------------------------------------------------------------------
def fused_router_applies(
    hidden_states: torch.Tensor, gate_weight: torch.Tensor, top_k: int
) -> bool:
    return (
        top_k == ROUTER_TOPK
        and hidden_states.dim() == 2
        and hidden_states.shape[0] <= ROUTER_MAX_TOKENS
        and hidden_states.shape[1] == ROUTER_HIDDEN
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and hidden_states.stride(1) == 1
        and hidden_states.stride(0) % 8 == 0
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.is_contiguous()
        and tuple(gate_weight.shape) == (ROUTER_NUM_EXPERTS, ROUTER_HIDDEN)
    )


# ---------------------------------------------------------------------------
# Finding the gate weight, and switching the stock gate GEMM off.
#
# In this fork the gate GEMM does not live in the model's MoE block: MoERunner
# owns it and applies it in _forward_impl ("router_logits, _ = self.gate(x)")
# just before handing the logits to the router.  The router therefore sees the
# logits already computed.  To make the fused kernel replace *both* the GEMM and
# the top-k without editing moe_runner.py, the first fused call swaps the gate
# module's bound forward for one that returns a persistent zero buffer of the
# right shape/dtype whenever the fused path will handle this call (M <= 16 and
# the shapes match); for larger M it calls the original forward, so prefill is
# untouched.  The swap is per gate *instance*, installed once, and only when the
# runner's configuration proves nothing else consumes router_logits:
#   * no naive dispatch/combine (EP all-gather of the logits),
#   * pcp_size == 1 (no PCP all-gather of the logits),
#   * gate weights are not fused with the shared-expert gate (_fse_fuse_gate).
# If any of those hold, the bypass is not installed and the stock GEMM keeps
# running (~5.9 us wasted per layer) while the fused kernel still replaces the
# cast + topkGating.  Set VLLM_GFX908_ROUTER_FUSED_SKIP_GATE=0 to force that.
# ---------------------------------------------------------------------------
def gfx908_router_will_fuse(runner, hidden_states: torch.Tensor) -> bool:
    """True when the fused kernel will produce the routing for this call, so
    MoERunner can skip the (otherwise discarded) gate GEMM. Mirrors the checks
    in maybe_fused_router; any mismatch only costs the stock GEMM, never
    correctness, because maybe_fused_router recomputes from hidden_states."""
    if not _env_flag("VLLM_GFX908_ROUTER_FUSED") or not _env_flag(
        "VLLM_GFX908_ROUTER_FUSED_SKIP_GATE", "1"
    ):
        return False
    router = getattr(runner, "router", None)
    gate = getattr(runner, "gate", None)
    if router is None or gate is None or not hasattr(gate, "weight"):
        return False
    if getattr(router, "scoring_func", "softmax") != "softmax":
        return False
    if getattr(runner, "do_naive_dispatch_combine", False):
        return False
    if getattr(getattr(runner, "moe_config", None), "pcp_size", 1) > 1:
        return False
    if getattr(runner, "_fse_fuse_gate", False):
        return False
    if not fused_router_available():
        return False
    return fused_router_applies(hidden_states, gate.weight, int(router.top_k))


def gfx908_zero_logits(runner, hidden_states: torch.Tensor) -> torch.Tensor:
    """Placeholder router_logits when the gate GEMM is skipped (never read)."""
    buf = getattr(runner, "_gfx908_zero_logits", None)
    if buf is None or buf.device != hidden_states.device:
        buf = torch.zeros(
            (ROUTER_MAX_TOKENS, ROUTER_NUM_EXPERTS),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        runner._gfx908_zero_logits = buf
    return buf[: hidden_states.shape[0]]


def maybe_fused_router(
    router,
    hidden_states: torch.Tensor,
    top_k: int,
    renormalize: bool,
    is_padding: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused gate GEMV + softmax + top-k as a (weights, ids) tuple; a logits
    tensor when the runner skipped the GEMM but fusion does not apply; None
    when the caller's router_logits should be used.

    `router` is the FusedTopKRouter instance; it is used only to locate the gate
    weight of the owning MoERunner (cached on the router after the first call).
    """
    # The gate is handed over by MoERunner (see gfx908_router_will_fuse) on
    # every call where it skipped the gate GEMM; without it there is nothing
    # to fuse and the caller's router_logits are the real ones.
    gate = getattr(router, "_gfx908_gate", None)
    skipped = getattr(router, "_gfx908_gate_skipped", False)
    router._gfx908_gate_skipped = False
    if gate is None or not hasattr(gate, "weight"):
        return None

    weight = gate.weight
    if not fused_router_applies(hidden_states, weight, top_k) or not (
        fused_router_available()
    ):
        if skipped:
            # Runner/router predicate mismatch: recompute the logits the stock
            # way so routing never sees the placeholder zeros.
            logits, _ = gate(hidden_states)
            return logits
        return None

    topk_weights, topk_ids, _ = router_topk_fused(
        hidden_states, weight, renormalize, is_padding
    )
    return topk_weights, topk_ids
