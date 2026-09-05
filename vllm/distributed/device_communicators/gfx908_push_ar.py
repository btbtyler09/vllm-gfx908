# SPDX-License-Identifier: Apache-2.0
"""gfx908 sentinel push all-reduce (env ``VLLM_GFX908_PUSH_AR=1``, default OFF).

Decode-sized TP all-reduces on 4x MI100 cost 7.7-21.8 us through
``cross_device_reduce_1stage`` (a start barrier, a staged copy, the reduce, an end barrier).
The push scheme replaces the two barriers and the staging copy with a data-carried handshake:

* every rank owns one uncached, IPC-shared receive buffer pre-armed with the bf16 ``-0.0``
  sentinel (``0x8000``);
* the producer kernel writes this rank's partial into slot ``(site, my_rank)`` of *all* four
  ranks' buffers (fire-and-forget nontemporal stores), sanitizing ``-0.0`` to ``+0.0`` on the way;
* the consumer kernel spins on its own 16 B of each of the four source rows until none of them
  still reads as the sentinel, sums them in fixed rank order in fp32 (bitwise identical to
  vLLM's ``packed_reduce``), writes the output, and re-arms the slot.

Graph-timed on an idle 4x MI100 box (agents/ar_push/REPORT.md section 5), us per all-reduce:

===========  ==============  ==============
T (2560 wide) stock custom AR  push+consume
===========  ==============  ==============
1  (5 KB)     7.71            5.38
4  (20 KB)    8.17            5.86
16 (80 KB)   11.95            8.08
48 (240 KB)  21.78           15.00
===========  ==============  ==============

Output is bit-identical to ``cross_device_reduce_1stage`` except for one inert case: when all four
ranks contribute exactly ``-0.0`` for an element the stock kernel yields ``-0.0`` and this path
yields ``+0.0``.

Site discipline (the correctness-critical part): a slot may not be pushed again before every rank
has consumed it.  Each all-reduce call site therefore gets its own slot, assigned deterministically
by a counter that is reset once per captured graph (keyed on the HIP stream-capture id) so that all
four ranks bake the identical site sequence into each graph.  Eager calls rotate through a separate
pool of slots.  A single-slot cycle was shown to race (REPORT.md section 4), hence >= 2 sites per
cycle is load-bearing; the defaults give 128 graph slots for a model that issues 96 all-reduces per
decode step.

Env knobs:
  ``VLLM_GFX908_PUSH_AR``         1 to enable (default 0)
  ``VLLM_GFX908_PUSH_AR_TMAX``    max rows of a push-eligible message (default 48)
  ``VLLM_GFX908_PUSH_AR_WIDTH``   slot width in elements (default: model hidden size, else 2560)
  ``VLLM_GFX908_PUSH_AR_SITES``   graph slots (default 128)
  ``VLLM_GFX908_PUSH_AR_EAGER``   eager slots (default 8)
  ``VLLM_GFX908_PUSH_AR_MAX_SPIN`` bounded spin, poll rounds (default 4194304, ~3 s)
  ``VLLM_GFX908_PUSH_AR_STATS``   1 to also collect the per-wave spin histogram (costs ~0.5 us)
"""

import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_push_ar.hip")

SENTINEL_U32 = 0x80008000

_DEFAULT_TMAX = 48
_DEFAULT_WIDTH = 2560
_DEFAULT_SITES = 128
_DEFAULT_EAGER = 8
_DEFAULT_MAX_SPIN = 1 << 22


@functools.cache
def _ext():
    """JIT-build (or load from the cache dir) the HIP push/consume extension."""
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP push-AR extension in %s", build_dir)
    return load(
        name="gfx908_push_ar_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def push_ar_requested() -> bool:
    return os.environ.get("VLLM_GFX908_PUSH_AR", "0") == "1"


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("gfx908 push AR: %s is not an integer; using %d", name, default)
        return default


def _default_width() -> int:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        h = int(cfg.model_config.get_hidden_size())
        if h > 0 and h % 8 == 0 and h <= 8192:
            return h
    except Exception:
        pass
    return _DEFAULT_WIDTH


class PushAllreduce:
    """Slot pool + dispatch for the sentinel push all-reduce.  4 ranks, bf16, same node."""

    def __init__(self, ca):
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

        self.ca = ca
        self.group = ca.group
        self.rank = ca.rank
        self.world_size = ca.world_size
        self.device = ca.device

        width = _envi("VLLM_GFX908_PUSH_AR_WIDTH", _default_width())
        tmax = _envi("VLLM_GFX908_PUSH_AR_TMAX", _DEFAULT_TMAX)
        self.graph_sites = _envi("VLLM_GFX908_PUSH_AR_SITES", _DEFAULT_SITES)
        self.eager_sites = _envi("VLLM_GFX908_PUSH_AR_EAGER", _DEFAULT_EAGER)
        self.max_spin = _envi("VLLM_GFX908_PUSH_AR_MAX_SPIN", _DEFAULT_MAX_SPIN)
        self.spin_stats = 1 if os.environ.get("VLLM_GFX908_PUSH_AR_STATS", "0") == "1" else 0
        assert self.graph_sites >= 2 and self.eager_sites >= 2, (
            "the push scheme needs >= 2 slots per cycle"
        )

        # bf16 elements per (site, source rank); the largest message this path accepts.
        self.slot_elems = ((tmax * width) // 8) * 8
        self.sites = self.graph_sites + self.eager_sites
        self.nbytes = self.sites * self.world_size * self.slot_elems * 2

        # Same allocation + IPC handle exchange vLLM already uses for the custom-AR buffers
        # (hipExtMallocWithFlags(hipDeviceMallocUncached) inside allocate_shared_buffer_and_handle).
        self.ptrs = CustomAllreduce.create_shared_buffer(
            self.nbytes, group=self.group, uncached=True
        )
        self.stats = torch.zeros(16, dtype=torch.int32, device=self.device)
        _ext().fill_u32(self.ptrs[self.rank], self.nbytes, SENTINEL_U32)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self.group)

        self._capture_id = 0
        self._capture_n = 0
        self._eager_n = 0
        self.calls = 0
        self.fallbacks = 0
        self._overflow_warned = False
        logger.info(
            "gfx908 push AR enabled: %d+%d slots x %d ranks x %d elems = %.1f MB/rank, "
            "max message %d bf16 elements (e.g. T <= %d at width %d)",
            self.graph_sites, self.eager_sites, self.world_size, self.slot_elems,
            self.nbytes / 1e6, self.slot_elems, self.slot_elems // width, width,
        )

    # -------------------------------------------------------------- slot assignment
    def _next_site(self) -> int | None:
        cid = _ext().capture_id()
        if cid:
            if cid != self._capture_id:
                self._capture_id = cid
                self._capture_n = 0
            if self._capture_n >= self.graph_sites:
                if not self._overflow_warned:
                    self._overflow_warned = True
                    logger.warning(
                        "gfx908 push AR: a captured graph holds more than %d all-reduces; "
                        "the extra ones use the stock path (raise VLLM_GFX908_PUSH_AR_SITES)",
                        self.graph_sites,
                    )
                return None
            site = self._capture_n
            self._capture_n += 1
            return site
        site = self.graph_sites + (self._eager_n % self.eager_sites)
        self._eager_n += 1
        return site

    # -------------------------------------------------------------- dispatch
    def eligible(self, inp: torch.Tensor) -> bool:
        if inp.dtype is not torch.bfloat16 or not inp.is_contiguous() or inp.dim() < 2:
            return False
        n = inp.shape[-1]
        if n % 8 or n > 8192 or n == 0:
            return False
        return inp.numel() <= self.slot_elems

    def maybe_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        """Push+consume all-reduce, or None when this message must take the stock path."""
        if not self.eligible(inp):
            self.fallbacks += 1
            return None
        site = self._next_site()
        if site is None:
            self.fallbacks += 1
            return None
        n = inp.shape[-1]
        t = inp.numel() // n
        x = inp.view(t, n)
        out = torch.empty_like(x)
        ext = _ext()
        ext.push(x, self.ptrs, (site * self.world_size + self.rank) * self.slot_elems)
        ext.consume(
            out,
            self.ptrs[self.rank] + site * self.world_size * self.slot_elems * 2,
            self.slot_elems,
            self.stats,
            self.max_spin,
            site,
            self.spin_stats,
        )
        self.calls += 1
        return out.view(inp.shape)

    # -------------------------------------------------------------- diagnostics / teardown
    def stats_dict(self) -> dict:
        v = self.stats.cpu().tolist()
        return {
            "timeouts": v[0], "max_spin": v[1], "waves_spun": v[2],
            "last_timeout_site": v[3], "last_timeout_row": v[4], "last_timeout_col": v[5],
            "calls": self.calls, "fallbacks": self.fallbacks,
        }

    def check_and_log(self, tag: str = "") -> bool:
        """Log the counters (one D2H sync of 16 ints); ERROR if a consumer ever timed out.

        Called from the gfx908 step timer every 200 steps and at teardown. A non-zero
        ``timeouts`` means a consumer gave up waiting for a rank's push and summed the
        sentinel: the run is corrupt and the site sequence diverged across ranks.
        Returns True when the counters are clean.
        """
        try:
            d = self.stats_dict()
        except Exception as exc:  # device already torn down
            logger.debug("gfx908 push AR stats unavailable (%s)", exc)
            return True
        if d["timeouts"]:
            logger.error("gfx908 push AR%s CORRUPT: %s", f" {tag}" if tag else "", d)
            return False
        logger.info("gfx908 push AR%s: %s", f" {tag}" if tag else "", d)
        return True

    def close(self):
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

        if self.ptrs is not None:
            if self.calls:
                self.check_and_log("at teardown")
            CustomAllreduce.free_shared_buffer(self.ptrs, rank=self.rank)
            self.ptrs = None


def maybe_create_push_ar(ca) -> PushAllreduce | None:
    """Build the push-AR slot pool for this CustomAllreduce, or None (any reason -> stock path)."""
    if not push_ar_requested():
        return None
    try:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return None
        from vllm.platforms.rocm import on_gfx908

        if not on_gfx908():
            logger.warning_once("VLLM_GFX908_PUSH_AR=1 ignored: not running on gfx908")
            return None
    except Exception as exc:
        logger.warning("gfx908 push AR: platform probe failed (%s); using the stock path", exc)
        return None
    if ca.world_size != 4:
        logger.warning_once(
            "gfx908 push AR: only world size 4 is implemented (got %d); using the stock path",
            ca.world_size,
        )
        return None
    try:
        return PushAllreduce(ca)
    except Exception as exc:  # hipcc missing, OOM, ... -> stock path
        logger.warning("gfx908 push AR unavailable (%s); using the stock custom all-reduce", exc)
        return None
