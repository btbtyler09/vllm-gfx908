# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 W8A16 skinny GEMV for unquantized (bf16) decode projections.

Opt-in via ``VLLM_GFX908_W8A16=1`` (default OFF).

At decode (M <= 4) the bf16 ``wvSplitK`` skinny GEMM is HBM-bound on the weight
stream.  This path keeps the *activations* in bf16 but streams an int8 copy of
the weight with symmetric group scales, halving the bytes.  Graph-timed on
MI100 (``agents/w8a16/REPORT.md``, L2-cold, M=1): lm_head 356 -> 181 us
(1.97x), GDN in_proj_qkvz 23.7 -> 13.6 (1.74x), GDN out_proj 10.7 -> 6.8
(1.58x).

Phase 1 scope: only the three whitelisted shapes below (GDN ``in_proj_qkvz``
and ``out_proj`` of the 36 GDN layers, plus ``lm_head``).  The router, the
indexer and the hyper-connection mixes are deliberately excluded (fused
epilogues / routing sensitivity).

Phase 2 (this file): the int8 copy can be made the **only** resident copy.
``prepare_w8a16_layer`` (called from ``process_weights_after_loading``, i.e.
before torch.compile and before any cudagraph capture) quantizes the weight,
releases the bf16 storage and rebinds ``layer.weight.data`` to a zero-stride
stub that keeps the shape/dtype/device (so vLLM's shape-derived logic and the
custom op's fake impl still work) but owns no memory of its own.  ``M > 4``
(prefill / batched decode) is then served by rematerialising the weight into a
small reusable bf16 scratch with the ``w8a16_dequant`` HIP kernel, in N-row
chunks, and running the stock bf16 GEMM dispatch per chunk.

Which weights lose their bf16 master copy is controlled by
``VLLM_GFX908_W8A16_FREE`` (``gdn`` | ``all`` | ``none``, default ``gdn``):
``lm_head`` is kept in bf16 by default because it is on the *batched decode*
path (logits for every sequence in the step), where a full rematerialisation
costs more than the int8 duplicate saves.

Numerics: with the bf16 copy released, ``M > 4`` no longer matches stock
bit-for-bit — it computes with the dequantized int8 weight, exactly the
approximation the ``M <= 4`` GEMV path has always used.

Phase 3 (``VLLM_GFX908_W8A16_MFMA=1``, default OFF): a hand-written MFMA GEMM
(``csrc/gfx908_w8a16_mfma.hip``, design D2) serves ``5 <= M <= 64`` directly from
the int8 weight instead of rematerialising it, using ``v_mfma_f32_16x16x8bf16``
over a pre-swizzled weight layout.  Graph-timed on MI100 (agents/mfma_w8):
lm_head 62080x2560 at M=8 444 -> 210 us (2.1x vs the stock bf16 dispatch, 5.5x vs
dequant+stock), GDN in_proj_qkvz 4096x2560 at M=8 39.8 -> 20.3 us.

Because the MFMA kernel needs a different byte order than the ``M <= 4`` GEMV,
keeping both int8 layouts would cost a second full int8 copy (+678 MB/rank for
the whitelisted set).  So with the flag on the **swizzled layout is the only
resident int8 copy** and every M is served from it: ``M <= 4`` also goes through
a GEMV that reads the swizzled bytes directly (``csrc/gfx908_w8sw_gemv.cuh``,
``w8sw_gemv``): one wave64 owns a 16-row n-tile and lane ``l`` owns the same 16 B
it owns in the MFMA kernel, so the k-tile is a single coalesced 1 KiB
``dwordx4`` per wave and there is no M padding.  ``M > 64`` rematerialises from
the swizzled copy with ``mfma_w8_dequant``.

Phase 4 (``VLLM_GFX908_W4_LOADTIME=gdn|all``, default OFF): the two GDN
projections are RTN-quantized to the **W4 GS32** layout the W4 GEMV kernels
consume (``csrc/gfx908_w4_loadtime.hip``) instead of int8, halving their weight
stream again (0.28x bf16 bytes vs 0.52x for int8+gs128 scales).  The bf16 master
is released and ``1 <= M <= 8`` is served by the W4 slab GEMV; above that the
weight is rematerialised with ``w4lt_dequant``.  The default activation mode
(``VLLM_GFX908_W4_LOADTIME_ACT=bf16``) folds the bf16 -> fp16 cast and the
per-8 permutation into the kernel's LDS staging, so the path costs exactly one
launch, like the int8 GEMV, and is numerically identical to reading fp16
activations (bf16 -> fp16 is lossless).  ``f16`` / ``int8`` select the
``VLLM_GFX908_W4A8_MODE`` dot types with a separate prep launch.

**Accuracy warning.**  Plain RTN at 4 bits costs ~9-11% relative error on these
tensors (measured on the real checkpoint, ``agents/w4_loadtime/correct.log``)
against ~1.5-2% for int8 gs128, so this flag must pass the PPL / GSM8K gate
before it is ever defaulted on.

The int8 + scale side tensors are cached by ``weight.data_ptr()``.  When
``prepare_w8a16_layer`` has not run, they are still built lazily on the first
*eager* call (and the bf16 copy is then never released, since swapping a
parameter's storage after compile would invalidate dynamo guards and captured
graph addresses).  A lookup that misses while a cudagraph is being captured
falls through to the bf16 path, so capture never allocates.
"""

import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SOURCES = [
    os.path.join(_CSRC_DIR, f)
    for f in (
        "gfx908_w8a16.hip",
        "gfx908_w8a16_dequant.hip",
        "gfx908_w8a16_mfma.hip",
        "gfx908_w8sw.hip",
        "gfx908_w8sw_nb1.hip",
        "gfx908_w8sw_nb2.hip",
        "gfx908_w8sw_nb3.hip",
        "gfx908_w8sw_nb4.hip",
        "gfx908_w8a16_nb1.hip",
        "gfx908_w8a16_nb2.hip",
        "gfx908_w8a16_nb3.hip",
        "gfx908_w8a16_nb4.hip",
    )
]

# Max batch (rows of x) the kernel is instantiated for.
W8A16_MAX_M = 4

# ---------------------------------------------------------------------------
# W4 load-time path (VLLM_GFX908_W4_LOADTIME), see the module docstring
# ---------------------------------------------------------------------------
_W4LT_SRC = os.path.join(_CSRC_DIR, "gfx908_w4_loadtime.hip")
# The W4 slab GEMV is instantiated for 1 <= M <= 8 (M is a runtime loop bound;
# above 8 the weight rematerialisation wins, see agents/w4_loadtime/REPORT.md).
W4LT_MAX_M = 8
# Shapes the ``gdn`` policy quantizes to W4.  lm_head is deliberately absent: it
# runs at M = #sequences every step and 4-bit RTN on the output embedding is the
# single most accuracy-sensitive weight in the model.
W4LT_GDN_SHAPES = frozenset({(4096, 2560), (2560, 1536)})
_W4LT_POLICY: str | None = None
_W4LT_GS: int | None = None
_W4LT_ACT: str | None = None
_W4LT_FREE: bool | None = None

# (N, K) of the weights this phase is allowed to touch.
#   (4096, 2560)  GDN in_proj_qkvz (per rank)
#   (2560, 1536)  GDN out_proj
#   (62080, 2560) lm_head
#   (4160, 2560)  GDN in_proj_qkvz + in_proj_ba merged (VLLM_GFX908_GDN_MERGED_PROJ=1):
#                 4096 + 24 rows, zero-padded to a multiple of 64 so every MFMA nw config
#                 of the 4096 shape stays legal (260 n-tiles)
W8A16_SHAPES = frozenset({(4096, 2560), (4160, 2560), (2560, 1536), (62080, 2560)})

# Shapes whose bf16 master copy may be released under VLLM_GFX908_W8A16_FREE=gdn.
# lm_head (62080, 2560) is excluded: it runs at M = #sequences every decode step,
# so rematerialising its 318 MB per step costs far more than the 164 MB the int8
# copy duplicates.  ``all`` adds it anyway (N-chunked dequant).
W8A16_FREE_SHAPES_GDN = frozenset({(4096, 2560), (4160, 2560), (2560, 1536)})

# (YTILE, UNRL, LPR, KS) per shape and M, from the microbench sweeps
# (agents/w8a16/REPORT.md "Recommended dispatch").
_CFG: dict[tuple[int, int], dict[int, tuple[int, int, int, int]]] = {
    (4096, 2560): {1: (2, 1, 32, 1), 2: (4, 1, 32, 1), 3: (4, 1, 32, 1), 4: (4, 1, 32, 2)},
    (4160, 2560): {1: (2, 1, 32, 1), 2: (4, 1, 32, 1), 3: (4, 1, 32, 1), 4: (4, 1, 32, 2)},
    (2560, 1536): {1: (2, 1, 32, 1), 2: (2, 1, 32, 1), 3: (2, 1, 32, 1), 4: (2, 1, 32, 1)},
    (62080, 2560): {1: (8, 1, 32, 1), 2: (4, 1, 32, 1), 3: (4, 1, 32, 1), 4: (4, 1, 32, 1)},
}
# Fallback config for a whitelisted shape with no per-M entry.
_CFG_DEFAULT = (2, 1, 32, 1)

_FLAG: bool | None = None
_GS: int | None = None
_FREE: str | None = None
_SCRATCH_ELEMS: int | None = None


class _QEntry:
    """int8 side copy of one whitelisted weight.

    ``owner`` pins the tensor the cache key (its ``data_ptr``) belongs to so the
    storage cannot be recycled underneath us.  For a *freed* weight the key is
    the stub's ``data_ptr``, which aliases ``q``'s storage, so ``q`` itself is
    the pin and ``owner`` is None.
    """

    __slots__ = ("owner", "q", "s", "gs", "freed", "qsw", "ssw", "q4", "s4", "gs4")

    def __init__(self, owner, q, s, gs, freed, qsw=None, ssw=None, q4=None, s4=None, gs4=32):
        self.owner = owner
        self.q = q          # row-major [N, K] int8; None when the MFMA/W4 path owns the copy
        self.s = s          # [N, K/gs] fp32;       None when the MFMA/W4 path owns the copy
        self.gs = gs
        self.freed = freed
        self.qsw = qsw      # MFMA-swizzled int8 (same bytes as q, permuted)
        self.ssw = ssw      # [K/gs, N] fp32
        self.q4 = q4        # W4 packed [N, K/8] int32 (the only copy when set)
        self.s4 = s4        # W4 scales [N, K/gs4] bf16
        self.gs4 = gs4


# key (data_ptr, N, K, dev.type, dev.index) -> _QEntry
_Q_CACHE: dict[tuple, _QEntry] = {}
# device index -> reusable bf16 dequantization scratch (flat)
_SCRATCH: dict[int, torch.Tensor] = {}


@functools.cache
def _ext():
    """JIT-build (or load from the cache dir) the HIP W8A16 GEMV extension."""
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP W8A16 GEMV extension in %s", build_dir)
    return load(
        name="gfx908_w8a16_ext",
        sources=_SOURCES,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def w8a16_group_size() -> int:
    """Scale group size along K; 0 = per-output-channel scales."""
    global _GS
    if _GS is None:
        gs = int(os.environ.get("VLLM_GFX908_W8A16_GS", "128"))
        if gs != 0 and (gs < 16 or (gs & (gs - 1)) != 0):
            logger.warning(
                "gfx908 W8A16: VLLM_GFX908_W8A16_GS=%d is not 0 or a power of two "
                ">= 16; using 128",
                gs,
            )
            gs = 128
        _GS = gs
    return _GS


# ---------------------------------------------------------------------------
# MFMA path (VLLM_GFX908_W8A16_MFMA=1): pre-swizzled weights + a bf16 MFMA GEMM
# ---------------------------------------------------------------------------

# MFMA tile: v_mfma_f32_16x16x8bf16.  32 (v_mfma_f32_32x32x4bf16) is supported by
# the kernel but is not instantiated (it never won the sweep) -- see the .hip.
MFMA_TILE = 16
# The kernel covers 1 <= M <= 64; above that we fall back to dequant + stock bf16.
MFMA_MAX_M = 64

# (N, K) -> {M bucket: (tile, nw, sk, ug)} from the graph-timed sweep
# (agents/mfma_w8/REPORT.md section 1 + the M <= 8 probe).  A shape/M that is not
# in the table, or whose entry the kernel declines, falls through to the stock
# path (M <= 4) / dequant+stock (M > 4).
_MFMA_CFG: dict[tuple[int, int], dict[int, tuple[int, int, int, int]]] = {
    (62080, 2560): {1: (16, 2, 4, 1), 2: (16, 2, 4, 1), 4: (16, 4, 4, 1),
                    8: (16, 2, 2, 1), 16: (16, 4, 4, 1), 32: (16, 4, 4, 1),
                    48: (16, 4, 1, 1), 64: (16, 2, 2, 1)},
    (4096, 2560): {1: (16, 1, 4, 1), 2: (16, 1, 4, 1), 4: (16, 1, 4, 1),
                   8: (16, 1, 4, 1), 16: (16, 2, 4, 1), 32: (16, 4, 4, 1),
                   48: (16, 2, 4, 1), 64: (16, 2, 4, 1)},
    # merged in_proj (4096 + 24 + pad): same configs as 4096 (260 n-tiles, nw | 260)
    (4160, 2560): {1: (16, 1, 4, 1), 2: (16, 1, 4, 1), 4: (16, 1, 4, 1),
                   8: (16, 1, 4, 1), 16: (16, 2, 4, 1), 32: (16, 4, 4, 1),
                   48: (16, 2, 4, 1), 64: (16, 2, 4, 1)},
    (2560, 1536): {1: (16, 1, 4, 1), 2: (16, 1, 4, 1), 4: (16, 1, 4, 1),
                   8: (16, 1, 4, 1), 16: (16, 2, 4, 1), 32: (16, 2, 6, 2),
                   48: (16, 2, 6, 1), 64: (16, 2, 6, 1)},
}
_MFMA_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64)

# Shapes where the MFMA kernel beat the stock bf16 dispatch at 5 <= M <= 64 in the
# sweep, per M bucket.  Outside these, a weight that still owns its bf16 copy is
# better served by stock bf16, so `w8a16_gemm` returns None there (a *freed*
# weight has no bf16 copy and is served by the MFMA kernel regardless, which is
# still 1.3-1.6x better than rematerialising).
_MFMA_BEATS_STOCK: dict[tuple[int, int], int] = {
    (62080, 2560): 64,    # lm_head: 1.41-2.47x vs stock at 5 <= M <= 64
    (4096, 2560): 64,    # gdn in_proj_qkvz: 1.21-2.44x vs stock at 5 <= M <= 64
    (4160, 2560): 64,    # merged gdn in_proj (same kernel, +1.5% rows)
    (2560, 1536): 64,    # gdn out_proj: 1.17-1.89x vs stock at 5 <= M <= 64
}

# M = 1 reads the swizzled copy with a plain GEMV (csrc/gfx908_w8sw_gemv.cuh) instead of
# padding the M-tile to 16 rows in the MFMA kernel: same bytes, same launch count, no second
# resident layout.  (N, K) -> {M: (NT, UNRL, KS)} from the graph-timed L2-cold sweep
# (agents/mfma_gemv/REPORT.md).  Only M = 1 is listed: the MFMA kernel is nearly M-independent
# below 16 rows, so it already wins at M >= 2 (0.88-0.95x of the GEMV there) and a shape/M that
# is absent from this table falls through to it.
_WSW_CFG: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    (4096, 2560): {1: (1, 1, 4)},
    (4160, 2560): {1: (1, 1, 4)},
    (2560, 1536): {1: (1, 1, 8)},
    (62080, 2560): {1: (2, 1, 1)},
}


_WSW_FLAG: bool | None = None


def w8sw_gemv_enabled() -> bool:
    """VLLM_GFX908_W8A16_SWGEMV=0 disables the M <= 4 swizzled GEMV (default on).

    Escape hatch only: with the MFMA path on there is no row-major copy left, so turning
    this off puts M <= 4 back on the MFMA kernel with the M-tile padded to 16 rows.
    """
    global _WSW_FLAG
    if _WSW_FLAG is None:
        _WSW_FLAG = os.environ.get("VLLM_GFX908_W8A16_SWGEMV", "1") == "1"
    return _WSW_FLAG


def _wsw_cfg(n: int, k: int, m: int) -> tuple[int, int, int] | None:
    if not w8sw_gemv_enabled():
        return None
    per_m = _WSW_CFG.get((n, k))
    return None if per_m is None else per_m.get(m)


_MFMA_FLAG: bool | None = None


def w8a16_mfma_enabled() -> bool:
    """VLLM_GFX908_W8A16_MFMA=1 (default off) and the W8A16 path is on."""
    global _MFMA_FLAG
    if _MFMA_FLAG is None:
        if os.environ.get("VLLM_GFX908_W8A16_MFMA", "1") != "1":
            _MFMA_FLAG = False
        elif not w8a16_enabled():
            _MFMA_FLAG = False
        elif w8a16_group_size() != 128:
            logger.warning_once(
                "gfx908 W8A16: MFMA path needs group size 128 (got %d); disabled",
                w8a16_group_size(),
            )
            _MFMA_FLAG = False
        else:
            _MFMA_FLAG = True
            logger.info_once(
                "gfx908: W8A16 MFMA GEMM enabled (tile=%d, M <= %d); the swizzled "
                "int8 copy is the only resident one",
                MFMA_TILE,
                MFMA_MAX_M,
            )
    return _MFMA_FLAG


def swizzle_w8_mfma(q: torch.Tensor, tile: int = MFMA_TILE) -> torch.Tensor:
    """[N, K] int8 -> MFMA-B fragments, 1 KiB per (tile n x 1024/tile k) tile.

    Same bytes as ``q``, permuted: lane ``l`` of a wave64 reads the 16 B at
    ``tile_base + 16*l``, i.e. n = n_tile*tile + l%tile,
    k = k_tile*(1024/tile) + 16*(l/tile) + 0..15.
    Built in row chunks so the permutation temporary stays small (lm_head is
    152 MiB and ``.contiguous()`` on the whole 5-D permute would peak at 2x).
    """
    n, k = q.shape
    kpt = 1024 // tile
    assert n % tile == 0 and k % kpt == 0, (n, k, tile)
    out = torch.empty_like(q)
    step = max(tile, ((1 << 24) // max(k, 1)) // tile * tile)
    ov = out.view(n // tile, tile * k)
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        ov[i0 // tile : i1 // tile] = (
            q[i0:i1]
            .view((i1 - i0) // tile, tile, k // kpt, kpt // 16, 16)
            .permute(0, 2, 3, 1, 4)
            .reshape((i1 - i0) // tile, tile * k)
        )
    return out


def swizzle_s_mfma(s: torch.Tensor) -> torch.Tensor:
    """[N, K/gs] fp32 -> [K/gs, N] fp32 contiguous (coalesced per n-tile)."""
    return s.t().contiguous()


def _mfma_bucket(m: int) -> int:
    for b in _MFMA_BUCKETS:
        if m <= b:
            return b
    return _MFMA_BUCKETS[-1]


def _mfma_cfg(n: int, k: int, m: int) -> tuple[int, int, int, int] | None:
    per_m = _MFMA_CFG.get((n, k))
    if per_m is None:
        return None
    return per_m.get(_mfma_bucket(m))


def w8a16_enabled() -> bool:
    """True when the opt-in env flag is set and the extension actually builds."""
    global _FLAG
    if _FLAG is None:
        if os.environ.get("VLLM_GFX908_W8A16", "0") != "1":
            _FLAG = False
            return _FLAG
        try:
            from vllm.platforms.rocm import on_gfx908

            _FLAG = bool(on_gfx908())
        except Exception:
            _FLAG = False
        if _FLAG:
            try:
                _ext()
            except Exception as exc:  # hipcc missing etc. -> stock bf16 path
                if os.environ.get("VLLM_GFX908_STRICT_EXT", "1") == "1":
                    raise RuntimeError("gfx908: W8A16 extension unavailable under its flag; set VLLM_GFX908_STRICT_EXT=0 to fall back") from exc
                logger.warning_once("gfx908: W8A16 extension unavailable (%s)", exc)
                _FLAG = False
        if _FLAG:
            logger.info_once(
                "gfx908: W8A16 decode GEMV enabled (group_size=%d, shapes=%s)",
                w8a16_group_size(),
                ", ".join(f"{n}x{k}" for n, k in sorted(W8A16_SHAPES)),
            )
    return _FLAG


@functools.cache
def _cu_count() -> int:
    from vllm.utils.platform_utils import num_compute_units

    return int(num_compute_units())


# ---------------------------------------------------------------------------
# W4 GS32 load-time quantization
# ---------------------------------------------------------------------------


@functools.cache
def _ext_w4lt():
    """The dense W4 GEMV / epilogue / dequant extension (shared with the HC path)."""
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w8a16")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading W4 load-time extension in %s", build_dir)
    return load(
        name="gfx908_w4lt_ext",
        sources=[_W4LT_SRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def w4lt_policy() -> str:
    """``VLLM_GFX908_W4_LOADTIME``: off | gdn | hc | all (default off).

    ``gdn`` quantizes the two GDN projections handled by this module, ``hc`` the
    two hyper-connection mixes (``models/qwen4_exp/amd/gfx908_hc_fused.py``),
    ``all`` both.
    """
    global _W4LT_POLICY
    if _W4LT_POLICY is None:
        v = os.environ.get("VLLM_GFX908_W4_LOADTIME", "off").strip().lower()
        if v in ("0", "", "none"):
            v = "off"
        if v in ("1", "true"):
            v = "all"
        if v not in ("off", "gdn", "hc", "all"):
            logger.warning(
                "gfx908 W4: VLLM_GFX908_W4_LOADTIME=%r is not off/gdn/hc/all; using off", v
            )
            v = "off"
        _W4LT_POLICY = v
    return _W4LT_POLICY


def w4lt_gs() -> int:
    """``VLLM_GFX908_W4_LOADTIME_GS``: 32 (default) or 64.

    64 saves only 5% of the weight bytes (0.266x bf16 vs 0.281x) and costs ~15%
    more quantization error on every real tensor measured, so it exists for the
    A/B only.
    """
    global _W4LT_GS
    if _W4LT_GS is None:
        gs = int(os.environ.get("VLLM_GFX908_W4_LOADTIME_GS", "32"))
        if gs not in (32, 64):
            logger.warning("gfx908 W4: group size %d is not 32/64; using 32", gs)
            gs = 32
        _W4LT_GS = gs
    return _W4LT_GS


def w4lt_act() -> str:
    """``VLLM_GFX908_W4_LOADTIME_ACT``: bf16 (default) | f16 | int8 | auto.

    ``bf16`` needs no prep launch (the cast + permutation ride along with the
    kernel's LDS staging) and is numerically identical to ``f16``.  ``auto``
    follows ``VLLM_GFX908_W4A8_MODE`` (``int8`` -> int8, otherwise ``bf16``).
    """
    global _W4LT_ACT
    if _W4LT_ACT is None:
        v = os.environ.get("VLLM_GFX908_W4_LOADTIME_ACT", "bf16").strip().lower()
        if v == "auto":
            v = "int8" if os.environ.get("VLLM_GFX908_W4A8_MODE", "int8") == "int8" else "bf16"
        if v not in ("bf16", "f16", "int8"):
            logger.warning("gfx908 W4: activation mode %r is not bf16/f16/int8; using bf16", v)
            v = "bf16"
        _W4LT_ACT = v
    return _W4LT_ACT


def w4lt_free() -> bool:
    """``VLLM_GFX908_W4_LOADTIME_FREE=0`` keeps the bf16 masters (default: release).

    Keeping them costs 0.5 GB/rank for the GDN pair but leaves ``M > 8`` on the
    untouched bf16 dispatch instead of paying a rematerialisation per call.
    """
    global _W4LT_FREE
    if _W4LT_FREE is None:
        _W4LT_FREE = os.environ.get("VLLM_GFX908_W4_LOADTIME_FREE", "1") == "1"
    return _W4LT_FREE


# lanes-per-row per shape, from the graph-timed sweep (agents/w4_loadtime/sweep.log).
# 16 and 32 are within 2% of each other at every shape/M; 64 always loses.
_W4LT_LPR: dict[tuple[int, int], int] = {(4096, 2560): 32, (2560, 1536): 32}

_W4LT_FLAG: bool | None = None


def w4lt_available() -> bool:
    """The extension builds (cached).  Independent of which shapes the policy covers."""
    global _W4LT_FLAG
    if _W4LT_FLAG is None:
        _W4LT_FLAG = False
        if w4lt_policy() != "off":
            try:
                from vllm.platforms.rocm import on_gfx908

                _W4LT_FLAG = bool(on_gfx908()) and _ext_w4lt() is not None
            except Exception as exc:
                if os.environ.get("VLLM_GFX908_STRICT_EXT", "1") == "1":
                    raise RuntimeError("gfx908: W4 load-time path unavailable under its flag; set VLLM_GFX908_STRICT_EXT=0 to fall back") from exc
                logger.warning_once("gfx908: W4 load-time path unavailable (%s)", exc)
                _W4LT_FLAG = False
            if _W4LT_FLAG:
                logger.info_once(
                    "gfx908: W4 load-time quantization enabled (policy=%s, gs=%d, act=%s, "
                    "free_bf16=%s) -- RTN at 4 bits, gate on PPL/GSM8K before shipping",
                    w4lt_policy(),
                    w4lt_gs(),
                    w4lt_act(),
                    w4lt_free(),
                )
    return _W4LT_FLAG


def _reset_env_cache():
    """Re-read every VLLM_GFX908_W8A16* / W4_LOADTIME env (tests only; extensions stay cached)."""
    global _FLAG, _GS, _FREE, _MFMA_FLAG, _W4LT_POLICY, _W4LT_GS, _W4LT_ACT, _W4LT_FREE
    global _WSW_FLAG
    global _W4LT_FLAG
    _FLAG = _GS = _FREE = _MFMA_FLAG = _WSW_FLAG = None
    _W4LT_POLICY = _W4LT_GS = _W4LT_ACT = _W4LT_FREE = _W4LT_FLAG = None
    _Q_CACHE.clear()


def w4lt_gdn_enabled() -> bool:
    """W4 for the GDN projections owned by this module."""
    return w4lt_policy() in ("gdn", "all") and w8a16_enabled() and w4lt_available()


def quantize_w4(
    weight: torch.Tensor, gs: int = 32, rule: str = "q4_0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric RTN to the W4 GS32 layout the kernels read.

    Returns ``(packed int32 [N, K/8], scales bf16 [N, K/gs])`` with nibble ``t``
    of word ``wd`` holding ``k = 8*wd + t`` and a zero point of 8, i.e.
    ``w ~= scale * (nibble - 8)`` -- byte-identical to the MoE checkpoint's W4
    layout, minus the expert dimension.

    ``rule`` ``q4_0`` (default, llama.cpp): the largest-magnitude element of the
    group defines the scale so that it lands exactly on level -8.  ``amax7``
    uses ``amax/7``, which never clips but wastes the -8 level; measured 11%
    worse on every tensor tried (agents/w4_loadtime/correct.log).

    Chunked over rows so the fp32 temporary stays small.
    """
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % gs or gs % 32:
        raise ValueError(f"quantize_w4: K={k} must be a multiple of gs={gs} (and gs of 32)")
    packed = torch.empty((n, k // 8), dtype=torch.int32, device=weight.device)
    scales = torch.empty((n, k // gs), dtype=torch.bfloat16, device=weight.device)
    step = max(1, (1 << 22) // max(k, 1))
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        wf = weight[i0:i1].float().view(i1 - i0, k // gs, gs)
        if rule == "amax7":
            sc = wf.abs().amax(-1).clamp_min(1e-12) / 7.0
        else:
            amax = wf.abs().amax(-1)
            idx = wf.abs().argmax(-1, keepdim=True)
            sc = torch.gather(wf, -1, idx).squeeze(-1) / -8.0
            sc = torch.where(amax > 0, sc, torch.ones_like(sc))
            sc = torch.where(sc.abs() < 1e-12, torch.full_like(sc, 1e-12), sc)
        q = (torch.round(wf / sc[..., None]) + 8.0).clamp_(0, 15).to(torch.int32)
        q = q.view(i1 - i0, k // 8, 8)
        acc = q[:, :, 0].clone()
        for t in range(1, 8):
            acc |= q[:, :, t] << (4 * t)
        packed[i0:i1] = acc
        scales[i0:i1] = sc.to(torch.bfloat16)
    return packed, scales


def w4lt_gemv(
    q4: torch.Tensor, s4: torch.Tensor, x: torch.Tensor, gs: int, n: int
) -> torch.Tensor | None:
    """W4 GEMV for ``1 <= M <= W4LT_MAX_M``; bf16 [M, N] out, or None if declined.

    ``x`` is bf16 [M, K] and contiguous.  The activation mode is
    ``w4lt_act()``; ``bf16`` needs no prep launch.
    """
    ext = _ext_w4lt()
    m, k = int(x.shape[0]), int(x.shape[1])
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    act = w4lt_act()
    try:
        if act == "bf16":
            ext.w4lt_gemv_bf16(q4, s4, x, out, gs, _W4LT_LPR.get((n, k), 16))
        elif act == "f16":
            from vllm.model_executor.layers.fused_moe.gfx908_w4a8 import cast_f16

            ext.w4lt_gemv_f16(q4, s4, cast_f16(x), out, gs, 16, 1, 4)
        else:
            from vllm.model_executor.layers.fused_moe.gfx908_w4a8 import quant_q8

            x8, xs, xsum = quant_q8(x)
            ext.w4lt_gemv_i8(q4, s4, x8, xs, xsum, out, gs, 16, 1, 4)
    except Exception as exc:  # unsupported K etc.
        logger.warning_once("gfx908 W4: GEMV declined %dx%d (%s)", n, k, exc)
        return None
    return out


def quantize_w8(weight: torch.Tensor, gs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax RTN int8 quantization of a [N, K] bf16/fp16 weight.

    gs == 0  -> per-output-channel scales, fp32 [N]
    gs  > 0  -> group scales along K,      fp32 [N, K // gs]

    Chunked over rows so the fp32 temporary stays small (lm_head would be 636 MB
    in one shot).
    """
    n, k = weight.shape
    assert gs == 0 or (gs >= 16 and (gs & (gs - 1)) == 0 and k % gs == 0)
    q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    s = torch.empty(
        (n,) if gs == 0 else (n, k // gs), dtype=torch.float32, device=weight.device
    )
    step = max(1, (1 << 22) // max(k, 1))  # ~4M elements per chunk
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        wf = weight[i0:i1].float()
        if gs == 0:
            sc = wf.abs().amax(1).clamp_min(1e-12) / 127.0
            q[i0:i1] = torch.round(wf / sc[:, None]).clamp_(-127, 127).to(torch.int8)
        else:
            v = wf.view(i1 - i0, k // gs, gs)
            sc = v.abs().amax(2).clamp_min(1e-12) / 127.0
            q[i0:i1] = (
                torch.round(v / sc[..., None])
                .clamp_(-127, 127)
                .to(torch.int8)
                .view(i1 - i0, k)
            )
        s[i0:i1] = sc
    return q, s


def _cache_key(weight: torch.Tensor) -> tuple:
    dev = weight.device
    return (weight.data_ptr(), weight.shape[0], weight.shape[1], dev.type, dev.index)


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# bf16 master-copy release
# ---------------------------------------------------------------------------


def w8a16_free_policy() -> str:
    """Which whitelisted shapes lose their bf16 master copy: gdn | all | none."""
    global _FREE
    if _FREE is None:
        v = os.environ.get("VLLM_GFX908_W8A16_FREE", "gdn").strip().lower()
        if v not in ("gdn", "all", "none"):
            logger.warning(
                "gfx908 W8A16: VLLM_GFX908_W8A16_FREE=%r is not gdn/all/none; using gdn",
                v,
            )
            v = "gdn"
        _FREE = v
    return _FREE


def _should_free(shape: tuple[int, int]) -> bool:
    pol = w8a16_free_policy()
    if pol == "none":
        return False
    if pol == "all":
        return shape in W8A16_SHAPES
    return shape in W8A16_FREE_SHAPES_GDN


def _scratch_elems_cap() -> int:
    """Upper bound (in bf16 elements) on the persistent dequantization scratch."""
    global _SCRATCH_ELEMS
    if _SCRATCH_ELEMS is None:
        mb = float(os.environ.get("VLLM_GFX908_W8A16_SCRATCH_MB", "32"))
        _SCRATCH_ELEMS = max(1 << 16, int(mb * (1 << 20)) // 2)
    return _SCRATCH_ELEMS


def _ensure_scratch(device: torch.device, elems: int) -> torch.Tensor:
    """Reusable flat bf16 scratch, grown on demand.  Allocated eagerly at
    prepare time so no allocation happens under cudagraph capture."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cur = _SCRATCH.get(idx)
    if cur is None or cur.numel() < elems:
        if cur is not None:
            del _SCRATCH[idx]
            del cur
        _SCRATCH[idx] = torch.empty(elems, dtype=torch.bfloat16, device=device)
        logger.info_once(
            "gfx908 W8A16: dequantization scratch %.1f MB on cuda:%d",
            elems * 2 / 2**20,
            idx,
        )
    return _SCRATCH[idx]


def _make_stub(q: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """A bf16 tensor with the weight's shape/dtype/device that owns no storage.

    It aliases the int8 copy's first bytes with stride (0, 0): ``.shape``,
    ``.dtype``, ``.device`` and ``.data_ptr()`` are all well defined (the last
    one uniquely identifies the weight and stays pinned by ``q``), while the
    N*K bf16 elements it claims cost nothing.  Nothing ever reads through it:
    every GEMM for a freed weight is served from ``q`` + scales.
    """
    return q.view(torch.bfloat16).as_strided(tuple(shape), (0, 0))


def is_w8a16_freed(weight: torch.Tensor) -> bool:
    """True if `weight` is a stub whose bf16 storage has been released."""
    if weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.is_meta:
        return False
    ent = _Q_CACHE.get(_cache_key(weight))
    return ent is not None and ent.freed


def _quantize_and_cache(weight: torch.Tensor, allow_free: bool) -> _QEntry:
    gs = w8a16_group_size()
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if w4lt_gdn_enabled() and (n, k) in W4LT_GDN_SHAPES:
        # W4 replaces the int8 copy entirely: 0.28x the bf16 bytes instead of 0.52x.
        gs4 = w4lt_gs()
        q4, s4 = quantize_w4(weight, gs4)
        freed = False
        if allow_free and w4lt_free():
            weight.data = _make_stub(q4, (n, k))   # releases the bf16 storage
            _ensure_scratch(q4.device, min(_scratch_elems_cap(), n * k))
            freed = True
        ent = _QEntry(None if freed else weight, None, None, gs, freed,
                      q4=q4, s4=s4, gs4=gs4)
        _Q_CACHE[_cache_key(weight)] = ent
        logger.info_once(
            "gfx908 W4: quantized weight %dx%d to W4 gs%d (act=%s, freed=%s); "
            "%.1f MB -> %.1f MB per weight",
            n, k, gs4, w4lt_act(), freed,
            n * k * 2 / 2**20, (n * k / 2 + n * (k // gs4) * 2) / 2**20,
        )
        return ent
    q, s = quantize_w8(weight, gs)
    qsw = ssw = None
    if w8a16_mfma_enabled() and (n, k) in _MFMA_CFG and n % MFMA_TILE == 0 and k % 128 == 0:
        # The MFMA kernel needs a different byte order than the M <= 4 GEMV and both
        # layouts would be a second full int8 copy, so the swizzled one becomes the
        # ONLY resident copy and the row-major original is released here.
        qsw = swizzle_w8_mfma(q, MFMA_TILE)
        ssw = swizzle_s_mfma(s)
        del q, s
        q = s = None
    pin = qsw if q is None else q
    if allow_free and _should_free((n, k)):
        stub = _make_stub(pin, (n, k))
        # Releases the bf16 storage: the parameter's TensorImpl now points at
        # the stub, and nothing else holds a reference to the old data.
        weight.data = stub
        ent = _QEntry(None, q, s, gs, True, qsw, ssw)
        _Q_CACHE[_cache_key(weight)] = ent
        _ensure_scratch(pin.device, min(_scratch_elems_cap(), n * k))
        logger.info_once(
            "gfx908 W8A16: released bf16 master copy for %dx%d weights "
            "(policy=%s); int8 is now the only resident copy",
            n,
            k,
            w8a16_free_policy(),
        )
    else:
        ent = _QEntry(weight, q, s, gs, False, qsw, ssw)
        _Q_CACHE[_cache_key(weight)] = ent
    logger.info_once(
        "gfx908 W8A16: quantized weight %dx%d (gs=%d, layout=%s); %d entries cached",
        n,
        k,
        gs,
        "mfma-swizzled" if qsw is not None else "row-major",
        len(_Q_CACHE),
    )
    return ent


def get_w8a16_weight(weight: torch.Tensor):
    """Return the cached (int8, scales) side tensors for `weight`, or None.

    Builds them on the first eager call.  Never quantizes while a cudagraph is
    being captured (allocation + a sync during capture is illegal / would leak
    into the graph pool) - the caller falls back to bf16 in that case.  The
    lazy path never releases the bf16 copy (see module docstring).
    """
    ent = _Q_CACHE.get(_cache_key(weight))
    if ent is None:
        if _capturing():
            return None
        ent = _quantize_and_cache(weight, allow_free=False)
    if ent.q is None:  # MFMA path: only the swizzled copy is resident
        return None
    return ent.q, ent.s


def prepare_w8a16_weight(weight: torch.Tensor) -> bool:
    """Eagerly materialise the int8 side tensors and, when the free policy says
    so, release the bf16 master copy in place (``weight.data`` is rebound).

    Must run before torch.compile / cudagraph capture, i.e. from
    ``process_weights_after_loading``.  Returns True when the weight is cached.
    """
    if not w8a16_enabled() or not w8a16_weight_applies(weight):
        return False
    if _cache_key(weight) in _Q_CACHE:
        return True
    _quantize_and_cache(weight, allow_free=True)
    return True


def prepare_w8a16_layer(layer, allow_free: bool = True) -> bool:
    """``process_weights_after_loading`` hook: quantize + (policy permitting)
    free ``layer.weight``.  Safe to call on any layer; a no-op unless the flag
    is on and the weight is whitelisted.

    ``allow_free=False`` (or ``layer._w8a16_keep_bf16 = True``) keeps the bf16
    master copy regardless of policy - used for layers whose weight may be read
    outside the GEMM dispatch (e.g. a ``ParallelLMHead`` tied to
    ``embed_tokens``, where ``F.embedding`` reads the weight directly).
    """
    w = getattr(layer, "weight", None)
    if w is None or not isinstance(w, torch.Tensor):
        return False
    if not w8a16_enabled() or not w8a16_weight_applies(w):
        return False
    if _cache_key(w) in _Q_CACHE:
        return True
    keep = getattr(layer, "_w8a16_keep_bf16", False)
    try:
        _quantize_and_cache(w, allow_free=allow_free and not keep)
        return True
    except Exception as exc:  # never break model loading over this
        logger.warning_once("gfx908 W8A16: prepare failed for %s (%s)", type(layer), exc)
        return False


def w8a16_weight_applies(weight: torch.Tensor) -> bool:
    if (
        weight.dim() != 2
        or weight.dtype != torch.bfloat16
        or weight.is_meta
        or (weight.shape[0], weight.shape[1]) not in W8A16_SHAPES
    ):
        return False
    # A freed weight is a zero-stride stub, hence not contiguous; it is valid
    # exactly because the cache already holds its int8 copy.
    return weight.is_contiguous() or _cache_key(weight) in _Q_CACHE


def w8a16_applies(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Cheap shape/dtype gate for the M <= 4 GEMV (does not touch the cache)."""
    if bias is not None or x.dtype != torch.bfloat16:
        return False
    if not w8a16_weight_applies(weight):
        return False
    m = x.numel() // x.size(-1)
    return 1 <= m <= W8A16_MAX_M


def _cfg_for(n: int, k: int, m: int) -> tuple[int, int, int, int]:
    per_m = _CFG.get((n, k))
    if per_m is None:
        return _CFG_DEFAULT
    return per_m.get(m, per_m.get(W8A16_MAX_M, _CFG_DEFAULT))


def dequant_w8_into(
    q: torch.Tensor, s: torch.Tensor, out: torch.Tensor, gs: int
) -> bool:
    """out[:] = bf16(q * s), via the HIP kernel (torch fallback if it declines)."""
    if _ext().w8a16_dequant(q, s, out, gs):
        return True
    if gs == 0:
        out.copy_((q.float() * s[:, None]).to(torch.bfloat16))
    else:
        r, k = q.shape
        out.copy_(
            (q.float().view(r, k // gs, gs) * s[..., None]).to(torch.bfloat16).view(r, k)
        )
    return True


def _dequant_into(ent: _QEntry, r0: int, r1: int, w: torch.Tensor, n: int, k: int) -> None:
    """w[:] = bf16(weight rows [r0, r1)), from whichever quantized layout is resident."""
    if ent.q4 is not None:
        _ext_w4lt().w4lt_dequant(ent.q4, ent.s4, w, r0, ent.gs4)
        return
    if ent.qsw is not None:
        qsw = ent.qsw.view(n // MFMA_TILE, MFMA_TILE * k)[r0 // MFMA_TILE : r1 // MFMA_TILE]
        if _ext().mfma_w8_dequant(qsw.reshape(-1), ent.ssw, w, r0, n):
            return
        raise RuntimeError("w8a16: mfma_w8_dequant declined")
    dequant_w8_into(ent.q[r0:r1], ent.s[r0:r1], w, ent.gs)


def _dequant_gemm(
    x: torch.Tensor,
    ent: _QEntry,
    n: int,
    k: int,
    bias: torch.Tensor | None,
    bf16_fn,
) -> torch.Tensor:
    """M > 64 (or an unsupported config): rematerialise the weight into the
    scratch in N-row chunks and run the stock bf16 GEMM dispatch per chunk."""
    dev = x.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    scratch = _SCRATCH.get(idx)
    want = min(_scratch_elems_cap(), n * k)
    if scratch is None or scratch.numel() < want:
        if _capturing():
            # Cannot grow the scratch under capture; caller falls back.
            raise RuntimeError("w8a16: dequant scratch missing during capture")
        scratch = _ensure_scratch(dev, want)
    fit = scratch.numel() // k
    # the swizzled dequant works on whole 16-row tiles
    align = MFMA_TILE if ent.qsw is not None else 8
    rows = min(n, max(align, (fit // align) * align))
    if rows >= n:
        w = scratch[: n * k].view(n, k)
        _dequant_into(ent, 0, n, w, n, k)
        return bf16_fn(x, w, bias)
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=dev)
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        w = scratch[: (r1 - r0) * k].view(r1 - r0, k)
        _dequant_into(ent, r0, r1, w, n, k)
        out[..., r0:r1] = bf16_fn(x, w, None if bias is None else bias[r0:r1])
    return out


def w8a16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    bf16_fn=None,
) -> torch.Tensor | None:
    """int8-weight / bf16-activation GEMM.  Returns None to fall back to bf16.

    Without ``VLLM_GFX908_W8A16_MFMA``: ``M <= 4`` uses the row-major W8A16 GEMV;
    ``M > 4`` is only served when the weight's bf16 master copy has been released
    (otherwise stock bf16 is both faster and bit-identical), by dequantizing into
    the scratch and calling ``bf16_fn`` (default ``F.linear``) per N-chunk.

    With ``VLLM_GFX908_W8A16_MFMA=1`` the swizzled int8 copy is the only resident
    one, so the MFMA GEMM serves every ``1 <= M <= MFMA_MAX_M`` (64): at ``M <= 4``
    unconditionally (there is no row-major copy left to run the GEMV on), above
    that only when the weight is freed or ``_MFMA_BEATS_STOCK`` says we beat the
    stock bf16 dispatch for this shape and M.  ``M > 64`` still goes through
    ``_dequant_gemm``, now rematerialising from the swizzled copy.

    With ``VLLM_GFX908_W4_LOADTIME`` covering this shape the W4 copy is the only
    resident one and serves ``1 <= M <= W4LT_MAX_M`` (8); above that
    ``_dequant_gemm`` rematerialises from it.
    """
    if weight.dim() != 2:
        return None
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (n, k) not in W8A16_SHAPES:
        return None
    m = x.numel() // x.size(-1)
    ent = _Q_CACHE.get(_cache_key(weight))
    if ent is None:
        # Lazy (no load-time hook): only worth it for the decode GEMV.
        max_lazy = (
            W4LT_MAX_M
            if (w4lt_gdn_enabled() and (n, k) in W4LT_GDN_SHAPES)
            else W8A16_MAX_M
        )
        if x.dtype != torch.bfloat16 or bias is not None or m < 1 or m > max_lazy:
            return None
        if not weight.is_contiguous() or weight.is_meta or _capturing():
            return None
        ent = _quantize_and_cache(weight, allow_free=False)
    elif x.dtype != torch.bfloat16:
        # A freed weight has no bf16 copy to fall back to, so serve it anyway
        # (the stock path would cast x to the weight dtype too); otherwise let
        # the caller take the bit-identical bf16 route.
        if not ent.freed:
            return None
        x = x.to(torch.bfloat16)

    if ent.q4 is not None:
        # W4 owns the only resident copy; it serves 1 <= M <= W4LT_MAX_M directly and
        # everything above through the rematerialisation path below.
        if bias is None and 1 <= m <= W4LT_MAX_M and x.dtype == torch.bfloat16:
            x2 = x.reshape(-1, k)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            w4out = w4lt_gemv(ent.q4, ent.s4, x2, ent.gs4, n)
            if w4out is not None:
                return w4out.reshape(*x.shape[:-1], n)
    elif ent.qsw is not None:
        # MFMA path owns the only int8 copy: it serves every M it covers.  At M <= 4 a plain
        # GEMV over the same swizzled bytes is cheaper than the MFMA kernel's 16-row M padding,
        # so try it first and fall through to the MFMA kernel if it declines this config.
        wcfg = _wsw_cfg(n, k, m) if (bias is None and 1 <= m <= W8A16_MAX_M) else None
        if wcfg is not None and x.dtype == torch.bfloat16:
            x2 = x.reshape(-1, k)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            nt, unrl, ks = wcfg
            if _ext().w8sw_gemv(
                ent.qsw, ent.ssw, x2, out, ent.gs, nt, unrl, ks, _cu_count()
            ):
                return out.reshape(*x.shape[:-1], n)
            del out
        if bias is None and 1 <= m <= MFMA_MAX_M:
            cfg = _mfma_cfg(n, k, m)
            # Above M = 4 a weight that still has its bf16 master copy is only
            # taken when the sweep says we beat the stock bf16 dispatch; a freed
            # weight has no bf16 copy, so we take it regardless (still well ahead
            # of rematerialising).
            ok = cfg is not None and (
                m <= W8A16_MAX_M
                or ent.freed
                or _mfma_bucket(m) <= _MFMA_BEATS_STOCK.get((n, k), 0)
            )
            if ok:
                x2 = x.reshape(-1, k)
                if not x2.is_contiguous():
                    x2 = x2.contiguous()
                out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
                tile, nw, sk, ug = cfg
                if _ext().mfma_w8_gemm(ent.qsw, ent.ssw, x2, out, tile, nw, sk, ug, 0):
                    return out.reshape(*x.shape[:-1], n)
                del out
    elif m <= W8A16_MAX_M and m >= 1 and bias is None:
        x2 = x.reshape(-1, k)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        ytile, unrl, lpr, ks = _cfg_for(n, k, m)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        if _ext().w8a16_gemv(
            ent.q, ent.s, x2, out, ent.gs, ytile, unrl, lpr, ks, 0, _cu_count()
        ):
            return out.reshape(*x.shape[:-1], n)
        del out
        # launcher declined this config -> fall through (dequant if freed)

    if not ent.freed:
        return None
    try:
        return _dequant_gemm(
            x, ent, n, k, bias, bf16_fn or torch.nn.functional.linear
        )
    except RuntimeError:
        # Freed weight and we cannot dequantize: there is no bf16 copy to fall
        # back to, so this must not silently return None.
        raise
