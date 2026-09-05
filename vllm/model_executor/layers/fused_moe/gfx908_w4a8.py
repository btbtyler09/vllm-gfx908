# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 W4 GEMV path for decode-sized M (<= 8): env ``VLLM_GFX908_W4A8=1`` (default off).

Weights stay W4 GS32 (symmetric, zero point 8) in the stock ``[E, N, K/8]`` nibble layout; only the
activation handling differs between the two modes of ``VLLM_GFX908_W4A8_MODE``:

``int8`` (default, the original behaviour)
    activations are quantized per block of 32 to int8 with an fp32 scale and fp32 scale*sum
    (llama.cpp Q8_1 style) and the GEMV runs int8 dot4 (``v_dot4_i32_i8``)
    -- csrc/gfx908_w4a8.hip, design D1/D2.  ~2.0-5.0x the stock W4A16 HIP GEMV; A8 error ~6e-3
    relative, so it needs the server accuracy gate.

``f16`` (W4A16-exact)
    activations are cast bf16 -> fp16 (lossless: bf16 has 8 mantissa bits, fp16 has 11) and the
    GEMV runs packed-fp16 dot2 (``v_dot2_f32_f16``), with the 1024 magic bias and the zero point
    folded away exactly by one ``v_pk_add_f16``
    -- csrc/gfx908_w4f16.hip, design D5.  1.8-4.2x the stock GEMV and numerically identical to it
    (5e-7 relative L2 = fp32 summation noise), i.e. a pure speed change with no accuracy risk;
    costs 6-41% more time than ``int8``.  Caveat: fp16 tops out at 65504 -- the microbench audited
    real activations at max|x| = 40, and ``cast_f16_audit`` in the .hip re-checks that on demand.

Both modes run with no split-K and feed the existing Triton reduce epilogues with ``SPLIT_K=1``.
Microbench numbers: agents/w4a8/REPORT.md and agents/w4f16/REPORT.md.

Three consumers, all inside custom ops / eager MoE code where M is concrete:
  * routed experts   (gfx908_moe_hip.gfx908_moe_hip)    -> moe_w4a8
  * shared expert    (gfx908_shared_expert)             -> shared_expert_w4a8
  * dense W4A16 GEMV (triton_w4a16._gfx908_w4a16_gemv)  -> dense_w4a8_gemv  (K in {2560, 1536})
The mode is selected inside this module, so the three hooks are mode-agnostic.
Dense / shared weights are ``[K, N/8]`` int32 K-major (N packed); they are repacked once to the
MoE ``[1, N, K/8]`` K-packed layout and cached by data_ptr (never during graph capture: the
caller falls back to the stock path until the first eager call has populated the cache).

``VLLM_GFX908_SHARED_AS_EXPERT=1`` (default off, needs the flag above) additionally folds the
shared expert into the routed GEMVs as "expert #E": the slab / rowlane kernels take an optional
extra weight+scale pointer pair selected when ``row_expert == E``, one extra row per token is
appended to the row lists with weight ``sigmoid(x . w_gate)`` (built by extra blocks of the
activation-prep kernel, so no extra launch), and the existing weighted-sum reduce emits
routed + shared in one pass.  See the "shared expert as expert #E" section below and
agents/sharedexp/INTEGRATION.md.
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# batch_research 2026-09-05: the W4A8 GEMV path beats the Triton gptq_awq MoE at every M measured up to 256
# (agents/batch_research/REPORT.md); above MOE_MR_MIN_M rows the expert-deduplicated multi-row kernels run.
W4A8_MAX_TOKENS = int(os.environ.get("VLLM_GFX908_W4A8_MAX_M", "256"))
DENSE_W4A8_MAX_TOKENS = 8            # dense projections keep the per-row GEMV only at decode M
MOE_MR_MIN_M = int(os.environ.get("VLLM_GFX908_MOE_MR_MIN_M", "24"))
# multi-row kernel configuration (measured best, M = 48..256): LDS-staged slab, 8 waves/block, 8 staged rows,
# occupancy-forced (pf=4), 960 persistent blocks; rowlane 4 waves/block, 960 blocks; sort chunk 16 rows.
MR_CFG = dict(rc=16, wpb=8, rcs=8, pf1=4, blocks=960, wpb2=4, blocks2=960, pf2=0)
GS = 32
_SLAB_K = (2560, 1536)          # K values instantiated for the slab kernels (SLAB_CASE in the .hip)
_ROWLANE_K = 160
_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_CSRC = os.path.join(_CSRC_DIR, "gfx908_w4a8.hip")
_CSRC_F16 = os.path.join(_CSRC_DIR, "gfx908_w4f16.hip")
_CSRC_MR = os.path.join(_CSRC_DIR, "gfx908_moe_mr.hip")
_MODES = ("int8", "f16")
_FLAG: bool | None = None
_MODE: str | None = None
_SHARED_AS_EXPERT: bool | None = None
_PREP_FOLD: bool | None = None
_SILU_FOLD: bool | None = None


def _load(name: str, src: str, subdir: str):
    from torch.utils.cpp_extension import load

    base = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    build_dir = os.path.join(base, subdir)
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP %s GEMV extension in %s", subdir, build_dir)
    return load(
        name=name,
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


@functools.cache
def _ext():
    """int8-activation (W4A8) extension."""
    return _load("gfx908_w4a8_ext", _CSRC, "w4a8")


@functools.cache
def _ext_mr():
    """expert-deduplicated multi-row kernels (int8 mode only)."""
    return _load("gfx908_moe_mr_ext", _CSRC_MR, "moe_mr")


def moe_mr_enabled() -> bool:
    """VLLM_GFX908_MOE_MR (default 1): multi-row flow for M > MOE_MR_MIN_M (int8 mode only)."""
    return os.environ.get("VLLM_GFX908_MOE_MR", "1") == "1" and w4a8_mode() == "int8"


@functools.cache
def _ext_f16():
    """fp16-activation (W4A16-exact) extension."""
    return _load("gfx908_w4f16_ext", _CSRC_F16, "w4f16")


def w4a8_mode() -> str:
    """VLLM_GFX908_W4A8_MODE: 'int8' (default) or 'f16' (W4A16-exact). Cached."""
    global _MODE
    if _MODE is None:
        m = os.environ.get("VLLM_GFX908_W4A8_MODE", "int8").strip().lower()
        if m not in _MODES:
            logger.warning_once(
                "gfx908: VLLM_GFX908_W4A8_MODE=%r is not one of %s; using 'int8'", m, _MODES
            )
            m = "int8"
        _MODE = m
    return _MODE


def _ext_active():
    return _ext_f16() if w4a8_mode() == "f16" else _ext()


def w4a8_enabled() -> bool:
    """VLLM_GFX908_W4A8=1 on gfx908 with a working extension build for the active mode (cached)."""
    global _FLAG
    if _FLAG is None:
        _FLAG = False
        if os.environ.get("VLLM_GFX908_W4A8", "0") == "1":
            try:
                from vllm.platforms.rocm import on_gfx908

                _FLAG = bool(on_gfx908()) and _ext_active() is not None
            except Exception as exc:  # hipcc missing etc. -> stock paths
                # 2026-09-05: a silent fallback here hid a build break for a whole night of
                # measurements (every boot ran the split-K W4A16 path).  Under the explicit flag
                # a failed build is an error unless VLLM_GFX908_STRICT_EXT=0.
                if os.environ.get("VLLM_GFX908_STRICT_EXT", "1") == "1":
                    raise RuntimeError(
                        "gfx908: VLLM_GFX908_W4A8=1 but the W4A8 extension failed to build; "
                        "set VLLM_GFX908_STRICT_EXT=0 to fall back to W4A16"
                    ) from exc
                logger.warning_once("gfx908: W4A8 GEMV path unavailable (%s); using W4A16", exc)
                _FLAG = False
            if _FLAG:
                logger.info_once(
                    "gfx908: W4 GEMV path enabled (VLLM_GFX908_W4A8=1, mode=%s)", w4a8_mode()
                )
    return _FLAG


def shared_as_expert_enabled() -> bool:
    """VLLM_GFX908_SHARED_AS_EXPERT=1 (default off): fold the shared expert into the routed
    GEMV launches as "expert #E" (see gfx908_shared_expert.py).  Requires the W4 GEMV path."""
    global _SHARED_AS_EXPERT
    if _SHARED_AS_EXPERT is None:
        _SHARED_AS_EXPERT = (
            os.environ.get("VLLM_GFX908_SHARED_AS_EXPERT", "0") == "1" and w4a8_enabled()
        )
        if _SHARED_AS_EXPERT:
            logger.info_once(
                "gfx908: shared expert folded into the routed W4 GEMVs "
                "(VLLM_GFX908_SHARED_AS_EXPERT=1, mode=%s)",
                w4a8_mode(),
            )
    return _SHARED_AS_EXPERT


def prep_fold_enabled() -> bool:
    """VLLM_GFX908_MOE_PREP_FOLD (default on since round 9; 0 disables): fold the gate_up projection's activation
    preparation (Q8_1 quant in ``int8`` mode, fp16 cast + per-8 permutation in ``f16`` mode) into
    the slab GEMV's LDS staging, so the separate ``quant_q8`` / ``cast_f16`` launch disappears.

    The staged bytes and the order they are read in are the same as the two-launch flow, so the
    result is **bit-identical**; only the launch count changes (-1 per gate_up GEMV).  The down
    projection is untouched: its activation already comes out of the silu*mul kernel in the
    prepared format.

    Not taken when the shared expert rides along as expert #E: that flow needs the prep kernel's
    extra blocks to build ``row_exp`` / ``wcomb`` *before* the slab kernel reads them, so folding
    the activation half would save no launch.
    """
    global _PREP_FOLD
    if _PREP_FOLD is None:
        _PREP_FOLD = os.environ.get("VLLM_GFX908_MOE_PREP_FOLD", "1") == "1" and w4a8_enabled()
        if _PREP_FOLD:
            logger.info_once(
                "gfx908: MoE gate_up activation prep folded into the slab GEMV "
                "(VLLM_GFX908_MOE_PREP_FOLD=1, mode=%s)",
                w4a8_mode(),
            )
    return _PREP_FOLD


# The folded kernel always runs the 4-waves-per-block LDS geometry (one staged token per block).
# Below this many tiles that geometry leaves most of the 120 CUs idle and the one saved launch
# does not pay for it -- the same 480-tile rule ``_slab_cfg_f16`` uses to pick its LDS variant.
_FOLD_MIN_TILES = 480


def silu_fold_enabled() -> bool:
    """VLLM_GFX908_MOE_SILU_FOLD: fold silu*mul + Q8_1 into the down GEMV (one launch fewer).

    Bit-exact vs silu_mul_quant + gemv_rowlane (the Triton sigmoid/exp/divide/rounding sequence is
    reproduced exactly), but the win is modest -- cold, two runs: M=1 -0.25/-0.30 us,
    M=4 -0.86/-0.94, M=8 -1.28/-1.35 -- because every down workgroup redoes the 5 groups of
    silu*mul, which costs the down GEMV ~1.5 us of the ~1.8 us the launch was worth.  Default off.
    """
    global _SILU_FOLD
    if _SILU_FOLD is None:
        _SILU_FOLD = os.environ.get("VLLM_GFX908_MOE_SILU_FOLD", "0") == "1"
        if _SILU_FOLD:
            logger.info_once("gfx908: MoE silu*mul+Q8_1 folded into the down GEMV "
                             "(VLLM_GFX908_MOE_SILU_FOLD=1)")
    return _SILU_FOLD


def silu_fold_applies(N: int, K2: int, P: int) -> bool:
    """Instantiated for the intermediate width 160 only; a block must not straddle two pairs."""
    if not (silu_fold_enabled() and K2 == 160 and N % 64 == 0):
        return False
    return (N // 64) % _silu_fold_wpb(N, P) == 0


def _silu_fold_wpb(N: int, P: int) -> int:
    """4 waves per block while the grid is small, 8 above it (measured best at M=1 vs M>=4)."""
    return 4 if P * (N // 64) <= 1024 else 8


def prep_fold_applies(N: int, K: int, P: int) -> bool:
    """The folded slab kernel stages one token per 4-wave block, so a block must not straddle two
    pairs: (N / ROWS) % 4 == 0 with ROWS = 64 / lpr = 4.  K must be an instantiated slab case, and
    there must be enough tiles (P rows x N/ROWS) to fill the machine with 4-wave blocks."""
    return (
        prep_fold_enabled()
        and K in _SLAB_K
        and N % 4 == 0
        and (N // 4) % 4 == 0
        and P * (N // 4) >= _FOLD_MIN_TILES
    )


_BF16_OUT: bool | None = None


def bf16_out_enabled() -> bool:
    """VLLM_GFX908_W4A8_BF16_OUT (default off): the dense slab GEMVs (QSA qkv_proj /
    index_qk_proj / o_proj) store bf16 straight from the slab epilogue instead of fp32 +
    the Triton SPLIT_K=1 cast launch.  Bit-exact (RNE both ways, -0.0 canonicalised like the
    Triton reduce); -1 launch per dense GEMV (36 per decode step).  int8 mode only."""
    global _BF16_OUT
    if _BF16_OUT is None:
        _BF16_OUT = (
            os.environ.get("VLLM_GFX908_W4A8_BF16_OUT", "0") == "1"
            or os.environ.get("VLLM_GFX908_W4A8_BF16_EPILOGUE", "0") == "1"
        ) and w4a8_mode() == "int8"
        if _BF16_OUT:
            logger.info_once("gfx908: dense W4A8 GEMVs store bf16 directly (VLLM_GFX908_W4A8_BF16_OUT=1)")
    return _BF16_OUT


def _reset_env_cache():
    """Re-read VLLM_GFX908_W4A8 / _MODE / _SHARED_AS_EXPERT / _PREP_FOLD (tests only;
    extensions stay cached)."""
    global _FLAG, _MODE, _SHARED_AS_EXPERT, _PREP_FOLD, _SILU_FOLD, _BF16_OUT
    _SILU_FOLD = None
    _BF16_OUT = None
    _FLAG = None
    _MODE = None
    _SHARED_AS_EXPERT = None
    _PREP_FOLD = None


# --------------------------------------------------------------------------- #
# "no extra expert" sentinel (an empty tensor -> nullptr in the .hip wrappers)
# --------------------------------------------------------------------------- #
@functools.cache
def _no_extra(device: torch.device):
    e = torch.empty(0, dtype=torch.int32, device=device)
    return e, e


def _extra_pair(extra, device):
    return _no_extra(device) if extra is None else (extra[0], extra[1])


# --------------------------------------------------------------------------- #
# thin wrappers -- int8 mode
# --------------------------------------------------------------------------- #
def quant_q8(x: torch.Tensor):
    """bf16 [R, K] -> (x8 int8 [R, K] per-8 de-interleaved, xs fp32 [R, K/32], xsum fp32 [R, K/32])."""
    R, K = x.shape
    x8 = torch.empty((R, K), dtype=torch.int8, device=x.device)
    xs = torch.empty((R, K // GS), dtype=torch.float32, device=x.device)
    xsum = torch.empty_like(xs)
    _ext().quant_q8(x, x8, xs, xsum)
    return x8, xs, xsum


def quant_q8_gate(x, wg, topk_ids, topk_w, row_exp, wcomb, E: int):
    """quant_q8 plus the shared-expert row prep, in ONE launch (M extra blocks)."""
    R, K = x.shape
    x8 = torch.empty((R, K), dtype=torch.int8, device=x.device)
    xs = torch.empty((R, K // GS), dtype=torch.float32, device=x.device)
    xsum = torch.empty_like(xs)
    _ext().quant_q8_gate(x, x8, xs, xsum, wg, topk_ids, topk_w, row_exp, wcomb, E)
    return x8, xs, xsum


def gemv_slab(w, s, x8, xs, xsum, row_tok, row_exp, wpb: int = 4, extra=None) -> torch.Tensor:
    """D1: w [E, N, K/8] int32 (K-packed nibbles), s [E, N, K/32] bf16 -> fp32 [P, N].
    ``extra`` = (w, s) of one more expert selected by row_exp == E (the shared expert)."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x8.device)
    wx, sx = _extra_pair(extra, x8.device)
    _ext().gemv_slab(w, s, x8, xs, xsum, row_tok, row_exp, out, 16, wpb, wx, sx)
    return out


def gemv_slab_prep(w, s, x, row_tok, row_exp, extra=None) -> torch.Tensor:
    """D1 with the Q8_1 prep folded in: x is the raw bf16 [R, K] activation, no quant_q8 launch.
    Bit-identical to ``gemv_slab(*quant_q8(x), ...)``."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x.device)
    wx, sx = _extra_pair(extra, x.device)
    _ext().gemv_slab_prep(w, s, x, row_tok, row_exp, out, 16, wx, sx)
    return out


def gemv_slab_prep_gate(w, s, x, row_tok, row_exp, wcomb, wg, topk_ids, topk_w, E, extra):
    """D1 with BOTH the Q8_1 prep and the shared-as-expert row/gate prep folded in (no quant_q8_gate
    launch).  ``row_exp`` / ``wcomb`` are written by M extra blocks of this same launch; the slab
    blocks derive their expert from ``topk_ids`` instead of reading ``row_exp``."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x.device)
    _ext().gemv_slab_prep_gate(w, s, x, row_tok, out, 16, extra[0], extra[1],
                               wg, topk_ids, topk_w, row_exp, wcomb, E)
    return out


def gemv_rowlane_silu(w, s, part, row_tok, row_exp, extra=None) -> torch.Tensor:
    """D2 with silu*mul + Q8_1 folded in: ``part`` is the raw fp32 [P, 2*K2] gate|up partial, so
    the silu_mul_quant launch disappears.  Bit-identical to
    ``gemv_rowlane(*silu_mul_quant(part), ...)``."""
    N = w.shape[1]
    P = row_tok.numel()
    out = torch.empty((P, N), dtype=torch.float32, device=part.device)
    wx, sx = _extra_pair(extra, part.device)
    wpb = _silu_fold_wpb(N, P)
    _ext().gemv_rowlane_silu(w, s, part, row_tok, row_exp, out, wpb, 0 if wpb == 4 else 1, wx, sx)
    return out


def gemv_rowlane(w, s, x8, xs, xsum, row_tok, row_exp, wpb: int = 4, extra=None) -> torch.Tensor:
    """D2 (K == 160): w [E, N, 20] int32, s [E, N, 5] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x8.device)
    wx, sx = _extra_pair(extra, x8.device)
    _ext().gemv_rowlane(w, s, x8, xs, xsum, row_tok, row_exp, out, wpb, wx, sx)
    return out


# --------------------------------------------------------------------------- #
# thin wrappers -- f16 (W4A16-exact) mode
# --------------------------------------------------------------------------- #
def cast_f16(x: torch.Tensor) -> torch.Tensor:
    """bf16 [R, K] -> fp16 [R, K], per-8 permuted to [k0 k4 k1 k5 k2 k6 k3 k7] (lossless)."""
    xh = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    _ext_f16().cast_f16(x, xh)
    return xh


def cast_f16_audit(x: torch.Tensor):
    """cast_f16 + the fp16 range audit: (xh, ovf) with ovf = [#overflow, #flush-to-zero, max|x|]."""
    xh = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    ovf = torch.zeros(3, dtype=torch.int32, device=x.device)
    _ext_f16().cast_f16_audit(x, xh, ovf)
    o = ovf.cpu()
    import struct

    return xh, (int(o[0]), int(o[1]), struct.unpack("<f", struct.pack("<I", int(o[2]) & 0xFFFFFFFF))[0])


def _slab_cfg_f16(P: int, N: int) -> tuple[int, int, int]:
    """(lpr, mode, wpb) for the fp16 slab: the LDS variant (fastest at every microbenched shape)
    whenever it is legal ((N/ROWS) % 4 == 0) and there is enough work to fill the 120 CUs with
    4-wave blocks; otherwise the activation-in-VGPRs variant with one wave per block."""
    tiles = P * (N // 4)          # ROWS = 64 // lpr = 4
    if (N // 4) % 4 == 0 and tiles >= 480:
        return 16, 1, 4
    return 16, 0, (1 if tiles <= 480 else 4)


def gemv_slab_f16(w, s, xh, row_tok, row_exp, cfg=None, extra=None) -> torch.Tensor:
    """D5 slab: w [E, N, K/8] int32, s [E, N, K/32] bf16, xh fp16 [R, K] -> fp32 [P, N].
    ``extra`` = (w, s) of one more expert selected by row_exp == E (the shared expert)."""
    P, N = row_tok.numel(), w.shape[1]
    lpr, mode, wpb = cfg if cfg is not None else _slab_cfg_f16(P, N)
    out = torch.empty((P, N), dtype=torch.float32, device=xh.device)
    wx, sx = _extra_pair(extra, xh.device)
    _ext_f16().gemv_slab_f16(w, s, xh, row_tok, row_exp, out, lpr, mode, wpb, wx, sx)
    return out


def gemv_slab_f16_prep(w, s, x, row_tok, row_exp, extra=None) -> torch.Tensor:
    """D5 slab with the bf16 -> fp16 cast + per-8 permutation folded into the LDS staging: x is the
    raw bf16 [R, K] activation, no cast_f16 launch.  Bit-identical to
    ``gemv_slab_f16(w, s, cast_f16(x), ..., cfg=(16, 1, 4))``."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x.device)
    wx, sx = _extra_pair(extra, x.device)
    _ext_f16().gemv_slab_f16_prep(w, s, x, row_tok, row_exp, out, 16, wx, sx)
    return out


def gemv_slab_f16_prep_gate(w, s, x, row_tok, row_exp, wcomb, wg, topk_ids, topk_w, E, extra):
    """D5 slab with BOTH the fp16 cast and the shared-as-expert row/gate prep folded in."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x.device)
    _ext_f16().gemv_slab_f16_prep_gate(w, s, x, row_tok, out, 16, extra[0], extra[1],
                                       wg, topk_ids, topk_w, row_exp, wcomb, E)
    return out


def gemv_rowlane_f16(w, s, xh, row_tok, row_exp, wpb: int = 4, extra=None) -> torch.Tensor:
    """D5 rowlane (K == 160): w [E, N, 20] int32, s [E, N, 5] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=xh.device)
    wx, sx = _extra_pair(extra, xh.device)
    _ext_f16().gemv_rowlane_f16(w, s, xh, row_tok, row_exp, out, wpb, wx, sx)
    return out


def cast_f16_gate(x, wg, topk_ids, topk_w, row_exp, wcomb, E: int) -> torch.Tensor:
    """cast_f16 plus the shared-expert row prep, in ONE launch (M extra blocks)."""
    xh = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    _ext_f16().cast_f16_gate(x, xh, wg, topk_ids, topk_w, row_exp, wcomb, E)
    return xh


# --------------------------------------------------------------------------- #
# silu*mul intermediate: fp32 [P, 2*K2] gate|up partial -> the down projection's activation
# --------------------------------------------------------------------------- #
@triton.jit
def _silu_mul_quant_kernel(
    part_ptr, x8_ptr, xs_ptr, xsum_ptr, N,          # part fp32 [P, N] (gate | up), N = 2 * K2
    stride_pp, stride_x8, stride_xs,
    GS: tl.constexpr,
):
    """silu(g) * u over one block of 32 outputs, then Q8_1 quant into the de-interleaved
    int8 layout the HIP kernels read (per 8: [k0 k2 k4 k6 | k1 k3 k5 k7])."""
    pair = tl.program_id(0)
    g = tl.program_id(1)
    half = N // 2
    offs = g * GS + tl.arange(0, GS)
    base = part_ptr + pair * stride_pp
    gv = tl.load(base + offs)
    uv = tl.load(base + half + offs)
    y = gv * tl.sigmoid(gv) * uv
    amax = tl.max(tl.abs(y), axis=0)
    d = amax / 127.0
    inv = tl.where(amax > 0, 127.0 / amax, 0.0)
    v = y * inv
    q = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    qi = q.to(tl.int32)
    dst = (offs // 8) * 8 + (offs % 2) * 4 + (offs % 8) // 2
    tl.store(x8_ptr + pair * stride_x8 + dst, qi.to(tl.int8))
    tl.store(xs_ptr + pair * stride_xs + g, d)
    tl.store(xsum_ptr + pair * stride_xs + g, d * tl.sum(qi.to(tl.float32), axis=0))


@triton.jit
def _silu_mul_cast_kernel(
    part_ptr, xh_ptr, N,                            # part fp32 [P, N] (gate | up), N = 2 * K2
    stride_pp, stride_xh,
    GS: tl.constexpr,
):
    """silu(g) * u over one block of 32 outputs, emitted as fp16 in the per-8 permuted layout the
    fp16 HIP kernels read ([k0 k4 k1 k5 k2 k6 k3 k7]) -- no requantization, so the down projection
    sees the exact intermediate (rounded once to fp16)."""
    pair = tl.program_id(0)
    g = tl.program_id(1)
    half = N // 2
    offs = g * GS + tl.arange(0, GS)
    base = part_ptr + pair * stride_pp
    gv = tl.load(base + offs)
    uv = tl.load(base + half + offs)
    y = gv * tl.sigmoid(gv) * uv
    m = offs % 8
    dst = (offs // 8) * 8 + tl.where(m < 4, 2 * m, 2 * (m - 4) + 1)
    tl.store(xh_ptr + pair * stride_xh + dst, y.to(tl.float16))


def silu_mul_quant(part: torch.Tensor):
    """fp32 [P, 2*K2] gate|up partial -> int8 [P, K2] + xs/xsum [P, K2/32] (one launch)."""
    P, N = part.shape
    K2 = N // 2
    assert K2 % GS == 0, K2
    x8 = torch.empty((P, K2), dtype=torch.int8, device=part.device)
    xs = torch.empty((P, K2 // GS), dtype=torch.float32, device=part.device)
    xsum = torch.empty_like(xs)
    _silu_mul_quant_kernel[(P, K2 // GS)](
        part, x8, xs, xsum, N, part.stride(0), x8.stride(0), xs.stride(0), GS=GS,
    )
    return x8, xs, xsum


def silu_mul_cast(part: torch.Tensor) -> torch.Tensor:
    """fp32 [P, 2*K2] gate|up partial -> fp16 [P, K2] per-8 permuted (one launch)."""
    P, N = part.shape
    K2 = N // 2
    assert K2 % GS == 0, K2
    xh = torch.empty((P, K2), dtype=torch.float16, device=part.device)
    _silu_mul_cast_kernel[(P, K2 // GS)](
        part, xh, N, part.stride(0), xh.stride(0), GS=GS,
    )
    return xh


@functools.cache
def _rows(M: int, device: torch.device):
    """(arange(M), zeros(M)) int32: row -> token, row -> expert 0 for single-matrix GEMVs."""
    return (
        torch.arange(M, device=device, dtype=torch.int32),
        torch.zeros(M, device=device, dtype=torch.int32),
    )


# --------------------------------------------------------------------------- #
# shared expert as expert #E: the M * (topk + 1) row lists.
# Layout is interleaved -- row (m, t) = m * (topk + 1) + t, t == topk being the shared expert --
# which is exactly what _moe_reduce_weighted_sum_kernel indexes with TOPK = topk + 1, so the
# routed + shared sum comes out of the existing epilogue.
#   row_token / row_self are constant for a given (M, topk) and are built once;
#   row_exp / wcomb are persistent OUTPUT buffers rewritten each call by the prep kernel
#   (stable addresses -> cudagraph safe).
# --------------------------------------------------------------------------- #
_FUSED_ROWS: dict[tuple, tuple] = {}


def _fused_rows(M: int, topk: int, device: torch.device):
    """(row_token, row_self, row_exp, wcomb) for M*(topk+1) rows, or None while capturing a
    graph before the first eager call for this (M, topk) has allocated them."""
    key = (M, topk, str(device))
    t = _FUSED_ROWS.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        P = M * (topk + 1)
        row_self = torch.arange(P, device=device, dtype=torch.int32)
        row_token = torch.div(row_self, topk + 1, rounding_mode="floor").to(torch.int32)
        row_exp = torch.zeros(P, device=device, dtype=torch.int32)
        wcomb = torch.zeros((M, topk + 1), device=device, dtype=torch.float32)
        t = (row_token, row_self, row_exp, wcomb)
        _FUSED_ROWS[key] = t
    return t


# --------------------------------------------------------------------------- #
# dense weight repack: [K, N/8] int32 K-major (N packed, triton_w4a16 layout) + scales [K/G, N]
# -> [1, N, K/8] int32 K-packed (low nibble = lowest k) + scales [1, N, K/G] bf16.
# (identical for both modes -- the weight layout does not depend on the activation type)
# --------------------------------------------------------------------------- #
# value = (w, s, source): the source weight is kept alive so its data_ptr (the key) cannot be
# recycled by a later allocation of the same shape while the entry exists.
_REPACK_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def repack_dense_to_nk(b_q: torch.Tensor, scales: torch.Tensor):
    K, N8 = b_q.shape
    N = N8 * 8
    shifts = torch.arange(8, device=b_q.device, dtype=torch.int32) * 4
    w_kn = ((b_q.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N)            # [K, N] nibbles
    w_nk = w_kn.t().contiguous().view(N, K // 2, 2).to(torch.uint8)
    packed = (w_nk[:, :, 0] | (w_nk[:, :, 1] << 4)).contiguous()           # uint8 [N, K/2]
    w = packed.view(torch.int32).unsqueeze(0).contiguous()                  # [1, N, K/8]
    s = scales.t().contiguous().to(torch.bfloat16).unsqueeze(0).contiguous()  # [1, N, K/G]
    return w, s


def dense_weight_nk(b_q: torch.Tensor, scales: torch.Tensor):
    """Cached repack keyed on the weight's data_ptr; None while capturing a graph and not yet
    cached (the caller then uses the stock path; the eager warmup call fills the cache)."""
    key = (b_q.data_ptr(), tuple(b_q.shape), str(b_q.device))
    t = _REPACK_CACHE.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        w, s = repack_dense_to_nk(b_q, scales)
        t = (w, s, b_q)
        _REPACK_CACHE[key] = t
    return t[0], t[1]


# --------------------------------------------------------------------------- #
# the three flows (the mode switch lives here, so the hooks stay mode-agnostic)
# --------------------------------------------------------------------------- #
def moe_w4a8_applies(K: int, N1: int, group_size: int, dtype: torch.dtype) -> bool:
    return (
        K in _SLAB_K and K % 64 == 0 and N1 % 4 == 0 and N1 // 2 == _ROWLANE_K
        and group_size == GS and dtype == torch.bfloat16
    )


def shared_pack_applies(pack, E: int, N1: int, K: int, K2: int) -> bool:
    """The shared expert can ride along as expert #E only if its two projections have exactly the
    routed experts' per-rank shapes (Qwen3.8-Flash-Next: both intermediates are 160/rank)."""
    if pack is None:
        return False
    w1x, s1x, w2x, s2x, wg = pack
    return (
        w1x.shape == (1, N1, K // 8) and s1x.shape == (1, N1, K // GS)
        and w2x.shape == (1, K, K2 // 8) and s2x.shape == (1, K, K2 // GS)
        and wg.numel() == K and wg.dtype == torch.bfloat16 and wg.is_contiguous()
    )


def moe_w4a8(
    output, hidden_states, w1_i, w2_i, w1_scale, w2_scale, topk_weights,
    row_token, row_self, row_expert, mul_routed_weight: bool, shared=None,
):
    """Routed experts: activation prep -> slab gate_up [P, N1] -> silu*mul (+prep) -> rowlane
    down [P, K] -> existing routed-weight sum. Same arguments/layouts as gfx908_moe_hip
    (w viewed as int32).

    ``shared`` = (w1, s1, w2, s2, w_gate) of the shared expert, already repacked to the routed
    ``[1, N, K/8]`` layout.  When given, the shared expert becomes expert #E: one extra row per
    token is appended to the row lists with weight sigmoid(x . w_gate) (both computed by extra
    blocks of the activation-prep kernel, so no extra launch), the GEMVs select the extra slab
    for those rows, and the existing weighted-sum reduce emits routed + shared in one pass.
    Returns ``output``, or None when the fused path could not be taken (caller falls back)."""
    from vllm.model_executor.layers.fused_moe.gfx908_moe_hip import _moe_reduce_weighted_sum_kernel

    M, K = hidden_states.shape
    P = row_expert.numel()
    topk = P // M
    E = w1_i.shape[0]
    N1 = w1_i.shape[1]
    K2 = N1 // 2
    extra1 = extra2 = None
    wsum = topk_weights.reshape(-1).to(torch.float32)
    rows = topk
    # the gate_up activation prep rides along with the slab kernel's LDS staging (-1 launch).
    # With the shared expert folded in as expert #E the row-list / gate prep rides along too
    # (fold_gate), so that flow also drops from five launches to four.
    fold = shared is None and prep_fold_applies(N1, K, P)
    fold_gate = False

    if shared is not None:
        if not (mul_routed_weight and shared_pack_applies(shared, E, N1, K, K2)):
            return None
        fr = _fused_rows(M, topk, hidden_states.device)
        if fr is None:
            return None
        row_token, row_self, row_expert_f, wcomb = fr
        ids = row_expert.view(M, topk)
        tkw = topk_weights.reshape(M, topk).to(torch.float32).contiguous()
        w1x, s1x, w2x, s2x, wg = shared
        extra1, extra2 = (w1x, s1x), (w2x, s2x)
        fold_gate = prep_fold_applies(N1, K, row_token.numel())
        if fold_gate:
            pass                              # folded into the gate_up slab below
        elif w4a8_mode() == "f16":
            xh = cast_f16_gate(hidden_states, wg, ids, tkw, row_expert_f, wcomb, E)
        else:
            x8, xs, xsum = quant_q8_gate(hidden_states, wg, ids, tkw, row_expert_f, wcomb, E)
        row_expert = row_expert_f
        wsum = wcomb.view(-1)
        rows = topk + 1
    elif fold:
        pass                                  # no separate prep launch
    elif w4a8_mode() == "f16":
        xh = cast_f16(hidden_states)
    else:
        x8, xs, xsum = quant_q8(hidden_states)

    if w4a8_mode() == "f16":
        if fold_gate:
            part1 = gemv_slab_f16_prep_gate(w1_i, w1_scale, hidden_states, row_token, row_expert,
                                            wcomb, wg, ids, tkw, E, extra1)
        elif fold:
            part1 = gemv_slab_f16_prep(w1_i, w1_scale, hidden_states, row_token, row_expert)
        else:
            part1 = gemv_slab_f16(w1_i, w1_scale, xh, row_token, row_expert, extra=extra1)
        ih = silu_mul_cast(part1)
        part2 = gemv_rowlane_f16(w2_i, w2_scale, ih, row_self, row_expert, wpb=4, extra=extra2)
    else:
        if fold_gate:
            part1 = gemv_slab_prep_gate(w1_i, w1_scale, hidden_states, row_token, row_expert,
                                        wcomb, wg, ids, tkw, E, extra1)
        elif fold:
            part1 = gemv_slab_prep(w1_i, w1_scale, hidden_states, row_token, row_expert)
        else:
            # gemv_flight (2026-09-04), cold M=1/4/8, two runs each: one wave per block wins
            # while the tile count is small (M=1 -0.23/-0.25 us, M=4 -0.61/-0.67) but REGRESSES
            # once it passes ~4k, because 7040 one-wave dispatches cost more than they save
            # (M=8 +1.24/+0.42 us).  Gate on the tile count, not on M, so it also holds for other
            # N1 / topk.  Empty-kernel dispatch is 1.89 us at 220 workgroups and 7.42 us at 3520.
            sw = 1 if P * (N1 // 4) <= 4096 else 4
            part1 = gemv_slab(w1_i, w1_scale, x8, xs, xsum, row_token, row_expert, wpb=sw, extra=extra1)
        if silu_fold_applies(K, K2, P):
            part2 = gemv_rowlane_silu(w2_i, w2_scale, part1, row_self, row_expert, extra=extra2)
        else:
            i8, isc, isum = silu_mul_quant(part1)
            # rowlane wpb: 1 vs 4 measured -0.02/-0.14 us over M=1/4/8, i.e. at the noise floor.
            # Kept at 1 (never a regression), but it is not a win worth quoting.
            part2 = gemv_rowlane(w2_i, w2_scale, i8, isc, isum, row_self, row_expert, wpb=1,
                                 extra=extra2)
    rb2 = 256  # 3 -> 10 workgroups; 3.13 -> 1.98 us at M=1 (agents/gemv_flight), same per-element order
    _moe_reduce_weighted_sum_kernel[(triton.cdiv(K, rb2), M)](
        part2, wsum, output, K,
        0, part2.stride(0), output.stride(0),
        TOPK=rows, SPLIT_K=1, BLOCK=rb2, MUL_W=mul_routed_weight,
    )
    return output


# --------------------------------------------------------------------------- #
# expert-deduplicated ("multi-row") flow for batched decode, agents/batch_research (2026-09-05).
# Same arithmetic as moe_w4a8 (bit-identical output), but the pairs are sorted by expert and one
# block streams an expert's weight tile once for all of its rows (csrc/gfx908_moe_mr.hip):
#   expert_sort -> slab_mr_lds (gate_up) -> silu_mul_quant -> rowlane_mr (down) -> weighted-sum reduce.
# The work lists are data dependent but the grids are persistent (fixed block count), so the
# launch sequence is cudagraph-stable.
# --------------------------------------------------------------------------- #
_MR_BUFS: dict[tuple, tuple] = {}


def _mr_bufs(P: int, E: int, rc: int, device):
    key = (P, E, rc, str(device))
    t = _MR_BUFS.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        t = tuple(torch.empty(P, dtype=torch.int32, device=device) for _ in range(5)) + (
            torch.empty(P + (E + 2) * rc, dtype=torch.int32, device=device),   # wl_tok
            torch.empty(E + 1, dtype=torch.int32, device=device),              # uniq_e
            torch.zeros(4, dtype=torch.int32, device=device),                  # counters
        )
        _MR_BUFS[key] = t
    return t


def _mr_gemvs(ext, w1_i, w1_scale, w2_i, w2_scale, x8, xs, xsum, row_tok, row_exp, E, has_extra,
              extra1, extra2, dev, cfg=None):
    """sort + gate_up + silu*mul + down for P rows -> fp32 [P, K] partials, or None during capture
    before the work-list buffers exist."""
    cfg = cfg or MR_CFG
    P = row_exp.numel()
    N1 = w1_i.shape[1]
    K = w2_i.shape[1]
    bufs = _mr_bufs(P, E, cfg["rc"], dev)
    if bufs is None:
        return None
    order, otok, wl_e, wl_r0, wl_cnt, wl_tok, uniq, counters = bufs
    ext.expert_sort(row_exp, row_tok, E, has_extra, cfg["rc"], order, otok, wl_e, wl_r0, wl_cnt,
                    wl_tok, uniq, counters)
    wx1, sx1 = _extra_pair(extra1, dev)
    wx2, sx2 = _extra_pair(extra2, dev)
    part1 = torch.empty((P, N1), dtype=torch.float32, device=dev)
    ext.gemv_slab_mr_lds(w1_i, w1_scale, x8, xs, xsum, order, otok, wl_e, wl_r0, wl_cnt, counters,
                         part1, cfg["wpb"], cfg["rcs"], cfg["blocks"], wx1, sx1, cfg["pf1"], cfg["rc"])
    i8, isc, isum = silu_mul_quant(part1)
    part2 = torch.empty((P, K), dtype=torch.float32, device=dev)
    # the down projection's activation is per pair (row_self == arange): its token table is `order`
    ext.gemv_rowlane_mr(w2_i, w2_scale, i8, isc, isum, order, order, wl_e, wl_r0, wl_cnt, counters,
                        part2, cfg["wpb2"], cfg["blocks2"], wx2, sx2, cfg["pf2"])
    return part2


def moe_w4a8_mr(
    output, hidden_states, w1_i, w2_i, w1_scale, w2_scale, topk_weights,
    row_token, row_self, row_expert, mul_routed_weight: bool, shared=None,
):
    """moe_w4a8 with expert deduplication (int8 mode). Returns output or None (caller falls back)."""
    from vllm.model_executor.layers.fused_moe.gfx908_moe_hip import _moe_reduce_weighted_sum_kernel

    if not moe_mr_enabled():
        return None
    M, K = hidden_states.shape
    P = row_expert.numel()
    topk = P // M
    E = w1_i.shape[0]
    N1 = w1_i.shape[1]
    K2 = N1 // 2
    if K not in _SLAB_K or K2 != _ROWLANE_K or N1 % 32 != 0:   # slab: (N1/4) % 8 == 0 (8 waves/block)
        return None
    ext = _ext_mr()
    dev = hidden_states.device
    extra1 = extra2 = None
    wsum = topk_weights.reshape(-1).to(torch.float32)
    rows = topk
    if shared is not None:
        if not (mul_routed_weight and shared_pack_applies(shared, E, N1, K, K2)):
            return None
        fr = _fused_rows(M, topk, dev)
        if fr is None:
            return None
        row_token, row_self, row_expert_f, wcomb = fr
        ids = row_expert.view(M, topk)
        tkw = topk_weights.reshape(M, topk).to(torch.float32).contiguous()
        w1x, s1x, w2x, s2x, wg = shared
        extra1, extra2 = (w1x, s1x), (w2x, s2x)
        x8, xs, xsum = quant_q8_gate(hidden_states, wg, ids, tkw, row_expert_f, wcomb, E)
        row_expert = row_expert_f
        wsum = wcomb.view(-1)
        rows = topk + 1
    else:
        x8, xs, xsum = quant_q8(hidden_states)
    part2 = _mr_gemvs(ext, w1_i, w1_scale, w2_i, w2_scale, x8, xs, xsum, row_token, row_expert, E,
                      shared is not None, extra1, extra2, dev)
    if part2 is None:
        return None
    rb2 = 256
    _moe_reduce_weighted_sum_kernel[(triton.cdiv(K, rb2), M)](
        part2, wsum, output, K,
        0, part2.stride(0), output.stride(0),
        TOPK=rows, SPLIT_K=1, BLOCK=rb2, MUL_W=mul_routed_weight,
    )
    return output


_DENSE_ROWS: dict[tuple, tuple] = {}


def shared_expert_mr(x, pack):
    """The un-fused shared expert for M > DENSE_W4A8_MAX_TOKENS through the multi-row kernels
    (the expert is one item of rc rows at a time instead of being streamed once per token)."""
    from vllm.model_executor.layers.gfx908_shared_expert import _reduce_gate_kernel

    w1x, s1x, w2x, s2x, wg = pack
    M, K = x.shape
    H = w2x.shape[1]
    N1 = w1x.shape[1]
    if K not in _SLAB_K or N1 // 2 != _ROWLANE_K or N1 % 32 != 0 or H % 64 != 0:
        return None
    key = (M, str(x.device))
    t = _DENSE_ROWS.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        # E = 0 routed experts: every row is the "extra" expert #0
        t = (torch.arange(M, device=x.device, dtype=torch.int32), torch.zeros(M, device=x.device, dtype=torch.int32))
        _DENSE_ROWS[key] = t
    rt, re = t
    x8, xs, xsum = quant_q8(x)
    part2 = _mr_gemvs(_ext_mr(), w1x, s1x, w2x, s2x, x8, xs, xsum, rt, re, 0, True, (w1x, s1x), (w2x, s2x),
                      x.device)
    if part2 is None:
        return None
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    rb2 = 1024
    _reduce_gate_kernel[(triton.cdiv(H, rb2), M)](
        part2, x, wg, out, H, K, 0, part2.stride(0), x.stride(0), out.stride(0),
        SPLIT_K=1, BLOCK=rb2, BLOCK_K=1024,
    )
    return out


def shared_pack(wq1, ws1, wq2, ws2, wg):
    """Repack the shared expert's dense [K, N/8] weights into the routed [1, N, K/8] layout (cached
    by data_ptr) and return (w1, s1, w2, s2, w_gate), or None (unsupported shape / graph capture
    before the first eager call)."""
    N1 = wq1.shape[1] * 8      # gate | up width
    H = wq2.shape[1] * 8       # hidden size
    K = wq1.shape[0]
    if (
        K not in _SLAB_K or N1 % 4 != 0 or N1 // 2 != _ROWLANE_K or H % 64 != 0
        or wq2.shape[0] != _ROWLANE_K or wg.numel() != K or wg.dtype != torch.bfloat16
    ):
        return None
    p1 = dense_weight_nk(wq1, ws1)
    p2 = dense_weight_nk(wq2, ws2)
    if p1 is None or p2 is None:
        return None
    return p1[0], p1[1], p2[0], p2[1], wg.contiguous()


def shared_expert_from_pack(x, pack):
    """The separate (unfused) shared-expert flow from an already repacked pack -> bf16 [M, H]."""
    from vllm.model_executor.layers.gfx908_shared_expert import _reduce_gate_kernel

    w1x, s1x, w2x, s2x, wg = pack
    M, K = x.shape
    H = w2x.shape[1]
    rt, re = _rows(M, x.device)
    fold = prep_fold_applies(w1x.shape[1], K, M)
    if w4a8_mode() == "f16":
        part1 = (
            gemv_slab_f16_prep(w1x, s1x, x, rt, re) if fold
            else gemv_slab_f16(w1x, s1x, cast_f16(x), rt, re)
        )
        ih = silu_mul_cast(part1)
        part2 = gemv_rowlane_f16(w2x, s2x, ih, rt, re, wpb=1)
    else:
        if fold:
            part1 = gemv_slab_prep(w1x, s1x, x, rt, re)
        else:
            x8, xs, xsum = quant_q8(x)
            part1 = gemv_slab(w1x, s1x, x8, xs, xsum, rt, re, wpb=1)
        i8, isc, isum = silu_mul_quant(part1)
        part2 = gemv_rowlane(w2x, s2x, i8, isc, isum, rt, re, wpb=1)
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    rb2 = 1024
    _reduce_gate_kernel[(triton.cdiv(H, rb2), M)](
        part2, x, wg, out, H, K,
        0, part2.stride(0), x.stride(0), out.stride(0),
        SPLIT_K=1, BLOCK=rb2, BLOCK_K=1024,
    )
    return out


def shared_expert_w4a8(x, wq1, ws1, wq2, ws2, wg):
    """Shared expert (dense [K, N/8] weights): returns bf16 [M, H] or None if not applicable."""
    M, K = x.shape
    if M > W4A8_MAX_TOKENS or x.dtype != torch.bfloat16 or wq1.shape[1] * 8 % 4 != 0:
        return None
    pack = shared_pack(wq1, ws1, wq2, ws2, wg)
    if pack is None:
        return None
    if M > DENSE_W4A8_MAX_TOKENS:
        # per-row GEMVs would stream the expert once per token: use the multi-row kernels
        out = shared_expert_mr(x, pack) if moe_mr_enabled() else None
        return out
    return shared_expert_from_pack(x, pack)


# --------------------------------------------------------------------------- #
# shared expert as expert #E -- the deferral handshake between the two hooks.
#
# The shared expert runs FIRST (moe_runner._apply_quant_method calls it before the routed
# experts, SharedExpertsOrder.NO_OVERLAP), so it cannot know whether the routed MoE will take the
# fused W4 path.  Handshake:
#   1. every shared-expert call registers its repacked weights (`shared_register`);
#   2. after a routed MoE has run the W4 path with matching shapes, `shared_arm` arms the fusion;
#   3. from then on the shared-expert hook DEFERS: it stashes (x, pack) and returns an all-zeros
#      tensor, so the runner's `shared_output + fused_output` add is a no-op add of 0;
#   4. the routed MoE pops the stash (`shared_take`) and folds it in; if for any reason it cannot,
#      it computes the shared expert separately and adds it to its own output -- so a stash is
#      never dropped and the result is always correct.
# The stand-in is a FRESH torch.zeros (one fill launch, ~5 KB) rather than a cached buffer: the
# runner's `shared_output + fused_output` is an Inductor pointwise node that may legally reuse its
# input buffer in place, which would clobber a cached one.  Net launch delta per MoE layer is
# therefore -4 (the shared expert's cast + slab + silu*mul + rowlane + reduce_gate = 5 kernels
# disappear; the zero fill is added; the routed GEMVs and their epilogue absorb the extra rows).
# --------------------------------------------------------------------------- #
_SHARED_LAST = None       # last registered pack, used to decide arming
_SHARED_PENDING = None    # (x, pack) deferred by the shared hook
_SHARED_ARMED = False


def shared_register(pack) -> None:
    global _SHARED_LAST
    _SHARED_LAST = pack


def shared_arm(E: int, N1: int, K: int, K2: int, mul_routed_weight: bool) -> None:
    """Called by the routed MoE after a successful W4 run: arm the fusion if the shared expert
    registered by this layer has the routed experts' shapes."""
    global _SHARED_ARMED
    if _SHARED_ARMED or not shared_as_expert_enabled() or not mul_routed_weight:
        return
    if not shared_pack_applies(_SHARED_LAST, E, N1, K, K2):
        return
    _SHARED_ARMED = True
    logger.info_once(
        "gfx908: shared expert armed as routed expert #%d (K=%d, N1=%d, K2=%d)", E, K, N1, K2
    )


def shared_disarm(reason: str) -> None:
    global _SHARED_ARMED, _SHARED_PENDING
    if _SHARED_ARMED:
        logger.warning("gfx908: shared-as-expert fusion disabled (%s)", reason)
    _SHARED_ARMED = False
    _SHARED_PENDING = None


def shared_defer(x, pack):
    """Stash (x, pack) for the routed MoE and return the zero stand-in, or None to run stock."""
    global _SHARED_PENDING
    if not _SHARED_ARMED:
        return None
    if _SHARED_PENDING is not None:
        shared_disarm("a previous deferral was never consumed by the routed MoE")
        return None
    _SHARED_PENDING = (x, pack)
    return torch.zeros((x.shape[0], pack[2].shape[1]), dtype=x.dtype, device=x.device)


def shared_take():
    """Pop the deferred shared expert (called unconditionally by the routed MoE hook)."""
    global _SHARED_PENDING
    p = _SHARED_PENDING
    _SHARED_PENDING = None
    return p


def _reset_shared_state():
    """Tests only."""
    global _SHARED_LAST, _SHARED_PENDING, _SHARED_ARMED
    _SHARED_LAST = None
    _SHARED_PENDING = None
    _SHARED_ARMED = False


def dense_w4a8_gemv(a, b_q, scales):
    """Dense symmetric W4 GEMV, a bf16 [M, K], b_q [K, N/8], scales [K/32, N] -> bf16 [M, N] or None."""
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        triton_w4a16_splitk_reduce_kernel,
    )

    M, K = a.shape
    N = b_q.shape[1] * 8
    if M > DENSE_W4A8_MAX_TOKENS or a.dtype != torch.bfloat16 or K not in _SLAB_K or N % 4 != 0:
        return None
    p = dense_weight_nk(b_q, scales)
    if p is None:
        return None
    rt, re = _rows(M, a.device)
    fold = prep_fold_applies(N, K, M)
    if bf16_out_enabled():
        # bf16 epilogue: the slab kernel stores RNE(bf16) itself, no cast launch.
        c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
        wx, sx = _no_extra(a.device)
        if fold:
            _ext().gemv_slab_prep(p[0], p[1], a, rt, re, c, 16, wx, sx)
        else:
            x8, xs, xsum = quant_q8(a)
            _ext().gemv_slab(p[0], p[1], x8, xs, xsum, rt, re, c, 16, 1 if N <= 512 else 4, wx, sx)
        return c
    if w4a8_mode() == "f16":
        part = (
            gemv_slab_f16_prep(p[0], p[1], a, rt, re) if fold
            else gemv_slab_f16(p[0], p[1], cast_f16(a), rt, re)
        )
    elif fold:
        part = gemv_slab_prep(p[0], p[1], a, rt, re)
    else:
        x8, xs, xsum = quant_q8(a)
        part = gemv_slab(p[0], p[1], x8, xs, xsum, rt, re, wpb=1 if N <= 512 else 4)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    rb = 1024
    triton_w4a16_splitk_reduce_kernel[(triton.cdiv(N, rb), M)](
        part, c, M, N, 0, part.stride(0), c.stride(0), SPLIT_K=1, BLOCK=rb,
    )
    return c
