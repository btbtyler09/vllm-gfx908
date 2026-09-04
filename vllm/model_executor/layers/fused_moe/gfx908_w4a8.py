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
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

W4A8_MAX_TOKENS = 8
GS = 32
_SLAB_K = (2560, 1536)          # K values instantiated for the slab kernels (SLAB_CASE in the .hip)
_ROWLANE_K = 160
_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_CSRC = os.path.join(_CSRC_DIR, "gfx908_w4a8.hip")
_CSRC_F16 = os.path.join(_CSRC_DIR, "gfx908_w4f16.hip")
_MODES = ("int8", "f16")
_FLAG: bool | None = None
_MODE: str | None = None


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
                logger.warning_once("gfx908: W4A8 GEMV path unavailable (%s); using W4A16", exc)
                _FLAG = False
            if _FLAG:
                logger.info_once(
                    "gfx908: W4 GEMV path enabled (VLLM_GFX908_W4A8=1, mode=%s)", w4a8_mode()
                )
    return _FLAG


def _reset_env_cache():
    """Re-read VLLM_GFX908_W4A8 / _MODE (tests only; the extensions stay cached)."""
    global _FLAG, _MODE
    _FLAG = None
    _MODE = None


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


def gemv_slab(w, s, x8, xs, xsum, row_tok, row_exp, wpb: int = 4) -> torch.Tensor:
    """D1: w [E, N, K/8] int32 (K-packed nibbles), s [E, N, K/32] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x8.device)
    _ext().gemv_slab(w, s, x8, xs, xsum, row_tok, row_exp, out, 16, wpb)
    return out


def gemv_rowlane(w, s, x8, xs, xsum, row_tok, row_exp, wpb: int = 4) -> torch.Tensor:
    """D2 (K == 160): w [E, N, 20] int32, s [E, N, 5] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x8.device)
    _ext().gemv_rowlane(w, s, x8, xs, xsum, row_tok, row_exp, out, wpb)
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


def gemv_slab_f16(w, s, xh, row_tok, row_exp, cfg=None) -> torch.Tensor:
    """D5 slab: w [E, N, K/8] int32, s [E, N, K/32] bf16, xh fp16 [R, K] -> fp32 [P, N]."""
    P, N = row_tok.numel(), w.shape[1]
    lpr, mode, wpb = cfg if cfg is not None else _slab_cfg_f16(P, N)
    out = torch.empty((P, N), dtype=torch.float32, device=xh.device)
    _ext_f16().gemv_slab_f16(w, s, xh, row_tok, row_exp, out, lpr, mode, wpb)
    return out


def gemv_rowlane_f16(w, s, xh, row_tok, row_exp, wpb: int = 4) -> torch.Tensor:
    """D5 rowlane (K == 160): w [E, N, 20] int32, s [E, N, 5] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=xh.device)
    _ext_f16().gemv_rowlane_f16(w, s, xh, row_tok, row_exp, out, wpb)
    return out


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


def moe_w4a8(
    output, hidden_states, w1_i, w2_i, w1_scale, w2_scale, topk_weights,
    row_token, row_self, row_expert, mul_routed_weight: bool,
):
    """Routed experts: activation prep -> slab gate_up [P, N1] -> silu*mul (+prep) -> rowlane
    down [P, K] -> existing routed-weight sum. Same arguments/layouts as gfx908_moe_hip
    (w viewed as int32)."""
    from vllm.model_executor.layers.fused_moe.gfx908_moe_hip import _moe_reduce_weighted_sum_kernel

    M, K = hidden_states.shape
    P = row_expert.numel()
    topk = P // M
    if w4a8_mode() == "f16":
        xh = cast_f16(hidden_states)
        part1 = gemv_slab_f16(w1_i, w1_scale, xh, row_token, row_expert)
        ih = silu_mul_cast(part1)
        part2 = gemv_rowlane_f16(w2_i, w2_scale, ih, row_self, row_expert, wpb=4)
    else:
        x8, xs, xsum = quant_q8(hidden_states)
        part1 = gemv_slab(w1_i, w1_scale, x8, xs, xsum, row_token, row_expert, wpb=4)
        i8, isc, isum = silu_mul_quant(part1)
        part2 = gemv_rowlane(w2_i, w2_scale, i8, isc, isum, row_self, row_expert, wpb=4)
    rb2 = 1024
    _moe_reduce_weighted_sum_kernel[(triton.cdiv(K, rb2), M)](
        part2, topk_weights.reshape(-1).to(torch.float32), output, K,
        0, part2.stride(0), output.stride(0),
        TOPK=topk, SPLIT_K=1, BLOCK=rb2, MUL_W=mul_routed_weight,
    )
    return output


def shared_expert_w4a8(x, wq1, ws1, wq2, ws2, wg):
    """Shared expert (dense [K, N/8] weights): returns bf16 [M, H] or None if not applicable."""
    from vllm.model_executor.layers.gfx908_shared_expert import _reduce_gate_kernel

    M, K = x.shape
    N1 = wq1.shape[1] * 8
    H = wq2.shape[1] * 8
    if (
        M > W4A8_MAX_TOKENS or x.dtype != torch.bfloat16 or K not in _SLAB_K
        or N1 % 4 != 0 or N1 // 2 != _ROWLANE_K or H % 64 != 0 or wq2.shape[0] != _ROWLANE_K
    ):
        return None
    p1 = dense_weight_nk(wq1, ws1)
    p2 = dense_weight_nk(wq2, ws2)
    if p1 is None or p2 is None:
        return None
    rt, re = _rows(M, x.device)
    if w4a8_mode() == "f16":
        xh = cast_f16(x)
        part1 = gemv_slab_f16(p1[0], p1[1], xh, rt, re)
        ih = silu_mul_cast(part1)
        part2 = gemv_rowlane_f16(p2[0], p2[1], ih, rt, re, wpb=1)
    else:
        x8, xs, xsum = quant_q8(x)
        part1 = gemv_slab(p1[0], p1[1], x8, xs, xsum, rt, re, wpb=1)
        i8, isc, isum = silu_mul_quant(part1)
        part2 = gemv_rowlane(p2[0], p2[1], i8, isc, isum, rt, re, wpb=1)
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    rb2 = 1024
    _reduce_gate_kernel[(triton.cdiv(H, rb2), M)](
        part2, x, wg, out, H, K,
        0, part2.stride(0), x.stride(0), out.stride(0),
        SPLIT_K=1, BLOCK=rb2, BLOCK_K=1024,
    )
    return out


def dense_w4a8_gemv(a, b_q, scales):
    """Dense symmetric W4 GEMV, a bf16 [M, K], b_q [K, N/8], scales [K/32, N] -> bf16 [M, N] or None."""
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        triton_w4a16_splitk_reduce_kernel,
    )

    M, K = a.shape
    N = b_q.shape[1] * 8
    if M > W4A8_MAX_TOKENS or a.dtype != torch.bfloat16 or K not in _SLAB_K or N % 4 != 0:
        return None
    p = dense_weight_nk(b_q, scales)
    if p is None:
        return None
    rt, re = _rows(M, a.device)
    if w4a8_mode() == "f16":
        part = gemv_slab_f16(p[0], p[1], cast_f16(a), rt, re)
    else:
        x8, xs, xsum = quant_q8(a)
        part = gemv_slab(p[0], p[1], x8, xs, xsum, rt, re, wpb=1 if N <= 512 else 4)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    rb = 1024
    triton_w4a16_splitk_reduce_kernel[(triton.cdiv(N, rb), M)](
        part, c, M, N, 0, part.stride(0), c.stride(0), SPLIT_K=1, BLOCK=rb,
    )
    return c
