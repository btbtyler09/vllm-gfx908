# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-top-k fast path for `apply_top_k_top_p` (gfx908 / small batches).

The PyTorch reference (`apply_top_k_top_p_pytorch`) sorts the full vocab,
runs softmax + cumsum over it and scatters back: ~0.9 ms (1 row) to ~2.8 ms
(3-8 rows) per call at V=248320 on MI100. This path reproduces its result
for rows whose top_k is at most `FASTK_CAP` (64) without ever touching a
full-vocab sort:

  1. four byte-radix histogram passes over the row (256-bin `tl.histogram`,
     multi-block, memory bound) -> the exact k-th largest logit `thr` of every
     row, the exact number of tokens equal to it (`n_t`) and strictly above it
     (`g`, always <= k-1 <= 63);
  2. one pass that masks `x < thr` to -inf (top-k, keeps every tie like the
     reference), counts ties per block, and compacts the <= 63 strictly-above
     tokens into a per-row candidate buffer;
  3. one pass whose programs each replay the (tiny, <= 64 candidate) top-p
     decision of their row -- softmax over survivors, ascending cumsum, mask
     where cumsum <= 1 - p, never mask the last element -- scatter -inf into
     the masked strictly-above tokens, and mask the `c` lowest-index tied
     tokens (the reference's ascending sort is stable, so tied survivors are
     consumed by top-p in index order).

Launches per call: 1 memset + 4 radix passes + 1 top-k apply (+ 1 top-p
apply) = 6 or 7, all multi-block except none; no per-row serial kernel.

Semantics vs the reference: the surviving *set* is identical (including all
tie rules) except where the fp32 cumulative probability of a survivor lands
within rounding of `1 - p`: the reference sums exp() over the full row and
scans in torch's tree order, this kernel sums over the <= 64 survivors, so the
two can round differently in the last ulp and flip that single boundary
token. Measured mismatch rate is reported in REPORT.md.

No host synchronisation, no allocation that depends on data: safe under
CUDA-graph capture and async scheduling. Callers must guarantee (from CPU-side
metadata) that every row has top_k enabled and top_k <= FASTK_CAP.
"""

import torch
import triton
import triton.language as tl

try:  # precise exp (ocml on AMD); vLLM exposes the same module as `tldevice`.
    from triton.language.extra import libdevice as tldevice
except ImportError:  # pragma: no cover
    tldevice = None

FASTK_CAP = 64  # max supported per-row top_k (candidate buffer is 64 wide)
_BLOCK = 4096
_NBINS = 256
_NUM_DIGITS = 4
_KEY_MIN = -(2**63)
# int32 workspace layout per row
_WS_HIST = 0  # [4][256]
_WS_GCNT = _NUM_DIGITS * _NBINS  # 1024: compaction counter
_WS_NTB = _WS_GCNT + 4  # per-block tie counts [NB_PAD] (slots 1-3 unused)


@triton.jit
def _ordered(x):
    """fp32 -> int32 with the same ordering as the floats (incl. +-inf)."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _unordered(o):
    bits = tl.where(o >= 0, o, o ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _digit(o, D: tl.constexpr):
    if D == 0:
        return (o >> 24) + 128
    else:
        return (o >> (24 - 8 * D)) & 0xFF


@triton.jit
def _select(hist_ptr, kk, D: tl.constexpr):
    """Replay the radix selection over the first D digit histograms of a row.

    Returns (prefix, kk_rem, above, hsel): `prefix` is the selected high bits
    right-aligned as a signed int32 (== thr_ord >> (32 - 8*D)), `kk_rem` the
    rank still to be resolved inside that bin, `above` the number of elements
    strictly above the bin, `hsel` the bin's population (after 4 digits: the
    number of elements exactly equal to the k-th largest).
    """
    bb = tl.arange(0, 256)
    prefix = kk * 0
    above = kk * 0
    hsel = kk * 0
    for d in tl.static_range(D):
        h = tl.load(hist_ptr + d * 256 + bb)
        cum_top = tl.flip(tl.cumsum(tl.flip(h, 0), 0), 0)  # sum_{b' >= b} h[b']
        sel = tl.max(tl.where(cum_top >= kk, bb, -1), 0)
        hsel = tl.sum(tl.where(bb == sel, h, 0), 0)
        csel = tl.sum(tl.where(bb == sel, cum_top, 0), 0)
        above_d = csel - hsel
        kk = kk - above_d
        above = above + above_d
        if d == 0:
            prefix = sel - 128
        else:
            prefix = (prefix << 8) | sel
    return prefix, kk, above, hsel


@triton.jit
def _radix_pass_kernel(
    x_ptr,
    x_stride,
    k_ptr,
    ws_ptr,
    ws_stride,
    V,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    x = tl.load(x_ptr + row * x_stride + offs, mask=m, other=float("-inf"))
    o = _ordered(x)
    kk = tl.load(k_ptr + row)
    hist_row = ws_ptr + row * ws_stride
    # Rows with k outside [1, FASTK_CAP] are left untouched by every kernel
    # (insurance only: the caller guarantees the range from CPU metadata).
    k_ok = (kk >= 1) & (kk <= 64)
    if D > 0:
        prefix, kk, above, hsel = _select(hist_row, kk, D)
        matched = m & k_ok & ((o >> (32 - 8 * D)) == prefix)
    else:
        matched = m & k_ok
    bins = tl.where(matched, _digit(o, D), 0)
    h = tl.histogram(bins, 256, mask=matched)
    bb = tl.arange(0, 256)
    tl.atomic_add(hist_row + D * 256 + bb, h, mask=h > 0)


@triton.jit
def _apply_topk_kernel(
    x_ptr,
    x_stride,
    k_ptr,
    ws_ptr,
    ws_stride,
    cand_ptr,
    V,
    HAS_P: tl.constexpr,
    BLOCK: tl.constexpr,
    WS_GCNT: tl.constexpr,
    WS_NTB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    xrow = x_ptr + row * x_stride
    x = tl.load(xrow + offs, mask=m, other=float("-inf"))
    o = _ordered(x)
    kk = tl.load(k_ptr + row)
    ws_row = ws_ptr + row * ws_stride
    k_ok = (kk >= 1) & (kk <= 64)
    thr_ord, kk_rem, g, n_t = _select(ws_row, kk, 4)
    # top-k: everything strictly below the k-th largest goes to -inf (ties kept)
    below = m & k_ok & (o < thr_ord) & (x != float("-inf"))
    tl.store(xrow + offs, float("-inf"), mask=below)
    if HAS_P:
        eq = m & k_ok & (o == thr_ord)
        tl.store(ws_row + WS_NTB + b, tl.sum(eq.to(tl.int32), 0))
        is_g = m & k_ok & (o > thr_ord)
        slot = tl.atomic_add(ws_row + WS_GCNT + offs * 0, 1, mask=is_g)
        key = (o.to(tl.int64) << 32) | offs.to(tl.int64)
        tl.store(cand_ptr + row * 64 + slot, key, mask=is_g & (slot < 64))


@triton.jit
def _decide(x_ptr, x_stride, k_ptr, p_ptr, ws_ptr, ws_stride, cand_ptr, row):
    """Top-p decision of one row over its survivors; scatters -inf into the
    masked strictly-above-threshold tokens and returns (thr_ord, c) where `c`
    is the number of tied-at-threshold tokens (lowest indices first) that
    top-p consumes. Pure function of completed workspace state, so every
    block program of a row may replay it (duplicate stores are identical)."""
    kk = tl.load(k_ptr + row)
    ws_row = ws_ptr + row * ws_stride
    k_ok = (kk >= 1) & (kk <= 64)
    thr_ord, kk_rem, g, n_t = _select(ws_row, kk, 4)
    g = tl.where(k_ok, g, 0)
    thr = _unordered(thr_ord)
    q = 1.0 - tl.load(p_ptr + row)  # fp32, same expression as the reference

    j = tl.arange(0, 64)
    valid = j < g
    key = tl.load(cand_ptr + row * 64 + j, mask=valid, other=-9223372036854775808)
    ks = tl.sort(key, descending=True)  # desc by value; equal values: larger index first
    v = _unordered((ks >> 32).to(tl.int32))
    idx = (ks & 0xFFFFFFFF).to(tl.int32)
    mx = tl.max(tl.where(valid, v, thr), 0)  # row max (== thr when g == 0)
    e = tl.where(valid, tldevice.exp(v - mx), 0.0)
    e_thr = tldevice.exp(thr - mx)
    n_tf = n_t.to(tl.float32)
    S = tl.sum(e, 0) + n_tf * e_thr
    prob = e / S
    p_t = e_thr / S
    # ascending (stable) order == flipped descending order; tied group first.
    prob_a = tl.flip(prob, 0)
    valid_a = tl.flip(valid, 0)
    idx_a = tl.flip(idx, 0)
    cs = tl.cumsum(prob_a, 0) + n_tf * p_t
    mask_g = valid_a & (cs <= q) & (j != 63)  # last element is never masked
    tl.store(x_ptr + row * x_stride + idx_a, float("-inf"), mask=mask_g)

    # number of tied tokens (lowest indices first) consumed by top-p
    all_ties = (e_thr == 0.0) | (S == float("inf"))
    jf = tl.minimum(tl.maximum(q / p_t, 0.0), n_tf)
    j0 = tl.where(all_ties, n_t, jf.to(tl.int32))
    ok_hi = ((j0 + 1).to(tl.float32) * p_t <= q) & (j0 + 1 <= n_t)
    ok_0 = j0.to(tl.float32) * p_t <= q
    ok_lo = ((j0 - 1).to(tl.float32) * p_t <= q) & (j0 >= 1)
    c = tl.where(ok_hi, j0 + 1, tl.where(ok_0, j0, tl.where(ok_lo, j0 - 1, 0)))
    c = tl.where(all_ties, n_t, c)
    c = tl.where(g == 0, tl.minimum(c, n_t - 1), c)
    c = tl.where(mx == float("-inf"), 0, c)  # all -inf row: reference masks nothing
    c = tl.where(k_ok, c, 0)
    return thr_ord, c


@triton.jit
def _apply_topp_kernel(
    x_ptr,
    x_stride,
    k_ptr,
    p_ptr,
    ws_ptr,
    ws_stride,
    cand_ptr,
    V,
    BLOCK: tl.constexpr,
    NB_PAD: tl.constexpr,
    WS_NTB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    thr_ord, c = _decide(x_ptr, x_stride, k_ptr, p_ptr, ws_ptr, ws_stride, cand_ptr, row)
    if c > 0:
        ws_row = ws_ptr + row * ws_stride
        bb = tl.arange(0, NB_PAD)
        prefix = tl.sum(tl.load(ws_row + WS_NTB + bb, mask=bb < b, other=0), 0)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        m = offs < V
        xrow = x_ptr + row * x_stride
        x = tl.load(xrow + offs, mask=m, other=float("-inf"))
        eq = m & (_ordered(x) == thr_ord)
        eqi = eq.to(tl.int32)
        rank = prefix + tl.cumsum(eqi, 0) - eqi
        mask_t = eq & (rank < c) & (x != float("-inf"))
        tl.store(xrow + offs, float("-inf"), mask=mask_t)


def apply_top_k_top_p_fastk(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """In-place top-k (+ optional top-p) mask; see module docstring.

    Preconditions (checked by the caller from CPU-side metadata, never here,
    to avoid a device sync): `k` is not None and 1 <= k[i] <= FASTK_CAP for
    every row.
    """
    assert logits.ndim == 2 and logits.dtype == torch.float32
    assert logits.stride(1) == 1
    B, V = logits.shape
    if B == 0:
        return logits
    NB = triton.cdiv(V, _BLOCK)
    NB_PAD = triton.next_power_of_2(NB)
    ws_stride = _WS_NTB + NB_PAD
    ws = torch.zeros(B, ws_stride, dtype=torch.int32, device=logits.device)
    k32 = k if k.dtype == torch.int32 else k.to(torch.int32)
    has_p = p is not None
    if has_p:
        cand = torch.empty(B, 64, dtype=torch.int64, device=logits.device)
        p32 = p if p.dtype == torch.float32 else p.to(torch.float32)
    else:
        cand = ws  # dummy pointer, never touched
        p32 = ws

    grid = (B, NB)
    for d in range(_NUM_DIGITS):
        _radix_pass_kernel[grid](
            logits, logits.stride(0), k32, ws, ws_stride, V,
            D=d, BLOCK=_BLOCK, num_warps=8,
        )
    _apply_topk_kernel[grid](
        logits, logits.stride(0), k32, ws, ws_stride, cand, V,
        HAS_P=has_p, BLOCK=_BLOCK, WS_GCNT=_WS_GCNT, WS_NTB=_WS_NTB, num_warps=8,
    )
    if has_p:
        _apply_topp_kernel[grid](
            logits, logits.stride(0), k32, p32, ws, ws_stride, cand, V,
            BLOCK=_BLOCK, NB_PAD=NB_PAD, WS_NTB=_WS_NTB, num_warps=8,
        )
    return logits
