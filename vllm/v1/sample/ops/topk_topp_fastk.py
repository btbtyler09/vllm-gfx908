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
_WS_ZERO = _WS_GCNT + 1  # the [0, _WS_ZERO) prefix is what needs zeroing
_WS_STATE = _WS_GCNT + 4  # carried radix state, one (prefix, kk_rem, above)
#                           slot per digit: pass D reads slot D-1, writes slot D
_WS_HAND = _WS_STATE + 3 * _NUM_DIGITS  # handoff to top-p: thr_ord, g, n_t
_WS_NTB = _WS_HAND + 3  # per-block tie counts [NB_PAD]


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
def _step(hist_ptr, prefix, kk, above, D: tl.constexpr):
    """One digit of the radix selection: identical arithmetic to one iteration
    of `_select`, but starting from the state the previous pass stored, so a
    pass never replays the digits before it (exact: the state is int32)."""
    bb = tl.arange(0, 256)
    h = tl.load(hist_ptr + D * 256 + bb)
    cum_top = tl.flip(tl.cumsum(tl.flip(h, 0), 0), 0)
    sel = tl.max(tl.where(cum_top >= kk, bb, -1), 0)
    hsel = tl.sum(tl.where(bb == sel, h, 0), 0)
    csel = tl.sum(tl.where(bb == sel, cum_top, 0), 0)
    above_d = csel - hsel
    kk = kk - above_d
    above = above + above_d
    if D == 0:
        prefix = sel - 128
    else:
        prefix = (prefix << 8) | sel
    return prefix, kk, above, hsel


@triton.jit
def _load_state(ws_row, k_ptr, row, S: tl.constexpr, D: tl.constexpr):
    """State after D digits, as stored by the previous pass (D == 0: initial)."""
    if D == 0:
        kk = tl.load(k_ptr + row)
        return kk * 0, kk, kk * 0
    q = S + 3 * D
    return (tl.load(ws_row + q + 0), tl.load(ws_row + q + 1), tl.load(ws_row + q + 2))


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
    WS_STATE_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    m = offs < V
    x = tl.load(x_ptr + row * x_stride + offs, mask=m, other=float("-inf"))
    o = _ordered(x)
    hist_row = ws_ptr + row * ws_stride
    kk0 = tl.load(k_ptr + row)
    # Rows with k outside [1, FASTK_CAP] are left untouched by every kernel
    # (insurance only: the caller guarantees the range from CPU metadata).
    k_ok = (kk0 >= 1) & (kk0 <= 64)
    if D > 0:
        prefix, kk, above = _load_state(hist_row, k_ptr, row, WS_STATE_C, D - 1)
        prefix, kk, above, hsel = _step(hist_row, prefix, kk, above, D - 1)
        # every program of the row computes the identical int32 state; it goes
        # to this digit's own slot, never over the slot the pass is reading
        tl.store(hist_row + WS_STATE_C + 3 * D + 0, prefix)
        tl.store(hist_row + WS_STATE_C + 3 * D + 1, kk)
        tl.store(hist_row + WS_STATE_C + 3 * D + 2, above)
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
    WS_STATE_C: tl.constexpr,
    WS_HAND_C: tl.constexpr,
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
    prefix, kkr, above = _load_state(ws_row, k_ptr, row, WS_STATE_C, 3)
    thr_ord, kk_rem, g, n_t = _step(ws_row, prefix, kkr, above, 3)
    if HAS_P:
        tl.store(ws_row + WS_HAND_C + 0, thr_ord)
        tl.store(ws_row + WS_HAND_C + 1, g)
        tl.store(ws_row + WS_HAND_C + 2, n_t)
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
def _decide(x_ptr, x_stride, k_ptr, p_ptr, ws_ptr, ws_stride, cand_ptr, row,
            WS_HAND_C: tl.constexpr):
    """Top-p decision of one row over its survivors; scatters -inf into the
    masked strictly-above-threshold tokens and returns (thr_ord, c) where `c`
    is the number of tied-at-threshold tokens (lowest indices first) that
    top-p consumes. Pure function of completed workspace state, so every
    block program of a row may replay it (duplicate stores are identical)."""
    kk = tl.load(k_ptr + row)
    ws_row = ws_ptr + row * ws_stride
    k_ok = (kk >= 1) & (kk <= 64)
    thr_ord = tl.load(ws_row + WS_HAND_C + 0)
    g = tl.load(ws_row + WS_HAND_C + 1)
    n_t = tl.load(ws_row + WS_HAND_C + 2)
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
    WS_HAND_C: tl.constexpr,
    WS_ZERO_C: tl.constexpr,
    ZB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    thr_ord, c = _decide(x_ptr, x_stride, k_ptr, p_ptr, ws_ptr, ws_stride, cand_ptr, row,
                         WS_HAND_C)
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
    # This kernel never reads the histograms (the top-k pass handed over the
    # three scalars it needs), so it is the one place where the workspace can
    # be zeroed for the next call without racing a reader: that removes the
    # per-call memset launch.  Each block program owns a disjoint slice.
    zz = b * ZB + tl.arange(0, ZB)
    tl.store(ws_ptr + row * ws_stride + zz, 0, mask=zz < WS_ZERO_C)


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
    k32 = k if k.dtype == torch.int32 else k.to(torch.int32)
    has_p = p is not None
    ZB = triton.next_power_of_2(triton.cdiv(_WS_ZERO, NB))
    persist = has_p and NB * ZB >= _WS_ZERO
    if persist:
        # The top-p kernel re-zeroes the workspace on its way out (it is the
        # only kernel that reads none of it), so the per-call memset is gone.
        ws = _workspace(B, ws_stride, logits.device)
        cand = _candidates(B, logits.device)
        p32 = p if p.dtype == torch.float32 else p.to(torch.float32)
    else:
        ws = torch.zeros(B, ws_stride, dtype=torch.int32, device=logits.device)
        cand = torch.empty(B, 64, dtype=torch.int64, device=logits.device) if has_p else ws
        p32 = (p if p.dtype == torch.float32 else p.to(torch.float32)) if has_p else ws

    grid = (B, NB)
    for d in range(_NUM_DIGITS):
        _radix_pass_kernel[grid](
            logits, logits.stride(0), k32, ws, ws_stride, V,
            D=d, BLOCK=_BLOCK, WS_STATE_C=_WS_STATE, num_warps=8,
        )
    _apply_topk_kernel[grid](
        logits, logits.stride(0), k32, ws, ws_stride, cand, V,
        HAS_P=has_p, BLOCK=_BLOCK, WS_GCNT=_WS_GCNT, WS_NTB=_WS_NTB,
        WS_STATE_C=_WS_STATE, WS_HAND_C=_WS_HAND, num_warps=8,
    )
    if has_p:
        _apply_topp_kernel[grid](
            logits, logits.stride(0), k32, p32, ws, ws_stride, cand, V,
            BLOCK=_BLOCK, NB_PAD=NB_PAD, WS_NTB=_WS_NTB, WS_HAND_C=_WS_HAND,
            WS_ZERO_C=(_WS_ZERO if persist else 0), ZB=ZB, num_warps=8,
        )
    return logits


# Persistent, zero-on-creation scratch: the top-p kernel leaves it zeroed for
# the next call, which is what removes the memset launch.  Keyed by shape, so a
# CUDA-graph capture always replays against the same pointers.
_WS_CACHE: dict = {}
_CAND_CACHE: dict = {}


def _workspace(B: int, ws_stride: int, device) -> torch.Tensor:
    key = (str(device), B, ws_stride)
    ws = _WS_CACHE.get(key)
    if ws is None:
        ws = torch.zeros(B, ws_stride, dtype=torch.int32, device=device)
        _WS_CACHE[key] = ws
    return ws


def _candidates(B: int, device) -> torch.Tensor:
    key = (str(device), B)
    c = _CAND_CACHE.get(key)
    if c is None:
        c = torch.empty(B, 64, dtype=torch.int64, device=device)
        _CAND_CACHE[key] = c
    return c


def reset_workspace() -> None:
    """Re-zero the cached scratch (only needed if a call was interrupted)."""
    for ws in _WS_CACHE.values():
        ws.zero_()
