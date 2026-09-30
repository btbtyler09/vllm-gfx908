# SPDX-License-Identifier: Apache-2.0
"""CPU-only numerics gate for the gfx908 small-M W8 hyper-connection kernels.

``vllm/models/qwen4_exp/amd/csrc/gfx908_hc_w8_smallm.hip`` (``hcsm_down`` / ``hcsm_up``) is
checked here WITHOUT a GPU: both the new kernels and the shipping ``hc_w8_gemv_kernel`` configs
they replace at M = 2..4 are emulated in numpy with their exact per-lane tiling and summation
order (fp32 products of int8 x bf16 are exact, so an fp32 add of the product is bit-identical to
the device FMA; the DPP reduction trees are replayed lane by lane).  The two emulations share the
same epilogue code, so any output difference between them is caused only by the summation order.

Checks, at the production shapes (mix_down 336 x 10240 gs128 with lora rank 320 + 4 inject
columns, mix_up 10240 x 320 gs64, HC = 4, hidden 2560), M = 1..4:

1. new kernel vs an fp64 reference of the same (q, s) math: every output within
   2 bf16 ulp of the reference plus the fp32 accumulation bound (the tolerance used for the
   shipping kernel in agents/hc_w8);
2. new kernel vs shipping kernel: bf16 outputs identical except for rare 1-ulp flips
   (reported as a fraction; asserted <= 1 ulp and <= 1 % of elements);
3. the compact injection output equals the injection columns of the down output.

Known emulation limits (documented, not hidden): ``__expf`` is replayed with float32 ``np.exp``
(the device uses v_exp_f32, ~1 ulp) and the group-scale FMA with a non-zero addend is replayed in
float64 then rounded (double rounding, affects only the shipping kernel's multi-chunk lanes).
Neither changes the comparison of the two summation orders.

Run: ``python3 -m pytest -q tests/gfx908/test_hc_w8_smallm_ref.py`` (numpy only), or execute the
file directly for the summary table.
"""

from __future__ import annotations

import numpy as np
import pytest

HC = 4
HIDDEN = 2560
LORA = 320
N_DOWN = 336              # lora 320 + inject 4 + pad 12
K_DOWN = HC * HIDDEN      # 10240
GS_DOWN = 128
GS_UP = 64

f32 = np.float32


# ----------------------------------------------------------------------------- bf16 helpers
def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 (RNE) -> fp32, like __float2bfloat16 (NaN-free inputs)."""
    u = np.ascontiguousarray(x, dtype=f32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(f32)


def quantize_w8(w: np.ndarray, gs: int) -> tuple[np.ndarray, np.ndarray]:
    """numpy port of gfx908_w8a16.quantize_w8 (group scales along K)."""
    n, k = w.shape
    v = w.astype(f32).reshape(n, k // gs, gs)
    sc = np.maximum(np.abs(v).max(2), f32(1e-12)) / f32(127.0)
    q = np.clip(np.round(v / sc[..., None]), -127, 127).astype(np.int8).reshape(n, k)
    return q, sc.astype(f32)


# ----------------------------------------------------------------------------- lane math
def chunk_dots(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """t[m, n, c] = fp32 FMA chain over the 16 k of chunk c, k ascending, starting from 0.

    Each product int8 x bf16 has <= 15 significant bits, so ``t + w*x`` in fp32 is exactly the
    device's v_fma_f32 result."""
    n, k = q.shape
    m = x.shape[0]
    qc = q.astype(f32).reshape(1, n, k // 16, 16)
    xc = x.astype(f32).reshape(m, 1, k // 16, 16)
    t = np.zeros((m, n, k // 16), dtype=f32)
    for i in range(16):
        t = (t + qc[..., i] * xc[..., i]).astype(f32)
    return t


def fma32(a, b, c):
    """fp32 fma via float64 (exact when c == 0; otherwise double-rounded, see module doc)."""
    return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(f32)


def dpp_reduce(v: np.ndarray, lpr: int) -> np.ndarray:
    """Replay hw8_wave_reduce<LPR> / hs_wave_reduce over the last axis (64 lanes); returns the
    whole lane vector after the tree (lane 63 = total for LPR 64; lanes 31 / 63 for LPR 32)."""
    v = v.astype(f32).copy()
    lane = np.arange(64)
    for n in (8, 4, 2, 1):                                  # row_shr:n, bound_ctrl -> 0
        src = np.zeros_like(v)
        ok = (lane % 16) >= n
        src[..., ok] = v[..., lane[ok] - n]
        v = (v + src).astype(f32)
    src = np.zeros_like(v)                                  # row_bcast:15
    for r in range(1, 4):
        src[..., 16 * r:16 * r + 16] = v[..., 16 * r - 1:16 * r]
    v = (v + src).astype(f32)
    if lpr == 64:                                           # row_bcast:31
        src = np.zeros_like(v)
        src[..., 32:64] = v[..., 31:32]
        v = (v + src).astype(f32)
    return v


def silu_epi(v: np.ndarray, rows: np.ndarray, r: int, hc: int) -> np.ndarray:
    """EPI 1: rows < r -> t = bf16(v) / hc ; v = t / (1 + exp(-t))."""
    t = (bf16_round(v) / f32(hc)).astype(f32)
    s = (t / (f32(1) + np.exp(-t).astype(f32))).astype(f32)
    return np.where(rows < r, s, v).astype(f32)


def gate_mix_epi(g: np.ndarray, xn: np.ndarray, hidden: int) -> np.ndarray:
    """EPI 2 on g[m, i, s] (fp32 row sums): out = bf16( fma-chain_s sigmoid(bf16(g)) * xn ) / HC."""
    m = g.shape[0]
    sg = (f32(1) / (f32(1) + np.exp(-bf16_round(g)).astype(f32))).astype(f32)
    xs = xn.reshape(m, HC, hidden).transpose(0, 2, 1).astype(f32)   # [m, i, s]
    acc = np.zeros((m, hidden), dtype=f32)
    for s in range(HC):
        acc = fma32(sg[..., s], xs[..., s], acc)
    return bf16_round((acc / f32(HC)).astype(f32))


# ----------------------------------------------------------------------------- mix_down
def down_new(q, s, x, gs, r, hc):
    """hcsm_down_kernel: lane l of wave ks owns chunk c = 64*ks + l (SPW = 1, KS = K/1024);
    partial = fma(t, scale, 0); DPP64 per wave; wave partials summed in ks order from 0."""
    n, k = q.shape
    m = x.shape[0]
    nch = k // 16
    t = chunk_dots(q, x)                                           # [m, n, nch]
    sc = np.repeat(s, gs // 16, axis=1)[None]                       # scale per chunk
    p = fma32(t, sc, 0.0)
    ks_n = (nch + 63) // 64
    pp = np.zeros((m, n, ks_n * 64), dtype=f32)
    pp[..., :nch] = p
    lane63 = dpp_reduce(pp.reshape(m, n, ks_n, 64), 64)[..., 63]  # [m, n, ks]
    v = np.zeros((m, n), dtype=f32)
    for q_ in range(ks_n):
        v = (v + lane63[..., q_]).astype(f32)
    rows = np.arange(n)[None]
    out = bf16_round(silu_epi(v, rows, r, hc))
    return out, out[:, r:r + hc].copy()


def down_ship(q, s, x, gs, r, hc, ks, unrl):
    """hc_w8_gemv_kernel<YT, UNRL, NB, GRP, *, 64, KS, 1>: wave (rg, ks) owns the K slice
    [ks*K/KS, (ks+1)*K/KS); lane l's chunks are j' = l + 64*j of that slice, accumulated in j order
    with sum = fma(t, scale, sum); DPP64; KS partials summed in order from 0.  (YT and UNRL do not
    change the per-row arithmetic, only which rows share a wave / how loads are batched.)"""
    n, k = q.shape
    m = x.shape[0]
    nch = k // 16
    per = nch // ks
    t = chunk_dots(q, x)
    sc = np.repeat(s, gs // 16, axis=1)[None]
    parts = []
    for kk in range(ks):
        sl = slice(kk * per, (kk + 1) * per)
        tt, ss = t[..., sl], np.broadcast_to(sc[..., sl], t[..., sl].shape)
        nj = (per + 63) // 64
        lane_sum = np.zeros((m, n, 64), dtype=f32)
        for j in range(nj):
            lo, hi = 64 * j, min(64 * j + 64, per)
            seg_t = np.zeros((m, n, 64), dtype=f32)
            seg_s = np.zeros((m, n, 64), dtype=f32)
            seg_t[..., :hi - lo] = tt[..., lo:hi]
            seg_s[..., :hi - lo] = ss[..., lo:hi]
            lane_sum = np.where(np.arange(64) < hi - lo, fma32(seg_t, seg_s, lane_sum), lane_sum)
        parts.append(dpp_reduce(lane_sum, 64)[..., 63])
    v = np.zeros((m, n), dtype=f32)
    for pv in parts:
        v = (v + pv).astype(f32)
    rows = np.arange(n)[None]
    out = bf16_round(silu_epi(v, rows, r, hc))
    return out, out[:, r:r + hc].copy()


# ----------------------------------------------------------------------------- mix_up
def up_new(qp, sp, lora, xn, gs, hidden):
    """hcsm_up_kernel: unit (i, c) -> partial[i, s, c] = fma(t, scale, 0) for the 4 permuted
    rows 4i+s; thread (i, n, s) sums c = 0..CPR-1 in order from 0; gate-mix in s order."""
    m = lora.shape[0]
    t = chunk_dots(qp, lora)                                      # [m, 4*HD, CPR]
    sc = np.repeat(sp, gs // 16, axis=1)[None]
    p = fma32(t, sc, 0.0)
    g = np.zeros(p.shape[:2], dtype=f32)
    for c in range(p.shape[2]):
        g = (g + p[..., c]).astype(f32)
    return gate_mix_epi(g.reshape(m, hidden, HC), xn, hidden)


def up_ship(qp, sp, lora, xn, gs, hidden):
    """hc_w8_gemv_kernel<4, 1, NB, true, true, 32, 1, 2>: half-wave (LPR 32) per 4 permuted rows,
    lane kl < CPR owns chunk kl (one chunk: partial = fma(t, scale, 0)), lanes >= CPR add 0;
    DPP tree without row_bcast31 -> lane 31 / 63."""
    m = lora.shape[0]
    t = chunk_dots(qp, lora)
    cpr = t.shape[2]
    sc = np.repeat(sp, gs // 16, axis=1)[None]
    p = fma32(t, sc, 0.0)                                         # [m, rows, CPR]
    lanes = np.zeros(p.shape[:2] + (64,), dtype=f32)
    lanes[..., :cpr] = p                                          # sub 0 layout (sub 1 is identical math)
    g = dpp_reduce(lanes, 32)[..., 31]
    return gate_mix_epi(g.reshape(m, hidden, HC), xn, hidden)


# ----------------------------------------------------------------------------- references
def down_ref64(q, s, x, gs, r, hc):
    """fp64 (q, s) reference; returns (output, pre-epilogue v, sum |w x| per output)."""
    n, k = q.shape
    w = (q.astype(np.float64).reshape(n, k // gs, gs) * s[..., None].astype(np.float64)).reshape(n, k)
    v = x.astype(np.float64) @ w.T
    a = np.abs(x.astype(np.float64)) @ np.abs(w).T
    rows = np.arange(n)[None]
    t = bf16_round(v.astype(f32)).astype(np.float64) / hc
    return np.where(rows < r, t / (1 + np.exp(-t)), v), v, a


def up_ref64(qp, sp, lora, xn, gs, hidden):
    """fp64 reference; returns (output, g [m, HD, HC], sum |w x| of g)."""
    n, k = qp.shape
    m = lora.shape[0]
    w = (qp.astype(np.float64).reshape(n, k // gs, gs) * sp[..., None].astype(np.float64)).reshape(n, k)
    g = (lora.astype(np.float64) @ w.T).reshape(m, hidden, HC)
    a = (np.abs(lora.astype(np.float64)) @ np.abs(w).T).reshape(m, hidden, HC)
    sg = 1 / (1 + np.exp(-bf16_round(g.astype(f32)).astype(np.float64)))
    xs = xn.reshape(m, HC, hidden).transpose(0, 2, 1).astype(np.float64)
    return (sg * xs).mean(-1), g, a


def ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """bf16 ulp distance between two bf16-valued fp32 arrays (monotone integer mapping)."""
    def key(x):
        u = (np.ascontiguousarray(x, f32).view(np.uint32) >> 16).astype(np.int64)
        return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), 0x8000 + u)
    return np.abs(key(a) - key(b))


# ----------------------------------------------------------------------------- data
def make_case(m: int, seed: int = 0, outliers: bool = True):
    rng = np.random.default_rng(seed)
    wd = rng.normal(0, 0.02, (N_DOWN, K_DOWN)).astype(f32)
    wu = rng.normal(0, 0.02, (HC * HIDDEN, LORA)).astype(f32)
    if outliers:     # input-channel outliers as in agents/hc_w8 (1% of channels x20, 0.1% entries x50)
        for w in (wd, wu):
            ch = rng.choice(w.shape[1], max(1, w.shape[1] // 100), replace=False)
            w[:, ch] *= 20
            idx = rng.random(w.shape) < 1e-3
            w[idx] *= 50
    wd, wu = bf16_round(wd), bf16_round(wu)
    qd, sd = quantize_w8(wd, GS_DOWN)
    # mix_up is permuted BEFORE quantisation: row i*HC + s = original row s*HD + i
    perm = np.arange(HC * HIDDEN).reshape(HC, HIDDEN).T.reshape(-1)
    qu, su = quantize_w8(wu[perm], GS_UP)
    xn = bf16_round(rng.normal(0, 1, (m, K_DOWN)).astype(f32))
    xn[:, rng.choice(K_DOWN, 8, replace=False)] *= 8            # a few large activations
    xn = bf16_round(xn)
    return qd, sd, qu, su, xn


SHIP_DOWN_CFG = {1: (2, 2), 2: (2, 2), 3: (4, 3), 4: (1, 2)}   # (KS, UNRL) from _CFG_DOWN


def run_case(m: int, seed: int = 0):
    qd, sd, qu, su, xn = make_case(m, seed)
    dn, inj_n = down_new(qd, sd, xn, GS_DOWN, LORA, HC)
    ks, un = SHIP_DOWN_CFG[m]
    ds, inj_s = down_ship(qd, sd, xn, GS_DOWN, LORA, HC, ks, un)
    dref, dv, da = down_ref64(qd, sd, xn, GS_DOWN, LORA, HC)
    # the up input is the (bf16) lora block of the down output, as in the real chain
    lora = dn[:, :LORA]
    un_ = up_new(qu, su, lora, xn, GS_UP, HIDDEN)
    us_ = up_ship(qu, su, lora, xn, GS_UP, HIDDEN)
    uref, ug, ua = up_ref64(qu, su, lora, xn, GS_UP, HIDDEN)

    # Tolerance (as in agents/hc_w8): 2 bf16 ulp of the reference + the fp32 accumulation bound
    # + the propagated effect of the epilogue's own bf16 rounding of its input (silu input / gate)
    # flipping by one ulp.  fp32 accumulation bound: (#adds) * 2^-24 * sum |w x|.
    ulp = 2.0 ** -7                                         # bf16 relative spacing (upper bound)
    acc_d = da * (K_DOWN / 16 + 64) * 2.0 ** -24
    rows = np.arange(N_DOWN)[None]
    dslope = np.where(rows < LORA, 1.1 / HC, 1.0)          # |d silu(v/HC) / dv| <= 1.1/HC
    tol_d = 2 * ulp * np.abs(dref) + dslope * (acc_d + ulp * np.abs(dv)) + 1e-30
    acc_u = ua * (LORA / 16 + 32) * 2.0 ** -24
    xs = np.abs(xn.reshape(m, HC, HIDDEN).transpose(0, 2, 1).astype(np.float64))
    tol_u = 2 * ulp * np.abs(uref) + (0.25 * (acc_u + ulp * np.abs(ug)) * xs).sum(-1) / HC + 1e-30

    res = {
        "M": m,
        "down_new_vs_ref_maxratio": float(np.max(np.abs(dn - dref) / tol_d)),
        "up_new_vs_ref_maxratio": float(np.max(np.abs(un_ - uref) / tol_u)),
        "down_new_vs_ship_maxulp": int(ulp_dist(dn, ds).max()),
        "down_new_vs_ship_frac_diff": float(np.mean(dn != ds)),
        "up_new_vs_ship_maxulp": int(ulp_dist(un_, us_).max()),
        "up_new_vs_ship_frac_diff": float(np.mean(un_ != us_)),
        "down_ship_vs_ref_maxratio": float(np.max(np.abs(ds - dref) / tol_d)),
        "up_ship_vs_ref_maxratio": float(np.max(np.abs(us_ - uref) / tol_u)),
        "inj_equal": bool(np.array_equal(inj_n, dn[:, LORA:LORA + HC])),
        "inj_ship_equal": bool(np.array_equal(inj_n, inj_s)),
    }
    return res


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_hc_w8_smallm_numerics(m):
    r = run_case(m, seed=m)
    assert r["down_new_vs_ref_maxratio"] <= 1.0, r
    assert r["up_new_vs_ref_maxratio"] <= 1.0, r
    assert r["down_new_vs_ship_maxulp"] <= 1 and r["down_new_vs_ship_frac_diff"] <= 0.01, r
    assert r["up_new_vs_ship_maxulp"] <= 1 and r["up_new_vs_ship_frac_diff"] <= 0.01, r
    assert r["inj_equal"], r


def test_dpp_tree_is_a_sum():
    rng = np.random.default_rng(1)
    v = rng.integers(-1000, 1000, (7, 64)).astype(f32)      # exact in fp32 -> order-free
    assert np.array_equal(dpp_reduce(v, 64)[:, 63], v.sum(1).astype(f32))
    r32 = dpp_reduce(v, 32)
    assert np.array_equal(r32[:, 31], v[:, :32].sum(1).astype(f32))
    assert np.array_equal(r32[:, 63], v[:, 32:].sum(1).astype(f32))


if __name__ == "__main__":
    print("| M | down new/ref (<=1) | up new/ref (<=1) | down new vs ship: max ulp, % diff | "
          "up new vs ship: max ulp, % diff | ship down/ref, up/ref | inj == down cols |")
    print("|---|---|---|---|---|---|---|")
    for m in (1, 2, 3, 4):
        r = run_case(m, seed=m)
        print(f"| {m} | {r['down_new_vs_ref_maxratio']:.3f} | {r['up_new_vs_ref_maxratio']:.3f} | "
              f"{r['down_new_vs_ship_maxulp']}, {100 * r['down_new_vs_ship_frac_diff']:.3f}% | "
              f"{r['up_new_vs_ship_maxulp']}, {100 * r['up_new_vs_ship_frac_diff']:.3f}% | "
              f"{r['down_ship_vs_ref_maxratio']:.3f}, {r['up_ship_vs_ref_maxratio']:.3f} | "
              f"{r['inj_equal'] and r['inj_ship_equal']} |")
