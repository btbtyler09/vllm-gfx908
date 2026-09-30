"""gfx908 HC: VLLM_GFX908_HC_FINAL_W8 (final mixer on the W8 fused chain).

Numerics (numpy, exact-order replay of the shipping ``hc_w8_gemv_kernel`` configs,
ported from the small-M gate on branch gfx908-mtp-smallm,
tests/gfx908/test_hc_w8_smallm_ref.py): the final mixer is the combining mixer's chain
without the 4 inject + 12 pad rows of mix_down, i.e. ``hc_w8_gemv`` epi 1 with Nrows ==
R == 320, then epi 2 on the same permuted mix_up.

1. W8 final chain vs an fp64 reference of the same (q, s) math, within the agents/hc_w8
   tolerance (2 bf16 ulp + fp32 accumulation bound) -- the tolerance the combining chain
   ships with;
2. the final chain's block output is bit-identical to the combining chain's when the
   combining mix_down carries the same first 320 rows (per-row arithmetic does not
   depend on Nrows), so it inherits the combining mixer's measured W8 accuracy exactly;
3. the W8-vs-bf16 error of the chain is reported (same number for both mixers by 2.).

Routing (torch, real gfx908_hc_fused.py with CPU stand-ins, see
test_hc_strided_inject.py): the flag gates, the final mixer's weights are prepared, the
W8 chain runs for M <= max_m (2 launches) and the freed-weight fallback above it.

Needs numpy + torch: run in a CPU-only container.
Run: python -m pytest --noconftest tests/gfx908/test_hc_final_w8.py
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

HC = 4
HIDDEN = 2560
LORA = 320
K_DOWN = HC * HIDDEN
GS_DOWN = 128
GS_UP = 64
f32 = np.float32
ROOT = pathlib.Path(__file__).resolve().parents[2]


# ------------------------------------------------------ exact-order replay (ported)
def bf16_round(x):
    u = np.ascontiguousarray(x, dtype=f32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(f32)


def quantize_w8(w, gs):
    n, k = w.shape
    v = w.astype(f32).reshape(n, k // gs, gs)
    sc = np.maximum(np.abs(v).max(2), f32(1e-12)) / f32(127.0)
    q = np.clip(np.round(v / sc[..., None]), -127, 127).astype(np.int8).reshape(n, k)
    return q, sc.astype(f32)


def chunk_dots(q, x):
    n, k = q.shape
    m = x.shape[0]
    qc = q.astype(f32).reshape(1, n, k // 16, 16)
    xc = x.astype(f32).reshape(m, 1, k // 16, 16)
    t = np.zeros((m, n, k // 16), dtype=f32)
    for i in range(16):
        t = (t + qc[..., i] * xc[..., i]).astype(f32)
    return t


def fma32(a, b, c):
    return (
        np.asarray(a, np.float64) * np.asarray(b, np.float64)
        + np.asarray(c, np.float64)
    ).astype(f32)


def dpp_reduce(v, lpr):
    v = v.astype(f32).copy()
    lane = np.arange(64)
    for n in (8, 4, 2, 1):
        src = np.zeros_like(v)
        ok = (lane % 16) >= n
        src[..., ok] = v[..., lane[ok] - n]
        v = (v + src).astype(f32)
    src = np.zeros_like(v)
    for r in range(1, 4):
        src[..., 16 * r : 16 * r + 16] = v[..., 16 * r - 1 : 16 * r]
    v = (v + src).astype(f32)
    if lpr == 64:
        src = np.zeros_like(v)
        src[..., 32:64] = v[..., 31:32]
        v = (v + src).astype(f32)
    return v


def silu_epi(v, rows, r, hc):
    t = (bf16_round(v) / f32(hc)).astype(f32)
    s = (t / (f32(1) + np.exp(-t).astype(f32))).astype(f32)
    return np.where(rows < r, s, v).astype(f32)


def gate_mix_epi(g, xn, hidden):
    m = g.shape[0]
    sg = (f32(1) / (f32(1) + np.exp(-bf16_round(g)).astype(f32))).astype(f32)
    xs = xn.reshape(m, HC, hidden).transpose(0, 2, 1).astype(f32)
    acc = np.zeros((m, hidden), dtype=f32)
    for s in range(HC):
        acc = fma32(sg[..., s], xs[..., s], acc)
    return bf16_round((acc / f32(HC)).astype(f32))


def down_ship(q, s, x, gs, r, hc, ks):
    """hc_w8_gemv_kernel<*, *, NB, GRP, *, 64, KS, 1> (per-row arithmetic only)."""
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
            seg_t[..., : hi - lo] = tt[..., lo:hi]
            seg_s[..., : hi - lo] = ss[..., lo:hi]
            lane_sum = np.where(
                np.arange(64) < hi - lo, fma32(seg_t, seg_s, lane_sum), lane_sum
            )
        parts.append(dpp_reduce(lane_sum, 64)[..., 63])
    v = np.zeros((m, n), dtype=f32)
    for pv in parts:
        v = (v + pv).astype(f32)
    return bf16_round(silu_epi(v, np.arange(n)[None], r, hc))


def up_ship(qp, sp, lora, xn, gs, hidden):
    """hc_w8_gemv_kernel<4, 1, NB, true, true, 32, 1, 2>."""
    t = chunk_dots(qp, lora)
    cpr = t.shape[2]
    sc = np.repeat(sp, gs // 16, axis=1)[None]
    p = fma32(t, sc, 0.0)
    lanes = np.zeros(p.shape[:2] + (64,), dtype=f32)
    lanes[..., :cpr] = p
    g = dpp_reduce(lanes, 32)[..., 31]
    return gate_mix_epi(g.reshape(lora.shape[0], hidden, HC), xn, hidden)


def down_ref64(q, s, x, gs, r, hc):
    n, k = q.shape
    w = (
        q.astype(np.float64).reshape(n, k // gs, gs) * s[..., None].astype(np.float64)
    ).reshape(n, k)
    v = x.astype(np.float64) @ w.T
    a = np.abs(x.astype(np.float64)) @ np.abs(w).T
    rows = np.arange(n)[None]
    t = bf16_round(v.astype(f32)).astype(np.float64) / hc
    return np.where(rows < r, t / (1 + np.exp(-t)), v), v, a


def up_ref64(qp, sp, lora, xn, gs, hidden):
    n, k = qp.shape
    m = lora.shape[0]
    w = (
        qp.astype(np.float64).reshape(n, k // gs, gs) * sp[..., None].astype(np.float64)
    ).reshape(n, k)
    g = (lora.astype(np.float64) @ w.T).reshape(m, hidden, HC)
    a = (np.abs(lora.astype(np.float64)) @ np.abs(w).T).reshape(m, hidden, HC)
    sg = 1 / (1 + np.exp(-bf16_round(g.astype(f32)).astype(np.float64)))
    xs = xn.reshape(m, HC, hidden).transpose(0, 2, 1).astype(np.float64)
    return (sg * xs).mean(-1), g, a


# KS per M from _CFG_DOWN (the per-row order depends only on KS)
SHIP_KS = {1: 2, 2: 2, 3: 4, 4: 1}
PERM = np.arange(HC * HIDDEN).reshape(HC, HIDDEN).T.reshape(-1)


def make_final(m, seed):
    rng = np.random.default_rng(seed)
    wd = rng.normal(0, 0.02, (LORA, K_DOWN)).astype(
        f32
    )  # final mix_down: no inject rows
    wu = rng.normal(0, 0.02, (HC * HIDDEN, LORA)).astype(f32)
    for w in (wd, wu):  # input-channel outliers as in agents/hc_w8
        ch = rng.choice(w.shape[1], max(1, w.shape[1] // 100), replace=False)
        w[:, ch] *= 20
        w[rng.random(w.shape) < 1e-3] *= 50
    wd, wu = bf16_round(wd), bf16_round(wu)
    xn = bf16_round(rng.normal(0, 1, (m, K_DOWN)).astype(f32))
    xn[:, rng.choice(K_DOWN, 8, replace=False)] *= 8
    return wd, wu, bf16_round(xn), rng


def bf16_stock_chain(wd, wu, xn):
    """The stock bf16 chain.

    gemm -> bf16 -> hc_silu -> bf16 -> gemm -> bf16 -> gate_mix.
    """
    down = bf16_round((xn.astype(np.float64) @ wd.astype(np.float64).T).astype(f32))
    t = down.astype(np.float64) / HC
    lora = bf16_round((t / (1 + np.exp(-t))).astype(f32))
    gate = bf16_round((lora.astype(np.float64) @ wu.astype(np.float64).T).astype(f32))
    m = xn.shape[0]
    sg = 1 / (1 + np.exp(-gate.astype(np.float64).reshape(m, HC, HIDDEN)))
    return (sg * xn.astype(np.float64).reshape(m, HC, HIDDEN)).mean(1)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_final_w8_chain_numerics(m):
    wd, wu, xn, rng = make_final(m, seed=10 + m)
    qd, sd = quantize_w8(wd, GS_DOWN)
    qu, su = quantize_w8(wu[PERM], GS_UP)
    ks = SHIP_KS[m]

    lora = down_ship(qd, sd, xn, GS_DOWN, LORA, HC, ks)  # [m, 320], every row silu'd
    out = up_ship(qu, su, lora, xn, GS_UP, HIDDEN)
    dref, dv, da = down_ref64(qd, sd, xn, GS_DOWN, LORA, HC)
    uref, ug, ua = up_ref64(qu, su, lora, xn, GS_UP, HIDDEN)

    # 1. agents/hc_w8 tolerance, as in the small-M gate
    ulp = 2.0**-7
    acc_d = da * (K_DOWN / 16 + 64) * 2.0**-24
    tol_d = 2 * ulp * np.abs(dref) + (1.1 / HC) * (acc_d + ulp * np.abs(dv)) + 1e-30
    acc_u = ua * (LORA / 16 + 32) * 2.0**-24
    xs = np.abs(xn.reshape(m, HC, HIDDEN).transpose(0, 2, 1).astype(np.float64))
    tol_u = (
        2 * ulp * np.abs(uref)
        + (0.25 * (acc_u + ulp * np.abs(ug)) * xs).sum(-1) / HC
        + 1e-30
    )
    assert np.max(np.abs(lora - dref) / tol_d) <= 1.0
    assert np.max(np.abs(out - uref) / tol_u) <= 1.0

    # 2. same arithmetic as the combining chain: append 4 inject + 12 pad rows
    extra = bf16_round(rng.normal(0, 0.02, (16, K_DOWN)).astype(f32))
    qd336, sd336 = quantize_w8(np.concatenate([wd, extra]), GS_DOWN)
    assert np.array_equal(qd336[:LORA], qd) and np.array_equal(sd336[:LORA], sd)
    comb_down = down_ship(qd336, sd336, xn, GS_DOWN, LORA, HC, ks)
    comb_out = up_ship(qu, su, comb_down[:, :LORA], xn, GS_UP, HIDDEN)
    assert np.array_equal(comb_down[:, :LORA], lora)
    assert np.array_equal(comb_out, out)

    # 3. W8 vs the bf16 chain the flag replaces: reported, loosely bounded (the
    #    ~1.5-2 % row error of int8 gs128/gs64 on outlier weights, agents/hc_w8)
    ref = bf16_stock_chain(wd, wu, xn)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    print(f"M={m}: final mixer W8 vs bf16 chain rel-L2 {rel:.4%}")
    assert rel < 0.03


# ----------------------------------------------------------- routing (torch)
def test_hyperconnection_routes_final_mixer_through_the_op():
    src = (ROOT / "vllm/models/qwen4_exp/amd/hyperconnection.py").read_text()
    tree = ast.parse(src)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GatedResidual"
    )
    fns = {f.name: ast.unparse(f) for f in cls.body if isinstance(f, ast.FunctionDef)}
    for name in ("mix", "combine_and_mix"):
        assert (
            "if not self.use_combine and self._gfx908_hc_final_op_applies(xn):"
            in fns[name]
        )
    assert (
        "self._gfx908_hc_shard or self._gfx908_hc_final_w8"
        in fns["_gfx908_hc_final_op_applies"]
    )
    init = fns["__init__"]
    assert "hc_final_w8_enabled()" in init and "install_hc_w8_prepare(self)" in init


def _torch_mod(monkeypatch, final_flag, w4=False, free=True):
    torch = pytest.importorskip("torch")
    import test_hc_strided_inject as S  # same directory (pytest rootdir import)

    monkeypatch.setenv("VLLM_GFX908_HC_FINAL_W8", "1" if final_flag else "0")
    mod = S._load(monkeypatch, strided=False, free=free)
    mod._W8_FLAG, mod._W4_FLAG = (False, True) if w4 else (True, False)
    return torch, S, mod


def test_flag_gating(monkeypatch):
    _, _, mod = _torch_mod(monkeypatch, final_flag=False)
    assert not mod.hc_final_w8_enabled()
    _, _, mod = _torch_mod(monkeypatch, final_flag=True)
    assert mod.hc_final_w8_enabled()
    _, _, mod = _torch_mod(monkeypatch, final_flag=True, w4=True)
    assert not mod.hc_final_w8_enabled()


@pytest.mark.parametrize("free", [True, False], ids=["free_bf16", "keep_bf16"])
def test_final_prepare_and_dispatch(monkeypatch, free):
    torch, S, mod = _torch_mod(monkeypatch, final_flag=True, free=free)
    BF = torch.bfloat16
    g = torch.Generator().manual_seed(3)
    wd = (torch.randn(LORA, K_DOWN, generator=g) * 0.02).to(BF)
    wu = (torch.randn(K_DOWN, LORA, generator=g) * 0.02).to(BF)

    # install_hc_w8_prepare on a final mixer (no input_mix_weight_down_block_inject)
    loaded = []

    def layer(w):
        qm = SimpleNamespace(
            process_weights_after_loading=lambda mod_: loaded.append(mod_)
        )
        return SimpleNamespace(weight=w, quant_method=qm)

    gr = SimpleNamespace(
        hc_count=HC,
        hidden_size=HIDDEN,
        input_mix_weight_down=layer(wd),
        input_mix_weight_up=layer(wu),
    )
    mod.install_hc_w8_prepare(gr)
    for lyr in (gr.input_mix_weight_down, gr.input_mix_weight_up):
        lyr.quant_method.process_weights_after_loading(lyr)
    assert len(loaded) == 2 and len(mod._W8_CACHE) == 2
    ed = mod._W8_CACHE[mod._w8_key(gr.input_mix_weight_down.weight)]
    eu = mod._W8_CACHE[mod._w8_key(gr.input_mix_weight_up.weight)]
    assert ed.kind == "down" and ed.q.shape == (LORA, K_DOWN) and ed.freed == free
    assert eu.kind == "up" and eu.q.shape == (K_DOWN, LORA) and eu.freed == free

    wds, wus = gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight
    # dequantized reference chain (what the freed fallback computes)
    wdq = (ed.q.float().view(LORA, -1, ed.gs) * ed.s[..., None]).view(LORA, K_DOWN)
    wuq = (eu.q.float().view(K_DOWN, -1, eu.gs) * eu.s[..., None]).view(K_DOWN, LORA)
    for m in (1, 2, 3, 4, 5, 6):
        xn = S._xn(m)
        S.FakeW8Ext.calls = 0
        y = mod._hc_final_mix_impl(xn, wds, wus, HC, LORA, HIDDEN)
        assert y.shape == (m, HIDDEN) and y.is_contiguous() and y.dtype == BF
        # M <= max_m (4): the 2-launch W8 chain; above: the freed-weight fallback
        # (FREE=1) or the untouched bf16 stock chain (FREE=0, the rc10 serve default)
        assert S.FakeW8Ext.calls == (2 if m <= 4 else 0), (m, S.FakeW8Ext.calls)
        lora = S._bf(xn.float() @ wdq.t())
        t = lora / HC
        lora = S._bf(t * torch.sigmoid(t))
        ref = S._gate_mix_perm_ref(lora @ wuq.t(), xn, HC, HIDDEN)
        assert torch.allclose(y.float(), ref, rtol=2e-2, atol=2e-2), m
    assert any("final mixer ran the W8 fused chain" in ln for ln in mod.logger_.lines)
