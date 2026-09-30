"""gfx908 HC: VLLM_GFX908_HC_STRIDED_INJECT (strided injection view, no copy).

The real ``gfx908_hc_fused.py`` is loaded from source twice (flag off / on) with stubbed
vLLM imports and CPU stand-ins for the HIP extensions that enforce the same argument
contracts as the TORCH_CHECKs in the .hip sources.  Every dispatch path of
``_hc_mix_local`` / ``_hc_mix_impl`` is run with the same inputs under both settings and
must give

* bit-identical block_input and injection values (the flag only changes the layout), and
* under the flag, exactly the layout ``_hc_mix_fake`` advertises (shape, strides,
  storage offset, storage size), with no copy on every path whose producer writes the
  [M, down_rows] buffer.

A second test compiles a producer/consumer pair of custom ops with inductor on CPU,
using the real fake impl, and checks that the consumer receives the strided view (no
copy inserted) and that compiled == eager.

Needs torch: run in a CPU-only container.
Run: python -m pytest --noconftest tests/gfx908/test_hc_strided_inject.py
"""

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm/models/qwen4_exp/amd/gfx908_hc_fused.py"

HC, HD, R = 4, 2560, 320
NDOWN = 336  # lora 320 + inject 4 + pad 12
K = HC * HD
BF = torch.bfloat16


def _bf(t):
    return t.to(BF).float()


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(msg % a if a else msg)

    info_once = warning = warning_once = debug = info


# ------------------------------------------------------- CPU stand-ins for the exts
def _silu_epi(v, r, hc):
    t = _bf(v) / hc
    y = t * torch.sigmoid(t)
    out = v.clone()
    out[:, :r] = y[:, :r]
    return out


def _gate_mix_perm_ref(g, xn, hc, hd):
    m = g.shape[0]
    sg = torch.sigmoid(_bf(g)).view(m, hd, hc)
    x = xn.float().view(m, hc, hd).transpose(1, 2)
    return (sg * x).mean(-1)


class FakeW8Ext:
    """hc_w8_gemv / hc_w8_dequant with the argument checks of gfx908_wv_fused_w8.hip."""

    calls = 0

    def hc_w8_gemv(self, W, S, X, out, xn, epi, gs, yt, un, lpr, ks, hc, r, cu):
        FakeW8Ext.calls += 1
        assert W.dtype == torch.int8 and W.is_contiguous()
        assert X.dtype == BF and X.stride(1) == 1 and X.shape[1] == W.shape[1]
        assert X.shape[0] == 1 or X.stride(0) % 8 == 0
        assert out.is_contiguous() and out.dtype == BF
        n, k = W.shape
        w = (W.float().view(n, k // gs, gs) * S[..., None]).view(n, k)
        v = X.float() @ w.t()
        if epi == 2:
            assert xn.is_contiguous()
            out.copy_(_gate_mix_perm_ref(v, xn, hc, out.shape[1]).to(BF))
        else:
            out.copy_((_silu_epi(v, r, hc) if epi == 1 else v).to(BF))
        return True

    def hc_w8_dequant(self, q, s, out, gs):
        return False  # exercise the module's torch fallback


class FakeBf16Ext:
    def wv_fused(self, w, x, out, xn, epi, hc, r, cu):
        assert x.stride(1) == 1 and out.is_contiguous()
        v = x.float() @ w.float().t()
        if epi == 2:
            out.copy_(_gate_mix_perm_ref(v, xn, hc, out.shape[1]).to(BF))
        else:
            out.copy_(_silu_epi(v, r, hc).to(BF))


class FakeW4Ext:
    """w4lt_hc_down / w4lt_hc_up; as in gfx908_w4_loadtime.hip, inj is contiguous."""

    weights: dict = {}

    def w4lt_hc_down(self, q4, s4, xn, lora, inj, gs4, hc, r, *cfg):
        assert inj.is_contiguous(), "w4lt_hc_down: TORCH_CHECK(inj.is_contiguous())"
        w = self.weights[q4.data_ptr()]
        v = xn.float() @ w.t()
        y = _silu_epi(v, r, hc)
        lora.copy_(y[:, :r].to(lora.dtype))
        inj.copy_(v[:, r : r + hc].to(BF))

    def w4lt_hc_up(self, q4, s4, lora, xn, y, gs4, hc, *cfg):
        w = self.weights[q4.data_ptr()]
        g = lora.float() @ w.t()
        y.copy_(_gate_mix_perm_ref(g, xn, hc, y.shape[1]).to(BF))


def _quantize_w8(w, gs):
    n, k = w.shape
    v = w.float().view(n, k // gs, gs)
    sc = v.abs().amax(2).clamp_min(1e-12) / 127.0
    q = torch.round(v / sc[..., None]).clamp_(-127, 127).to(torch.int8).view(n, k)
    return q, sc


def _quantize_w4(w, gs):
    q4 = torch.zeros(w.shape[0], w.shape[1] // 8, dtype=torch.int32)
    FakeW4Ext.weights[q4.data_ptr()] = _bf(w)
    return q4, torch.ones(w.shape[0], w.shape[1] // gs, dtype=BF)


# ---------------------------------------------------------------- module loading
_LIB = None


def _ensure_gemm_op():
    global _LIB
    if _LIB is not None:
        return
    _LIB = torch.library.Library("vllm", "FRAGMENT")
    _LIB.define("rocm_unquantized_gemm_gfx908(Tensor x, Tensor w, Tensor? b) -> Tensor")
    _LIB.impl(
        "rocm_unquantized_gemm_gfx908",
        lambda x, w, b: F.linear(x.float(), w.float()).to(x.dtype),
        "CPU",
    )


def _load(monkeypatch, strided: bool, free: bool = True):
    _ensure_gemm_op()
    log = _Log()

    def stub(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)

    for n in (
        "vllm",
        "vllm.utils",
        "vllm.models",
        "vllm.models.qwen4_exp",
        "vllm.models.qwen4_exp.amd",
        "vllm.models.qwen4_exp.amd.ops",
        "vllm.model_executor",
        "vllm.model_executor.layers",
    ):
        stub(n)
    stub("vllm.logger", init_logger=lambda _n: log)
    stub("vllm.utils.torch_utils", direct_register_custom_op=lambda **kw: None)
    stub("vllm.utils.platform_utils", num_compute_units=lambda: 120)
    stub("vllm.platforms", current_platform=None)
    stub("vllm.platforms.rocm", on_gfx908=lambda: True)
    stub(
        "vllm.models.qwen4_exp.amd.ops.hc",
        hc_silu=lambda x, hc: (lambda t: t * torch.sigmoid(t))(_bf(x) / hc).to(x.dtype),
        hc_gate_mix=lambda xn, g, hc: (
            (
                torch.sigmoid(g.float()).view(g.shape[0], hc, -1)
                * xn.float().view(xn.shape[0], hc, -1)
            ).mean(1)
        ).to(xn.dtype),
    )
    stub(
        "vllm.model_executor.layers.gfx908_w8a16",
        quantize_w8=_quantize_w8,
        quantize_w4=_quantize_w4,
        w4lt_free=lambda: True,
        w4lt_gs=lambda: 32,
        _ext_w4lt=lambda: FakeW4Ext(),
    )
    # CPU tensors have device.index None; the scratch helpers then ask for the GPU.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("VLLM_GFX908_HC_STRIDED_INJECT", "1" if strided else "0")
    # the rc10 image bakes VLLM_GFX908_HC_W8_FREE=0; pin it per test
    monkeypatch.setenv("VLLM_GFX908_HC_W8_FREE", "1" if free else "0")
    spec = importlib.util.spec_from_file_location(f"hcf_{int(strided)}", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ext = lambda: FakeBf16Ext()
    mod._ext_w8 = lambda: FakeW8Ext()
    mod._ext_w8_big = lambda: FakeW8Ext()
    mod._FLAG = True
    mod._SHARD_FLAG = False
    mod._MAX_M = 4
    mod.logger_ = log
    return mod


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    wd = (torch.randn(NDOWN, K, generator=g) * 0.02).to(BF)
    wu = (torch.randn(K, R, generator=g) * 0.02).to(BF)
    return wd, wu


def _xn(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, K, generator=g).to(BF)


def _prep(mod, wd, wu, kind):
    """kind: bf16 (no cache) | w8 | w4 -> the (w_down, w_up) the op gets."""
    wd, wu = wd.clone(), wu.clone()
    mod._W8_CACHE.clear()
    if kind == "w8":
        mod._W8_FLAG, mod._W4_FLAG = True, False
        assert mod.prepare_hc_w8_weight(wd, "down", HC, HD)
        assert mod.prepare_hc_w8_weight(wu, "up", HC, HD)
    elif kind == "w4":
        mod._W8_FLAG, mod._W4_FLAG = False, True
        assert mod.prepare_hc_w8_weight(wd, "down", HC, HD)
        assert mod.prepare_hc_w8_weight(wu, "up", HC, HD)
    else:
        mod._W8_FLAG = mod._W4_FLAG = False
    return wd, wu


# path -> (weights kind, VLLM_GFX908_HC_W8_FREE, M values, expected copies into
# the layout under the flag)
PATHS = {
    "w8_fused": ("w8", True, (1, 2, 3, 4), 0),
    "w8_fused_kept_bf16": ("w8", False, (1, 2, 3, 4), 0),
    "w8_freed_dequant": ("w8", True, (5, 7), 0),
    "w8_kept_bf16_stock": ("w8", False, (5, 7), 0),  # the rc10 serve (FREE=0), M > 4
    "bf16_fused": ("bf16", True, (1, 2, 3, 4), 0),
    "stock": ("bf16", True, (5, 9), 0),
    "w4_chain": (
        "w4",
        True,
        (1, 2, 4),
        1,
    ),  # kernel needs a contiguous inj: one copy kept
}


@pytest.mark.parametrize("path", list(PATHS))
def test_paths_identical_values_and_fake_layout(monkeypatch, path):
    kind, free, ms, copies = PATHS[path]
    wd0, wu0 = _weights()
    off = _load(monkeypatch, strided=False, free=free)
    on = _load(monkeypatch, strided=True, free=free)
    n_layout = {"n": 0}
    real_layout = on._inject_layout

    def counting_layout(*a):
        n_layout["n"] += 1
        return real_layout(*a)

    on._inject_layout = counting_layout
    for m in ms:
        xn = _xn(m)
        wd, wu = _prep(off, wd0, wu0, kind)
        b_off, i_off = off._hc_mix_impl(xn, wd, wu, HC, R, HD)
        wd, wu = _prep(on, wd0, wu0, kind)
        n_layout["n"] = 0
        b_on, i_on = on._hc_mix_impl(xn, wd, wu, HC, R, HD)

        assert torch.equal(b_on, b_off) and torch.equal(i_on, i_off), (path, m)
        assert i_off.is_contiguous()
        fb, fi = on._hc_mix_fake(xn, wd, wu, HC, R, HD)
        assert b_on.shape == fb.shape and b_on.stride() == fb.stride()
        assert i_on.shape == fi.shape == (m, HC)
        assert i_on.stride() == fi.stride() == (NDOWN, 1), (path, m, i_on.stride())
        assert i_on.storage_offset() == fi.storage_offset() == R
        assert i_on.untyped_storage().nbytes() == m * NDOWN * 2
        assert n_layout["n"] == copies, (path, m, n_layout["n"])
        # the off-mode fake is the historical contiguous contract
        _, fi_off = off._hc_mix_fake(xn, wd, wu, HC, R, HD)
        assert fi_off.is_contiguous() and fi_off.storage_offset() == 0


def test_shard_path_is_copied_into_the_layout(monkeypatch):
    on = _load(monkeypatch, strided=True)
    wd, wu = _weights()
    xn = _xn(6)
    inj_c = torch.arange(6 * HC, dtype=torch.float32).view(6, HC).to(BF)
    on.hc_shard_applies = lambda m: True
    on.hc_shard_mix = lambda xn, chain, hidden, hc: (
        torch.zeros(xn.shape[0], hidden, dtype=BF),
        inj_c.clone(),
    )
    _, inj = on._hc_mix_impl(xn, wd, wu, HC, R, HD)
    _, fi = on._hc_mix_fake(xn, wd, wu, HC, R, HD)
    assert torch.equal(inj, inj_c)
    assert inj.stride() == fi.stride() and inj.storage_offset() == fi.storage_offset()


def test_foreign_down_layout_is_normalised(monkeypatch):
    """A producer output that is not exactly the fresh [M, down_rows] buffer (here: a
    view with an offset) is copied, never returned in a layout the fake does not
    promise."""
    on = _load(monkeypatch, strided=True)
    big = torch.randn(3, NDOWN + 8).to(BF)
    down = big[:, 8:]
    inj = on._injection_from_down(down, NDOWN, R, HC)
    assert torch.equal(inj, down[:, R : R + HC])
    assert inj.stride() == (NDOWN, 1) and inj.storage_offset() == R


# ---------------------------------------------------------------- inductor contract
def test_inductor_keeps_the_view(monkeypatch):
    on = _load(monkeypatch, strided=True)
    lib = torch.library.Library("hcstride_test", "DEF")
    seen = []

    def mix_impl(xn: torch.Tensor, w_down: torch.Tensor):
        down = (xn.float() @ w_down.float().t()).to(BF)  # fresh [M, 336]
        block = (xn[:, :HD].float() * 0.5).to(BF)
        return block, on._injection_from_down(down, NDOWN, R, HC)

    def mix_fake(xn, w_down):
        b, i = on._hc_mix_fake(xn, w_down, None, HC, R, HD)
        return b, i

    def combine_impl(res: torch.Tensor, block: torch.Tensor, inj: torch.Tensor):
        assert inj.stride(1) == 1  # the only assertion the real consumers make
        seen.append((inj.stride(), inj.storage_offset()))
        g = 2.0 * torch.sigmoid(inj.float() / HC)  # [M, HC]
        out = (
            res.float().view(res.shape[0], HC, HD)
            + block.float()[:, None] * g[..., None]
        )
        return out.view(res.shape).to(res.dtype)

    lib.define("mix(Tensor xn, Tensor w_down) -> (Tensor, Tensor)")
    lib.impl("mix", mix_impl, "CPU")
    lib._register_fake("mix", mix_fake)
    lib.define("combine(Tensor res, Tensor block, Tensor inj) -> Tensor")
    lib.impl("combine", combine_impl, "CPU")
    lib._register_fake("combine", lambda res, block, inj: res.new_empty(res.shape))

    def f(xn, w_down, res):
        block, inj = torch.ops.hcstride_test.mix(xn, w_down)
        block = block * 2  # something for inductor to fuse between the two ops
        return torch.ops.hcstride_test.combine(res, block, inj)

    wd, _ = _weights()
    cf = torch.compile(f, backend="inductor", fullgraph=True, dynamic=True)
    for m in (2, 3, 5):
        xn, res = _xn(m), _xn(m, seed=2)
        seen.clear()
        ref = f(xn, wd, res)
        eager_seen = list(seen)
        seen.clear()
        out = cf(xn, wd, res)
        assert torch.equal(out, ref), m
        assert seen == eager_seen == [((NDOWN, 1), R)], (m, seen, eager_seen)
