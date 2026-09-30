"""gfx908 W8 prep: no cache entry may outlive the module it belongs to.

Simulates a target load, a draft (MTP) load and load_eagle_model's
embed/lm_head sharing with stub modules, running the REAL code loaded from
source with stubbed imports:

* ``gfx908_w8a16.py`` (the cache, quantize, prep, release, summary), whole file;
* ``UnquantizedEmbeddingMethod.process_weights_after_loading``;
* the loader's ``process_weights_after_loading`` (model-level head pass);
* ``load_eagle_model`` and its helpers from ``spec_decode/eagle/utils.py``.

Importing vllm on a CPU host fails (aiter needs a GPU arch), hence the stubs.
Needs torch: skipped on a bare host, run it in a CPU-only container.

Run: python -m pytest --noconftest tests/gfx908/test_w8_prep_leak.py
"""

import __future__

import ast
import contextlib
import gc
import importlib.util
import logging
import pathlib
import sys
import types
import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
W8_SRC = ROOT / "vllm/model_executor/layers/gfx908_w8a16.py"
VPE_SRC = ROOT / "vllm/model_executor/layers/vocab_parallel_embedding.py"
LOADER_SRC = ROOT / "vllm/model_executor/model_loader/utils.py"
EAGLE_SRC = ROOT / "vllm/v1/worker/gpu/spec_decode/eagle/utils.py"

V, H = 512, 256  # stands in for the (62080, 2560) vocab head
LIN = (128, 256)  # stands in for a whitelisted linear (e.g. GDN out_proj)


class _Log:
    def __init__(self):
        self.records = []
        self._l = logging.getLogger("w8test")

    def _rec(self, level, msg, *a):
        self.records.append((level, msg % a if a else msg))

    def info(self, msg, *a):
        self._rec("info", msg, *a)

    def warning(self, msg, *a):
        self._rec("warning", msg, *a)

    info_once = info
    warning_once = warning

    def debug(self, *a):
        pass


def _extract(src, names, cls=None):
    tree = ast.parse(src.read_text())
    body = tree.body
    if cls is not None:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    keep = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {k.name for k in keep} == set(names), (src, names)
    for k in keep:
        k.decorator_list = []
    return ast.Module(body=keep, type_ignores=[])


def _exec(mod_ast, src, ns):
    exec(
        compile(
            mod_ast,
            str(src),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        ns,
    )
    return ns


@pytest.fixture
def env(monkeypatch, request):
    mfma = getattr(request, "param", True)
    log = _Log()

    def stub(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.compilation",
    ):
        stub(name)
    stub("vllm.logger", init_logger=lambda _n: log)
    platform = SimpleNamespace(is_cpu=lambda: False, is_rocm=lambda: True)
    stub("vllm.platforms", current_platform=platform)
    stub("vllm.platforms.rocm", on_gfx908=lambda: True)
    stub(
        "vllm.compilation.backends",
        set_model_tag=lambda _t: contextlib.nullcontext(),
    )

    for k in ("VLLM_GFX908_W4_LOADTIME", "VLLM_GFX908_W8A16_GS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_GFX908_W8A16", "1")
    monkeypatch.setenv("VLLM_GFX908_W8A16_MFMA", "1" if mfma else "0")

    spec = importlib.util.spec_from_file_location("gfx908_w8a16_t", W8_SRC)
    w8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w8)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.gfx908_w8a16", w8)
    w8._reset_env_cache()
    w8._FLAG = True  # the HIP extension is not built on a CPU host
    w8.W8A16_SHAPES = frozenset({(V, H), LIN})
    w8.W8A16_VOCAB_SHAPES = frozenset({(V, H)})
    w8._MFMA_CFG = {(V, H): {1: (1, 1, 1, 1)}, LIN: {1: (1, 1, 1, 1)}}

    # ---- quant methods / layers -------------------------------------------
    class QuantizeMethodBase:
        pass

    class VocabParallelEmbedding(nn.Module):
        def __init__(self, v=V, h=H):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn(v, h).to(torch.bfloat16), requires_grad=False
            )
            self.quant_method = UnquantizedEmbeddingMethod()

    class ParallelLMHead(VocabParallelEmbedding):
        pass

    vpe_ns = _exec(
        _extract(
            VPE_SRC, {"process_weights_after_loading"}, "UnquantizedEmbeddingMethod"
        ),
        VPE_SRC,
        {
            "current_platform": platform,
            "os": __import__("os"),
            "ParallelLMHead": ParallelLMHead,
            "bind_rocm_unquantized_gemm_gfx908": lambda layer: None,
        },
    )

    class UnquantizedEmbeddingMethod(QuantizeMethodBase):
        process_weights_after_loading = vpe_ns["process_weights_after_loading"]

    class LinearMethod(QuantizeMethodBase):
        # linear.py's gfx908 hook: prepare_w8a16_layer(layer)
        def process_weights_after_loading(self, layer):
            w8.prepare_w8a16_layer(layer)

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn(*LIN).to(torch.bfloat16), requires_grad=False
            )
            self.quant_method = LinearMethod()

    loader_ns = _exec(
        _extract(LOADER_SRC, {"process_weights_after_loading"}),
        LOADER_SRC,
        {
            "nn": nn,
            "torch": torch,
            "maybe_retie_word_embeddings": lambda m, c: None,
            "QuantizeMethodBase": QuantizeMethodBase,
            "device_loading_context": lambda m, d: contextlib.nullcontext(),
            "release_device_memory_under_pressure": lambda d: None,
            "is_deferred_attention_layer": lambda m: False,
            "HpcModule": type("HpcModule", (), {}),
            "set_torchao_reload_attrs": lambda m, c: None,
        },
    )

    def load(model):
        loader_ns["process_weights_after_loading"](
            model,
            SimpleNamespace(quantization=None, dtype=torch.bfloat16),
            torch.device("cpu"),
        )
        return model

    return SimpleNamespace(
        w8=w8,
        log=log,
        VPE=VocabParallelEmbedding,
        Head=ParallelLMHead,
        Linear=Linear,
        load=load,
        eagle_ns=_eagle_ns,
    )


def _eagle_ns(get_model, release=True):
    ns = {
        "torch": torch,
        "nn": nn,
        "replace": lambda *a, **k: a[0],
        "get_pp_group": lambda: SimpleNamespace(world_size=1),
        "BaseLayerWithLoRA": type("BaseLayerWithLoRA", (), {}),
        "get_model": get_model,
    }
    _exec(
        _extract(
            EAGLE_SRC,
            {
                "_should_share",
                "_release_discarded_w8",
                "get_target_lm_head",
                "load_eagle_model",
            },
        ),
        EAGLE_SRC,
        ns,
    )
    if not release:  # the pre-fix behaviour, as a negative control
        ns["_release_discarded_w8"] = lambda *a, **k: None
    return ns


class _Inner(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.embed_tokens = e.VPE()
        self.proj = e.Linear()


class _LM(nn.Module):
    def __init__(self, e, tie=False):
        super().__init__()
        self.model = _Inner(e)
        self.lm_head = self.model.embed_tokens if tie else e.Head()


_VCFG = SimpleNamespace(
    speculative_config=SimpleNamespace(draft_model_config=None, kv_cache_dtype=None)
)


def _run(e, draft_tie=False, target_tie=False, release=True, share_param=False):
    target = e.load(_LM(e, tie=target_tie))
    target_head_ent = e.w8._Q_CACHE.get(e.w8._cache_key(target.lm_head.weight))
    draft_holder = {}

    def get_model(vllm_config, model_config):
        d = _LM(e, tie=draft_tie)
        if share_param:
            d.lm_head.weight = target.lm_head.weight  # tied to the target
        d = e.load(d)
        draft_holder["refs"] = [
            weakref.ref(d.model.embed_tokens.weight),
            weakref.ref(d.lm_head.weight),
        ]
        return d

    draft = e.eagle_ns(get_model, release)["load_eagle_model"](target, _VCFG)
    gc.collect()
    return target, draft, target_head_ent, draft_holder["refs"]


def _live_keys(w8, *models):
    return {
        w8._cache_key(p)
        for m in models
        for p in m.parameters()
        if p.dim() == 2 and w8._cache_key(p) in w8._Q_CACHE
    }


@pytest.mark.parametrize("env", [True, False], indirect=True, ids=["mfma", "rowmajor"])
def test_only_live_head_survives(env):
    w8 = env.w8
    target, draft, head_ent, refs = _run(env)

    # The draft now uses the target's head and embedding objects.
    assert draft.lm_head is target.lm_head
    assert draft.model.embed_tokens is target.model.embed_tokens

    # Exactly: target head + target linear + draft linear.  No embedding table
    # (target's or draft's) and no discarded draft head.
    expected = {
        w8._cache_key(target.lm_head.weight),
        w8._cache_key(target.model.proj.weight),
        w8._cache_key(draft.model.proj.weight),
    }
    assert set(w8._Q_CACHE) == expected
    assert _live_keys(w8, target, draft) == expected
    # The live head's entry is the very object built at target load
    # (numerics untouched by the draft load/release).
    assert w8._Q_CACHE[w8._cache_key(target.lm_head.weight)] is head_ent
    # The discarded draft bf16 Parameters are actually freed.
    assert all(r() is None for r in refs)

    s = w8.log_w8a16_prep_summary([target, draft])
    assert len(s["heads"]) == 1 and s["orphans"] == []
    line = [m for lvl, m in env.log.records if m.startswith("gfx908 W8 prep:")]
    assert line and "1 vocab head (512x256)" in line[-1] and "orphans 0" in line[-1]


def test_negative_control_prefix_behaviour_leaks(env):
    """Without the release in load_eagle_model the draft head entry survives,
    pins its bf16 weight, and the summary flags it as an orphan."""
    w8 = env.w8
    target, draft, _, refs = _run(env, release=False)
    s = w8.log_w8a16_prep_summary([target, draft])
    assert len(s["orphans"]) == 1 and len(s["heads"]) == 2
    assert refs[1]() is not None  # draft lm_head bf16 still pinned by the cache
    assert any(
        lvl == "warning" and "belong to no live module" in m
        for lvl, m in env.log.records
    )


def test_tied_heads(env):
    """lm_head = embed_tokens (plain VocabParallelEmbedding) on both sides: the
    loader's model-level pass preps the tied module as a head; the draft's copy
    is released when the target's embedding/head replace it."""
    w8 = env.w8
    target, draft, head_ent, refs = _run(env, draft_tie=True, target_tie=True)
    assert head_ent is not None  # tied VPE head was prepped at target load
    assert draft.lm_head is target.lm_head
    s = w8.w8a16_prep_summary([target, draft])
    assert len(s["heads"]) == 1 and s["orphans"] == []
    assert all(r() is None for r in refs)


def test_shared_parameter_is_not_released(env):
    """A draft head whose Parameter IS the target's must not drop the target's
    entry."""
    w8 = env.w8
    target, draft, head_ent, _ = _run(env, share_param=True)
    assert w8._Q_CACHE.get(w8._cache_key(target.lm_head.weight)) is head_ent
    assert w8.w8a16_prep_summary([target, draft])["orphans"] == []


def test_plain_embedding_is_never_prepped(env):
    w8 = env.w8
    m = env.load(_Inner(env))
    assert w8._cache_key(m.embed_tokens.weight) not in w8._Q_CACHE
    assert w8._cache_key(m.proj.weight) in w8._Q_CACHE
