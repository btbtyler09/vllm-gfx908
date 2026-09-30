"""Logic test (host-runnable, no torch/GPU): the eager-only QSA fast paths (dense-short, prefill-tiled)
must refuse spec-verify shaped batches (q_len <= 8) and anything under graph capture; real prefill
shapes keep them.  Loads the gate functions straight from qsa.py source with stubbed globals, because
importing the module needs a live ROCm/Triton stack.  Run: python -m pytest tests/gfx908/test_qsa_eager_gate.py"""
import __future__, ast, enum, os, pathlib
from types import SimpleNamespace
import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "vllm/models/qwen4_exp/amd/qsa.py"
FUNCS = {"_dense_short_enabled", "_prefill_tiled_enabled", "_eager_fast_path_min_qlen", "_stream_capturing"}
METHODS = {"_dense_short_eligible", "_prefill_tiled_eligible"}

class CUDAGraphMode(enum.Enum):
    NONE = 0; PIECEWISE = 1; FULL = 2

def _load(capturing=False, mode=CUDAGraphMode.NONE):
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpQSAAttention":
            keep += [m for m in n.body if isinstance(m, ast.FunctionDef) and m.name in METHODS]
    assert {k.name for k in keep} == FUNCS | METHODS, "gate functions renamed/missing"
    for k in keep: k.decorator_list = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing))
    ns = dict(os=os, torch=torch, CUDAGraphMode=CUDAGraphMode,
              get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), ns)
    return ns

def _self(topk=2048, spec_q=1):
    return SimpleNamespace(indexer=SimpleNamespace(skip_topk=False, token_topk=topk), _spec_query_len=spec_q)
def _md(q, s): return SimpleNamespace(max_query_len=q, max_seq_len=s)

@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VLLM_GFX908_QSA_DENSE_SHORT", "1"); monkeypatch.setenv("VLLM_GFX908_QSA_PREFILL_TILED", "1")
    monkeypatch.delenv("VLLM_GFX908_QSA_EAGER_FAST_MIN_QLEN", raising=False)

@pytest.mark.parametrize("spec_q,q,ok", [
    (1, 1, False), (1, 2, True), (1, 512, True),            # no spec decode: any prefill-shaped batch (>1) qualifies
    (2, 2, False), (2, 3, True),                             # K=1: verify q_len 2 stays on the sparse decode path
    (4, 2, False), (4, 4, False), (4, 5, True),              # K=3: verify q_len <= 4 stays sparse
    (8, 8, False), (8, 9, True)])                            # K=7
def test_gate_derived_from_spec_config(spec_q, q, ok):
    ns = _load()
    assert ns["_prefill_tiled_eligible"](_self(spec_q=spec_q), _md(q, 32768)) is ok
    assert ns["_dense_short_eligible"](_self(spec_q=spec_q), _md(q, 1024)) is ok

def test_capture_refuses_even_prefill_shapes():
    ns = _load(capturing=True)
    assert not ns["_prefill_tiled_eligible"](_self(), _md(512, 32768))
    assert not ns["_dense_short_eligible"](_self(), _md(512, 1024))

def test_graph_runtime_modes_refuse():
    ns = _load(mode=CUDAGraphMode.FULL)
    assert not ns["_prefill_tiled_eligible"](_self(), _md(512, 32768))

def test_env_override_restores_old_behaviour(monkeypatch):
    monkeypatch.setenv("VLLM_GFX908_QSA_EAGER_FAST_MIN_QLEN", "2")
    assert _load()["_prefill_tiled_eligible"](_self(spec_q=2), _md(2, 32768))

def test_missing_attr_defaults_to_no_spec():
    ns = _load(); me = SimpleNamespace(indexer=SimpleNamespace(skip_topk=False, token_topk=2048))
    assert ns["_prefill_tiled_eligible"](me, _md(2, 32768))
