# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 fused hyper-connection projections (M <= 3).

A copy of vLLM's wvSplitK small-M skinny GEMM (csrc/rocm/skinny_gemms.cu) with
two fused epilogues, JIT-built with torch.utils.cpp_extension:

  epi 1: mix_down (+ block inject) GEMV with silu(v / HC) applied to the lora
         columns (< lora_rank) — replaces hc_silu.
  epi 2: mix_up GEMV over a row-permuted weight (row i*HC + s = original row
         s*HD + i) with YTILE = HC so one wave holds the HC stream values of an
         output column; the epilogue writes mean_s sigmoid(g_s) * xn[s*HD + i]
         — replaces hc_gate_mix.

Graph-timed on MI100 (bit-exact vs the stock chain): mix_down+silu 7.2 vs 8.3
us, mix_up+gate_mix 8.3 vs 11.6 us at M=1. HC chain 5 -> 3 launches.

W8 variant (``VLLM_GFX908_HC_W8=1``, default OFF)
------------------------------------------------
The two HC mixes are the largest bf16 weight bucket left at decode: 336x10240
+ 10240x320 bf16 per HC module, replicated on every rank, ~1.3 GB of weight
traffic per decode token per rank.  ``csrc/gfx908_wv_fused_w8.hip`` is the same
kernel with the weight stream switched to int8 + fp32 group scales (the W8A16
GEMV of ``agents/w8a16``, plus its LPR=32 / split-K extensions) and *the same
two epilogues*, so the HC chain keeps its 3 launches.

The weights are quantized once at load time (``process_weights_after_loading``
of the two Linear sub-layers, wrapped by :func:`install_hc_w8_prepare`), the
mix_up weight is permuted *before* quantization so the permutation cache holds
the int8 + scale layout, and — unless ``VLLM_GFX908_HC_W8_FREE=0`` — the bf16
master copies are released (``weight.data`` is rebound to a zero-stride stub
that keeps shape/dtype/device but owns no memory).  ``M > 3`` then
rematerialises each weight into a reusable bf16 scratch with a dequant kernel
and runs the stock chain, exactly like the W8A16 phase-2 path.
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

HC_FUSED_MAX_M = 3
_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_CSRC = os.path.join(_CSRC_DIR, "gfx908_wv_fused.hip")
_CSRC_W8 = [os.path.join(_CSRC_DIR, "gfx908_wv_fused_w8.hip")] + [
    os.path.join(_CSRC_DIR, f"gfx908_wv_fused_w8_nb{i}.hip") for i in (1, 2, 3)
]
_FLAG: bool | None = None
_W8_FLAG: bool | None = None


def _build_dir() -> str:
    d = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(d, exist_ok=True)
    return d


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    build_dir = _build_dir()
    logger.info_once("gfx908: building/loading fused HC extension in %s", build_dir)
    return load(
        name="gfx908_wv_fused_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


@functools.cache
def _ext_w8():
    from torch.utils.cpp_extension import load

    build_dir = _build_dir()
    logger.info_once("gfx908: building/loading fused HC W8 extension in %s", build_dir)
    return load(
        name="gfx908_wv_fused_w8_ext",
        sources=_CSRC_W8,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def hc_fused_enabled() -> bool:
    global _FLAG
    if _FLAG is None:
        from vllm.platforms.rocm import on_gfx908

        _FLAG = on_gfx908() and os.environ.get("VLLM_GFX908_HC_FUSED", "1") == "1"
        if _FLAG:
            try:
                _ext()
            except Exception as exc:
                logger.warning_once("gfx908: fused HC extension unavailable (%s)", exc)
                _FLAG = False
    return _FLAG


def hc_w8_enabled() -> bool:
    """int8 weights for the two HC mixes.  Requires the fused path (the freed
    weights are only ever read by the fused custom op)."""
    global _W8_FLAG
    if _W8_FLAG is None:
        if os.environ.get("VLLM_GFX908_HC_W8", "0") != "1":
            _W8_FLAG = False
            return _W8_FLAG
        _W8_FLAG = hc_fused_enabled()
        if _W8_FLAG:
            try:
                _ext_w8()
            except Exception as exc:
                logger.warning_once("gfx908: HC W8 extension unavailable (%s)", exc)
                _W8_FLAG = False
        else:
            logger.warning_once(
                "gfx908: VLLM_GFX908_HC_W8=1 ignored (the fused HC path is off)"
            )
        if _W8_FLAG:
            logger.info_once(
                "gfx908: HC W8 mixes enabled (gs down=%d up=%d, free_bf16=%s)",
                hc_w8_gs("down"),
                hc_w8_gs("up"),
                hc_w8_free(),
            )
    return _W8_FLAG


@functools.cache
def hc_w8_gs(kind: str) -> int:
    """Scale group size along K; 0 = per-output-channel.

    Defaults follow the outlier study in ``agents/w8a16/REPORT.md`` /
    ``agents/hc_w8/REPORT.md``: per-channel RTN degrades to 3-9% row error on
    weights with input-channel outliers, group scales hold ~1.5-2%.  K=10240
    (mix_down) uses 128, K=320 (mix_up) uses 64.
    """
    env = "VLLM_GFX908_HC_W8_GS_DOWN" if kind == "down" else "VLLM_GFX908_HC_W8_GS_UP"
    gs = int(os.environ.get(env, "128" if kind == "down" else "64"))
    if gs != 0 and (gs < 16 or (gs & (gs - 1)) != 0):
        logger.warning("gfx908 HC W8: %s=%d is not 0 or a power of two >= 16; ignored", env, gs)
        gs = 128 if kind == "down" else 64
    return gs


@functools.cache
def hc_w8_free() -> bool:
    """Release the bf16 master copies once the int8 copy exists (default on).

    Set ``VLLM_GFX908_HC_W8_FREE=0`` to keep them: that costs ~650 MB per rank
    but leaves ``M > 3`` (prefill and batched decode) on the untouched bf16
    path instead of paying a full weight rematerialisation per call.
    """
    return os.environ.get("VLLM_GFX908_HC_W8_FREE", "1") == "1"


@functools.cache
def _cu_count() -> int:
    from vllm.utils.platform_utils import num_compute_units

    return int(num_compute_units())


def permute_up_weight(w_up: torch.Tensor, hc_count: int, hidden: int) -> torch.Tensor:
    """[HC*HD, R] -> rows reordered so row i*HC + s = original row s*HD + i."""
    perm = torch.arange(hc_count * hidden, device=w_up.device).view(hc_count, hidden).t().reshape(-1)
    return w_up[perm].contiguous()


_PERM_CACHE: dict[tuple, torch.Tensor] = {}


def _perm_key(w: torch.Tensor) -> tuple:
    return (w.data_ptr(), tuple(w.shape), str(w.device))


def _w_up_perm(w_up: torch.Tensor, hc_count: int, hidden: int) -> torch.Tensor:
    key = _perm_key(w_up)
    t = _PERM_CACHE.get(key)
    if t is None:
        t = permute_up_weight(w_up, hc_count, hidden)
        _PERM_CACHE[key] = t
    return t


# ---------------------------------------------------------------------------
# W8: load-time quantization, bf16 release, M > 3 scratch
# ---------------------------------------------------------------------------


class _W8Entry:
    """int8 side copy of one HC mix weight.

    For ``kind == "up"`` the *permuted* weight is quantized, so the permutation
    cache of the bf16 path is replaced by this entry (nothing permutes at run
    time any more).  ``owner`` pins the tensor whose ``data_ptr`` is the cache
    key when the bf16 copy was kept; for a freed weight the stub aliases ``q``,
    so ``q`` is the pin.
    """

    __slots__ = ("owner", "q", "s", "gs", "kind", "freed")

    def __init__(self, owner, q, s, gs, kind, freed):
        self.owner = owner
        self.q = q
        self.s = s
        self.gs = gs
        self.kind = kind
        self.freed = freed


_W8_CACHE: dict[tuple, _W8Entry] = {}
_W8_SCRATCH: dict[int, torch.Tensor] = {}


def _w8_key(w: torch.Tensor) -> tuple:
    dev = w.device
    return (w.data_ptr(), int(w.shape[0]), int(w.shape[1]), dev.type, dev.index)


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _make_stub(q: torch.Tensor, shape) -> torch.Tensor:
    """bf16 view with the weight's shape/dtype/device that owns no storage: it
    aliases the int8 copy with stride (0, 0).  ``data_ptr()`` still uniquely
    identifies the weight (and is pinned by ``q``); nothing ever reads it."""
    return q.view(torch.bfloat16).as_strided(tuple(shape), (0, 0))


def _ensure_scratch(device: torch.device, elems: int) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cur = _W8_SCRATCH.get(idx)
    if cur is None or cur.numel() < elems:
        if cur is not None:
            del _W8_SCRATCH[idx]
            del cur
        _W8_SCRATCH[idx] = torch.empty(elems, dtype=torch.bfloat16, device=device)
        logger.info_once(
            "gfx908 HC W8: dequantization scratch %.1f MB on cuda:%d", elems * 2 / 2**20, idx
        )
    return _W8_SCRATCH[idx]


def prepare_hc_w8_weight(
    weight: torch.Tensor, kind: str, hc_count: int, hidden: int
) -> bool:
    """Quantize one HC mix weight (permuting first for ``kind == "up"``) and,
    unless ``VLLM_GFX908_HC_W8_FREE=0``, release its bf16 storage.

    Must run before torch.compile / cudagraph capture.
    """
    if not hc_w8_enabled():
        return False
    if weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.is_meta:
        return False
    if _w8_key(weight) in _W8_CACHE:
        return True
    from vllm.model_executor.layers.gfx908_w8a16 import quantize_w8

    gs = hc_w8_gs(kind)
    src = weight if kind == "down" else permute_up_weight(weight, hc_count, hidden)
    if src.shape[1] % 16 or (gs and src.shape[1] % gs):
        logger.warning_once(
            "gfx908 HC W8: skipping %s weight %s (K not a multiple of 16/gs)", kind, tuple(src.shape)
        )
        return False
    q, s = quantize_w8(src.contiguous(), gs)
    del src
    if hc_w8_free():
        stub = _make_stub(q, weight.shape)
        weight.data = stub  # releases the bf16 storage
        ent = _W8Entry(None, q, s, gs, kind, True)
    else:
        ent = _W8Entry(weight, q, s, gs, kind, False)
    _W8_CACHE[_w8_key(weight)] = ent
    _ensure_scratch(q.device, q.shape[0] * q.shape[1])
    logger.info_once(
        "gfx908 HC W8: quantized %s weight %dx%d (gs=%d, freed=%s); %d cached",
        kind,
        q.shape[0],
        q.shape[1],
        gs,
        ent.freed,
        len(_W8_CACHE),
    )
    return True


def install_hc_w8_prepare(gated_residual) -> None:
    """Wrap the two HC Linear sub-layers' ``process_weights_after_loading`` so
    the int8 copies are built (and the bf16 released) at load time.

    ``UnquantizedLinearMethod`` is instantiated per layer (``LinearBase``), so
    replacing the bound method on ``layer.quant_method`` affects only that one
    layer.  The model loader calls it once, before torch.compile and before any
    cudagraph capture.
    """
    hc, hidden = gated_residual.hc_count, gated_residual.hidden_size
    pairs = (
        (gated_residual.input_mix_weight_down_block_inject, "down"),
        (gated_residual.input_mix_weight_up, "up"),
    )
    for layer, kind in pairs:
        qm = getattr(layer, "quant_method", None)
        if qm is None or getattr(qm, "_gfx908_hc_w8_wrapped", False):
            continue
        orig = qm.process_weights_after_loading

        def wrapped(mod, _orig=orig, _kind=kind):
            _orig(mod)
            try:
                prepare_hc_w8_weight(mod.weight, _kind, hc, hidden)
            except Exception as exc:  # never break model loading over this
                logger.warning_once("gfx908 HC W8: prepare failed (%s)", exc)

        qm.process_weights_after_loading = wrapped
        qm._gfx908_hc_w8_wrapped = True


# (YTILE, UNRL, LPR, KS) per M, from the sweeps in agents/hc_w8/REPORT.md.
# mix_down (336 x 10240): few rows, long K -> intra-WG split-K so all 16 waves stream.
# mix_up (10240 x 320): YTILE must be HC=4 for the gate-mix epilogue; K=320 -> two rows
# per wave-step (LPR=32) keeps 62% of the lanes busy instead of 31%.
_CFG_DOWN = {1: (2, 2, 64, 2), 2: (2, 2, 64, 2), 3: (1, 3, 64, 4)}
_CFG_UP = {1: (4, 1, 32, 1), 2: (4, 1, 32, 1), 3: (4, 1, 32, 1)}
_CFG_DOWN_DEFAULT = (2, 2, 64, 2)
_CFG_UP_DEFAULT = (4, 1, 32, 1)


def _w8_dequant_into(ent: _W8Entry, out: torch.Tensor) -> None:
    if not _ext_w8().hc_w8_dequant(ent.q, ent.s, out, ent.gs):
        r, k = ent.q.shape
        if ent.gs == 0:
            out.copy_((ent.q.float() * ent.s[:, None]).to(torch.bfloat16))
        else:
            out.copy_(
                (ent.q.float().view(r, k // ent.gs, ent.gs) * ent.s[..., None])
                .to(torch.bfloat16)
                .view(r, k)
            )


def _w8_scratch_view(ent: _W8Entry) -> torch.Tensor:
    n, k = ent.q.shape
    dev = ent.q.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    sc = _W8_SCRATCH.get(idx)
    if sc is None or sc.numel() < n * k:
        if _capturing():
            raise RuntimeError("gfx908 HC W8: dequant scratch missing during capture")
        sc = _ensure_scratch(dev, n * k)
    return sc[: n * k].view(n, k)


def _gate_mix_perm(xn: torch.Tensor, gate_perm: torch.Tensor, hc: int, hidden: int):
    """hc_gate_mix on a *permuted* gate ([M, i*HC + s] instead of [M, s*HD + i])."""
    m = gate_perm.shape[0]
    g = gate_perm.view(m, hidden, hc).float()
    x = xn.view(m, hc, hidden).transpose(1, 2).float()
    return (torch.sigmoid(g) * x).mean(-1).to(xn.dtype)


def _hc_mix_impl(
    xn: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor,
    hc_count: int, lora_rank: int, hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque op: fused wvSplitK epilogues for M <= 3, stock math otherwise."""
    M = xn.shape[0]
    ed = eu = None
    if _W8_CACHE:
        ed = _W8_CACHE.get(_w8_key(w_down))
        eu = _W8_CACHE.get(_w8_key(w_up))
        if ed is None or eu is None:
            ed = eu = None

    if ed is not None and M <= HC_FUSED_MAX_M and xn.is_contiguous():
        ext = _ext_w8()
        cu = _cu_count()
        yt, un, lpr, ks = _CFG_DOWN.get(M, _CFG_DOWN_DEFAULT)
        down = torch.empty((M, ed.q.shape[0]), dtype=xn.dtype, device=xn.device)
        ok = ext.hc_w8_gemv(ed.q, ed.s, xn, down, xn, 1, ed.gs, yt, un, lpr, ks,
                            hc_count, lora_rank, cu)
        if ok:
            # the kernel takes X's row stride, so the lora block is passed as a view
            # (no extra copy launch: the chain stays at the bf16 path's launch count)
            lora = down[:, :lora_rank]
            injection = down[:, lora_rank : lora_rank + hc_count].contiguous()
            yt, un, lpr, ks = _CFG_UP.get(M, _CFG_UP_DEFAULT)
            y = torch.empty((M, hidden), dtype=xn.dtype, device=xn.device)
            if ext.hc_w8_gemv(eu.q, eu.s, lora, y, xn, 2, eu.gs, yt, un, lpr, ks,
                              hc_count, lora_rank, cu):
                return y, injection
        del down

    if ed is not None and ed.freed:
        # No bf16 master copy: rematerialise into the scratch and run the stock chain.
        from vllm.models.qwen4_exp.amd.ops.hc import hc_silu

        wd = _w8_scratch_view(ed)
        _w8_dequant_into(ed, wd)
        down = torch.ops.vllm.rocm_unquantized_gemm_gfx908(xn, wd, None)
        lora = hc_silu(down[:, :lora_rank].contiguous(), hc_count)
        injection = down[:, lora_rank : lora_rank + hc_count].contiguous()
        wu = _w8_scratch_view(eu)
        _w8_dequant_into(eu, wu)
        gate_perm = torch.ops.vllm.rocm_unquantized_gemm_gfx908(lora, wu, None)
        return _gate_mix_perm(xn, gate_perm, hc_count, hidden), injection

    if M <= HC_FUSED_MAX_M and xn.is_contiguous():
        ext = _ext()
        cu = _cu_count()
        w_up_perm = _w_up_perm(w_up, hc_count, hidden)
        down = torch.empty((M, w_down.shape[0]), dtype=xn.dtype, device=xn.device)
        ext.wv_fused(w_down, xn, down, xn, 1, hc_count, lora_rank, cu)
        lora = down[:, :lora_rank]
        injection = down[:, lora_rank : lora_rank + hc_count].contiguous()
        y = torch.empty((M, hidden), dtype=xn.dtype, device=xn.device)
        ext.wv_fused(w_up_perm, lora, y, xn, 2, hc_count, lora_rank, cu)
        return y, injection
    # stock chain (same dispatch as the ReplicatedLinear layers would take)
    from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu

    down = torch.ops.vllm.rocm_unquantized_gemm_gfx908(xn, w_down, None)
    lora = hc_silu(down[:, :lora_rank].contiguous(), hc_count)
    injection = down[:, lora_rank : lora_rank + hc_count].contiguous()
    gate = torch.ops.vllm.rocm_unquantized_gemm_gfx908(lora, w_up, None)
    return hc_gate_mix(xn, gate, hc_count), injection


def _hc_mix_fake(xn, w_down, w_up, hc_count, lora_rank, hidden):
    return (
        xn.new_empty((xn.shape[0], hidden)),
        xn.new_empty((xn.shape[0], hc_count)),
    )


direct_register_custom_op(
    op_name="gfx908_hc_fused_mix",
    op_func=_hc_mix_impl,
    fake_impl=_hc_mix_fake,
)


def hc_fused_mix(xn, w_down, w_up, hc_count, lora_rank, hidden):
    return torch.ops.vllm.gfx908_hc_fused_mix(xn, w_down, w_up, hc_count, lora_rank, hidden)
