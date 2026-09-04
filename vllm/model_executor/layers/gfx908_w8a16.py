# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 W8A16 skinny GEMV for unquantized (bf16) decode projections.

Opt-in via ``VLLM_GFX908_W8A16=1`` (default OFF).

At decode (M <= 4) the bf16 ``wvSplitK`` skinny GEMM is HBM-bound on the weight
stream.  This path keeps the *activations* in bf16 but streams an int8 copy of
the weight with symmetric group scales, halving the bytes.  Graph-timed on
MI100 (``agents/w8a16/REPORT.md``, L2-cold, M=1): lm_head 356 -> 181 us
(1.97x), GDN in_proj_qkvz 23.7 -> 13.6 (1.74x), GDN out_proj 10.7 -> 6.8
(1.58x).

Phase 1 scope: only the three whitelisted shapes below (GDN ``in_proj_qkvz``
and ``out_proj`` of the 36 GDN layers, plus ``lm_head``).  The router, the
indexer and the hyper-connection mixes are deliberately excluded (fused
epilogues / routing sensitivity).

Phase 2 (this file): the int8 copy can be made the **only** resident copy.
``prepare_w8a16_layer`` (called from ``process_weights_after_loading``, i.e.
before torch.compile and before any cudagraph capture) quantizes the weight,
releases the bf16 storage and rebinds ``layer.weight.data`` to a zero-stride
stub that keeps the shape/dtype/device (so vLLM's shape-derived logic and the
custom op's fake impl still work) but owns no memory of its own.  ``M > 4``
(prefill / batched decode) is then served by rematerialising the weight into a
small reusable bf16 scratch with the ``w8a16_dequant`` HIP kernel, in N-row
chunks, and running the stock bf16 GEMM dispatch per chunk.

Which weights lose their bf16 master copy is controlled by
``VLLM_GFX908_W8A16_FREE`` (``gdn`` | ``all`` | ``none``, default ``gdn``):
``lm_head`` is kept in bf16 by default because it is on the *batched decode*
path (logits for every sequence in the step), where a full rematerialisation
costs more than the int8 duplicate saves.

Numerics: with the bf16 copy released, ``M > 4`` no longer matches stock
bit-for-bit — it computes with the dequantized int8 weight, exactly the
approximation the ``M <= 4`` GEMV path has always used.

The int8 + scale side tensors are cached by ``weight.data_ptr()``.  When
``prepare_w8a16_layer`` has not run, they are still built lazily on the first
*eager* call (and the bf16 copy is then never released, since swapping a
parameter's storage after compile would invalidate dynamo guards and captured
graph addresses).  A lookup that misses while a cudagraph is being captured
falls through to the bf16 path, so capture never allocates.
"""

import functools
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SOURCES = [
    os.path.join(_CSRC_DIR, f)
    for f in (
        "gfx908_w8a16.hip",
        "gfx908_w8a16_dequant.hip",
        "gfx908_w8a16_nb1.hip",
        "gfx908_w8a16_nb2.hip",
        "gfx908_w8a16_nb3.hip",
        "gfx908_w8a16_nb4.hip",
    )
]

# Max batch (rows of x) the kernel is instantiated for.
W8A16_MAX_M = 4

# (N, K) of the weights this phase is allowed to touch.
#   (4096, 2560)  GDN in_proj_qkvz (per rank)
#   (2560, 1536)  GDN out_proj
#   (62080, 2560) lm_head
W8A16_SHAPES = frozenset({(4096, 2560), (2560, 1536), (62080, 2560)})

# Shapes whose bf16 master copy may be released under VLLM_GFX908_W8A16_FREE=gdn.
# lm_head (62080, 2560) is excluded: it runs at M = #sequences every decode step,
# so rematerialising its 318 MB per step costs far more than the 164 MB the int8
# copy duplicates.  ``all`` adds it anyway (N-chunked dequant).
W8A16_FREE_SHAPES_GDN = frozenset({(4096, 2560), (2560, 1536)})

# (YTILE, UNRL, LPR, KS) per shape and M, from the microbench sweeps
# (agents/w8a16/REPORT.md "Recommended dispatch").
_CFG: dict[tuple[int, int], dict[int, tuple[int, int, int, int]]] = {
    (4096, 2560): {1: (2, 1, 32, 1), 2: (4, 1, 32, 1), 3: (4, 1, 32, 1), 4: (4, 1, 32, 2)},
    (2560, 1536): {1: (2, 1, 32, 1), 2: (2, 1, 32, 1), 3: (2, 1, 32, 1), 4: (2, 1, 32, 1)},
    (62080, 2560): {1: (8, 1, 32, 1), 2: (4, 1, 32, 1), 3: (4, 1, 32, 1), 4: (4, 1, 32, 1)},
}
# Fallback config for a whitelisted shape with no per-M entry.
_CFG_DEFAULT = (2, 1, 32, 1)

_FLAG: bool | None = None
_GS: int | None = None
_FREE: str | None = None
_SCRATCH_ELEMS: int | None = None


class _QEntry:
    """int8 side copy of one whitelisted weight.

    ``owner`` pins the tensor the cache key (its ``data_ptr``) belongs to so the
    storage cannot be recycled underneath us.  For a *freed* weight the key is
    the stub's ``data_ptr``, which aliases ``q``'s storage, so ``q`` itself is
    the pin and ``owner`` is None.
    """

    __slots__ = ("owner", "q", "s", "gs", "freed")

    def __init__(self, owner, q, s, gs, freed):
        self.owner = owner
        self.q = q
        self.s = s
        self.gs = gs
        self.freed = freed


# key (data_ptr, N, K, dev.type, dev.index) -> _QEntry
_Q_CACHE: dict[tuple, _QEntry] = {}
# device index -> reusable bf16 dequantization scratch (flat)
_SCRATCH: dict[int, torch.Tensor] = {}


@functools.cache
def _ext():
    """JIT-build (or load from the cache dir) the HIP W8A16 GEMV extension."""
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP W8A16 GEMV extension in %s", build_dir)
    return load(
        name="gfx908_w8a16_ext",
        sources=_SOURCES,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def w8a16_group_size() -> int:
    """Scale group size along K; 0 = per-output-channel scales."""
    global _GS
    if _GS is None:
        gs = int(os.environ.get("VLLM_GFX908_W8A16_GS", "128"))
        if gs != 0 and (gs < 16 or (gs & (gs - 1)) != 0):
            logger.warning(
                "gfx908 W8A16: VLLM_GFX908_W8A16_GS=%d is not 0 or a power of two "
                ">= 16; using 128",
                gs,
            )
            gs = 128
        _GS = gs
    return _GS


def w8a16_enabled() -> bool:
    """True when the opt-in env flag is set and the extension actually builds."""
    global _FLAG
    if _FLAG is None:
        if os.environ.get("VLLM_GFX908_W8A16", "0") != "1":
            _FLAG = False
            return _FLAG
        try:
            from vllm.platforms.rocm import on_gfx908

            _FLAG = bool(on_gfx908())
        except Exception:
            _FLAG = False
        if _FLAG:
            try:
                _ext()
            except Exception as exc:  # hipcc missing etc. -> stock bf16 path
                logger.warning_once("gfx908: W8A16 extension unavailable (%s)", exc)
                _FLAG = False
        if _FLAG:
            logger.info_once(
                "gfx908: W8A16 decode GEMV enabled (group_size=%d, shapes=%s)",
                w8a16_group_size(),
                ", ".join(f"{n}x{k}" for n, k in sorted(W8A16_SHAPES)),
            )
    return _FLAG


@functools.cache
def _cu_count() -> int:
    from vllm.utils.platform_utils import num_compute_units

    return int(num_compute_units())


def quantize_w8(weight: torch.Tensor, gs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric absmax RTN int8 quantization of a [N, K] bf16/fp16 weight.

    gs == 0  -> per-output-channel scales, fp32 [N]
    gs  > 0  -> group scales along K,      fp32 [N, K // gs]

    Chunked over rows so the fp32 temporary stays small (lm_head would be 636 MB
    in one shot).
    """
    n, k = weight.shape
    assert gs == 0 or (gs >= 16 and (gs & (gs - 1)) == 0 and k % gs == 0)
    q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    s = torch.empty(
        (n,) if gs == 0 else (n, k // gs), dtype=torch.float32, device=weight.device
    )
    step = max(1, (1 << 22) // max(k, 1))  # ~4M elements per chunk
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        wf = weight[i0:i1].float()
        if gs == 0:
            sc = wf.abs().amax(1).clamp_min(1e-12) / 127.0
            q[i0:i1] = torch.round(wf / sc[:, None]).clamp_(-127, 127).to(torch.int8)
        else:
            v = wf.view(i1 - i0, k // gs, gs)
            sc = v.abs().amax(2).clamp_min(1e-12) / 127.0
            q[i0:i1] = (
                torch.round(v / sc[..., None])
                .clamp_(-127, 127)
                .to(torch.int8)
                .view(i1 - i0, k)
            )
        s[i0:i1] = sc
    return q, s


def _cache_key(weight: torch.Tensor) -> tuple:
    dev = weight.device
    return (weight.data_ptr(), weight.shape[0], weight.shape[1], dev.type, dev.index)


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# bf16 master-copy release
# ---------------------------------------------------------------------------


def w8a16_free_policy() -> str:
    """Which whitelisted shapes lose their bf16 master copy: gdn | all | none."""
    global _FREE
    if _FREE is None:
        v = os.environ.get("VLLM_GFX908_W8A16_FREE", "gdn").strip().lower()
        if v not in ("gdn", "all", "none"):
            logger.warning(
                "gfx908 W8A16: VLLM_GFX908_W8A16_FREE=%r is not gdn/all/none; using gdn",
                v,
            )
            v = "gdn"
        _FREE = v
    return _FREE


def _should_free(shape: tuple[int, int]) -> bool:
    pol = w8a16_free_policy()
    if pol == "none":
        return False
    if pol == "all":
        return shape in W8A16_SHAPES
    return shape in W8A16_FREE_SHAPES_GDN


def _scratch_elems_cap() -> int:
    """Upper bound (in bf16 elements) on the persistent dequantization scratch."""
    global _SCRATCH_ELEMS
    if _SCRATCH_ELEMS is None:
        mb = float(os.environ.get("VLLM_GFX908_W8A16_SCRATCH_MB", "32"))
        _SCRATCH_ELEMS = max(1 << 16, int(mb * (1 << 20)) // 2)
    return _SCRATCH_ELEMS


def _ensure_scratch(device: torch.device, elems: int) -> torch.Tensor:
    """Reusable flat bf16 scratch, grown on demand.  Allocated eagerly at
    prepare time so no allocation happens under cudagraph capture."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cur = _SCRATCH.get(idx)
    if cur is None or cur.numel() < elems:
        if cur is not None:
            del _SCRATCH[idx]
            del cur
        _SCRATCH[idx] = torch.empty(elems, dtype=torch.bfloat16, device=device)
        logger.info_once(
            "gfx908 W8A16: dequantization scratch %.1f MB on cuda:%d",
            elems * 2 / 2**20,
            idx,
        )
    return _SCRATCH[idx]


def _make_stub(q: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """A bf16 tensor with the weight's shape/dtype/device that owns no storage.

    It aliases the int8 copy's first bytes with stride (0, 0): ``.shape``,
    ``.dtype``, ``.device`` and ``.data_ptr()`` are all well defined (the last
    one uniquely identifies the weight and stays pinned by ``q``), while the
    N*K bf16 elements it claims cost nothing.  Nothing ever reads through it:
    every GEMM for a freed weight is served from ``q`` + scales.
    """
    return q.view(torch.bfloat16).as_strided(tuple(shape), (0, 0))


def is_w8a16_freed(weight: torch.Tensor) -> bool:
    """True if `weight` is a stub whose bf16 storage has been released."""
    if weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.is_meta:
        return False
    ent = _Q_CACHE.get(_cache_key(weight))
    return ent is not None and ent.freed


def _quantize_and_cache(weight: torch.Tensor, allow_free: bool) -> _QEntry:
    gs = w8a16_group_size()
    q, s = quantize_w8(weight, gs)
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if allow_free and _should_free((n, k)):
        stub = _make_stub(q, (n, k))
        # Releases the bf16 storage: the parameter's TensorImpl now points at
        # the stub, and nothing else holds a reference to the old data.
        weight.data = stub
        ent = _QEntry(None, q, s, gs, True)
        _Q_CACHE[_cache_key(weight)] = ent
        _ensure_scratch(q.device, min(_scratch_elems_cap(), n * k))
        logger.info_once(
            "gfx908 W8A16: released bf16 master copy for %dx%d weights "
            "(policy=%s); int8 is now the only resident copy",
            n,
            k,
            w8a16_free_policy(),
        )
    else:
        ent = _QEntry(weight, q, s, gs, False)
        _Q_CACHE[_cache_key(weight)] = ent
    logger.info_once(
        "gfx908 W8A16: quantized weight %dx%d (gs=%d); %d entries cached",
        n,
        k,
        gs,
        len(_Q_CACHE),
    )
    return ent


def get_w8a16_weight(weight: torch.Tensor):
    """Return the cached (int8, scales) side tensors for `weight`, or None.

    Builds them on the first eager call.  Never quantizes while a cudagraph is
    being captured (allocation + a sync during capture is illegal / would leak
    into the graph pool) - the caller falls back to bf16 in that case.  The
    lazy path never releases the bf16 copy (see module docstring).
    """
    ent = _Q_CACHE.get(_cache_key(weight))
    if ent is not None:
        return ent.q, ent.s
    if _capturing():
        return None
    ent = _quantize_and_cache(weight, allow_free=False)
    return ent.q, ent.s


def prepare_w8a16_weight(weight: torch.Tensor) -> bool:
    """Eagerly materialise the int8 side tensors and, when the free policy says
    so, release the bf16 master copy in place (``weight.data`` is rebound).

    Must run before torch.compile / cudagraph capture, i.e. from
    ``process_weights_after_loading``.  Returns True when the weight is cached.
    """
    if not w8a16_enabled() or not w8a16_weight_applies(weight):
        return False
    if _cache_key(weight) in _Q_CACHE:
        return True
    _quantize_and_cache(weight, allow_free=True)
    return True


def prepare_w8a16_layer(layer, allow_free: bool = True) -> bool:
    """``process_weights_after_loading`` hook: quantize + (policy permitting)
    free ``layer.weight``.  Safe to call on any layer; a no-op unless the flag
    is on and the weight is whitelisted.

    ``allow_free=False`` (or ``layer._w8a16_keep_bf16 = True``) keeps the bf16
    master copy regardless of policy - used for layers whose weight may be read
    outside the GEMM dispatch (e.g. a ``ParallelLMHead`` tied to
    ``embed_tokens``, where ``F.embedding`` reads the weight directly).
    """
    w = getattr(layer, "weight", None)
    if w is None or not isinstance(w, torch.Tensor):
        return False
    if not w8a16_enabled() or not w8a16_weight_applies(w):
        return False
    if _cache_key(w) in _Q_CACHE:
        return True
    keep = getattr(layer, "_w8a16_keep_bf16", False)
    try:
        _quantize_and_cache(w, allow_free=allow_free and not keep)
        return True
    except Exception as exc:  # never break model loading over this
        logger.warning_once("gfx908 W8A16: prepare failed for %s (%s)", type(layer), exc)
        return False


def w8a16_weight_applies(weight: torch.Tensor) -> bool:
    if (
        weight.dim() != 2
        or weight.dtype != torch.bfloat16
        or weight.is_meta
        or (weight.shape[0], weight.shape[1]) not in W8A16_SHAPES
    ):
        return False
    # A freed weight is a zero-stride stub, hence not contiguous; it is valid
    # exactly because the cache already holds its int8 copy.
    return weight.is_contiguous() or _cache_key(weight) in _Q_CACHE


def w8a16_applies(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Cheap shape/dtype gate for the M <= 4 GEMV (does not touch the cache)."""
    if bias is not None or x.dtype != torch.bfloat16:
        return False
    if not w8a16_weight_applies(weight):
        return False
    m = x.numel() // x.size(-1)
    return 1 <= m <= W8A16_MAX_M


def _cfg_for(n: int, k: int, m: int) -> tuple[int, int, int, int]:
    per_m = _CFG.get((n, k))
    if per_m is None:
        return _CFG_DEFAULT
    return per_m.get(m, per_m.get(W8A16_MAX_M, _CFG_DEFAULT))


def dequant_w8_into(
    q: torch.Tensor, s: torch.Tensor, out: torch.Tensor, gs: int
) -> bool:
    """out[:] = bf16(q * s), via the HIP kernel (torch fallback if it declines)."""
    if _ext().w8a16_dequant(q, s, out, gs):
        return True
    if gs == 0:
        out.copy_((q.float() * s[:, None]).to(torch.bfloat16))
    else:
        r, k = q.shape
        out.copy_(
            (q.float().view(r, k // gs, gs) * s[..., None]).to(torch.bfloat16).view(r, k)
        )
    return True


def _dequant_gemm(
    x: torch.Tensor,
    ent: _QEntry,
    n: int,
    k: int,
    bias: torch.Tensor | None,
    bf16_fn,
) -> torch.Tensor:
    """M > 4: rematerialise the weight into the scratch (N-row chunks) and run
    the stock bf16 GEMM dispatch per chunk."""
    dev = x.device
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    scratch = _SCRATCH.get(idx)
    want = min(_scratch_elems_cap(), n * k)
    if scratch is None or scratch.numel() < want:
        if _capturing():
            # Cannot grow the scratch under capture; caller falls back.
            raise RuntimeError("w8a16: dequant scratch missing during capture")
        scratch = _ensure_scratch(dev, want)
    fit = scratch.numel() // k
    rows = min(n, max(1, (fit & ~7) or fit))
    if rows >= n:
        w = scratch[: n * k].view(n, k)
        dequant_w8_into(ent.q, ent.s, w, ent.gs)
        return bf16_fn(x, w, bias)
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=dev)
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        w = scratch[: (r1 - r0) * k].view(r1 - r0, k)
        dequant_w8_into(ent.q[r0:r1], ent.s[r0:r1], w, ent.gs)
        out[..., r0:r1] = bf16_fn(x, w, None if bias is None else bias[r0:r1])
    return out


def w8a16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    bf16_fn=None,
) -> torch.Tensor | None:
    """int8-weight / bf16-activation GEMM.  Returns None to fall back to bf16.

    ``M <= 4`` uses the W8A16 GEMV.  ``M > 4`` is only served when the weight's
    bf16 master copy has been released (otherwise stock bf16 is both faster and
    bit-identical), by dequantizing into the scratch and calling ``bf16_fn``
    (default ``F.linear``) per N-chunk.
    """
    if weight.dim() != 2:
        return None
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if (n, k) not in W8A16_SHAPES:
        return None
    m = x.numel() // x.size(-1)
    ent = _Q_CACHE.get(_cache_key(weight))
    if ent is None:
        # Lazy (no load-time hook): only worth it for the decode GEMV.
        if x.dtype != torch.bfloat16 or bias is not None or m < 1 or m > W8A16_MAX_M:
            return None
        if not weight.is_contiguous() or weight.is_meta or _capturing():
            return None
        ent = _quantize_and_cache(weight, allow_free=False)
    elif x.dtype != torch.bfloat16:
        # A freed weight has no bf16 copy to fall back to, so serve it anyway
        # (the stock path would cast x to the weight dtype too); otherwise let
        # the caller take the bit-identical bf16 route.
        if not ent.freed:
            return None
        x = x.to(torch.bfloat16)

    if m <= W8A16_MAX_M and m >= 1 and bias is None:
        x2 = x.reshape(-1, k)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        ytile, unrl, lpr, ks = _cfg_for(n, k, m)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        if _ext().w8a16_gemv(
            ent.q, ent.s, x2, out, ent.gs, ytile, unrl, lpr, ks, 0, _cu_count()
        ):
            return out.reshape(*x.shape[:-1], n)
        del out
        # launcher declined this config -> fall through (dequant if freed)

    if not ent.freed:
        return None
    try:
        return _dequant_gemm(
            x, ent, n, k, bias, bf16_fn or torch.nn.functional.linear
        )
    except RuntimeError:
        # Freed weight and we cannot dequantize: there is no bf16 copy to fall
        # back to, so this must not silently return None.
        raise
