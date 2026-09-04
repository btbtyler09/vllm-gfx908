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
epilogues / routing sensitivity).  The bf16 weight is kept, so anything with
M > 4 (prefill, batched decode) is bit-identical to stock; the int8 copy is a
duplicate (~1.4 GB per rank for this model).

The int8 + scale side tensors are built lazily on the first *eager* call for a
given weight and cached by ``weight.data_ptr()``; a lookup that misses while a
cudagraph is being captured falls through to the bf16 path, so capture never
allocates and the captured graph only ever references already-materialised
buffers.
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
# key -> (weight_ref, q int8 [N, K], s fp32 [N] or [N, K/gs])
_Q_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


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


def get_w8a16_weight(weight: torch.Tensor):
    """Return the cached (int8, scales) side tensors for `weight`, or None.

    Builds them on the first eager call.  Never quantizes while a cudagraph is
    being captured (allocation + a sync during capture is illegal / would leak
    into the graph pool) — the caller falls back to bf16 in that case.
    """
    key = _cache_key(weight)
    ent = _Q_CACHE.get(key)
    if ent is not None:
        return ent[1], ent[2]
    if _capturing():
        return None
    gs = w8a16_group_size()
    q, s = quantize_w8(weight, gs)
    # Hold a reference to `weight` so its storage (and therefore its data_ptr,
    # the cache key) cannot be recycled under a different tensor.
    _Q_CACHE[key] = (weight, q, s)
    logger.info_once(
        "gfx908 W8A16: quantized weight %dx%d (gs=%d); %d entries cached",
        weight.shape[0],
        weight.shape[1],
        gs,
        len(_Q_CACHE),
    )
    return q, s


def prepare_w8a16_weight(weight: torch.Tensor) -> bool:
    """Eagerly materialise the int8 side tensors (e.g. from a
    process_weights_after_loading hook).  Returns True when cached."""
    if not w8a16_enabled() or not w8a16_weight_applies(weight):
        return False
    return get_w8a16_weight(weight) is not None


def w8a16_weight_applies(weight: torch.Tensor) -> bool:
    return (
        weight.dim() == 2
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and not weight.is_meta
        and (weight.shape[0], weight.shape[1]) in W8A16_SHAPES
    )


def w8a16_applies(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Cheap shape/dtype gate (does not touch the quantization cache)."""
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


def w8a16_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor | None:
    """int8-weight / bf16-activation GEMV.  Returns None to fall back to bf16."""
    if not w8a16_applies(x, weight, bias):
        return None
    qs = get_w8a16_weight(weight)
    if qs is None:
        return None
    q, s = qs
    n, k = weight.shape
    x2 = x.reshape(-1, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    m = x2.shape[0]
    ytile, unrl, lpr, ks = _cfg_for(n, k, m)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ok = _ext().w8a16_gemv(
        q, s, x2, out, w8a16_group_size(), ytile, unrl, lpr, ks, 0, _cu_count()
    )
    if not ok:
        return None
    return out.reshape(*x.shape[:-1], n)
