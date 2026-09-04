# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 W4A8 GEMV path for decode-sized M (<= 8): env ``VLLM_GFX908_W4A8=1`` (default off).

Weights stay W4 GS32 (symmetric, zero point 8); activations are quantized per block of 32 to
int8 with an fp32 scale and fp32 scale*sum (llama.cpp Q8_1 style), and the GEMV runs int8 dot4
(``v_dot4_i32_i8``) with all loads in flight, lanes along K, DPP reductions and no split-K
(csrc/gfx908_w4a8.hip; design docs/mi100_decode_opt/research/w4_gemv_kernels.md D1/D2).
Microbench vs the W4A16 HIP GEMV (agents/w4a8/REPORT.md): MoE gate_up 21.7 -> 8.9 us, down
11.7 -> 5.8 us at M=1 (4.8x / 2.1x at M=8); A8 error ~6e-3 relative.

Three consumers, all inside custom ops / eager MoE code where M is concrete:
  * routed experts   (gfx908_moe_hip.gfx908_moe_hip)    -> moe_w4a8
  * shared expert    (gfx908_shared_expert)             -> shared_expert_w4a8
  * dense W4A16 GEMV (triton_w4a16._gfx908_w4a16_gemv)  -> dense_w4a8_gemv  (K in {2560, 1536})
Dense / shared weights are ``[K, N/8]`` int32 K-major (N packed); they are repacked once to the
MoE ``[1, N, K/8]`` K-packed layout and cached by data_ptr (never during graph capture: the
caller falls back to the stock path until the first eager call has populated the cache).
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

W4A8_MAX_TOKENS = 8
GS = 32
_SLAB_K = (2560, 1536)          # K values instantiated for the slab kernel (SLAB_CASE in the .hip)
_ROWLANE_K = 160
_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gfx908_w4a8.hip")
_FLAG: bool | None = None


@functools.cache
def _ext():
    from torch.utils.cpp_extension import load

    base = os.environ.get(
        "VLLM_GFX908_HIP_BUILD_DIR", os.path.expanduser("~/.cache/vllm/gfx908_w4gemv")
    )
    build_dir = os.path.join(base, "w4a8")
    os.makedirs(build_dir, exist_ok=True)
    logger.info_once("gfx908: building/loading HIP W4A8 GEMV extension in %s", build_dir)
    return load(
        name="gfx908_w4a8_ext",
        sources=[_CSRC],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--offload-arch=gfx908"],
        verbose=False,
    )


def w4a8_enabled() -> bool:
    """VLLM_GFX908_W4A8=1 on gfx908 with a working extension build (cached)."""
    global _FLAG
    if _FLAG is None:
        _FLAG = False
        if os.environ.get("VLLM_GFX908_W4A8", "0") == "1":
            try:
                from vllm.platforms.rocm import on_gfx908

                _FLAG = bool(on_gfx908()) and _ext() is not None
            except Exception as exc:  # hipcc missing etc. -> stock paths
                logger.warning_once("gfx908: W4A8 GEMV path unavailable (%s); using W4A16", exc)
                _FLAG = False
            if _FLAG:
                logger.info_once("gfx908: W4A8 GEMV path enabled (VLLM_GFX908_W4A8=1)")
    return _FLAG


# --------------------------------------------------------------------------- #
# thin wrappers
# --------------------------------------------------------------------------- #
def quant_q8(x: torch.Tensor):
    """bf16 [R, K] -> (x8 int8 [R, K] per-8 de-interleaved, xs fp32 [R, K/32], xsum fp32 [R, K/32])."""
    R, K = x.shape
    x8 = torch.empty((R, K), dtype=torch.int8, device=x.device)
    xs = torch.empty((R, K // GS), dtype=torch.float32, device=x.device)
    xsum = torch.empty_like(xs)
    _ext().quant_q8(x, x8, xs, xsum)
    return x8, xs, xsum


def gemv_slab(w, s, x8, xs, xsum, row_tok, row_exp, wpb: int = 4) -> torch.Tensor:
    """D1: w [E, N, K/8] int32 (K-packed nibbles), s [E, N, K/32] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x8.device)
    _ext().gemv_slab(w, s, x8, xs, xsum, row_tok, row_exp, out, 16, wpb)
    return out


def gemv_rowlane(w, s, x8, xs, xsum, row_tok, row_exp, wpb: int = 4) -> torch.Tensor:
    """D2 (K == 160): w [E, N, 20] int32, s [E, N, 5] bf16 -> fp32 [P, N]."""
    out = torch.empty((row_tok.numel(), w.shape[1]), dtype=torch.float32, device=x8.device)
    _ext().gemv_rowlane(w, s, x8, xs, xsum, row_tok, row_exp, out, wpb)
    return out


@triton.jit
def _silu_mul_quant_kernel(
    part_ptr, x8_ptr, xs_ptr, xsum_ptr, N,          # part fp32 [P, N] (gate | up), N = 2 * K2
    stride_pp, stride_x8, stride_xs,
    GS: tl.constexpr,
):
    """silu(g) * u over one block of 32 outputs, then Q8_1 quant into the de-interleaved
    int8 layout the HIP kernels read (per 8: [k0 k2 k4 k6 | k1 k3 k5 k7])."""
    pair = tl.program_id(0)
    g = tl.program_id(1)
    half = N // 2
    offs = g * GS + tl.arange(0, GS)
    base = part_ptr + pair * stride_pp
    gv = tl.load(base + offs)
    uv = tl.load(base + half + offs)
    y = gv * tl.sigmoid(gv) * uv
    amax = tl.max(tl.abs(y), axis=0)
    d = amax / 127.0
    inv = tl.where(amax > 0, 127.0 / amax, 0.0)
    v = y * inv
    q = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    qi = q.to(tl.int32)
    dst = (offs // 8) * 8 + (offs % 2) * 4 + (offs % 8) // 2
    tl.store(x8_ptr + pair * stride_x8 + dst, qi.to(tl.int8))
    tl.store(xs_ptr + pair * stride_xs + g, d)
    tl.store(xsum_ptr + pair * stride_xs + g, d * tl.sum(qi.to(tl.float32), axis=0))


def silu_mul_quant(part: torch.Tensor):
    """fp32 [P, 2*K2] gate|up partial -> int8 [P, K2] + xs/xsum [P, K2/32] (one launch)."""
    P, N = part.shape
    K2 = N // 2
    assert K2 % GS == 0, K2
    x8 = torch.empty((P, K2), dtype=torch.int8, device=part.device)
    xs = torch.empty((P, K2 // GS), dtype=torch.float32, device=part.device)
    xsum = torch.empty_like(xs)
    _silu_mul_quant_kernel[(P, K2 // GS)](
        part, x8, xs, xsum, N, part.stride(0), x8.stride(0), xs.stride(0), GS=GS,
    )
    return x8, xs, xsum


@functools.cache
def _rows(M: int, device: torch.device):
    """(arange(M), zeros(M)) int32: row -> token, row -> expert 0 for single-matrix GEMVs."""
    return (
        torch.arange(M, device=device, dtype=torch.int32),
        torch.zeros(M, device=device, dtype=torch.int32),
    )


# --------------------------------------------------------------------------- #
# dense weight repack: [K, N/8] int32 K-major (N packed, triton_w4a16 layout) + scales [K/G, N]
# -> [1, N, K/8] int32 K-packed (low nibble = lowest k) + scales [1, N, K/G] bf16.
# --------------------------------------------------------------------------- #
# value = (w, s, source): the source weight is kept alive so its data_ptr (the key) cannot be
# recycled by a later allocation of the same shape while the entry exists.
_REPACK_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def repack_dense_to_nk(b_q: torch.Tensor, scales: torch.Tensor):
    K, N8 = b_q.shape
    N = N8 * 8
    shifts = torch.arange(8, device=b_q.device, dtype=torch.int32) * 4
    w_kn = ((b_q.unsqueeze(-1) >> shifts) & 0xF).reshape(K, N)            # [K, N] nibbles
    w_nk = w_kn.t().contiguous().view(N, K // 2, 2).to(torch.uint8)
    packed = (w_nk[:, :, 0] | (w_nk[:, :, 1] << 4)).contiguous()           # uint8 [N, K/2]
    w = packed.view(torch.int32).unsqueeze(0).contiguous()                  # [1, N, K/8]
    s = scales.t().contiguous().to(torch.bfloat16).unsqueeze(0).contiguous()  # [1, N, K/G]
    return w, s


def dense_weight_nk(b_q: torch.Tensor, scales: torch.Tensor):
    """Cached repack keyed on the weight's data_ptr; None while capturing a graph and not yet
    cached (the caller then uses the stock path; the eager warmup call fills the cache)."""
    key = (b_q.data_ptr(), tuple(b_q.shape), str(b_q.device))
    t = _REPACK_CACHE.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        w, s = repack_dense_to_nk(b_q, scales)
        t = (w, s, b_q)
        _REPACK_CACHE[key] = t
    return t[0], t[1]


# --------------------------------------------------------------------------- #
# the three flows
# --------------------------------------------------------------------------- #
def moe_w4a8_applies(K: int, N1: int, group_size: int, dtype: torch.dtype) -> bool:
    return (
        K in _SLAB_K and K % 64 == 0 and N1 % 4 == 0 and N1 // 2 == _ROWLANE_K
        and group_size == GS and dtype == torch.bfloat16
    )


def moe_w4a8(
    output, hidden_states, w1_i, w2_i, w1_scale, w2_scale, topk_weights,
    row_token, row_self, row_expert, mul_routed_weight: bool,
):
    """Routed experts: quant -> slab gate_up [P, N1] -> silu*mul+quant -> rowlane down [P, K]
    -> existing routed-weight sum. Same arguments/layouts as gfx908_moe_hip (w viewed as int32)."""
    from vllm.model_executor.layers.fused_moe.gfx908_moe_hip import _moe_reduce_weighted_sum_kernel

    M, K = hidden_states.shape
    P = row_expert.numel()
    topk = P // M
    x8, xs, xsum = quant_q8(hidden_states)
    part1 = gemv_slab(w1_i, w1_scale, x8, xs, xsum, row_token, row_expert, wpb=4)
    i8, isc, isum = silu_mul_quant(part1)
    part2 = gemv_rowlane(w2_i, w2_scale, i8, isc, isum, row_self, row_expert, wpb=4)
    rb2 = 1024
    _moe_reduce_weighted_sum_kernel[(triton.cdiv(K, rb2), M)](
        part2, topk_weights.reshape(-1).to(torch.float32), output, K,
        0, part2.stride(0), output.stride(0),
        TOPK=topk, SPLIT_K=1, BLOCK=rb2, MUL_W=mul_routed_weight,
    )
    return output


def shared_expert_w4a8(x, wq1, ws1, wq2, ws2, wg):
    """Shared expert (dense [K, N/8] weights): returns bf16 [M, H] or None if not applicable."""
    from vllm.model_executor.layers.gfx908_shared_expert import _reduce_gate_kernel

    M, K = x.shape
    N1 = wq1.shape[1] * 8
    H = wq2.shape[1] * 8
    if (
        M > W4A8_MAX_TOKENS or x.dtype != torch.bfloat16 or K not in _SLAB_K
        or N1 % 4 != 0 or N1 // 2 != _ROWLANE_K or H % 64 != 0 or wq2.shape[0] != _ROWLANE_K
    ):
        return None
    p1 = dense_weight_nk(wq1, ws1)
    p2 = dense_weight_nk(wq2, ws2)
    if p1 is None or p2 is None:
        return None
    rt, re = _rows(M, x.device)
    x8, xs, xsum = quant_q8(x)
    part1 = gemv_slab(p1[0], p1[1], x8, xs, xsum, rt, re, wpb=1)
    i8, isc, isum = silu_mul_quant(part1)
    part2 = gemv_rowlane(p2[0], p2[1], i8, isc, isum, rt, re, wpb=1)
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    rb2 = 1024
    _reduce_gate_kernel[(triton.cdiv(H, rb2), M)](
        part2, x, wg, out, H, K,
        0, part2.stride(0), x.stride(0), out.stride(0),
        SPLIT_K=1, BLOCK=rb2, BLOCK_K=1024,
    )
    return out


def dense_w4a8_gemv(a, b_q, scales):
    """Dense symmetric W4 GEMV, a bf16 [M, K], b_q [K, N/8], scales [K/32, N] -> bf16 [M, N] or None."""
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        triton_w4a16_splitk_reduce_kernel,
    )

    M, K = a.shape
    N = b_q.shape[1] * 8
    if M > W4A8_MAX_TOKENS or a.dtype != torch.bfloat16 or K not in _SLAB_K or N % 4 != 0:
        return None
    p = dense_weight_nk(b_q, scales)
    if p is None:
        return None
    rt, re = _rows(M, a.device)
    x8, xs, xsum = quant_q8(a)
    part = gemv_slab(p[0], p[1], x8, xs, xsum, rt, re, wpb=1 if N <= 512 else 4)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    rb = 1024
    triton_w4a16_splitk_reduce_kernel[(triton.cdiv(N, rb), M)](
        part, c, M, N, 0, part.stride(0), c.stride(0), SPLIT_K=1, BLOCK=rb,
    )
    return c
