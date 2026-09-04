# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 fused shared-expert MLP for small M (Qwen3-Next / Qwen4Exp).

Stock path per layer at decode: gate_up GEMV (2 launches) -> SiluAndMul ->
down GEMV (2) -> expert gate einsum (2 rocBLAS kernels) -> sigmoid -> mul,
~10 launches / ~45 us for ~14 us of GEMV work. This path keeps the two
W4A16 GEMV partial kernels and fuses everything else into their reduces:

  K1 gate_up partials   [SK1, M, 2*I]    (triton_w4a16_gemv_partial_kernel)
  K2 reduce + silu*mul  -> inter [M, I]  bf16
  K3 down partials      [SK2, M, H]
  K4 reduce + sigmoid(x . w_gate) * (.)  -> out [M, H]

Runs as an opaque custom op so M is concrete under torch.compile.
"""

import torch

from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    _gfx908_gemv_config,
    triton_w4a16_gemv_partial_kernel,
)
from vllm.model_executor.layers.fused_moe.gfx908_w4a8 import (
    W4A8_MAX_TOKENS,
    shared_as_expert_enabled,
    shared_defer,
    shared_expert_from_pack,
    shared_pack,
    shared_register,
    w4a8_enabled,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

SHARED_EXPERT_MAX_M = 16


@triton.jit
def _reduce_silu_mul_kernel(
    part_ptr, out_ptr, N, stride_pk, stride_pm, stride_om,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.program_id(1)
    half = N // 2
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half
    g = tl.zeros((BLOCK,), dtype=tl.float32)
    u = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        base = part_ptr + s * stride_pk + m * stride_pm
        g += tl.load(base + offs, mask=mask, other=0.0)
        u += tl.load(base + half + offs, mask=mask, other=0.0)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + m * stride_om + offs, y.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def _reduce_gate_kernel(
    part_ptr, x_ptr, wg_ptr, out_ptr, N, K,
    stride_pk, stride_pm, stride_xm, stride_om,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * stride_pk + m * stride_pm + offs, mask=mask, other=0.0)
    # expert gate: sigmoid(x[m] . w_gate)
    dot = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        xv = tl.load(x_ptr + m * stride_xm + offs_k, mask=mk, other=0.0).to(tl.float32)
        wv = tl.load(wg_ptr + offs_k, mask=mk, other=0.0).to(tl.float32)
        dot += tl.sum(xv * wv, axis=0)
    gate = tl.sigmoid(dot)
    tl.store(out_ptr + m * stride_om + offs, (acc * gate).to(out_ptr.type.element_ty), mask=mask)


def _gemv_partials(a, b_q, scales, group_size, zp_bias):
    M, K = a.shape
    N = b_q.shape[1] * 8
    block_k = min(32, group_size)
    block_m = 1 if M == 1 else 16
    k_tiles = triton.cdiv(K, block_k)
    block_n, split_k = _gfx908_gemv_config(M, K, N, k_tiles)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    triton_w4a16_gemv_partial_kernel[(triton.cdiv(N, block_n), split_k)](
        a, b_q, scales, part, M, N, K,
        a.stride(0), b_q.stride(0), part.stride(0), part.stride(1),
        group_size, ZP_BIAS=zp_bias, BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_K=block_k, SPLIT_K=split_k,
    )
    return part, N


def _shared_expert_forward(
    x: torch.Tensor,
    wq1: torch.Tensor, ws1: torch.Tensor,
    wq2: torch.Tensor, ws2: torch.Tensor,
    wg: torch.Tensor,
    group_size: int, zp_bias: int,
) -> torch.Tensor:
    M, K = x.shape
    if (
        w4a8_enabled() and group_size == 32 and zp_bias == 8
        and M <= W4A8_MAX_TOKENS and x.dtype == torch.bfloat16
    ):
        # VLLM_GFX908_W4A8=1: int8/fp16-activation GEMVs (gfx908_w4a8.py); None -> stock path
        pack = shared_pack(wq1, ws1, wq2, ws2, wg)
        if pack is not None:
            if shared_as_expert_enabled():
                # VLLM_GFX908_SHARED_AS_EXPERT=1: hand the shared expert to the routed W4 GEMVs
                # (which run right after this call) and return the zero stand-in.  `shared_defer`
                # returns None until the routed MoE has confirmed it can consume the hand-off.
                shared_register(pack)
                z = shared_defer(x, pack)
                if z is not None:
                    return z
            return shared_expert_from_pack(x, pack)
    part1, n1 = _gemv_partials(x, wq1, ws1, group_size, zp_bias)
    inter = torch.empty((M, n1 // 2), dtype=x.dtype, device=x.device)
    rb = 256
    _reduce_silu_mul_kernel[(triton.cdiv(n1 // 2, rb), M)](
        part1, inter, n1, part1.stride(0), part1.stride(1), inter.stride(0),
        SPLIT_K=part1.shape[0], BLOCK=rb,
    )
    part2, n2 = _gemv_partials(inter, wq2, ws2, group_size, zp_bias)
    out = torch.empty((M, n2), dtype=x.dtype, device=x.device)
    rb2 = 1024
    _reduce_gate_kernel[(triton.cdiv(n2, rb2), M)](
        part2, x, wg, out, n2, K,
        part2.stride(0), part2.stride(1), x.stride(0), out.stride(0),
        SPLIT_K=part2.shape[0], BLOCK=rb2, BLOCK_K=1024,
    )
    return out


def _shared_expert_forward_fake(x, wq1, ws1, wq2, ws2, wg, group_size, zp_bias):
    return x.new_empty((x.shape[0], wq2.shape[1] * 8))


direct_register_custom_op(
    op_name="gfx908_shared_expert",
    op_func=_shared_expert_forward,
    fake_impl=_shared_expert_forward_fake,
)


def gfx908_shared_expert_applies(mlp: torch.nn.Module, x: torch.Tensor) -> bool:
    """True when both projections are symmetric TritonW4A16 GPTQ, the down
    projection does not reduce (FusedMoE reduces the sum), and M is small."""
    if x.dim() != 2 or x.shape[0] > SHARED_EXPERT_MAX_M or x.dtype != torch.bfloat16:
        return False
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        TritonW4A16LinearKernel, _gfx908_gemv_enabled,
    )
    if not _gfx908_gemv_enabled():
        return False
    if getattr(mlp.down_proj, "reduce_results", True):
        return False
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        qm = getattr(proj, "quant_method", None)
        k = getattr(qm, "kernel", None)
        if not isinstance(k, TritonW4A16LinearKernel) or not k.config.weight_type.has_bias():
            return False
    return True


def gfx908_shared_expert_forward(mlp: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    k1 = mlp.gate_up_proj.quant_method.kernel
    k2 = mlp.down_proj.quant_method.kernel
    wq1, ws1, _, _ = k1._get_weight_params(mlp.gate_up_proj)
    wq2, ws2, _, _ = k2._get_weight_params(mlp.down_proj)
    gs = k1.config.group_size if k1.config.group_size != -1 else k1.config.partition_weight_shape[0]
    zp = k1.config.weight_type.bias
    wg = mlp.expert_gate.weight.reshape(-1)
    return torch.ops.vllm.gfx908_shared_expert(x, wq1, ws1, wq2, ws2, wg, gs, zp)
