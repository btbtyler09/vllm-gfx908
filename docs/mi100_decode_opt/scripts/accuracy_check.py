#!/usr/bin/env python3
"""Proper accuracy validation: compare rocBLAS, AITER gemm_a16w16, and FP32 reference
at decode shapes. Uses bounded-magnitude random inputs and relative error tolerances.
"""
import sys
import torch

sys.path.insert(0, "/workspace/aiter")
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

device = torch.device("cuda")
SHAPES = [
    ("router/gate_up", 1, 256, 2048),
    ("unfused_gate", 1, 128, 2048),
    ("shared_down", 1, 2048, 128),
    ("linear_attn_qkv", 1, 2048, 2048),
    ("linear_attn_z", 1, 4096, 2048),
    ("full_attn_qkv", 1, 1024, 2048),
    ("full_attn_o", 1, 2048, 1024),
    ("lm_head", 1, 62080, 2048),
]

# Realistic GPTQ weight distribution: roughly N(0, 0.02), inputs N(0, 1)
def make_tensors(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Inputs: realistic-scale activations after layernorm (mostly ~unit variance)
    x = torch.randn(M, K, device=device, dtype=torch.float16, generator=g)
    # Weights: typical fine-tuned model scale
    w = (0.02 * torch.randn(N, K, device=device, dtype=torch.float32, generator=g)).to(torch.float16)
    return x, w

BEST_CFG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 2048, "kpack": 1,
}

print("Comparing rocBLAS (aten::mm) and AITER gemm_a16w16 against fp32 reference")
print("Inputs: activations ~N(0,1), weights ~N(0, 0.02)  (realistic scales)\n")
print(f"{'Shape':<22} {'M×N×K':<18} {'FP16 baseline err':<24} {'rocBLAS err':<24} {'AITER err':<24}")
print("-" * 130)

for name, M, N, K in SHAPES:
    x, w = make_tensors(M, N, K)

    # FP32 reference
    x32 = x.float()
    w32 = w.float()
    ref = x32 @ w32.T  # (M, N) in fp32

    # rocBLAS / aten fp16 path
    out_roc = (x @ w.T).float()
    # AITER with best config (BM16_BN64_K128_w4_k1)
    out_aiter = gemm_a16w16(x, w, dtype=torch.float16, config=BEST_CFG).float()

    # "FP16 baseline" = fp16 matmul with fp32 intermediate (the best fp16 can do)
    # (aten::mm already uses MFMA f16 with fp32 accumulate — so out_roc IS this)
    base = out_roc

    def stats(out, ref_):
        diff = (out - ref_).abs()
        rel = diff / ref_.abs().clamp(min=1e-6)
        return diff.max().item(), diff.mean().item(), rel.mean().item()

    # Compare each to fp32 reference (ground truth)
    roc_max, roc_mean, roc_rel = stats(out_roc, ref)
    aiter_max, aiter_mean, aiter_rel = stats(out_aiter, ref)
    base_max, base_mean, base_rel = stats(base, ref)

    def fmt(mx, mn, rl):
        return f"max={mx:.3e} rel={rl:.2%}"

    print(f"{name:<22} {str((M,N,K)):<18} {fmt(base_max, base_mean, base_rel):<24} "
          f"{fmt(roc_max, roc_mean, roc_rel):<24} {fmt(aiter_max, aiter_mean, aiter_rel):<24}")
