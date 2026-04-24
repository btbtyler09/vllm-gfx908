#!/usr/bin/env python3
"""Stage 2 last-call: try the M=1 BEST_CFG from microbench2 vs default."""
import os, sys, torch
os.environ["AITER_TRITON_LOG_LEVEL"] = "WARNING"
sys.path.insert(0, "/workspace/aiter")
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

device = torch.device("cuda")

BEST_CFG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 2048, "kpack": 1,
}

# K-split config for skinny shapes
KSPLIT_CFG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512, "kpack": 1,
}

def bench(fn, n_iter=300, n_warm=30):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000

SHAPES = [
    ("lm_head",          1, 62080, 2048),
    ("full_attn_qkv",    1,  3072, 2048),
    ("linear_attn_qkvz", 1,  2560, 2048),
    ("attn_out_1024",    1,  2048, 1024),
]

print("{:<22s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
    "shape", "rocBLAS", "ai_dflt", "ai_BEST", "ai_KSPLIT", "best_sp"))
print("-" * 82)
for name, M, N, K in SHAPES:
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    t_roc = bench(lambda: x @ w.T)
    t_dflt = bench(lambda: gemm_a16w16(x, w, dtype=torch.float16))
    try:
        t_best = bench(lambda: gemm_a16w16(x, w, dtype=torch.float16, config=BEST_CFG))
    except Exception:
        t_best = float("inf")
    try:
        t_ksp = bench(lambda: gemm_a16w16(x, w, dtype=torch.float16, config=KSPLIT_CFG))
    except Exception:
        t_ksp = float("inf")
    best = min(t_dflt, t_best, t_ksp)
    sp = t_roc / best
    print("{:<22s} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>9.2f}x".format(
        name, t_roc, t_dflt, t_best, t_ksp, sp))
