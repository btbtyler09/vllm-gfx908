#!/usr/bin/env python3
"""MFMA tile sweep for lm_head shape (1, 62080, 2048)."""
import os, sys, torch
os.environ["AITER_TRITON_LOG_LEVEL"] = "WARNING"
sys.path.insert(0, "/workspace/aiter")
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

device = torch.device("cuda")

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

def cfg(BM, BN, BK, nw, mfma, ksplit=1, kbs=2048, ws=2):
    return dict(
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=1, num_warps=nw, num_stages=2, waves_per_eu=ws,
        matrix_instr_nonkdim=mfma, cache_modifier=".cg",
        NUM_KSPLIT=ksplit, SPLITK_BLOCK_SIZE=kbs, kpack=1,
    )

x = torch.randn(1, 2048, dtype=torch.float16, device=device)
w = torch.randn(62080, 2048, dtype=torch.float16, device=device)

print("=== rocBLAS baseline ===")
print(f"{bench(lambda: x @ w.T):.2f} us")
print()
print("=== AITER MFMA tile sweep for lm_head (1, 62080, 2048) ===")
print(f"{'cfg':<50s} {'us':>8s} {'speedup':>8s}")
print("-" * 70)
roc = bench(lambda: x @ w.T)
configs = [
    ("BM16_BN64_BK128_w4_mfma16_k1",  cfg(16, 64, 128, 4, 16)),  # current BEST_CFG
    ("BM16_BN64_BK128_w4_mfma32_k1",  cfg(16, 64, 128, 4, 32)),
    ("BM16_BN128_BK64_w4_mfma16_k1",  cfg(16, 128, 64, 4, 16)),
    ("BM16_BN128_BK64_w4_mfma32_k1",  cfg(16, 128, 64, 4, 32)),
    ("BM16_BN256_BK64_w4_mfma16_k1",  cfg(16, 256, 64, 4, 16)),
    ("BM16_BN256_BK64_w4_mfma32_k1",  cfg(16, 256, 64, 4, 32)),
    ("BM32_BN64_BK128_w4_mfma32_k1",  cfg(32, 64, 128, 4, 32)),
    ("BM32_BN128_BK64_w8_mfma32_k1",  cfg(32, 128, 64, 8, 32)),
    ("BM16_BN64_BK64_w4_mfma16_k2",   cfg(16, 64, 64, 4, 16, ksplit=2, kbs=1024)),
    ("BM16_BN64_BK64_w4_mfma16_k4",   cfg(16, 64, 64, 4, 16, ksplit=4, kbs=512)),
    ("BM16_BN64_BK64_w4_mfma32_k2",   cfg(16, 64, 64, 4, 32, ksplit=2, kbs=1024)),
    ("BM16_BN128_BK64_w4_mfma32_k2",  cfg(16, 128, 64, 4, 32, ksplit=2, kbs=1024)),
]
for name, c in configs:
    try:
        t = bench(lambda c=c: gemm_a16w16(x, w, dtype=torch.float16, config=c))
        sp = roc / t
        flag = " <- WIN" if sp > 1.10 else ""
        print(f"{name:<50s} {t:>8.2f} {sp:>7.2f}x{flag}")
    except Exception as e:
        print(f"{name:<50s} FAIL: {str(e)[:40]}")
