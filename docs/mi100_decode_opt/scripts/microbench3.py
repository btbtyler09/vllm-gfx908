#!/usr/bin/env python3
"""Microbench REAL decode-time unquantized shapes (from probe data)."""
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
    return s.elapsed_time(e) / n_iter * 1000  # μs

# (name, M, N, K) — REAL shapes hitting rocm_unquantized_gemm_impl at decode
SHAPES = [
    ("lm_head",          1, 62080, 2048),
    ("full_attn_qkv",    1,  3072, 2048),
    ("linear_attn_qkvz", 1,  2560, 2048),
    ("attn_out_1024",    1,  2048, 1024),
    ("attn_proj_128",    1,  2048,  128),
    ("router_256",       1,   256, 2048),
    ("z_gate_16",        1,    16, 2048),
    ("misc_1_2048",      1,     1, 2048),
]

hdr = "{:<22s} {:<22s} {:>12s} {:>12s} {:>10s}".format(
    "shape", "M*N*K", "rocBLAS_us", "AITER_us", "speedup")
print(hdr)
print("-" * 84)
for name, M, N, K in SHAPES:
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    w = torch.randn(N, K, device=device, dtype=torch.float16)
    try:
        t_roc = bench(lambda: x @ w.T)
        t_ai = bench(lambda: gemm_a16w16(x, w, dtype=torch.float16))
        sp = t_roc / t_ai
        flag = "<- WIN" if sp > 1.05 else ("<- LOSS" if sp < 0.95 else "")
        print("{:<22s} {:<22s} {:>12.2f} {:>12.2f} {:>9.2f}x {}".format(
            name, str((M, N, K)), t_roc, t_ai, sp, flag))
    except Exception as e:
        print("{:<22s} FAILED: {}".format(name, str(e)[:60]))
