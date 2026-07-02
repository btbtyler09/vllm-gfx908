#!/usr/bin/env python3
"""Microbenchmark AITER W8A8 vs W8A16 on gfx908."""
import sys
import time

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

import torch
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import gemm_a16w8_blockscale
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale


def quantize_per_token(x):
    """x: [M,K] -> (int8 x_q, fp32 scale per token)."""
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    x_q = (x / scale).clamp(-128, 127).round().to(torch.int8)
    return x_q, scale.squeeze(-1)


def quantize_per_block(x, block_k=128):
    """x: [M,K] -> per-block scales along K."""
    M, K = x.shape
    assert K % block_k == 0
    x_blocks = x.reshape(M, K // block_k, block_k)
    absmax = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    x_q = (x_blocks / scale).clamp(-128, 127).round().to(torch.int8).reshape(M, K)
    scale = scale.squeeze(-1)  # [M, K//block_k]
    return x_q, scale


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    shapes = [
        (1, 4352, 17408),    # decode down
        (8, 4352, 17408),
        (1, 8704, 5120),     # decode gate/up
        (8, 8704, 5120),
        (5000, 4352, 17408), # prefill down
        (5000, 8704, 5120),  # prefill gate/up
    ]
    print(f"{'M':>5} {'N':>6} {'K':>6} | W8A16 ms | W8A8 ms | speedup")
    for M, N, K in shapes:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        w = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
        w_s = torch.rand((N, K // 128), dtype=torch.float16, device="cuda") * 0.01

        # W8A16 (current)
        cfg16 = {
            "BLOCK_SIZE_M": 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128)),
            "BLOCK_SIZE_N": 64 if M <= 64 else 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
            "SPLITK_BLOCK_SIZE": 2048,
        }
        t_w8a16 = bench(lambda: gemm_a16w8_blockscale(x, w, w_s, dtype=torch.float16, config=cfg16))

        # W8A8
        x_q, x_s = quantize_per_block(x, block_k=128)
        # gemm_a8w8_blockscale expects w_scale shape (scale_n, scale_k) = (N, K//128)
        w_s_8 = w_s  # already [N, K//128]
        cfg8 = {
            "BLOCK_SIZE_M": 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128)),
            "BLOCK_SIZE_N": 64 if M <= 64 else 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 1,
            "SPLITK_BLOCK_SIZE": 2048,
            "cache_modifier": ".cg",
            "num_stages": 2,
        }
        try:
            t_w8a8 = bench(lambda: gemm_a8w8_blockscale(x_q, w, x_s, w_s_8, dtype=torch.float16, config=cfg8))
        except Exception as e:
            print(f"W8A8 error for {M},{N},{K}: {e}")
            t_w8a8 = float("inf")

        print(f"{M:>5} {N:>6} {K:>6} | {t_w8a16:>8.3f} | {t_w8a8:>7.3f} | {t_w8a16/t_w8a8 if t_w8a8 > 0 else float('inf'):>7.2f}x")


if __name__ == "__main__":
    main()
