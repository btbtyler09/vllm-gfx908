"""W8A16 microbench: TritonW8A16 vs ops.gptq_gemm timing on 27B per-rank shapes.

Times decode hot path (M=1) on:
  qkv (5120, 3584), o_proj (1536, 5120), gate_up (5120, 8704), down (4352, 5120)

Reports per-call time + estimated HBM bandwidth utilization.

Usage:
  python3 test_b_microbench.py [--sweep]
  --sweep: try a grid of (BLOCK_M, BLOCK_N, BLOCK_K) for the kernel
"""
import argparse
import sys
import time
import torch
import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
    triton_w8a16_gemm,
    triton_w8a16_gemm_kernel,
    triton_w8a16_decode,
    triton_w8a16_decode_kernel,
)
from vllm.triton_utils import triton


# Realistic synthesis (matches test_b_numerical.py)
def pack_w8_K(w_int8_KN: torch.Tensor) -> torch.Tensor:
    K, N = w_int8_KN.shape
    K4 = K // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=w_int8_KN.device)
    w_view = w_int8_KN.to(torch.int32).view(K4, 4, N)
    return ((w_view & 0xFF) << shifts[None, :, None]).sum(dim=1, dtype=torch.int32)


def pack_w8_N(w_int8_KN: torch.Tensor) -> torch.Tensor:
    K, N = w_int8_KN.shape
    N4 = N // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=w_int8_KN.device)
    w_view = w_int8_KN.to(torch.int32).view(K, N4, 4)
    return ((w_view & 0xFF) << shifts[None, None, :]).sum(dim=2, dtype=torch.int32)


def pack_zeros_N(z_GN: torch.Tensor) -> torch.Tensor:
    G, N = z_GN.shape
    N4 = N // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=z_GN.device)
    z_view = z_GN.to(torch.int32).view(G, N4, 4)
    return ((z_view & 0xFF) << shifts[None, None, :]).sum(dim=2, dtype=torch.int32)


def make_tensors(K, N, group_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    G = K // group_size
    a = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1
    w_int8 = torch.randint(0, 256, (K, N), dtype=torch.int32, device=device).to(torch.uint8)
    scales = torch.rand(G, N, dtype=torch.float16, device=device) * 0.05 + 0.001
    zeros = torch.full((G, N), 128, dtype=torch.int32, device=device).to(torch.uint8)
    b_q_K = pack_w8_K(w_int8.to(torch.int32))     # [K//4, N]   for ops.gptq_gemm
    b_q_N = pack_w8_N(w_int8.to(torch.int32))     # [K, N//4]   for our kernel
    zeros_packed = pack_zeros_N(zeros.to(torch.int32))  # [K//G, N//4]
    g_idx = torch.arange(K, dtype=torch.int32, device=device) // group_size
    return a, b_q_K, b_q_N, scales, zeros, zeros_packed, g_idx


def time_fn(fn, warmup=20, iters=200) -> float:
    """Returns mean us-per-call after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / iters


def time_triton(a, b_q_N, scales, zeros_packed, group_size, has_zp=False):
    qz = zeros_packed if has_zp else None
    return time_fn(lambda: triton_w8a16_gemm(
        a=a, b_q=b_q_N, scales=scales, qzeros=qz,
        group_size=group_size, zp_bias=128 if not has_zp else 0,
    ))


def time_ops_gptq(a, b_q_K, zeros_packed, scales, g_idx):
    return time_fn(lambda: ops.gptq_gemm(
        a, b_q_K, zeros_packed, scales, g_idx,
        True,   # use_exllama
        False,  # use_v2_format (gptq, not gptq_v2 -- 27B checkpoint format)
        8,
    ))


def hbm_ideal_us(K: int, N: int, bw_tbps: float = 1.2) -> float:
    """Min time to read W8 weights at HBM bandwidth (no overhead)."""
    weight_bytes = K * N * 1.0  # 1 byte per W8 weight
    return weight_bytes / (bw_tbps * 1e12) * 1e6


def run_default_compare():
    print("\n=== W8A16 microbench: tl.dot kernel + decode kernel vs ops.gptq_gemm ===")
    print(f"{'shape':<26} | {'ideal':>8} | {'triton':>10} | {'decode':>10} | {'ops':>10} | "
          f"{'best/ops':>8} | {'best bw':>10}")
    print("-" * 120)
    shapes = [
        ("qkv (5120, 3584)", 5120, 3584, 32),
        ("o_proj (1536, 5120)", 1536, 5120, 32),
        ("gate_up (5120, 8704)", 5120, 8704, 32),
        ("down_proj (4352, 5120)", 4352, 5120, 32),
    ]
    for name, K, N, gs in shapes:
        a, b_q_K, b_q_N, scales, zeros, zeros_packed, g_idx = make_tensors(K, N, gs)

        # tl.dot kernel (HAS_ZP=False)
        t_tdot = time_triton(a, b_q_N, scales, zeros_packed, gs, has_zp=False)

        # decode-specialized kernel
        t_dec = time_fn(lambda: triton_w8a16_decode(
            a=a, b_q=b_q_N, scales=scales, qzeros=None,
            group_size=gs, zp_bias=128,
        ))

        # ops.gptq_gemm reference
        try:
            t_ops = time_ops_gptq(a, b_q_K, zeros_packed, scales, g_idx)
        except Exception as e:
            t_ops = float("nan")
            print(f"  ops.gptq_gemm failed for {name}: {e}")

        ideal = hbm_ideal_us(K, N, 1.2)
        t_best = min(t_tdot, t_dec)
        best_bw = (K * N * 1.0) / (t_best * 1e-6) / 1e12 if t_best > 0 else 0
        speedup = t_ops / t_best if t_best > 0 and t_ops > 0 else 0
        print(f"{name:<26} | {ideal:>6.2f}us | {t_tdot:>8.2f}us | {t_dec:>8.2f}us | "
              f"{t_ops:>8.2f}us | {speedup:>6.2f}x | {best_bw:>5.2f}TB/s ({best_bw/1.2*100:.0f}%)")


def run_sweep():
    print("\n=== W8A16 sweep: tl.dot kernel + decode kernel with split-K ===")
    shapes = [
        ("qkv", 5120, 3584, 32),
        ("o_proj", 1536, 5120, 32),
        ("gate_up", 5120, 8704, 32),
        ("down", 4352, 5120, 32),
    ]
    # tl.dot kernel sweep
    print("\n--- tl.dot kernel (no split-K; N-only parallelism) ---")
    for name, K, N, gs in shapes:
        a, b_q_K, b_q_N, scales, zeros, zeros_packed, g_idx = make_tensors(K, N, gs)
        results = []
        for bm in [16, 32]:
            for bn in [32, 64, 128, 256]:
                for nw in [2, 4, 8]:
                    for ns in [1, 2, 3]:
                        try:
                            def fn(bm=bm, bn=bn, nw=nw, ns=ns):
                                grid = (triton.cdiv(1, bm), triton.cdiv(N, bn))
                                c = torch.empty((1, N), dtype=a.dtype, device=a.device)
                                triton_w8a16_gemm_kernel[grid](
                                    a, b_q_N, scales, b_q_N, c,
                                    1, N, K,
                                    a.stride(0), a.stride(1),
                                    b_q_N.stride(0), b_q_N.stride(1),
                                    c.stride(0), c.stride(1),
                                    group_size=gs,
                                    HAS_ZP=False, ZP_BIAS=128,
                                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=32,
                                    num_warps=nw, num_stages=ns,
                                )
                            t = time_fn(fn, warmup=10, iters=100)
                            results.append((t, bm, bn, nw, ns))
                        except Exception:
                            pass
        results.sort()
        print(f"\n  {name} (K={K}, N={N}):")
        for t, bm, bn, nw, ns in results[:3]:
            bw = (K * N * 1.0) / (t * 1e-6) / 1e12
            print(f"    {t:>7.2f}us  BM={bm:>3} BN={bn:>3} nw={nw} ns={ns}  "
                  f"bw={bw:>5.2f}TB/s ({bw/1.2*100:>4.1f}%)")

    # Decode kernel (split-K) sweep
    print("\n--- Decode kernel (split-K + atomicAdd) ---")
    for name, K, N, gs in shapes:
        a, b_q_K, b_q_N, scales, zeros, zeros_packed, g_idx = make_tensors(K, N, gs)
        results = []
        for bn in [32, 64, 128, 256]:
            # Compute valid split-K (must divide K cleanly into blocks of 32)
            for sk in [1, 2, 4, 8]:
                if (K // sk) % 32 != 0:
                    continue
                for nw in [2, 4, 8]:
                    for ns in [1, 2, 3]:
                        try:
                            def fn(bn=bn, sk=sk, nw=nw, ns=ns):
                                triton_w8a16_decode(
                                    a=a, b_q=b_q_N, scales=scales, qzeros=None,
                                    group_size=gs, zp_bias=128,
                                    split_k=sk, block_n=bn, block_k=32,
                                    num_warps=nw, num_stages=ns,
                                )
                            t = time_fn(fn, warmup=10, iters=100)
                            results.append((t, bn, sk, nw, ns))
                        except Exception:
                            pass
        results.sort()
        print(f"\n  {name} (K={K}, N={N}):")
        for t, bn, sk, nw, ns in results[:3]:
            bw = (K * N * 1.0) / (t * 1e-6) / 1e12
            print(f"    {t:>7.2f}us  BN={bn:>3} SK={sk} nw={nw} ns={ns}  "
                  f"bw={bw:>5.2f}TB/s ({bw/1.2*100:>4.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run block size sweep")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}")

    run_default_compare()

    if args.sweep:
        run_sweep()


if __name__ == "__main__":
    main()
