"""Round-6 Phase 1 / Phase 3 microbench: scalar baseline + MFMA A/B.

Phase 1 use: measure ops.gptq_gemm (current scalar kernel) on the 4 production
shapes at M=1, 4, 16. Saves baseline JSON for Phase 3 comparison.

Phase 3 use: compare MFMA-built _C.abi3.so against baseline. Run with
--baseline <path> to load and compare.

Usage:
    # Phase 1 baseline capture
    python3 test_mfma_microbench.py --output baseline.json

    # Phase 3 A/B vs baseline (same script, different .so via overlay mount)
    python3 test_mfma_microbench.py --output mfma.json --baseline baseline.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import vllm._custom_ops as ops


SHAPES = [
    ("qkv",     5120, 3584, 32),
    ("o_proj",  1536, 5120, 32),
    ("gate_up", 5120, 8704, 32),
    ("down",    4352, 5120, 32),
]
M_VALUES = [1, 2, 4, 8, 16]


def pack_w8_K(w_int8_KN: torch.Tensor) -> torch.Tensor:
    K, N = w_int8_KN.shape
    K4 = K // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=w_int8_KN.device)
    w_view = w_int8_KN.to(torch.int32).view(K4, 4, N)
    return ((w_view & 0xFF) << shifts[None, :, None]).sum(dim=1, dtype=torch.int32)


def pack_zeros_N(z_GN: torch.Tensor) -> torch.Tensor:
    G, N = z_GN.shape
    N4 = N // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=z_GN.device)
    z_view = z_GN.to(torch.int32).view(G, N4, 4)
    return ((z_view & 0xFF) << shifts[None, None, :]).sum(dim=2, dtype=torch.int32)


def make_tensors(M, K, N, group_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    G = K // group_size
    a = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    w_int8 = torch.randint(0, 256, (K, N), dtype=torch.int32, device=device).to(torch.uint8)
    scales = torch.rand(G, N, dtype=torch.float16, device=device) * 0.05 + 0.001
    zeros = torch.full((G, N), 128, dtype=torch.int32, device=device).to(torch.uint8)
    b_q_K = pack_w8_K(w_int8.to(torch.int32))
    zeros_packed = pack_zeros_N(zeros.to(torch.int32))
    g_idx = torch.arange(K, dtype=torch.int32, device=device) // group_size
    return a, b_q_K, zeros_packed, scales, g_idx


def time_fn(fn, warmup=30, iters=300):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / iters


def bench_shape(M, name, K, N, gs):
    a, b_q_K, zeros_packed, scales, g_idx = make_tensors(M, K, N, gs)

    def call():
        return ops.gptq_gemm(
            a, b_q_K, zeros_packed, scales, g_idx,
            True,   # use_exllama
            False,  # use_v2_format
            8,      # bit
        )

    # quick correctness check (output not NaN)
    out = call()
    if torch.isnan(out).any():
        return {"error": "NaN in output", "M": M, "K": K, "N": N}

    avg_us = time_fn(call)
    weight_bytes = K * N * 1.0  # W8 weight bytes
    bw = weight_bytes / (avg_us * 1e-6) / 1e12  # TB/s
    return {
        "M": M, "K": K, "N": N, "group_size": gs,
        "us": round(avg_us, 3),
        "bw_tbps": round(bw, 4),
        "bw_pct": round(bw / 1.2 * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--baseline", help="Optional baseline JSON for A/B")
    parser.add_argument("--label", default="scalar", help="Label for this run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available", file=sys.stderr)
        sys.exit(1)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"label:  {args.label}")
    print()

    results = {"label": args.label, "device": torch.cuda.get_device_name(0), "shapes": {}}

    print(f"{'shape':<12} {'M':>3} {'K':>5} {'N':>5}  {'us':>8}  {'bw':>8}  {'%peak':>6}")
    print("-" * 60)
    for name, K, N, gs in SHAPES:
        for M in M_VALUES:
            r = bench_shape(M, name, K, N, gs)
            if "error" in r:
                print(f"{name:<12} {M:>3}  ERROR: {r['error']}")
                continue
            key = f"{name}_M{M}"
            results["shapes"][key] = r
            print(f"{name:<12} {r['M']:>3} {r['K']:>5} {r['N']:>5}  {r['us']:>6.2f}us  "
                  f"{r['bw_tbps']:>5.3f}TB/s  {r['bw_pct']:>5.1f}%")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output}")

    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print(f"\n=== A/B vs baseline ({base['label']}) ===")
        print(f"{'shape':<12}  {'baseline':>10}  {'this':>10}  {'speedup':>8}")
        print("-" * 50)
        for key, this in results["shapes"].items():
            b = base["shapes"].get(key)
            if not b:
                continue
            sp = b["us"] / this["us"] if this["us"] > 0 else 0
            print(f"{key:<12}  {b['us']:>7.2f}us  {this['us']:>7.2f}us  {sp:>6.2f}x")


if __name__ == "__main__":
    main()
