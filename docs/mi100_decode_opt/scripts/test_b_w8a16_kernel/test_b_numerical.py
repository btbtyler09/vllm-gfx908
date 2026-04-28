"""W8A16 numerical correctness test.

Tests TritonW8A16LinearKernel against a pure-PyTorch dequant+matmul reference
on representative shapes for Qwen3.6-27B-GPTQ-8bit (per-rank, TP=4):
  qkv (5120, 3584), o_proj (1536, 5120), gate_up (5120, 8704), down (4352, 5120)

Tolerance: ULP-aware (5x fp16 ULP at the output magnitude, floor 5e-2).
fp16 has ~10-bit mantissa so accumulating thousands of fp32 products and
casting to fp16 yields ~1-2 ULP rounding noise per output cell.

Run inside the aiter container with patched files overlay-mounted.
"""
import math
import sys
import torch
from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
    triton_w8a16_gemm,
)


def pack_w8_N(w_int8_KN: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] int8 to [K, N//4] int32 (N-packed; our kernel format)."""
    K, N = w_int8_KN.shape
    assert N % 4 == 0
    N4 = N // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=w_int8_KN.device)
    w_view = w_int8_KN.to(torch.int32).view(K, N4, 4)
    w_packed = ((w_view & 0xFF) << shifts[None, None, :]).sum(dim=2, dtype=torch.int32)
    return w_packed


def pack_zeros_N(z_int8_GN: torch.Tensor) -> torch.Tensor:
    """Pack [K//G, N] int8 zeros to [K//G, N//4] int32 (N-packed)."""
    G, N = z_int8_GN.shape
    assert N % 4 == 0
    N4 = N // 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=z_int8_GN.device)
    z_view = z_int8_GN.to(torch.int32).view(G, N4, 4)
    z_packed = ((z_view & 0xFF) << shifts[None, None, :]).sum(dim=2, dtype=torch.int32)
    return z_packed


def make_gptq_w8_tensors(K: int, N: int, group_size: int, device="cuda", seed=0):
    """Build a synthetic W8 weight set (asymmetric).

    Returns:
      a:                [1, K] fp16 input
      w_int8_KN:        [K, N] uint8 ground-truth weights
      scales_GN:        [K//G, N] fp16 per-group scales
      zeros_GN:         [K//G, N] uint8 per-group zeros (random near 128)
      b_q_KN_packed:    [K, N//4] int32 N-packed (kernel format)
      zeros_NG_packed:  [K//G, N//4] int32 N-packed (kernel HAS_ZP path)
    """
    torch.manual_seed(seed)
    G = K // group_size

    a = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

    w_int8_KN = torch.randint(
        0, 256, (K, N), dtype=torch.int32, device=device
    ).to(torch.uint8)

    scales_GN = (torch.rand(G, N, dtype=torch.float16, device=device) * 0.05 + 0.001)

    zeros_GN = torch.randint(
        120, 137, (G, N), dtype=torch.int32, device=device
    ).to(torch.uint8)

    b_q_KN_packed = pack_w8_N(w_int8_KN.to(torch.int32))
    zeros_NG_packed = pack_zeros_N(zeros_GN.to(torch.int32))

    return a, w_int8_KN, scales_GN, zeros_GN, b_q_KN_packed, zeros_NG_packed


def reference_dequant_matmul(a, w_int8_KN, scales_GN, zeros_GN, group_size):
    """Pure-PyTorch reference: dequantize and matmul in fp32, cast to fp16."""
    K, N = w_int8_KN.shape
    G = K // group_size
    w_int8_g = w_int8_KN.view(G, group_size, N).to(torch.float32)
    z_g = zeros_GN.view(G, 1, N).to(torch.float32)
    s_g = scales_GN.view(G, 1, N).to(torch.float32)
    w_fp = ((w_int8_g - z_g) * s_g).view(K, N)
    return (a.to(torch.float32) @ w_fp).to(torch.float16)


def _ulp_atol(ref_max: float) -> float:
    """5x fp16 ULP at the output magnitude, with a 5e-2 floor."""
    if ref_max <= 0:
        return 5e-2
    e = int(math.floor(math.log2(ref_max)))
    fp16_ulp = 2.0 ** (e - 10)
    return max(5.0 * fp16_ulp, 5e-2)


def test_one(K: int, N: int, group_size: int, name: str) -> bool:
    print(f"\n=== {name}: K={K}, N={N}, group={group_size} ===")
    if K % group_size != 0:
        print(f"  SKIP -- K {K} not divisible by group_size {group_size}")
        return True

    a, w_int8_KN, scales_GN, zeros_GN, b_q_KN, zeros_NG = make_gptq_w8_tensors(
        K, N, group_size, seed=K * 7 + N * 13 + group_size
    )

    ref_asym = reference_dequant_matmul(a, w_int8_KN, scales_GN, zeros_GN, group_size)
    ref_max = ref_asym.abs().max().item()
    atol = _ulp_atol(ref_max)
    print(f"  ref range: pm{ref_max:.4f}  atol={atol:.4f}")

    # HAS_ZP=True: pass zeros tensor, kernel reads them
    out_zp = triton_w8a16_gemm(
        a=a, b_q=b_q_KN, scales=scales_GN, qzeros=zeros_NG,
        group_size=group_size, zp_bias=0,
    )
    diff_zp = (out_zp.float() - ref_asym.float()).abs()
    pass_zp = diff_zp.max().item() < atol
    print(f"  HAS_ZP=True: max_diff={diff_zp.max().item():.4f}  "
          f"mean={diff_zp.mean().item():.4f}  "
          f"{'PASS' if pass_zp else 'FAIL'}")
    if not pass_zp:
        idx = diff_zp.argmax().item()
        print(f"    worst: ref={ref_asym.flatten()[idx].item()}, "
              f"got={out_zp.flatten()[idx].item()}")

    # HAS_ZP=False: zeros all = 128, kernel uses ZP_BIAS=128 constant
    zeros_GN_sym = torch.full_like(zeros_GN, 128)
    ref_sym = reference_dequant_matmul(a, w_int8_KN, scales_GN, zeros_GN_sym, group_size)
    out_sym = triton_w8a16_gemm(
        a=a, b_q=b_q_KN, scales=scales_GN, qzeros=None,
        group_size=group_size, zp_bias=128,
    )
    diff_sym = (out_sym.float() - ref_sym.float()).abs()
    pass_sym = diff_sym.max().item() < atol
    print(f"  HAS_ZP=False: max_diff={diff_sym.max().item():.4f}  "
          f"mean={diff_sym.mean().item():.4f}  "
          f"{'PASS' if pass_sym else 'FAIL'}")

    return pass_zp and pass_sym


def main():
    if not torch.cuda.is_available():
        print("CUDA/HIP not available")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}")

    failures = []
    if not test_one(128, 64, 32, "smoke"):
        failures.append("smoke")
    shapes = [
        (5120, 3584, 32, "qkv (5120, 3584)"),
        (1536, 5120, 32, "o_proj (1536, 5120)"),
        (5120, 8704, 32, "gate_up (5120, 8704)"),
        (4352, 5120, 32, "down_proj (4352, 5120)"),
    ]
    for K, N, gs, name in shapes:
        if not test_one(K, N, gs, name):
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("All tests passed")


if __name__ == "__main__":
    main()
