# Round-5 Phase 2 — Lever B audit (TritonW8A16 LinearKernel) — closed without ship

**Date:** 2026-04-27
**Outcome:** **Lever B doesn't ship.** Triton W8A16 implementation cannot
beat the hand-tuned C++ `ops.gptq_gemm` reference on gfx908. The original
yield model (predicting -50% to -70% reduction in the gptq-gemm bucket)
was wrong about how much headroom existed above the C++ kernel — the C++
is already at 30-50% of HBM peak, near the realistic ceiling for any
implementation on this hardware.

## What ran

1. **Wrote `vllm/model_executor/kernels/linear/mixed_precision/triton_w8a16.py`** —
   forked from the W4A16 kernel; W4→W8 changes touched every interleave/
   shift/mask/pack-dim, so a fresh file was cleaner than `W_BITS` constexpr
   branching the W4 kernel (per user direction "default to fork if W_BITS
   diverges significantly").
   - Two kernels: `triton_w8a16_gemm_kernel` (general `tl.dot` MFMA path,
     mirrors W4A16) and `triton_w8a16_decode_kernel` (M=1 specialized,
     no MFMA, split-K via atomicAdd).
2. **Registered `TritonW8A16LinearKernel` in `_POSSIBLE_KERNELS[ROCM]`**
   between TritonW4A16 and Conch.
3. **Tried unblocking GPTQ→GPTQMarlin promotion on ROCm** in `gptq_marlin.py`
   (extending the `is_cuda() or is_cpu()` guard to also allow ROCm gfx9).
   Verified `apply_gptq_marlin_linear` calls `self.kernel.apply_weights()`
   only — no CUDA-only marlin C++ in the apply path. Stop condition for
   Phase 2a not triggered.
4. **Numerical correctness verified** on all 4 dominant 27B per-rank
   shapes (qkv 5120×3584, o_proj 1536×5120, gate_up 5120×8704, down
   4352×5120). Both HAS_ZP=True and HAS_ZP=False (ZP_BIAS=128) paths
   match a pure-PyTorch reference within fp16 ULP. Test:
   `docs/mi100_decode_opt/scripts/test_b_w8a16_kernel/test_b_numerical.py`.
5. **Microbench against `ops.gptq_gemm`** (`gemm_half_q_half_gptq_8bit_kernel`):

| Shape (per-rank, decode M=1) | HBM ideal | TritonW8A16 best | ops.gptq_gemm | speedup | Triton bw |
|---|---:|---:|---:|---:|---:|
| qkv (5120, 3584) | 15.3 µs | 57 µs (decode SK=8) | 44 µs | **0.77× (slower)** | 27% peak |
| o_proj (1536, 5120) | 6.6 µs | 43 µs (decode SK=8) | 25 µs | **0.58× (slower)** | 15% peak |
| gate_up (5120, 8704) | 37.1 µs | 119 µs (decode SK=8 BN=256) | 73 µs | **0.61× (slower)** | 31% peak |
| down_proj (4352, 5120) | 18.6 µs | 68 µs (decode SK=8) | 49 µs | **0.72× (slower)** | 27% peak |
| **avg** | | | | **~0.67× (33% slower)** | |

Sweep covered BLOCK_M ∈ {16, 32}, BLOCK_N ∈ {32, 64, 128, 256},
SPLIT_K ∈ {1, 2, 4, 8}, num_warps ∈ {2, 4, 8}, num_stages ∈ {1, 2, 3}.
**No combination beat ops.gptq_gemm on any shape.** Best Triton config
hits 27-31% of HBM peak; ops.gptq_gemm hits 30-50% via hand-tuned
vectorized loads (int4 = 16-byte vectorized HBM reads), per-thread
width-4 N-vectorization, and split-K-via-atomicAdd reduction.

## Why the original yield model was wrong

**Original assumption** (in plan / Phase 0 audit): W8 dense GEMMs at
M=1 are bandwidth-bound, and the current `ops.gptq_gemm` is "scalar
HIP" (not MFMA), so a Triton kernel using `tl.dot` could approach HBM
peak (1.2 TB/s) and yield 2-7× speedup.

**What's actually true:**
- `ops.gptq_gemm` is **NOT scalar** — it uses width-4 N-vectorized
  threads and `int4`-vectorized HBM loads (16 bytes per cacheline),
  achieving 30-50% of HBM peak. See `csrc/quantization/gptq/q_gemm.cu`
  lines 583-682 (`gemm_half_q_half_gptq_8bit_kernel`).
- It also uses split-K via 3D grid + `atomicAdd` reduction
  (`blockIdx.z` slices K). Already maximizes CU utilization.
- Triton's `tl.interleave` + shift/mask pattern for unpacking int8
  from int32 has overhead the C++ avoids by reading int8 directly via
  `int4` casts.
- For decode M=1, MFMA wastes 15/16 lanes per `tl.dot`. A scalar
  reduction (the `triton_w8a16_decode_kernel`) avoids this waste but
  doesn't recover the gap from the unpack/load overhead.
- Realistic Triton ceiling on gfx908 for this workload: **match
  ops.gptq_gemm**, not 2-3× beat it.

## Corrected Lever B yield ceiling

Even if I had spent another 2-3 days hand-optimizing the Triton kernel
to hit 50% of HBM peak (matching the C++), the speedup over current
production would be **0%** — same time, same bucket size. The only way
to materially beat the C++ would be:

1. **Tune the C++ kernel itself.** Theoretical headroom from 30-50%
   HBM peak → 60-80% peak. Yield: maybe -30-40% on gptq-gemm bucket =
   -3.7 to -5.0 ms TPOT = **+24% to +35% throughput** (53 → 66-72
   tok/s). Multi-day work, requires C++ rebuild iteration loop.
2. **Reduce GPTQ_GEMM call count.** 256 calls per token. Grouped GEMM
   batching would help if shapes were repeated, but each call is a
   different layer's projection (sequential dependencies). Not feasible
   without breaking the dataflow model.
3. **Quant-format change** (e.g. AWQ marlin path with hardware-friendly
   tensor layouts). Big effort, requires re-quantizing the model.

None of these get to the **+100% throughput** that "doubling" would
require.

## Disposition

- **`gptq_marlin.py:285-286` reverted.** ROCm promotion guard restored
  to original `is_cuda() or is_cpu()`. GPTQ stays on the
  `GPTQLinearMethod.apply()` → `ops.gptq_gemm()` C++ path (current
  production behavior).
- **`triton_w8a16.py` source file kept.** Numerically correct, dead
  code unless promotion is re-enabled. Useful starting point for round-6
  if/when we attempt C++-tune-by-Triton-port or a different angle.
- **`TritonW8A16LinearKernel` registered in `_POSSIBLE_KERNELS[ROCM]`.**
  Harmless (won't fire because GPTQMarlin doesn't promote on ROCm), but
  wired up for future use.
- **Tests kept** at
  `docs/mi100_decode_opt/scripts/test_b_w8a16_kernel/`:
  - `test_b_numerical.py` — ULP-aware numerical correctness test
  - `test_b_microbench.py` — sweep + comparison vs ops.gptq_gemm

## Honest "doubling" assessment

Doubling Qwen3.6-27B-GPTQ-8bit tok/s (49 → 98) is **not achievable in
round-5 via Lever B**. Realistic round-5 ceiling, given:

- **Lever A (CAR auto-apply):** +8.5% throughput, **shipped** (49 → 53
  tok/s).
- **Lever B (TritonW8A16):** **DEAD via Triton.** Possible round-6 path
  is tuning the C++ kernel directly (yield est. +25-35% throughput on
  top of A).
- **Lever C (GDN MFMA):** GDN bucket is 0.75 ms = 3.6% TPOT; refactor
  ceiling ~0.3 ms saved = +1.6% throughput. Not worth multi-day work.
- **Lever D (sampler softmax):** 0.47 ms bucket; +1-2% throughput
  ceiling. Marginal.

**Realistic round-5 outcome:** +8-10% throughput vs round-3 (~53-54
tok/s). To get to doubling needs round-6 effort on C++ kernel tuning
or an architectural change (e.g. AWQ-marlin port).

## Round-6 entry points

1. **Tune `csrc/quantization/gptq/q_gemm.cu` `gemm_half_q_half_gptq_8bit_kernel`
   for gfx908.** Current `BLOCK_KN_SIZE = 128`; try 256/384. Verify K-split
   factor heuristic. Check inner-loop unroll count. Profile with rocprof.
   Multi-day, requires container rebuild iteration loop.
2. **AWQ port** — re-quantize 27B as AWQ format, route through marlin's
   AWQ kernel (which DOES exist on ROCm via the LinearKernel priority
   list). Different model checkpoint required.
3. **Hand-tuned HIP M=1 W8 kernel** — bespoke gfx908 ASM, similar effort
   to the LLGemm1 entry from 35B-A3B round-4. Yield bounded by HBM peak
   anyway.

## Files

- `vllm/model_executor/kernels/linear/mixed_precision/triton_w8a16.py`
  (NEW — kept; dead code unless promotion re-enabled)
- `vllm/model_executor/kernels/linear/__init__.py` (TritonW8A16
  registered in priority list — harmless)
- `vllm/model_executor/layers/quantization/gptq_marlin.py:285-286`
  (REVERTED to original guard)
- `docs/mi100_decode_opt/scripts/test_b_w8a16_kernel/` (kept; tests
  + sweep harness)
- `/tmp/decode_opt/w8a16_microbench.log` (sweep results)
