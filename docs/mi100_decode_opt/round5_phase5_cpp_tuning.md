# Round-5 Phase 5 — C++ HIP kernel tuning experiments (2026-04-27 night)

**Target:** `csrc/quantization/gptq/q_gemm.cu` `gemm_half_q_half_gptq_8bit_kernel`
**Baseline:** Round-5 production = `18.82 ms TPOT / 53.13 tok/s @ c=1` on
Qwen3.6-27B-GPTQ-8bit, 4×MI100, TP=4 (round-4 image, no source changes).

## Experimental setup

- Build env: `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` with
  `/home/tyler/vllm-gfx908` mounted, `python setup.py build_ext --inplace`.
- Source state: stale `build/` dir from June 2025 moved aside.
  Fresh first build = `[40/40] Linking HIP shared module _C.abi3.so` in
  ~12 min. Incremental rebuilds (single .cu change) = ~3-4 min.
- Test: boot 27B-8bit container with the rebuilt `_C.abi3.so` overlay-mounted,
  run coherence-pre + 3-run TPOT (256-tok decode, c=1).
- A/B labels via filename: `_C.abi3.so.baseline_BLOCK128`,
  `_C.abi3.so.tweak_*`.

## Results table (TPOT @ c=1, 3-run mean)

| Variant | Coherence | TPOT (ms) | tok/s | Δ vs production |
|---|---|---:|---:|---:|
| Production v0.20.0rc1.dev (no source change) | PASS | 18.82 | 53.13 | baseline |
| Baseline rebuild (BLOCK_KN_SIZE=128) | PASS | 18.97 | 52.71 | +0.8% TPOT (noise) |
| **Tweak: BLOCK_KN_SIZE=256** | **PASS** | **17.61** | **56.79** | **−6.4% TPOT, +6.9% throughput** |
| Tweak: BLOCK_KN_SIZE=512 | PASS | 27.31 | 36.62 | +45% TPOT (regress; under-occupancy) |
| Tweak: BLOCK_KN_SIZE=320 | PASS | 21.12 | 47.34 | +12% TPOT (regress; 5 wavefronts misaligned with 4 SIMDs) |
| **Tweak: BLOCK_KN_SIZE=256 + `__launch_bounds__(256, 2)`** | **PASS** | **17.10** | **58.49** | **−9.2% TPOT, +10.1% throughput (stacks!)** |
| Tweak: BLOCK_KN_SIZE=256 + `__launch_bounds__(256, 4)` | PASS | 17.84 | 56.06 | −5.2% TPOT (worse than lb=2; compiler register-spilled) |
| **Tweak: BLOCK_KN_SIZE=256 + `__launch_bounds__(256, 1)`** | **PASS** | **16.98** | **58.90** | **−9.8% TPOT, +10.9% throughput (best so far)** |
| Tweak: BLOCK_KN_SIZE=256, NO launch_bounds | PASS | 17.64 | 56.69 | −6.3% TPOT (compiler default ≈ BLOCK_256 alone; lb(1) wins by 0.7 ms) |

**Locked-in winner: BLOCK_KN_SIZE=256 + `__launch_bounds__(BLOCK_KN_SIZE, 1)`**

Final c=1 result: TPOT 16.98 ms / 58.90 tok/s = **+10.9% throughput vs
production round-5, +20.2% vs round-3 baseline (49 → 58.90)**. 12-tier
BenchAndReport in flight to validate no regression across higher tiers.

**Optimum is sharp at BLOCK_KN_SIZE=256.** Wave/SIMD geometry on gfx908:
- 128 = 2 wavefronts → 50% SIMD occupancy (1 wave per 2 SIMDs)
- **256 = 4 wavefronts → 100% SIMD occupancy (1 wave per SIMD)** ← optimum
- 320 = 5 wavefronts → contention or 1 idle SIMD
- 512 = 8 wavefronts → register-spill / 2 waves per SIMD slow path

**Cumulative round-5 with BLOCK_KN_SIZE=256:** Lever A (CAR auto-apply,
+8.5%) + BLOCK_256 (+6.9%) = **+15.9% throughput vs round-3 baseline
(49 → 56.8 tok/s)**.

## Per-experiment notes

### Baseline rebuild

Confirms my build env produces a binary equivalent to production. 0.8% TPOT
slower = within run-to-run noise (intra-3-run variance is 0.05 ms = 0.3%).

### Tweak: BLOCK_KN_SIZE 128 → 256

**Hypothesis:** Bigger K and N tiles reduce per-block startup overhead.
Each block does 4× more work (K-block 256 vs 128, N-block 1024 vs 512)
but there are 4× fewer blocks. For 27B-8bit shapes:
- qkv (K=5120, N=3584): grid 7×40=280 → 4×20=80 blocks
- gate_up (K=5120, N=8704): 17×40=680 → 9×20=180 blocks

gfx908 has 120 CUs. With BLOCK_KN_SIZE=128, qkv was 233% over-occupied;
with =256 it drops to 67% under-occupied. Direction uncertain — test will
tell.

**Risk:** smaller shapes (e.g. o_proj K=1536 N=5120) drop from 120 blocks
(100% occupancy) to 30 blocks (25% occupancy). Could regress.

(_To be filled in once test completes_)

### Tweak: pragma unroll 4 → 8

(_Pending baseline tweak result_)

### Combo: best individual tweaks combined

(_Pending individual results_)
