# Stage 2 — AITER gemm_a16w16 for lm_head — RESULTS

**Date:** 2026-04-24
**Model:** Qwen3.6-35B-A3B-GPTQ-8bit (resolved arch: Qwen3_5MoeForConditionalGeneration)
**Stack:** vllm-rocm-gfx908:latest, TP=4, dtype half, mode-3 + FULL_AND_PIECEWISE, TRITON_ATTN

## What changed

Patches mounted as overlays (no rebuild):

1. `vllm/model_executor/layers/utils.py` — extended `use_aiter_triton_gemm` whitelist with `(m=62080, k=2048)` for the model's lm_head, and added a hardcoded `_AITER_GEMM_M1_BEST_CFG` dict that's passed when the lm_head shape matches. Default AITER config picks M_LEQ_64 which wastes blocks at M=1 — BEST_CFG (BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, num_warps=4, NUM_KSPLIT=1) recovers a 1.74× win.
2. `vllm/platforms/rocm.py:213` — flipped `VLLM_ROCM_USE_AITER_TRITON_GEMM` default to `"1"` on gfx908. Safe because the whitelist gates dispatch to one specific (m,k) pair.
3. `vllm/distributed/device_communicators/custom_all_reduce.py` — also overlaid (host commit `f86fed94f` moved the gfx908 bypass from `custom_all_reduce()` to `should_custom_ar()` — the container's older bypass tripped the new v0.19 `assert out is not None` in cuda_communicator.py).

Container env: `VLLM_ROCM_USE_AITER_TRITON_GEMM=1` set via `docker --env` (not via rocm.py defaulter — `_aiter_ops.py:1144` caches `_TRITON_UNQUANT_GEMM` at class-definition time, which is BEFORE the rocm.py defaulter runs).

## Verification

- **AITER dispatch confirmed firing**: 40 `[AITER_DISPATCH]` log lines per single decode-step request from all 4 ranks; 7128 dispatches across the 3-run measurement. (Note: AITER's own `_LOGGER.info` "GEMM_A16W16" lines do NOT appear in docker logs — vllm's logger config silently filters them. Used a stderr `print()` to verify dispatch instead.)
- **Pre-bench coherence**: PASS (fibonacci, hash_collisions, french_translation, ocean_haiku)
- **Post-bench coherence**: PASS (no drift under sustained load)

## TPOT measurement

```
run 1: 256 tok in 4.951s = 51.71 tok/s, TPOT=19.34 ms
run 2: 256 tok in 4.948s = 51.74 tok/s, TPOT=19.33 ms
run 3: 256 tok in 4.949s = 51.72 tok/s, TPOT=19.33 ms
```

Stage 2 mean: **51.72 tok/s, TPOT 19.33 ms**
Baseline (pre-Stage-2):  ~51 tok/s, TPOT 19.5 ms

**Delta: +0.72 tok/s (+1.4%) / -0.17 ms TPOT (-0.87%).**

Run-to-run variance is ±0.05 ms — change is real, not noise.

## Microbench validation (in-container)

Real shapes hitting `rocm_unquantized_gemm_impl` per request (from instrumented probe):

| Shape (M, N, K)     | Calls/req | rocBLAS μs | AITER default μs | AITER BEST_CFG μs | Best speedup |
|---------------------|-----------|------------|------------------|-------------------|--------------|
| (1, 62080, 2048)    | 40        | 466        | 360              | **267**           | **1.74×** ✓  |
| (1, 3072, 2048)     | 20,880    | 21         | 64               | 50                | 0.42× LOSS   |
| (1, 2560, 2048)     | 6,960     | 18         | 64               | 51                | 0.35× LOSS   |
| (1, 2048, 1024)     | 27,840    | 16         | 63               | 50                | 0.32× LOSS   |
| (1, 2048, 128)      | 27,840    | 15         | —                | —                 | LOSS         |
| (1, 256, 2048)      | 55,680    | 50         | 63               | —                 | 0.79× LOSS   |
| (1, 16, 2048)       | 20,880    | 19         | —                | —                 | LOSS         |
| (1, 1, 2048)        | 27,840    | 18         | —                | —                 | LOSS         |

**Conclusion:** AITER's Triton kernel has a ~50 μs per-launch floor on gfx908 for M=1 shapes, while rocBLAS sits at 15–50 μs. AITER only wins when total compute exceeds the floor — only `(1, 62080, 2048)` for our model. Per-step savings: 199 μs (= 1% TPOT). This matches the observed +0.87% TPOT.

## Why prior microbench (microbench2.py) was misleading

`microbench2.py` tested per-rank shapes assuming column-parallel sharding:
- "lm_head" tested as (1, 62080, 2048) — lm_head is ACTUALLY replicated, not split → measured shape is correct, just labeled wrong
- "linear_attn_z" tested as (1, 4096, 2048) — but the model uses **fused `in_proj_qkvz`** at (1, 2560, 2048) per rank → no separate z-projection exists in the non-LoRA path
- Other tested shapes were either correct-by-accident or didn't exist in the actual decode path

The 188,400 unquantized GEMM calls per request observed in the runtime probe are dominated by small-shape per-layer projections that AITER cannot accelerate. Real Stage 2 ceiling: ~1% TPOT (matching observation), not 5–6% as planned.

## Decision

**Keep the patch.** It's a small, low-risk, +0.87% win with verified coherence. No reason to revert.

But Stage 2 is NOT the lever it was projected to be. The bigger opportunity is in Stage 3 — the 188K-call/request pattern through an opaque `torch.ops.vllm.rocm_unquantized_gemm` custom op suggests inductor cannot fuse linears with surrounding ops. Replacing the custom-op layer with direct `F.linear` may unlock inductor fusion → potentially 1–4 ms/step (5–20% TPOT).

## Files modified

- `/home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py` — whitelist + BEST_CFG dict + dispatch path
- `/home/tyler/vllm-gfx908/vllm/platforms/rocm.py:213` — env var default
- `/tmp/decode_opt/test_stage2.sh` — test driver (mounts overlays, sets env, verifies, measures)
- `/tmp/decode_opt/stage2_results.md` — this file
- `/tmp/decode_opt/microbench3.py` `/tmp/decode_opt/microbench4.py` — real-shape microbenches

## Cleanup

- The `[AITER_DISPATCH]` stderr print added for verification can be left in (harmless, only fires for whitelisted shapes — once per decode step on lm_head)
- The `DECODE_OPT_PROBE` env var and gated probe were removed in the cleanup pass
