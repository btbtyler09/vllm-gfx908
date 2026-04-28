# Decode Optimization Test Harness — gfx908 / Qwen3.6-35B-A3B

Reproduction scripts for the decode optimization investigation.
Findings docs: see [parent dir](../) — start with `README.md` for the methodology overview, or `qwen3_6_35b.md` for the consolidated 35B story (rounds 1–3, all shipped).

## Quick layout

| Script | Purpose |
|--------|---------|
| `coherence.sh CONTAINER MODEL` | 4-prompt coherence smoke (fibonacci, hash collisions, French translation, ocean haiku). Rejects `!!!!!`-style degenerate output. |
| `test_stage2.sh` | Spin up vllm-rocm-gfx908:latest with overlays + env, verify lm_head AITER dispatch, run pre/post coherence + 3-run TPOT. |
| `test_stage3.sh` | Same as stage2, after the custom-op bypass refactor in `dispatch_unquantized_gemm()`. |
| `microbench3.py` | Real decode-time shape rocBLAS vs AITER comparison (run inside container). |
| `microbench4.py` | BEST_CFG vs default-config comparison for AITER. |
| `microbench5.py` | MFMA tile sweep (16x16 vs 32x32) for lm_head. |
| `parse_profile.py` | Bucket torch profiler trace into linear-rocblas / all-reduce / moe-gemm / etc. |
| `accuracy_check.py` | fp32 reference comparison for AITER kernel correctness. |

## Required env (set via docker `--env`)

```
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_TRITON_GEMM=1   # MUST be docker --env, NOT rocm.py defaulter
VLLM_MI100_TORCH_COMPILE=1
AITER_TRITON_LOG_LEVEL=INFO          # for any debug visibility
VLLM_GFX908_DEBUG_DISPATCH=1         # optional: enable [AITER_DISPATCH] stderr prints
```

`VLLM_ROCM_USE_AITER_TRITON_GEMM` must be set BEFORE python imports (rocm.py's `_GFX908_DEFAULTS` defaulter runs too late: `_aiter_ops.py:1144` caches `_TRITON_UNQUANT_GEMM` at class-definition time).
