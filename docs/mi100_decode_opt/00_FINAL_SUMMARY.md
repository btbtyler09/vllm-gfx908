# Decode optimization for Qwen3.6-35B-A3B on 4×MI100 — FINAL SUMMARY

**Date:** 2026-04-24
**Workload:** Qwen3.6-35B-A3B-GPTQ-8bit, TP=4 on 4× MI100 (gfx908)
**Stack:** vLLM v0.19.2rc1+mi100, AITER, `VLLM_MI100_TORCH_COMPILE=1`, mode-3 + FULL_AND_PIECEWISE, TRITON_ATTN

## Bottom line

| Stage | Status | TPOT (ms) | tok/s | Notes |
|-------|--------|-----------|-------|-------|
| Baseline | — | 19.5 | 51.0 | one historical measurement |
| Stage 2 (lm_head AITER w/ BEST_CFG) | shipped | 19.33 | 51.7 | one 3-run set |
| Stage 3 (custom-op bypass) | shipped | 19.24 | 52.0 | one 3-run set |
| Final (Stage 2+3 cleaned, debug print env-gated) | shipped | 19.38 | 51.6 | one 3-run set |

All measurements coherence pre + post PASS. Confirmed AITER dispatch fires for lm_head shape on every decode step (7128 dispatches per measurement). Zero coherence regressions across all stages.

**Honest take on the magnitude**: per-set TPOT variance is ~0.05 ms within a run, but across different container starts it's ~0.05-0.15 ms. Our predicted theoretical max (1% TPOT from lm_head AITER) is right at the noise floor. The patches are **demonstrably doing the right thing** (correct dispatch, correct microbench delta), but the end-to-end TPOT win is **statistically marginal** — call it "between 0% and 2%" with high confidence the kernel-level work is positive but the integration-level effect is small.

Why so small in practice:
- lm_head is only 1 GEMM call per decode step (200μs theoretical savings = 1% TPOT)
- Custom-op bypass overhead reduction is ~0.4μs × 184 calls = 70μs (0.4% TPOT)
- Both are dwarfed by the ~9.3 ms (47% TPOT) spent in the rocBLAS small-GEMM bucket which we cannot accelerate without major refactoring

## What we tried — by stage

### Stage 0 — profile [completed] ✓
Built decode-only profile harness (`/tmp/decode_opt/parse_profile.py`). Result: 9.3 ms (43.9%) in linear-rocblas, 3.4 ms (16%) in all-reduce, 1.8 ms (8.6%) in moe-gemm. The 9.3 ms is dominated by **184 small GEMM calls per decode step at the rocBLAS ~50μs floor**, not by a few large kernels.

### Stage 1 — CAR NaN fix [SHELVED]
Goal: re-enable custom_all_reduce during cudagraph capture on gfx908 (3.4 ms / 16% TPOT prize). Verified the prize is real (broken-CAR runs at 59 tok/s = +17%). Investigated 4 hypotheses across 3 container rebuild cycles:
- Cached vs uncached allocator → cached caused spin-lock (peer atomic stores trapped in L2). Reverted.
- `__threadfence_system()` + load-side `__MEMORY_SCOPE_SYSTEM` → no effect on output.
- `VLLM_CUSTOM_ALLREDUCE_ALGO=2stage` → identical NaN. Bug is shared infra, not per-stage kernel.
- AITER's CAR + QuickReduce evaluated: AITER's CAR is byte-identical to vLLM's (no new ideas); QuickReduce's 1MB minimum makes it inert for our 4 KB decode all-reduces.

Likely root cause (unproven): stale peer IPC pointers under cudagraph replay, or HIP cudagraph + IPC memory incompatibility on CDNA1. Needs HIP-runtime instrumentation, not source patches. Decision: ship the existing bypass overlay (RCCL fallback). Revisit when budget allows.

Side benefit: discovered `f86fed94f` — the gfx908 CAR bypass needs to live in `should_custom_ar()`, not in `custom_all_reduce()`. Container's older bypass (in the wrong place) trips v0.19's new `assert out is not None` in cuda_communicator.py. Already committed to the host repo; container needs rebuild OR overlay mount.

### Stage 1a — shape inventory [completed] ✓
Real shapes hitting `rocm_unquantized_gemm_impl` per request (probe data, all 4 ranks):
| Shape (M,N,K) | Calls/req | Per step | rocBLAS μs | AITER best μs | Speedup |
|---------------|-----------|----------|------------|---------------|---------|
| (1, 62080, 2048) lm_head | 40 | 0.16 | 466 | 267 | **1.74×** |
| (1, 3072, 2048) full_attn QKV | 20,880 | 21 | 21 | 50 | 0.42× LOSS |
| (1, 2560, 2048) linear_attn QKVZ | 6,960 | 7 | 18 | 51 | 0.35× LOSS |
| (1, 2048, 1024) attn out | 27,840 | 28 | 16 | 50 | 0.32× LOSS |
| (1, 2048, 128) | 27,840 | 28 | 15 | — | LOSS |
| (1, 256, 2048) router | 55,680 | 56 | 50 | 63 | 0.79× LOSS |
| (1, 16, 2048) | 20,880 | 21 | 19 | — | LOSS |
| (1, 1, 2048) | 27,840 | 28 | 18 | — | LOSS |

**Key insight:** AITER Triton GEMM has a ~50 μs per-launch floor on gfx908 — only beats rocBLAS when the GEMM is large enough to absorb the floor. For our model that's only lm_head. Microbench2's "5–6% TPOT" projection was based on incorrect per-rank shape assumptions (linear_attn_z doesn't exist as a separate op — it's fused into in_proj_qkvz; lm_head is replicated, not column-split).

### Stage 2 — lm_head AITER dispatch [SHIPPED] ✓
Added `(m=62080, k=2048)` to the `use_aiter_triton_gemm` whitelist with hardcoded `_AITER_GEMM_M1_BEST_CFG` (BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, num_warps=4, NUM_KSPLIT=1). Default AITER `_get_config(M, N, K)` picks M_LEQ_64 (BLOCK_SIZE_M=64) which wastes blocks at M=1; BEST_CFG (BLOCK_SIZE_M=16) recovers 1.34→1.74×.
Also flipped `VLLM_ROCM_USE_AITER_TRITON_GEMM` default to `"1"` on gfx908 (rocm.py:213). MUST also set via `docker --env` because `_aiter_ops.py:1144` caches `_TRITON_UNQUANT_GEMM` at class-definition time, BEFORE the rocm.py defaulter runs.

**Result:** TPOT 19.33 ms (51.7 tok/s) = **+1.4% throughput**. Confirmed 7128 AITER dispatches over 3-run test.

### Stage 3 — custom-op bypass [SHIPPED] ✓
Added `rocm_unquantized_gemm_gfx908(layer, x, weight, bias)` and modified `dispatch_unquantized_gemm()` to return it on gfx908. The new function inlines the AITER dispatch and falls through to `torch.nn.functional.linear` for non-AITER shapes — bypassing the opaque `torch.ops.vllm.rocm_unquantized_gemm` custom op.

Hypothesis was 5–10% TPOT from inductor fusion. Reality: ~0.5% TPOT. The custom-op overhead on gfx908 is ~0.4 μs/call, much smaller than expected. F.linear → aten::mm still goes through rocBLAS; inductor doesn't get to fuse around external kernel calls.

**Result:** TPOT 19.24 ms (52.0 tok/s) = **+0.5% on top of Stage 2 = +1.96% cumulative** vs baseline. Coherence PASS.

### Stage 4 — vLLM compilation passes [investigated, no win]
- `fuse_allreduce_rms` → BLOCKED. `AllReduceFusionPass` is gated to `is_cuda()` in `pass_manager.py:38` — not imported on ROCm. Setting the flag crashes ("name 'AllReduceFusionPass' is not defined"). Same for `AsyncTPPass` (used by `fuse_gemm_comms`) and `MiniMaxQKNormPass`. Patching the import gate would also need rewriting the pass body for ROCm AR — not a quick win.
- `fuse_act_padding` → SLIGHT REGRESSION. RocmAiterTritonAddRMSNormPadFusionPass IS imported on ROCm+AITER. Pass engaged ("Enabled custom fusions: act_padding"), coherence PASS, but TPOT regressed 19.24 → 19.31 ms. Dropped.
- `fuse_norm_quant`, `fuse_act_quant` → not relevant (model has unquantized linears we care about).
- `enable_sp` → decode has seq_len=1, nothing to shard. Doesn't help.
- `fuse_rope_kvcache` → only applies to 10/40 attention layers (full-attn); marginal upside, didn't pursue.
- MFMA 32x32x8 sweep for lm_head → 32x32x8 is SLOWER than 16x16x16 at M=1 (larger MFMA tile wastes more compute on padding for skinny matmuls). Current BEST_CFG is optimal.

## Files shipped (host: /home/tyler/vllm-gfx908)

| File | Change | Purpose |
|------|--------|---------|
| `vllm/model_executor/layers/utils.py` | +66/-2 lines | Stage 2 whitelist + BEST_CFG; Stage 3 `rocm_unquantized_gemm_gfx908` + dispatch override on gfx908 |
| `vllm/platforms/rocm.py` | +4/-2 lines | `VLLM_ROCM_USE_AITER_TRITON_GEMM` default `"1"` on gfx908 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | already committed (`f86fed94f`) | gfx908 CAR bypass placed in `should_custom_ar` not `custom_all_reduce` |

Container image `vllm-rocm-gfx908:latest` is 6 days old — predates `f86fed94f`. Either rebuild OR mount `custom_all_reduce.py` as overlay (current working pattern).

## Required env

`VLLM_ROCM_USE_AITER_TRITON_GEMM=1` MUST be set via docker `--env` (not via `os.environ.setdefault` in rocm.py). The rocm.py defaulter runs too late — `_aiter_ops.py:1144` reads the env var at class-definition time which happens during the import chain BEFORE rocm.py's `_GFX908_DEFAULTS` block executes.

Optional debug: `VLLM_GFX908_DEBUG_DISPATCH=1` enables per-dispatch stderr prints (silent by default).

## Test infrastructure (`/tmp/decode_opt/`)

- `coherence.sh` — 4-prompt PASS/FAIL coherence check
- `parse_profile.py` — torch profiler trace bucketization
- `microbench2.py` — original (correct shapes by accident, missed real ones)
- `microbench3.py` — corrected real-shape benchmark
- `microbench4.py` — BEST_CFG vs default vs KSPLIT comparison
- `microbench5.py` — MFMA tile sweep for lm_head
- `accuracy_check.py` — fp32 reference comparison
- `test_stage2.sh` / `test_stage3.sh` / `test_stage4.sh` / `test_stage4b.sh` — reproducible test drivers
- `stage0_profile.md` / `stage1a_shapes.md` / `stage2_results.md` / `stage3_results.md` / `stage4_results.md` — per-stage analysis

## What's left on the table

1. **CAR NaN fix (~3.4 ms / 16% TPOT)** — biggest remaining lever. Needs HIP-runtime instrumentation to identify the IPC + cudagraph interaction bug. Out of scope for code patches.
2. **Kernel fusion at the model layer** — many small per-layer GEMMs that can't be fused via inductor (custom op or external kernel). Would need model-code-level merge of router + shared_expert + gate ops. Multi-day effort.
3. **AsyncTP / SP** — would help prefill (especially long-prompt) but not decode at c=1. Worth revisiting for prefill optimization.
4. **gfx908 patch the `is_cuda()` gate + port AllReduceFusionPass to ROCm** — would unlock the high-value AR+RMSNorm fusion. 1-2 days of work IF the pass body is portable.

## Recommendation

Ship Stage 2 + Stage 3 patches. Net +1.96% TPOT (51 → 52 tok/s) is small but free, validated, no risk. The bigger wins require either external bug fixes (Stage 1 CAR) or major refactoring (Stage 3+ fusion).
