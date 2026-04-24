# Stage 0 decode profile — Qwen3.6-35B-A3B-GPTQ-8bit on 4×MI100

**Date:** 2026-04-24
**Workload:** `Qwen3.6-35B-A3B-GPTQ-8bit`, TP=4, `--attention-backend TRITON_ATTN`, `VLLM_MI100_TORCH_COMPILE=1`, `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`.
**Baseline:** 256-tok single-request: 50 tok/s = **19.5 ms TPOT** (matches last rebench).
**Coherence:** pre-bench PASS, post-bench PASS.

## Methodology

- Two profiles captured via `VLLM_TORCH_PROFILER_DIR` → `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/profiles","torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"ignore_frontend":true}'`:
  1. **Mixed (prefill+decode):** 10 warmups → `/start_profile` → 1 request (28 in / 256 out) → `/stop_profile`. Captured 1 prefill + 2 decode iterations of GPU kernels.
  2. **Decode-only (clean):** long request kicked off in background → 1.5 s delay (let prefill + ~75 decodes elapse) → `/start_profile` → 1.5 s sleep → `/stop_profile`. Captured ~1 decode iteration of GPU kernels (profile auto-caps at `active_iterations`, but `user_annotation`s span all 78 decodes it was armed for).
- All analysis below uses the **decode-only** trace.
- Per-rank traces (`.pt.trace.json.gz`) in `/tmp/decode_opt/profiles/`. Rank 0 and rank 1 have near-identical kernel mixes (confirmed) — TP symmetry.

## Headline findings

### 1. Decode IS GPU-compute-bound (reframing #1)

The initial mixed-profile read suggested ~half of TPOT was "gap time" between steps (CPU/scheduler/launch overhead), because the prefill user_annotation was 42 ms and the decode gpu_user_annotation aggregate was only ~10 ms. **That was a profiler-scheduling artifact, not reality.**

Decode-only profile: rank 0 GPU kernel time totals **~21 ms per decode iteration**, matching the 19.5 ms TPOT almost exactly (small overage = cross-stream overlap).

There is no hidden 10 ms of scheduler/launch overhead to attack. **The GPU is busy for the full 19.5 ms per step.** This is good news for optimization focus — kernel work is the target.

### 2. B1 as originally written is dead (reframing #2)

The bottleneck inventory memo (`project_decode_bottleneck_inventory.md`) hypothesized that GPTQ W8A16 falls back to scalar Exllama (no MFMA). The profile disproves that.

- Non-MoE linears (QKV, O, router, etc.) run as **rocBLAS Tensile** kernels with MFMA. Top kernel: `Cijk_Alik_Bljk_HHS_BH_MT128x128x32_MI32x32x8x1_SN_...` — explicit MI32x32x8 MFMA instruction.
- MoE experts use `fused_moe_kernel_gptq_awq` (Triton), also MFMA-lowered.
- There is **no scalar/Exllama path** running on this model.

The replacement finding is subtler: rocBLAS picks **MT128x128x32 for decode M=1**, which computes 127/128 output rows of garbage per workgroup. The small-M variant `Cijk_Ailk_Bljk_HHS_BH_MT4x16x64` runs 4.9× faster per call (14 μs vs 69 μs). So the opportunity is not "write a Triton W8 kernel from scratch" but **"force rocBLAS to pick a decode-sized tile, or route decode-path linears through a Triton GEMM we control."**

### 3. Qwen3.6-35B-A3B is a hybrid-attention model (reframing #3)

- 40 layers total: **10 full-attention** (`unified_attention_with_output`, 10 calls) + **30 linear-attention** (`chunk_gated_delta_rule_*`, `fused_recurrent_gated_delta_rule_*`, 30 calls each).
- Linear-attn contributes only 0.58 ms per step (2.7% of decode). Not a decode target.

## Per-bucket breakdown (rank 0, decode-only, 1 iteration captured)

| Bucket | GPU ms | % of decode |
|---|---:|---:|
| **linear-rocblas** (QKV/O/router/shared-expert projections) | **9.30** | **43.9%** |
| **all-reduce** (RCCL ring via `ncclDevKernel_Generic_2`) | **3.38** | **16.0%** |
| **moe-gemm** (`fused_moe_kernel_gptq_awq`) | **1.82** | **8.6%** |
| moe-routing (topkGating, moe_align, act_and_mul) | 1.49 | 7.0% |
| triton-misc (fused small kernels) | 1.04 | 4.9% |
| elementwise (aten:: / native elementwise) | 0.80 | 3.8% |
| other (reduce_kernel, rocprim) | 1.03 | 4.9% |
| norm (rms_norm, moe_forward_shared_rms_norm) | 0.65 | 3.1% |
| linear-attn (DeltaNet chunk kernels) | 0.58 | 2.7% |
| sampler (cunn_SoftMax) | 0.45 | 2.1% |
| memcpy (DtoD) | 0.44 | 2.1% |
| rope | 0.04 | 0.2% |
| attention (full attn, 10 layers only) | 0.16 | 0.7% |
| **TOTAL** | **21.19** | **100%** |

## Top kernels by GPU time (per decode iteration, rank 0)

| Kernel | Bucket | Calls/iter | Total ms | Avg μs |
|---|---|---:|---:|---:|
| `Cijk_Alik_Bljk_HHS_BH_MT128x128x32_MI32x32x8` | linear-rocblas | 80 | 5.49 | 68.6 |
| `ncclDevKernel_Generic_2` | all-reduce | 82 | 3.39 | 41.3 |
| `fused_moe_kernel_gptq_awq` | moe-gemm | 80 | 1.80 | 22.5 |
| `Cijk_Alik_Bljk_HHS_BH_MT16x16x128_MI16x16x16` | linear-rocblas | 40 | 0.87 | 21.7 |
| `Cijk_Alik_Bljk_HHS_BH_MT128x192x32_MI32x32x8` | linear-rocblas | 2 | 0.85 | 427.4 |
| `Cijk_Alik_Bljk_HHS_BH_MT128x64x64_MI32x32x8` | linear-rocblas | 40 | 0.77 | 19.3 |
| `vllm::moe::topkGating<8,256,4,16,64,int,__half>` | moe-routing | 40 | 0.77 | 19.3 |
| `Cijk_Ailk_Bljk_HHS_BH_MT4x16x64` | linear-rocblas | 40 | 0.56 | 14.0 |
| `cunn_SoftMaxForwardGmem` | sampler | 4 | 0.46 | 115.0 |
| `Memcpy DtoD` | memcpy | 101 | 0.45 | 4.4 |

**Observations on linear-rocblas:**
- 5 distinct rocBLAS Cijk variants dispatched per decode step. 80 calls of MT128x128x32 vs 40 calls of the smaller MT16x16x128 / MT128x64x64 / MT4x16x64 — the selector is not M-uniform.
- MT128x128x32 at 69 μs × 80 calls = 5.5 ms alone — this is the single fattest kernel pattern in decode.
- MT4x16x64 at 14 μs × 40 calls = 0.56 ms — demonstrates rocBLAS *can* pick decode-friendly tiles for some shapes; just not for the hot ones.

## Status of the B1–B7 candidate list

| ID | Memo claim | Profile-confirmed status |
|---|---|---|
| B1 | Exllama scalar dequant is the linear-layer bottleneck | **Wrong.** Linears use rocBLAS MFMA. Real bottleneck is *tile selection* for decode (MT128 for M=1). Pivot: small-M rocBLAS override OR Triton GEMM for decode path. |
| B2 | gfx908 fmoe_2stages has zero precompiled .co binaries | Still true (directory doesn't even exist). BUT: MoE is only 8.6% of decode (`fused_moe_kernel_gptq_awq` is Triton, not ASM 2-stage). The ASM path would help MoE but impact is bounded by MoE's share. |
| B3 | `tuned_fmoe.csv` has no cu=60 entries | Still true. Potential to shave 30-50% off MoE (~0.5-1 ms/step). |
| B4 | Triton MoE config is M≥32 while decode is M≈1 | Plausible but secondary — MoE is only 8.6% of decode. |
| B5 | Custom all-reduce BYPASSED on gfx908 during graph capture | **Confirmed by profile.** RCCL is 16% of decode (3.4 ms). Fixing CAR likely saves 1.5-2 ms/step. |
| B6 | Attention kernel is well-tuned | Confirmed. Only 0.16 ms for attention, 0.04 ms for RoPE. Don't touch. |
| B7 | Sampler is generic, ~1 ms | Over-estimated; sampler is 0.45 ms (2.1%). Skip. |

## Updated opportunity ranking

| Candidate | Expected save (ms/step) | Expected TPOT after | Effort | Risk |
|---|---:|---:|---|---|
| **Small-M rocBLAS / Triton decode-linear kernel** (revised B1) | 3-5 | ~15-17 ms | 3-7 d | High (requires rocBLAS hacking or new kernel) |
| **B5 CAR NaN fix** | 1.5-2.5 | ~17-18 ms | 1-2 d | Medium (root-cause HIP IPC NaN) |
| **B3+B4 MoE tuning** | 0.5-1.5 | ~18-19 ms | 0.5-1 d | Low (CSV + JSON config) |

## Stage 1 recommendation

**Execute in this order:**

1. **B3+B4 first (quick validation win).** ~0.5-1 day, lowest risk, no kernel writing, no vLLM rebuild. Teaches us the AITER tuning infrastructure (tune_fmoe flow, cu=60 codegen behavior, CSV mount/reload). Expected 5-8% TPOT improvement; coherence-validate end-to-end.
2. **B5 second (medium-risk known target).** 1-2 days. With Stage 1 tuning infra in place and coherence framework working, pivot to the HIP IPC atomic NaN. Expected 8-13% TPOT improvement. Isolated fix, easy to toggle on/off.
3. **Revised B1 third (big prize, bigger project).** 3-7 days. With ~5-10 ms/step already shaved, attack the rocBLAS MT128 decode-tile problem. Options: rocBLAS Tensile override file, Triton decode-GEMM replacing gptq_gemm dispatch on decode shapes, or a combined approach.

**Why not jump straight to revised B1?** The 9.3 ms rocBLAS bucket is the single biggest target, but it's also the deepest hole. Quick MoE + CAR wins tighten our measurement methodology and give us pre/post deltas to validate the profiling harness before we spend a week on a rocBLAS deep dive. If B3+B4 and B5 together deliver the predicted 2-3.5 ms savings, we've already moved TPOT from 19.5 ms to ~16-17 ms (15-20% improvement) in 2-3 days — a real milestone before we tackle the biggest piece.

## Files and artifacts

- `/tmp/decode_opt/profiles/` — clean decode-only traces (4 ranks, ~1.2 MB each .gz)
- `/tmp/decode_opt/profiles_initial/` — original mixed-profile traces (saved for reference)
- `/tmp/decode_opt/rank0_parsed.md`, `rank1_parsed.md` — raw parser output
- `/tmp/decode_opt/parse_profile.py` — reusable bucketing parser
- `/tmp/decode_opt/coherence.sh` — 4-prompt smoke check (PASS pre + post)
- `/tmp/decode_opt/capture_decode_only.sh` — reusable decode-only profile harness
- `/tmp/decode_opt/pre_bench.log`, `post_bench.log` — coherence transcripts
