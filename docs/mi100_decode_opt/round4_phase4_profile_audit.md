# Round-4 Phase 4 — profile audit before round-4 sign-off

**Date:** 2026-04-27
**Image:** `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev`
(digest `sha256:a15a6adb31d7…`)
**Model:** Qwen3.6-35B-A3B-GPTQ-8bit, TP=4 on 4× MI100 (gfx908)
**Capture:** decode-only torch profile, mid-stream `/start_profile` →
1.5s decode → `/stop_profile` (≈150 decode steps)
**Coherence-pre:** 4/4 PASS

## TL;DR

The "elephant" the audit was supposed to find — `MT128x128x32` rocBLAS
GEMM at 43.8% of round-3 baseline — is **already 70% smaller** in the
shipped round-4 image. Round-3 stage 5h's custom-op inductor escape hatch
silently switched dispatch from rocBLAS Tensile (`MT128x128x32` wasted
tile) to **vLLM's `LLGemm1_kernel` skinny GEMM** (custom kernel for M=1).
This was an unintended-but-real win we never measured.

After reclassifying the parser's "other" bucket (which lumps LLGemm1 and
custom-AR under "other"), the round-4 picture is:

| Bucket (reclassified) | Round-4 ms | % of TPOT |
|---|---:|---:|
| linear (rocBLAS Tensile + LLGemm1) | **2.59** | **22.3%** |
| moe-routing (topkGating + sort + align) | 1.37 | 11.7% |
| moe-gemm (`fused_moe_kernel_gptq_awq`) | 1.26 | 10.9% |
| triton-misc | 0.98 | 8.4% |
| memcpy DtoD | 0.93 | 8.0% |
| all-reduce (RCCL + `cross_device_reduce_1stage`) | 0.82 | 7.0% |
| elementwise | 0.73 | 6.3% |
| norm | 0.61 | 5.2% |
| linear-attn (gated delta rule) | 0.49 | 4.2% |
| sampler | 0.46 | 4.0% |
| other (residual after reclass) | 1.23 | 10.6% |
| attention (paged) | 0.12 | 1.1% |
| rope | 0.04 | 0.3% |
| **TOTAL** | **11.63** | **100%** |

**No single bucket is ≥20%.** Largest is the "linear" bucket at 22.3%, but
it's already a custom skinny-GEMM kernel — not a mistuned dispatch
problem. Next 4 buckets sit at 8-12%; each is a separate engineering
project for 2-5% TPOT individually.

## Per-bucket diff: round-3 baseline → round-4

Round-3 baseline trace: `/tmp/decode_opt/profiles/dp0_pp0_tp0_*.pt.trace.json.gz`
(taken pre-stage-5h era, TPOT 21.19 ms).

Round-4 trace: `/tmp/decode_opt/profiles_round4/dp0_pp0_tp0_*.pt.trace.json.gz`
(shipped image, TPOT 11.63 ms — wall-clock 8.75 ms via direct measurement).

| Bucket | R3 ms | R3 % | R4 ms | R4 % | Δ ms | Notes |
|---|---:|---:|---:|---:|---:|---|
| linear (rocblas + LLGemm1) | 9.28 | 43.8% | 2.59 | 22.3% | **-6.69** | **Dispatch fix in round-3 stage 5h: MT128x128x32 → LLGemm1** |
| all-reduce (RCCL + custom AR) | 3.39 | 16.0% | 0.82 | 7.0% | **-2.57** | **Round-4 E shipped: RCCL 3.39 → 0.12 ms; custom AR 0 → 0.70 ms** |
| moe-gemm | 1.80 | 8.5% | 1.26 | 10.9% | -0.54 | **Round-4 B1' shipped — modest absolute improvement** |
| moe-routing | 1.51 | 7.1% | 1.37 | 11.7% | -0.14 | held |
| memcpy | 0.45 | 2.1% | 0.93 | 8.0% | +0.48 | **UP — likely CAR staging copies + LLGemm1 setup** |
| norm | 0.66 | 3.1% | 0.61 | 5.2% | -0.05 | held |
| sampler | 0.46 | 2.2% | 0.46 | 4.0% | 0 | held |
| linear-attn | 0.56 | 2.7% | 0.49 | 4.2% | -0.07 | held |
| triton-misc | 1.04 | 4.9% | 0.98 | 8.4% | -0.06 | held |
| elementwise | 0.79 | 3.7% | 0.73 | 6.3% | -0.06 | held |
| attention | 0.16 | 0.7% | 0.12 | 1.1% | -0.04 | held |
| **TOTAL** | **21.19** | | **11.63** | | **-9.56** | -45% |

**Where the 9.56 ms came from:**
- 70% from linear dispatch fix (stage 5h side-effect)
- 27% from all-reduce fix (E shipped)
- 6% from MoE config tune (B1' shipped)
- ~0% from everything else

## Top 10 round-4 kernels by GPU time

| Kernel | Bucket | Calls | ms | µs/call |
|---|---|---:|---:|---:|
| `LLGemm1_kernel<c10::Half, 4>` | linear (was "other") | 232 | 1.94 | 8.3 |
| `fused_moe_kernel_gptq_awq` | moe-gemm | 80 | 1.26 | 15.8 |
| `Memcpy DtoD` | memcpy | 175 | 0.93 | 5.3 |
| `topkGating<8, 256, 4, 16, 64>` | moe-routing | 40 | 0.72 | 18.1 |
| `cross_device_reduce_1stage<__half, 4>` | all-reduce (was "other") | 81 | 0.70 | 8.7 |
| `Cijk_…MT4x16x64…` rocBLAS | linear | 40 | 0.49 | 12.3 |
| `cunn_SoftMaxForwardGmem<4, float, …>` | sampler | 4 | 0.46 | 115.7 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | linear-attn | 30 | 0.37 | 12.4 |
| `moe_align_block_size_kernel<int>` | moe-routing | 40 | 0.33 | 8.4 |
| `at::native::reduce_kernel<128, 4>` | other | 40 | 0.32 | 8.0 |

**LLGemm1_kernel** at 232 calls × 8.3 µs is now the single biggest
contributor. For typical attention QKV/O size (M=1, K=2048-7168, N=2048-7168),
math is ~10 MFLOPs per call → MI100 fp16 MFMA peak (184 TFLOPS) gives a
~50 ns floor. So 8.3 µs is **launch-overhead dominated**, not
math-dominated. Reducing **launch count** via fusion (e.g. fused QKV with
split-output, fused gate+up_proj) is higher-leverage than making each
kernel faster.

## Decision-branch fire

Plan's decision table (from `velvet-inventing-badger.md` Phase 4):

| Profile finding | Action |
|---|---|
| Single bucket >15% TPOT, not yet attacked, with credible lever | Round-5 phase, separate `/plan` |
| 5-15% bucket with low-cost lever | In-flight extension to round-4 |
| All buckets <5% / fragmented | Ship round-4 as-is |
| B1' / E didn't actually work | STOP, re-investigate |

**Outcome: hybrid** — the table didn't anticipate LLGemm1 hiding under
"other." Reclassified, no bucket exceeds 22.3%, and that 22.3% bucket is
already a custom skinny-GEMM kernel where per-call cost is dominated by
launch overhead. Several 5-15% buckets exist but **each requires its own
engineering project** (no low-cost lever like a config-file-only ship).

**Verified that round-4 ship works as designed:**
- E (CAR fix): all-reduce bucket dropped 76% (3.39 → 0.82 ms). Custom AR
  is the dominant path; RCCL fallback only fires 2 times per token (was
  82). ✓
- B1' (MoE config): moe-gemm dropped 30% absolute (1.80 → 1.26 ms). The
  W8 wna16 JSON is loading from the wheel. ✓
- Both were measurement-confirmed; the wall-clock TPOT improvements are
  real, not noise.

## Remaining levers ranked (vs current 9.59 ms baseline)

**Lever-1 candidate (QKV/gate+up fusion) was killed by validation
pass on 2026-04-27.** Both are already fused at the model-architecture
level: `Qwen3MoeAttention.qkv_proj` is `QKVParallelLinear`,
`Qwen3MoeMLP.gate_up_proj` is `MergedColumnParallelLinear`, and
`Qwen3NextGatedDeltaNet.in_proj_qkvz` / `in_proj_ba` are
`MergedColumnParallelLinear`. The 232 LLGemm1 calls already represent
the fused versions. This is a **launch-overhead problem, not a fusion
problem** — Python-side fusion lever doesn't exist.

| # | Lever | Bucket | Yield (TPOT %) | Cost | Risk | Validated? |
|---|---|---|---|---|---|---|
| 2 | Custom HIP MFMA skinny-GEMM kernel for M=1 → make LLGemm1 ~2× faster | linear | 6-10% | 5-10 days | High (kernel work) | Hypothesis only — needs microbench |
| 3 | Lever F: custom gfx908 MFMA fused-MoE kernel | moe-gemm | 4-5% | 2-3 days | Medium | Hypothesis only |
| 4 | Identify + reduce 175 memcpy DtoD calls | memcpy | 3-4% | 1-2 days | Low | Need root-cause |
| 5 | Custom Triton softmax kernel (4×115µs sampler softmax → ~60µs) | sampler | 2-3% | 1-2 days | Low | Yield ceiling clear |
| 6 | Lever L: per-layer closure binding for n>4 dispatch | (off-decode) | c=64+ tier only | 1 day | Low | Won't help c=1 |
| 7 | Lower-call cross_device_reduce (multi-stage path) | all-reduce | 1-2% | 2-3 days | Medium | Marginal |
| 8 | Persistent / batched-launch wrapper for LLGemm1 (one kernel handles N consecutive layers' GEMMs) | linear | 4-8% theoretical | 5-10 days | High (capture-graph integration risk) | Speculative |

**Cumulative ceiling if EVERY lever shipped:** ~15-25% TPOT (best case,
no overlap) = TPOT 9.59 → ~7.5-8.0 ms. Realistic stack with friction:
~10-15% = TPOT 9.59 → ~8.0-8.5 ms.

**No single tractable lever delivers >10% by itself.** The biggest is
lever #2 (custom M=1 MFMA kernel) — 1-2 weeks of HIP work, high risk,
and would need a microbench to confirm whether the launch-overhead floor
on gfx908 is even *beatable* by hand-rolled HIP (a custom kernel may
still hit the same ~5-8 µs HIP launch floor, capping yield).

## Honest verdict for the user

The user's framing was: "I don't want to sign off on round 4 for a small
gain, I'd rather go for something significant."

**The data says:**
- The "obvious" significant lever the audit was supposed to find (the
  43.8% rocBLAS bucket) **already shrunk 70% indirectly** via round-3
  stage 5h. We never measured that — round-4 is benefiting from it
  silently.
- The **single biggest remaining bucket is 22.3%** (linear), but it's
  already running an optimized custom kernel; further wins require
  kernel-level work, not config tweaks.
- **No "one config file ship" win remains** that mirrors the B1' pattern.
- The largest tractable lever is **#1 (QKV/gate+up fusion)** at 4-8% for
  ~3 days work — but even this is "modest gain" not "significant."
- Anything >10% requires multi-week kernel engineering (#2).

**Recommendation:** ship round-4 as-is. Document the LLGemm1 launch-
overhead ceiling and Lever F as the round-5 entry points. Round-5 should
be scoped as a kernel-engineering round (custom MFMA M=1 GEMM and/or
custom gfx908 MFMA fmoe kernel + persistent-launch wrapper), not a
config/dispatch round, because the dispatch-level wins are exhausted.

**Microbench gate before round-5:** before committing weeks of HIP work
on a custom M=1 MFMA kernel, write a 1-day microbench that hand-rolls
the absolute minimum-viable M=1 K=2048 N=2048 fp16 MFMA launch and
measures it against LLGemm1's 8.3 µs. If hand-rolled hits the same ~5-8
µs floor, the kernel-engineering ceiling is launch overhead — and lever
#2 caps at maybe 1-2 µs/call save = 3-5% TPOT, not 6-10%. **Run this
microbench first** to validate that lever #2 is even worth pursuing.

## Critical files for round-5 entry

- `vllm/csrc/quantization/llmm/` — search for `LLGemm1_kernel`
  implementation source. (Round-5 lever #1/#2 starting point.)
- `vllm/model_executor/layers/linear.py` — where QKV is currently
  invoked; check whether multi-call vs fused-call patterns exist.
- `aiter/ops/triton/_triton_kernels/moe/` — Lever F starting point if
  that's preferred over linear work.
- `/tmp/decode_opt/profiles_round4/` — round-4 trace files for any future
  comparison.
- `/tmp/decode_opt/profiles_round4_parsed/` — parsed bucket breakdowns
  per rank.

## Stop conditions hit

None of plan's STOP triggers fired:
- Profile capture succeeded ✓
- Coherence-pre 4/4 PASS ✓
- Buckets did NOT match round-0 baseline (round-4 ship is real) ✓
- No >15% bucket on a kernel-level fix path ✓ (LLGemm1's 16.7% is
  launch-overhead-dominated, not a tile-mistuned dispatch)

## Container teardown

After audit doc accepted by user: tear down container and free GPUs.
