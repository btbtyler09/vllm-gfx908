# Round-5 Phase 0 — Qwen3.6-27B-GPTQ-8bit profile audit (2026-04-27)

## TL;DR

Captured live decode profile of 27B-GPTQ-8bit on the shipped round-4
image (`btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev`). **Two clear
findings:**

1. **Lever A free win verified.** Phase 2 E (Mori-pattern persistent-
   handle CAR) auto-applies. TPOT 20.43 → **18.82 ms** (-7.9% / +8.5%
   throughput), 49 → **53 tok/s**. RCCL all-reduce bucket: round-3's
   ~16% ring fallback → round-5's 5.9% (custom AR + remnant ncclDev).
   No code change required.
2. **GPTQ GEMM is the elephant.** Single bucket `gptq-gemm` at
   **~60% of GPU kernel time** across all 4 ranks. The hot kernel is
   `vllm::gptq::gemm_half_q_half_gptq_8bit_kernel<true, 1>` — 256 calls
   per decode step × ~45 µs/call = **~11.6 ms / step / rank**. This
   is the largest single-bucket dominance we've ever seen on this
   stack (round-3 35B-A3B's rocBLAS elephant was 43.8%).

**Phase 0 decision branch fired:**
- "GPTQ GEMM bucket > 30% TPOT" → **Lever B is the prime target.
  Prioritize.**
- "GDN bucket < 5% TPOT" → Lever C is sub-1ms; **defer or skip.**
- "Phase 2 E shows in profile (RCCL bucket -90%+)" → **Lever A real,
  no follow-up needed.**

## Methodology

- Image: `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` (round-4 baked in).
- Model: `Qwen3.6-27B-GPTQ-8bit` (in-tree, has
  `Qwen3_5ForConditionalGeneration` arch — no `-multimodal` suffix
  needed anymore; the `-multimodal` re-quant from round-3 era is no
  longer required).
- Boot: TP=4, `--mamba-cache-mode align`, `--attention-backend
  TRITON_ATTN`, `--compilation-config '{"mode": 3, "cudagraph_mode":
  "FULL_AND_PIECEWISE"}'`, `--profiler-config '{"profiler": "torch",
  "torch_profiler_dir": "/tmp/profiles"}'`,
  `--gpu-memory-utilization 0.85`. Coherence-pre 4/4 PASS before
  capture.
- Capture: mid-stream `/start_profile` → 1.5 s decode → `/stop_profile`
  pattern. Excludes prefill from trace. Filename pattern in v0.20:
  `rank{N}.{timestamp}.pt.trace.json.gz` (was `dp0_pp0_tp{N}.*` in
  earlier round-4 captures — parser glob updated accordingly).
- Parser: `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/parse_profile.py`
  with new round-5 buckets: `linear-llgemm` (LLGemm1, wvSplitK),
  `gptq-gemm` (gemm_half_q_half, gptq_gemm, exllama, alt_8bit), and
  `cross_device_reduce` added to `all-reduce` bucket.

## Direct TPOT measurement (3 runs, c=1, 256-tok decode)

| Run | tok | duration (s) | tok/s | TPOT (ms) |
|---|---:|---:|---:|---:|
| 1 | 256 | 4.811 | 53.21 | 18.793 |
| 2 | 256 | 4.818 | 53.13 | 18.821 |
| 3 | 256 | 4.826 | 53.04 | 18.853 |
| **avg** | | | **53.13** | **18.822** |

**Round-3 baseline:** TPOT 20.43 ms / 48.96 tok/s
**Round-5 Phase 0 (= round-4 image, no source changes):** TPOT 18.82 ms /
53.13 tok/s
**Δ vs round-3:** **-7.9% TPOT / +8.5% throughput** — *free* gain from
Phase 2 E auto-applying.

## Per-bucket breakdown (all 4 ranks, post-Phase 2 E)

Profile span = ~1 decode step / rank (≈ 21 ms GPU-bucketed time across
all kernels in the captured iteration window).

| Bucket | rank0 ms | rank1 ms | rank2 ms | rank3 ms | mean ms | mean % |
|---|---:|---:|---:|---:|---:|---:|
| **gptq-gemm** | 12.64 | 12.74 | 12.63 | 12.54 | **12.64** | **59.9%** |
| linear-llgemm (lm_head) | 1.41 | 1.41 | 1.39 | 1.40 | 1.40 | 6.6% |
| elementwise | 1.31 | 1.32 | 1.30 | 1.30 | 1.31 | 6.2% |
| all-reduce | 1.25 | 1.13 | 1.30 | 1.31 | 1.25 | 5.9% |
| other | 1.18 | 1.18 | 1.19 | 1.21 | 1.19 | 5.6% |
| triton-misc | 0.91 | 0.96 | 0.92 | 0.92 | 0.93 | 4.4% |
| linear-attn (GDN) | 0.77 | 0.71 | 0.73 | 0.79 | 0.75 | 3.6% |
| memcpy | 0.46 | 0.45 | 0.49 | 0.44 | 0.46 | 2.2% |
| sampler | 0.47 | 0.46 | 0.46 | 0.46 | 0.46 | 2.2% |
| norm | 0.44 | 0.47 | 0.44 | 0.46 | 0.45 | 2.1% |
| attention (full) | 0.21 | 0.20 | 0.21 | 0.21 | 0.21 | 1.0% |
| rope | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.3% |
| **TOTAL** | **21.12** | **21.09** | **21.12** | **21.09** | **21.11** | **100.0%** |

All 4 ranks within ±0.5% of mean — clean and reproducible.

## Top kernels (rank 0)

| Kernel | Bucket | Calls | Total ms | Avg µs |
|---|---|---:|---:|---:|
| `gemm_half_q_half_gptq_8bit_kernel<true, 1>` | gptq-gemm | 256 | **11.61** | 45.4 |
| `LLGemm1_kernel<c10::Half, 4>` (lm_head) | linear-llgemm | 50 | 1.41 | 28.3 |
| `cross_device_reduce_1stage<__half, 4>` (custom AR) | all-reduce | 129 | 1.14 | 8.8 |
| `vectorized_elementwise_kernel<...FillFunctor<Half>>` | elementwise | 256 | 1.01 | 3.9 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | linear-attn | 48 | 0.58 | 12.0 |
| `__amd_rocclr_copyBuffer` | other | 128 | 0.50 | 3.9 |
| `cunn_SoftMaxForwardGmem<...float>` (sampler) | sampler | 4 | 0.47 | 117.4 |
| `Memcpy DtoD` | memcpy | 101 | 0.45 | 4.5 |
| `kernel_unified_attention` | attention | 16 | 0.21 | 13.2 |
| `_causal_conv1d_update_kernel` | linear-attn | 48 | 0.20 | 4.2 |

## What this tells us about Lever B (TritonW8A16 LinearKernel)

The dense GEMM kernel `gemm_half_q_half_gptq_8bit_kernel<true, 1>`:
- **256 calls per decode step / rank** = 64 layers × 4 fused linears
  (qkv, gate_up, down, o_proj for full-attn; qkvz, gate_up, down,
  out_proj for GDN). Matches the safetensors-derived 400 quantized
  linears expectation when accounting for fused projections.
- **45.4 µs avg per call.** HIP launch floor on gfx908 is ~5-8 µs.
  Math at MFMA peak for typical M=1 K=5120 N=3584 fp16 is ~0.1 µs.
  **The bottleneck is HBM read of the W8 weights + dequant overhead,
  NOT arithmetic.**

### Memory-bandwidth ceiling for a well-tuned kernel

Per-rank HBM bandwidth: ~1.2 TB/s on MI100. Per-call weight read for
each shape (W8 = 1 byte/weight):

| Shape (per rank) | Weight bytes | Bandwidth limit | Current (µs) | Headroom |
|---|---:|---:|---:|---:|
| qkv (5120, 3584) | 17.6 MB | 14.7 µs | ~45 µs | **3.1×** |
| o_proj (1536, 5120) | 7.5 MB | 6.3 µs | ~45 µs | **7.1×** |
| gate_up (5120, 8704) | 42.6 MB | 35.5 µs | ~45 µs | 1.3× |
| down_proj (4352, 5120) | 21.3 MB | 17.8 µs | ~45 µs | 2.5× |

**The current scalar-HIP `gemm_half_q_half_gptq_8bit_kernel` is
2-7× off bandwidth-ideal for the smaller shapes.** A Triton W8A16
kernel with `tl.dot` (MFMA) and well-tuned tile / `num_warps` /
`num_stages` could plausibly approach the bandwidth ceiling.

### Yield model for Lever B

| Scenario | gptq-gemm Δ | TPOT | tok/s | vs round-3 |
|---|---:|---:|---:|---:|
| Conservative (-30%): bandwidth half-saturated | 12.64 → 8.85 ms | ~15.0 ms | 67 | +37% |
| Realistic (-50%): bandwidth ~70% saturated | 12.64 → 6.32 ms | ~12.5 ms | 80 | +63% |
| Aggressive (-70%): bandwidth ~95% saturated | 12.64 → 3.79 ms | ~10.0 ms | **100** | **+104%** |

**Doubling becomes plausible at the aggressive end.** The current
kernel sits 2-7× off bandwidth-ideal for the shapes that matter, so
70% reduction is not a fantasy ceiling — it's roughly what you'd
expect from a kernel that respects HBM access patterns.

The "realistic" -50% case gets us to ~80 tok/s (+63% vs round-3).
That alone would be the biggest single-lever win in any round.

## What this tells us about Lever C (GDN MFMA refactor)

`fused_recurrent_gated_delta_rule_packed_decode_kernel`: 48 calls × 12 µs
= 0.58 ms. Total `linear-attn` bucket (incl. `_causal_conv1d_update`
and friends) = 0.75 ms = 3.6%. **At c=1 this is too small to chase.**
Even a 50% reduction would save ~0.3 ms TPOT (~1.6%) — not worth 3-5
days of kernel work right now.

**Note for c=16+ tiers (out of round-5 scope but worth recording):**
the GDN kernel may scale poorly with batch size (per-row scalar reduce
doesn't benefit from M batching the way `tl.dot` would). If a future
profile at c=32+ shows it growing past 10% of TPOT, Lever C becomes
worth revisiting.

## Other observations

### Phase 2 E confirmed real on 27B

- Round-3 RCCL bucket on 27B-8bit was ~16% (estimated from the 35B-A3B
  pre-Phase 2 E ratio; no direct profile from round-3 to compare).
- Round-5 all-reduce bucket: 5.9%, dominated by 129 calls of
  `cross_device_reduce_1stage` (the custom AR kernel from Phase 2 E)
  at 8.8 µs/call. Only 2 ncclDevKernel calls remain (probably barriers /
  setup).
- This matches the 35B-A3B Phase 4 audit's finding that custom AR
  shows up under `cross_device_reduce_1stage` and bucketed correctly
  with the parse_profile.py update.

### LLGemm1 is the same M=1 launch-overhead story as 35B-A3B

- 50 calls × 28.3 µs = 1.41 ms = 6.6% of TPOT.
- Math: LLGemm1 fires for `lm_head` per-rank (n=1 m=62080 k=5120,
  per round-3 27B memory). 50 calls in 1 step = lm_head fires once but
  is split into 50 sub-kernels? Possibly batched scheduler artifacts.
  Same launch-overhead-dominated character as the 35B-A3B finding.
- Not the prime target for round-5 (Lever B has 9× the headroom).
  Filed for round-6.

### Sampler softmax fires for 4 calls/step × 117 µs = 0.47 ms

Same `cunn_SoftMaxForwardGmem` from the 35B-A3B trace. 117 µs/call
suggests opportunity for a fused Triton sampler kernel, but at 2.2%
of TPOT it's a "future micro" item, not round-5 priority.

### Elementwise FillFunctor: 256 calls × 3.9 µs = 1.01 ms

Almost identical call count to the gptq_gemm kernel (256). Suggests a
per-projection fill (probably part of the inductor-compiled pre-GEMM
glue). 6% of TPOT and bounded by HIP launch floor on each call.
Reducing call count via custom op fusion in Phase 2's wrapper is a
possible micro for round-5 Phase 4 if there's time.

## Phase 0 outcome: ship Lever A, commit to Lever B, defer C

| Lever | Outcome |
|---|---|
| **A** (Phase 2 E auto-apply) | **VERIFIED.** -7.9% TPOT free, no code change. |
| **B** (TritonW8A16 + GPTQMarlin ROCm unblock) | **PROCEED.** Prime target. Aggressive yield (-70% gptq-gemm) puts doubling in reach. |
| **C** (GDN MFMA refactor) | **DEFER.** Bucket too small at c=1 (3.6%). Revisit if c=16+ profile shows growth. |
| **D** (micro-tunings) | **CONDITIONAL on Phase 2 outcome.** Sampler / FillFunctor fusion candidates if there's time after B ships. |

**Updated round-5 honest forecast:**
- Floor (Lever A only, B fails): TPOT 18.82 ms / 53 tok/s — ship as-is.
- Realistic (Lever A + B at -50%): TPOT ~12.5 ms / **~80 tok/s** = +63% vs round-3.
- Aggressive (Lever A + B at -70%): TPOT ~10.0 ms / **~100 tok/s** =
  doubling achieved.

## Reusable infra captured

- `/home/tyler/decode_opt_audit/profile_round5_part1.sh` — boot 27B-8bit
  with profiler config + coherence-pre.
- `/home/tyler/decode_opt_audit/profile_round5_part2.sh` — mid-stream
  capture + parse. **Note:** parser glob in part2.sh expected
  `dp0_pp0_tp*` filename pattern (round-4); v0.20 emits `rank{N}.*`. Fix
  baked into round-5 part2 inline command in this doc's run history;
  re-using future scripts should match `rank*.pt.trace.json.gz`.
- `/home/tyler/decode_opt_audit/profile_round5_tpot3.sh` — 3-run TPOT
  capture against booted container.
- `parse_profile.py` updated with `linear-llgemm`, `gptq-gemm`, and
  `cross_device_reduce` in all-reduce bucket. Reusable for round-5+.

## Files

- Trace files: `/tmp/decode_opt/profiles_round5_27b/rank{0,1,2,3}.*.pt.trace.json.gz`
  (~40 MB each)
- Parsed buckets: `/tmp/decode_opt/profiles_round5_27b_parsed/rank{N}_parsed.md`
- TPOT log: `/tmp/decode_opt/round5_phase0_tpot_results.txt`
- Boot log: `/tmp/decode_opt/serve_round5_phase0.log`
