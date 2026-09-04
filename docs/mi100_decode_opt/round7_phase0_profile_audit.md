# Round-7 Phase 0 — torch profile audit on 27B-GPTQ-8bit (post-round-6)

**Date:** 2026-04-27
**Image:** `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev`
(round-6-shipped, digest `sha256:3d4aaaf51c08…`)
**Model:** Qwen3.6-27B-GPTQ-8bit, TP=4 on 4× MI100 (gfx908)
**Capture:** decode-only torch profile, mid-stream `/start_profile` →
1.5s decode → `/stop_profile` (per-rank ~1 decode step captured)
**Coherence-pre:** 4/4 PASS

## TL;DR

**`gemm_half_q_half_gptq_8bit_kernel<true, 1>` is still the elephant.**
Three rounds of optimization (4 + 5 + 6) and the GPTQ q_gemm bucket
still dominates at **56.1% of GPU kernel time**. The single C++ HIP
kernel (`<m_count=1>` template specialization) accounts for **9.93 ms /
51% of total** by itself — **256 calls per decode step × 38.8 µs/call.**

No other bucket exceeds ~7%. The next four buckets sit between 4-7%
and would each need a separate engineering project for ~1-3% TPOT
yield. Lever A (q_gemm wave-occupancy on the M=1 hot path) is the
**only credible big-win path** and attacks the bucket the profile
points at directly.

## Per-bucket breakdown (avg across 4 ranks)

Per-rank totals: 19.37 / 19.36 / 19.33 / 19.36 ms (mean **19.36 ms**
of GPU kernel time per decode step). Round-6 ship TPOT is ~16.65 ms /
60 tok/s c=1 — the ~16% gap between trace-sum and wall-clock is
overlap with CAR / async memcpy / etc. (same pattern as round-4 Phase
4, which saw 11.63 ms trace-sum vs 8.75 ms wall-clock).

| Bucket | Avg ms | % of total | Notes |
|---|---:|---:|---|
| **gptq-gemm** | **10.87** | **56.1%** | C++ scalar kernel @ 9.93 ms + ~9 small Triton fused-RMS/copy kernels @ ~0.94 ms |
| linear-llgemm | 1.38 | 7.1% | LLGemm1_kernel (50 calls × 27.4 µs) — non-GPTQ linears (lm_head, norms?) |
| all-reduce | 1.33 | 6.9% | cross_device_reduce_1stage 0.94 + nccl 0.4 (CAR is dominant; RCCL fallback small) |
| elementwise | 1.30 | 6.7% | **FillFunctor 1.00 ms / 256 calls × 3.9 µs** — likely zero-init scratch buffers |
| other | 1.17 | 6.0% | __amd_rocclr_copyBuffer, rocprim trampolines, ArgMaxOps reduce |
| triton-misc | 0.90 | 4.7% | small fused kernels (rms+add, view+silu+slice, etc.) |
| linear-attn | 0.77 | 4.0% | fused_recurrent_gated_delta_rule + causal_conv1d_update (Qwen3.6 hybrid) |
| memcpy | 0.51 | 2.6% | 109 calls × 4.8 µs DtoD |
| sampler | 0.46 | 2.4% | cunn_SoftMaxForwardGmem (4 calls × 112.4 µs) — same shape as 35B-A3B |
| norm | 0.41 | 2.1% | rmsnorm Triton |
| attention | 0.19 | 1.0% | unified_attention (16 calls × 12.1 µs) |
| rope | 0.06 | 0.3% | reshape_and_cache_kernel_flash |
| **TOTAL** | **19.36** | **100.0%** | |

**MoE buckets (`moe-routing`, `moe-gemm`) are zero** — 27B is dense.

## Top 10 kernels by GPU time (rank 0)

| Kernel | Bucket | Calls | Total ms | Avg µs |
|---|---|---:|---:|---:|
| `gemm_half_q_half_gptq_8bit_kernel<true, 1>` | gptq-gemm | **256** | **9.93** | **38.8** |
| `LLGemm1_kernel<c10::Half, 4>` | linear-llgemm | 50 | 1.37 | 27.4 |
| `vectorized_elementwise_kernel<8, FillFunctor<c10::Half>>` | elementwise | 256 | 1.00 | 3.9 |
| `cross_device_reduce_1stage<__half, 4>` | all-reduce | 129 | 0.94 | 7.3 |
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | linear-attn | 48 | 0.66 | 13.7 |
| `Memcpy DtoD` | memcpy | 109 | 0.52 | 4.8 |
| `__amd_rocclr_copyBuffer` | other | 128 | 0.50 | 3.9 |
| `cunn_SoftMaxForwardGmem<4, float, ...>` | sampler | 4 | 0.45 | 112.4 |
| `triton_red_fused__to_copy_add_gptq_gemm_rms_norm_3` | gptq-gemm (Triton helper) | 48 | 0.33 | 6.9 |
| `triton_red_fused__to_copy_add_copy__rms_norm_5` | norm | 32 | 0.32 | 9.9 |

**Call-count math sanity check:** 27B has 64 layers × 4 GEMMs/layer
(qkv_proj + o_proj + gate_up_proj + down_proj) = **256 calls/decode-step
of `gemm_half_q_half_gptq_8bit_kernel`**. Confirms profile is one decode
step per rank.

## Decision-branch fire (vs plan inventory)

Plan's profile-finding decision table:

| Profile finding | Lever | Outcome |
|---|---|---|
| q_gemm bucket > 50% TPOT | A: q_gemm wave occupancy | **FIRES — bucket is 56.1%** |
| q_gemm bucket 30-50% | A' (occupancy) OR direct-asm v_dot2c | not applicable |
| LLGemm1 / linear bucket > 20% | B: persistent-launch wrapper | not applicable (LLGemm1 only 7.1%) |
| Attention bucket > 15% | C: Triton attn audit | not applicable (1.0%) |
| All-reduce bucket > 10% | D: CAR follow-up | not applicable (6.9%) |
| Sampler / softmax > 5% | E: custom Triton softmax | not applicable (2.4%) |
| Memcpy DtoD > 8% | F: copy-count root-cause | not applicable (2.6%) |

**Lever A (q_gemm wave occupancy) wins outright.** No competing lever
in the inventory has a bucket large enough to threaten its yield
ceiling.

## Why Lever A is the right call

1. **The bucket is real.** 56.1% of GPU time in the C++ scalar kernel
   on the M=1 path. Even a 10% improvement on the kernel = 5.6% TPOT.
2. **Phase-0 rocprof from round-6 already pointed at the lever.**
   `gemm_half_q_half_gptq_8bit_kernel` measured at 14% VALUBusy on qkv,
   25% on gate_up, 0.14% MemUnitStalled, 4 waves/CU. SIMD is idle 75-86%
   of the time waiting for `s_waitcnt vmcnt(...)` returns. **Adding more
   waves per CU directly attacks this idle time.**
3. **Round-5/6 wins are orthogonal to occupancy.** Round-5
   `BLOCK_KN_SIZE=256` + `__launch_bounds__(256, 1)` was a SIMD-alignment
   tune. Round-6 `v_dot2c` conditional dispatch only fires on
   m_count >= 2. **The M=1 hot path (this 56% bucket) hasn't been
   touched at the occupancy level.**
4. **Headroom is real.** Kernel uses 72 VGPR/wave (per round-6 rocprof).
   With launch_bounds(256, 2), we'd fit 2 blocks/CU = 8 waves/CU at
   2 waves/SIMD × 72 VGPR = 144 VGPR/SIMD used (256 budget). Plenty
   of room. launch_bounds(256, 3) → 12 waves/CU at 216 VGPR/SIMD —
   tight but feasible.
5. **Failure mode is recoverable.** The lever is a 1-character change
   to launch_bounds. If the sweep regresses, we revert. Worst case is
   a 0.5-day microbench investment with zero source change.

## What can go wrong (Lever A risks)

1. **Empirical optimum at (256, 1) was found via round-5 sweep.** It's
   possible higher occupancy doesn't help if memory bandwidth saturates
   (more waves → no extra parallel work, just contention). If the
   profile is actually memory-bandwidth-saturated and not latency-bound,
   the lever caps at 0%.
   - **Mitigation:** round-6 rocprof showed MemUnitStalled = 0.14%
     (negligible). NOT memory-bandwidth-saturated. Latency-bound.
2. **LDS pressure.** `block_a` LDS array is `m_count × 256 × sizeof(half) =
   up to 4 KB/block at m_count=8`. For decode (m_count=1) it's 512 B/
   block. With 8 blocks/CU we'd use 4 KB / 64 KB LDS — fine.
3. **block_a is loaded into LDS once at block start.** Higher occupancy
   doesn't change the load pattern, so no LDS bank-conflict risk.
4. **Other m_count specializations may regress.** Round-5/6 work tuned
   for m_count={1,2,4,8} with the same launch_bounds. Lever A may need
   per-template-specialization launch_bounds — `__launch_bounds__(256,
   2)` for m_count=1 only, leaving (256, 1) for higher m_count.

## Round-7 phase 1 lever pick: **A (q_gemm wave occupancy)**

### Phase 2 design (Lever A)

**Microbench sweep first** (~0.5 day):
1. Use the round-6 microbench harness (`test_mfma_microbench.py`) at
   M=1, 2, 4, 8, 16 on 4 production shapes (qkv, o_proj, gate_up,
   down_proj).
2. Build _C.abi3.so variants with `__launch_bounds__(256, N)` for
   N ∈ {1, 2, 3, 4} on the m_count=1 specialization only (keep round-5
   `(256, 1)` for higher m_count specializations).
3. Compare per-shape µs at M=1; ship variant only if M=1 improves
   ≥3% on 3 of 4 shapes AND M=2/4/8/16 don't regress.

**If sweep finds a winner:**
- 3-run TPOT vs round-6 baseline (16.65 ms target lower)
- Coherence-pre 4/4 PASS gate
- 12-tier BenchAndReport on 27B-8bit + 27B-4bit
- Coherence-post 4/4 PASS gate
- Tier-by-tier compare with `compare_reports.py` — ship-gate ≥3% c=1
  improvement, no tier regresses >2%

**If sweep finds no winner:**
- Reconsider the lever. Possible follow-ups:
  - **A.2: VGPR reduction** — restructure the inner loop to use fewer
    VGPRs, allowing higher occupancy without changing launch_bounds
    (compiler infers occupancy from register usage)
  - **A.3: prefetch pattern** — manual double-buffering of B-tile loads
    (tricky on gfx908 without `__builtin_amdgcn_global_load_lds`)
  - **G: bundle small wins** — FillFunctor reduction, sampler softmax
    custom kernel, copy-count audit (each 1-3% TPOT, sum 5-8%)

### Yield estimate

| Outcome | TPOT delta | Notes |
|---|---|---|
| Best case (launch_bounds(256, 2) win on M=1) | -8 to -15% | 5-15% TPOT improvement |
| Mid case (small win on M=1, neutral elsewhere) | -3 to -7% | 2-5% TPOT |
| Worst case (no winner, fall back to G) | -3 to -7% via bundle | from FillFunctor + sampler + 4-bit c=1 mitigation |

## Stop conditions check

| Condition | Status |
|---|---|
| Profiler `/start_profile` activated | ✓ PASS |
| Coherence-pre 4/4 | ✓ PASS |
| Trace files appeared in container | ✓ PASS (5 files: 4 ranks + 1 async_llm) |
| Bucket profile parsed without error | ✓ PASS |
| Buckets match round-6 expectations (q_gemm dominant) | ✓ PASS (56.1%, expected 50-70%) |
| Found at least one >15% bucket with credible lever | ✓ PASS (q_gemm @ 56.1% → Lever A) |

**No stop conditions hit. Round-7 Phase 1 lever pick: A.**

## Container teardown

Decode_opt_r7 container will be torn down after this audit doc is
saved. GPU resources freed for Phase 2 microbench (which will rebuild
`_C.abi3.so` with launch_bounds variants and re-overlay).

## Files

- Raw traces: `/tmp/decode_opt/profiles_round7/rank{0..3}.*.pt.trace.json.gz`
- Parsed per-rank: `/tmp/decode_opt/profiles_round7_parsed/rank{0..3}_parsed.md`
- Boot script: `/home/tyler/decode_opt_audit/profile_round7_27b8_part1.sh`
- Capture script: `/home/tyler/decode_opt_audit/profile_round7_27b8_part2.sh`
- Audit doc (this file)

## Phase 2 Lever A — execution result (FAILED)

Tested launch_bounds variants on `gemm_half_q_half_gptq_8bit_kernel`'s
m_count=1 specialization only (kept round-5 `(BLOCK_KN_SIZE, 1)` for
m_count >= 2 to preserve round-5/6 wins). Built 4 .so variants and
microbenched at M=1,2,4,8,16 on 4 production shapes vs round-6 baseline.

| Variant | qkv M=1 | o_proj M=1 | gate_up M=1 | down M=1 | Verdict |
|---|---:|---:|---:|---:|---|
| baseline `(BLOCK_KN_SIZE, 1)` | 39.59 µs | 30.27 µs | 60.41 µs | 41.24 µs | reference |
| `(BLOCK_KN_SIZE, 2)` | 40.34 (-1.9%) | 31.07 (-2.6%) | 60.87 (-0.8%) | 41.65 (-1.0%) | REJECT |
| `(BLOCK_KN_SIZE, 3)` | 39.97 (-1.0%) | 30.77 (-1.6%) | 60.87 (-0.8%) | 41.56 (-0.8%) | REJECT |
| `(BLOCK_KN_SIZE, 4)` | 44.00 (-10%) | 35.27 (-14%) | 62.33 (-3.2%) | 44.50 (-7.3%) | REJECT (spill) |

**All variants regress M=1.** N=2 and N=3 show 1-3% codegen perturbation
losses; N=4 shows clear register-spill (-10 to -14%). M >= 2 paths
unchanged across all variants (they keep round-6 dispatch).

**Why occupancy didn't help:** Block-count math reveals the kernel may
already be at runtime occupancy limits for these shapes:
- qkv M=1: 5120/256 = 20 K-blocks × 1 M-block × ceil(3584/(256*4))=4 N-blocks = 80 blocks. With 120 CUs, 0.67 blocks/CU avg.
- o_proj M=1: 30 blocks → 0.25 blocks/CU
- gate_up M=1: 180 blocks → 1.5 blocks/CU
- down M=1: 85 blocks → 0.71 blocks/CU

For 3 of 4 shapes, **total block count is sub-CU-count**. Adding
`launch_bounds(_, N>1)` cannot create more blocks; it only constrains
codegen. Result: codegen loss without occupancy gain.

The 14-25% VALUBusy figure from round-6 phase-0 rocprof reflects this:
**not enough work for the GPU to do**, not "wave-stall bound" as
originally hypothesized. The kernel is *block-count-starved* at small M.

## Phase 2 Lever H attempt — FillFunctor elimination via memset (FAILED)

Hypothesis: torch::zeros() in `gptq_gemm()` wrapper triggers FillFunctor
kernel (5-7% TPOT bucket). Replace with `cudaMemsetAsync` for lower
per-call overhead.

Result: 1-3% **regression** across all shapes/M values. PyTorch's
torch::zeros has well-optimized fast-path for small tensors; cudaMemsetAsync
launches its own generic init kernel with worse per-call overhead.

Race-condition analysis: kernel-internal init (skip torch::zeros, have
z=0 block do direct store, z>=1 atomicAdd) has a race because block
execution order is non-deterministic. gfx908 has no `__syncgrid()`.
Cleanly eliminating FillFunctor requires either invasive 2-phase
launches or kernel-level scratch coordination — both add overhead that
likely defeats the 5% savings.

## Round-7 status: no big win lever found

Two attacked levers (A: launch_bounds, H: FillFunctor memset) both
failed. Remaining levers from inventory all sub-3% individual yield:
- E (sampler softmax): ~1-2% TPOT
- F (memcpy DtoD investigation): ~1-3% TPOT (uncertain root cause)
- 4-bit c=1 mitigation: helps 4-bit only, not 8-bit
- A.2 (VGPR reduction via inner-loop restructure): 2-3 days, also
  uncertain since the bottleneck appears to be block-count, not VGPR

Bundle ceiling for round-7: ~3-5% TPOT (Lever G). Falls short of the
"big win" round-6 produced (+3.3-5.2%) and the user's stated preference
for significant moves.

**STOP for Phase 1 lever re-selection with user.**

## Phase 2 E+F bundle results (2026-04-27)

Per user direction, bundled Lever E (sampler softmax via AITER Triton) +
Lever F (mamba memcpy via mode change). Both failed.

### Lever E — sampler softmax ABORTED by microbench

Static analysis of the production decode path (Sampler config:
`logprobs_mode="raw_logprobs"`, no logprobs requested, no min_p,
`temperature=0.7, top_k=20, top_p=0.95`) showed only **2 of 5** candidate
softmax sites actually fire — both in `topk_topp_sampler.py`:

- Line 112: `probs = logits.softmax(dim=-1, dtype=torch.float32)` (forward_native)
- Line 291: `probs_sort = logits_sort.softmax(dim=-1)` (apply_top_k_top_p_pytorch top-p mask)

Other 3 candidates ruled out:
- `sampler.py:292` log_softmax — only fires when `num_logprobs is not None` (logprobs requested). Default chat path doesn't request.
- `topk_topp_sampler.py:239` compiled_random_sample — only called from `forward_cpu`, never on GPU.
- `builtin.py:106` MinPLogitsProcessor — only fires when `min_p_count > 0` (default min_p=0).

The 4 calls in the original profile = 2 sites × 2 decode steps captured.

**Microbench on `[1, 152064]` fp32**:

| Variant | torch.softmax | aiter.softmax | speedup |
|---|---:|---:|---:|
| contiguous input | 70.47 µs | 106.28 µs | **0.66× (-50.8%)** |
| sorted input    | 71.25 µs | 106.24 µs | **0.67× (-49.1%)** |

AITER softmax wraps `_softmax_kernel_online` with `BLOCK_SIZE=16384`, which
forces 9 sequential chunks per row at vocab=152064. PyTorch's
`cunn_SoftMaxForwardGmem<4, float, ...>` is better tuned for this
shape/dtype combination. Correctness was fine (max abs err 2.9e-11) but
perf is worse — plan stop condition triggered, Lever E abandoned.

Artifact: `/home/tyler/decode_opt_audit/round7_lever_e/microbench_softmax_v1.json`.

### Lever F — `mamba_cache_mode=all` is NULL

`MambaCacheMode` is `Literal["none", "all", "align"]` — there's no `bypass` mode.
Switching `align` → `all` should skip the per-step
`postprocess_mamba()`/`do_mamba_copy_block()` (gated by `if cache_config.mamba_cache_mode == "align"` at
`gpu_model_runner.py:1439, 3944`) and only do block-boundary copies (every
mamba block_size=528 tokens).

Coherence-pre 4/4 PASSED with `mamba_cache_mode=all` — no functionality break.
Re-captured profile vs round-7 baseline:

| Bucket | align (baseline) | all (Lever F) | Δ |
|---|---:|---:|---:|
| memcpy | 0.53 ms (109 calls) | 0.49 ms (109 calls) | -0.04 ms / 0 calls |
| all-reduce | 1.06 ms | 1.16 ms | +0.10 ms |
| gptq-gemm | 10.98 ms | 10.92 ms | -0.06 ms |
| **TOTAL** | **19.37 ms** | **19.40 ms** | **+0.03 ms** |

`__amd_rocclr_copyBuffer`: 128→128 calls. **Memcpy DtoD calls unchanged**
despite fully disabling postprocess_mamba. Phase 1 attribution that
"~50-60% of DtoD comes from mamba postprocess" is **falsified by experiment**.

Real source of the 109 DtoD + 128 copyBuffer calls per profile (≈55+64 per
decode step) is unknown — candidates: `.contiguous()` calls in
`gdn_linear_attn.py:90-95` (5 per mamba layer × 48 layers = 240 potential
copies if all trigger), torch.compile graph artifacts, or KV cache writes.
Tier-2 `.contiguous()` audit risky (kernel correctness depends on
contiguous inputs) and small expected yield (<1% TPOT).

Lever F mode-flip path dead. Tier-2 not pursued.

### Round-7 final status

| Lever | Result |
|---|---|
| A (launch_bounds sweep) | FAIL: kernel block-count-starved, not occupancy-bound |
| H (FillFunctor → cudaMemsetAsync) | FAIL: torch::zeros has small-tensor fast-path |
| E (sampler softmax via AITER) | ABORT: AITER 50% slower than PyTorch on production shape |
| F (mamba_cache_mode=all) | NULL: 109 DtoD calls unchanged, mode is not the source |

Round-7 ships no changes. Round-6 (`v0.20.0rc1.dev`, sha256 `3d4aaaf51c08...`)
remains the production image for 27B-8bit. Source on `mi100-optimized`
remains at HEAD `b159ce9f8` (round-6 ship), unchanged.

Source `_C.abi3.so` matches `_C.abi3.so.r7_baseline` (sha256 `3432b51ffad4…`)
which matches the round-6 ship — verified clean, no in-flight edits.
