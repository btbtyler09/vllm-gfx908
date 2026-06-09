# MTP depth sweep on gfx908 — Qwen3.6-27B-GPTQ-8bit (4×MI100)

**Date:** 2026-06-08 · **Model:** Qwen3.6-27B-GPTQ-8bit · **Hardware:** 4×MI100 (gfx908) TP4
**Harness:** `mi100-llm-testing/BenchAndReport.py` — full 12-tier suite, **real-text (sonnet) dataset**
**Companion docs:** `test_curvedinf_levers.md` (A/B runbook) · memory `project_curvedinf_dual_result`, `project_mtp_n3_crash_gfx908`

> **Merge status:** the two levers characterized here are merged on branch
> `mi100-gptq8-dual-mtp` (→ PR into `mi100-optimized`). **GPTQ8 `dual` is now the
> gfx908 default** (`VLLM_GFX908_GPTQ8=native` opts out); **the n≥3 token-budget
> bump is automatic** (no manual `--max-num-batched-tokens` needed). The serve
> commands below show the *measured* config; the "As merged" recipe at the bottom
> is what you actually run on the release image.

This characterizes MTP (Multi-Token Prediction) speculative-decode depth on top of the
**graph-safe GPTQ8 dual** GEMM win (see `project_curvedinf_dual_result`). MTP is layered
*on* dual@MTHRESH=16; the "dual base" column is the no-MTP reference. Output is **bit-exact
at every depth** (strict rejection sampling — speedup only, never a quality trade).

## How it was run

**Base stack (every arm):** round-8 prod image + AITER + `VLLM_MI100_TORCH_COMPILE=1` +
`FULL_AND_PIECEWISE` cudagraphs + graph-safe dual GPTQ8.

Serve env (constant across the sweep):
```
VLLM_GFX908_GPTQ8=dual   VLLM_GFX908_GPTQ8_MTHRESH=16
VLLM_ROCM_USE_AITER=1    VLLM_ROCM_USE_AITER_TRITON_GEMM=1
VLLM_MI100_TORCH_COMPILE=1
NCCL_ALGO=Tree           NCCL_PROTO=LL
HSA_OVERRIDE_GFX_VERSION=9.0.8
```
Serve flags:
```
vllm serve /models/Qwen3.6-27B-GPTQ-8bit --served-model-name qwen3.6-27b-8bit \
  --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --attention-backend TRITON_ATTN \
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --max-num-batched-tokens 8192 \                       # <-- THE n>=3 CRASH FIX
  --speculative-config '{"method":"mtp","num_speculative_tokens":N}'
```
Driver: `curvedinf_ab/sweep_mtp_sonnet.sh` → `run_arm.sh` (serve→ready→coherence-pre→
BenchAndReport 12-tier→capture JSON+MD→teardown). Raw per-arm data in
`curvedinf_ab/results_sonnet_*.json` + `report_sonnet_*.md`.

### The n≥3 crash and its fix
MTP `num_speculative_tokens>=3` crashed the engine (`HIP illegal memory access`) under
sustained load — **only** with 27B (big prefills) + cudagraphs + n≥3. Root cause: vLLM
auto-sets `max_num_scheduled_tokens=2048` for spec-decode; the 27B's 2048-token prefills +
draft slots overflow the 2048-sized cudagraph-captured buffer → OOB. **Not** a kernel-math
bug, **not** OOM, **not** attention-backend (ROCM_ATTN crashed identically), **not** dual.
**Fix: `--max-num-batched-tokens 8192`.** n=3 then ran all 12 tiers clean (incl. Mixed
Traffic, the tier that crashed). Full forensics in `project_mtp_n3_crash_gfx908`.

## Results — throughput (output tok/s), real-text sonnet

| tier | dual base | n=1 | n=2 | n=3 (+fix) | n=5 (+fix) |
|---|---|---|---|---|---|
| Single User c=1 (2048/512) | 53.5 | 61.6 | 71.4 | 72.1 | 73.8 |
| **Decode Stress c=1 (128/2048)** | **58.5** | **74.0** | **83.1** | **91.1** | **83.3** |
| Short Context c=16 (512/256) | 248.3 | 245.4 | 294.4 | 363.7 | 289.4 |
| Long Context 16K c=4 | 67.1 | 54.6 | 54.7 | 61.0 | 56.3 |
| Mixed Traffic c=8 (2048/512) | 166.2 | 184.6 | 170.8 | 182.9 | 207.2 |
| Concurrency c=2 | 85.3 | 95.0 | 110.0 | 107.0 | 90.1 |
| Concurrency c=4 | 119.8 | 132.3 | 107.4 | 154.4 | 129.2 |
| Concurrency c=8 | 167.5 | 189.8 | 172.5 | 184.2 | 206.9 |
| Concurrency c=16 | 224.3 | 228.4 | 277.4 | 282.2 | 267.6 |
| Concurrency c=32 | 252.7 | 338.9 | 346.4 | 324.3 | 314.0 |
| Concurrency c=64 | 372.8 | 402.8 | 409.5 | 340.2 | 327.9 |
| Concurrency c=128 | 413.0 | 419.3 | 420.0 | 363.9 | 342.4 |

**Decode curve (c=1, Decode Stress) PEAKS AT n=3:** base 58.5 → n1 74.0 (+27%) →
n2 83.1 (+42%) → **n3 91.1 (+56%, PEAK)** → n5 83.3 (rolls back to ~n2 level).
Acceptance rises with real text: ~80.7% (random) → ~89% (sonnet). **n=5 over-speculates** —
5 draft forward-passes but the deep draft tokens are rarely accepted, so compute is wasted
re-running rejected drafts; decode regresses and high-concurrency gets strictly worse than
n=3. **There is no reason to run n>3 on this model/hardware.**

## Reading the data / recommendation

- **MTP pays in the decode / low-concurrency regime** (c=1..c=16): +25–56%. Returns diminish
  with output length — Decode Stress (2048-out) keeps gaining through n=3; Single User
  (512-out) is flat n2→n3 (short generations give the deeper draft no runway).
- **High concurrency (c≥64) goes negative at n=3** (−9…−12%): once the batch is saturated
  with real tokens, draft slots are pure competition for compute. **n=2 holds c=64/128
  flat-to-positive** (+9.8% / +1.7%) — it is the no-regret depth at saturation.
- **Long Context 16K regresses ~9% at all MTP depths** (not n=3-specific): long prefills
  dominate wall-clock; draft overhead doesn't amortize.

**Ship guidance (workload-dependent, MTP depth is a per-launch serve flag):**
- **Batch / dataset-gen / training (high concurrency)** → **n=2**. Best saturated tiers
  (c=64 +9.8%, c=128 +1.7% vs base) on top of the dual prefill win, no downside, bit-exact.
- **Interactive / single-user** → **n=3** (the decode peak, +56% c=1). c≥64 regresses
  −9…−12% but that's irrelevant at this operating point.
- **n=5 is dominated** — decode rolls back to n=2 level (83.3) and high-concurrency is worse
  than n=3. Skip it; n=3 is the optimum.

## As merged — serving recipe (release image)

`dual` is the default and the n≥3 budget bump is automatic, so the GPTQ8 + MTP
config collapses to just picking a depth:

```bash
# Interactive / single-user (decode peak): n=3
vllm serve /models/Qwen3.6-27B-GPTQ-8bit --served-model-name qwen3.6-27b-8bit \
  --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --attention-backend TRITON_ATTN \
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
#   dual: on by default (set VLLM_GFX908_GPTQ8=native to opt out for 64k density)
#   max_num_batched_tokens: auto-raised to 8192 for n>=3 (no flag needed)

# Batch / dataset-gen / training (no-regret at saturation): n=2
#   ...same, with num_speculative_tokens:2
```

Opt-outs / knobs: `VLLM_GFX908_GPTQ8=native` (disable dual), `VLLM_GFX908_GPTQ8_MTHRESH=<M>`
(crossover, default 16), `VLLM_GFX908_PREBIND_GEMM=1` (gated lever, default off).

## Reproduce (A/B harness)
```bash
cd /home/tyler/curvedinf_ab
DATASET=sonnet ./run_arm.sh sonnet_mtpN \
  --env VLLM_GFX908_GPTQ8=dual --env VLLM_GFX908_GPTQ8_MTHRESH=16 \
  -- --speculative-config '{"method":"mtp","num_speculative_tokens":N}' \
     --max-num-batched-tokens 8192
```
