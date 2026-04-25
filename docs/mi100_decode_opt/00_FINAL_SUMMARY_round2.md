# Decode optimization round 2 — gfx908 / Qwen3.6-35B-A3B / 4×MI100

**Date:** 2026-04-25
**Workload:** Qwen3.6-35B-A3B-GPTQ-8bit at TP=4 on 4× MI100 (gfx908)
**Stack:** vLLM v0.19.2rc1+mi100 + AITER, `VLLM_MI100_TORCH_COMPILE=1`, mode-3 + FULL_AND_PIECEWISE, TRITON_ATTN

## Bottom line

| Stage | Status | TPOT (ms) | tok/s | Δ vs round-1 baseline |
|-------|--------|-----------|-------|-----------------------|
| Round-1 baseline (Stages 0–4) | shipped | 19.36 | 51.65 | — |
| **Round-2 Stage 5g (LLMM1+wvSplitK)** | **shipped** | **13.82** | **72.35** | **−28.6 % TPOT / +40.1 % throughput** |

3-run TPOT spread ±0.01 ms. Coherence pre + post both PASS. Single confirmed-shippable patch this round; everything else either neutral or already-incorporated.

## What worked

### Stage 5g — LLMM1 + wvSplitK in our gfx908 dispatch (the headline win)

`vllm/_custom_ops` already exposes `wvSplitK` and `LLMM1` skinny-GEMM kernels (built for gfx908 since larkinwc/vllm-gfx908#4 added `__gfx908__` to the C++ compile guard, which is in our base image). They were never firing because:
1. Standard `rocm_unquantized_gemm_impl` gates them behind `(on_gfx9() or on_gfx1x())` and `_ON_GFX9` excludes gfx908 (`rocm.py:190` lists only gfx90a/942/950).
2. Our gfx908 path (`rocm_unquantized_gemm_gfx908`, Stage 3) bypasses the standard impl entirely.

Patched our path to dispatch:
- LLMM1 if `n==1, m % 4 == 0, k <= 8192, bias is None` (decode-time, output-features divisible by 4)
- wvSplitK if `m > 8 and 0 < n <= 4` (skinny-M small batches)
- AITER for the lm_head whitelist (Stage 2 fallback; LLMM1 is now picked first since it's faster)
- F.linear otherwise (inductor-fusable)

Microbench wins (eager, gfx908 single-GPU): **1.9–6.7× over rocBLAS**. Biggest win at `(1, 256, 2048)` router/gate_up_proj where rocBLAS hits a 50 µs floor and LLMM1 finishes in 7.4 µs.

End-to-end: **TPOT 19.36→13.82 ms** = +40.1 % throughput. Coherence PASS pre+post. Dispatch verified via env-gated stderr prints.

Files:
- `vllm/model_executor/layers/utils.py` — extended `rocm_unquantized_gemm_gfx908`
- `vllm/platforms/rocm.py` — `VLLM_ROCM_USE_SKINNY_GEMM` default `"1"` (no-op for our patched path; flipped for cleanliness)

Inspired by larkinwc/vllm-gfx908#4 (compile-guard fix; already applied to our base) — but the actual dispatch wiring on gfx908 is new in this commit.

Commit: `7db67cd92` on `mi100-optimized`.

Per-stage docs: `stage5g_results.md`.

## What didn't move TPOT (and why)

### Stage 5b — hipBLASLt probe

Microbenched all hot M=1 shapes: hipBLASLt is **1.08× to 4.4× SLOWER** than rocBLAS on gfx908. Don't enable. (Closed permanently.)

### Stage 5e — `VLLM_ROCM_USE_SKINNY_GEMM=1` env flag alone

Flipping the env default did nothing because the standard impl never reaches gfx908 (see Stage 5g rationale). The flag is harmless when set, and Stage 5g made it actually do something (via our patched dispatch).

### Stage 6a — Python-level fusion of MoE block (`gate` + `shared_expert_gate`)

Implemented and verified correct (coherence PASS), but **TPOT delta −0.01 ms = 0.05 % (within noise).** Cudagraph already amortizes per-launch overhead at the Python level to ~1 µs. The 50 µs "launch floor" was always GPU-side kernel cost, not CPU launch overhead. Reverted.

This was the strongest test of the "fewer launches via Python fusion" thesis. It failed; the lever was the kernel itself, not the launch count. Stage 5g's switch from rocBLAS to wvSplitK/LLMM1 *replaces the kernel with a faster one*, which is what cudagraph cannot hide.

### Stage 7 — CAR NaN under cudagraph deep-dive

Larkinwc PR #7 + #10 confirm root cause is **"IPC buffer addresses captured in the graph become stale on replay"** — fundamental HIP runtime / IPC behavior on CDNA1, not source-level fixable. Both PRs sidestep with the same `should_custom_ar()` bypass we already have. Capping investigation here; the 3.4 ms / 16 % TPOT prize is unattainable without a HIP runtime patch. (Closed.)

## What was already in our base image (no work needed)

- **PR #4 (skinny GEMM compile guard)** — `csrc/rocm/skinny_gemms.cu:25` already has `defined(__gfx908__)`. We just had to wire the dispatch (Stage 5g).
- **PR #6 (FlashSplitK adaptive split-K)** — `triton_attn.py:62-96` has `_compute_flash_decoding_splits()`, lines 229-263 enable it on MI100, `triton_unified_attention.py:1198-1220` uses it dynamically. So Stage 5g's 13.82 ms result already includes FlashSplitK benefit.
- **PR #3 (Triton attention tuning, TILE_SIZE=32, etc.)** — `triton_attn.py:51-56` has `_get_mi100_tuned_constants()` returning `(64, 8)`.

## Still on the table

### Stage 5f — TunableOp tuning (small marginal expected)
Wired into `test_stage5_baseline.sh` via `TUNABLEOP=tune|replay` env. Tunes rocBLAS dispatch which would only help our remaining inductor-compiled GEMMs (QKV/QKVZ/o_proj that go through `aten::mm` → rocBLAS). Estimated marginal gain ~1–2 % TPOT.

### Stage 5c — vLLM fused_moe config tune for E=256/N=128/MI100/int8_w8a16
Default config used for our MoE expert kernels; missing `…device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int8_w8a16.json`. Would need to run `benchmarks/kernels/benchmark_moe.py --tune` (Ray-based, may or may not work on gfx908). Estimated marginal gain ~1–2 % TPOT.

### Inductor escape hatch for QKV/QKVZ/o_proj
Currently those layers are inside the inductor compile graph and never hit our wvSplitK dispatch. Routing them too would add ~700 µs / step (3.6 % TPOT) but requires either disabling inductor for those layers or registering wvSplitK as a Torch op inductor knows about. Open-ended.

### CK FA backend (larkinwc PR #19, `ROCM_CK_FA`)
Not present in our base. Reports prefill 1.69–2.20× speedup. Decode benefit unclear, would need C++ build. High effort, decode-irrelevant.

## Test infrastructure carry-overs

All under `docs/mi100_decode_opt/scripts/`:
- `test_stage5_baseline.sh` — primary 3-run TPOT bench (with optional TUNABLEOP=tune|replay)
- `test_stage5a_inductor.sh` — inductor compile cache audit
- `test_stage5d_probe.sh` — runtime shape probe (env-gated)
- `test_stage6a.sh` — gate+shared_expert_gate fusion
- `test_stage7a_car_eager.sh` / `test_stage7a_car_piecewise.sh` — CAR no-bypass with mitigation env vars
- `coherence.sh` — 4-prompt coherence smoke

## Memory recommendations

Update `project_decode_opt_results_2026_04_24.md` (rename to `_round2`?) with:
- Round-2 baseline = 13.82 ms TPOT (51 → 72 tok/s, +40 %)
- wvSplitK/LLMM1 dispatch lives in `rocm_unquantized_gemm_gfx908`
- "wvSplitK assertion crash" memory entry is OBSOLETE — the kernel works fine post-larkinwc#4 + our dispatch
- hipBLASLt is universally slower on gfx908 — don't enable
- CAR fix is not source-fixable; cap further investigation
- Python-level fusion does NOT help on gfx908+cudagraph — kernel-level swaps are the lever
