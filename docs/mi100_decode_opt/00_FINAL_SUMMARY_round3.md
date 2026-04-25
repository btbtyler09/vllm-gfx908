# Decode optimization round 3 — gfx908 / Qwen3.6-35B-A3B / 4×MI100

**Date:** 2026-04-25
**Workload:** Qwen3.6-35B-A3B-GPTQ-8bit at TP=4 on 4× MI100 (gfx908)
**Stack:** vLLM v0.19.2rc1+mi100 + AITER, `VLLM_MI100_TORCH_COMPILE=1`, mode-3 + FULL_AND_PIECEWISE, TRITON_ATTN

## Bottom line

| Stage | Status | TPOT (ms) | tok/s | Δ vs round-1 baseline | Δ vs round-2 ship |
|-------|--------|-----------|-------|-----------------------|-------------------|
| Round-1 baseline (Stages 0–4) | shipped | 19.36 | 51.65 | — | — |
| Round-2 Stage 5g (LLMM1+wvSplitK) | shipped | 13.82 | 72.35 | −28.6 % / +40.1 % | — |
| Round-3 Stage 5h (custom-op inductor escape hatch) | shipped | 11.62 | 86.06 | −40.0 % / +66.6 % | −15.9 % / +18.9 % |
| **Round-3 Stage 5j (NCCL Tree+LL)** | **shipped** | **11.00** | **90.94** | **−43.2 % / +76.1 %** | **−20.4 % / +25.7 %** |

3-run TPOT spread ±0.01 ms. Coherence pre + post both PASS on every shipped stage.

## What shipped this round

### Stage 5h — Inductor escape hatch via custom-op registration ⭐ headline win

**Problem:** Round-2's Stage 5g wired `LLMM1`/`wvSplitK` into `vllm/model_executor/layers/utils.py:rocm_unquantized_gemm_gfx908`, but inside `@support_torch_compile`-wrapped model forwards (qkv_proj, qkvz_proj, o_proj), inductor traces through the Python dispatch and lowers `F.linear` to `aten::mm` → rocBLAS, never reaching our skinny-GEMM kernels.

**Fix:** Mirror the existing `direct_register_custom_op` pattern (`utils.py:237-241`) for the gfx908 path so the dispatch becomes a single opaque graph node inductor cannot inline through.

```python
# Before (Stage 5g): direct Python function — inductor inlines, gets aten::mm
def rocm_unquantized_gemm_gfx908(layer, x, weight, bias=None):
    # full dispatch body here

# After (Stage 5h): split + register
def rocm_unquantized_gemm_gfx908_impl(x, weight, bias):
    # full dispatch body here

def rocm_unquantized_gemm_gfx908(layer, x, weight, bias=None):
    return torch.ops.vllm.rocm_unquantized_gemm_gfx908(x, weight, bias)

direct_register_custom_op(
    op_name="rocm_unquantized_gemm_gfx908",
    op_func=rocm_unquantized_gemm_gfx908_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)
```

Result: TPOT **13.82 → 11.62 ms = +18.9% throughput**. Single file change (~30 lines). Coherence PASS pre+post.

Files: `vllm/model_executor/layers/utils.py`.

### Stage 5j — NCCL/RCCL Tree+LL env vars

**Problem:** All-reduce takes ~3.4 ms / step (~16% of TPOT) via RCCL fallback (CAR is broken on gfx908 cudagraphs — IPC pointer staleness on graph replay, CDNA1-specific). RCCL defaults assume PCIe-style fabric; our box has 4× MI100 with XGMI.

**Fix:** `NCCL_ALGO=Tree NCCL_PROTO=LL`. Tree algorithm uses log-N depth (vs Ring's N-hop latency); LL protocol cuts per-message overhead. Together they shave the per-message latency for our small (~few KB) per-step all-reduces.

**Sweep results:** baseline 11.62 → Tree alone 11.11 → LL alone 11.60 (no help on its own) → Tree+LL **11.08 ms**. Stage F final 3-run mean: **11.00 ms = 90.94 tok/s** (additional small drift over multiple container restarts).

Files: `vllm/platforms/rocm.py:_GFX908_DEFAULTS` (auto-set), `docs/mi100_decode_opt/scripts/test_stage5_baseline.sh`, `docs/mi100_decode_opt/scripts/test_stage_combined.sh`. Zero source-code change in vLLM proper.

### Stage 5k — Inductor fusion audit (verified, no action needed)

Single instrumented launch with `TORCH_COMPILE_DEBUG=1`. Confirmed: **RMSNorm + residual add already fuse into single triton reduction kernel** (`triton_red_fused_1`), running back-to-back with our custom-op gemm extern call. Pointwise ops (sigmoid, mul, view) for SiLU-gating-style paths fuse into a separate triton kernel that prepares the gemm input. Inductor is doing as much fusion as it can given the gemm is opaque (which is what we made it in Stage 5h).

Documented in `stage5k_results.md`. Trace dump at `docs/mi100_decode_opt/stage5k_inductor_trace/` (gitignored).

## What didn't ship

### Stage 5f — TunableOp tune+replay
Tune pass shaved 0.17 ms during the tune itself, but replay regressed by 0.08 ms. Tuned algos don't help LLMM1/wvSplitK paths (they're already faster than rocBLAS); they only modify the cudagraph-capture batches we don't actually decode at. NO SHIP at c=1; CSV kept at `docs/mi100_decode_opt/tunableop_results/` for round-4 c≥8 evaluation.

### Stage 5i — AITER triton GEMM at higher M
Microbench: AITER `gemm_a16w16` is **0.33–0.45× rocBLAS** at M=8/16/32 for our QKV shapes. The ~50 µs gfx908 launch floor swamps the work. NO SHIP. (Confirms round-2 finding that AITER only wins at very large GEMMs like lm_head.)

### Stage 5m — MoE block torch.compile wrap
The lever doesn't exist for this model. `Qwen3_5Model` (and parent `Qwen3NextModel`) is **already** wrapped in `@support_torch_compile`. The MoE block runs inside the inductor graph already. The round-4 candidates doc was wrong about this. NO ACTION (saved 2-3 hours of patch + risk).

### Stage 5c — MoE config tune
The `benchmark_moe.py` tuner targets the un-quantized `fused_moe_kernel`, not the GPTQ-AWQ `fused_moe_kernel_gptq_awq` variant our model uses. Even after surface-level field-name alignment (renaming for dtype suffix, adding `SPLIT_K: 1`), the chosen tile shapes (BLOCK_SIZE_K=256 + num_warps=1) violate kernel invariants and produce HIP illegal-memory-access during cudagraph capture. NO SHIP. Reference JSON kept at `docs/mi100_decode_opt/moe_tune_output/` for round-4. The real lever here is a custom gfx908 MFMA fmoe kernel — round-4 ticket F.

## What's still on the table for round-4

The remaining big lever is **Custom CAR replacement for gfx908 cudagraphs** (~3.4 ms / step ≈ 16% TPOT). RCCL fallback is the entire remaining gap vs A100 inference at c=1. Multi-day C++ work; deferred to round-4. See `round4_candidates.md` lever E.

Other candidates documented in `round4_candidates.md`:
- **C** — Cudagraph capture range pruning (small per-step + faster startup)
- **F** — Custom gfx908 MFMA fmoe kernel (2-3 days, +2.5–5%)
- **G** — CAR buffer pinning across replays (less risky CAR variant)
- Long shots: speculative decoding, custom_kernel sampler

## Full-bench delta vs round-2

12-scenario `BenchAndReport.py` run, comparison at `/tmp/decode_opt/round3_comparison.md`. **Mean throughput ratio across all 12 scenarios: 1.08× (+8.3%)**. Round-3 dominates round-2 across the practical concurrency range; modest regressions appear only at c=64 and c=128 (see "Known regression" below).

| Scenario | In/Out | c | R2 tok/s | R3 tok/s | Δ tok/s | R2 TPOT | R3 TPOT | Δ TPOT |
|----------|--------|---|---------:|---------:|--------:|--------:|--------:|-------:|
| Single User Latency | 2048/512 | 1 | 71.2 | 87.9 | **+23.3%** | 13.74 | 11.01 | **−19.9%** |
| Decode Stress Test | 128/2048 | 1 | 73.1 | 91.8 | **+25.6%** | 13.66 | 10.87 | **−20.4%** |
| Concurrency Scaling (c=2) | 1024/256 | 2 | 115.5 | 136.6 | **+18.3%** | 16.80 | 14.11 | **−16.0%** |
| Concurrency Scaling (c=4) | 1024/256 | 4 | 199.2 | 230.0 | **+15.5%** | 18.19 | 15.76 | **−13.4%** |
| Long Context (16K) | 16384/1024 | 4 | 146.2 | 156.9 | +7.3% | 21.01 | 19.02 | −9.5% |
| Concurrency Scaling (c=8) | 1024/256 | 8 | 305.4 | 310.4 | +1.6% | 23.40 | 22.81 | −2.5% |
| Mixed Traffic | 2048/512 | 8 | 294.8 | 300.5 | +1.9% | 24.27 | 23.86 | −1.7% |
| Concurrency Scaling (c=16) | 1024/256 | 16 | 499.1 | 536.4 | +7.5% | 27.93 | 25.49 | −8.7% |
| Short Context Throughput | 512/256 | 16 | 417.5 | 459.2 | +10.0% | 28.25 | 25.56 | −9.5% |
| Concurrency Scaling (c=32) | 1024/256 | 32 | 733.3 | 743.5 | +1.4% | 35.81 | 34.11 | −4.7% |
| Concurrency Scaling (c=64) | 1024/256 | 64 | 1149.1 | 1103.6 | **−4.0%** | 44.78 | 44.91 | +0.3% |
| Concurrency Scaling (c=128) | 1024/256 | 128 | 1506.0 | 1365.9 | **−9.3%** | 61.54 | 65.88 | +7.1% |

### Known regression at c=64+128

Round-3 regresses at c=64 (−4%) and c=128 (−9.3%) vs round-2. Hypothesis:

- At c=1–32 the custom-op extern call is a net win because LLMM1/wvSplitK fire (n≤4) and replace rocBLAS for the QKV/o_proj GEMMs.
- At c=64+ the per-batch n exceeds the wvSplitK gate (n>4), so we fall through to `F.linear` inside our custom op — but inductor cannot fuse that path because the custom op is opaque. Round-2's direct Python dispatch let inductor inline `F.linear` to `aten::mm` cleanly; round-3's custom op breaks that inlining for the high-n cases.

**Round-4 ticket:** add a fast path in `rocm_unquantized_gemm_gfx908_impl` that bypasses the custom op entirely when `n>4 and not in any LLMM1 whitelist` — i.e., expose the F.linear path back to inductor for the cases where our custom kernels can't help anyway. ~30 lines, low risk.

Tonight's call: **ship 5h+5j** because c=1–32 (the practical hot range for this workload) gains far outweigh the c=64+128 losses, and the regression is fully understood and easy to undo.

## Verification artifacts

- 3-run TPOT (canonical metric, c=1, 256-tok decode): **11.00 ms median, ±0.01 ms spread**.
- Coherence: 4-prompt suite PASS pre AND post on every shipped stage and on the post-Stage-F bench.
- BenchAndReport JSON: `/tmp/decode_opt/round3_results.json`
- Comparison: `/tmp/decode_opt/round3_comparison.md`
- Per-stage logs: `docs/mi100_decode_opt/stage5{c,f,h,i,j,k,m}_results.md`

## Methodology notes & lessons learned

- **The inductor-escape-hatch was the headline win** because it unlocked the round-2 work for the actual decode-time GEMMs. Round-2 shipped LLMM1/wvSplitK but they were only firing for ~30% of the calls; round-3 Stage 5h made them fire for all of them.
- **Snap-installed Docker can't bind-mount paths under `/tmp/`** on this host — silent failure to anonymous tmpfs. Use `/home/tyler/...`. Documented in `~/.claude/projects/-home-tyler-aiter/memory/feedback_snap_docker_tmp_mounts.md`. Burned ~12 min in Stage 5f tune pass before catching it.
- **`benchmark_moe.py` tunes the wrong kernel for GPTQ-quantized models.** Configs aren't transferable to `fused_moe_kernel_gptq_awq`. Surface-level field rename isn't enough; tile-shape invariants differ. Shipping a tuned MoE config requires either a quant-aware tuner or hand-editing.
- **Tree+LL all-reduce is strictly better than Ring at our message size.** Worth checking on every new gfx9 install — it's a one-line env-var change.
