# Round-4 candidates — gfx908 / Qwen3.6-35B-A3B / 4×MI100

After round-3 (Stages 5g/5h/5f, ~+70% throughput vs round-1) the cheap single-patch wins are exhausted. This is the punch-list for next time, ordered by yield × effort.

## Cheap (under 1 day each)

### A — NCCL/RCCL parameter sweep
- **Where the time is:** all-reduce ~3.4 ms / step (~16% of TPOT at c=1) — currently RCCL fallback because CAR is broken on gfx908 cudagraphs (round-2 conclusion).
- **Try (env vars only, no source change):**
  - `NCCL_ALGO=Tree` vs `Ring` (default)
  - `NCCL_PROTO=Simple` vs `LL` vs `LL128`
  - `NCCL_NET_GDR_LEVEL` (PHB / NVL / SYS)
  - `NCCL_P2P_LEVEL`
  - `NCCL_BUFFSIZE`
- **Method:** rotate one var at a time, 3-run TPOT each, keep best. `coherence.sh` after each.
- **Expected:** 0.3–1.0% TPOT if any non-default wins. Worst case: same.
- **Effort:** ~30 min including bench.

### B — Inductor fusion audit around the new custom op
- **Question:** does inductor fuse the surrounding `RMSNorm` + residual `add_` into the kernel that *calls* `torch.ops.vllm.rocm_unquantized_gemm_gfx908`? If yes, no work needed — confirm and stop. If no, we may need to add a `@torch.library.register_fake_tensor_with_meta` or restructure the op signature.
- **Method:** export the inductor cache for the patched container (`TORCH_COMPILE_DEBUG=1`, `TORCHINDUCTOR_TRACE=/tmp/round4_inductor`) and grep `output_code.py` for fused triton kernels referencing `rocm_unquantized_gemm_gfx908` — vs the M=1/4 GEMMs sitting alone as extern calls.
- **Expected:** if not fused, pulling in surrounding norms saves 100–500 µs / step (~0.5–2.5%).
- **Effort:** 1–2 hr audit, 1–4 hr fix if needed.

### C — Cudagraph capture range pruning — **ATTEMPTED 2026-04-26, REVERTED (marginal regressions, startup not a priority)**
- **Where:** Default capture ladder is `[1,2,4] + range(8,256,8) + range(256,max,16)` = ~50 sizes. We pruned to `[1,2,4,8,16,32,64,128]` in `apply_config_platform_defaults` at `vllm/platforms/rocm.py:778`.
- **Result (round-4 Phase 1 C-only bench, 2026-04-26 on Qwen3.6-35B-A3B-GPTQ-8bit, 4×MI100 TP=4):** marginal regressions on two tiers:
  - Short Context Throughput (c=16): tps **-3.42%** (455.66 → 440.08), TPOT +0.43%
  - Concurrency Scaling c=2: TPOT **+1.15%** (13.96 → 14.12 ms), tps -0.85%
  - All other 10 tiers within ±0.5% (decode c=1 even slightly better at 91.63 vs 91.17 tok/s)
- **Root cause hypothesis:** padding-up at non-captured intermediate sizes. Default has [16,24,32,40,...] — actual decode batches of 17-23 pad to 24 (waste 1-7 slots). With our prune [16,32,...], 17-31 pads to 32 (waste 1-15 slots). Short Context Throughput at c=16 likely sees more variance in actual batch size because of prefill mixing with decode, which makes padding waste matter more.
- **Verdict:** the regressions could be single-run variance (1-3% on one bench run), but per "no regression across the test map" we reverted. Startup time savings (~10 min) and ~0.5% c=1 TPOT improvement weren't worth even the variance risk.
- **If retried later:** safer prune is `[1,2,4] + range(8, 128+1, 8) = [1,2,4,8,16,24,32,...,128]` = 19 sizes. Drops only the 256-512 range. ~60% fewer captures than default, no padding-up cost on intermediate sizes. Worth a try only if startup time becomes a stated objective.
- **Bench artifact:** `~/mi100-llm-testing/Model_Reports/benchmark_Qwen3.6-35B-A3B-GPTQ-8bit_v0.20_round4_phase1_Conly.md`

### D — MoE block torch.compile wrap
- **Where:** the MoE block (router + shared_expert + gate_up_proj/down_proj) currently runs in **eager Python** (Stage 5g notes "the MoE block is NOT in graph"). Cudagraph captures around it but Python launch overhead survives.
- **Risk:** wrapping in `@support_torch_compile` could break our LLMM1 dispatch (we'd need the custom-op pattern from Stage 5h to extend correctly to MoE expert calls).
- **Method:** add a `@maybe_compile`-style wrapper around the model's `MoEBlock.forward`. Verify dispatch trace still shows `[LLMM1]` firing for `n=1 m=256 k=2048` (router/gate_up_proj).
- **Expected:** 0.2–0.8% TPOT (Python launch overhead is ~1 µs / op × tens of ops in the MoE block).
- **Effort:** 2–4 hr including the dispatch verification.

## Medium (1–3 days each)

### E — Custom CAR replacement for gfx908 cudagraphs
- **Where:** the canonical CAR is unfixable from source — root cause is HIP IPC pointers going stale on cudagraph replay (CDNA1-specific runtime behavior, confirmed by larkinwc PR #7+#10). RCCL fallback costs 3.4 ms / step. **This is the single biggest remaining lever (~16% TPOT).**
- **Approach:** write a minimal point-to-point ring all-reduce in HIP that:
  - Uses fresh IPC handles per cudagraph capture (re-exchanged in worker startup, not embedded in graph)
  - OR uses XGMI peer copies via `hipMemcpyPeerAsync` with a fixed handshake buffer that's stable across replays
- **Risk:** new C++ code, must coexist with existing CAR for non-graph paths.
- **Expected:** 2–3.5 ms / step (~10–16% TPOT).
- **Effort:** 2–3 days including correctness validation.

### F — Custom gfx908 MFMA fmoe expert kernel
- **Where:** `fused_moe_kernel` (Triton) is ~1.8 ms / step at c=1. The current kernel is CK-derived and not MI100-tuned.
- **Approach:** write a tight gfx908-specific MFMA fused-MoE kernel. Use `v_mfma_f32_16x16x16_f16` (CDNA1-supported), avoid `v_pk_mul_f32` (gfx90a+). LDS budget ~64 KB.
- **Expected:** 0.5–1.0 ms / step (~2.5–5% TPOT).
- **Effort:** 2–3 days including JSON tune output for E=256/N=128/cu=120.

### G — Custom_all_reduce buffer pinning across replays
- **Approach less risky than E:** patch CAR's IPC handle exchange to happen *outside* the cudagraph capture region. Re-exchange handles after every capture cycle. May or may not work depending on HIP runtime.
- **Effort:** 1–2 days; high uncertainty.

## Long shots / research

### H — Speculative decoding
- Different feature class entirely. Multi-week implementation but potentially 2–3× speedup at high acceptance rate. Out of scope for "small fixes" rounds.

### I — CK FA on gfx908
- Pre-evaluated (memory: `project_ck_fa_gfx908_evaluation.md`). v3 pipeline can't compile (uses `v_pk_mul_f32`); non-v3 untested. Narrow upside post-FlashSplitK; low priority.

### J — vLLM v1 sampler optimization
- Memory mentions "generic sampler" as a bottleneck. Currently runs once per token. ~0.1–0.3% TPOT lever, low priority unless we hit a wall on everything else.

## Summary

If round-4 happens, suggested order: **A (cheap probe) → B (audit) → E (the prize)**. A and B are confirmable in a single day; E is the only remaining lever big enough to warrant multi-day effort.


---

## Added 2026-04-25 (post-round-3 ship)

### K — Custom-op fast path for high-n GEMMs (c=64+ regression fix) — **ATTEMPTED 2026-04-26, FAILED, REVERTED**
- **Where:** `vllm/model_executor/layers/utils.py:rocm_unquantized_gemm_gfx908_impl` — currently runs the custom op for ALL calls, even when LLMM1/wvSplitK can't help (n>4) and we fall back to F.linear.
- **Why:** Round-3 Stage 5h shipped the custom-op inductor escape hatch (+18.9% TPOT at c=1) but introduced a regression at c=64 (−4%) and c=128 (−9.3%). The opaque custom-op extern call breaks inductor's ability to inline `F.linear` to `aten::mm` cleanly when the dispatch falls through.
- **Fix attempted:** Hoist the n>4 check up to the wrapper level (`rocm_unquantized_gemm_gfx908`). Wrapper computes `n = x.numel() // x.size(-1)`, checks LLMM1/wvSplitK/AITER conditions, and either calls the opaque custom op (when skinny path can fire) or `F.linear` directly (when not, so inductor can inline to `aten::mm`).
- **Result (round-4 Phase 1 K bench, 2026-04-26 on Qwen3.6-35B-A3B-GPTQ-8bit, 4×MI100 TP=4):** REGRESSED c=1 by **+17% TPOT** (12.94 ms vs round-3 baseline 11.04 ms — Decode Stress Test went 10.95 → 12.87 ms). c=2/c=4 also regressed +10–12%. c=8 through c=128 flat — and crucially, **c=64/c=128 did NOT recover** (still at round-3 perf). The hoist made everything worse and fixed nothing.
- **Root cause:** the Python-level shape conditional (`if n == 1 and m % 4 == 0 ...`) breaks inductor's clean trace through the wrapper. Inductor can't fold it (`x.numel()` looks dynamic), so it inserts a graph break and runs the wrapper in eager Python on every GEMM call. Per-call Python overhead (~10 µs) × ~129 GEMM calls per token ≈ 1.3 ms/token — matches the observed ~1.9 ms regression at c=1.
- **Why c=64+ didn't recover either:** the per-layer dispatch still went through the custom op for those shapes too (the wrapper's conditional structure routes large-n shapes through F.linear, but the *eager-Python* execution of the wrapper added overhead that swamped any inductor benefit on the F.linear path).
- **Verdict: do NOT retry this approach.** The "right fix" exists but is invasive and doesn't help c=1 — see lever **L** below.
- **Bench artifact:** `~/mi100-llm-testing/Model_Reports/benchmark_Qwen3.6-35B-A3B-GPTQ-8bit_v0.20_round4_phase1_K_v0.19_round3.md`

### L — Per-layer dispatch closure binding (the "right fix" for K's intent)
- **Why this exists:** lever K above failed because runtime shape conditionals in the wrapper break inductor tracing. The right structural fix is to make the dispatch decision ONCE per layer at construction time (when shapes are static and the cost is amortized over millions of forward calls), not on every call.
- **Where the binding happens:** `vllm/model_executor/layers/linear.py:UnquantizedLinearMethod.apply()` at line 228 currently calls `dispatch_unquantized_gemm()(layer, x, layer.weight, bias)` per forward. Two complementary edit points:
  - `process_weights_after_loading()` (line 214): inspect `layer.weight.shape`, dtype, and contiguity. Pre-decide which path the dispatch would take (LLMM1 / wvSplitK / AITER / F.linear). Cache the chosen callable on the layer as `layer._gfx908_gemm_op`.
  - `apply()` (line 220): fast-path `if hasattr(layer, "_gfx908_gemm_op"): return layer._gfx908_gemm_op(x, layer.weight, bias)` — single op call, no conditional.
- **Pre-bound callables (one of):**
  - `_call_llmm1` — wraps `ops.LLMM1(weight, x_view, 4)` + reshape (for n==1, m%4==0, k≤8192, no bias)
  - `_call_wvsplitk` — wraps `ops.wvSplitK(weight, x_view, cu_count, bias)` + reshape (for m>8 layers; n must be ≤4 at call time, but in practice the caller controls this)
  - `_call_aiter` — wraps `gemm_a16w16(...)` for the lm_head whitelist
  - `_call_flinear` — wraps `F.linear(x, weight, bias)` (the high-N path)
  - Each closure could itself be `direct_register_custom_op`-registered (preserves the inductor-opaque guarantee for the dispatch ops, while letting the F.linear closure remain inductor-inlinable)
- **Important subtlety:** wvSplitK's eligibility depends on BOTH static `m > 8` AND dynamic `0 < n ≤ 4`. Per-layer binding fixes m statically, but n still varies per call. Two options:
  1. Bind `_call_wvsplitk` for `m > 8` layers, accept that we lose dispatch when n>4 (i.e., wvSplitK forwards to its own internal F.linear fallback — which is fine, it's a single op call, no Python conditional)
  2. Bind a "decode-vs-prefill" pair: use cudagraph capture size as a static signal (every captured size has a static n), and pre-build per-capture-size dispatch callables. Cleaner but more invasive.
  - Option 1 is the recommended starting point.
- **Effort:** ~1 day end-to-end. Touches `linear.py`, `utils.py`, possibly `vocab_parallel_embedding.py` (lm_head consumer at line 75) and `compressed_tensors/transform/module.py:119,129`. Test with full BenchAndReport.
- **Expected yield:** restore round-2 perf at c=64 (+~4%) and c=128 (+~9%), zero change at c=1 (still hits the same dispatch — just via pre-bound closure instead of runtime conditional).
- **Why this is NOT round-4 priority:** round-4 is targeting c=1 throughput (break 100 tok/s). Lever L only moves c=64/c=128 numbers — useful for high-throughput serving but orthogonal to our goal. Park this until c=64+ recovery becomes a stated objective (e.g., a future "round-4.5: high-concurrency cleanup" if c=128 throughput becomes important to a downstream user).
- **Stop conditions if attempted later:** any c=1 regression > 0.5% (verifies the closure-binding doesn't have its own Python-overhead surprise); coherence 4/4 pre+post.
