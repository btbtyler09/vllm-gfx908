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

### C — Cudagraph capture range pruning
- **Where:** vllm captures M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} cudagraph shapes by default. Many of these are never hit in our workload (we mostly run c=1–32). Pruning trims warmup (already ~11 min) and reduces dispatcher branch count per step.
- **Method:** look at `_get_default_cudagraph_capture_sizes()` in `vllm/v1/worker/`; reduce to {1, 2, 4, 8, 16, 32, 64} via `--compilation-config '{...,"capture_sizes":[...]}`.
- **Expected:** small per-step win (~0.05 ms), bigger startup savings.
- **Effort:** 1 hr.

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

### K — Custom-op fast path for high-n GEMMs (c=64+ regression fix)
- **Where:** `vllm/model_executor/layers/utils.py:rocm_unquantized_gemm_gfx908_impl` — currently runs the custom op for ALL calls, even when LLMM1/wvSplitK can't help (n>4) and we fall back to F.linear.
- **Why:** Round-3 Stage 5h shipped the custom-op inductor escape hatch (+18.9% TPOT at c=1) but introduced a regression at c=64 (−4%) and c=128 (−9.3%). The opaque custom-op extern call breaks inductor's ability to inline `F.linear` to `aten::mm` cleanly when the dispatch falls through.
- **Fix:** Hoist the n>4 check up to the wrapper level so the custom op is only invoked when LLMM1/wvSplitK can fire. For the high-n cases, return F.linear directly so inductor can inline it.
- **Effort:** ~30 lines, half a day including bench validation.
- **Expected:** restore round-2 perf at c=64+128 while keeping all round-3 c=1–32 wins. Net: a small absolute win at c=64+128 (~5-9% recovery), no change at c=1–32.
