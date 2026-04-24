# MI100 Decode Optimization — Qwen3.6-35B-A3B

Investigation: 2026-04-24
Workload: Qwen3.6-35B-A3B-GPTQ-8bit, TP=4 on 4× MI100 (gfx908)
Stack: vLLM v0.19.2rc1+mi100, AITER, mode-3 + FULL_AND_PIECEWISE, TRITON_ATTN

**Start with [00_FINAL_SUMMARY.md](00_FINAL_SUMMARY.md)** for the consolidated findings + bottom line.

Per-stage forensic detail:
- [stage0_profile.md](stage0_profile.md) — Decode-only torch profile + bucketization
- [stage1a_shapes.md](stage1a_shapes.md) — vLLM dispatch path + real shape inventory
- [stage2_results.md](stage2_results.md) — AITER `gemm_a16w16` for lm_head with hand-tuned BEST_CFG
- [stage3_results.md](stage3_results.md) — Custom-op bypass (`rocm_unquantized_gemm_gfx908`)
- [stage4_results.md](stage4_results.md) — vLLM compilation fusion passes (mostly blocked on ROCm)

Reproduction scripts: [scripts/](scripts/)
