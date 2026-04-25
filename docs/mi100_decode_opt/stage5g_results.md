# Stage 5g — LLMM1 + wvSplitK skinny-GEMM dispatch on gfx908 (BIG WIN)

**Date:** 2026-04-25
**Result:** TPOT 19.36 → 13.82 ms = **−28.6% TPOT / +40.1% throughput** (51.65 → 72.35 tok/s)

## Background

Inspired by larkinwc/vllm-gfx908#4 (which fixed the C++ compile guard for `wvSplitK`/`LLMM1` on gfx908). Our base image already had the C++ fix; what was missing was the dispatch path.

`vllm/model_executor/layers/utils.py:rocm_unquantized_gemm_impl` already dispatches to `wvSplitK` and `LLMM1` for the right shape conditions, but its `use_skinny` gate is `(on_gfx9() or on_gfx1x())`. **`on_gfx9()` excludes gfx908** (`rocm.py:190` lists only gfx90a/gfx942/gfx950). So the standard impl never picks `wvSplitK` for us.

Our gfx908-specific dispatch (`rocm_unquantized_gemm_gfx908`, Stage 3) bypassed the standard impl entirely to enable inductor fusion on the `F.linear` fallback. That meant we were leaving `wvSplitK`/`LLMM1` on the table.

## Microbench (gfx908 single-GPU, eager)

| Shape (M, N, K)         | rocBLAS µs | wvSplitK µs | LLMM1 µs | Best vs rocBLAS |
|-------------------------|-----------:|-------------:|---------:|----------------:|
| (1, 3072, 2048) QKV/QKVZ | 21.19      | 9.77         | 10.83    | 2.17×           |
| (1, 2048, 1024) o_proj   | 15.88      | 7.61         | 7.23     | 2.20×           |
| (1, 2048, 128) shared dn | 15.82      | 7.49         | 7.39     | 2.14×           |
| (1, 256, 2048) router/gu | 49.68      | 7.56         | 7.41     | **6.71×**       |
| (1, 1, 2048) shared_gate | 19.21      | 7.84         | —        | 2.45×           |
| (1, 16, 2048) lin_attn ba| 20.21      | 7.73         | 7.58     | 2.67×           |
| (1, 62080, 2048) lm_head | 464.19     | 295.72       | 244.33   | 1.90×           |

`wvSplitK` and `LLMM1` both have a ~7.5 µs gfx908 floor. rocBLAS hits a brutal ~50 µs floor at the (1, 256, 2048) router shape (where the 6.71× win lives). LLMM1 even beats AITER's BEST_CFG for the lm_head shape (244 vs 267 µs).

## Patch

`vllm/model_executor/layers/utils.py` — extended `rocm_unquantized_gemm_gfx908`:

```
Dispatch priority (fp16/bf16, k % 8 == 0, contiguous weight):
  1. LLMM1   if n==1, m % 4 == 0, k <= 8192, bias is None
  2. wvSplitK if m > 8 and 0 < n <= 4
  3. AITER gemm_a16w16 (whitelisted lm_head shape)  [Stage 2 still in place but
                                                    LLMM1 wins for lm_head now]
  4. F.linear fallback (inductor-fusable)
```

`vllm/platforms/rocm.py:218` — flipped `VLLM_ROCM_USE_SKINNY_GEMM` default to `"1"` (this env is read only by the standard impl path, not strictly required for our patched dispatch but kept set in case any non-bypassed code path checks it).

## End-to-end measurement

Container: `vllm-rocm-gfx908:latest` + Stage 2/3/5g overlays. Workload: Qwen3.6-35B-A3B-GPTQ-8bit, TP=4, dtype half, c=1, 256-tok decode, FULL_AND_PIECEWISE cudagraph, TRITON_ATTN backend.

| Run | Tokens | Wall (s) | tok/s | TPOT (ms) |
|-----|--------|----------|-------|-----------|
| 1   | 256    | 3.538    | 72.35 | 13.82     |
| 2   | 256    | 3.539    | 72.34 | 13.82     |
| 3   | 256    | 3.536    | 72.39 | 13.81     |
| **median** |  |          | **72.35** | **13.82** |

Pre-bench coherence PASS. Post-bench coherence PASS.

## Dispatch verification (from container stderr)

`[LLMM1]` and `[wvSplitK]` prints fire during graph capture for these decode-step shapes (gated on `VLLM_GFX908_DEBUG_DISPATCH=1`):
- LLMM1: `n=1 m=256 k=2048` (router/gate_up_proj), `n=1 m=2048 k=128` (shared_expert.down_proj), `n=1 m=62080 k=2048` (lm_head)
- wvSplitK: `n=2 m=256 k=2048`, `n=4 m=256 k=2048`, `n=2 m=2048 k=128`, `n=4 m=2048 k=128` (prefill batches inside the same shapes)

Notably **not** seen: `n=1 m=3072 k=2048` (full-attn QKV / linear-attn QKVZ) and `n=1 m=2048 k=1024` (attn out). Those go through inductor's compiled subgraph and end up at `aten::mm` → rocBLAS regardless of our dispatch. Estimated additional headroom ~700 µs / step (3.6% TPOT) if we could route them too — would require either disabling inductor for those layers or registering wvSplitK as an op inductor knows about.

## Why this works (when other launch-overhead reductions didn't)

Stage 6a Python-level fusion was neutral because cudagraph already amortizes the *Python launch overhead* to ~1 µs. But the rocBLAS GEMM kernel itself spends ~50 µs of GPU time on the (1, 256, 2048) shape — that's GPU-side compute floor inside the rocBLAS launcher and small-M tile. wvSplitK/LLMM1 *replace the kernel* with one that's 2-7× faster at small M. That's the lever cudagraph can't hide.

## What's next on the table

- Stage 5h (port larkinwc PR #6 adaptive FlashSplitK to triton_unified_attention.py) — claims +35% on Qwen3.5-9B; for our 10/40 full-attn layers, expected ~5-9% additional TPOT.
- Stage 5c (MoE config tune for E=256, N=128, MI100, int8_w8a16) — small expected gain, but free if benchmark_moe.py --tune works on gfx908.
- Inductor escape hatch for QKV/QKVZ/o_proj — get those onto wvSplitK too. Risk vs reward unclear.
