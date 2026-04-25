# Stage 5h — Inductor escape hatch via torch custom op (BIG WIN)

**Date:** 2026-04-25
**Result:** TPOT 13.82 → **11.62 ms** = **−15.9% TPOT / +18.9% throughput** (72.35 → 86.06 tok/s)
**On top of Stage 5g** (which itself was +40.1% over round-1 baseline).
**Combined round-1 → round-3:** 19.36 → 11.62 ms = **−40.0% TPOT / +66.6% throughput** (51.65 → 86.06 tok/s)

## Background

Stage 5g wired LLMM1 + wvSplitK into `rocm_unquantized_gemm_gfx908`. Verified in `stage5g_results.md` that the dispatch fires for the **MoE block** (`gate`, `shared_expert`, `router`) and **lm_head**. But the dispatch trace also showed the **QKV/QKVZ/o_proj** shapes (`n=1 m=3072 k=2048`, `n=1 m=2048 k=1024`, `n=1 m=2560 k=2048`) **missing** — those layers are inside the `@support_torch_compile` model forward, and inductor inlined our Python wrapper down to `aten::mm` → rocBLAS at runtime.

## Patch

`vllm/model_executor/layers/utils.py` — split `rocm_unquantized_gemm_gfx908` into two pieces and registered the body as a torch custom op:

```python
def rocm_unquantized_gemm_gfx908_impl(x, weight, bias=None) -> torch.Tensor:
    """Existing dispatch body (LLMM1 / wvSplitK / AITER / F.linear)."""
    ...

def rocm_unquantized_gemm_gfx908(layer, x, weight, bias=None) -> torch.Tensor:
    """Thin wrapper that goes through the custom op so inductor sees one node."""
    return torch.ops.vllm.rocm_unquantized_gemm_gfx908(x, weight, bias)

direct_register_custom_op(
    op_name="rocm_unquantized_gemm_gfx908",
    op_func=rocm_unquantized_gemm_gfx908_impl,
    fake_impl=rocm_unquantized_gemm_fake,  # output shape is identical
)
```

`dispatch_unquantized_gemm()` is unchanged — it still returns `rocm_unquantized_gemm_gfx908`, which now routes through the custom op.

This is the same pattern vLLM uses for the platform-default `rocm_unquantized_gemm` at lines 228-241 — we just mirror it for the gfx908 path.

## Why this works

When a `Linear` layer is called inside a `@support_torch_compile`-wrapped model forward, inductor traces the entire forward into a single FX graph. Without the custom-op wrapping, inductor sees our Python branches (`if n == 1 and m % 4 == 0 ...`) and lowers the trailing `F.linear` call to `extern_kernels.mm` (rocBLAS) at compile time — bypassing the runtime dispatch entirely.

With `direct_register_custom_op`, the dispatch becomes a single opaque node in the FX graph. Inductor cannot inline through it. At runtime, the custom op runs eagerly with all its branching, picking LLMM1 / wvSplitK / etc. based on the actual tensor shape.

## End-to-end measurement

Container: `vllm-rocm-gfx908:latest` + Stage 5g + Stage 5h overlays. Workload: Qwen3.6-35B-A3B-GPTQ-8bit, TP=4, dtype half, c=1, 256-tok decode, FULL_AND_PIECEWISE cudagraph, TRITON_ATTN backend.

| Run | Tokens | Wall (s) | tok/s | TPOT (ms) |
|-----|--------|----------|-------|-----------|
| 1   | 256    | 2.975    | 86.06 | 11.62     |
| 2   | 256    | 2.974    | 86.09 | 11.62     |
| 3   | 256    | 2.977    | 86.00 | 11.63     |
| **median** |  |     | **86.06** | **11.62** |

Spread: ±0.005 ms (the cleanest 3-run set in the project). Pre-bench coherence PASS (4/4 prompts). Post-bench coherence PASS (4/4 prompts).

## Dispatch verification — new shapes now firing

From the `VERIFY_DISPATCH=1` run, all three previously-missing QKV/QKVZ/o_proj shapes now fire through LLMM1:

```
[LLMM1] n=1 m=3072 k=2048   ← QKV / QKVZ          (NEW in 5h)
[LLMM1] n=1 m=2048 k=1024   ← o_proj              (NEW in 5h)
[LLMM1] n=1 m=2560 k=2048   ← per-layer QKV variant  (NEW in 5h)
[LLMM1] n=1 m=2048 k=128    ← shared_expert.down_proj   (already 5g)
[LLMM1] n=1 m=256  k=2048   ← router / gate_up_proj     (already 5g)
[LLMM1] n=1 m=62080 k=2048  ← lm_head                   (already 5g)
[LLMM1] n=1 m=16   k=2048   ← lin_attn ba              (already 5g)
```

Plus the matching `wvSplitK n=2/n=4` variants (cudagraph capture batches) for every shape.

## Files

- `vllm/model_executor/layers/utils.py:333-440` — the split + registration
- `docs/mi100_decode_opt/scripts/test_stage5_baseline.sh` — pre-flight cleanup; `VERIFY_DISPATCH=1` env opts in to stderr dispatch prints + shape probe (off by default in prod runs).

## Why this didn't show up sooner

Round-2 stage5g_results explicitly noted "would require either disabling inductor for those layers or registering wvSplitK as an op inductor knows about" and estimated **~700 µs / step (3.6 % TPOT)** of headroom. The actual realized win is **2.20 ms / step (15.9 % TPOT)** — *4×* the estimate. The estimate was based on the rocBLAS launcher floor for those shapes; the actual win includes the entire ~10× speedup of LLMM1 vs rocBLAS for M=1.

## Decision

**SHIPPED.** Same `mi100-optimized` branch as Stage 5g. To be combined with Stage 5f (TunableOp) on the residual rocBLAS calls (M ≥ 8 fallthrough).
