# Stage 3 — Custom-op bypass for inductor fusion — RESULTS

**Date:** 2026-04-24
**Model:** Qwen3.6-35B-A3B-GPTQ-8bit, TP=4, dtype half, mode-3 + FULL_AND_PIECEWISE

## What changed (on top of Stage 2)

`vllm/model_executor/layers/utils.py`:
- New `rocm_unquantized_gemm_gfx908(layer, x, weight, bias)` — inline dispatch path.
- `dispatch_unquantized_gemm()` returns `rocm_unquantized_gemm_gfx908` when `on_gfx908()` is True.
- For shapes matching `use_aiter_triton_gemm` (lm_head 62080×2048): call AITER `gemm_a16w16` directly, with the `_AITER_GEMM_M1_BEST_CFG` for that specific shape.
- For all other shapes: call `torch.nn.functional.linear(x, weight, bias)` directly.

The original `rocm_unquantized_gemm` path goes:
```
F.linear(layer, x, weight, bias)
  → return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)
    → opaque op call
      → rocm_unquantized_gemm_impl(x, weight, bias)
        → eventually returns F.linear (on gfx908 for non-AITER shapes)
```

The new gfx908 path goes directly:
```
rocm_unquantized_gemm_gfx908(layer, x, weight, bias)
  → return torch.nn.functional.linear(x, weight, bias)
```

## Hypothesis

Bypassing the `torch.ops.vllm.rocm_unquantized_gemm` custom op should:
1. Remove per-call op-dispatch overhead (~few μs × 184 calls/step = ~1 ms = 5%)
2. Let inductor inline F.linear → aten::mm and fuse with adjacent ops (norms, residuals, activations)

## Results

```
run 1: 256 tok in 4.925s = 51.98 tok/s, TPOT=19.24 ms
run 2: 256 tok in 4.925s = 51.98 tok/s, TPOT=19.24 ms
run 3: 256 tok in 4.923s = 52.00 tok/s, TPOT=19.23 ms
```

Stage 3 mean: **51.99 tok/s, TPOT 19.24 ms**
Stage 2:      51.72 tok/s, TPOT 19.33 ms
Baseline:     ~51 tok/s,    TPOT ~19.5 ms

**Stage 3 delta vs Stage 2: -0.10 ms TPOT (-0.5%)**
**Stage 2+3 cumulative vs baseline: -0.27 ms TPOT (-1.4%) / +1.0 tok/s (+1.96%)**

Coherence pre + post: **PASS** both.
AITER dispatch confirmed firing 7128 times (lm_head only).

## Interpretation

The custom-op bypass yielded a real but small win (~0.5%). The effective custom-op overhead on gfx908 is ~70 μs/step / 184 calls ≈ 0.4 μs per call, much smaller than my ~5 μs hypothesis.

Inductor fusion of F.linear with adjacent ops did NOT materialize as a big win — likely because the GEMM lowers to `_extern_kernels.mm` (rocBLAS) anyway, and inductor cannot fuse around an external kernel call. The fusion benefit is only real for things inductor can codegen itself (pointwise ops between ops).

## What's actually bottlenecking decode

Stage 0 profile + microbench data point to **per-launch overhead × many small GEMMs** as the dominant cost. There are ~184 unquantized linear calls per decode step, each at the rocBLAS ~15-50 μs floor for M=1 small-N shapes. The total ~9.3 ms is just `184 × ~50 μs`. Reducing per-call overhead by 0.4 μs (Stage 3) saves only 0.07 ms.

The bigger lever is **fewer kernel launches**. That requires either:
- Model-architecture-level fusion (merge several adjacent linears into a single bigger one — but the fused `in_proj_qkvz` shows the architecture is already doing this where it makes sense)
- Pre-compiled CUDA graph that batches the launches (FULL_AND_PIECEWISE captures the model body, but launches are still serial within the graph — capture saves Python overhead, not GPU overhead)
- True inductor fusion of GEMMs (epilogue fusion / persistent kernels) — inductor's GEMM epilogue support is improving but may not work on gfx908

## Decision

**Keep the patch.** +1.4% cumulative TPOT improvement, zero coherence drift. Effort low, blast radius contained to gfx908.

## Next: Stage 4 — vLLM fusion passes

The compile config currently has these vllm fusion passes DISABLED:
- `fuse_norm_quant: False`
- `fuse_act_quant: False`
- `fuse_attn_quant: False`
- `enable_sp: False`  (sequence parallelism)
- `fuse_gemm_comms: False`  (fuse GEMM + AR)
- `fuse_allreduce_rms: False`  (fuse AR + RMSNorm — known winner)
- `fuse_act_padding: False`
- `fuse_rope_kvcache: False`

`fuse_allreduce_rms` and `fuse_gemm_comms` are the biggest potential wins. Test enabling them via `--compilation-config`.

## Files modified (Stage 2 + 3)

- `/home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py` — added `rocm_unquantized_gemm_gfx908`, modified `dispatch_unquantized_gemm`
- `/home/tyler/vllm-gfx908/vllm/platforms/rocm.py:213` — env var default
- `/home/tyler/vllm-gfx908/vllm/distributed/device_communicators/custom_all_reduce.py` — host's f86fed94f bypass placement
- `/tmp/decode_opt/test_stage3.sh` — test driver
- `/tmp/decode_opt/stage3_results.md` — this file
