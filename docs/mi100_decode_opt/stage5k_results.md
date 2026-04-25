# Stage 5k — Inductor fusion audit (NO-SHIP — already optimal)

**Date:** 2026-04-25
**Result:** RMSNorm + residual add ARE fused into a single triton reduction kernel that runs immediately before our `rocm_unquantized_gemm_gfx908` custom-op extern call. Pointwise ops (sigmoid, mul, view) for SiLU-gating-style paths are fused into a separate triton kernel that prepares the gemm input.
**Decision: NO SHIP, NO ACTION.** Inductor is already doing as much fusion as it can given that the gemm is an opaque extern call (which we deliberately made it in Stage 5h).

## Method

1. Launched container with `TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=/host_inductor_trace`.
2. Mounted `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/stage5k_inductor_trace/` as the host trace dir (avoids snap-docker /tmp gotcha).
3. Sent 3 priming chat-completion requests to trace all relevant decode + prefill graphs.
4. Tore down container, grepped `output_code.py` for `rocm_unquantized_gemm_gfx908` and surrounding triton kernels.

Trace artifacts saved to `docs/mi100_decode_opt/stage5k_inductor_trace/torch_compile_debug/run_*-pid_431/torchinductor/model__*_inference_*.*/output_code.py` (~6.4 MB total, 4 ranks × ~50 graphs each).

## Findings

### Pattern 1: linear_attention path (model__0)

```python
# Topologically Sorted Source Nodes: [float_1, add, , rocm_unquantized_gemm_gfx908],
# Original ATen: [aten._to_copy, aten.add, vllm_ir.rms_norm, vllm.rocm_unquantized_gemm_gfx908]
triton_red_fused_1.run(arg1_1, arg0_1, buf1, buf4, ...)   # RMSNorm + residual add (FUSED)
buf2 = torch.ops.vllm.rocm_unquantized_gemm_gfx908.default(buf1, arg3_1, bias=None)
```

The triton kernel `triton_red_fused_1` is a single reduction kernel that takes the input + the residual, computes RMSNorm, and writes both the normalized output (for the gemm) and the updated residual (for downstream use). One kernel launch, two outputs, no intermediate global-memory round-trip.

### Pattern 2: full_attention + MoE block path (model__40)

```python
# Topologically Sorted Source Nodes: [view, sigmoid, mul, rocm_unquantized_gemm_gfx908],
# Original ATen: [aten.view, aten.sigmoid, aten.mul, vllm.rocm_unquantized_gemm_gfx908]
triton_poi_fused_mul_rocm_unquantized_gemm_gfx908_sigmoid_view_0.run(...)   # SiLU-gating (FUSED)
buf0 = torch.ops.vllm.rocm_unquantized_gemm_gfx908.default(...)

# Source Nodes: [_to_copy, add, moe_forward_shared, rms_norm], Original ATen: [aten._to_copy, aten.add, vllm.moe_forward_shared, vllm_ir.rms_norm]
triton_red_fused__to_copy_add_moe_forward_shared_rms_norm_1.run(...)       # Residual + RMSNorm + MoE setup (FUSED)
torch.ops.vllm.moe_forward_shared.default(...)

# Source Nodes: [add, all_reduce], Original ATen: [aten.add, vllm.all_reduce]
triton_poi_fused_add_all_reduce_2.run(...)                                 # Add + all-reduce setup (FUSED)
torch.ops.vllm.all_reduce.default(...)
```

The kernel **names literally include the names of the source ops** they were fused from (`triton_poi_fused_mul_rocm_unquantized_gemm_gfx908_sigmoid_view_0` etc.) — inductor topologically clusters these source nodes into a single triton kernel even though the actual gemm/all-reduce/MoE compute is opaque.

### What's not fused — and why it doesn't matter

The custom-op extern calls (`rocm_unquantized_gemm_gfx908`, `all_reduce`, `moe_forward_shared`) are **not** themselves fused into the triton kernels — they remain as separate Python-level extern calls. This is a fundamental property of `direct_register_custom_op`-registered ops: they're opaque to the inductor lowering pass (that's the entire point — Stage 5h needed this opacity to keep our LLMM1 dispatch from being inlined into rocBLAS).

The theoretical lost-fusion cost is one extra global-memory round-trip per gemm (the fused norm kernel writes its output to HBM; the gemm extern call reads it back). For the 2048-element norm output that's ~4 KB per gemm. Versus the gemm's own HBM traffic of ~MB per call, this is < 0.1% overhead.

Plus, **everything is captured in a cudagraph**. Per-launch dispatcher overhead drops to sub-µs in graph-replay mode, so having N+1 kernel launches vs N kernel launches is essentially free at runtime.

## Decision

- ✅ RMSNorm + residual add: fused into one reduction kernel.
- ✅ SiLU-gating (view + sigmoid + mul): fused into one pointwise kernel.
- ✅ MoE-prep (residual add + RMSNorm + the moe_forward_shared op's output buffers): fused into one reduction kernel.
- ✅ Post-all-reduce add: fused into one pointwise kernel.
- ❌ Custom-op extern calls (gemm, moe, all_reduce) remain as separate calls — by design.

**No code change is warranted.** The remaining fusion gap is theoretical (~4 KB/gemm of redundant HBM traffic) and dwarfed by everything else on the critical path.

## What this rules out

- Item B in `round4_candidates.md` (fusion audit + potential restructuring): **closed**. The hypothetical "100-500 µs / step (~0.5–2.5%)" estimate was based on the assumption that norm/add weren't fused. They are.
- Any future "fuse X into the gemm" lever: would require either (a) breaking the custom-op opacity (would re-introduce the inductor-inlining problem 5h fixed), or (b) writing a new fused gemm kernel that includes norm/add inside. Option (b) is a multi-week effort with bounded upside.

## Trace dir

Kept at `docs/mi100_decode_opt/stage5k_inductor_trace/` (6.4 MB). Useful reference for any future inductor-pass investigation. Should be `.gitignore`d before committing — the trace contains paths and PIDs that change per run.
