# Stage 4 — vLLM compilation fusion passes — RESULTS

**Date:** 2026-04-24
**Stack:** vllm-rocm-gfx908:latest, TP=4, dtype half, mode-3 + FULL_AND_PIECEWISE

## Stage 4a — `fuse_allreduce_rms` — BLOCKED

Tried adding `pass_config: {fuse_allreduce_rms: true}` to compilation-config.

**Result: container crashed during cudagraph capture.** Worker raised:
```
RuntimeError: Worker failed with error 'name 'AllReduceFusionPass' is not defined'
```

**Root cause** in `vllm/compilation/passes/pass_manager.py:38`:
```python
if current_platform.is_cuda():
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass
    from .fusion.collective_fusion import AsyncTPPass
    from .fusion.minimax_qk_norm_fusion import MiniMaxQKNormPass
```

Three high-value passes (`AllReduceFusionPass`, `AsyncTPPass` (used for `fuse_gemm_comms`), `MiniMaxQKNormPass`) are gated to NVIDIA-only. Not importable on ROCm. The `pass_config.fuse_allreduce_rms` flag exists but flips a no-op switch that crashes when actually triggered.

**To unlock these on gfx908 would require:**
1. Patching the `is_cuda()` gate to `is_cuda_alike()` so it imports on ROCm too
2. Patching the pass implementation to use ROCm-compatible all-reduce kernels (the pass likely matches against NVIDIA-specific torch ops or expects NVIDIA-specific custom kernels for the fused output)

Step 2 is the hard part. Realistically not achievable as a quick patch — would need a custom AllReduceRMS fused kernel for gfx908.

## Stage 4b — `fuse_act_padding` — SMALL REGRESSION

Tried `pass_config: {fuse_act_padding: true}`. This invokes `RocmAiterTritonAddRMSNormPadFusionPass` (imported when `rocm_aiter_ops.is_enabled()` — that's our path).

**Result: container started, pass engaged ("Enabled custom fusions: act_padding"), coherence PASS, but TPOT regressed.**

```
run 1: 256 tok in 4.943s = 51.79 tok/s, TPOT=19.31ms
run 2: 256 tok in 4.945s = 51.77 tok/s, TPOT=19.32ms
run 3: 256 tok in 4.944s = 51.78 tok/s, TPOT=19.31ms
```

Stage 4b mean: **51.78 tok/s, TPOT 19.31 ms**
Stage 3:       51.99 tok/s, TPOT 19.24 ms

**Delta: +0.07 ms (+0.4%) — SLIGHT REGRESSION. Drop the pass.**

The fused AITER kernel for AddRMSNormPad on gfx908 is either not well-tuned for our shapes, or doesn't actually engage with our model graph at the right pattern. Either way, no improvement.

## Other passes considered

- `fuse_norm_quant`: model isn't quantized in the right places (only MoE experts; lm_head/linear_attn/full_attn are unquantized) — wouldn't engage
- `fuse_act_quant`: similar — quant fusion not relevant
- `enable_sp` (sequence parallelism): decode workload has seq_len=1, nothing to shard along. Would only help prefill.
- `fuse_rope_kvcache`: requires `SplitCoalescingPass` + `ScatterSplitReplacementPass`. Available on ROCm, but model uses linear_attn (no RoPE) for 30/40 layers; only 10 full-attn layers have RoPE. Marginal upside, didn't pursue.

## Decision

**No Stage 4 patch retained.** The vLLM passes that work on gfx908+AITER don't help our workload; the ones that would help (AllReduce fusion) aren't available on ROCm.

## Recommended final state

Keep **Stage 2 + Stage 3** patches (cumulative +1.4% TPOT):
- `vllm/model_executor/layers/utils.py` — added `rocm_unquantized_gemm_gfx908` (custom-op bypass + AITER lm_head dispatch with BEST_CFG)
- `vllm/platforms/rocm.py:213` — `VLLM_ROCM_USE_AITER_TRITON_GEMM` default `"1"`
- `vllm/distributed/device_communicators/custom_all_reduce.py` — host's `f86fed94f` bypass placement (in `should_custom_ar` not `custom_all_reduce`)

Set `VLLM_ROCM_USE_AITER_TRITON_GEMM=1` via env (rocm.py defaulter is too late for `_aiter_ops.py:1144` class-load caching).

Final TPOT: 19.24 ms = 52.0 tok/s, vs baseline 19.5 ms = 51 tok/s. +1.96% throughput.
