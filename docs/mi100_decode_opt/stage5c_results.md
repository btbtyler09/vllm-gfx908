# Stage 5c — MoE config tune (NO SHIP — tuner / kernel mismatch)

**Date:** 2026-04-25
**Result:** Tuner output is incompatible with the GPTQ-AWQ MoE kernel path. Loading it produces an HIP illegal memory access during cudagraph capture.
**Decision: NO SHIP, NO ACTION.** Document and move on. The shipped CSV is harmless to keep in `docs/mi100_decode_opt/moe_tune_output/` as a reference for round-4 if a different tuner / a hand-fixed config can be made to work.

## V1 (initial attempt) — Ray module missing

`vllm-rocm-gfx908:latest` doesn't ship `ray`. `benchmark_moe.py` imports it at line 14. Easy fix: `pip install ray[default]` in an ephemeral container.

## V2 (rewrite with ray installed) — runtime errors with int8_w8a16 dtype

`--dtype int8_w8a16` triggers `RuntimeError: expected scalar type Float but found Half` in `scaled_int8_quant`. This is independent of our model; it's a bug in benchmark_moe.py's int8_w8a16 dispatch path on this build (likely a recent vLLM regression). Switched to `--dtype auto` which lets the script infer from `model_config.dtype`.

## V3 (--dtype auto) — multi-batch sweep timed out before save

Default `--batch-size` sweeps multiple values (1, 8, 32, 64, 128). Saving happens **at end of script**, not per-batch. Our internal `timeout 1500` (25 min) fired in the middle of batch_size=8, after batch_size=1 had completed. No save. Confirmed by container exit log: `Completed tuning for batch_size=1` followed by SIGTERM.

## V4 (--batch-size 1 only) — saved JSON, but...

Restricting to `--batch-size 1` made the script complete in ~25 min and write the output JSON:

```
docs/mi100_decode_opt/moe_tune_output/E=256,N=128,device_name=Arcturus_GL-XL_[Instinct_MI100].json
```

```json
{
  "triton_version": "3.6.0",
  "1": {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 1, "num_warps": 1, "num_stages": 2,
    "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1
  }
}
```

## V5 — config rejected at runtime

### Issue 1 (fixable): missing `dtype=int8_w8a16` suffix

`get_config_file_name()` in `vllm/model_executor/layers/fused_moe/fused_moe.py:1018` builds the filename with `,dtype=int8_w8a16` for our quantized weights. The tuner's saved file omits it. Renamed the file to add the suffix; then it loads.

### Issue 2 (fixable): missing `SPLIT_K` field

The runtime path uses `fused_moe_kernel_gptq_awq` (not the un-quantized `fused_moe_kernel` the tuner exercises). That kernel signature requires `SPLIT_K`. Without it: `TypeError: dynamic_func() missing 1 required positional argument: 'SPLIT_K'`. Added `SPLIT_K: 1` (matches the existing `E=512,N=128,...,dtype=int4_w4a16.json` pattern shipped in tree).

### Issue 3 (fatal — gives up): HIP illegal memory access during cudagraph capture

After fixing 1+2 the model loads, but cudagraph capture crashes with:

```
[fused_moe.py:1077] Using configuration from .../E=256,N=128,...,dtype=int8_w8a16.json for MoE layer.
...
HIP warning: an illegal memory access was encountered
Failed: Cuda error custom_all_reduce_hip.cuh:514 'an illegal memory access was encountered'
```

The OOB happens inside the MoE triton kernel (subsequently propagating to CAR). Most likely: `BLOCK_SIZE_K=256` with `num_warps=1` doesn't satisfy a tile-coverage invariant for the GPTQ-AWQ kernel variant. The shipped `int4_w4a16` Arcturus config uses `BLOCK_SIZE_K=128` and `num_warps={1, 2, 4}` — a much more conservative pattern.

## Root cause

`benchmark_moe.py` tunes against the **un-quantized `fused_moe_kernel`** (no GPTQ groups, no AWQ zero-points, no SPLIT_K param). Configs it produces aren't directly transferable to the **`fused_moe_kernel_gptq_awq`** variant our model uses, even after surface-level field-name alignment. The tile shapes that are valid for one kernel can violate invariants in the other.

## What would actually work

1. **Hand-edit a known-good config**: take the existing `E=512,N=128,...,dtype=int4_w4a16.json` block-size pattern (BLOCK_SIZE_K=128) and just change `E=512` → `E=256` and dtype suffix. That config is already shape-compatible. Bench it. **Defer to round-4** because we already have a working baseline.

2. **Custom tuner targeting `fused_moe_kernel_gptq_awq`**: not a one-night job. **Defer to round-4.**

3. **Wait for upstream fix**: `benchmark_moe.py` may grow `--quant-mode int8_w8a16` (or similar) that exercises the right kernel. Track issue.

## Disposition

- The tuner JSON is kept at `docs/mi100_decode_opt/moe_tune_output/` as artifact (300 bytes; no harm).
- No source change shipped from this stage.
- Wallclock cost: ~50 min (V1 + V2 + V3 + V4 + V5 attempts). Worth it to prove the bug exists and document; not worth pushing further tonight.

## What this means for round-4

`round4_candidates.md` item F (custom gfx908 MFMA fmoe expert kernel, 2-3 days) becomes a stronger candidate now that we know the off-the-shelf tuner doesn't help. A hand-tuned kernel gets us tile/warp choices that respect the GPTQ-AWQ kernel's invariants.

The MoE-gemm time (~1.8 ms / 11.08 ms TPOT now ≈ 16% of TPOT — a higher fraction than at round-2 since we shrank the rest) is still a real lever; it's just one that needs handwritten code, not autotuning.
