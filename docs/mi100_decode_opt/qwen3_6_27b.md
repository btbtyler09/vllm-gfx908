# Qwen3.6-27B-GPTQ-8bit decode optimization on 4× MI100

**Model:** `/models/Qwen3.6-27B-GPTQ-8bit-multimodal` (the patched VLM variant — text path uses the same compute, but `Qwen3_5ForConditionalGeneration` is the registered class so it loads cleanly)
**Hardware:** 4× MI100 (gfx908) with XGMI peer-to-peer, TP=4, dtype=half
**Stack:** vLLM v0.19.2rc1+mi100 + AITER, `mi100-optimized` branch @ `2ae323c98` (round-3 of 35B already applied), `--mamba-cache-mode align`, max_model_len 32768, FULL_AND_PIECEWISE cudagraph, TRITON_ATTN
**Investigation date:** 2026-04-25

## Headline result

**Round 27B is complete and shipped nothing.** None of the round-3 levers transfer to a fully-GPTQ workload like 27B. `mi100-optimized` stays at `2ae323c98` (the 35B round-3 ship). The real lever for 27B — a W8A16 GPTQ kernel for dense linears — is deferred to a future round.

| Stage | Status | TPOT @ c=1 (ms) | tok/s @ c=1 | Δ vs baseline |
|---|---|---:|---:|---:|
| Pre-flight baseline | reference | **20.43** | **48.96** | — |
| 27-A — n>4 wrapper hoist (round-4 ticket K) | NULL — reverted | 20.83 | 48.02 | −2.0% (Python overhead, no-op for 27B) |
| 27-B — AITER for gate_up M=1 | NULL — reverted | (bundled w/ 27-A) | — | (microbench is fp16, prod is GPTQ) |
| 27-C — TunableOp tune+replay | SKIPPED | — | — | (TunableOp targets rocBLAS; 27B's only rocBLAS call is lm_head, already AITER) |
| 27-D — NCCL re-sweep | DONE — no change | LL128: 27.69 / LL+buf: 20.48 | — | LL128 −35.5%, LL+buf within noise. Tree+LL stays optimal. |
| 27-F — final BenchAndReport + commit | DONE — nothing to commit | — | — | — |

## Key finding — round-3 levers don't transfer to 27B

**The round-27B plan was built around a wrong assumption.** Round-3 wins came from optimizing the **unquantized GEMM dispatch** (`rocm_unquantized_gemm_gfx908`). That dispatch fired for 35B-A3B's QKV/o_proj because **35B has `dynamic` quant overrides that exclude those layers from GPTQ**:

```json
"dynamic": {
    "-:.*linear_attn\\.in_proj_qkv": {},
    "-:.*linear_attn\\.in_proj_z": {},
    "-:.*linear_attn\\.out_proj": {},
    "-:.*shared_expert\\.down_proj": {},
    "-:.*shared_expert\\.gate_proj": {},
    "-:.*shared_expert\\.up_proj": {}
}
```

**27B has no such exclusions** — every projection (qkv_proj, gate_up_proj, down_proj, etc.) is GPTQ-quantized. Only `lm_head` (with `lm_head: false` meaning "not quantized") goes through the unquantized path.

Verified via `VERIFY_DISPATCH=1` startup probe:
- 2,172 `[LLMM1]` prints, ALL for shape `n=1 m=62080 k=5120` (lm_head per-rank)
- ZERO `[wvSplitK]` prints
- ZERO `[AITER_DISPATCH]` prints

→ 27B's hot path is **`vllm._custom_ops.gptq_gemm`** (the HIP/exllama-style GPTQ kernel inside `GPTQLinearMethod.apply`), which the round-3 patches don't touch at all.

## Pre-flight surprises

- **Architecture string mismatch.** 27B's text-only weights have `Qwen3_5ForCausalLM`; vLLM only registers `Qwen3_5ForConditionalGeneration`. Use the `Qwen3.6-27B-GPTQ-8bit-multimodal` variant — same text compute path, fits the registered class.
- **Hybrid attention ratio.** 27B is 3 linear_attn : 1 full_attn. Need `--mamba-cache-mode align` AND let the platform auto-bump `block_size` (it picks 400 to align attn page size with mamba page).
- **`attn_output_gate=true`.** QKV per-rank shape is **(M, K=5120, N=3584)**, NOT 2048. Doubles q-head count (24·2 + 4 + 4 = 56 heads / 4 ranks).
- 64 layers vs 35B's 40, hidden 5120 vs 2048.

## Stages

### 27-A — n>4 wrapper hoist (NULL)

The 35B round-3 ship has a known c=64+128 regression because the `direct_register_custom_op` opacity blocks inductor from inlining the `F.linear` fallback (round 4 ticket K in `round4_candidates.md`). Patch tested:

```python
def rocm_unquantized_gemm_gfx908(layer, x, weight, bias=None):
    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]
    if n > 4 and not use_aiter_triton_gemm(n, m, k, x.dtype):
        return torch.nn.functional.linear(x, weight, bias)
    return torch.ops.vllm.rocm_unquantized_gemm_gfx908(x, weight, bias)
```

Result on 27B (3-run TPOT @ c=1):

| Configuration | TPOT (ms) | tok/s | Δ |
|---|---:|---:|---:|
| Baseline (round-3 HEAD) | 20.43 | 48.96 | (ref) |
| 27-A applied | 20.83 | 48.02 | −2.0% |

**Why null:** the c=64+ opacity issue only affects unquantized GEMM, and 27B has almost none. The −2% is small Python overhead from the wrapper's extra check on the lm_head call path. Below the +5% improvement threshold needed to ship for c=64+128.

The lever **remains valid for 35B-A3B** (mixed-quantization workload). Should be benched there, not 27B.

### 27-B — AITER for gate_up M=1 (NULL)

Microbench `aiter.ops.triton.gemm.basic.gemm_a16w16` vs LLMM1 vs rocBLAS at M=1 for 27B's per-rank decode shapes (TP=4, fp16 weights):

| Shape (M, K, N) | rocBLAS µs | LLMM1 µs | AITER BC µs | Best |
|---|---:|---:|---:|---|
| qkv (1, 5120, 3584) | 38.85 | **33.75** | 50.77 | LLMM1 |
| o_proj (1, 1536, 5120) | 19.14 | **13.16** | 50.00 | LLMM1 |
| **gate_up (1, 5120, 8704)** | 93.21 | 84.89 | **50.91** | **AITER 1.67×** (fp16) |
| down_proj (1, 4352, 5120) | 46.55 | **41.72** | 53.38 | LLMM1 |

Tested patch: added `(m=8704, k=5120)` to the `use_aiter_triton_gemm` whitelist + bypass LLMM1 for that shape. With `VERIFY_DISPATCH=1` and full priming, ZERO `[wvSplitK]` / ZERO `[AITER_DISPATCH]` prints fired for `n=1 m=8704 k=5120` — confirming gate_up_proj never reaches the unquantized dispatch path. Only lm_head appears.

**Why null:** the microbench is fp16×fp16. 27B's gate_up is **GPTQ-W8A16** dispatched via `ops.gptq_gemm`. AITER's `gemm_a16w16` requires fp16 weights; can't operate on packed int8 GPTQ weights without dequantizing them first (slower AND wastes the memory savings). The fp16 microbench numbers are the **ceiling** for each per-rank shape — useful as a target for a future W8A16 implementation, not as a drop-in.

### 27-C — TunableOp (SKIPPED)

TunableOp tunes `torch.mm` / `F.linear` (rocBLAS) calls. For 27B the only such call is lm_head, which AITER already handles via the existing whitelist. Almost certainly null on 27B; skipped. (Round-2 finding for 35B carries: TunableOp may help c≥8 if any decode shape ever falls back to rocBLAS — not the case here.)

### 27-D — NCCL re-sweep (NO CHANGE)

27B has 1.6× more layers (64 vs 40) and 2.5× hidden (5120 vs 2048), so the all-reduce profile differs from 35B. Tested two alternatives to round-3's Tree+LL default:

| Config | NCCL env | TPOT (ms) | tok/s | Δ vs ctrl |
|---|---|---:|---:|---:|
| **Control (round-3 default)** | `NCCL_ALGO=Tree NCCL_PROTO=LL` | **20.43** | **48.96** | (ref) |
| LL128 | `NCCL_ALGO=Tree NCCL_PROTO=LL128` | 27.69 | 36.12 | **−35.5%** |
| Buffered LL | `NCCL_ALGO=Tree NCCL_PROTO=LL NCCL_BUFFSIZE=8388608 NCCL_NTHREADS=512` | 20.48 | 48.83 | −0.24% (noise) |

**Findings:** LL128 is sharply worse (opposite of the round-27B plan's hypothesis). Most all-reduces still fall under LL's payload threshold even at hidden=5120; LL128's protocol overhead dominates. Buffer/thread tweaks don't help. Tree+LL stays the right default for gfx908 / XGMI / 4-rank — confirmed across both 40-layer and 64-layer model profiles.

## Baseline (12-scenario BenchAndReport)

| Scenario | tok/s | TTFT (ms) | TPOT (ms) |
|---|---:|---:|---:|
| Single User Latency (c=1, 2048+512) | 43.7 | 1431 | 20.1 |
| Decode Stress (c=1, 128+2048) | 49.7 | 181 | 20.1 |
| Short Context (c=16, 512+256) | 208.2 | 3515 | 56.5 |
| Long Context 16K (c=4, 16384+1024) | 49.8 | 22707 | 52.7 |
| Mixed Traffic (c=8 ±50%) | 129.1 | 3239 | 52.9 |
| Scaling c=2 | 69.5 | 1130 | 24.4 |
| Scaling c=4 | 100.5 | 1702 | 31.4 |
| Scaling c=8 | 134.9 | 3004 | 44.8 |
| Scaling c=16 | 170.6 | 5444 | 68.7 |
| Scaling c=32 | 164.5 | 11782 | 135.7 |
| Scaling c=64 | 211.3 | 23757 | 206.6 |
| Scaling c=128 | 238.5 | 47383 | 339.9 |

Reports: `~/mi100-llm-testing/Model_Reports/round27b_baseline_qwen3.6-27b-8bit_2026-04-25.md`. Raw JSON: `/tmp/decode_opt/27b_baseline.json`, `27b_baseline_raw.json`.

## What would actually win on 27B GPTQ

The fp16 microbench numbers in 27-B above are the **ceiling** for each per-rank shape. A W8A16 GPTQ kernel that gets within ~1.5× of those numbers would beat current `ops.gptq_gemm`. Per-rank decode shapes (all GPTQ on 27B):

- `qkv_proj`: K=5120, N=3584 (`attn_output_gate=true` → 48 q + 4 kv + 4 kv heads × 256 / 4)
- `o_proj`: K=1536, N=5120
- `gate_up_proj`: K=5120, N=8704 (TP-split, 2× intermediate)
- `down_proj`: K=4352, N=5120
- `lm_head`: K=5120, N=62080 — UNQUANTIZED, LLMM1 already fires

Candidates for a future round of 27B work:

- **AITER MoE-style `fused_moe_kernel_gptq_awq` adapted for dense layers** (vLLM's `quantization/utils/marlin_utils*` lineage). Already W8/W4-aware Triton; needs dense-layer adapter.
- **Custom W8A16 Triton GEMM** with the same tile pattern as `_AITER_GEMM_M1_BEST_CFG` (BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, num_warps=4, mfma=16).
- **Shape-tuned port of the existing `gptq_gemm` HIP kernel** for gfx908.
- **Marlin-style packed GEMM port** if an int8 marlin variant exists for AMD (the team's prior memory says marlin is NVIDIA-only — would need fresh evaluation).

## Files & artifacts

- Branch: `mi100-optimized` (unchanged at `2ae323c98`)
- Bench script: `docs/mi100_decode_opt/scripts/test_27b_baseline.sh`
- Microbenches: `/tmp/decode_opt/microbench_27b_aiter.py`, `microbench_27b_llmm1.py`, `microbench_27b_lmhead.py`
- Logs: `/tmp/decode_opt/27b_*`, `/tmp/decode_opt/serve_27b.log`, `/tmp/decode_opt/27d_ncclB_LL128.log`, `27d_ncclC_LLbuf.log`
- Coherence logs: `/tmp/decode_opt/27d_LL128_pre.log`, `27d_LL128_post.log`, `27b_pre.log`, `27b_post.log`
- Reports: `~/mi100-llm-testing/Model_Reports/round27b_*`
