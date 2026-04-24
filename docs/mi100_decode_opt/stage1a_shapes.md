# Stage 1a shape inventory — Qwen3.6-35B-A3B-GPTQ-8bit at TP=4

## Raw model params (from config.json)

- `hidden_size = 2048` (notably small for a 35B model — carried by MoE width)
- `num_hidden_layers = 40`
- `head_dim = 256`
- `num_attention_heads = 16` → Q dim = 16 × 256 = 4096
- `num_key_value_heads = 2` → K,V dim = 2 × 256 = 512 (GQA)
- `attn_output_gate = true` (extra gate projection in attention)
- Full-attention interval = 4 → **10 full-attn + 30 linear-attn layers**
- `linear_num_key_heads = 16`, `linear_key_head_dim = 128` → linear-K dim = 2048
- `linear_num_value_heads = 32`, `linear_value_head_dim = 128` → linear-V dim = 4096
- `linear_conv_kernel_dim = 4` (causal conv1d)
- `num_experts = 256`, `num_experts_per_tok = 8`
- `moe_intermediate_size = 512`, `shared_expert_intermediate_size = 512`
- `vocab_size = 248320` (extremely large — multimodal tokenizer)

## Per-rank linear GEMM shapes at TP=4, M=1 (decode)

### Full-attention block (10 layers × per-step)

| Op | M | N (per rank) | K | Notes |
|---|---:|---:|---:|---|
| Q proj | 1 | 1024 | 2048 | 16 heads / 4 rank = 4 heads × 256 = 1024 |
| K proj | 1 | 128 | 2048 | 2 kv heads / 4 rank (each replicated across heads?), or kept full and sharded differently |
| V proj | 1 | 128 | 2048 | |
| O proj | 1 | 2048 | 1024 | All-reduce across ranks after |
| Attn gate | 1 | 1024 | 2048 | `attn_output_gate=true` adds this |

### Linear-attention block (30 layers × per-step)

| Op | M | N (per rank) | K | Notes |
|---|---:|---:|---:|---|
| Q proj (delta) | 1 | 512 | 2048 | 16 K-heads × 128 / 4 ranks = 512 |
| K proj (delta) | 1 | 512 | 2048 | |
| V proj (delta) | 1 | 1024 | 2048 | 32 V-heads × 128 / 4 = 1024 |
| alpha proj | 1 | ? | 2048 | DeltaNet gating — check actual |
| beta proj | 1 | ? | 2048 | |
| out gate | 1 | ? | 2048 | |
| O proj | 1 | 2048 | 1024 | All-reduce after |

### Shared components (every layer)

| Op | M | N (per rank) | K | Notes |
|---|---:|---:|---:|---|
| MoE router | 1 | 256 | 2048 | Replicated (not TP-split) |
| Shared expert gate_up (fused?) | 1 | 256 | 2048 | 512/4 = 128 per MLP direction, gate+up likely fused → 256 |
| Shared expert down | 1 | 2048 | 128 | All-reduce after |

### Final layer (once per step)

| Op | M | N (per rank) | K | Notes |
|---|---:|---:|---:|---|
| lm_head | 1 | 62080 | 2048 | **248320 / 4 = 62080** — huge N |

## Expected rocBLAS dispatch vs observed kernels

Observed (from profile, rank 0, per decode iter):
- `MT128x128x32_MI32x32x8` × 80 calls — **biggest cost, 5.49 ms**
- `MT16x16x128_MI16x16x16` × 40 calls — 0.87 ms
- `MT128x192x32_MI32x32x8` × 2 calls — 0.85 ms ← almost certainly lm_head (N=62080)
- `MT128x64x64_MI32x32x8` × 40 calls — 0.77 ms
- `MT4x16x64_MI16x16x16` × 40 calls — 0.56 ms ← smallest tile, best for M=1

### Hypothesis for the 80 MT128 calls

80 calls per decode iter = 2 calls × 40 layers. Suspects:
- **(Q proj + O proj) × 40 layers** of some kind? Need rocBLAS logging to confirm.
- OR **(gate_up + down) × 40 layers** for shared expert? shared_expert_intermediate=512/rank=128 — (1, 2048) × (2048, 256) for gate_up would be N=256 per rank, and (1, 128) × (128, 2048) for down is N=2048.
- OR **(QKV fused + O) × 40 layers**? If QKV is fused to a single GEMM per layer it'd be N = 1024+128+128 = 1280 per rank (full-attn) or similar for linear-attn, then O is a separate call.

The 40-layer multiplier suggests it's per-layer. Need rocBLAS dispatch log to disambiguate.

### Hypothesis for the 40 MT4x16x64 calls

40 calls = 1 per layer. Suspects:
- Router (N=256, K=2048) — rocBLAS might pick MT4 because N=256 isn't huge. Plausible.
- OR some other single-op-per-layer projection.

### Critical investigation: why MT128 for 80 decode ops?

Hypotheses to test via rocBLAS logging:
1. rocBLAS heuristic picks MT128 because **K=2048 is moderate-large** and it's optimizing for a different performance regime than pure small-M.
2. The specific (M=1, N, K=2048) combination **doesn't match any small-M tuned entry in rocBLAS's Tensile DB for gfx908**, so it falls back to a default.
3. vLLM/GPTQ dispatch path is calling rocBLAS with a **batched dimension != 1** that looks larger than M=1 (e.g., M=32 for padded tile shapes, making MT128 look reasonable).

## Next investigation steps

1. ✅ Shape inventory (this doc)
2. → Enable `ROCBLAS_LAYER=2` + `ROCBLAS_LOG_TRACE_PATH` during a decode request; match log entries to `MT128` calls
3. → Read vLLM GPTQ-W8 dispatch path in container (`vllm/model_executor/layers/quantization/gptq*.py`, kernel priority registry)
4. → Inventory AITER's `gemm_a16w16.py` + peers on gfx908
5. → Micro-bench 3 hot shapes

## Open questions

- Is QKV a fused GEMM (N = Q+K+V per rank) or 3 separate GEMMs? vLLM's `QKVParallelLinear` usually fuses.
- Is the shared expert gate_up fused (2×intermediate)? `MergedColumnParallelLinear` usually fuses.
- Is lm_head running on all 4 ranks or only rank 0? The 2 `MT128x192x32` calls across all ranks vs just 2 on rank 0 suggests one-per-rank with replication or vocab-splitting.
