# Stage 5 — Re-baseline (round 2 starting point)

**Date:** 2026-04-24
**Branch:** `mi100-optimized` @ `23bbb8696` (Stage 2+3 patches committed)

## TPOT measurement (3-run set, 256-tok decode, c=1)

| Run | Tokens | Wall (s) | tok/s | TPOT (ms) |
|-----|--------|----------|-------|-----------|
| 1   | 256    | 4.956    | 51.65 | 19.36 |
| 2   | 256    | 4.956    | 51.66 | 19.36 |
| 3   | 256    | 4.957    | 51.64 | 19.37 |
| **median** |    |          | **51.65** | **19.36** |

Pre-bench coherence: PASS. Post-bench coherence: PASS.

This is **noise-floor consistency** (±0.01 ms across runs) — much better than the prior session's ±0.15 ms across container restarts. Use this as the round-2 baseline.

## What's in the compiled graph (inductor cache audit)

Inspected `/root/.cache/vllm/torch_compile_cache/15bcfd771b/rank_0_0/backbone/` and the AOT inductor cache.

**In the compiled graph (per layer):**
- `input_layernorm`, `post_attention_layernorm`
- For full-attn layers (10): `qkv_proj`, `q_norm`, `k_norm`, `o_proj`
- For linear-attn layers (30): `in_proj_qkvz`, `in_proj_ba`, `out_proj`, GDN `norm`

**NOT in the compiled graph (executes eager):**
- `conv1d` (linear-attn)
- The entire MoE block: `gate`, `shared_expert_gate`, `shared_expert.gate_up_proj`, `shared_expert.down_proj`, `experts` (FusedMoE Triton path)
- attention itself (lives behind a custom op)
- RoPE

**Inductor cache stats (240 sub-graph .py files):**
- 60 files contain at least one `triton_fused_*` kernel definition
- 132 fused-kernel definitions total
- 56 `extern_kernels.mm/addmm/bmm` calls remain (unfused → rocBLAS)

**Implication for round 2:**
- Stage 3's custom-op bypass DID help inductor fuse: ~132 fused kernels exist where pre-Stage 3 they would not. This explains the +0.5 % TPOT we measured for Stage 3.
- MoE block bypasses inductor entirely → the 80+ small Linear calls in the MoE block (gate, shared_expert_gate, shared_expert MLP, etc.) are all eager. **Stage 6 Python-level fusion has full freedom on this code.**

## Per-step Linear shape map (from static analysis of Qwen3.6-35B-A3B config)

Confirmed via reading `qwen3_5.py`, `qwen3_next.py`, `gdn_linear_attn.py`, and `config.json`.

Per-rank (TP=4) UNQUANTIZED Linear calls per decode step:

| Shape (M, N, K)        | Calls/step | Origin |
|------------------------|------------|--------|
| (1, 3072, 2048)        | 40 | full-attn `qkv_proj` (10 layers) + linear-attn `in_proj_qkvz` (30 layers) |
| (1, 2048, 1024)        | 40 | full-attn `o_proj` (10) + linear-attn `out_proj` (30) |
| (1, 256, 2048)         | 80 | MoE `gate` (40) + MoE `shared_expert.gate_up_proj` (40) |
| (1, 1, 2048)           | 40 | MoE `shared_expert_gate` (40) |
| (1, 2048, 128)         | 40 | MoE `shared_expert.down_proj` (40) |
| (1, 16, 2048)          | 30 | linear-attn `in_proj_ba` (30) |
| **TOTAL**              | **270** | (excludes lm_head, MoE expert kernels, attention, conv1d) |

Plus:
- lm_head (1, 62080, 2048) × 1 per request — already AITER-dispatched (Stage 2)
- MoE expert kernels: vLLM's fused_moe Triton path (1 launch per stage per layer × 40 layers × 2 stages) ≈ 80 kernel launches, but each handles multiple experts at once
- attention kernels: vLLM `unified_attention` for full-attn (10/step), FLA `fused_recurrent_gated_delta_rule_packed_decode` for linear-attn (30/step)

## Key fusion targets identified

### Tier 1 (high-leverage, simple Python merge)

**(1, 256, 2048) gate + (1, 1, 2048) shared_expert_gate** — same input (`hidden_states` post-`post_attention_layernorm`), both `ReplicatedLinear`. Merge into a single `(1, 257, 2048)` `ReplicatedLinear`. Saves 1 launch × 40 layers × ~50 µs = **~2 ms / step / ~10 % TPOT**.

### Tier 2 (medium-effort)

**hipBLASLt swap** — RULED OUT. hipBLASLt is 1.08x–4.4x SLOWER than rocBLAS at all our M=1 shapes on gfx908. See microbench in stage5_baseline_run2.log.

**vLLM fused_moe config gap** — `E=256,N=128,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int8_w8a16.json` is missing. Tune via `benchmarks/kernels/benchmark_moe.py --tune`. Estimated win: 5-15 % of MoE-expert time = 0.1–0.3 ms / step / 0.5–1.5 % TPOT.

### Tier 3 (large-effort, biggest prize)

**CAR NaN under HIP cudagraph** — 3.4 ms / 16 % TPOT. See Stage 7 in plan.

## Stage 5b — hipBLASLt probe (CONCLUSIVE NEGATIVE)

| Shape (M, N, K)        | rocBLAS µs | hipBLASLt µs | Ratio |
|------------------------|-----------|--------------|-------|
| (1, 3072, 2048)        | 21.6      | 94.6         | 4.38× LOSE |
| (1, 2048, 1024)        | 15.9      | 45.0         | 2.84× LOSE |
| (1, 256, 2048)         | 49.8      | 81.3         | 1.63× LOSE |
| (1, 62080, 2048)       | 465       | 502          | 1.08× LOSE |
| (1, 16, 2048)          | 19.4      | 77.0         | 3.97× LOSE |
| (1, 1, 2048)           | 19.0      | 47.8         | 2.52× LOSE |

hipBLASLt is universally slower. Don't enable.
