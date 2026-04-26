# Round-4 Phase 3 audit + B1'-Tune ship

**Date:** 2026-04-26
**Branch:** `mi100-optimized` @ `134809438` (Phase 2 E baseline)
**Outcome:** B1'-Tune branch fired; **one JSON config file shipped**, no code changes.

## Premise correction

The original Phase 3 plan (extend `TritonW4A16LinearKernel` to W8) was based
on the bottleneck inventory's claim that ExllamaLinearKernel scalar dequant
fires on QKV+output × ~64 layers. **Static analysis 2026-04-26 falsified
this** for Qwen3.6-35B-A3B-GPTQ-8bit:

- `quantize_config.json` dynamic exclusions skip `self_attn.*`, linear-attn
  `in_proj_*` / `out_proj`, shared experts, and `lm_head`.
- The model checkpoint's safetensors index confirms attention layers carry
  `.weight` only (no `.qweight`) — they're fp16.
- Only routed MoE expert linears (`mlp.experts.{N}.{gate,up,down}_proj`) are
  GPTQ-8bit quantized.
- Routed experts dispatch through `MoeWNA16Method.apply()` →
  `dispatch_fused_experts_func()` →
  `invoke_fused_moe_wna16_triton_kernel()` at
  `vllm/model_executor/layers/fused_moe/fused_moe.py:636`. **Never** lands
  on `LinearKernel` → `Exllama`.

Implementing TritonW8A16 LinearKernel would have yielded zero on this model.
The bottleneck inventory memo (`project_decode_bottleneck_inventory.md`) was
stale and partially wrong; the audit pivots Phase 3 to the kernel that
actually fires.

## Audit static reads

### Step 1 — kernel structure (`fused_moe_kernel_gptq_awq`, `fused_moe.py:78-310`)

Findings (all good — no refactor needed):

- **Line 290:** `accumulator = tl.dot(a, b, acc=accumulator)` — uses `tl.dot`
  which lowers to **MFMA on gfx908** via Triton's AMDGPU backend. ✓
- **Line 233:** `accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)`
  — **fp32 accumulator** (no precision loss in the GEMM). ✓
- **Lines 213-219 (W8 path):** `b_ptrs = b_ptr + ... + offs_k * stride_bk + ...`
  — for W8, qweight is byte-aligned int8 (no shift/mask unpack needed). ✓
- **Lines 287/289:** dequant `((b - b_zp) * b_scale).to(compute_type)` happens
  inside the K-loop but is **vectorized** (block-wide elementwise). ✓
  - Dequant could in theory be hoisted outside the K-loop *if* group_size were
    larger than BLOCK_SIZE_K, but our group_size=32 < BLOCK_SIZE_K=64, so
    per-K-block dequant is necessary anyway. Not a kernel issue.

**Verdict for the kernel itself: structurally fine. B1'-Refactor branch is
OFF the table.**

### Step 2 — block-config dispatch (`get_moe_wna16_block_config`, `fused_moe.py:1141-1210`, and `get_default_config`, `fused_moe.py:1259-1272`)

For decode at M=1, top_k=8, dtype=int8_w8a16:

1. `try_get_optimal_moe_config` calls `get_moe_configs(E, N, "int8_w8a16", 0, group_size)`.
2. JSON lookup misses → falls to `get_default_config(M=1, ..., "int8_w8a16", block_shape=[0, 32])`.
3. `get_default_config` returns `{BLOCK_SIZE_M: 16, GROUP_SIZE_M: 1, SPLIT_K: 1}` (M ≤ 20 branch). **No `num_warps`, no `num_stages`, no `BLOCK_SIZE_N`, no `BLOCK_SIZE_K`, no AMD MFMA hints.**
4. `invoke_fused_moe_wna16_triton_kernel` then calls `get_moe_wna16_block_config` which adds:
   - For `num_valid_tokens // top_k == 1` (decode): `{BLOCK_SIZE_N: 32, BLOCK_SIZE_K: 64}`
5. Triton fills missing meta-params with defaults: `num_warps=4, num_stages=1`.

Net effective decode config: `BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, SPLIT_K=1, num_warps=4, num_stages=1`. **No `waves_per_eu`, no `matrix_instr_nonkdim`, no `kpack`.**

### Step 3 — autotune file lookup (`get_config_file_name`, `fused_moe.py:1019-1030`)

For our model: `E=256, N=128, dtype=int8_w8a16, device_name="Arcturus GL-XL [Instinct MI100]"` → looks up `E=256,N=128,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int8_w8a16.json`.

**File did not exist.** The MI100 W4 wna16 config (`...int4_w4a16.json`) is the only MI100-specific autotune for any wna16 path; W8 was using the un-tuned default. Boot warning confirmed: `Using default MoE config. Performance might be sub-optimal!`

The W4 reference exists with full per-M-bucket gfx908 MFMA tuning
(`waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, optimized `num_warps`).
This is the template Phase 3 ports to W8.

## Audit live runs (3-run TPOT @ c=1, 256-token completion, fixed seed)

| Config | TPOT (ms) | tok/s | Δ vs baseline |
|---|---:|---:|---:|
| baseline (default, no JSON) | 9.13 | 109.5 | — |
| **cand_b** (W4-mirror with N=32 + AMD MFMA params at M=1) | **8.78** | **113.85** | **+4.0%** |
| cand_a (W4-mirror exact: N=16 + AMD params) | 8.89 | 112.4 | +2.7% |
| cand_c (default-style + K=128 only, no AMD params) | 9.12 | 109.67 | +0.2% (noise) |

**Cand_c is the control:** same `BLOCK_SIZE_K=128` as A/B but without AMD MFMA hints → no gain. Proves the lever is the AMD-specific params, not the K bump.

(Note: per-call TPOT here is wall-clock minus 50ms TTFT estimate, so it understates true TPOT vs BenchAndReport methodology. Relative deltas measured the same way are what matter.)

## Decision branch fire

| Branch | Trigger | Status |
|---|---|---|
| **None** | profile shows fused MoE <15% of TPOT, OR no tuple beats default by >1% | **NOT TAKEN** — cand_b clears 4% over default |
| **B1'-Refactor** | kernel uses scalar dequant in K-loop or no `tl.dot` | **NOT TAKEN** — kernel is `tl.dot` + fp32 accum + vectorized dequant |
| **B1'-Tune** | swept tuple beats default ≥3% on at least c=1 | **TAKEN** ✓ |

## B1'-Tune full BenchAndReport — cand_b round 1

Full 12-tier BenchAndReport on cand_b (W4-mirror config, all M buckets borrowed verbatim from W4):

| Scenario | E baseline tok/s | cand_b tok/s | Δ |
|---|---:|---:|---:|
| Single Latency c=1 | 96.69 | 100.14 | +3.6% |
| Decode Stress c=1 | 101.37 | 105.97 | +4.5% |
| Conc c=2 | 148.04 | 164.47 | +11.1% |
| Conc c=4 | 244.32 | 263.47 | +7.8% |
| Conc c=8 | 329.71 | 343.31 | +4.1% |
| **Conc c=16** | **558.49** | **531.74** | **−4.8% ⚠** |
| **Short Ctx (c=16)** | **481.12** | **465.39** | **−3.3% ⚠** |
| Conc c=32 | 773.65 | 845.00 | +9.2% |
| Conc c=64 | 1126.05 | 1153.59 | +2.4% |
| Conc c=128 | 1402.75 | 1385.76 | −1.2% |
| Long Ctx 16K | 164.88 | 172.20 | +4.4% |
| Mixed (c=8) | 317.81 | 330.48 | +4.0% |

Coherence 4/4 PASS pre + post.

**c=16 regression** tripped the plan's "Regression > 2% at any tier → STOP and triage" guardrail. Both c=16 scenarios slipped together → both hit the M=16 JSON bucket → bucket misconfigured for W8.

## Triage and v2 fix

W4's M=16 entry: `BLOCK_SIZE_N=64, num_warps=4, waves_per_eu=1, kpack=2, matrix_instr_nonkdim=16`.
W4's M=8 entry (worked, +4.1% in cand_b): `waves_per_eu=2, kpack=2`.
W4's M=32 entry (worked, +9.2% in cand_b): `waves_per_eu=4, kpack=2`.

Hypothesis: `waves_per_eu=1` at M=16 is too aggressive for W8's per-thread state pressure (W8 dequant has bigger intermediate fp32 values than W4's nibble unpack). Both adjacent buckets used `waves_per_eu ≥ 2` and gained.

**cand_b_v2: change one knob — M=16 `waves_per_eu: 1 → 2`**.

## B1'-Tune full BenchAndReport — cand_b_v2 (shipped)

| Scenario | E baseline tok/s | cand_b tok/s | **cand_b_v2 tok/s** | v2 vs E |
|---|---:|---:|---:|---:|
| Single Latency c=1 | 96.69 | 100.14 | **99.97** | **+3.4%** |
| Decode Stress c=1 | 101.37 | 105.97 | **105.91** | **+4.5%** |
| Conc c=2 | 148.04 | 164.47 | **164.19** | **+10.9%** |
| Conc c=4 | 244.32 | 263.47 | **259.35** | **+6.2%** |
| Conc c=8 | 329.71 | 343.31 | **343.60** | **+4.2%** |
| **Conc c=16** | **558.49** | **531.74 ⚠** | **561.20** | **+0.5% ✓** |
| **Short Ctx (c=16)** | **481.12** | **465.39 ⚠** | **491.20** | **+2.1% ✓** |
| Conc c=32 | 773.65 | 845.00 | **846.44** | **+9.4%** |
| Conc c=64 | 1126.05 | 1153.59 | **1145.61** | **+1.7%** |
| Conc c=128 | 1402.75 | 1385.76 | 1381.78 | −1.5% (within noise) |
| Long Ctx 16K | 164.88 | 172.20 | **169.87** | **+3.0%** |
| Mixed (c=8) | 317.81 | 330.48 | **331.38** | **+4.3%** |

**Coherence: 4/4 PASS pre, 4/4 PASS post.**

## What shipped

**One file, no code changes:**
- `vllm/model_executor/layers/fused_moe/configs/E=256,N=128,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int8_w8a16.json`

The file is the W4 MI100 reference with one knob changed (M=16 `waves_per_eu`).
All other M-bucket entries are W4 verbatim — they happened to transfer cleanly
to W8 for our shapes.

## Cumulative round-4 picture (Qwen3.6-35B-A3B-GPTQ-8bit)

| Stage | TPOT (ms) | tok/s c=1 |
|---|---:|---:|
| Round-3 baseline | 11.04 | 87.6 |
| Round-4 Phase 2 E (Mori-pattern CAR) | 9.97 | 96.7 (Decode Stress 101.4) |
| **Round-4 Phase 3 B1'-Tune (cand_b_v2)** | **9.59** | **99.97 (Decode Stress 105.9)** |

Cumulative round-3 → round-4: **+14.1% TPOT improvement at c=1**, on top of the round-1 → round-3 improvements.

## Open items

- **c=128 −1.5%**: in noise band, not worth a separate iteration. M=128 entry is W4 verbatim. If a future round wants to chase, the natural lever is the same `waves_per_eu` knob — but at the high-M end the kernel is more compute-bound and less sensitive.
- **Re-bench the other 3 production models** (35B-4bit, 27B-8bit, 27B-4bit). The new JSON only fires for the lookup key `E=256, N=128, ..., int8_w8a16` — other models won't load it (different E/N or different dtype). Validate that there's no surprise interaction.
- **Bottleneck inventory memo update**: correct the "Exllama on QKV × 64 layers" claim and document that for MoE-quantized models the W8 hot path is `fused_moe_wna16_triton_kernel`, not `LinearKernel`.

## Test infrastructure

Audit candidates and runner under `/home/tyler/decode_opt_audit/`:
- `configs/cand_a_w4mirror/`, `configs/cand_b_hybrid/`, `configs/cand_b_v2/`,
  `configs/cand_c_default_k128/`
- `bench_cand_b.sh` — monolithic runner (used for cand_b round 1)
- `bench_cand_b_v2_part1.sh` + `_part2.sh` — split runner that pauses after
  coherence-pre per the new feedback rule (use this pattern going forward).

These are kept on host (not in the repo) — they're scratch infra, not part of
the round-4 ship.
