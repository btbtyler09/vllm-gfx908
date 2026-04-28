# Qwen3.6-35B-A3B decode optimization on 4× MI100

**Model:** `Qwen3.6-35B-A3B-GPTQ-8bit` (registered arch: `Qwen3_5MoeForConditionalGeneration`)
**Hardware:** 4× MI100 (gfx908) with XGMI peer-to-peer, TP=4, dtype=half
**Stack:** vLLM v0.19.2rc1+mi100 + AITER, `VLLM_MI100_TORCH_COMPILE=1`, mode-3 + FULL_AND_PIECEWISE cudagraph, TRITON_ATTN
**Branch:** `mi100-optimized`
**Investigation window:** 2026-04-24 → 2026-04-25 (3 rounds, all shipped)

## Headline result

| Milestone | TPOT (ms) | tok/s | Δ throughput | Commit |
|---|---:|---:|---:|---|
| Pre-investigation baseline | 19.50 | 51.0 | — | — |
| End of round 1 (AITER lm_head + custom-op bypass) | 19.24 | 52.0 | +1.96% | `23bbb8696` |
| End of round 2 (LLMM1 + wvSplitK dispatch) | 13.82 | 72.4 | +40.1% | `7db67cd92`, `fb72c78b8` |
| End of round 3 (inductor escape hatch + NCCL Tree+LL) | **11.00** | **90.9** | **+76.1%** vs baseline | `2ae323c98` |

3-run TPOT spread ±0.01 ms once the optimizations stabilized. Coherence (4-prompt smoke: fibonacci, hash collisions, French translation, ocean haiku) PASS pre + post on every shipped stage.

12-scenario `BenchAndReport.py` confirms wins extend across the practical concurrency range (c=1 through c=32). Modest regression at c=64 (−4%) and c=128 (−9.3%) is fully understood (see "round 4 ticket K" below).

## What ships in `mi100-optimized`

| File | Role |
|---|---|
| `vllm/model_executor/layers/utils.py` | The whole story. Adds `rocm_unquantized_gemm_gfx908` + `_impl` split, AITER-whitelist for lm_head with hand-tuned BEST_CFG, LLMM1 / wvSplitK dispatch ladder, and `direct_register_custom_op` registration so inductor sees one opaque node. |
| `vllm/platforms/rocm.py` | gfx908 defaults — `VLLM_ROCM_USE_AITER_TRITON_GEMM=1`, `VLLM_ROCM_USE_SKINNY_GEMM=1`, `NCCL_ALGO=Tree`, `NCCL_PROTO=LL` |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | gfx908 CAR bypass placed in `should_custom_ar()` (commit `f86fed94f`) — predates this investigation but is required for round 1 to work |
| `docs/mi100_decode_opt/scripts/test_stage5_baseline.sh` | NCCL Tree+LL env vars + `VERIFY_DISPATCH=1` opt-in for stderr dispatch prints |
| `docs/mi100_decode_opt/scripts/test_stage_combined.sh` | Same env defaults; always sets `TUNABLEOP=replay` for read-only runs |

## Required environment

`VLLM_ROCM_USE_AITER_TRITON_GEMM=1` MUST be set via `docker --env`, not via `os.environ.setdefault` in `rocm.py`. The defaulter runs too late: `_aiter_ops.py:1144` caches `_TRITON_UNQUANT_GEMM` at class-definition time, BEFORE the rocm.py `_GFX908_DEFAULTS` block executes. The other env vars (NCCL Tree+LL, USE_SKINNY_GEMM) are read later and the platform-defaulter is sufficient.

Optional debug: `VLLM_GFX908_DEBUG_DISPATCH=1` enables per-dispatch stderr prints (silent by default) for `[LLMM1]`, `[wvSplitK]`, `[AITER_DISPATCH]`.

---

# Round 1 — profiling and AITER lm_head dispatch (+1.96%)

**Commit:** `23bbb8696` — *Surgical AITER lm_head dispatch + custom-op bypass on gfx908*

## Stage 0 — decode profile

Decode-only torch profile (long request → 1.5s warmup → `/start_profile` → 1.5s sleep → `/stop_profile`) showed GPU kernel time totals ~21 ms / decode iter, matching the 19.5 ms TPOT directly. **There is no hidden launch / scheduler overhead to attack** — the GPU is busy for the full step.

Decode bucket breakdown (rank 0):

| Bucket | GPU ms | % of decode |
|---|---:|---:|
| **linear-rocblas** (QKV / O / router / shared-expert) | **9.30** | **43.9%** |
| **all-reduce** (RCCL ring via `ncclDevKernel_Generic_2`) | **3.38** | **16.0%** |
| **moe-gemm** (`fused_moe_kernel_gptq_awq`) | **1.82** | **8.6%** |
| moe-routing, triton-misc, elementwise | 3.33 | 15.7% |
| norm, sampler, memcpy, attn, rope | 1.92 | 9.0% |
| linear-attn, other | 1.61 | 7.6% |

Reframings vs the prior bottleneck inventory:
- **Not Exllama-scalar.** Linears already use rocBLAS Tensile MFMA (`Cijk_Alik_Bljk_HHS_BH_MT128x128x32_MI32x32x8x1` etc.). The lever is *tile selection* (rocBLAS picks MT128 for M=1, computing 127/128 rows of garbage), not "write a Triton kernel from scratch."
- **Hybrid-attention model.** 10 full-attn + 30 linear-attn (DeltaNet) layers; linear-attn is only 0.58 ms / step (2.7%) — not a decode target.
- **CAR is broken under cudagraphs.** RCCL fallback's 16% is real and reproducible (broken-CAR runs at 59 tok/s = +17% vs working RCCL). Confirmed root cause is HIP IPC pointer staleness on graph replay (CDNA1-specific) — see "Things that did not ship" below.

## Stage 1a — shape inventory

Per-rank (TP=4) UNQUANTIZED Linear shapes per decode step:

| Shape (M, N, K) | Calls/req | Per step | rocBLAS µs | AITER best µs | Speedup |
|---|---:|---:|---:|---:|---|
| (1, 62080, 2048) lm_head | 40 | 0.16 | 466 | 267 | **1.74×** |
| (1, 3072, 2048) full_attn QKV | 20,880 | 21 | 21 | 50 | 0.42× LOSS |
| (1, 2560, 2048) linear_attn QKVZ | 6,960 | 7 | 18 | 51 | 0.35× LOSS |
| (1, 2048, 1024) attn out | 27,840 | 28 | 16 | 50 | 0.32× LOSS |
| (1, 256, 2048) router / gate_up | 55,680 | 56 | 50 | 63 | 0.79× LOSS |
| (1, 2048, 128) shared down | 27,840 | 28 | 15 | — | LOSS |
| (1, 16, 2048) lin_attn ba | 20,880 | 21 | 19 | — | LOSS |
| (1, 1, 2048) shared gate | 27,840 | 28 | 18 | — | LOSS |

**Key insight:** AITER Triton GEMM has a ~50 µs per-launch floor on gfx908 — only beats rocBLAS when the GEMM is large enough to absorb the floor. For our model that's only lm_head. The earlier microbench's "5–6% TPOT" projection was based on incorrect shape assumptions (linear_attn_z doesn't exist as a separate op — fused into `in_proj_qkvz`; lm_head is replicated, not column-split).

## Stage 2 — lm_head AITER dispatch (shipped)

Added `(m=62080, k=2048)` to `use_aiter_triton_gemm` whitelist with hardcoded `_AITER_GEMM_M1_BEST_CFG = (BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, num_warps=4, NUM_KSPLIT=1)`. AITER's default `_get_config(M, N, K)` picks M_LEQ_64 (BLOCK_SIZE_M=64) which wastes blocks at M=1; BEST_CFG (BLOCK_SIZE_M=16) recovers 1.34→1.74×. Result: TPOT 19.5 → 19.33 ms (**+1.4%**), 7128 AITER dispatches verified across 3-run test.

MFMA tile sweep confirmed 16x16x16 beats 32x32x8 at M=1 — larger tiles waste compute on padding for skinny matmuls.

## Stage 3 — custom-op bypass (shipped)

Added `rocm_unquantized_gemm_gfx908(layer, x, weight, bias)` and modified `dispatch_unquantized_gemm()` to return it on gfx908. The new function inlines AITER dispatch and falls through to `torch.nn.functional.linear` for non-AITER shapes — bypassing the opaque `torch.ops.vllm.rocm_unquantized_gemm` custom op so inductor can fuse around `F.linear`.

Hypothesis was 5–10% TPOT from inductor fusion. Reality: ~0.5%. Custom-op overhead is only ~0.4 µs/call on gfx908 (not the ~5 µs assumed). `F.linear → aten::mm` still goes through rocBLAS; inductor doesn't get to fuse around external kernel calls. Audit of the inductor cache (240 sub-graph .py files) showed 132 fused triton kernels exist that didn't pre-Stage 3 — fusion DID happen, just for surrounding pointwise ops, not the GEMM itself.

Result: TPOT 19.33 → 19.24 ms (+0.5% on top of Stage 2 = **+1.96% cumulative**).

## Stage 4 — vLLM compilation passes (no win)

- `fuse_allreduce_rms` → **BLOCKED.** `AllReduceFusionPass`, `AsyncTPPass`, `MiniMaxQKNormPass` are gated to `is_cuda()` in `vllm/compilation/passes/pass_manager.py:38` — not importable on ROCm. Setting the flag crashes (`name 'AllReduceFusionPass' is not defined`). Unlocking would need patching the import gate AND porting the pass body to ROCm AR — not a quick win.
- `fuse_act_padding` → **slight regression** (19.24 → 19.31 ms). `RocmAiterTritonAddRMSNormPadFusionPass` engages but doesn't help our shapes.
- `fuse_norm_quant`, `fuse_act_quant` — model has unquantized linears we care about; quant fusion not relevant.
- `enable_sp` (sequence parallelism) — decode has seq_len=1, nothing to shard.
- `fuse_rope_kvcache` — only 10/40 layers have RoPE (linear-attn doesn't); marginal upside, didn't pursue.

No Stage 4 patch retained.

---

# Round 2 — LLMM1 + wvSplitK dispatch (+40%)

**Commits:** `7db67cd92` (Stage 5g dispatch), `fb72c78b8` (round 2 summary)

## The setup

After round 1 the round-2 baseline (3-run, ±0.01 ms) was **19.36 ms / 51.65 tok/s**. Inductor cache audit at this point: 56 `extern_kernels.mm/addmm/bmm` calls remain (unfused, hitting rocBLAS), 132 `triton_fused_*` kernels exist for surrounding pointwise ops. The MoE block runs entirely eager (cudagraph captures around it but inductor doesn't).

## Stage 5g — LLMM1 + wvSplitK in our gfx908 dispatch (the headline win)

`vllm/_custom_ops` already exposes `wvSplitK` and `LLMM1` skinny-GEMM kernels. They were built for gfx908 since `csrc/rocm/skinny_gemms.cu:25` had `defined(__gfx908__)` added in larkinwc/vllm-gfx908#4 (already in our base image). They were never firing because:

1. Standard `rocm_unquantized_gemm_impl` gates them behind `(on_gfx9() or on_gfx1x())` — but `on_gfx9()` excludes gfx908 (`rocm.py:190` lists only gfx90a/942/950).
2. Our gfx908 path (`rocm_unquantized_gemm_gfx908` from Stage 3) bypasses the standard impl entirely.

Patched our path to dispatch in this priority order (fp16/bf16, k % 8 == 0, contiguous weight):

1. **LLMM1** if `n==1, m % 4 == 0, k <= 8192, bias is None`
2. **wvSplitK** if `m > 8 and 0 < n <= 4`
3. **AITER `gemm_a16w16`** for the lm_head whitelist (LLMM1 actually beats it now — 244 vs 267 µs — but kept as fallback)
4. **`F.linear`** otherwise (inductor-fusable)

Microbench (gfx908 single-GPU, eager):

| Shape (M, N, K) | rocBLAS µs | wvSplitK µs | LLMM1 µs | Best vs rocBLAS |
|---|---:|---:|---:|---:|
| (1, 3072, 2048) QKV/QKVZ | 21.19 | 9.77 | 10.83 | 2.17× |
| (1, 2048, 1024) o_proj | 15.88 | 7.61 | 7.23 | 2.20× |
| (1, 2048, 128) shared dn | 15.82 | 7.49 | 7.39 | 2.14× |
| (1, 256, 2048) router/gu | 49.68 | 7.56 | 7.41 | **6.71×** |
| (1, 1, 2048) shared_gate | 19.21 | 7.84 | — | 2.45× |
| (1, 16, 2048) lin_attn ba | 20.21 | 7.73 | 7.58 | 2.67× |
| (1, 62080, 2048) lm_head | 464.19 | 295.72 | 244.33 | 1.90× |

`wvSplitK`/`LLMM1` both have a ~7.5 µs gfx908 floor. rocBLAS hits a brutal ~50 µs floor at the (1, 256, 2048) router shape — that's where the 6.71× lives.

End-to-end: **TPOT 19.36 → 13.82 ms = +40.1% throughput.** Dispatch trace from `VLLM_GFX908_DEBUG_DISPATCH=1`:
- `[LLMM1]` for n=1 router/gate_up_proj (m=256, k=2048), shared_expert.down_proj (m=2048, k=128), lm_head (m=62080, k=2048)
- `[wvSplitK]` for n=2/n=4 prefill batches of those same shapes
- **NOT seen:** `n=1 m=3072 k=2048` (full-attn QKV / linear-attn QKVZ) or `n=1 m=2048 k=1024` (attn out). Those go through inductor's compiled subgraph and end up at `aten::mm` regardless of dispatch. Estimated headroom **~700 µs / step (3.6% TPOT)** if we could route them too — became Round 3's headline.

Also flipped `VLLM_ROCM_USE_SKINNY_GEMM` default to `"1"` in `vllm/platforms/rocm.py` (no-op for our patched path; flipped for cleanliness in case standard impl ever runs).

**Why this works when launch-overhead reductions didn't:** cudagraph already amortizes Python launch overhead to ~1 µs. The 50 µs "launch floor" was always GPU-side compute floor inside rocBLAS. `wvSplitK`/`LLMM1` *replace the kernel* with a faster one — the lever cudagraph cannot hide.

## Stage 5b — hipBLASLt probe (closed permanently)

Microbench across all hot M=1 shapes: hipBLASLt is **1.08× to 4.4× SLOWER** than rocBLAS on gfx908. Don't enable.

## Stage 6a — Python-level fusion of MoE block (no-op)

Implemented `gate + shared_expert_gate` Python-level merge, coherence PASS, but TPOT delta −0.01 ms (within noise). This was the strongest test of the "fewer launches via Python fusion" thesis. **It failed.** Cudagraph already amortizes per-launch overhead at the Python level to ~1 µs. The kernel-itself is the lever, not the launch count. Reverted.

## Stage 7 — CAR NaN deep-dive (closed)

larkinwc PRs #7 + #10 confirm root cause: **"IPC buffer addresses captured in the graph become stale on replay"** — fundamental HIP runtime / IPC behavior on CDNA1, not source-level fixable. Both PRs sidestep with the same `should_custom_ar()` bypass we already have. The 3.4 ms / 16% TPOT prize is unattainable without a HIP runtime patch. Capped further investigation.

## Already in our base image (free wins, no work needed)

- **PR #4** — skinny GEMM compile guard at `csrc/rocm/skinny_gemms.cu:25`. Round 2 Stage 5g wires the dispatch.
- **PR #6** — FlashSplitK adaptive split-K at `triton_attn.py:62-96` (`_compute_flash_decoding_splits()`), enabled on MI100 at lines 229-263, used dynamically at `triton_unified_attention.py:1198-1220`. Already in the 13.82 ms result.
- **PR #3** — Triton attention tuning, `triton_attn.py:51-56` `_get_mi100_tuned_constants()` returns `(64, 8)`.

---

# Round 3 — inductor escape hatch + NCCL Tree+LL (+25.7%)

**Commit:** `2ae323c98` — *round-3 decode opt: inductor escape hatch + NCCL Tree+LL*

## Stage 5h — Inductor escape hatch via custom-op registration ⭐ headline win

**The problem:** Round 2's dispatch fired for the MoE block and lm_head but missed QKV/QKVZ/o_proj. Those layers are inside `@support_torch_compile`-wrapped model forwards — inductor traces the entire forward into a single FX graph and lowers our `F.linear` fallback to `extern_kernels.mm` (rocBLAS) at compile time, before runtime dispatch ever sees the call.

**The fix:** Mirror the existing `direct_register_custom_op` pattern (`utils.py:237-241`) for the gfx908 path. Split into wrapper + impl:

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
    fake_impl=rocm_unquantized_gemm_fake,
)
```

`dispatch_unquantized_gemm()` is unchanged — it still returns `rocm_unquantized_gemm_gfx908`, which now routes through the custom op. The opaque op cannot be inlined; at runtime the dispatch runs eagerly with all its branching.

End-to-end: **TPOT 13.82 → 11.62 ms = +18.9% throughput.** Spread ±0.005 ms (cleanest 3-run set in the project). Dispatch trace shows the previously-missing shapes now firing through LLMM1:

```
[LLMM1] n=1 m=3072 k=2048   ← QKV / QKVZ          (NEW in 5h)
[LLMM1] n=1 m=2048 k=1024   ← o_proj              (NEW in 5h)
[LLMM1] n=1 m=2560 k=2048   ← per-layer QKV variant  (NEW in 5h)
```

The realized win was **4× the round-2 estimate** (2.20 ms / step vs the 700 µs estimate) because the original estimate counted only the rocBLAS launcher floor — the actual win includes the entire ~10× LLMM1-vs-rocBLAS speedup at M=1.

## Stage 5j — NCCL Tree+LL (shipped)

Default NCCL all-reduce on 4 ranks uses Ring (latency = `2(N−1)·msg/N·BW`). Our per-step all-reduces are tiny (~few KB) — Tree algorithm with `log_2(N)` depth wins for small messages. LL protocol cuts protocol overhead by inlining the data flag bit into the payload.

| Config | env vars | TPOT median | vs 11.62 baseline |
|---|---|---:|---:|
| Baseline | (none) | 11.62 ms | — |
| Tree algo | `NCCL_ALGO=Tree` | 11.11 ms | −4.4% |
| LL proto only | `NCCL_PROTO=LL` | 11.60 ms | −0.2% (noise) |
| **Tree + LL** | both | **11.08 ms** | **−4.6%** |

Final 3-run mean **11.00 ms / 90.94 tok/s** (additional small drift over multiple container restarts). LL on its own does little; on top of Tree it adds a marginal 0.03 ms. Free to enable since it costs nothing.

Defaults wired into `_GFX908_DEFAULTS` in `vllm/platforms/rocm.py` so they auto-apply on every gfx908 launch. Test scripts also set them explicitly.

## Stage 5k — Inductor fusion audit (no-ship; already optimal)

Single instrumented launch with `TORCH_COMPILE_DEBUG=1`. **RMSNorm + residual add already fuse into a single triton reduction kernel** (`triton_red_fused_1`) running back-to-back with our custom-op gemm extern call. SiLU-gating-style paths fuse into a separate triton kernel that prepares the gemm input. Inductor topologically clusters source nodes into a single triton kernel even though gemm/all_reduce/moe_forward_shared remain opaque extern calls — kernel names literally include the source op names (`triton_poi_fused_mul_rocm_unquantized_gemm_gfx908_sigmoid_view_0`).

The remaining "lost fusion" is ~4 KB / gemm of redundant HBM round-trip — dwarfed by everything else. No code change warranted. Closes round 4 candidate B.

Trace dump kept at `docs/mi100_decode_opt/stage5k_inductor_trace/` (6.4 MB; gitignored — the trace contains paths and PIDs that change per run).

## Round 3 full-bench delta

12-scenario `BenchAndReport.py` against round 2:

| Scenario | In/Out | c | R2 tok/s | R3 tok/s | Δ tok/s |
|---|---|---|---:|---:|---:|
| Single User Latency | 2048/512 | 1 | 71.2 | 87.9 | **+23.3%** |
| Decode Stress Test | 128/2048 | 1 | 73.1 | 91.8 | **+25.6%** |
| Concurrency Scaling c=2 | 1024/256 | 2 | 115.5 | 136.6 | **+18.3%** |
| Concurrency Scaling c=4 | 1024/256 | 4 | 199.2 | 230.0 | **+15.5%** |
| Long Context (16K) | 16384/1024 | 4 | 146.2 | 156.9 | +7.3% |
| Concurrency Scaling c=8 | 1024/256 | 8 | 305.4 | 310.4 | +1.6% |
| Mixed Traffic | 2048/512 | 8 | 294.8 | 300.5 | +1.9% |
| Concurrency Scaling c=16 | 1024/256 | 16 | 499.1 | 536.4 | +7.5% |
| Short Context Throughput | 512/256 | 16 | 417.5 | 459.2 | +10.0% |
| Concurrency Scaling c=32 | 1024/256 | 32 | 733.3 | 743.5 | +1.4% |
| Concurrency Scaling c=64 | 1024/256 | 64 | 1149.1 | 1103.6 | **−4.0%** |
| Concurrency Scaling c=128 | 1024/256 | 128 | 1506.0 | 1365.9 | **−9.3%** |

**Mean throughput ratio across all 12 scenarios: 1.08× (+8.3%).** Round 3 dominates round 2 across the practical range; regressions appear only at c=64 / c=128.

### Known regression at c=64+128

At c=1–32 the custom-op extern call is a net win because LLMM1/wvSplitK fire (n≤4) and replace rocBLAS for QKV / o_proj. At c=64+ the per-batch n exceeds the wvSplitK gate (n>4); we fall through to `F.linear` inside our custom op — but inductor cannot fuse that path because the custom op is opaque. Round 2's direct Python dispatch let inductor inline `F.linear → aten::mm` cleanly; round 3's custom op breaks that inlining for the high-n cases.

**Round 4 ticket K** (in `round4_candidates.md`): hoist the n>4 check up to the wrapper level so the custom op is only invoked when LLMM1/wvSplitK can fire. For high-n cases, return F.linear directly so inductor can inline it. ~30 lines, low risk; restores round 2 perf at c=64+128 while keeping all round 3 c=1–32 wins.

Tonight's call: **ship 5h+5j** because c=1–32 (the practical hot range) gains far outweigh c=64+128 losses, and the regression is fully understood and reversible.

---

# Things that did not ship and why

Stop here next time you're tempted to retry one of these on this model.

## Stage 5b — hipBLASLt
hipBLASLt is **1.08×–4.4× slower** than rocBLAS at every hot M=1 shape on gfx908. Don't enable.

## Stage 5c — vLLM `fused_moe` config tune (round-2 attempt)
`benchmark_moe.py` tunes the un-quantized `fused_moe_kernel`, not our `fused_moe_kernel_gptq_awq`. Even after surface-level field-name alignment (renaming the JSON to add `dtype=int8_w8a16` suffix and adding `SPLIT_K: 1`), the chosen tile shapes (`BLOCK_SIZE_K=256` + `num_warps=1`) violate kernel invariants and produce HIP illegal-memory-access during cudagraph capture. The tuner targets a different kernel; configs aren't transferable. Reference JSON kept at `docs/mi100_decode_opt/moe_tune_output/` for round-4 hand-edit attempts.

## Stage 5e — `VLLM_ROCM_USE_SKINNY_GEMM=1` env flag alone
No-op without Stage 5g. The flag is harmless when set; Stage 5g made it actually do something via our patched dispatch.

## Stage 5f — PyTorch TunableOp tune+replay
Tune pass shaved 0.17 ms during the tune itself, but replay regressed by 0.08 ms. Tuned algos don't help the LLMM1/wvSplitK paths (already faster than rocBLAS); they only modify cudagraph-capture batches we don't actually decode at. NO SHIP at c=1. CSVs kept at `docs/mi100_decode_opt/tunableop_results/` for round-4 c≥8 evaluation if rocBLAS becomes the hot path again at higher concurrency.

## Stage 5i — AITER triton GEMM at M=8/16/32
AITER `gemm_a16w16` is **0.33–0.45×** rocBLAS at M=8/16/32 for our QKV shapes. The ~50 µs gfx908 launch floor swamps the work. Confirms round 2 finding that AITER only wins at very large GEMMs (lm_head).

## Stage 5m — MoE block torch.compile wrap
The `round4_candidates.md` item D claim that the MoE block runs eager is **wrong** for this model. `Qwen3NextModel` (`vllm/model_executor/models/qwen3_next.py:458`) and `Qwen3_5Model` (`vllm/model_executor/models/qwen3_5.py:197-207`) both already carry `@support_torch_compile`. The MoE block runs inside the inductor graph already. Round 5g's note "MoE block is NOT in graph" referred to `SharedFusedMoE` extern Triton calls being opaque — not fixable by another decorator. NO ACTION (saved 2-3 hours of patch + risk).

## Stage 6a — Python-level Linear fusion
Strong test of the "fewer launches" thesis. Cudagraph already amortizes Python launch overhead to ~1 µs. The kernel itself is the lever, not launch count. Reverted.

## Stage 7 — CAR NaN under cudagraph
Root cause is HIP IPC pointer staleness on graph replay (CDNA1). Not source-level fixable; would need a HIP runtime patch. The 3.4 ms / 16% TPOT prize is real but unattainable without runtime work. See round 4 candidate E for the C++-rewrite alternative.

---

# Methodology notes

- **The inductor-escape-hatch was the headline win** because it unlocked round-2 work for the actual decode-time GEMMs. Round 2 shipped LLMM1/wvSplitK but they were only firing for ~30% of calls; round 3's Stage 5h made them fire for all of them.
- **Tree+LL all-reduce is strictly better than Ring at our message size.** Worth checking on every new gfx9 install — a one-line env-var change for ~5% TPOT.
- **Snap-installed Docker can't bind-mount paths under `/tmp/`** — silent failure to anonymous tmpfs. Use `/home/tyler/...` for docker bind-mount sources on this host. Burned ~12 min in Stage 5f tune pass before catching it.
- **`benchmark_moe.py` tunes the wrong kernel for GPTQ-quantized models.** Configs aren't transferable to `fused_moe_kernel_gptq_awq`; surface-level rename isn't enough. Need a quant-aware tuner or hand-edit.
- **Cudagraph hides Python launch overhead, not GPU compute floors.** Two stages (5e env flag, 6a Python fusion) confirmed this independently. Kernel-level swaps (5g, 5h) are the lever.

# Verification artifacts

- 3-run TPOT (canonical metric, c=1, 256-tok decode): **11.00 ms median, ±0.01 ms spread**
- Coherence: 4-prompt suite PASS pre AND post on every shipped stage
- BenchAndReport JSON: `/tmp/decode_opt/round3_results.json`
- Round 2→3 comparison: `/tmp/decode_opt/round3_comparison.md`
- Inductor trace dump (Stage 5k): `docs/mi100_decode_opt/stage5k_inductor_trace/` (gitignored)
- Tunableop CSVs (Stage 5f): `docs/mi100_decode_opt/tunableop_results/`
- MoE tuner output (Stage 5c): `docs/mi100_decode_opt/moe_tune_output/`

To inspect any shipped change: `git show 23bbb8696` (round 1), `git show 7db67cd92` (round 2 dispatch), `git show 2ae323c98` (round 3). Reproduction scripts under `docs/mi100_decode_opt/scripts/` — see `scripts/README.md`. Forward-looking work tracked in `round4_candidates.md`.
