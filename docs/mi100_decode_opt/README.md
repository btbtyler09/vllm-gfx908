# MI100 Decode Optimization — Qwen3.6 on 4× MI100

Decode-throughput investigation for Qwen3.6 models on vLLM, targeting 4× AMD Instinct MI100 (gfx908) with XGMI peer-to-peer.

## Models tracked

| Model | Status | Headline result | Doc |
|---|---|---|---|
| Qwen3.6-35B-A3B-GPTQ-8bit | shipped (3 rounds, branch `mi100-optimized`) | TPOT 19.5 → 11.00 ms (**+76% throughput**) at c=1 | [qwen3_6_35b.md](qwen3_6_35b.md) |
| Qwen3.6-27B-GPTQ-8bit | investigated, no ship | baseline 20.43 ms TPOT @ c=1; round-3 levers don't transfer to a fully-GPTQ workload | [qwen3_6_27b.md](qwen3_6_27b.md) |

## The tuning playbook

This is the sequence we re-use per model. Each step either ships a patch or rules out a lever cheaply.

1. **Profile decode-only.** Long request → 1.5s warmup → `/start_profile` → 1.5s sleep → `/stop_profile`. Bucket kernels by name (linear-rocblas, all-reduce, moe-gemm, …). The biggest bucket is your target.
2. **Inventory real shapes.** Probe `rocm_unquantized_gemm_impl` (or the equivalent quantized path) for actual per-rank `(M, N, K)` calls per decode step. Microbench projections based on guessed shapes are usually wrong.
3. **Microbench candidate kernels** at the real shapes (rocBLAS vs LLMM1 vs wvSplitK vs AITER vs hipBLASLt). Single-GPU eager mode. Establish each kernel's per-launch floor on gfx908 (~7 µs for skinny kernels, ~50 µs for AITER, ~15–50 µs for rocBLAS).
4. **Wire the dispatch** in `vllm/model_executor/layers/utils.py` for the model's hot path. For unquantized layers this is `rocm_unquantized_gemm_gfx908`; for GPTQ layers the lever is in `vllm._custom_ops.gptq_gemm` (or a Triton replacement).
5. **Verify dispatch fires** with `VLLM_GFX908_DEBUG_DISPATCH=1` — count `[LLMM1]` / `[wvSplitK]` / `[AITER_DISPATCH]` lines per request. Layers wrapped in `@support_torch_compile` may inline your Python branches at compile time and skip the dispatch entirely — see step 6.
6. **Escape inductor where it hurts.** Wrap the dispatch as a `direct_register_custom_op` so inductor sees one opaque node (and runtime branching is preserved). Trade-off: the custom op also blocks fusion of `F.linear` fallback paths at high concurrency — provide a fast path that skips the custom op when no skinny kernel will fire.
7. **Tune NCCL** for the cluster's fabric. On 4× MI100 with XGMI: `NCCL_ALGO=Tree NCCL_PROTO=LL` saves ~5% TPOT vs the default Ring at small message sizes.
8. **Re-baseline with 3-run TPOT** (c=1, 256-tok decode) and full `BenchAndReport.py` (12 scenarios, c=1–128). Coherence smoke pre+post every change.

## Constraints worth knowing before you start

- **CAR is broken on gfx908 cudagraphs** — HIP IPC pointers go stale on graph replay (CDNA1 runtime behavior, not source-fixable). RCCL fallback costs ~3.4 ms / step (~16% TPOT). Fixing it would need either a HIP runtime patch or a custom point-to-point ring all-reduce in HIP.
- **`VLLM_ROCM_USE_AITER_TRITON_GEMM=1` MUST be set via `docker --env`,** not via `os.environ.setdefault`. `vllm/_aiter_ops.py:1144` caches `_TRITON_UNQUANT_GEMM` at class-definition time, before `vllm/platforms/rocm.py`'s `_GFX908_DEFAULTS` block runs.
- **Snap-installed Docker silently drops `/tmp` bind mounts** to anonymous tmpfs. Use `/home/tyler/...` for any host-side mount source on this box.
- **AITER Triton GEMM has a ~50 µs launch floor** on gfx908. Only beats rocBLAS when the GEMM is large enough to absorb the floor (lm_head only, for these models).
- **hipBLASLt is universally slower than rocBLAS** on gfx908 at our M=1 decode shapes. Don't enable.
- **`benchmark_moe.py` tunes the wrong kernel** for GPTQ-quantized models — its configs target `fused_moe_kernel`, not `fused_moe_kernel_gptq_awq`. Tile-shape invariants differ; a tuned config can crash with HIP illegal memory access.

## Forward-looking work

[`round4_candidates.md`](round4_candidates.md) — punch-list of levers that haven't shipped yet. Notable items:
- **K** (cheap, ~30 lines): hoist the n>4 check above the custom op so high-concurrency `F.linear` fallback gets back into the inductor graph. Restores round-2 perf at c=64+128 without losing round-3 wins at c=1–32.
- **A / D** (cheap): NCCL re-sweep on a new model, MoE block compile-wrap (model-dependent, doesn't apply to current Qwen3.6 variants).
- **E** (multi-day, ~16% TPOT prize): custom CAR replacement in HIP that re-exchanges IPC handles outside cudagraph capture.
- **F** (multi-day, ~5% TPOT): custom gfx908 MFMA fmoe expert kernel.

## Reproduction

[`scripts/`](scripts/) — test harnesses, microbenches, profiler tooling. See [`scripts/README.md`](scripts/README.md) for the per-script index. All scripts assume the `vllm-rocm-gfx908:latest` container with the patched `vllm/` mounted as overlays.

## Subdirectories

| Path | Contents |
|---|---|
| `scripts/` | Reproduction scripts (test harnesses, microbenches, profiler bucketizer, coherence smoke) |
| `tunableop_results/` | PyTorch TunableOp CSVs from Stage 5f (per-rank, 358 entries each). NOT enabled by default — kept for potential c≥8 evaluation. |
| `moe_tune_output/` | `benchmark_moe.py` output for E=256/N=128/MI100. Doesn't load against `fused_moe_kernel_gptq_awq` — kept as artifact for round-4 hand-edit. |
| `stage5k_inductor_trace/` | `TORCH_COMPILE_DEBUG=1` dump from Stage 5k fusion audit (~6.4 MB). Gitignored — paths/PIDs change per run. |

## Hardware & environment

- **Cluster:** 4× AMD Instinct MI100 (gfx908) with XGMI peer-to-peer
- **Workflow:** Docker — `vllm-rocm-gfx908:latest` with patched `vllm/` mounted as overlays
- **Branch:** `mi100-optimized`
- **Stack:** vLLM v0.19.2rc1+mi100 + AITER, `VLLM_MI100_TORCH_COMPILE=1`, mode-3 + FULL_AND_PIECEWISE cudagraph, TRITON_ATTN backend, dtype=half, TP=4
