# Stage 5f — PyTorch TunableOp tune + replay (no-ship at c=1)

**Date:** 2026-04-25
**Result:** TPOT 11.62 → **11.70 ms** = **−0.7% (small regression at c=1)**
**Decision: NO SHIP at c=1.** Defer ship/no-ship to full BenchAndReport — may still help at c≥8 where rocBLAS is on the hot path.

## Background

Round-2 summary listed Stage 5f as ~1–2 % expected gain by tuning rocBLAS dispatch for the QKV/QKVZ/o_proj GEMMs that go through `aten::mm` after inductor inlining (Stage 5g). With Stage 5h (custom-op wrapper) shipping, those GEMMs now route through LLMM1 / wvSplitK at runtime — meaning **rocBLAS only runs for cudagraph capture batches we don't actually decode at**. So the expected lever has effectively been swallowed by 5h.

## What we did

1. Ran `TUNABLEOP=tune bash test_stage5_baseline.sh` — 358 entries × 4 ranks captured to `tunableop_results{0,1,2,3}.csv` under `docs/mi100_decode_opt/tunableop_results/`. Tune-pass TPOT was **11.45 ms** (faster than 5h baseline by 0.17 ms).
2. Ran `TUNABLEOP=replay bash test_stage5_baseline.sh` — clean replay, no per-call timing. TPOT **11.70 ms** (slower than 5h baseline by 0.08 ms).

## Why tune was fast and replay is slow

In **tune** mode, PyTorch only uses a tuned algo if it found one. Untuned shapes (those not yet seen during the run) fall back to the default rocBLAS path. Apparently the default is faster than the "tuned best" for our cudagraph-capture batches.

In **replay** mode, every shape with a CSV entry uses the chosen algo, no fallback. Net: a small regression because the tuned algos are not faster than the defaults for shapes that don't dominate decode time.

## End-to-end measurement

Container: `vllm-rocm-gfx908:latest` + Stage 5g + Stage 5h overlays + TunableOp env. Same workload as Stage 5h.

| Mode | Run 1 | Run 2 | Run 3 | Median TPOT | vs Stage 5h |
|------|-------|-------|-------|-------------|------------|
| Stage 5h (no TunableOp) | 11.62 | 11.62 | 11.63 | **11.62 ms** | — |
| 5f tune                  | 11.45 | 11.45 | 11.46 | 11.45 ms     | −1.5 % (suspect) |
| 5f replay                | 11.70 | 11.70 | 11.71 | **11.70 ms** | **+0.7 % (regression)** |

Coherence PASS pre+post on both passes.

## CSV inventory

`docs/mi100_decode_opt/tunableop_results/tunableop_results{0..3}.csv` — 358 entries each (one per TP rank). Includes:
- `GemmAndBiasTunableOp_Half_TN` for prefill batches (M=65536)
- `GemmTunableOp_Half_TN` for cudagraph capture batches (M ∈ {2,4,16,256,480,496,512,...})
- Mostly `Gemm_Rocblas_-XXX` solution IDs (not `Default`) — meaning rocBLAS *did* find specific tuned algos. They just don't help the decode path because decode goes LLMM1.

## Decision

- **At c=1: NO SHIP.** Tuned algos don't help our hot path (LLMM1) and slightly hurt the cudagraph capture replay paths.
- **At higher c: TBD.** When concurrency ≥ 8, M for QKV/QKVZ becomes 8/16/32 — wvSplitK refuses (n>4 internal gate), so we fall back to rocBLAS. TunableOp would then be on the hot path. Need to measure via full BenchAndReport (Stage F).
- **Action for now:** keep CSVs in repo (under `docs/mi100_decode_opt/tunableop_results/`) but DO NOT enable TunableOp by default in any test script. If Stage F shows c≥8 wins, flip default; otherwise leave dormant for round-4.

## Snap-docker /tmp gotcha (important footgun)

The first tune run wrote CSVs into a tmpfs inside the container because the host bind mount `-v /tmp/decode_opt/tunableop_results:/host_tunableop` silently failed under snap-installed Docker (snap confinement disallows bind-mounts under `/tmp/`). The CSVs were visible inside the container but never reached the host, and were destroyed on container teardown. **Cost: 12 min of one tune-pass wallclock.** Documented in `~/.claude/projects/-home-tyler-aiter/memory/feedback_snap_docker_tmp_mounts.md`.

**Always use `/home/tyler/...` paths for docker bind-mount sources on this host.**
