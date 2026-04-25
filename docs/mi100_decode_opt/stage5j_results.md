# Stage 5j — NCCL/RCCL parameter sweep (SHIP)

**Date:** 2026-04-25
**Result:** TPOT 11.62 → **11.08 ms** = **−4.6% (Tree+LL combined)**
**Decision: SHIP `NCCL_ALGO=Tree NCCL_PROTO=LL`** as default env vars in test scripts and Dockerfile/launch invocations.

## Background

Per round-2 conclusion, all-reduce takes ~3.4 ms / step (~16% of TPOT) because canonical CAR is broken on gfx908 cudagraphs (CDNA1 IPC pointer staleness on graph replay) and we fall back to RCCL. RCCL is configurable via env vars; defaults assume PCIe-style fabric, but our box is 4× MI100 with XGMI peer-to-peer.

## Method

Each launch reused the Stage 5h overlay container (custom-op patched `utils.py`, no TunableOp). 3-run TPOT @ c=1, 256-tok decode. Coherence pre + post.

## Results

| Config | env vars added | run 1 | run 2 | run 3 | TPOT median | vs 11.62 baseline |
|--------|---------------|------:|------:|------:|------------:|-----------------:|
| Baseline (Stage 5h ship) | (none — Ring algo, default proto) | — | — | — | 11.62 ms | — |
| Tree algo | `NCCL_ALGO=Tree` | 11.11 | 11.11 | 11.11 | **11.11 ms** | **−4.4%** |
| LL proto only | `NCCL_PROTO=LL` | 11.59 | 11.60 | 11.61 | 11.60 ms | −0.2% (noise) |
| **Tree + LL** | both | 11.08 | 11.07 | 11.08 | **11.08 ms** | **−4.6%** ✅ |

Coherence PASS pre + post on all three configs.

## Why Tree wins

Default NCCL all-reduce on 4 ranks uses Ring algorithm (latency = 2(N−1)·msg/N·BW). Tree algorithm uses log-N depth (2·log2(N)·msg/BW for binary tree). For our small per-step all-reduces (~few KB at most across 4 GPUs), Tree has lower latency because the Ring algorithm pays N·hop-latency that dominates for small messages. Once messages hit the bandwidth-bound regime (large-batch training all-reduces), Ring wins. Our decode-time all-reduces are firmly in the latency-bound small-message regime.

## Why LL adds little on top of Tree

`NCCL_PROTO=LL` (low-latency) cuts protocol overhead by inlining the data flag bit into the payload (saves a separate sync). Tree already cuts the latency-sensitive part of the cost; LL adds a marginal further trim (0.03 ms ≈ within run-to-run noise). Free to enable since it costs nothing.

## What this means at higher concurrency

At c≥8 the per-step compute time grows but the all-reduce message size stays roughly constant (or grows linearly with hidden_size, not concurrency). So the absolute 0.5 ms savings should hold; the relative percentage shrinks because per-step total time grows.

## Action

- Default env vars `NCCL_ALGO=Tree NCCL_PROTO=LL` should be added to:
  - `docs/mi100_decode_opt/scripts/test_stage5_baseline.sh`
  - `docs/mi100_decode_opt/scripts/test_stage_combined.sh`
  - The `_GFX908_DEFAULTS` dict in `vllm/platforms/rocm.py` so they stick across all gfx908 launches automatically.
- Document in user-facing notes (`docker_run_mi100_pattern.md` or equivalent) that these env vars are the recommended defaults for 4×MI100 RCCL fallback.

## Stop conditions hit?

No. All coherence PASS, all configs stable, sweep was ≤60 min wallclock (3 container restarts). Moving on to Stage 5k.
