# Round-6 Phase 0 — rocprof instruction profile of `gemm_half_q_half_gptq_8bit_kernel`

**Date:** 2026-04-27
**Image:** `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` (round-5 ship)
**Tooling:** rocprofv3 PMC capture, microbench harness invoking `ops.gptq_gemm` directly
**Dispatches per pass:** 250 (warmup 50 + iters 200)

## TL;DR

Kernel is **latency / wave-occupancy-bound**, NOT compute-bound and NOT
memory-bound. The plan's Phase 0 decision branch (compute-bound vs
memory-bound vs balanced) didn't anticipate this fourth case.

**Decision:** Proceed to Phase 2 with **lever B' (16×16×16 MFMA)** — smaller
VGPR footprint reduces regression risk. Realistic yield estimate revised
**downward to 5-15%** (was 15-30%). If Phase 3 microbench shows MFMA
can't beat the scalar kernel on any production shape, **STOP per plan
rule.**

## Captured counters

| Counter | qkv (5120×3584) | gate_up (5120×8704) | Reading |
|---|---:|---:|---|
| GPUBusy | 100% | — | Kernel dispatched, fills all CUs |
| **VALUBusy** | **13.9%** | **25.2%** | Vector ALU mostly idle |
| **MemUnitBusy** | **20.5%** | **37.1%** | Memory unit moderately active |
| **MemUnitStalled** | **0.14%** | **0.11%** | Memory unit not blocked on credit/queue |
| LDSBankConflict | 0 | — | Clean LDS access |
| SALUBusy | 51% | — | Scalar ALU half-busy (address compute) |
| SQ_INSTS_VALU | 1,332,539 | — | Total VALU instructions per dispatch |
| SQ_INSTS_VMEM_RD | 23,600 | — | Total VMEM read instructions |
| SQ_INSTS_VMEM_WR | 3,801 | — | Mostly atomicAdd output writes |
| FetchSize | 19.6 MB | 46.5 MB | Bytes fetched from HBM per dispatch |
| VGPR_Count | 72 | 72 | Per-wave VGPR allocation |
| Accum_VGPR_Count | 72 | 72 | Per-wave AGPR allocation (reserved but unused) |
| SGPR_Count | 32 | 32 | Per-wave SGPR allocation |

Per-shape FetchSize ÷ weight bytes:
- qkv: 19.6 MB / 17.5 MB (W8 weights) = 112% (W + scales/zeros + activation reuse)
- gate_up: 46.5 MB / 42.5 MB = 109%

Both shapes show `MemUnitStalled` ≈ 0.1% — kernel is not waiting on
memory unit credits. Memory **demand** is moderate but well within
hardware limits.

## Derived metrics

- **VALU/VMEM_RD ratio:** 1,332,539 / 23,600 = **56.5** instructions per memory read
  - Compute-bound regions: > 200
  - Memory-bound regions: < 10
  - **Moderate arithmetic intensity** — not pinned to either limit

- **Wave occupancy estimate:**
  - 256 threads/block → 4 waves/block
  - `__launch_bounds__(256, 1)` → 1 block/CU at most
  - Per-CU active waves = **4 / 40 max = 10% wave occupancy**
  - VGPR (72) + AGPR (72) = 144/512 → 28% per-wave register footprint;
    occupancy is **not VGPR-limited**, it's gated by `__launch_bounds__`
  - The round-5 sweep showed `__launch_bounds__(256, 2)` regressed
    (compiler register-spilled when forced to fit 2 blocks/CU) — so 4
    waves/CU is the empirical sweet spot for the scalar kernel

## Why the kernel sits at 14% VALUBusy

With 4 waves per CU and 4 SIMDs per CU = **1 wave per SIMD** at peak.
gfx908 issues 1 wave's instruction per cycle per SIMD, so per-SIMD
peak utilization = 100% (in theory). We're at 14% (qkv).

The 86% gap is **wave stalls**:
- `s_waitcnt vmcnt(N)` waiting for memory loads to return (200-400 cycles)
- Pipeline dependencies (`dot22_8_h` chained mul-adds can't issue back-to-back)
- The SIMD has no other resident wave to context-switch to (we're at 1 wave / SIMD)

This is **classic latency-bound, low-occupancy** kernel behavior.

## Why this is NOT a clean MFMA win

The plan assumed compute-bound → MFMA brings 32× compute density
→ 15-30% throughput. Reality:

1. **VALU isn't the bottleneck.** It's idle 86% of the time. Replacing
   scalar with MFMA fills the same idle gaps; doesn't speed memory return.
2. **MFMA may worsen latency.** `v_mfma_f32_32x32x8f16` has a 16-cycle
   issue latency. During those cycles, the wave can't issue other
   instructions — same wave-stall problem in a different form.
3. **VGPR pressure risk.** A 32×32 fp32 accumulator needs 32 VGPRs
   per wave (one fp32 per output slot per lane). Current kernel uses 72.
   Adding 32 → 104. With `__launch_bounds__(256, 1)` we're already
   capped at 1 block/CU; adding VGPR forces compiler to spill OR drop
   below the binding cap → regression.
4. **AGPR offers a partial mitigation.** `Accum_VGPR_Count = 72` shows
   the compiler already reserves 72 AGPRs (unused by scalar kernel).
   MFMA's `c` operand (accumulator) can use AGPRs natively → could
   absorb the 32-AGPR accumulator without touching VGPR. **This is the
   only architectural reason MFMA might still help here.**

## Why MFMA might still give 5-15%

- **Instruction-issue rate reduction.** Each MFMA replaces ~32 scalar
  fmac. Per-wave instruction count drops ~32×. Wave finishes its
  compute faster → spends LESS time issuing → MORE time effectively
  hidden by memory latency overlap. Hard to size without a working
  prototype.
- **Better K-axis utilization.** The current kernel does 4 unrolled
  `dot22_8_h` calls per outer K iteration; each call has a serial
  dependency chain. MFMA computes the same K-fan-in in 1 instruction
  with no per-element dependency.

## Decision branch outcome

| Plan classification | Reality | Action |
|---|---|---|
| VALUBusy > 70% → COMPUTE-BOUND | NO (14-25%) | — |
| VALUBusy 50-70% & MemBusy 50-70% → BALANCED | NO (low both) | — |
| VALUBusy < 50% & MemBusy > 70% → MEMORY-BOUND → STOP | NO (MemBusy 20-37%) | — |
| **NEW: low VALU + low Mem + zero MemStalled → LATENCY/OCCUPANCY-BOUND** | **YES** | **Proceed to Phase 2 with B' (smaller tile)** |

## Updated lever inventory & ceiling

| Lever | Original yield | Revised yield | Rationale |
|---|---|---|---|
| B (32×32×8 MFMA) | +15-30% | +5-15% | VGPR risk too high; tile-waste at M=1 |
| **B' (16×16×16 MFMA)** | +10-20% | **+5-15%** | Smaller VGPR; AGPR mitigation possible |
| C (4-bit MFMA port) | +10-20% | +5-10% | Same dynamics as 8-bit |

**Round-6 realistic ceiling:** TPOT 16.98 → ~14.4 ms = **+15-18%
throughput** (~68 tok/s c=1) if everything lands at the high end.
Conservative outcome: 0% (no win, revert).

## Path forward

1. **Phase 1** — build microbench scaffolding to measure scalar-kernel
   per-shape µs at `M = {1, 4, 16}`. Establishes the A/B target.
2. **Phase 2** — prototype 16×16×16 MFMA kernel using AGPR for
   accumulator. Keep VGPR ≤ 80.
3. **Phase 3** — numerical + microbench gate. **Hard rule from plan:
   if MFMA slower at every shape, STOP.**
4. **Phase 4** — full BenchAndReport on 27B-8bit. Ship gate ≥5% TPOT
   improvement, no tier regresses >2%.
5. **Phase 5** — 4-bit port if Phase 4 ships.

## Files

- Microbench harness: `/home/tyler/decode_opt_audit/round6_phase0/profile_round6_microbench.py`
- Counter set definitions: `counters_pass{1,2,3}.txt`
- Capture runner: `profile_round6_rocprof.sh`
- Counter aggregator: `parse_counters.py`
- Captured CSVs: `/home/tyler/decode_opt_audit/round6_phase0/captures/pass{1,2,3}_{qkv,gate_up}/pmc_1/capture_counter_collection.csv`

## Caveats

- Microbench runs the kernel under rocprofv3 instrumentation, which
  imposes ~9× overhead on wall-clock time. The **counter values** are
  valid; the **timing** in this capture is not representative.
- Per-kernel HBM utilization measured here (~38% gate_up) is not the
  same as system-level HBM utilization at decode (where 256+ kernel
  calls/token compete). Round-5's "30-50% HBM peak" was system-level.
- We did not capture ATT (Advanced Thread Trace) instruction-level
  data — would have given dependency-chain stall details. Reserved
  for round-7 if MFMA proves marginal.
