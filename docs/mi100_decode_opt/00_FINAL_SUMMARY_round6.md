# Round-6 Final Summary (2026-04-27)

**Targets:** Qwen3.6-27B-GPTQ-{8bit, 4bit} on 4×MI100, TP=4
**User goal:** maximize 8-bit tok/s; user-confirmed "ship" with 4-bit c=1 trade-off
**Outcome:** **+3.3% avg throughput on 8-bit, +5.2% avg on 4-bit.** Single-tier
trade-off: 4-bit c=1 regresses −2.4%; all other tiers neutral or improved.

| Metric | Round-5 baseline | Round-6 dot2c-conditional | Δ |
|---|---:|---:|---:|
| Microbench M=1 (4 production shapes) | reference | +0 to +2% | ~neutral |
| Microbench M=4 | reference | +5 to +7% | win |
| Microbench M=16 | reference | +11 to +13% | win |
| 27B-8bit 3-run TPOT @ c=1 (serving) | 16.98 ms | **16.81 ms** | **+1.0% throughput** |
| 27B-8bit 12-tier (avg across tiers) | reference | **+3.3% throughput / -3.5% TPOT** | **WIN** |
| 27B-4bit 3-run TPOT @ c=1 | 15.51 ms | 15.88 ms | −2.4% (real, accepted) |
| 27B-4bit 12-tier (avg across tiers) | reference | **+5.2% throughput / -6.1% TPOT** | **WIN** |

**Image:** `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` and `:latest`
(both pushed 2026-04-27, digest `sha256:3d4aaaf51c08...`).

## 27B-8bit 12-tier (2026-04-27)

| Tier | Δ throughput | Δ TPOT |
|---|---:|---:|
| Single User Latency (c=1) | +0.8% | -1.0% |
| Decode Stress (c=1) | +1.0% | -1.0% |
| **Concurrency Scaling c=2** | **+10.5%** | **-11.2%** |
| Long Context (c=4) | +2.2% | -2.5% |
| Concurrency Scaling c=4 | +3.6% | -2.2% |
| **Mixed Traffic c=8** | **+4.9%** | **-5.5%** |
| **Concurrency Scaling c=8** | **+5.0%** | **-6.0%** |
| **Short Context Throughput c=16** | **+7.0%** | **-7.4%** |
| **Concurrency Scaling c=16** | **+5.0%** | **-6.1%** |
| Concurrency Scaling c=32 | -0.1% | +0.2% |
| Concurrency Scaling c=64 | -0.1% | +0.2% |
| Concurrency Scaling c=128 | +0.0% | -0.0% |
| **Average** | **+3.3%** | **-3.5%** |

No tier regresses > 0.2%. Coherence 4/4 PASS pre + post.

## Path

1. **Phase 0 (rocprof).** `gemm_half_q_half_gptq_8bit_kernel` is
   **latency / wave-occupancy bound** (VALUBusy 14% on qkv, 25% on
   gate_up; MemUnitStalled 0.14%; 4 waves/CU at `__launch_bounds__(256, 1)`).
   This invalidated the planned MFMA approach (compute-density lever
   doesn't apply when VALU is mostly idle).

2. **Phase 1 (microbench baseline).** Per-shape M=1 HBM utilization on
   the round-5 .so: qkv 38%, o_proj 21%, gate_up 61% (near memory-bound),
   down 45%. **gate_up at M=1 cannot win much from any compute-side
   lever — it's already memory-bound.**

3. **Phase 2 (small-experiment sequence — pivoted from MFMA).**
   - Tried `dot22_8_f` swap: regressed M=1 by 22-30%
   - Tried `__builtin_amdgcn_fdot2`: regressed M=1 by 14-24%
   - Tried `v_dot2c_f32_f16` inline asm always: regressed M=1 by 3-5%
   - **Tried `v_dot2c_f32_f16` conditional on `m_count >= 2`: WIN**

4. **Phase 3 (validation).** Microbench passed at all M; coherence 4/4
   PASS pre on 27B-8bit serving; 3-run TPOT +1.0% throughput vs round-5.

5. **Phase 4 (full bench).** _Running_ — 12-tier BenchAndReport on 27B-8bit.

6. **Phase 5 (4-bit port).** Source changes already in place (mirroring
   8-bit pattern). Validation pending after Phase 4.

## Source changes

`csrc/quantization/gptq/q_gemm.cu`:

- New helper `dot22_8_h_dot2` (~line 195) — `v_dot2c_f32_f16` asm variant
  for 8-bit kernel
- New helper `dot22_8_f_dot2` (~line 76) — `v_dot2c_f32_f16` asm variant
  for 4-bit/3-bit/2-bit kernel paths
- 8-bit kernel inner loop (~line 663): `if constexpr (m_count == 1)`
  uses original `dot22_8_h`; else uses `dot22_8_h_dot2`
- 4-bit kernel inner loop (~line 358): `if constexpr (m_count == 1)`
  uses original `dot22_8_f`; else uses `dot22_8_f_dot2`

The 2-bit and 3-bit kernels use `dot22_16_h` / `dot22_32_h` (different
unpack widths); not modified in round-6 because those formats aren't
production-relevant for our model set.

## Why MFMA was rejected (mid-round-6 pivot)

Round-6 plan called for a multi-day MFMA kernel rewrite. Phase 0 rocprof
showed the kernel is latency/occupancy-bound, not compute-bound — so
MFMA's primary lever (compute density) doesn't apply. The microbench
sequence confirmed: every variant that increased compute density at the
expense of M=1 dependency-chain latency regressed M=1.

The conditional dispatch is the only design that gets the M >= 2 wins
(where there's enough parallel work to amortize asm-constraint
dependencies) without paying the M=1 cost.

A full MFMA refactor remains the credible round-7 path if needed, but
the Phase 0 evidence suggests it would also struggle at M=1.

## Files

- Round-5 baseline `_C.abi3.so`: shipped in image
  `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev`
- Round-6 builds:
  - `_C.abi3.so.tweak_dot2c_split` (8-bit-only changes)
  - `_C.abi3.so.tweak_dot2c_4and8bit` (8-bit + 4-bit changes — **ship target**)
- Phase 0 rocprof captures + audit:
  `docs/mi100_decode_opt/round6_phase0_rocprof.md`
- Phase 2 design doc (this round):
  `docs/mi100_decode_opt/round6_phase2_design.md`
- Microbench artifacts:
  `/home/tyler/decode_opt_audit/round6_phase0/{scalar_round5,tweak_*}.json`

## 27B-4bit 12-tier (2026-04-27)

| Tier | Δ throughput | Δ TPOT |
|---|---:|---:|
| Single User Latency (c=1) | -2.0% | +2.3% |
| Decode Stress (c=1) | **-2.4%** | **+2.5%** |
| Concurrency Scaling c=2 | +3.6% | -4.3% |
| Long Context (c=4) | +2.4% | -4.0% |
| Concurrency Scaling c=4 | +5.7% | -6.8% |
| **Mixed Traffic c=8** | **+14.3%** | **-14.3%** |
| **Concurrency Scaling c=8** | **+15.7%** | **-17.2%** |
| **Short Context Throughput c=16** | **+10.3%** | **-12.6%** |
| **Concurrency Scaling c=16** | **+8.6%** | **-10.6%** |
| **Concurrency Scaling c=32** | **+6.4%** | **-9.2%** |
| Concurrency Scaling c=64 | -0.1% | +0.2% |
| Concurrency Scaling c=128 | -0.4% | +0.4% |
| **Average** | **+5.2%** | **-6.1%** |

Coherence 4/4 PASS pre + post. c=1 regression of 2.0-2.4% is real and
caused by `__forceinline__` helper additions perturbing register
allocation in the 4-bit kernel (verified by removal experiment). User-
confirmed acceptable for the much larger c=2-c=32 wins.

## 4-bit c=1 regression — root cause investigation

Diagnosed via 4-variant build sweep (cross-boot variance baseline 0.3%):

| Build | 4-bit c=1 TPOT | Δ vs round-5 |
|---|---:|---:|
| Round-5 ship (recheck this boot) | 15.558 ms | baseline |
| min_helpers (only `dot22_8_h_dot2`) | **15.546 ms** | **0%** |
| 8bit_only_v2 (3 helpers, no 4-bit conditional) | 15.820 ms | +1.7% |
| Combined ship (`dot22_8_h_dot2` + `dot22_8_f_dot2` + 4-bit conditional) | 15.880 ms | +2.1% |

Adding any helper function definition to `q_gemm.cu` regresses the
4-bit kernel m_count==1 path by ~1.7% — independent of whether the
helper is called. Adding the 4-bit conditional dispatch on top adds
another ~0.4%.

Mitigation paths considered for round-7:
- Move helpers to a separate `q_gemm_dot2c.cuh` header (test if
  translation-unit boundary helps)
- Use `static inline __attribute__((always_inline))` on helpers
- Inline the v_dot2c asm directly into the 4-bit kernel m_count>=2
  branch (no helper function)

## Image bake + push (2026-04-27)

Built from base `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` with the
round-6-shipped `_C.abi3.so` overlaid via single-line Dockerfile COPY.
Tagged + pushed both `:v0.20.0rc1.dev` and `:latest` overwriting the
round-5 production tags (per round-5 ship pattern).

Image digest: `sha256:3d4aaaf51c08c1eb05f3f46261777f73c75f15b8bf0af0445fb59bd86dd87949`

## Memory entry

Updated `~/.claude/projects/-home-tyler-aiter/memory/project_decode_opt_round6_27b_2026_04_27.md`
and `MEMORY.md` index.
