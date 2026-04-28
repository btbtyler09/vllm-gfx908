# Round-6 Phase 2 — v_dot2c-conditional dispatch in GPTQ kernels (DRAFT)

**Date:** 2026-04-27
**Lever:** Replace inner `dot22_8_h` / `dot22_8_f` calls with v_dot2c-based
variants for `m_count >= 2` template instantiations only. Keeps original code
path for `m_count == 1` (decode at c=1) which microbenched as already-optimal.

## Why this instead of MFMA

Phase 0 rocprof identified the kernel as **latency / wave-occupancy bound**,
not compute-bound. MFMA's primary lever is compute throughput density, which
isn't our bottleneck. Three small experiments confirmed this:

| Variant | M=1 microbench | M=4 microbench | M=16 microbench |
|---|---:|---:|---:|
| Round-5 baseline (`dot22_8_h` fp32 fma chain) | 100% | 100% | 100% |
| `dot22_8_f` swap (matches 4-bit kernel pattern) | **−22 to −30%** | −9% | +3 to +19% |
| `dot22_8_h_dot2` asm `v_dot2c_f32_f16` (always) | −2 to −5% | +5 to +7% | +11 to +13% |
| `__builtin_amdgcn_fdot2` (always) | **−14 to −24%** | 0 to +1% | +5 to +25% |
| **`v_dot2c_f32_f16` conditional on `m_count >= 2`** | **0 to +2%** | **+5 to +7%** | **+11 to +13%** |

The conditional dispatch is the only variant that doesn't regress M=1 while
delivering the M >= 2 gains. Fits the user's "find lots of small improvements"
framing.

## Why the conditional split works

- `v_dot2c_f32_f16` does 2 fp16 mul-adds per instruction (vs scalar fma's 1).
  Halves the inner-loop instruction count.
- At `m_count == 1`, only 16 dot22 calls per outer K iter (4 N positions × 4 j
  unroll). Compiler can't interleave enough work to hide the asm-constraint
  dependency chain → small regression vs hand-tuned fma.
- At `m_count >= 2`, the inner loop has `m_count`× more dot22 calls. Compiler
  has enough parallel work in flight to fully amortize the asm pattern → wins
  from instruction-count reduction land.

## Source change summary

`csrc/quantization/gptq/q_gemm.cu`:

1. **New helper** `dot22_8_h_dot2` (after `dot22_8_h`, ~line 162):
   ```cpp
   __forceinline__ __device__ half dot22_8_h_dot2(half2 (&dq)[4],
                                                   const half* a_ptr,
                                                   const half g_result,
                                                   const half qs_h) {
     float result = 0.0f;
     const half2* a2_ptr = (const half2*)a_ptr;
   #pragma unroll
     for (int i = 0; i < 4; i++) {
       half2 w01 = dq[i];
       half2 a01 = *a2_ptr++;
   #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
       defined(__gfx941__) || defined(__gfx942__)
       asm("v_dot2c_f32_f16 %0, %2, %3"
           : "=v"(result)
           : "0"(result), "v"(w01), "v"(a01));
   #else
       /* fallback: scalar fma chain */
   #endif
     }
     return __hadd(__float2half_rn(result * __half2float(qs_h)), g_result);
   }
   ```

2. **New helper** `dot22_8_f_dot2` (after `dot22_8_f`, ~line 70): same idea,
   returns float (no scale applied) for use with the 4-bit kernel's
   pre-existing `fma(dot22_8_f(...), scales[i], block_c[m][i])` accumulator
   pattern.

3. **8-bit kernel inner loop** (line ~663): conditional dispatch
   ```cpp
   if constexpr (m_count == 1) {
     block_c[0][0] = dot22_8_h(dq[0], a_ptr, block_c[0][0], scales[0]);
     // ... 4 calls ...
   } else {
     #pragma unroll
     for (int m = 0; m < m_count; m++) {
       block_c[m][0] = dot22_8_h_dot2(dq[0], a_ptr + m * a_stride,
                                      block_c[m][0], scales[0]);
       // ... 4 calls ...
     }
   }
   ```

4. **4-bit kernel inner loop** (line ~296): same pattern with
   `dot22_8_f` / `dot22_8_f_dot2`.

## Microbench

Captured 2026-04-27 on round-5 image (`btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev`),
GPU 0 only, single-shape per measurement, 200 iters after 30 warmup, no
rocprof overhead.

**Combined 4-bit + 8-bit `_C.abi3.so.tweak_dot2c_4and8bit`** vs round-5
baseline (4 production shapes × M ∈ {1, 4, 16}):

To be filled in after benchmark validates.

## Validation

| Stage | Status |
|---|---|
| Compile gfx908 | PASS |
| Microbench M=1, M=4, M=16 (8-bit) | PASS — see above |
| 27B-8bit coherence-pre 4/4 | PASS (2026-04-27) |
| 27B-8bit 3-run TPOT @ c=1 | **16.81 ms / 59.49 tok/s** vs round-5 16.98 ms / 58.9 tok/s = **+1.0% throughput** |
| 27B-8bit 12-tier BenchAndReport | _running_ |
| 27B-8bit coherence-post 4/4 | _pending_ |
| 27B-4bit 3-run TPOT @ c=1 | _pending_ |
| 27B-4bit 12-tier BenchAndReport | _pending_ |

## Files

- Kernel changes: `csrc/quantization/gptq/q_gemm.cu`
  (8-bit kernel ~566-690, 4-bit kernel ~192-330, helper functions ~139-200)
- Built artifact: `/home/tyler/vllm-gfx908/vllm/_C.abi3.so.tweak_dot2c_4and8bit`
  (8-bit-only intermediate at `.tweak_dot2c_split`)
- Microbench artifacts: `/home/tyler/decode_opt_audit/round6_phase0/*.json`
- Microbench script: `docs/mi100_decode_opt/scripts/test_mfma_kernel/test_mfma_microbench.py`

## What didn't work (rejected variants)

1. **`dot22_8_f` swap on 8-bit kernel** — regressed M=1 by 22-30%. The fp16
   hfma2 + fp32 final reduction takes more VGPRs (4 floats vs 4 halfs in
   accumulator) which interacts poorly with `__launch_bounds__(256, 1)`.
2. **`__builtin_amdgcn_fdot2` (clang built-in)** — regressed M=1 by 14-24%.
   Compiler emits a different instruction encoding (probably `v_dot2_f32_f16`
   3-source form) requiring extra register copies vs the asm `v_dot2c` C-form.
3. **`v_dot2c` always** (no conditional) — small −3 to −5% regression at M=1.

The conditional dispatch (only fire `v_dot2c` when `m_count >= 2`) recovers
the M=1 baseline while keeping the M >= 2 wins.

## Open questions

1. Does the 12-tier BenchAndReport show tier-by-tier wins matching the
   microbench (M=1 ~0%, M=4 ~+5-7%, M=16 ~+11-13%)?
2. Does the 4-bit version of the same change deliver similar wins?
3. Is there a similar conditional opportunity for the 2-bit / 3-bit kernels?
   (Not pursued in round-6; low impact since those formats are rare.)
