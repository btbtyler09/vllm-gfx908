# Faster W4 GEMV kernels for gfx908 decode — design study (2026-09-03)

Scope: weight-only-int4 (GPTQ sym, GS32, bf16 scales) GEMV/GEMM for M <= 16 on MI100
(gfx908, CDNA1, 120 CU, 4 SIMD/CU, wave64, 1.2 TB/s HBM2, sclk 1502 MHz). Read-only
study; nothing here has been run. All cycle/byte numbers are estimates from the ISA and
the kernel structure, labelled as such. Sources are in section 6.

## 0. TL;DR

1. The MoE HIP GEMV (`w4gemv3_kernel`, 96 launches x 23.9 us = 2.3 ms/token at c=1) moves
   ~6.9 MB/layer/rank in 48 us = **~145 GB/s, ~12% of HBM** (not 40-50%). It is not
   bandwidth-bound; it is latency + VALU bound with ~1 wave per SIMD and a serialized load
   chain. A bandwidth-shaped rewrite should bring the pair of launches to ~9-11 us/layer
   (**-1.8 ms/token, ~-11% step time at c=1**), and at M=4 from 157 us/layer to ~35
   (**-5.9 ms/step at c=4**).
2. The winning structure on this GPU is the one llama.cpp, wvSplitK and exllama all use:
   lanes along K (coalesced 256 B-1 KB per wave-load), every weight load for the wave issued
   before the first use, int8 (`v_dot4_i32_i8`) or packed-f16 (`v_dot2_f32_f16`) inner
   products, DPP row reductions, no split-K, direct output.
3. W4A8 with **block-32 int8 activations** (Q8_1 style, fp32 scale + fp32 scale*sum) is what
   every llama.cpp HIP user runs by default; its error is ~8x below the W4-GS32 weight error.
   It cuts VALU per MAC ~10x vs the current fp32 path. Still gate on PPL 3.14 + GSM8K.
4. Int8/f16 MFMA is bandwidth-neutral at M=1 (no gain over a good dot kernel) but is the
   only way to stay bandwidth-bound for 4 <= M <= 16 (dense qkv/o_proj at c=16 are
   VALU-bound today, ~26 us vs ~6 us achievable).
5. Multi-GEMV descriptor launches (MoE experts + shared expert in one kernel) save ~4
   launches/layer (~0.25 ms/token). Do NOT bring back the last-block finalize.

## 1. Where the time goes now (c=1, per token, rank 0)

From `_profile_c1_launch_map.txt` (profiler adds ~2.4 us per kernel; wvSplitK and w4gemv
numbers are ~real):

| kernel | launches | mean us | ms/token | note |
|---|---|---|---|---|
| `w4gemv3_kernel` (MoE, HIP) | 96 | 23.9 | 2.30 | 2/layer: gate_up (N=320,K=2560) and down (N=2560,K=160), P=10 pairs |
| `triton_w4a16_gemv_partial_kernel` (dense) | 120 | 8.0 | 0.96 | qkv 2560->3584 (4.6 MB), o_proj 1536->2560 (2 MB), shared gate_up 2560->320 (0.4), shared down 160->2560 (0.2) |
| MoE reduces (`_reduce_silu_mul`, `_reduce_gate`, splitk_reduce) | ~167 | 4-9 | ~0.9 | split-K partial sums + epilogues |
| `wvSplitK_hf_sml_` (bf16 GDN/HC/router) | 365 | 11.9 | 4.34 | already 65-98% of HBM (campaign log); out of scope |

Bytes per layer per rank (10 routed pairs): gate_up 320x2560/2 = 410 KB + 51 KB scales;
down 2560x160/2 = 205 KB + 26 KB scales; x10 = **6.9 MB**. At 1.2 TB/s that is 5.8 us;
at a realistic 70-80% for a 7 MB working set with ramp, 8-9 us. Current: 48 us.

### 1.1 Why `w4gemv_kernel` is slow (reading the source, `gfx908_w4gemv.hip`)

Config at M=1 (`gfx908_moe_hip.py::_CFG[1]` = SK1 8, BN 64, SK2 1, BN 128):
- gate_up grid = (320/64=5, 8, 10) = **400 blocks of ONE wave**, down grid = (20,1,10) = 200
  blocks of 2 waves. 480 SIMDs -> at most ~1 wave per SIMD. Nothing overlaps with anything.
- Thread-per-column: lane n reads its own row of `[N, K/8]` uint32 -> lanes are 1280 B apart
  (gate_up). Each `uint4` wave-load touches 64 distinct 64 B lines for 1 KB of payload
  (TA processes ~1 line/cycle -> ~64 cycles/instruction vs 16 for a contiguous 1 KB).
- Load chain: 10 `uint4` loads per lane (k_per_split=320) with a **1-deep prefetch**
  (`q4n`), i.e. ~5 exposed HBM round-trips (~0.7-1 us each) = 4-5 us just waiting.
- Scales: `bf2f(sp[(k_begin+kk)/gs])` — a 2-byte global load per uint32, **issued inline
  and consumed immediately**, uncoalesced (lanes 160 B apart). 40 such loads per lane.
- Inner loop per nibble: shift, and, cvt_f32, sub 8, mul scale, ds_read x, fma = ~6 VALU
  + LDS per MAC. Per wave 320 k x 6 = ~2000 VALU = ~8000 cycles = **5.3 us of pure VALU**
  on its SIMD, serialized after the loads because there is no second wave to overlap.
- LDS staging of x: 320 bf16 per wave by 64 lanes (5 dependent 2 B loads), then `__syncthreads`.

Sum of the serialized parts (~5 us loads + ~5 us VALU + ~1 us x staging + ~1.3 us launch +
tail) ≈ the observed 24 us. The down launch (K=160, 5 loads/lane, 200 blocks) is latency-
bound the same way. Conclusion: the fix is **memory-level parallelism and VALU count**,
not micro-tuning of the current loop.

### 1.2 Bandwidth arithmetic for gfx908 (use these when sizing any design)

- HBM 1.2 TB/s / 1.502 GHz = **800 B/clk chip = 6.7 B/clk/CU = 1.7 B/clk/SIMD**.
- Loaded-latency ~1.5-3k cycles -> to sustain 6.7 B/clk a CU needs **>= 10-20 KB of loads
  in flight**, e.g. 4 waves x 4 outstanding `dwordx4` (1 KB each per wave), or 2 waves x 8.
- VALU: one wave64 op = 4 cycles on a SIMD16 -> 16 lane-ops/clk/SIMD. Budget so that
  VALU-cycles per KB of weights <= ~600 (= 1 KB / 1.7 B/clk), i.e. **<= 150 VALU ops per
  wave per 1 KB of weights** to stay memory-bound with a single wave per SIMD. Current
  kernel: ~640 ops/KB (VALU-bound even with perfect overlap). dot4 design below: ~20 ops
  per 16 B per lane = 20 ops per 1 KB per wave. 30x headroom.
- MFMA (matrix pipe, separate from VALU): `v_mfma_i32_16x16x16i8` 4096 MAC / 32 cycles,
  `v_mfma_i32_4x4x4i8` (16 blocks) 1024 MAC / 8 cycles, both 128 MAC/clk/SIMD (= 184.6 TOPS
  chip). `v_mfma_f32_16x16x8bf16` is half that rate (92 TF).
- Launch/dependency floor in a graph ~1.3 us per kernel (measured earlier this campaign).

### 1.3 gfx908 instruction facts (verified with ROCm 7.2 `llvm-mc -mcpu=gfx908` on the host)

Present: `v_dot4_i32_i8`, `v_dot4c_i32_i8`, `v_dot8_i32_i4` (int4 x int4 only), `v_dot2_f32_f16`,
`v_dot2c_f32_f16`, `v_dot2_i32_i16`, `v_pk_fma_f16`, `v_pk_mul_f16`, `v_pk_add_f16`, `v_pk_mad_i16`,
`v_and_or_b32`, `v_or3_b32`, `v_lshl_or_b32`, `v_bfi_b32`, `v_bfe_u32`, `v_perm_b32`, `v_fma_mix_f32`,
`v_cvt_pkrtz_f16_f32`, `v_sat_pk_u8_i16`, SDWA byte selects, `v_mfma_i32_{4x4x4,16x16x4,16x16x16,32x32x4,32x32x8}i8`,
`v_mfma_f32_{4x4x4,16x16x16,32x32x8}f16`, `v_mfma_f32_{4x4x2,16x16x2,16x16x8,32x32x2,32x32x4}bf16`
(non-1k, 2 bf16/lane/operand), MFMA `cbsz/abid/blgp`, `v_accvgpr_read`, `global_atomic_add_f32`
and `global_atomic_pk_add_f16` (**no-return form only**), `buffer_atomic_add_f32`, `ds_bpermute_b32`,
`ds_swizzle_b32`, `ds_read_b128`, `s_load_dwordx16`, `s_buffer_load_dwordx16`, `buffer_load_dwordx4`,
DPP `row_shr/row_shl/row_bcast:15/row_bcast:31`, `v_readlane_b32`. `v_dot4_i32_i8` accepts an
SGPR operand (`v_dot4_i32_i8 v0, v1, s2, v3` assembles).
Absent: `v_mfma_*bf16_1k`, `v_pk_fma_f32`, `v_dot2_f32_bf16`, `v_mad_mix_f32` (use `v_fma_mix_f32`),
returning float atomics, `lop3`/`prmt` (use `v_and_or_b32` / `v_perm_b32`).
Note: `reference_gfx908_isa.md` lists `ds_bpermute_b32` as gfx90a+; it assembles for gfx908
(it is GFX8+). Correct that memory when convenient.

## 2. What the reference implementations do, and what ports

### 2.1 llama.cpp `mul_mat_vec_q` (HIP/CDNA path) — the model to copy
- `ggml/src/ggml-cuda/common.cuh::ggml_cuda_dp4a`: `#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)`
  -> `__builtin_amdgcn_sdot4(a, b, c, false)`. So on MI100 every quantized GEMV is an int8 dot product.
- Activations are quantized per op to **Q8_1** (`ggml-common.h`: `{half d; half s /* d*sum(qs) */; int8 qs[32]}`)
  by `quantize.cu::quantize_q8_1` (warp-reduce absmax over 32, store d and d*sum). The sum
  lets the weight zero-point be folded: `vec_dot_q4_0_q8_1_impl`:
  `vi0 = v & 0x0F0F0F0F; vi1 = (v>>4) & 0x0F0F0F0F; sumi = dp4a(vi0,u0,dp4a(vi1,u1,..));`
  `return d4 * (sumi*ds8.x - 8*ds8.y)` — **no per-element subtract, no float dequant**.
- `mmvq.cu::calc_nwarps` GCN table: 2 warps (wave64) for ncols_dst <= 4, 1 above; 1 row per
  block, `blocks_per_iter = vdr*nwarps*warp_size/qi` — lanes stride along K (coalesced),
  `warp_reduce_sum` (shfl_xor) at the end, batched `ncols_dst` up to 8 tokens reuse each
  weight load (`tmp[j][i]`). Multi-token MoE path (`mul_mat_vec_q_moe`) uses `ids` per column.
- Public MI100 number (llama.cpp discussion #15021): Llama-2-7B Q4_0 tg128 = **110.5 t/s**
  (ROCm 6.4.1). 3.8 GB of weights x 110/s = ~420 GB/s end to end including attention, norms
  and launch gaps, so the mmvq kernels themselves run at >= ~50% of HBM on gfx908.
  MI210 (same design, 1.6 TB/s) gets 124-130 t/s; MI50 (gfx906, 1 TB/s) 99-106 t/s.
- gfx906 fork (iacopPBK/llama.cpp-gfx906): DPP-based warp reductions, "half-warp (32 thread)
  dispatch for MoE small matrices", software-pipelined loads to hide `v_perm` latency.

### 2.2 exllamav2 / vLLM `q_gemm.cu` (in tree, hipified, gfx908-tuned)
- `csrc/libtorch_stable/quantization/gptq/qdq_4.cuh::dequant_4bit_8_gptq`: the 0x6400 trick,
  `(q & 0x000f000f) | 0x64006400` = half2(1024+q0, 1024+q1), `(q & 0x00f000f0)|0x6400..` =
  half2(16*q2+1024, ..) then `__hfma2(.., 1/16, -1024-z)`. Needs nibbles pre-shuffled
  (`shuffle_4bit_8`: `77775555 33331111 66664444 22220000`) so only ONE shift is needed.
  On gfx908: `(q & m) | c` is a single `v_and_or_b32`; `__hfma2` is `v_pk_fma_f16`. Ports 1:1.
- `q_gemm.cu` 4-bit kernel (used by the 27B GPTQ models): `BLOCK_KN_SIZE 256` ("4 wavefronts
  x 64 = optimal SIMD occupancy on gfx908"), 4 columns per thread, `block_a` in LDS, 2-stage
  prefetch (comment: "latency-bound at decode M (MemUnitBusy ~9%, VALUBusy ~13%); prefetch
  worth ~10%"), `v_dot2c_f32_f16` inline asm for m_count >= 2, fp16 `atomicAdd` split-K.
  Same disease as our MoE kernel (thread-per-column, shallow prefetch); the numbers confirm
  the diagnosis in 1.1.

### 2.3 vLLM `moe_wna16.cu` (CUDA only)
`csrc/libtorch_stable/moe/moe_wna16.cu`: thread-per-column (`offset_n = blockIdx.y*BLOCK_SIZE_N + threadIdx.x`),
activations staged to `extern __shared__` with a permuted k order matching the dequant output,
one `float4` weight load per 4 uint32, scales for `GROUPS` groups loaded once as a `float4`,
`dequant<half2,4>` via `lop3` + 0x6400 then `__hfma2(__hmul2(__hsub2(w,z),s), x, acc)`,
`float res[64]` over tokens, fp16 `atomicAdd` to output (split-K over `blockIdx.z`), topk weight
applied in the epilogue. Ports: everything except `lop3/prmt` (-> `v_and_or_b32`/`v_perm_b32`).
Its per-token loop structure (reuse one weight fragment across all `num_valid_tokens`) is
the right way to batch tokens routed to the same expert at M > 8.

### 2.4 AWQ gemv
`csrc/libtorch_stable/quantization/awq/dequantize.cuh::dequantize_s4_to_fp16x2`: 4 `lop3` with
masks 0x000f000f / 0x00f000f0 and one `>> 8`, relying on AWQ's interleaved packing
(elements 0,2,4,6 | 1,3,5,7). Same magic; same port.

### 2.5 wvSplitK (`csrc/rocm/skinny_gemms.cu:350-570`, the 65-98%-of-HBM bf16 kernel)
- Whole activation matrix (`Kap*N` elements, <= 32 KB) copied to LDS once per WG.
- One wave per `YTILE` weight columns; lanes stride along K with `A_CHUNK=8` elements
  (16 B) per lane per step, `UNRL` steps' loads issued back to back (`loadnt` = non-temporal),
  then `DOT2C` (`v_dot2c_f32_f16`) per pair; persistent loop `m += CuCount*_WvPrGrp*YTILE`.
- Reduction: `row_shr:8,4,2,1` DPP adds then `row_bcast:15` / `row_bcast:31` (gfx9 path),
  lane 63 writes `C`. No split-K, no atomics, no second kernel.
This is the skeleton to reuse; replace the bf16 pair-dot with nibble unpack + int8 dot.

## 3. Ranked designs

### D1 — "K-slab" W4A8 dot4 GEMV (MoE gate_up, dense qkv/o_proj, any K >= 512)  [rank 1]

Geometry: `LPR` lanes per output row, `64/LPR` rows per wave, each lane owns whole GS32
groups (one `uint4` = 32 nibbles = exactly one group = one bf16 scale). For K=2560
(80 groups) use LPR=16: 4 rows/wave, lane `l` owns groups `l, l+16, l+32, l+48, l+64`
(5 x 16 B). A wave-load instruction then reads 4 rows x 256 B contiguous = 16 full
lines, no repack of the stock `[E, N, K/2]` (or dense `[N, K/8]`) layout. For small N
(shared-expert gate_up N=320) use LPR=64 (1 row/wave, lanes 0-15 take a second group) to
get >= 300 waves without split-K.

Activation format (produced by the preceding kernel, see 3.6): `x8` int8 `[P][K]` with the
8-element blocks de-interleaved (even k then odd k within each 8, so that the stock nibble
order `lo=k even, hi=k odd` needs no weight shuffle), `xs` fp32 `[P][K/32]` (absmax/127),
`xsum` fp32 `[P][K/32]` = `xs * sum(q8)`. fp32 scales, not fp16 (ik_llama.cpp #196: fp16 `d`
overflows on models with activations beyond fp16 range).

```cpp
// grid: tiles = P * ceil(N / ROWS); block 256 = 4 waves; one 4-row tile per wave.
template <int K, int LPR = 16>                 // K=2560: G=80 groups, GPL=5 groups/lane
__global__ __launch_bounds__(256) void w4a8_slab(
    const uint32_t* w, const uint16_t* s, const int8_t* x8, const float* xs,
    const float* xsum, const int* row_tok, const int* row_exp, float* out,
    int N, long stride_we, long stride_se) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = G / LPR;   // (K % (32*LPR) == 0 case)
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR;
  const int tile = blockIdx.x * 4 + (threadIdx.x >> 6);
  const int p = tile / (N / ROWS), n0 = (tile % (N / ROWS)) * ROWS, n = n0 + r;
  const int e = row_exp[p], t = row_tok[p];
  if (e < 0) { if (l == 0) out[(long)p * N + n] = 0.f; return; }     // capture routing
  const uint4*    wp = (const uint4*)(w + e * stride_we + (long)n * (K / 8)) + l;
  const uint16_t* sp = s + e * stride_se + (long)n * G + l;
  const uint4*    xp = (const uint4*)(x8 + (long)t * K) + 2 * l;      // 2 x 16 B per group
  // 1) issue every load for the wave: 5 x 16 B weights, 5 x 2 B scales, 10 x 16 B x8, 10 floats
  uint4 q[GPL], xa[GPL], xb[GPL]; uint16_t sc[GPL]; float sx[GPL], sxs[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    q[j]  = wp[LPR * j];  sc[j] = sp[LPR * j];                        // groups l + LPR*j
    xa[j] = xp[2 * LPR * j]; xb[j] = xp[2 * LPR * j + 1];
    sx[j] = xs[(long)t * G + l + LPR * j]; sxs[j] = xsum[(long)t * G + l + LPR * j];
  }
  // 2) int8 dot per group: 4 uint32 -> 8 x v_dot4_i32_i8 (llama.cpp vec_dot_q4_0_q8_1_impl)
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    const int* xe = (const int*)&xa[j];  const int* xo = (const int*)&xb[j];
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {                                  // 8 k per uint32
      sumi = __builtin_amdgcn_sdot4(qw[wd] & 0x0F0F0F0Fu, xe[wd], sumi, false);       // k even
      sumi = __builtin_amdgcn_sdot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xo[wd], sumi, false);// k odd
    }
    const float ws = __uint_as_float((uint32_t)sc[j] << 16);          // bf16 -> f32
    acc += ws * (sx[j] * (float)sumi - 8.f * sxs[j]);                 // zero point folded
  }
  // 3) reduce the LPR lanes of this DPP row (row_shr within 16 lanes; for LPR=64 add
  //    row_bcast:15 and row_bcast:31 as wvSplitK does), lane LPR-1 writes.
  acc += __builtin_amdgcn_mov_dpp(acc, 0x111, 0xf, 0xf, true);  // row_shr:1
  acc += __builtin_amdgcn_mov_dpp(acc, 0x112, 0xf, 0xf, true);  // row_shr:2
  acc += __builtin_amdgcn_mov_dpp(acc, 0x114, 0xf, 0xf, true);  // row_shr:4
  acc += __builtin_amdgcn_mov_dpp(acc, 0x118, 0xf, 0xf, true);  // row_shr:8
  if (l == LPR - 1) out[(long)p * N + n] = acc;                    // fp32 (or bf16 + epilogue)
}
```
Budgets (LPR=16, K=2560): VGPRs ~ q 20 + x 40 + sc/sx/sxs 15 + misc ~15 = ~90 (<= 128 ->
4 waves/SIMD possible). VALU per lane: 5 x (12 unpack + 8 dot4 + 4 scale) + 4 DPP + ~30
address = ~145 ops = ~580 cycles = 0.4 us. Bytes in flight per wave at issue: 5 KB weights +
10 KB (L2-hot) x. Waves: gate_up 10 pairs x 320 rows / 4 = **800 waves (200 WGs)**; the entire
4.1 MB is requested in the first ~1 us, so the kernel is HBM-bound: ~3.5 us at peak, expect
**5-6 us** with ramp/tail vs 24 us now. Dense qkv (4.6 MB, 3584 rows -> 896 waves): ~5.5 us
in one launch and no reduce (vs 8 + 4.8 profiled). o_proj K=1536 = 48 groups -> LPR=16,
3 groups/lane. Odd K (K/32 not divisible by LPR): let lanes `l < G % LPR` take one extra group
(predicated), or pick LPR=8.
M > 1 with the same expert (dense M<=16, or MoE tokens sharing an expert): loop `j` over tokens
inside the group loop reusing `q[]` (llama.cpp `ncols_dst`); VALU grows 8 dot4 per token per
group, still ~16x under the memory budget up to M=4; beyond that use D4.

Why not thread-per-column with all loads in flight (minimal change to today's kernel)?
It would fix the latency chain but keeps 64 lines per wave-load (TA-bound at ~16 B/clk/CU,
only 2.4x over the HBM share) and needs x in LDS or registers per lane (K=2560 -> 2.5 KB
per lane: impossible) — hence the K-slab. Keep it as the "cheap ablation" in the microbench.

### D2 — down-proj GEMV, K=160: thread-per-row, activation in SGPRs  [rank 1, pairs with D1]

K=160 = 5 groups = 80 B per row, N=2560 rows x 10 pairs. Lanes along K would waste 3/4 of a
wave; instead each lane owns one row (5 `uint4` issued together, 400 waves = whole 2.3 MB in
flight), and the activation is **wave-uniform** (one pair per wave): 160 int8 = 40 dwords
via `s_load_dwordx16` x2 + `x8`, plus 5+5 floats of `xs/xsum`, all in SGPRs. `v_dot4_i32_i8`
takes the SGPR directly — zero LDS, zero VGPRs for x, no `__syncthreads`.

```cpp
// wave = (pair p, 64 rows); lane = row n. x8 for pair p lives in SGPRs (compiler will use
// s_load for __attribute__((uniform)) loads from a wave-uniform pointer; verify in the ISA dump).
const uint4* wp = (const uint4*)(w2 + e*stride_we + (long)n*(160/8));   // 80 B per row
uint4 q[5]; uint16_t sc[5];
#pragma unroll for (int j=0;j<5;++j){ q[j]=wp[j]; sc[j]=sp[j]; }         // all in flight
const int* xu = (const int*)(x8 + (long)p*160);                           // uniform -> SGPRs
float acc=0.f;
#pragma unroll for (int j=0;j<5;++j){ int sumi=0;
  const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
  #pragma unroll for(int wd=0;wd<4;++wd){
    sumi=__builtin_amdgcn_sdot4(qw[wd]&0x0F0F0F0F, xu[8*j+wd],   sumi,false);
    sumi=__builtin_amdgcn_sdot4((qw[wd]>>4)&0x0F0F0F0F, xu[8*j+4+wd], sumi,false);}
  acc += bf2f(sc[j]) * (xs_u[j]*(float)sumi - 8.f*xsum_u[j]); }
out[(long)p*N + n] = acc;                                                 // fp32 [P, N]
```
Loads: lanes 80 B apart -> one `dwordx4` wave-load spans 5 KB / 80 lines; the five loads
together consume every byte (L1 16 KB, L2 otherwise). ~100 VALU/lane. Expected **3-4 us** vs
24 us. Output stays `[P, N]` fp32 for the existing `_moe_reduce_weighted_sum_kernel`
(top-k weighted sum), which the in-server test showed beats an in-kernel finalize; partials
shrink from `[SK, P, N]` to `[P, N]`. If N x P grows (M=8: 25,600 rows -> 400 waves, fine).

Alternative for the sum over 10 experts: `global_atomic_add_f32` (no-return, exists on gfx908)
into a zeroed fp32 `[M, K]` buffer — fire-and-forget, no serialization, but non-deterministic
summation order and still needs the bf16 cast kernel. Prefer the reduce kernel; test atomics
only if the reduce shows up in the profile.

### D3 — multi-GEMV descriptor launch  [rank 2, +1.5% c=1, easy once D1/D2 exist]

One kernel, one launch, a small descriptor table in constant/SGPR memory:
`struct Desc { const void *w,*s,*x8,*xs,*xsum; void *out; const int *row_tok,*row_exp;
int N,K,lpr,first_tile,n_tiles; }`. `blockIdx.x` -> descriptor by scanning `first_tile`
(<= 16 entries, uniform branch), then D1 or D2 body by `K` template dispatch inside the
kernel (`switch` on desc.K with a handful of instantiated bodies; wave-uniform).
Per layer today: MoE = 2 GEMV + 2 reduces, shared expert = 2 GEMV + 2 reduces (8 launches).
With D3: launch A = {10 routed gate_up pairs, shared gate_up} ; B = reduce/silu-mul/quant
(one Triton kernel over P+1 rows) ; C = {10 down pairs, shared down} ; D = weighted sum +
sigmoid(x.w_gate) x shared + add. 8 -> 4 launches = ~5 us/layer = **~0.25 ms/token**.
QSA layers: qkv(+gate) and the indexer projections share the normed input — same trick.
cudagraph: descriptor tables are per-layer static buffers (weights static, activation
buffers static under capture); expert ids come from `row_exp` device tensors as today.

### D4 — MFMA paths for 2 <= M <= 16  [rank 2 for batched decode; not a c=1 lever]

Bandwidth-neutral at M=1 (a 16x16 tile does 16x the MACs for the same bytes), but the only
way to keep the dense W4 GEMMs memory-bound once M >= 4: the Triton BLOCK_M=16 fp32 path
does 2 VALU ops per MAC -> qkv at M=16 = 147M MAC -> ~38k cycles/SIMD = **~26 us** (VALU-bound)
vs 5.5 us of bytes. Two shapes:

(a) `v_mfma_i32_4x4x4i8` with `cbsz:4 abid:0` (broadcast block 0's A to all 16 blocks —
verified to assemble): A = 4 tokens x 4 k of int8 activation held by lanes 0-3, B = lane
`4b+j` holds W[col 4b+j][k..k+3], D = lane (col) holds 4 accumulators = 4 tokens. Per group:
8 MFMA (8 cycles each), 12 unpack VALU, 4 cvt + 8 fma rescale with wave-uniform `xs[m][g]`,
`xsum[m][g]` in SGPRs. Thread-per-column semantics, so loads are strided like today —
acceptable only with all 5 loads in flight; best for MoE at M<=4 pairs sharing an expert
(rare at 512 experts) — low priority.

(b) `v_mfma_i32_16x16x16i8` for dense M<=16: A = x8 (16 tokens x 16 k), lane `l` holds
A[l%16][4*(l/16)..+3]; B = W (16 k x 16 cols), lane `l` holds B[4*(l/16)..+3][l%16]; D lane
`l` reg `r` = D[4*(l/16)+r][l%16]. **Constraint: one MFMA's K=16 must lie inside one GS32
group** (the int32 accumulator cannot be rescaled per column x group after mixing groups),
so each lane loads a **dword** (4 B: row j, group g, word kb) — 16 rows x 16 B contiguous
per wave-load (16 lines, 256 B). Keep 16-20 dwords in flight per lane. Per group: 2 MFMA
+ 6 unpack + 4 cvt + 8 fma (per-token `xs[m][g]`: 4 tokens per lane -> 4 floats from LDS
`ds_read_b128`, 16 x 80 x 4 B = 5 KB). Matrix time for qkv M=16: 147M / (128 x 480) = 2.4k
cycles = 1.6 us, hidden under 5.5 us of bytes. Also the natural kernel for the GPTQ4 27B
"verify" M=4-8 tiers.
(c) fp16 alternative `v_mfma_f32_16x16x16f16` with the 0x6400 dequant (W4A16, no activation
quant): 4 `v_and_or_b32` + 4 `v_pk_fma_f16` per uint32 to produce scaled fp16 weights, then
the group constraint disappears (scale applied on B). Risk: bf16 -> fp16 activation overflow
(Qwen3-Next hidden states carry large outliers); would need a per-token pre-scale. Use (b).
Weight layout for (b) needs no repack (dword loads from the stock row-major-K layout).
A Marlin-style pre-shuffle (lane order = MFMA operand order, 16 B per lane) would give
4x fewer load instructions but a second weight copy is impossible (22.5 GB/GPU of W4) and
the prefill Triton kernel needs the stock layout — so no.

### D5 — W4A16-exact fallback: packed-f16 dot with the sum trick  [rank 3, only if A8 fails the gate]

Same geometry as D1, activation kept as fp16 pairs pre-permuted to (x0,x2),(x1,x3),(x4,x6),
(x5,x7) per 8-block; per uint32: `v_and_or_b32` x4 with masks 0x000F000F / 0x0F000F00>>... and
0x64006400 (pairs (t,t+2) as exllama/AWQ do with one shift), then 4 x `v_dot2_f32_f16`
against the fp16 pairs — **no subtract**: the 1024 bias and the zero point 8 are folded with
`acc += ws * (dot - 1032 * xsum_g)` where `xsum_g = sum of the 32 fp16 x` (fp32, precomputed
like Q8_1's `s`). ~11 VALU per 8 MAC (vs 5 for D1, 48 today). fp16 range caveat applies to
x only (weights are exact integers in fp16); if activations exceed 65504 pre-scale per token
by a power of two. Bit-exact vs the current fp32 path up to summation order.

### What not to do (evidence in the campaign log)
- Fused "last block finalizes" split-K: +25% in the microbench, -4% in the server. D1/D2
  avoid split-K entirely; if a shape ever needs it, use no-return fp32 atomics + a separate
  parallel epilogue.
- Triton for the GEMV inner loop: nibble unpack codegen tops out ~80 GB/s on gfx908.
- Any second copy of the expert weights (no HBM headroom).

## 4. Expected speedups (estimates; validate per section 7)

| path | today | design | expected | per-token / per-step effect |
|---|---|---|---|---|
| MoE gate_up, M=1 (4.1 MB) | 24 us | D1 | 5-6 us | |
| MoE down, M=1 (2.3 MB) | 24 us | D2 | 3-4 us | |
| MoE per layer incl. 2 reduces | ~64 us | D1+D2 | ~14 us | **-2.3 ms/token, c=1 16.7 -> ~14.4 ms (-14%)**; -11% counting only the GEMVs |
| MoE per layer, M=4 (27.6 MB) | 157 us | D1+D2 | ~35 us | **-5.9 ms/step at c=4**; M=8: ~65 vs ~187 us |
| dense qkv M=1 (4.6 MB) | 8.0 + 4.8 (prof.) | D1, no split | ~5.5 | -0.1-0.2 ms/token over the 120 dense launches |
| dense qkv/o_proj M=16 | ~26 us (VALU) | D4(b) | ~6 us | c=16: ~-0.5 ms/step on the 12 QSA layers + shared experts |
| launches (MoE + shared expert) | 8/layer | D3 | 4/layer | -0.25 ms/token |

Basis: bytes / (0.7 x 1.2 TB/s) + 1.3 us launch + ~1 us tail, checked against the VALU budget
of 1.2 (all designs are >= 10x under it). The c=1 step is 16.7 ms with ~14 ms GPU.

## 5. Risks and accuracy notes

- **W4A8 block-32 activation error.** Symmetric absmax int8 over 32 values: max error
  0.5/127 = 0.4% of the block max; the W4-GS32 weight error is up to 1/15 = 3.3% of the group
  max (RMS ~8x larger). llama.cpp ships this for every quantized matmul on HIP/CUDA (Q8_1).
  QServe reports +<= 0.16 ppl for W4A8 with **per-token** int8 (far coarser than block-32) plus
  rotations; block-32 needs none of that. Expected PPL delta << 0.01. Gate anyway: PPL 3.14
  and GSM8K at production TP (kernel-parity rule). Compare logprob fingerprints to the W4A16
  path (noise floor 0.004-0.007 nats).
- **Scale precision**: keep `xs`, `xsum` in fp32 (ik_llama.cpp #196: fp16 Q8_1 scales fail on
  models whose activations exceed fp16 range). Weight scales stay bf16 as stored; `sumi` is
  exact int32 (max 32 x 15 x 127 = 61k per group).
- **Where activation quant runs**: fold into the producer (`_hc_combine_norm_kernel` for the
  MoE/shared/qkv input; `_moe_reduce_silu_mul_kernel` for `inter`; norms for the dense
  inputs) — one extra int8 output + two fp32 vectors, no extra launch. A standalone quant
  kernel costs 1.3 us x ~3 per layer = ~0.2 ms/token and would eat most of D3's gain.
- **Coalescing on the stock layout**: D1 is fully coalesced; D2/D4(a) rely on L1 line reuse
  across the 5 in-flight loads — measure `TCP` hit rate; if poor, D2 can switch to LPR=4
  with 20 B per lane (dword+dwordx4 pair) at the cost of a 4-lane DPP reduce.
- **DPP semantics**: `row_shr:n` shifts within 16-lane rows; out-of-row lanes read 0 only
  with bound_ctrl set (the builtin's last arg `true`, spelled `bound_ctrl:0` in asm — the
  same inversion wvSplitK lives with). Lane 15 of each row holds the row sum. For LPR=64 add
  `row_bcast:15` (row_mask 0xa) and `row_bcast:31` (row_mask 0xc) as in `skinny_gemms.cu:490-495`.
- **cudagraph capture**: expert ids can be -1/out of range at capture (dummy routing); keep the
  zero-fill branch from the current kernel. Descriptor tables must be pre-allocated per layer.
- **Compiler**: `__builtin_amdgcn_sdot4` is enabled by `-mcpu=gfx908` (dot1-insts); check the
  ISA dump (`--save-temps`) for `v_dot4_i32_i8`, `s_load_dwordx16` (D2), 5 back-to-back
  `global_load_dwordx4` before the first `s_waitcnt vmcnt`, and `v_mov_b32_dpp`. hipcc tends
  to hoist scale `bf16->f32` shifts fine; watch VGPR count (<= 128 for 4 waves/SIMD).
- **Occupancy vs. MLP**: the design deliberately uses many single-tile waves rather than
  persistent waves; if the dispatcher becomes the limit (>= 3000 WGs at M=8), switch to 2
  tiles per wave with the second tile's loads issued before the first tile's math.
- Register-pressure cliff: at LPR=16, K=2560, x8 occupies 40 VGPRs per lane. For K=3584-class
  dense inputs (not present here) drop to LPR=32 or stream x8 from LDS via `ds_read_b128`.

## 6. Sources

Local:
- `/home/tyler/vllm-gfx908/vllm/model_executor/layers/fused_moe/csrc/gfx908_w4gemv.hip` (current MoE GEMV);
  `.../fused_moe/gfx908_moe_hip.py` (`_CFG`, reduces); `.../mixed_precision/triton_w4a16.py`
  (`triton_w4a16_gemv_partial_kernel`, `_GFX908_GEMV_TABLE`); `.../layers/gfx908_shared_expert.py`.
- `/home/tyler/vllm-gfx908/csrc/rocm/skinny_gemms.cu:350-570` (`wvSplitK_hf_sml_`: LDS-staged A,
  `A_CHUNK=8`/`UNRL` loads, `DOT2C`, DPP `row_shr`/`row_bcast` reduce, persistent loop).
- `/home/tyler/vllm-gfx908/csrc/libtorch_stable/quantization/gptq/{qdq_4.cuh,qdq_util.cuh,q_gemm.cu}`
  (exllamav2 dequant + the gfx908 `BLOCK_KN_SIZE 256` / `v_dot2c_f32_f16` / prefetch tuning notes).
- `/home/tyler/vllm-gfx908/csrc/libtorch_stable/moe/{moe_wna16.cu,moe_wna16_utils.h}` (vLLM PR #13321 kernel).
- `/home/tyler/vllm-gfx908/csrc/libtorch_stable/quantization/awq/dequantize.cuh`.
- `/home/tyler/llama.cpp/llama.cpp` @ 319146247 (2026-03-01): `ggml/src/ggml-cuda/common.cuh`
  (`ggml_cuda_dp4a`), `mmvq.cu` (`mul_mat_vec_q`, `calc_nwarps` GCN table), `vecdotq.cuh`
  (`vec_dot_q4_0_q8_1_impl`), `quantize.cu` (`quantize_q8_1`), `ggml/src/ggml-common.h` (`block_q8_1`).
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/qwen38_flash_next_gfx908.md` (finalize regression,
  Triton GEMV ceiling, 57/91/157 us MoE pipeline numbers at M=1/2/4, wvSplitK 65-98% of HBM).
- `/opt/rocm-7.2.0/lib/llvm/bin/llvm-mc -mcpu=gfx908` (instruction inventory in 1.3).

Public:
- llama.cpp: https://github.com/ggml-org/llama.cpp — `ggml/src/ggml-cuda/{common.cuh,mmvq.cu,vecdotq.cuh,quantize.cu}`;
  MI100/MI210/MI50 Q4_0 numbers: https://github.com/ggml-org/llama.cpp/discussions/15021
- gfx906 fork with DPP reductions / half-wave MoE dispatch: https://github.com/iacopPBK/llama.cpp-gfx906
- exllamav2 `exllamav2/exllamav2_ext/cuda/quant/qdq_4.cuh`: https://github.com/turboderp-org/exllamav2
- vLLM `csrc/moe/moe_wna16.cu` (PR https://github.com/vllm-project/vllm/pull/13321), `csrc/quantization/gptq/q_gemm.cu`
- AWQ gemv/dequant: https://github.com/mit-han-lab/llm-awq (`awq/kernels/csrc/gemv/gemv_cuda.cu`, `dequantize.cuh`)
- QServe W4A8KV4 accuracy: https://arxiv.org/abs/2405.04532 ; Q8_1 fp16-scale caveat:
  https://github.com/ikawrakow/ik_llama.cpp/issues/196
- MFMA operand layouts / `cbsz`/`abid`/`blgp`: https://github.com/ROCm/amd_matrix_instruction_calculator ,
  https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html ; ISA:
  AMD Instinct MI100 CDNA1 ISA reference (amd.com), https://llvm.org/docs/AMDGPUUsage.html

## 7. Microbenchmark plan (one idle GPU in the `aiter-mi100` / vllm image, graph-timed)

Harness: reuse the graph-timed pattern of the campaign's `bench_hip.py --moe-only` /
`mb_w4a16_gemv.py` (hipEvent around a captured graph of 50 iterations, report us and GB/s =
weight bytes / us; never quote torch-profiler numbers, they inflate by ~2.4 us). Build the
new kernels with `torch.utils.cpp_extension.load(..., extra_cuda_cflags=['-O3','--offload-arch=gfx908','--save-temps'])`
exactly like `gfx908_moe_hip.py::_ext`, so the same file drops into the server later.

Shapes (per TP4 rank, real): MoE gate_up E-slice N=320,K=2560 P in {10,20,40,80,160};
MoE down N=2560,K=160 same P; shared gate_up 2560->320, shared down 160->2560 (M=1..16);
qkv 2560->3584, o_proj 1536->2560 (M in {1,2,4,8,16}). Random expert ids in [0,512), plus a
capture-style run with ids = -1.

Steps:
1. Reference + correctness: fp32 dequant GEMV in torch; report max|err| for (a) current
   kernel, (b) D5 (must match (a) to fp32-summation-order noise), (c) D1/D2 W4A8 (report the
   error ratio vs. the W4 rounding error itself; expect <= 1/8).
2. Ablation ladder on gate_up P=10 (each step is one variable):
   current -> current + all 10 loads issued up front -> + scales prefetched -> K-slab LPR=16
   fp32 FMA -> + `v_dot2_f32_f16` (D5) -> + int8 `v_dot4` (D1) -> LPR=64 variant ->
   waves/SIMD sweep via `__launch_bounds__(256, {1,2,4})`.
   Record us, GB/s, VGPRs (from `--save-temps` .s), and rocprof `FETCH_SIZE`, `MemUnitBusy`,
   `VALUBusy`, `TCP_TOTAL_CACHE_ACCESSES`/hit, `Wavefronts`.
3. Down: D2 with x in SGPRs vs x in VGPRs (broadcast via `ds_read_b128`) vs current.
4. D3: {10 pairs + shared} in one launch vs two, both GEMVs; count launches in the graph.
5. D4(b) at M=4/8/16 on qkv/o_proj vs the Triton BLOCK_M=16 path; check MFMA issue with
   `SQ_INSTS_MFMA` (or equivalent) and that the kernel is still memory-bound.
6. Sanity: total bytes moved per layer x 48 vs. the measured per-token delta.
7. In-server (rule from the log: >= 3 c=1 probes per arm before believing a fusion): c=1 x3,
   then the full BenchAndReport 12-tier, then PPL (3.136 reference) and GSM8K at TP4; print
   the launches/ms-per-token profile table after every run. Only then extend
   `MOE_HIP_MAX_TOKENS` (8 -> 16/32) if the M=8/16 tiers confirm the bandwidth scaling.

Decision points: if step 2 shows the K-slab fp32-FMA variant already at >= 60% of HBM, D5
(exact W4A16) may be enough and the A8 accuracy gate can be skipped for the dense path; keep
D1's int8 path for the MoE where VALU per byte matters most at M >= 4.
