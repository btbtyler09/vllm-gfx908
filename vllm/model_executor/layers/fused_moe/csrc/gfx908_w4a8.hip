// gfx908 W4A8 GEMV kernels for small M (MoE experts, shared expert, dense projections).
// Microbench + design: docs/mi100_decode_opt/research/w4_gemv_kernels.md (D1, D2),
// agents/w4a8/REPORT.md (2026-09-04).
//
// Weights: [E, N, K/8] uint32, 8 nibbles along K (nibble t of word wd = k 8*wd+t),
// symmetric zero point 8; scales [E, N, K/32] bf16 (group size 32; one uint4 = one group).
//
// Activations (Q8_1 style, block 32, produced by quant_q8 or the Triton silu*mul+quant kernel):
//   x8   int8 [R, K]    per 8-element block DE-INTERLEAVED: bytes [k0 k2 k4 k6 | k1 k3 k5 k7], so the
//                       dword 2*wd holds the even k of word wd (pairs with q & 0x0F0F0F0F) and dword
//                       2*wd+1 the odd k (pairs with (q >> 4) & 0x0F0F0F0F). No weight shuffle needed.
//   xs   fp32 [R, K/32] block scale d = absmax/127
//   xsum fp32 [R, K/32] d * sum(q8)   -> sum(w x) = ws * (d * sumi - 8 * xsum)  (zero point folded)
//
// Output rows are pairs: out[p, :] = W[row_exp[p]] . x[row_tok[p]] as fp32 [P, N].
// row_exp[p] outside [0, E) -> zero row (cudagraph capture routing).
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

__device__ __forceinline__ float bf2f(uint32_t v) { return __uint_as_float(v << 16); }

__device__ __forceinline__ int dot4(uint32_t a, int b, int c) {
#ifdef W4A8_ASM_DOT4
  int r;
  asm("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
#else
  return __builtin_amdgcn_sdot4((int)a, b, c, false);
#endif
}

template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
  return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true));
}

// Sum over LPR consecutive lanes; result in lane LPR-1 of each group (row_shr prefix sums).
template <int LPR>
__device__ __forceinline__ float reduce_lpr(float v) {
  if constexpr (LPR >= 2) v = dpp_add<0x111>(v);   // row_shr:1
  if constexpr (LPR >= 4) v = dpp_add<0x112>(v);   // row_shr:2
  if constexpr (LPR >= 8) v = dpp_add<0x114>(v);   // row_shr:4
  if constexpr (LPR >= 16) v = dpp_add<0x118>(v);  // row_shr:8
  if constexpr (LPR >= 32) v += __shfl_xor(v, 16, 64);
  if constexpr (LPR >= 64) v += __shfl_xor(v, 32, 64);
  return v;
}

// ---------------------------------------------------------------------------------------------
// "shared expert as expert #E" row preparation (VLLM_GFX908_SHARED_AS_EXPERT=1).  One block of
// 256 threads per token m, launched as EXTRA BLOCKS of the activation prep kernel so it costs no
// extra launch and needs no sync (the consumers are later kernels).  For token m it writes
//   row_exp[m, 0:topk] = topk_ids[m, :]   row_exp[m, topk] = E   (the shared-expert row)
//   wcomb [m, 0:topk] = topk_w  [m, :]    wcomb [m, topk] = sigmoid(x[m] . w_gate)
// so the existing weighted-sum reduce produces routed + shared in one pass.
struct GateArgs {
  const uint32_t* x2;    // bf16 [M, K] viewed as uint32 [M, K/2]
  const uint32_t* wg2;   // bf16 [K]    viewed as uint32 [K/2]
  const int* topk_ids;   // int32 [M, topk]
  const float* topk_w;   // fp32  [M, topk]
  int* row_exp;          // int32 [M, topk+1]  (written)
  float* wcomb;          // fp32  [M, topk+1]  (written)
  int khalf, topk, E;
};

__device__ __forceinline__ void gate_row(const GateArgs& g, int m) {
  __shared__ float red[256];
  const int tid = threadIdx.x, stride = g.topk + 1;
  for (int i = tid; i < g.topk; i += 256) {
    g.row_exp[m * stride + i] = g.topk_ids[m * g.topk + i];
    g.wcomb[m * stride + i] = g.topk_w[m * g.topk + i];
  }
  const uint32_t* xr = g.x2 + (long)m * g.khalf;
  float acc = 0.f;
  // x4 unrolled: four (x, wg) pairs in flight instead of one.  Identical k sequence and identical
  // accumulation order into `acc`, so the value is unchanged.
  int k = tid;
  for (; k + 3 * 256 < g.khalf; k += 4 * 256) {
    uint32_t a[4], b[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) { a[u] = xr[k + u * 256]; b[u] = g.wg2[k + u * 256]; }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      acc = fmaf(bf2f(a[u] & 0xFFFFu), bf2f(b[u] & 0xFFFFu), acc);
      acc = fmaf(__uint_as_float(a[u] & 0xFFFF0000u), __uint_as_float(b[u] & 0xFFFF0000u), acc);
    }
  }
  for (; k < g.khalf; k += 256) {
    const uint32_t a = xr[k], b = g.wg2[k];
    acc = fmaf(bf2f(a & 0xFFFFu), bf2f(b & 0xFFFFu), acc);
    acc = fmaf(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), acc);
  }
  // gemv_flight (2026-09-04): the o = 128..1 LDS tree cost 8 __syncthreads and 8 LDS round trips on
  // a block that is pure latency (quant_q8_gate is a 2-workgroup launch, and under MOE_PREP_FOLD
  // this is the tail of the gate blocks).  Only o = 128 and o = 64 cross a wave boundary; the rest
  // fit inside wave 0.  The pairing order red[t] += red[t + o] is preserved exactly, so the sigmoid
  // input -- and the shared-expert weight -- are bit-identical (verified torch.equal, bench.log s5).
  // 4.08/4.09 -> 3.30/3.32 us at M=1, 4.11/4.12 -> 3.28/3.32 at M=4, 4.07/4.03 -> 3.20/3.19 at M=8.
  red[tid] = acc;
  __syncthreads();
  if (tid < 128) red[tid] += red[tid + 128];
  __syncthreads();
  if (tid < 64) {
    float v = red[tid] + red[tid + 64];
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
    if (tid == 0) {
      g.row_exp[m * stride + g.topk] = g.E;
      g.wcomb[m * stride + g.topk] = 1.f / (1.f + expf(-v));
    }
  }
}

// Pick the weight/scale base row for pair p: the "extra expert" slab when row_exp[p] == E and an
// extra pointer was given, the routed expert e when 0 <= e < E, otherwise "zero this row".
__device__ __forceinline__ bool pick_expert(
    int e, int E, const uint32_t* w, const uint16_t* s, long stride_we, long stride_se,
    const uint32_t* w_extra, const uint16_t* s_extra, const uint32_t*& wb, const uint16_t*& sb) {
  if (w_extra != nullptr && e == E) { wb = w_extra; sb = s_extra; return true; }
  if (e < 0 || e >= E) return false;
  wb = w + (long)e * stride_we;
  sb = s + (long)e * stride_se;
  return true;
}

// ---------------------------------------------------------------------------------------------
// Activation quant: one thread per (row, group of 32). x bf16 [R, K] -> x8 / xs / xsum.
// GATE=1 appends the M row-prep blocks described above.
template <int GATE>
__global__ void __launch_bounds__(256) quant_q8_kernel(
    const uint32_t* __restrict__ x, uint4* __restrict__ x8, float* __restrict__ xs,
    float* __restrict__ xsum, int total, int nbase, GateArgs ga) {
  if constexpr (GATE) {
    if ((int)blockIdx.x >= nbase) { gate_row(ga, (int)blockIdx.x - nbase); return; }
  }
  const int gid = blockIdx.x * 256 + threadIdx.x;
  if (gid >= total) return;
  const uint4* xp = reinterpret_cast<const uint4*>(x) + (long)gid * 4;  // 64 B = 32 bf16
  uint4 v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = xp[i];
  float f[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t d[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      f[i * 8 + j * 2] = __uint_as_float(d[j] << 16);
      f[i * 8 + j * 2 + 1] = __uint_as_float(d[j] & 0xFFFF0000u);
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
  const float d = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int q[32], qsum = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) { q[i] = __float2int_rn(f[i] * inv); qsum += q[i]; }
  uint32_t o[8];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int* qb = q + b * 8;
    o[2 * b] = (qb[0] & 0xFF) | ((qb[2] & 0xFF) << 8) | ((qb[4] & 0xFF) << 16) | ((qb[6] & 0xFF) << 24);
    o[2 * b + 1] = (qb[1] & 0xFF) | ((qb[3] & 0xFF) << 8) | ((qb[5] & 0xFF) << 16) | ((qb[7] & 0xFF) << 24);
  }
  x8[(long)gid * 2] = make_uint4(o[0], o[1], o[2], o[3]);
  x8[(long)gid * 2 + 1] = make_uint4(o[4], o[5], o[6], o[7]);
  xs[gid] = d;
  xsum[gid] = d * (float)qsum;
}

// ---------------------------------------------------------------------------------------------
// D1 "K-slab": LPR lanes per output row, 64/LPR rows per wave, lane l owns groups l + LPR*j
// (one uint4 = one GS32 group = one scale). Every load of the wave is issued before the first
// use (sched_barrier); int8 dot4 per group; DPP row reduction; direct fp32 store, no split-K.
template <int K, int LPR, int WPB>
__global__ void __launch_bounds__(64 * WPB) w4a8_slab_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ row_exp, float* __restrict__ out, int N, int P, int E, long stride_we,
    long stride_se, const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR;
  const int tiles_per_p = N / ROWS;
  // readfirstlane: the tile (and so p, e, t) is wave-uniform -> s_load for the ids.
  const int tile = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  if (tile >= P * tiles_per_p) return;
  const int p = tile / tiles_per_p, n = (tile % tiles_per_p) * ROWS + r;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wbase;
  const uint16_t* sbase;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8)) + l;
  const uint16_t* sp = sbase + (long)n * G + l;
  const uint4* xp = reinterpret_cast<const uint4*>(x8 + (long)t * K) + 2 * l;
  const float* xsp = xs + (long)t * G + l;
  const float* xsump = xsum + (long)t * G + l;
  uint4 q[GPL], xa[GPL], xb[GPL];
  uint32_t sc[GPL];
  float sx[GPL], sxs[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) { q[j] = wp[LPR * j]; sc[j] = sp[LPR * j]; }
    else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) {
      xa[j] = xp[2 * LPR * j]; xb[j] = xp[2 * LPR * j + 1];
      sx[j] = xsp[LPR * j]; sxs[j] = xsump[LPR * j];
    } else { xa[j] = make_uint4(0, 0, 0, 0); xb[j] = xa[j]; sx[j] = 0.f; sxs[j] = 0.f; }
  }
  __builtin_amdgcn_sched_barrier(0);   // all loads above the first use
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    const int xd[8] = {(int)xa[j].x, (int)xa[j].y, (int)xa[j].z, (int)xa[j].w,
                       (int)xb[j].x, (int)xb[j].y, (int)xb[j].z, (int)xb[j].w};
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
      sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
    }
    acc += bf2f(sc[j]) * (sx[j] * (float)sumi - 8.f * sxs[j]);
  }
  acc = reduce_lpr<LPR>(acc);
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// Q8_1 quantization of ONE group of 32 bf16 (64 B) -- the exact arithmetic of quant_q8_kernel,
// factored out so the fused-prep slab kernel below is bit-identical to the two-launch flow.
//   src : the group's 4 uint4 (32 bf16)
//   o[8]: the de-interleaved int8 payload (bytes [k0 k2 k4 k6 | k1 k3 k5 k7] per 8)
//   d   : block scale, dsum: d * sum(q)
__device__ __forceinline__ void q8_group(const uint4* __restrict__ src, uint32_t o[8], float& d,
                                         float& dsum) {
  uint4 v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = src[i];
  // two passes over the 16 packed dwords instead of one pass over 32 unpacked floats: the shifts
  // are free next to the 25 VGPRs of weight already live here, and 32 fp32 in flight cost a wave
  // of occupancy.  The value sequence (and so the result) is exactly quant_q8_kernel's.
  float amax = 0.f;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t dw[4] = {v[b].x, v[b].y, v[b].z, v[b].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      amax = fmaxf(amax, fabsf(__uint_as_float(dw[j] << 16)));
      amax = fmaxf(amax, fabsf(__uint_as_float(dw[j] & 0xFFFF0000u)));
    }
  }
  d = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int qsum = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t dw[4] = {v[b].x, v[b].y, v[b].z, v[b].w};
    int q[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[2 * j] = __float2int_rn(__uint_as_float(dw[j] << 16) * inv);
      q[2 * j + 1] = __float2int_rn(__uint_as_float(dw[j] & 0xFFFF0000u) * inv);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) qsum += q[j];
    o[2 * b] = (q[0] & 0xFF) | ((q[2] & 0xFF) << 8) | ((q[4] & 0xFF) << 16) | ((q[6] & 0xFF) << 24);
    o[2 * b + 1] = (q[1] & 0xFF) | ((q[3] & 0xFF) << 8) | ((q[5] & 0xFF) << 16) | ((q[7] & 0xFF) << 24);
  }
  dsum = d * (float)qsum;
}

// ---------------------------------------------------------------------------------------------
// isa_research (2026-09-05): the gate-fold slab's critical path in front of its staging barrier is
// three dependent scalar round trips + one 32-element Q8_1 chain on 80 of 256 lanes, and only then
// do the weight loads issue.  Three bit-exact changes take the M=1 kernel from 9.35 to 8.88 us cold
// (-5%, agents/isa_research/bench_variants2.log; M=4 25.1 -> 22.7):
//   * SPLITQ: two lanes per group.  max and the int sum are order-free, so exchanging them with one
//     DPP quad_perm swap each gives the same d / inv / q / dsum as q8_group (151 VALU on the critical
//     path instead of 249, 35 VGPR instead of 47);
//   * HOIST: with the staging that cheap, the weight/scale loads are issued BEFORE the staging and
//     stay in flight across the barrier (59 VGPR, 4 waves/SIMD -- measured better than 5 waves without);
//   * the pair's token is p / (topk+1) (the layout invariant the kernel already relies on for `slot`),
//     the expert id is one s_load_dword, and all scalar kernargs are fetched in one batch at entry.
__device__ __forceinline__ float w4a8_swap1_f(float v) { return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xf, 0xf, true)); }
__device__ __forceinline__ int w4a8_swap1_i(int v) { return __builtin_amdgcn_mov_dpp(v, 0xB1, 0xf, 0xf, true); }

// Half of one Q8_1 group (16 bf16) on one lane; the partner lane (xor 1) holds the other half.
__device__ __forceinline__ void q8_group_half(const uint4* __restrict__ src2, uint32_t o[4], float& d,
                                              float& dsum) {
  uint4 v[2];
  v[0] = src2[0]; v[1] = src2[1];
  float amax = 0.f;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const uint32_t dw[4] = {v[b].x, v[b].y, v[b].z, v[b].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      amax = fmaxf(amax, fabsf(__uint_as_float(dw[j] << 16)));
      amax = fmaxf(amax, fabsf(__uint_as_float(dw[j] & 0xFFFF0000u)));
    }
  }
  amax = fmaxf(amax, w4a8_swap1_f(amax));
  d = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int qsum = 0;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const uint32_t dw[4] = {v[b].x, v[b].y, v[b].z, v[b].w};
    int q[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[2 * j] = __float2int_rn(__uint_as_float(dw[j] << 16) * inv);
      q[2 * j + 1] = __float2int_rn(__uint_as_float(dw[j] & 0xFFFF0000u) * inv);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) qsum += q[j];
    o[2 * b] = (q[0] & 0xFF) | ((q[2] & 0xFF) << 8) | ((q[4] & 0xFF) << 16) | ((q[6] & 0xFF) << 24);
    o[2 * b + 1] = (q[1] & 0xFF) | ((q[3] & 0xFF) << 8) | ((q[5] & 0xFF) << 16) | ((q[7] & 0xFF) << 24);
  }
  qsum += w4a8_swap1_i(qsum);
  dsum = d * (float)qsum;
}

// The gate-fold slab body: weight loads hoisted, two-lane staging.  Same value as w4a8_slab_prep_body.
template <int K, int LPR>
__device__ __forceinline__ float w4a8_slab_prep_body_hoist(
    const uint32_t* __restrict__ wbase, const uint16_t* __restrict__ sbase,
    const uint32_t* __restrict__ x, int tok, int n, int l,
    uint32_t* __restrict__ xl, float* __restrict__ xsl, float* __restrict__ xsuml) {
  constexpr int G = K / 32, GPL = (G + LPR - 1) / LPR;
  static_assert(G % LPR == 0 && 2 * G <= 256, "hoisted gate-fold body: K/32 % LPR == 0, two lanes per group");
  const uint4* xg = reinterpret_cast<const uint4*>(x) + (long)tok * (K / 8);
  const uint16_t* sp = sbase + (long)n * G + l;
  const uint4* wp4 = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8)) + l;
  uint4 q[GPL];
  uint32_t sc[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) { q[j] = wp4[LPR * j]; sc[j] = sp[LPR * j]; }
  __builtin_amdgcn_sched_barrier(0);   // weight loads in flight across the staging + barrier
  {
    const int g = threadIdx.x >> 1, h = threadIdx.x & 1;
    if (g < G) {
      uint32_t o[4];
      float d, dsum;
      q8_group_half(xg + (long)g * 4 + 2 * h, o, d, dsum);
      reinterpret_cast<uint4*>(xl)[2 * g + h] = make_uint4(o[0], o[1], o[2], o[3]);
      if (h == 0) { xsl[g] = d; xsuml[g] = dsum; }
    }
  }
  __syncthreads();
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const int g = l + LPR * j;
    const uint4 xa = reinterpret_cast<const uint4*>(xl)[2 * g];
    const uint4 xb = reinterpret_cast<const uint4*>(xl)[2 * g + 1];
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    const int xd[8] = {(int)xa.x, (int)xa.y, (int)xa.z, (int)xa.w,
                       (int)xb.x, (int)xb.y, (int)xb.z, (int)xb.w};
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
      sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
    }
    acc += bf2f(sc[j]) * (xsl[g] * (float)sumi - 8.f * xsuml[g]);
  }
  return reduce_lpr<LPR>(acc);
}

// ---------------------------------------------------------------------------------------------
// D1 with the ACTIVATION PREP FOLDED IN (VLLM_GFX908_MOE_PREP_FOLD=1).
//
// Same geometry and same math as w4a8_slab_kernel, but the block reads the pair's activation as
// bf16 and does the Q8_1 quantization while staging it in LDS, so the separate quant_q8 launch
// disappears.  One block = 4 waves = 4 consecutive tiles of the SAME pair (needs (N/ROWS) % 4 == 0,
// exactly like the fp16 LDS kernel), so `t` is block-uniform and the zero-row branch is
// block-uniform too (no __syncthreads hazard).
//
// The staged bytes / scales are bit-identical to what quant_q8 writes (q8_group above is the same
// code), and the dot product then reads them in the same order, so the result is bit-identical to
// quant_q8 + gemv_slab.
// The folded slab's per-block work, shared by the plain and the shared-as-expert kernels so the
// two can never drift: stage+quantize the pair's token into LDS, then the usual dot product.
template <int K, int LPR>
__device__ __forceinline__ float w4a8_slab_prep_body(
    const uint32_t* __restrict__ wbase, const uint16_t* __restrict__ sbase,
    const uint32_t* __restrict__ x, int tok, int n, int l,
    uint32_t* __restrict__ xl, float* __restrict__ xsl, float* __restrict__ xsuml) {
  constexpr int G = K / 32, GPL = (G + LPR - 1) / LPR;
  // The weight loads are issued AFTER the barrier on purpose: keeping q[GPL] + sc[GPL] (25 VGPRs
  // at K = 2560) live across the staging code costs a whole wave of occupancy, and the staging is
  // LDS-write bound, not latency bound.
  const uint4* xg = reinterpret_cast<const uint4*>(x) + (long)tok * (K / 8);   // K bf16 = K/8 uint4
  for (int g = threadIdx.x; g < G; g += 256) {
    uint32_t o[8];
    float d, dsum;
    q8_group(xg + (long)g * 4, o, d, dsum);
    reinterpret_cast<uint4*>(xl)[2 * g] = make_uint4(o[0], o[1], o[2], o[3]);
    reinterpret_cast<uint4*>(xl)[2 * g + 1] = make_uint4(o[4], o[5], o[6], o[7]);
    xsl[g] = d;
    xsuml[g] = dsum;
  }
  __syncthreads();
  const uint32_t* wp = wbase + (long)n * (K / 8);
  const uint16_t* sp = sbase + (long)n * G + l;
  const uint4* wp4 = reinterpret_cast<const uint4*>(wp) + l;
  uint4 q[GPL];
  uint32_t sc[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) { q[j] = wp4[LPR * j]; sc[j] = sp[LPR * j]; }
    else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }
  __builtin_amdgcn_sched_barrier(0);   // all weight loads above the first use
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const int g = l + LPR * j;
    if (G % LPR == 0 || g < G) {
      const uint4 xa = reinterpret_cast<const uint4*>(xl)[2 * g];
      const uint4 xb = reinterpret_cast<const uint4*>(xl)[2 * g + 1];
      const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
      const int xd[8] = {(int)xa.x, (int)xa.y, (int)xa.z, (int)xa.w,
                         (int)xb.x, (int)xb.y, (int)xb.z, (int)xb.w};
      int sumi = 0;
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
        sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
      }
      acc += bf2f(sc[j]) * (xsl[g] * (float)sumi - 8.f * xsuml[g]);
    }
  }
  return reduce_lpr<LPR>(acc);
}

// ---------------------------------------------------------------------------------------------
// D1 with the ACTIVATION PREP FOLDED IN (VLLM_GFX908_MOE_PREP_FOLD=1).
//
// Same geometry and same math as w4a8_slab_kernel, but the block reads the pair's activation as
// bf16 and does the Q8_1 quantization while staging it in LDS, so the separate quant_q8 launch
// disappears.  One block = 4 waves = 4 consecutive tiles of the SAME pair (needs (N/ROWS) % 4 == 0,
// exactly like the fp16 LDS kernel), so `t` is block-uniform and the zero-row branch is
// block-uniform too (no __syncthreads hazard).
//
// The staged bytes / scales are bit-identical to what quant_q8 writes (q8_group above is the same
// code), and the dot product then reads them in the same order, so the result is bit-identical to
// quant_q8 + gemv_slab.
template <int K, int LPR>
__global__ void __launch_bounds__(256) w4a8_slab_prep_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const uint32_t* __restrict__ x,
    const int* __restrict__ row_tok, const int* __restrict__ row_exp, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32;
  __shared__ __align__(16) uint32_t xl[K / 4];    // K int8, de-interleaved per 8
  __shared__ float xsl[G], xsuml[G];
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles_per_p = N / ROWS;
  const int tile0 = __builtin_amdgcn_readfirstlane(blockIdx.x * 4);
  const int p = tile0 / tiles_per_p;
  const int n = ((tile0 + wv) % tiles_per_p) * ROWS + r;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wbase;
  const uint16_t* sbase;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;   // block-uniform: no barrier below is reached
    return;
  }
  const float acc = w4a8_slab_prep_body<K, LPR>(wbase, sbase, x, t, n, l, xl, xsl, xsuml);
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// D1 prep fold for the "shared expert as expert #E" flow (VLLM_GFX908_SHARED_AS_EXPERT=1): the
// activation quant AND the row-list / gate preparation both ride along with the gate_up slab, so
// quant_q8_gate disappears as well and the layer keeps its four launches.
//
// The slab blocks must NOT read `row_exp`: the gate blocks of this same launch write it.  They do
// not need to -- the expert of a pair is a pure function of topk_ids given the interleaved row
// layout (row p = m*(topk+1) + slot, m = row_tok[p]):
//     slot == topk  ->  E (the shared expert)      slot < topk  ->  topk_ids[m*topk + slot]
// which is exactly the value gate_row writes.  `row_exp` / `wcomb` are still produced, by the M
// extra blocks at the end of the grid, for the later rowlane GEMV and the weighted-sum reduce --
// both separate launches, so there is no intra-launch dependency.
template <int K, int LPR>
__global__ void __launch_bounds__(256) w4a8_slab_prep_gate_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const uint32_t* __restrict__ x,
    const int* __restrict__ row_tok, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra,
    int nbase, GateArgs ga) {
  // kernarg prefetch: every scalar argument the slab path touches is forced into SGPRs here, so the
  // compiler emits the kernarg s_loads in one batch at entry instead of lazily across the branches.
  asm volatile("" :: "s"(w), "s"(s), "s"(x), "s"(out), "s"(N), "s"(P), "s"(E), "s"(stride_we), "s"(stride_se),
               "s"(w_extra), "s"(s_extra), "s"(nbase), "s"(ga.topk_ids), "s"(ga.topk), "s"(ga.x2), "s"(ga.wg2),
               "s"(ga.row_exp), "s"(ga.wcomb), "s"(ga.khalf), "s"(ga.topk_w));
  if ((int)blockIdx.x >= nbase) { gate_row(ga, (int)blockIdx.x - nbase); return; }
  constexpr int ROWS = 64 / LPR, G = K / 32;
  __shared__ __align__(16) uint32_t xl[K / 4];
  __shared__ float xsl[G], xsuml[G];
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles_per_p = N / ROWS;
  const int tile0 = __builtin_amdgcn_readfirstlane(blockIdx.x * 4);
  const int p = tile0 / tiles_per_p;
  const int n = ((tile0 + wv) % tiles_per_p) * ROWS + r;
  // row p = m*(topk+1) + slot (the layout the shared-as-expert flow builds and this kernel already
  // relies on for `slot`), so the token needs no row_tok load, and the expert id is one scalar load.
  const int stride = ga.topk + 1;
  const int t = p / stride;
  const int slot = p - t * stride;
  const int* tp = ga.topk_ids + __builtin_amdgcn_readfirstlane(t * ga.topk + slot);
  int ev;
  asm volatile("s_load_dword %0, %1, 0x0\n\ts_waitcnt lgkmcnt(0)" : "=s"(ev) : "s"(tp) : "memory");
  const int e = (slot == ga.topk) ? E : ev;
  (void)row_tok;
  const uint32_t* wbase;
  const uint16_t* sbase;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;
    return;
  }
  // batch_research 2026-09-05: the hoisted body needs K/32 % LPR == 0 (two lanes per group); the K = 160
  // instantiation (G = 5) kept in SLABPG_CASE tripped its static_assert and broke the whole extension build,
  // so fall back to the plain body where the hoisted one is not applicable (same value either way).
  float acc;
  if constexpr (G % LPR == 0 && 2 * G <= 256) {
    acc = w4a8_slab_prep_body_hoist<K, LPR>(wbase, sbase, x, t, n, l, xl, xsl, xsuml);
  } else {
    acc = w4a8_slab_prep_body<K, LPR>(wbase, sbase, x, t, n, l, xl, xsl, xsuml);
  }
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// D2: thread-per-row for K=160 (5 groups = 80 B per row). One pair per wave; the pair's
// activation (40 dwords + 10 floats) is wave-uniform and lives in SGPRs.
template <int K, int WPB>
__global__ void __launch_bounds__(64 * WPB) w4a8_rowlane_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ row_exp, float* __restrict__ out, int N, int P, int E, long stride_we,
    long stride_se, const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int G = K / 32;
  const int lane = threadIdx.x & 63;
  const int waves_per_p = N / 64;
  const int wave = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  if (wave >= P * waves_per_p) return;
  const int p = wave / waves_per_p, n = (wave % waves_per_p) * 64 + lane;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wbase;
  const uint16_t* sbase;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
    out[(long)p * N + n] = 0.f;
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8));
  const uint16_t* sp = sbase + (long)n * G;
  uint4 q[G];
  uint32_t sc[G];
#pragma unroll
  for (int j = 0; j < G; ++j) { q[j] = wp[j]; sc[j] = sp[j]; }
  const int* xu = reinterpret_cast<const int*>(x8 + (long)t * K);  // wave-uniform -> SGPRs
  const float* xsu = xs + (long)t * G;
  const float* xsumu = xsum + (long)t * G;
  int xv[8 * G]; float xsv[G], xsumv[G];
#pragma unroll
  for (int i = 0; i < 8 * G; ++i) xv[i] = xu[i];
#pragma unroll
  for (int j = 0; j < G; ++j) { xsv[j] = xsu[j]; xsumv[j] = xsumu[j]; }
  __builtin_amdgcn_sched_barrier(0);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < G; ++j) {
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xv[8 * j + 2 * wd], sumi);
      sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xv[8 * j + 2 * wd + 1], sumi);
    }
    acc += bf2f(sc[j]) * (xsv[j] * (float)sumi - 8.f * xsumv[j]);
  }
  out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// D2 with silu*mul + Q8_1 FOLDED IN (VLLM_GFX908_MOE_SILU_FOLD=1).
//
// Same idea as MOE_PREP_FOLD one stage later: the down GEMV recomputes its own pair's
// silu(gate)*up + Q8_1 into LDS, so the separate silu_mul_quant launch disappears.  One block =
// WPB waves of the SAME pair (needs (N/64) % WPB == 0), so the staging is block-uniform and there
// is no cross-workgroup dependency, no counter and no atomic.
//
// Bit-exactness: this reproduces `_silu_mul_quant_kernel`'s value sequence instruction for
// instruction, read off its own AMDGCN.  tl.sigmoid(x) = 1 / (1 + tl.exp(-x)), and llvm.exp.f32
// on gfx908 lowers to a denormal-guarded exp2:
//     under = (-x) * log2e < -126 ;  t = fma(log2e, 0 - x, under ? 64 : 0)
//     e = exp2(t) ;  e = ldexp(e, under ? -64 : 0)
// and 1/(1+e) is the FULL IEEE divide expansion (v_div_scale / v_rcp / v_div_fmas / v_div_fixup),
// i.e. plain `/` without -ffast-math -- NOT a bare v_rcp.  The quantizer's rounding is
// round-half-AWAY-from-zero via the CONTRACTED fma(y, inv, +/-0.5) -> floor/ceil, with a PLAIN
// y*inv only for the sign test; note this differs from q8_group's __float2int_rn (half-to-even).
// Verified: int8 / xs / xsum and the layer output are bitwise identical (NaN payloads included)
// to silu_mul_quant + gemv_rowlane over 6 input distributions covering exact zeros, denormals,
// the exp underflow branch and inf-producing overflow (agents/gemv_flight/test_silu.log).
__device__ __forceinline__ float triton_expf(float y) {
  const float LOG2E = 0x1.715476p+0f;                  // 0x3fb8aa3b
  const bool under = (y * LOG2E) < -126.0f;
  const float t = fmaf(LOG2E, y, under ? 64.0f : 0.0f);
  const float e = __builtin_amdgcn_exp2f(t);
  return e * (under ? 0x1p-64f : 1.0f);                // == v_ldexp_f32 by 0 / -64 (exact)
}
__device__ __forceinline__ float triton_sigmoid(float x) {
  return 1.0f / (1.0f + triton_expf(0.0f - x));
}

// 32 lanes cooperate on one group of 32 -- the same width the Triton kernel uses.  Putting all 32
// elements (and so 32 v_exp_f32 + 32 full IEEE divides) in one thread instead made the fold
// 2.6-6.4 us SLOWER than the launch it removes.  max is order-independent and the int sum is exact
// in fp32 (|qsum| <= 32*127), so the cross-lane reduction does not change the value.
__device__ __forceinline__ void silu_q8_group_lp(const float* __restrict__ gp,
                                                 const float* __restrict__ up, int i,
                                                 int8_t* __restrict__ dst8, float& d, float& dsum) {
  const float gv = gp[i];
  const float y = (gv * triton_sigmoid(gv)) * up[i];
  float amax = fabsf(y);
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) amax = fmaxf(amax, __shfl_xor(amax, m, 32));
  d = amax / 127.0f;
  const float inv = amax > 0.f ? 127.0f / amax : 0.f;
  const float v = y * inv;
  const float r = (v >= 0.f) ? __builtin_floorf(fmaf(y, inv, 0.5f))
                             : __builtin_ceilf(fmaf(y, inv, -0.5f));
  const int q = (int)r;
  int qsum = q;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) qsum += __shfl_xor(qsum, m, 32);
  dsum = d * (float)qsum;
  dst8[(i / 8) * 8 + (i % 2) * 4 + (i % 8) / 2] = (int8_t)q;    // per-8 de-interleave
}

// `part` is [P, 2*K] (gate | up) fp32 with K = the INTERMEDIATE width (160); `N` is the down
// projection's OUTPUT width (2560).  HOIST=1 issues the weight loads before the staging barrier.
template <int K, int WPB, int HOIST>
__global__ void __launch_bounds__(64 * WPB) w4a8_rowlane_silu_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const float* __restrict__ part,
    const int* __restrict__ row_tok, const int* __restrict__ row_exp, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se, long stride_pp,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int G = K / 32, NTHR = 64 * WPB;
  __shared__ __align__(16) int8_t ly[K];
  __shared__ float lys[G], lysum[G];
  const int lane = threadIdx.x & 63;
  const int waves_per_p = N / 64;
  const int wave0 = blockIdx.x * WPB;
  if (wave0 >= P * waves_per_p) return;
  const int p = wave0 / waves_per_p;                          // block-uniform
  const int wave = wave0 + (int)(threadIdx.x >> 6);
  const int n = (wave % waves_per_p) * 64 + lane;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wbase;
  const uint16_t* sbase;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
    out[(long)p * N + n] = 0.f;                               // block-uniform: no barrier below
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8));
  const uint16_t* sp = sbase + (long)n * G;
  uint4 q[G];
  uint32_t sc[G];
  if constexpr (HOIST) {
#pragma unroll
    for (int j = 0; j < G; ++j) { q[j] = wp[j]; sc[j] = sp[j]; }
    __builtin_amdgcn_sched_barrier(0);
  }
  {
    const int i = threadIdx.x & 31;
    for (int g = threadIdx.x >> 5; g < G; g += NTHR / 32) {
      const float* pr = part + (long)t * stride_pp + g * 32;
      float d, ds;
      silu_q8_group_lp(pr, pr + K, i, ly + g * 32, d, ds);
      if (i == 0) { lys[g] = d; lysum[g] = ds; }
    }
  }
  __syncthreads();
  if constexpr (!HOIST) {
#pragma unroll
    for (int j = 0; j < G; ++j) { q[j] = wp[j]; sc[j] = sp[j]; }
    __builtin_amdgcn_sched_barrier(0);
  }
  const int* lyi = reinterpret_cast<const int*>(ly);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < G; ++j) {
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      sumi = dot4(qw[wd] & 0x0F0F0F0Fu, lyi[8 * j + 2 * wd], sumi);
      sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, lyi[8 * j + 2 * wd + 1], sumi);
    }
    acc += bf2f(sc[j]) * (lys[j] * (float)sumi - 8.f * lysum[j]);
  }
  out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// Host wrappers
static void check_act(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& xsum, int K) {
  TORCH_CHECK(x8.scalar_type() == at::kChar && x8.is_contiguous() && x8.size(1) == K);
  TORCH_CHECK(xs.scalar_type() == at::kFloat && xsum.scalar_type() == at::kFloat);
  TORCH_CHECK(xs.is_contiguous() && xsum.is_contiguous() && xs.size(1) == K / 32);
}

static void launch_quant(const at::Tensor& x, at::Tensor& x8, at::Tensor& xs, at::Tensor& xsum,
                         const GateArgs* ga, int mrows) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  const int R = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 32 == 0);
  check_act(x8, xs, xsum, K);
  const int total = R * (K / 32);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (total <= 0) return;
  const int nbase = (total + 255) / 256;
  const uint32_t* xp = reinterpret_cast<const uint32_t*>(x.data_ptr());
  uint4* qp = reinterpret_cast<uint4*>(x8.data_ptr());
  if (ga != nullptr) {
    quant_q8_kernel<1><<<nbase + mrows, 256, 0, stream>>>(
        xp, qp, xs.data_ptr<float>(), xsum.data_ptr<float>(), total, nbase, *ga);
  } else {
    quant_q8_kernel<0><<<nbase, 256, 0, stream>>>(
        xp, qp, xs.data_ptr<float>(), xsum.data_ptr<float>(), total, nbase, GateArgs{});
  }
}

// x bf16 [R, K] (K % 32 == 0) -> x8 int8 [R, K], xs fp32 [R, K/32], xsum fp32 [R, K/32]
void quant_q8(at::Tensor x, at::Tensor x8, at::Tensor xs, at::Tensor xsum) {
  launch_quant(x, x8, xs, xsum, nullptr, 0);
}

// Shared expert as expert #E: the row lists / weights for M*(topk+1) pairs, prepared by M extra
// blocks of the quant kernel (no extra launch).  wg bf16 [K], topk_ids int32 [M, topk],
// topk_w fp32 [M, topk] -> row_exp int32 [M, topk+1], wcomb fp32 [M, topk+1].
static GateArgs make_gate_args(const at::Tensor& x, const at::Tensor& wg,
                               const at::Tensor& topk_ids, const at::Tensor& topk_w,
                               at::Tensor& row_exp, at::Tensor& wcomb, int64_t E) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 2 == 0, "gate: K % 2 != 0");
  TORCH_CHECK(wg.scalar_type() == at::kBFloat16 && wg.is_contiguous() && wg.numel() == K);
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt && topk_ids.is_contiguous() && topk_ids.dim() == 2);
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.is_contiguous() && topk_w.dim() == 2);
  TORCH_CHECK(topk_ids.size(0) == M && topk_w.size(0) == M && topk_ids.size(1) == topk_w.size(1));
  const int topk = topk_ids.size(1);
  TORCH_CHECK(row_exp.scalar_type() == at::kInt && row_exp.is_contiguous() &&
              row_exp.numel() == (long)M * (topk + 1));
  TORCH_CHECK(wcomb.scalar_type() == at::kFloat && wcomb.is_contiguous() &&
              wcomb.numel() == (long)M * (topk + 1));
  GateArgs g;
  g.x2 = reinterpret_cast<const uint32_t*>(x.data_ptr());
  g.wg2 = reinterpret_cast<const uint32_t*>(wg.data_ptr());
  g.topk_ids = topk_ids.data_ptr<int32_t>();
  g.topk_w = topk_w.data_ptr<float>();
  g.row_exp = row_exp.data_ptr<int32_t>();
  g.wcomb = wcomb.data_ptr<float>();
  g.khalf = K / 2;
  g.topk = topk;
  g.E = (int)E;
  return g;
}

void quant_q8_gate(at::Tensor x, at::Tensor x8, at::Tensor xs, at::Tensor xsum, at::Tensor wg,
                   at::Tensor topk_ids, at::Tensor topk_w, at::Tensor row_exp, at::Tensor wcomb,
                   int64_t E) {
  const GateArgs g = make_gate_args(x, wg, topk_ids, topk_w, row_exp, wcomb, E);
  launch_quant(x, x8, xs, xsum, &g, (int)x.size(0));
}

// Optional "extra expert" (row_exp == E) weight slab: [1, N, K/8] int32 + [1, N, K/32] bf16.
// An empty tensor means "no extra expert" (the pre-fusion behaviour: row_exp >= E -> zero row).
static void check_extra(const at::Tensor& w_extra, const at::Tensor& s_extra, int N, int K) {
  if (w_extra.numel() == 0) {
    TORCH_CHECK(s_extra.numel() == 0, "extra: w/s must both be empty or both given");
    return;
  }
  TORCH_CHECK(w_extra.is_contiguous() && s_extra.is_contiguous());
  TORCH_CHECK(w_extra.dim() == 3 && s_extra.dim() == 3 && w_extra.size(0) == 1 && s_extra.size(0) == 1);
  TORCH_CHECK(w_extra.size(1) == N && s_extra.size(1) == N && s_extra.size(2) == K / 32);
  TORCH_CHECK(s_extra.scalar_type() == at::kBFloat16);
  TORCH_CHECK((w_extra.scalar_type() == at::kInt && w_extra.size(2) * 8 == K) ||
              (w_extra.scalar_type() == at::kByte && w_extra.size(2) * 2 == K));
}

static void check_gemv(const at::Tensor& w, const at::Tensor& s, const at::Tensor& row_tok,
                       const at::Tensor& row_exp, const at::Tensor& out, int K) {
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous() && s.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w.scalar_type() == at::kInt || w.scalar_type() == at::kByte);
  TORCH_CHECK(row_tok.scalar_type() == at::kInt && row_exp.scalar_type() == at::kInt);
  TORCH_CHECK(row_tok.is_contiguous() && row_exp.is_contiguous() && row_tok.numel() == row_exp.numel());
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous());
  TORCH_CHECK(w.dim() == 3 && s.dim() == 3 && s.size(1) == w.size(1) && s.size(2) == K / 32);
  TORCH_CHECK(out.size(0) == row_tok.numel() && out.size(1) == w.size(1));
}

template <int K, int LPR, int WPB>
static void launch_slab(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x8,
                        const at::Tensor& xs, const at::Tensor& xsum, const at::Tensor& row_tok,
                        const at::Tensor& row_exp, at::Tensor& out, const at::Tensor& w_extra,
                        const at::Tensor& s_extra) {
  constexpr int ROWS = 64 / LPR;
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0, "gemv_slab: N must be a multiple of ", ROWS);
  check_extra(w_extra, s_extra, N, K);
  const long tiles = (long)P * (N / ROWS);
  const int blocks = (int)((tiles + WPB - 1) / WPB);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (blocks > 0) {
    w4a8_slab_kernel<K, LPR, WPB><<<blocks, 64 * WPB, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
        reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(),
        row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E,
        (long)N * (K / 8), (long)N * (K / 32),
        w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr,
        w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr);
  }
}

// D1. w [E, N, K/8] int32 (or uint8 [E, N, K/2] viewed), s [E, N, K/32] bf16, out fp32 [P, N].
// K in {2560, 1536, 160}; lpr = 16; wpb (waves per block) in {1, 4}.
void gemv_slab(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
               at::Tensor row_tok, at::Tensor row_exp, at::Tensor out, int64_t lpr, int64_t wpb,
               at::Tensor w_extra, at::Tensor s_extra) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) * 8 == K) || (w.scalar_type() == at::kByte && w.size(2) * 2 == K),
              "gemv_slab: w K mismatch");
#define SLAB_CASE(KK, L, W) \
  if (K == KK && lpr == L && wpb == W) { launch_slab<KK, L, W>(w, s, x8, xs, xsum, row_tok, row_exp, out, w_extra, s_extra); return; }
  SLAB_CASE(2560, 16, 4) SLAB_CASE(2560, 16, 1)
  SLAB_CASE(1536, 16, 4) SLAB_CASE(1536, 16, 1)
  SLAB_CASE(160, 16, 4) SLAB_CASE(160, 16, 1)
#undef SLAB_CASE
  TORCH_CHECK(false, "gemv_slab: unsupported K/lpr/wpb ", K, "/", lpr, "/", wpb);
}

template <int K, int LPR>
static void launch_slab_prep(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x,
                             const at::Tensor& row_tok, const at::Tensor& row_exp, at::Tensor& out,
                             const at::Tensor& w_extra, const at::Tensor& s_extra) {
  constexpr int ROWS = 64 / LPR;
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0, "gemv_slab_prep: N must be a multiple of ", ROWS);
  TORCH_CHECK((N / ROWS) % 4 == 0, "gemv_slab_prep: (N/ROWS) % 4 != 0");
  check_extra(w_extra, s_extra, N, K);
  const long tiles = (long)P * (N / ROWS);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (tiles > 0) {
    w4a8_slab_prep_kernel<K, LPR><<<(tiles + 3) / 4, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
        reinterpret_cast<const uint32_t*>(x.data_ptr()),
        row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E,
        (long)N * (K / 8), (long)N * (K / 32),
        w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr,
        w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr);
  }
}

// D1 with the Q8_1 activation prep folded into the LDS staging (no quant_q8 launch).
// x is the raw bf16 activation [R, K]; everything else matches gemv_slab.
void gemv_slab_prep(at::Tensor w, at::Tensor s, at::Tensor x, at::Tensor row_tok,
                    at::Tensor row_exp, at::Tensor out, int64_t lpr, at::Tensor w_extra,
                    at::Tensor s_extra) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2,
              "gemv_slab_prep: x must be contiguous bf16 [R, K]");
  const int K = x.size(1);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) * 8 == K) ||
                  (w.scalar_type() == at::kByte && w.size(2) * 2 == K),
              "gemv_slab_prep: w K mismatch");
#define SLABP_CASE(KK, L) \
  if (K == KK && lpr == L) { launch_slab_prep<KK, L>(w, s, x, row_tok, row_exp, out, w_extra, s_extra); return; }
  SLABP_CASE(2560, 16) SLABP_CASE(1536, 16) SLABP_CASE(160, 16)
#undef SLABP_CASE
  TORCH_CHECK(false, "gemv_slab_prep: unsupported K/lpr ", K, "/", lpr);
}

template <int K, int LPR>
static void launch_slab_prep_gate(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x,
                                  const at::Tensor& row_tok, at::Tensor& out,
                                  const at::Tensor& w_extra, const at::Tensor& s_extra,
                                  const GateArgs& ga, int mrows) {
  constexpr int ROWS = 64 / LPR;
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0, "gemv_slab_prep_gate: N must be a multiple of ", ROWS);
  TORCH_CHECK((N / ROWS) % 4 == 0, "gemv_slab_prep_gate: (N/ROWS) % 4 != 0");
  TORCH_CHECK(w_extra.numel() != 0, "gemv_slab_prep_gate: needs the shared expert's slab");
  check_extra(w_extra, s_extra, N, K);
  const long tiles = (long)P * (N / ROWS);
  const int nbase = (int)((tiles + 3) / 4);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (nbase > 0) {
    w4a8_slab_prep_gate_kernel<K, LPR><<<nbase + mrows, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
        reinterpret_cast<const uint32_t*>(x.data_ptr()),
        row_tok.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E,
        (long)N * (K / 8), (long)N * (K / 32),
        reinterpret_cast<const uint32_t*>(w_extra.data_ptr()),
        reinterpret_cast<const uint16_t*>(s_extra.data_ptr()), nbase, ga);
  }
}

// D1 with BOTH the Q8_1 activation prep and the shared-as-expert row/gate prep folded in, so the
// quant_q8_gate launch disappears.  x is the raw bf16 activation [M, K]; row_tok is the
// M*(topk+1) token list; row_exp / wcomb are OUTPUTS (written by the M extra blocks).
void gemv_slab_prep_gate(at::Tensor w, at::Tensor s, at::Tensor x, at::Tensor row_tok,
                         at::Tensor out, int64_t lpr, at::Tensor w_extra, at::Tensor s_extra,
                         at::Tensor wg, at::Tensor topk_ids, at::Tensor topk_w,
                         at::Tensor row_exp, at::Tensor wcomb, int64_t E) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2,
              "gemv_slab_prep_gate: x must be contiguous bf16 [M, K]");
  const int K = x.size(1);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) * 8 == K) ||
                  (w.scalar_type() == at::kByte && w.size(2) * 2 == K),
              "gemv_slab_prep_gate: w K mismatch");
  const GateArgs ga = make_gate_args(x, wg, topk_ids, topk_w, row_exp, wcomb, E);
  const int M = x.size(0);
  TORCH_CHECK(row_tok.numel() == (long)M * (ga.topk + 1),
              "gemv_slab_prep_gate: row_tok must have M*(topk+1) entries");
#define SLABPG_CASE(KK, L) \
  if (K == KK && lpr == L) { \
    launch_slab_prep_gate<KK, L>(w, s, x, row_tok, out, w_extra, s_extra, ga, M); return; }
  SLABPG_CASE(2560, 16) SLABPG_CASE(1536, 16) SLABPG_CASE(160, 16)
#undef SLABPG_CASE
  TORCH_CHECK(false, "gemv_slab_prep_gate: unsupported K/lpr ", K, "/", lpr);
}

// D2. K == 160 only; N % 64 == 0; out fp32 [P, N]; wpb in {1, 4}.
void gemv_rowlane(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
                  at::Tensor row_tok, at::Tensor row_exp, at::Tensor out, int64_t wpb,
                  at::Tensor w_extra, at::Tensor s_extra) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK(K == 160, "gemv_rowlane: K must be 160");
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) == 20) || (w.scalar_type() == at::kByte && w.size(2) == 80));
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % 64 == 0, "gemv_rowlane: N % 64 != 0");
  check_extra(w_extra, s_extra, N, 160);
  const long waves = (long)P * (N / 64);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (waves == 0) return;
  const uint32_t* wx = w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr;
  const uint16_t* sx = w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr;
#define ROWLANE_CASE(W) \
  if (wpb == W) { \
    w4a8_rowlane_kernel<160, W><<<(waves + W - 1) / W, 64 * W, 0, stream>>>( \
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()), \
        reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(), \
        row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E, \
        (long)N * (160 / 8), (long)N * 5, wx, sx); \
    return; \
  }
  ROWLANE_CASE(1) ROWLANE_CASE(4)
#undef ROWLANE_CASE
  TORCH_CHECK(false, "gemv_rowlane: wpb must be 1 or 4");
}


// D2 + silu fold.  K (intermediate) == 160 only; part fp32 [P, 320]; N % 64 == 0 and
// (N / 64) % wpb == 0; out fp32 [P, N].
void gemv_rowlane_silu(at::Tensor w, at::Tensor s, at::Tensor part, at::Tensor row_tok,
                       at::Tensor row_exp, at::Tensor out, int64_t wpb, int64_t hoist,
                       at::Tensor w_extra, at::Tensor s_extra) {
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.dim() == 2 && part.size(1) == 320,
              "gemv_rowlane_silu: part must be fp32 [P, 320]");
  check_gemv(w, s, row_tok, row_exp, out, 160);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) == 20) ||
              (w.scalar_type() == at::kByte && w.size(2) == 80));
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % 64 == 0, "gemv_rowlane_silu: N % 64 != 0");
  check_extra(w_extra, s_extra, N, 160);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const long waves = (long)P * (N / 64);
  if (waves == 0) return;
  const uint32_t* wx = w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr;
  const uint16_t* sx = w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr;
#define RLSILU_CASE(W, H) \
  if (wpb == W && hoist == H) { \
    TORCH_CHECK((N / 64) % W == 0, "gemv_rowlane_silu: (N/64) % wpb != 0"); \
    w4a8_rowlane_silu_kernel<160, W, H><<<(int)(waves / W), 64 * W, 0, stream>>>( \
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()), \
        part.data_ptr<float>(), row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), \
        out.data_ptr<float>(), N, P, E, (long)N * 20, (long)N * 5, part.stride(0), wx, sx); \
    return; \
  }
  RLSILU_CASE(4, 0) RLSILU_CASE(8, 1) RLSILU_CASE(4, 1) RLSILU_CASE(8, 0)
#undef RLSILU_CASE
  TORCH_CHECK(false, "gemv_rowlane_silu: unsupported wpb/hoist ", wpb, "/", hoist);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_q8", &quant_q8, "x bf16 [R,K] -> x8 int8 [R,K] (per-8 de-interleaved), xs/xsum fp32 [R,K/32]");
  m.def("quant_q8_gate", &quant_q8_gate,
        "quant_q8 + M extra blocks: row_exp/wcomb [M,topk+1] from topk_ids/topk_w and sigmoid(x.wg)");
  m.def("gemv_slab", &gemv_slab,
        "w, s, x8, xs, xsum, row_tok, row_exp, out fp32 [P,N], lpr(16), wpb(1|4), w_extra, s_extra");
  m.def("gemv_slab_prep", &gemv_slab_prep,
        "w, s, x bf16 [R,K], row_tok, row_exp, out fp32 [P,N], lpr(16), w_extra, s_extra "
        "(Q8_1 prep folded into the LDS staging)");
  m.def("gemv_slab_prep_gate", &gemv_slab_prep_gate,
        "w, s, x bf16 [M,K], row_tok, out fp32 [P,N], lpr(16), w_extra, s_extra, wg, topk_ids, "
        "topk_w, row_exp(out), wcomb(out), E  (Q8_1 prep + shared-as-expert row/gate prep folded "
        "into the slab GEMV)");
  m.def("gemv_rowlane_silu", &gemv_rowlane_silu,
        "w, s, part fp32 [P,320] (gate|up), row_tok, row_exp, out fp32 [P,N], wpb, hoist, "
        "w_extra, s_extra  (silu*mul + Q8_1 folded into the down GEMV; bit-exact)");
  m.def("gemv_rowlane", &gemv_rowlane,
        "w, s, x8, xs, xsum, row_tok, row_exp, out fp32 [P,N], wpb(1|4), w_extra, s_extra  (K=160)");
}
