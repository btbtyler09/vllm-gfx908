// gfx908 W4A16-EXACT GEMV kernels (packed-fp16 `v_dot2_f32_f16`), the `f16` mode of
// VLLM_GFX908_W4A8_MODE.  Same geometry / layouts / call pattern as the int8 kernels in
// gfx908_w4a8.hip (D1 slab, D2 rowlane), but the activation is NOT quantized: bf16 -> fp16 is
// lossless (bf16 has 8 mantissa bits, fp16 has 11) so the result matches the stock W4A16 GEMV to
// fp32 summation-order noise (~5e-7 relative L2, measured in agents/w4f16/REPORT.md).
// Design: docs/mi100_decode_opt/research/w4_gemv_kernels.md section D5.
//
// Weights: [E, N, K/8] uint32, 8 nibbles along K (nibble t of word wd = k 8*wd+t), symmetric
// zero point 8; scales [E, N, K/32] bf16 (group size 32, one uint4 = one group).  NO REPACK.
//
// Nibble -> fp16 with the 0x6400 magic number (0x6400 = fp16 1024.0; 1024+q is exact in fp16
// because ulp = 1 in [1024, 2048)).  One `v_and_or_b32` builds two biased weights at once:
//     (q        & 0x000F000F) | 0x64006400  ->  (1024+n0, 1024+n4)
//     ((q >> 4) & 0x000F000F) | 0x64006400  ->  (1024+n1, 1024+n5)
//     ((q >> 8) & 0x000F000F) | 0x64006400  ->  (1024+n2, 1024+n6)
//     ((q >>12) & 0x000F000F) | 0x64006400  ->  (1024+n3, 1024+n7)
// so the activation is stored per 8-block in the order [k0 k4 k1 k5 k2 k6 k3 k7] (dword j of the
// block = the fp16 pair (k_j, k_{j+4})) -- what `cast_f16` and the Triton silu*mul kernel write.
// The 1024 bias and the zero point 8 are removed by a single `v_pk_add_f16` of -1032 on the
// packed pair before the dot; that is exact (Sterbenz), which is why this path needs no per-token
// activation scale, no `xsum`/`xbias` vector and no accuracy gate.  (The microbench's cheaper but
// lossy BIAS=0 "sum trick" / BIAS=2 "mixed magic" variants measured within noise of this one in
// LDS mode, so only the exact variant is ported here -- see agents/w4f16/REPORT.md section 2.)
//
// Output rows are pairs: out[p, :] = W[row_exp[p]] . x[row_tok[p]] as fp32 [P, N], ready for the
// same Triton reduce epilogues as the int8 path (SPLIT_K = 1).
// row_exp[p] outside [0, E) -> zero row (cudagraph capture routing).
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf2f(uint32_t v) { return __uint_as_float(v << 16); }

// (a & b) | c  -> v_and_or_b32.  hipcc does NOT contract this (it emits v_and_b32 + v_or_b32),
// and forcing the asm form is worth -21% VALU in the slab kernel and 63 -> 33 VGPRs in the
// rowlane kernel (occupancy 4 -> 7 waves/SIMD).  -DW4F16_NO_ASM_ANDOR falls back to C.
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
#ifndef W4F16_NO_ASM_ANDOR
  uint32_t r;
  asm("v_and_or_b32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
#else
  return (a & b) | c;
#endif
}

// c + a.lo*b.lo + a.hi*b.hi -> v_dot2_f32_f16 / v_dot2c_f32_f16
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  union { uint32_t u; h2_t v; } ua, ub;
  ua.u = a; ub.u = b;
  return __builtin_amdgcn_fdot2(ua.v, ub.v, c, false);
}

// packed fp16 pair minus 1032 (= 1024 magic bias + zero point 8); exact -> v_pk_add_f16
__device__ __forceinline__ uint32_t sub_1032(uint32_t a) {
  union { uint32_t u; h2_t v; } ua, ub;
  ua.u = a; ub.u = 0xE408E408u;   // fp16(-1032) twice
  ua.v = ua.v + ub.v;
  return ua.u;
}

template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
  return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true));
}

// Sum over LPR consecutive lanes; result in lane LPR-1 of each group (row_shr prefix sums).
template <int LPR>
__device__ __forceinline__ float reduce_lpr(float v) {
  if constexpr (LPR >= 2) v = dpp_add<0x111>(v);   // row_shr:1
  if constexpr (LPR >= 4) v = dpp_add<0x112>(v);
  if constexpr (LPR >= 8) v = dpp_add<0x114>(v);
  if constexpr (LPR >= 16) v = dpp_add<0x118>(v);
  if constexpr (LPR >= 32) v += __shfl_xor(v, 16, 64);
  if constexpr (LPR >= 64) v += __shfl_xor(v, 32, 64);
  return v;
}

// one group of 32 weights (uint4 q) x 32 activations (4 uint4 of packed fp16 pairs)
__device__ __forceinline__ float group_dot(const uint4& q, const uint4* x) {
  const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
  float g = 0.f;
#pragma unroll
  for (int wd = 0; wd < 4; ++wd) {
    const uint32_t xw[4] = {x[wd].x, x[wd].y, x[wd].z, x[wd].w};
    uint32_t p[4];
    p[0] = and_or(qw[wd], 0x000F000Fu, 0x64006400u);
    p[1] = and_or(qw[wd] >> 4, 0x000F000Fu, 0x64006400u);
    p[2] = and_or(qw[wd] >> 8, 0x000F000Fu, 0x64006400u);
    p[3] = and_or(qw[wd] >> 12, 0x000F000Fu, 0x64006400u);
#pragma unroll
    for (int i = 0; i < 4; ++i) g = dot2(sub_1032(p[i]), xw[i], g);
  }
  return g;
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
  for (int k = tid; k < g.khalf; k += 256) {
    const uint32_t a = xr[k], b = g.wg2[k];
    acc = fmaf(bf2f(a & 0xFFFFu), bf2f(b & 0xFFFFu), acc);
    acc = fmaf(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), acc);
  }
  red[tid] = acc;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  if (tid == 0) {
    g.row_exp[m * stride + g.topk] = g.E;
    g.wcomb[m * stride + g.topk] = 1.f / (1.f + expf(-red[0]));
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
// Activation cast: one thread per (row, group of 32). x bf16 [R, K] -> xh fp16 [R, K], per-8
// permuted to [k0 k4 k1 k5 k2 k6 k3 k7].  CHECK=1 also reports fp16 overflow / flush-to-zero
// counts and max|x| in ovf[0..2] (the fp16 range audit; CHECK=0 is the production path).
// GATE=1 appends the M row-prep blocks described above.
template <int CHECK, int GATE>
__global__ void __launch_bounds__(256) cast_f16_kernel(
    const uint32_t* __restrict__ x, uint4* __restrict__ xh, uint32_t* __restrict__ ovf, int total,
    int nbase, GateArgs ga) {
  if constexpr (GATE) {
    if ((int)blockIdx.x >= nbase) { gate_row(ga, (int)blockIdx.x - nbase); return; }
  }
  const int gid = blockIdx.x * 256 + threadIdx.x;
  if (gid >= total) return;
  const uint4* xp = reinterpret_cast<const uint4*>(x) + (long)gid * 4;   // 64 B = 32 bf16
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
  _Float16 h[32];
  int nov = 0, nun = 0;
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    h[i] = (_Float16)f[i];
    if constexpr (CHECK) {
      const float hf = (float)h[i];
      amax = fmaxf(amax, fabsf(f[i]));
      if (!isfinite(hf) && isfinite(f[i])) ++nov;
      if (hf == 0.f && f[i] != 0.f) ++nun;
    }
  }
  uint32_t o[16];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      union { _Float16 hh; uint16_t u; } a, c;
      a.hh = h[8 * b + j]; c.hh = h[8 * b + j + 4];
      o[4 * b + j] = (uint32_t)a.u | ((uint32_t)c.u << 16);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
    xh[(long)gid * 4 + i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
  if constexpr (CHECK) {
    if (nov) atomicAdd(ovf, (uint32_t)nov);
    if (nun) atomicAdd(ovf + 1, (uint32_t)nun);
    atomicMax(ovf + 2, __float_as_uint(amax));
  }
}

// ---------------------------------------------------------------------------------------------
// D5-slab (D1 geometry): LPR lanes per output row, 64/LPR rows per wave, lane l owns groups
// l + LPR*j.  MODE 0 = the pair's activation preloaded into VGPRs before the math.
template <int K, int LPR, int WPB>
__global__ void __launch_bounds__(64 * WPB) w4f16_slab_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const uint16_t* __restrict__ xh,
    const int* __restrict__ row_tok, const int* __restrict__ row_exp, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR;
  const int tiles_per_p = N / ROWS;
  const int tile = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  if (tile >= P * tiles_per_p) return;
  const int p = tile / tiles_per_p, n = (tile % tiles_per_p) * ROWS + r;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wb;
  const uint16_t* sb;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wb, sb)) {
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(wb + (long)n * (K / 8)) + l;
  const uint16_t* sp = sb + (long)n * G + l;
  const uint4* xp = reinterpret_cast<const uint4*>(xh + (long)t * K) + 4 * l;  // 4 uint4 per group
  uint4 q[GPL], xv[GPL * 4];
  uint32_t sc[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) { q[j] = wp[LPR * j]; sc[j] = sp[LPR * j]; }
    else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      xv[4 * j + i] = (G % LPR == 0 || l + LPR * j < G) ? xp[4 * LPR * j + i]
                                                        : make_uint4(0, 0, 0, 0);
    }
  }
  __builtin_amdgcn_sched_barrier(0);   // all loads above the first use
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) acc = fmaf(bf2f(sc[j]), group_dot(q[j], &xv[4 * j]), acc);
  acc = reduce_lpr<LPR>(acc);
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// D5-slab with the pair's activation staged in LDS once per block (4 waves = 4*ROWS rows of the
// same pair).  Needs (N / ROWS) % 4 == 0 so a block never straddles two pairs -- which also makes
// the zero-fill branch block-uniform, so there is no __syncthreads hazard.  This is the fastest
// variant at every shape and every M in the microbench (41 VGPRs / 5 waves per SIMD at K=2560).
template <int K, int LPR>
__global__ void __launch_bounds__(256) w4f16_slab_lds_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const uint16_t* __restrict__ xh,
    const int* __restrict__ row_tok, const int* __restrict__ row_exp, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  __shared__ __align__(16) uint16_t xl[K];
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles_per_p = N / ROWS;
  const int tile0 = __builtin_amdgcn_readfirstlane(blockIdx.x * 4);
  const int p = tile0 / tiles_per_p;
  const int n = ((tile0 + wv) % tiles_per_p) * ROWS + r;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wb;
  const uint16_t* sb;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wb, sb)) {  // block-uniform
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(wb + (long)n * (K / 8)) + l;
  const uint16_t* sp = sb + (long)n * G + l;
  uint4 q[GPL];
  uint32_t sc[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) { q[j] = wp[LPR * j]; sc[j] = sp[LPR * j]; }
    else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }
  const uint4* xg = reinterpret_cast<const uint4*>(xh + (long)t * K);
  for (int i = threadIdx.x; i < K / 8; i += 256) reinterpret_cast<uint4*>(xl)[i] = xg[i];
  __syncthreads();
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const int g = l + LPR * j;
    if (G % LPR == 0 || g < G)
      acc = fmaf(bf2f(sc[j]), group_dot(q[j], reinterpret_cast<const uint4*>(xl) + 4 * g), acc);
  }
  acc = reduce_lpr<LPR>(acc);
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// bf16 -> fp16 cast + per-8 permutation of ONE group of 32 (64 B) -- exactly the arithmetic of
// cast_f16_kernel<0, 0>, factored out so the fused-prep kernel below is bit-identical to it.
__device__ __forceinline__ void cast_group(const uint4* __restrict__ src, uint4 o[4]) {
  uint4 v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = src[i];
  // one packed dword-quad at a time (8 fp16 in flight) rather than 32 fp32 + 32 fp16: same values,
  // same order, ~20 fewer VGPRs live next to the weight registers.
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint32_t dw[4] = {v[b].x, v[b].y, v[b].z, v[b].w};
    _Float16 h[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      h[2 * j] = (_Float16)__uint_as_float(dw[j] << 16);
      h[2 * j + 1] = (_Float16)__uint_as_float(dw[j] & 0xFFFF0000u);
    }
    uint32_t p[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      union { _Float16 hh; uint16_t u; } a, c;
      a.hh = h[j]; c.hh = h[j + 4];
      p[j] = (uint32_t)a.u | ((uint32_t)c.u << 16);
    }
    o[b] = make_uint4(p[0], p[1], p[2], p[3]);
  }
}

// ---------------------------------------------------------------------------------------------
// D5-slab-LDS with the ACTIVATION PREP FOLDED IN (VLLM_GFX908_MOE_PREP_FOLD=1): the block reads
// the pair's activation as raw bf16 and does the fp16 cast + per-8 permutation while staging it in
// LDS, so the separate cast_f16 launch disappears.  Same geometry / block-uniformity requirements
// as w4f16_slab_lds_kernel ((N / ROWS) % 4 == 0), and bit-identical to cast_f16 + gemv_slab_f16
// (mode 1) because the staged bytes and the read order are the same.
template <int K, int LPR>
__global__ void __launch_bounds__(256) w4f16_slab_prep_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const uint32_t* __restrict__ x,
    const int* __restrict__ row_tok, const int* __restrict__ row_exp, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  __shared__ __align__(16) uint16_t xl[K];
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles_per_p = N / ROWS;
  const int tile0 = __builtin_amdgcn_readfirstlane(blockIdx.x * 4);
  const int p = tile0 / tiles_per_p;
  const int n = ((tile0 + wv) % tiles_per_p) * ROWS + r;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wb;
  const uint16_t* sb;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wb, sb)) {  // block-uniform
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;
    return;
  }
  // Weight loads are issued AFTER the staging barrier: holding q[GPL] + sc[GPL] live across
  // the cast code costs a wave of occupancy (see the int8 kernel's note).
  const uint4* xg = reinterpret_cast<const uint4*>(x) + (long)t * (K / 8);   // K bf16 = K/8 uint4
  for (int g = threadIdx.x; g < G; g += 256) {
    uint4 o[4];
    cast_group(xg + (long)g * 4, o);
#pragma unroll
    for (int i = 0; i < 4; ++i) reinterpret_cast<uint4*>(xl)[4 * g + i] = o[i];
  }
  __syncthreads();
  const uint4* wp = reinterpret_cast<const uint4*>(wb + (long)n * (K / 8)) + l;
  const uint16_t* sp = sb + (long)n * G + l;
  uint4 q[GPL];
  uint32_t sc[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) { q[j] = wp[LPR * j]; sc[j] = sp[LPR * j]; }
    else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }
  __builtin_amdgcn_sched_barrier(0);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const int g = l + LPR * j;
    if (G % LPR == 0 || g < G)
      acc = fmaf(bf2f(sc[j]), group_dot(q[j], reinterpret_cast<const uint4*>(xl) + 4 * g), acc);
  }
  acc = reduce_lpr<LPR>(acc);
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// D5-rowlane (D2 geometry): one thread per output row, for small K (K=160 -> 5 groups, 320 B of
// fp16 activation per pair).  One pair per wave, so the activation is wave-uniform (SGPRs).
template <int K, int WPB>
__global__ void __launch_bounds__(64 * WPB) w4f16_rowlane_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const uint16_t* __restrict__ xh,
    const int* __restrict__ row_tok, const int* __restrict__ row_exp, float* __restrict__ out,
    int N, int P, int E, long stride_we, long stride_se,
    const uint32_t* __restrict__ w_extra, const uint16_t* __restrict__ s_extra) {
  constexpr int G = K / 32;
  const int lane = threadIdx.x & 63;
  const int waves_per_p = N / 64;
  const int wave = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  if (wave >= P * waves_per_p) return;
  const int p = wave / waves_per_p, n = (wave % waves_per_p) * 64 + lane;
  const int e = row_exp[p], t = row_tok[p];
  const uint32_t* wb;
  const uint16_t* sb;
  if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wb, sb)) {
    out[(long)p * N + n] = 0.f;
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(wb + (long)n * (K / 8));
  const uint16_t* sp = sb + (long)n * G;
  uint4 q[G];
  uint32_t sc[G];
#pragma unroll
  for (int j = 0; j < G; ++j) { q[j] = wp[j]; sc[j] = sp[j]; }
  const uint4* xu = reinterpret_cast<const uint4*>(xh + (long)t * K);   // wave-uniform -> SGPRs
  uint4 xv[4 * G];
#pragma unroll
  for (int i = 0; i < 4 * G; ++i) xv[i] = xu[i];
  __builtin_amdgcn_sched_barrier(0);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < G; ++j) acc = fmaf(bf2f(sc[j]), group_dot(q[j], &xv[4 * j]), acc);
  out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// Host wrappers
static void check_act(const at::Tensor& xh, int K) {
  TORCH_CHECK(xh.scalar_type() == at::kHalf && xh.is_contiguous() && xh.dim() == 2 && xh.size(1) == K);
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

static void launch_cast(const at::Tensor& x, at::Tensor& xh, uint32_t* ovf, bool check,
                        const GateArgs* ga, int mrows) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2);
  const int R = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 32 == 0, "cast_f16: K % 32 != 0");
  check_act(xh, K);
  TORCH_CHECK(xh.size(0) == R);
  const int total = R * (K / 32);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (total <= 0) return;
  const uint32_t* xp = reinterpret_cast<const uint32_t*>(x.data_ptr());
  uint4* hp = reinterpret_cast<uint4*>(xh.data_ptr());
  const int nbase = (total + 255) / 256;
  if (ga != nullptr) {
    cast_f16_kernel<0, 1><<<nbase + mrows, 256, 0, stream>>>(xp, hp, nullptr, total, nbase, *ga);
  } else if (check) {
    cast_f16_kernel<1, 0><<<nbase, 256, 0, stream>>>(xp, hp, ovf, total, nbase, GateArgs{});
  } else {
    cast_f16_kernel<0, 0><<<nbase, 256, 0, stream>>>(xp, hp, nullptr, total, nbase, GateArgs{});
  }
}

// x bf16 [R, K] (K % 32 == 0) -> xh fp16 [R, K], per-8 permuted [k0 k4 k1 k5 k2 k6 k3 k7]
void cast_f16(at::Tensor x, at::Tensor xh) { launch_cast(x, xh, nullptr, false, nullptr, 0); }

// same, plus the fp16 range audit: ovf int32 [>=3] = {#overflows, #flush-to-zero, bits(max|x|)}
void cast_f16_audit(at::Tensor x, at::Tensor xh, at::Tensor ovf) {
  TORCH_CHECK(ovf.numel() >= 3 && ovf.scalar_type() == at::kInt && ovf.is_contiguous());
  launch_cast(x, xh, reinterpret_cast<uint32_t*>(ovf.data_ptr()), true, nullptr, 0);
}

// Shared expert as expert #E: the row lists / weights for M*(topk+1) pairs, prepared by M extra
// blocks of the cast kernel (no extra launch).  wg bf16 [K], topk_ids int32 [M, topk],
// topk_w fp32 [M, topk] -> row_exp int32 [M, topk+1], wcomb fp32 [M, topk+1].
static GateArgs make_gate_args(const at::Tensor& x, const at::Tensor& wg, const at::Tensor& topk_ids,
                               const at::Tensor& topk_w, at::Tensor& row_exp, at::Tensor& wcomb,
                               int64_t E) {
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

void cast_f16_gate(at::Tensor x, at::Tensor xh, at::Tensor wg, at::Tensor topk_ids,
                   at::Tensor topk_w, at::Tensor row_exp, at::Tensor wcomb, int64_t E) {
  GateArgs g = make_gate_args(x, wg, topk_ids, topk_w, row_exp, wcomb, E);
  launch_cast(x, xh, nullptr, false, &g, (int)x.size(0));
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

template <int K, int LPR, int WPB, int MODE>
static void launch_slab(const at::Tensor& w, const at::Tensor& s, const at::Tensor& xh,
                        const at::Tensor& row_tok, const at::Tensor& row_exp, at::Tensor& out,
                        const at::Tensor& w_extra, const at::Tensor& s_extra) {
  constexpr int ROWS = 64 / LPR;
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0, "gemv_slab_f16: N must be a multiple of ", ROWS);
  check_extra(w_extra, s_extra, N, K);
  const long tiles = (long)P * (N / ROWS);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (tiles <= 0) return;
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(w.data_ptr());
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(s.data_ptr());
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(xh.data_ptr());
  const uint32_t* wx = w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr;
  const uint16_t* sx = w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr;
  if constexpr (MODE == 1) {
    TORCH_CHECK(WPB == 4 && (N / ROWS) % 4 == 0, "gemv_slab_f16: LDS mode needs wpb=4 and (N/ROWS) % 4 == 0");
    w4f16_slab_lds_kernel<K, LPR><<<(tiles + 3) / 4, 256, 0, stream>>>(
        wp, sp, xp, row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(),
        N, P, E, (long)N * (K / 8), (long)N * (K / 32), wx, sx);
  } else {
    w4f16_slab_kernel<K, LPR, WPB><<<(tiles + WPB - 1) / WPB, 64 * WPB, 0, stream>>>(
        wp, sp, xp, row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(),
        N, P, E, (long)N * (K / 8), (long)N * (K / 32), wx, sx);
  }
}

// D5 slab. w [E, N, K/8] int32 (or uint8 [E, N, K/2] viewed), s [E, N, K/32] bf16, out fp32 [P, N].
// K in {2560, 1536}; lpr = 16; mode 0 = x in VGPRs, 1 = x staged in LDS (needs wpb = 4).
void gemv_slab_f16(at::Tensor w, at::Tensor s, at::Tensor xh, at::Tensor row_tok,
                   at::Tensor row_exp, at::Tensor out, int64_t lpr, int64_t mode, int64_t wpb,
                   at::Tensor w_extra, at::Tensor s_extra) {
  const int K = xh.size(1);
  check_act(xh, K);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) * 8 == K) ||
                  (w.scalar_type() == at::kByte && w.size(2) * 2 == K),
              "gemv_slab_f16: w K mismatch");
#define SLAB_CASE(KK, L, W, MO)                                                       \
  if (K == KK && lpr == L && wpb == W && mode == MO) {                                \
    launch_slab<KK, L, W, MO>(w, s, xh, row_tok, row_exp, out, w_extra, s_extra); return; }
  SLAB_CASE(2560, 16, 4, 1) SLAB_CASE(2560, 16, 4, 0) SLAB_CASE(2560, 16, 1, 0)
  SLAB_CASE(1536, 16, 4, 1) SLAB_CASE(1536, 16, 4, 0) SLAB_CASE(1536, 16, 1, 0)
#undef SLAB_CASE
  TORCH_CHECK(false, "gemv_slab_f16: unsupported K/lpr/mode/wpb ", K, "/", lpr, "/", mode, "/", wpb);
}

template <int K, int LPR>
static void launch_slab_prep(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x,
                             const at::Tensor& row_tok, const at::Tensor& row_exp, at::Tensor& out,
                             const at::Tensor& w_extra, const at::Tensor& s_extra) {
  constexpr int ROWS = 64 / LPR;
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0, "gemv_slab_f16_prep: N must be a multiple of ", ROWS);
  TORCH_CHECK((N / ROWS) % 4 == 0, "gemv_slab_f16_prep: (N/ROWS) % 4 != 0");
  check_extra(w_extra, s_extra, N, K);
  const long tiles = (long)P * (N / ROWS);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (tiles > 0) {
    w4f16_slab_prep_kernel<K, LPR><<<(tiles + 3) / 4, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
        reinterpret_cast<const uint32_t*>(x.data_ptr()),
        row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E,
        (long)N * (K / 8), (long)N * (K / 32),
        w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr,
        w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr);
  }
}

// D5 slab with the bf16 -> fp16 cast + permutation folded into the LDS staging (no cast_f16
// launch).  x is the raw bf16 activation [R, K]; everything else matches gemv_slab_f16 (mode 1).
void gemv_slab_f16_prep(at::Tensor w, at::Tensor s, at::Tensor x, at::Tensor row_tok,
                        at::Tensor row_exp, at::Tensor out, int64_t lpr, at::Tensor w_extra,
                        at::Tensor s_extra) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2,
              "gemv_slab_f16_prep: x must be contiguous bf16 [R, K]");
  const int K = x.size(1);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) * 8 == K) ||
                  (w.scalar_type() == at::kByte && w.size(2) * 2 == K),
              "gemv_slab_f16_prep: w K mismatch");
#define SLABP_CASE(KK, L) \
  if (K == KK && lpr == L) { launch_slab_prep<KK, L>(w, s, x, row_tok, row_exp, out, w_extra, s_extra); return; }
  SLABP_CASE(2560, 16) SLABP_CASE(1536, 16)
#undef SLABP_CASE
  TORCH_CHECK(false, "gemv_slab_f16_prep: unsupported K/lpr ", K, "/", lpr);
}

// D5 rowlane. K == 160 only; N % 64 == 0; out fp32 [P, N]; wpb in {1, 4}.
void gemv_rowlane_f16(at::Tensor w, at::Tensor s, at::Tensor xh, at::Tensor row_tok,
                      at::Tensor row_exp, at::Tensor out, int64_t wpb, at::Tensor w_extra,
                      at::Tensor s_extra) {
  const int K = xh.size(1);
  check_act(xh, K);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK(K == 160, "gemv_rowlane_f16: K must be 160");
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) == 20) ||
              (w.scalar_type() == at::kByte && w.size(2) == 80));
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % 64 == 0, "gemv_rowlane_f16: N % 64 != 0");
  check_extra(w_extra, s_extra, N, 160);
  const long waves = (long)P * (N / 64);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (waves <= 0) return;
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(w.data_ptr());
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(s.data_ptr());
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(xh.data_ptr());
  const uint32_t* wx = w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr;
  const uint16_t* sx = w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr;
#define ROWLANE_CASE(W)                                                            \
  if (wpb == W) {                                                                  \
    w4f16_rowlane_kernel<160, W><<<(waves + W - 1) / W, 64 * W, 0, stream>>>(      \
        wp, sp, xp, row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(),      \
        out.data_ptr<float>(), N, P, E, (long)N * (160 / 8), (long)N * 5, wx, sx); \
    return;                                                                        \
  }
  ROWLANE_CASE(1) ROWLANE_CASE(4)
#undef ROWLANE_CASE
  TORCH_CHECK(false, "gemv_rowlane_f16: wpb must be 1 or 4");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cast_f16", &cast_f16, "x bf16 [R,K] -> xh fp16 [R,K] (per-8 permuted [k0 k4 k1 k5 ...])");
  m.def("cast_f16_audit", &cast_f16_audit, "cast_f16 + ovf int32 [>=3] {#ovf, #flush0, bits(max|x|)}");
  m.def("cast_f16_gate", &cast_f16_gate,
        "cast_f16 + M extra blocks: row_exp/wcomb [M,topk+1] from topk_ids/topk_w and sigmoid(x.wg)");
  m.def("gemv_slab_f16", &gemv_slab_f16,
        "w, s, xh, row_tok, row_exp, out fp32 [P,N], lpr(16), mode(0|1), wpb(1|4), w_extra, s_extra");
  m.def("gemv_slab_f16_prep", &gemv_slab_f16_prep,
        "w, s, x bf16 [R,K], row_tok, row_exp, out fp32 [P,N], lpr(16), w_extra, s_extra "
        "(bf16->fp16 cast + permute folded into the LDS staging)");
  m.def("gemv_rowlane_f16", &gemv_rowlane_f16,
        "w, s, xh, row_tok, row_exp, out fp32 [P,N], wpb(1|4), w_extra, s_extra  (K=160)");
}
