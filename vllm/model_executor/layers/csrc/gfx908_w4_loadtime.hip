// gfx908 W4 GEMV kernels for *load-time quantized* dense decode projections
// (VLLM_GFX908_W4_LOADTIME), i.e. weights that ship as bf16 and are RTN-quantized to the
// W4 GS32 layout at load time instead of being kept as int8.
//
// This is the dense (E = 1, no row lists) sibling of
// fused_moe/csrc/gfx908_w4a8.hip (D1 slab, int8 activations) and
// fused_moe/csrc/gfx908_w4f16.hip (D5 slab, fp16 activations).  Three differences:
//
//   1. no expert indirection: pair p == token m, so there is no row_tok / row_exp read and
//      no zero-row branch;
//   2. the output stage is bf16 (or one of the two hyper-connection epilogues), not fp32,
//      so the caller needs no reduce/convert launch;
//   3. LPR = 64 (one output row per wave) is instantiated as well as LPR = 16, for the very
//      narrow / very long mix_down shape (N = 336, K = 10240).
//
// Weight layout (identical to the MoE one, minus the expert dim):
//   w  [N, K/8] uint32   8 nibbles along K, nibble t of word wd = k 8*wd + t, zero point 8
//   s  [N, K/gs] bf16    group scales, gs in {32, 64}; the group-of-32 index is shifted right
//                        by `gshift` (0 for gs=32, 1 for gs=64) to address it
//
// Activations, exactly as the MoE kernels consume them (produced by cast_f16 / quant_q8 of
// gfx908_w4f16.hip / gfx908_w4a8.hip -- this file does NOT duplicate the prep kernels):
//   ACT 0 (f16, W4A16-exact): xh fp16 [M, K], per-8 permuted [k0 k4 k1 k5 k2 k6 k3 k7]
//   ACT 1 (int8, Q8_1):       x8 int8 [M, K], per-8 de-interleaved [k0 k2 k4 k6 | k1 k3 k5 k7]
//                             + xs / xsum fp32 [M, K/32]
//   ACT 2 (bf16 direct):      x bf16 [M, K] *natural order* -- the bf16 -> fp16 cast and the
//                             per-8 permutation are done while staging the token into LDS, so
//                             this mode needs NO activation-prep launch at all and is exactly
//                             as accurate as ACT 0 (bf16 -> fp16 is lossless).  MODE must be 1.
//
// Epilogues (EPI), ported from models/qwen4_exp/amd/csrc/gfx908_wv_fused.hip:
//   0  plain:      C bf16 [M, NT], C[m, n] = acc
//   1  mix_down:   n < R  -> t = bf16(acc)/HC; silu(t) written as **fp16 in the permuted
//                            activation layout** of xh_out [M, R], so the mix_up GEMV can read
//                            it with no prep launch;
//                  R <= n < R+HC -> inj bf16 [M, HC] (the block-inject columns, written
//                            compactly so the chain needs no `.contiguous()` either).
//   2  mix_up:     weight rows are pre-permuted (row i*HC + s = original row s*HD + i) and
//                  ROWS == HC, so one wave holds the HC gates of output feature i = tile:
//                  Y[m, i] = bf16( mean_s sigmoid(bf16(acc_s)) * xn[m, s*HD + i] ).
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

// 16-byte non-temporal load (the weight is streamed once per token and never reused, so it must
// not evict the tiny, hot activation from L2).  -DW4LT_NO_NT falls back to a normal load.
__device__ __forceinline__ uint4 load_nt(const uint4* p) {
#ifdef W4LT_NO_NT
  return *p;
#else
  const u32x4_t v = __builtin_nontemporal_load(reinterpret_cast<const u32x4_t*>(p));
  return make_uint4(v.x, v.y, v.z, v.w);
#endif
}

__device__ __forceinline__ float bf2f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  // round-to-nearest-even, same as __float2bfloat16
  uint32_t u = __float_as_uint(f);
  uint32_t r = ((u >> 16) & 1u) + 0x7FFFu;
  return (uint16_t)((u + r) >> 16);
}

// (a & b) | c -> v_and_or_b32 (hipcc does not contract this; worth -21% VALU, see gfx908_w4f16.hip)
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t b, uint32_t c) {
#ifndef W4LT_NO_ASM_ANDOR
  uint32_t r;
  asm("v_and_or_b32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
#else
  return (a & b) | c;
#endif
}

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  union { uint32_t u; h2_t v; } ua, ub;
  ua.u = a; ub.u = b;
  return __builtin_amdgcn_fdot2(ua.v, ub.v, c, false);
}

__device__ __forceinline__ int dot4(uint32_t a, int b, int c) {
  return __builtin_amdgcn_sdot4((int)a, b, c, false);
}

// packed fp16 pair minus 1032 (1024 magic bias + zero point 8); exact -> v_pk_add_f16
__device__ __forceinline__ uint32_t sub_1032(uint32_t a) {
  union { uint32_t u; h2_t v; } ua, ub;
  ua.u = a; ub.u = 0xE408E408u;
  ua.v = ua.v + ub.v;
  return ua.u;
}

template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
  return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true));
}

// Sum over LPR consecutive lanes; result in lane LPR-1 of each group.
template <int LPR>
__device__ __forceinline__ float reduce_lpr(float v) {
  if constexpr (LPR >= 2) v = dpp_add<0x111>(v);
  if constexpr (LPR >= 4) v = dpp_add<0x112>(v);
  if constexpr (LPR >= 8) v = dpp_add<0x114>(v);
  if constexpr (LPR >= 16) v = dpp_add<0x118>(v);
  if constexpr (LPR >= 32) v += __shfl_xor(v, 16, 64);
  if constexpr (LPR >= 64) v += __shfl_xor(v, 32, 64);
  return v;
}

// one group of 32 weights (uint4 q) x 32 fp16 activations (4 uint4 of packed pairs)
__device__ __forceinline__ float group_dot_f16(const uint4& q, const uint4* x) {
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

// one group of 32 weights x 32 int8 activations (2 uint4, per-8 de-interleaved)
__device__ __forceinline__ int group_dot_i8(const uint4& q, const uint4& xa, const uint4& xb) {
  const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
  const int xd[8] = {(int)xa.x, (int)xa.y, (int)xa.z, (int)xa.w,
                     (int)xb.x, (int)xb.y, (int)xb.z, (int)xb.w};
  int sumi = 0;
#pragma unroll
  for (int wd = 0; wd < 4; ++wd) {
    sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
    sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
  }
  return sumi;
}

// per-8 permutation of the fp16 activation layout: k_j lands at slot 2j (j<4) / 2(j-4)+1
__device__ __forceinline__ int act_slot(int k) {
  const int j = k & 7;
  return (k & ~7) + (j < 4 ? 2 * j : 2 * (j - 4) + 1);
}

struct EpiArgs {
  uint16_t* out;       // EPI 0: bf16 [M, NT]
  uint16_t* xh_out;    // EPI 1: fp16 [M, R]  (permuted activation layout)
  uint16_t* inj;       // EPI 1: bf16 [M, HC]
  const uint16_t* xn;  // EPI 2: bf16 [M, HC*HD]
  int NT, R, HC, HD;
};

// Write one finished output element (row n of the weight, token m).
template <int EPI>
__device__ __forceinline__ void epi_store(const EpiArgs& e, int m, int n, float acc) {
  if constexpr (EPI == 0) {
    e.out[(long)m * e.NT + n] = f2bf(acc);
  } else if constexpr (EPI == 1) {
    if (n < e.R) {
      const float t = bf2f((uint32_t)f2bf(acc)) / (float)e.HC;
      const float v = t / (1.f + __expf(-t));
      union { _Float16 h; uint16_t u; } c;
      c.h = (_Float16)v;
      e.xh_out[(long)m * e.R + act_slot(n)] = c.u;
    } else if (n < e.R + e.HC) {
      e.inj[(long)m * e.HC + (n - e.R)] = f2bf(acc);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Slab kernel.  LPR lanes per output row, ROWS = 64/LPR rows per wave, lane l owns groups
// l + LPR*j.  MODE 0 = the token's activation preloaded into VGPRs, MODE 1 = staged in LDS once
// per 4-wave block (needs tiles_per_m % 4 == 0 so a block never straddles two tokens).
template <int K, int LPR, int WPB, int ACT, int MODE, int EPI>
__global__ void __launch_bounds__(64 * WPB) w4lt_slab(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s,
    const void* __restrict__ xact, const float* __restrict__ xs, const float* __restrict__ xsum,
    int M, int Gs, int gshift, EpiArgs e) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  static_assert(MODE == 0 || WPB == 4, "LDS mode needs 4 waves per block");
  __shared__ __align__(16) uint16_t xl[MODE == 1 ? K : 1];

  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles_per_m = e.NT / ROWS;
  int tile, m, n;
  if constexpr (MODE == 1) {
    const int tile0 = __builtin_amdgcn_readfirstlane(blockIdx.x * 4);
    m = tile0 / tiles_per_m;
    tile = tile0 + wv;
    n = ((tile0 + wv) % tiles_per_m) * ROWS + r;
  } else {
    tile = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + wv);
    if (tile >= M * tiles_per_m) return;
    m = tile / tiles_per_m;
    n = (tile % tiles_per_m) * ROWS + r;
  }

  const uint4* wp = reinterpret_cast<const uint4*>(w + (long)n * (K / 8)) + l;
  const uint16_t* sp = s + (long)n * Gs;
  uint4 q[GPL];
  uint32_t sc[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const int g = l + LPR * j;
    if (G % LPR == 0 || g < G) {
      // the weight is streamed once per token and never reused: non-temporal keeps it out
      // of L2, where only the (tiny, hot) activation belongs.
      q[j] = load_nt(wp + LPR * j);
      sc[j] = sp[g >> gshift];
    } else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }

  float acc = 0.f;
  if constexpr (ACT == 0 || ACT == 2) {
    const uint4* xg = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint16_t*>(xact) + (long)m * K);
    if constexpr (MODE == 1) {
      for (int i = threadIdx.x; i < K / 8; i += 256) {
        if constexpr (ACT == 2) {
          // bf16 -> fp16 + the per-8 permutation, done on the way into LDS: dword j of the
          // 8-block must be the packed fp16 pair (k_j, k_{j+4}).
          const uint4 v = xg[i];
          const uint32_t d[4] = {v.x, v.y, v.z, v.w};
          _Float16 h[8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            h[2 * j] = (_Float16)bf2f(d[j] & 0xFFFFu);
            h[2 * j + 1] = (_Float16)__uint_as_float(d[j] & 0xFFFF0000u);
          }
          uint32_t o[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            union { _Float16 hh; uint16_t u; } a, c;
            a.hh = h[j];
            c.hh = h[j + 4];
            o[j] = (uint32_t)a.u | ((uint32_t)c.u << 16);
          }
          reinterpret_cast<uint4*>(xl)[i] = make_uint4(o[0], o[1], o[2], o[3]);
        } else {
          reinterpret_cast<uint4*>(xl)[i] = xg[i];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < GPL; ++j) {
        const int g = l + LPR * j;
        if (G % LPR == 0 || g < G)
          acc = fmaf(bf2f(sc[j]), group_dot_f16(q[j], reinterpret_cast<const uint4*>(xl) + 4 * g),
                     acc);
      }
    } else {
      static_assert(ACT == 0, "ACT 2 (bf16 direct) requires the LDS staging mode");
      uint4 xv[GPL * 4];
#pragma unroll
      for (int j = 0; j < GPL; ++j) {
        const int g = l + LPR * j;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          xv[4 * j + i] = (G % LPR == 0 || g < G) ? xg[4 * g + i] : make_uint4(0, 0, 0, 0);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int j = 0; j < GPL; ++j) acc = fmaf(bf2f(sc[j]), group_dot_f16(q[j], &xv[4 * j]), acc);
    }
  } else {
    const uint4* xg = reinterpret_cast<const uint4*>(
        reinterpret_cast<const int8_t*>(xact) + (long)m * K);
    const float* xsp = xs + (long)m * G;
    const float* xsump = xsum + (long)m * G;
    uint4 xa[GPL], xb[GPL];
    float sx[GPL], sxs[GPL];
#pragma unroll
    for (int j = 0; j < GPL; ++j) {
      const int g = l + LPR * j;
      if (G % LPR == 0 || g < G) {
        xa[j] = xg[2 * g]; xb[j] = xg[2 * g + 1];
        sx[j] = xsp[g]; sxs[j] = xsump[g];
      } else { xa[j] = make_uint4(0, 0, 0, 0); xb[j] = xa[j]; sx[j] = 0.f; sxs[j] = 0.f; }
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int j = 0; j < GPL; ++j)
      acc += bf2f(sc[j]) * (sx[j] * (float)group_dot_i8(q[j], xa[j], xb[j]) - 8.f * sxs[j]);
  }

  acc = reduce_lpr<LPR>(acc);

  if constexpr (EPI == 2) {
    // ROWS == HC: rows n0..n0+HC-1 are the HC gates of output feature i = tile % tiles_per_m.
    // The per-row sums sit in lanes LPR-1, 2*LPR-1, ...; gather them into the last lane.
    float g0 = __shfl(acc, LPR - 1, 64);
    float g1 = __shfl(acc, 2 * LPR - 1, 64);
    float g2 = __shfl(acc, 3 * LPR - 1, 64);
    float g3 = __shfl(acc, 4 * LPR - 1, 64);
    if (lane == 63) {
      const int i = tile % tiles_per_m;
      const float gs4[4] = {g0, g1, g2, g3};
      float o = 0.f;
#pragma unroll
      for (int y = 0; y < 4; ++y) {
        const float gg = bf2f((uint32_t)f2bf(gs4[y]));
        o += (1.f / (1.f + __expf(-gg))) *
             bf2f(e.xn[(long)m * (e.HC * e.HD) + (long)y * e.HD + i]);
      }
      e.out[(long)m * e.HD + i] = f2bf(o / (float)e.HC);
    }
  } else {
    if (l == LPR - 1) epi_store<EPI>(e, m, n, acc);
  }
}

// ---------------------------------------------------------------------------------------------
// Multi-token slab: the same geometry, but a wave owns its output rows for ALL M tokens and
// loads the weight ONCE.  The whole [M, K] activation is staged in LDS (converted from bf16 and
// permuted on the way in, ACT 2), so the weight stream is M-independent -- which is the whole
// point at M = 2..8, where the per-token kernel above re-reads the weight M times.
// LDS = MMAX * K * 2 bytes, so MMAX is instantiated per K (8 for K <= 2560, less for mix_down).
template <int K, int LPR, int WPB, int MMAX, int EPI>
__global__ void __launch_bounds__(64 * WPB) w4lt_slab_m(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s,
    const uint16_t* __restrict__ xact, int M, int Gs, int gshift, EpiArgs e) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  __shared__ __align__(16) uint16_t xl[MMAX * K];
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles = e.NT / ROWS;
  const int tile = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + wv);
  const int n = (tile < tiles) ? tile * ROWS + r : 0;

  uint4 q[GPL];
  uint32_t sc[GPL];
  if (tile < tiles) {
    const uint4* wp = reinterpret_cast<const uint4*>(w + (long)n * (K / 8)) + l;
    const uint16_t* sp = s + (long)n * Gs;
#pragma unroll
    for (int j = 0; j < GPL; ++j) {
      const int g = l + LPR * j;
      if (G % LPR == 0 || g < G) { q[j] = load_nt(wp + LPR * j); sc[j] = sp[g >> gshift]; }
      else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
    }
  }
  // stage all M tokens: bf16 -> fp16 with the per-8 permutation
  const uint4* xg = reinterpret_cast<const uint4*>(xact);
  for (int i = threadIdx.x; i < M * (K / 8); i += 64 * WPB) {
    const uint4 v = xg[i];
    const uint32_t d[4] = {v.x, v.y, v.z, v.w};
    _Float16 h[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      h[2 * j] = (_Float16)bf2f(d[j] & 0xFFFFu);
      h[2 * j + 1] = (_Float16)__uint_as_float(d[j] & 0xFFFF0000u);
    }
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      union { _Float16 hh; uint16_t u; } a, c;
      a.hh = h[j];
      c.hh = h[j + 4];
      o[j] = (uint32_t)a.u | ((uint32_t)c.u << 16);
    }
    reinterpret_cast<uint4*>(xl)[i] = make_uint4(o[0], o[1], o[2], o[3]);
  }
  __syncthreads();
  if (tile >= tiles) return;

  for (int m = 0; m < M; ++m) {
    const uint4* xm = reinterpret_cast<const uint4*>(xl + (long)m * K);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < GPL; ++j) {
      const int g = l + LPR * j;
      if (G % LPR == 0 || g < G) acc = fmaf(bf2f(sc[j]), group_dot_f16(q[j], xm + 4 * g), acc);
    }
    acc = reduce_lpr<LPR>(acc);
    if constexpr (EPI == 2) {
      float g0 = __shfl(acc, LPR - 1, 64);
      float g1 = __shfl(acc, 2 * LPR - 1, 64);
      float g2 = __shfl(acc, 3 * LPR - 1, 64);
      float g3 = __shfl(acc, 4 * LPR - 1, 64);
      if (lane == 63) {
        const int i = tile;
        const float gs4[4] = {g0, g1, g2, g3};
        float o = 0.f;
#pragma unroll
        for (int y = 0; y < 4; ++y) {
          const float gg = bf2f((uint32_t)f2bf(gs4[y]));
          o += (1.f / (1.f + __expf(-gg))) *
               bf2f(e.xn[(long)m * (e.HC * e.HD) + (long)y * e.HD + i]);
        }
        e.out[(long)m * e.HD + i] = f2bf(o / (float)e.HC);
      }
    } else {
      if (l == LPR - 1) epi_store<EPI>(e, m, n, acc);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// W4 -> bf16 rematerialisation for M > W4LT_MAX_M (prefill / batched decode).
// out bf16 [rows, K]; row j of out is weight row r0 + j.  One thread per uint32 word (8 values).
__global__ void __launch_bounds__(256) w4lt_dequant_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, uint16_t* __restrict__ out,
    int r0, int rows, int Kw, int Gs, int gshift) {
  const long idx = (long)blockIdx.x * 256 + threadIdx.x;
  const long total = (long)rows * Kw;
  if (idx >= total) return;
  const int row = (int)(idx / Kw), wd = (int)(idx % Kw);
  const uint32_t q = w[(long)(r0 + row) * Kw + wd];
  const float sc = bf2f(s[(long)(r0 + row) * Gs + ((wd >> 2) >> gshift)]);
  uint16_t o[8];
#pragma unroll
  for (int t = 0; t < 8; ++t) o[t] = f2bf(sc * (float)(int)((q >> (4 * t)) & 0xFu) - 8.f * sc);
  uint16_t* dst = out + idx * 8;
#pragma unroll
  for (int t = 0; t < 8; ++t) dst[t] = o[t];
}

// ---------------------------------------------------------------------------------------------
// Host side
static void check_w(const at::Tensor& w, const at::Tensor& s, int N, int K, int gs) {
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous(), "w4lt: w/s must be contiguous");
  TORCH_CHECK(w.scalar_type() == at::kInt && w.dim() == 2 && w.size(0) == N && w.size(1) == K / 8,
              "w4lt: w must be int32 [N, K/8]");
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.dim() == 2 && s.size(0) == N &&
                  s.size(1) == K / gs,
              "w4lt: s must be bf16 [N, K/gs]");
}

static int gshift_of(int gs) {
  TORCH_CHECK(gs == 32 || gs == 64, "w4lt: group size must be 32 or 64, got ", gs);
  return gs == 64 ? 1 : 0;
}

template <int K, int LPR, int WPB, int ACT, int MODE, int EPI>
static void launch_slab(const at::Tensor& w, const at::Tensor& s, const void* xp, const float* xsp,
                        const float* xsump, int M, int Gs, int gshift, const EpiArgs& e) {
  constexpr int ROWS = 64 / LPR;
  const int tiles_per_m = e.NT / ROWS;
  TORCH_CHECK(e.NT % ROWS == 0, "w4lt: NT must be a multiple of ", ROWS);
  const long tiles = (long)M * tiles_per_m;
  if (tiles <= 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(w.data_ptr());
  const uint16_t* sp = reinterpret_cast<const uint16_t*>(s.data_ptr());
  if constexpr (MODE == 1) {
    TORCH_CHECK(tiles_per_m % 4 == 0, "w4lt: LDS mode needs (NT/ROWS) % 4 == 0");
    w4lt_slab<K, LPR, WPB, ACT, MODE, EPI><<<(tiles + 3) / 4, 256, 0, st>>>(
        wp, sp, xp, xsp, xsump, M, Gs, gshift, e);
  } else {
    w4lt_slab<K, LPR, WPB, ACT, MODE, EPI><<<(tiles + WPB - 1) / WPB, 64 * WPB, 0, st>>>(
        wp, sp, xp, xsp, xsump, M, Gs, gshift, e);
  }
}

// ---- EPI 0: plain dense GEMV, bf16 out --------------------------------------------------------
// f16 activations.  w [N, K/8] int32, s [N, K/gs] bf16, xh fp16 [M, K] (permuted), out bf16 [M, N].
void w4lt_gemv_f16(at::Tensor w, at::Tensor s, at::Tensor xh, at::Tensor out, int64_t gs,
                   int64_t lpr, int64_t mode, int64_t wpb) {
  const int M = xh.size(0), K = xh.size(1), N = out.size(1);
  TORCH_CHECK(xh.scalar_type() == at::kHalf && xh.is_contiguous(), "w4lt: xh must be fp16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M);
  check_w(w, s, N, K, (int)gs);
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  e.NT = N;
  const void* xp = xh.data_ptr();
#define C0(KK, L, W, MO)                                                                   \
  if (K == KK && lpr == L && wpb == W && mode == MO) {                                     \
    launch_slab<KK, L, W, 0, MO, 0>(w, s, xp, nullptr, nullptr, M, K / (int)gs, gsh, e);   \
    return;                                                                                \
  }
  C0(2560, 16, 4, 1) C0(2560, 16, 4, 0) C0(2560, 16, 1, 0)
  C0(1536, 16, 4, 1) C0(1536, 16, 4, 0) C0(1536, 16, 1, 0)
#undef C0
  TORCH_CHECK(false, "w4lt_gemv_f16: no instantiation for K/lpr/mode/wpb ", K, "/", lpr, "/", mode,
              "/", wpb);
}

// bf16 activations, cast + permuted inside the LDS stage: NO prep launch.  x bf16 [M, K].
void w4lt_gemv_bf16(at::Tensor w, at::Tensor s, at::Tensor x, at::Tensor out, int64_t gs,
                    int64_t lpr) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "w4lt: x must be bf16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M);
  check_w(w, s, N, K, (int)gs);
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  e.NT = N;
  const void* xp = x.data_ptr();
#define C2(KK, L)                                                                       \
  if (K == KK && lpr == L) {                                                            \
    launch_slab<KK, L, 4, 2, 1, 0>(w, s, xp, nullptr, nullptr, M, K / (int)gs, gsh, e);  \
    return;                                                                             \
  }
  C2(2560, 16) C2(2560, 32) C2(2560, 64)
  C2(1536, 16) C2(1536, 32) C2(1536, 64)
#undef C2
  TORCH_CHECK(false, "w4lt_gemv_bf16: no instantiation for K/lpr ", K, "/", lpr);
}

// int8 activations (Q8_1).  x8 int8 [M, K] + xs/xsum fp32 [M, K/32].
void w4lt_gemv_i8(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
                  at::Tensor out, int64_t gs, int64_t lpr, int64_t mode, int64_t wpb) {
  const int M = x8.size(0), K = x8.size(1), N = out.size(1);
  TORCH_CHECK(x8.scalar_type() == at::kChar && x8.is_contiguous(), "w4lt: x8 must be int8");
  TORCH_CHECK(xs.scalar_type() == at::kFloat && xsum.scalar_type() == at::kFloat);
  TORCH_CHECK(xs.is_contiguous() && xsum.is_contiguous() && xs.size(1) == K / 32);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M);
  check_w(w, s, N, K, (int)gs);
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  e.NT = N;
  const void* xp = x8.data_ptr();
  const float* a = xs.data_ptr<float>();
  const float* b = xsum.data_ptr<float>();
#define C1(KK, L, W, MO)                                                              \
  if (K == KK && lpr == L && wpb == W && mode == MO) {                                \
    launch_slab<KK, L, W, 1, MO, 0>(w, s, xp, a, b, M, K / (int)gs, gsh, e);          \
    return;                                                                           \
  }
  C1(2560, 16, 4, 1) C1(2560, 16, 4, 0) C1(2560, 16, 1, 0)
  C1(1536, 16, 4, 1) C1(1536, 16, 4, 0) C1(1536, 16, 1, 0)
#undef C1
  TORCH_CHECK(false, "w4lt_gemv_i8: no instantiation for K/lpr/mode/wpb ", K, "/", lpr, "/", mode,
              "/", wpb);
}

// ---- EPI 1: hyper-connection mix_down + silu, emitting the next GEMV's activation -------------
// w [Mc, K/8] (Mc = R + HC + pad), xh fp16 [M, K] permuted, xh_out fp16 [M, R] permuted,
// inj bf16 [M, HC].
void w4lt_hc_down(at::Tensor w, at::Tensor s, at::Tensor xh, at::Tensor xh_out, at::Tensor inj,
                  int64_t gs, int64_t hc, int64_t r, int64_t lpr, int64_t mode, int64_t wpb,
                  int64_t act) {
  const int M = xh.size(0), K = xh.size(1);
  TORCH_CHECK(act == 0 || act == 2, "w4lt_hc_down: act must be 0 (fp16 in) or 2 (bf16 in)");
  TORCH_CHECK(xh.scalar_type() == (act == 2 ? at::kBFloat16 : at::kHalf) && xh.is_contiguous());
  TORCH_CHECK(xh_out.scalar_type() == at::kHalf && xh_out.is_contiguous());
  TORCH_CHECK(inj.scalar_type() == at::kBFloat16 && inj.is_contiguous());
  TORCH_CHECK(xh_out.size(0) == M && xh_out.size(1) == r && inj.size(0) == M && inj.size(1) == hc);
  const int N = w.size(0);
  check_w(w, s, N, K, (int)gs);
  const int NT = (int)(r + hc);
  TORCH_CHECK(NT <= N, "w4lt_hc_down: R + HC exceeds the weight rows");
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.xh_out = reinterpret_cast<uint16_t*>(xh_out.data_ptr());
  e.inj = reinterpret_cast<uint16_t*>(inj.data_ptr());
  e.NT = NT;
  e.R = (int)r;
  e.HC = (int)hc;
  const void* xp = xh.data_ptr();
#define CD(KK, L, W, MO)                                                              \
  if (K == KK && lpr == L && wpb == W && mode == MO) {                                \
    launch_slab<KK, L, W, 0, MO, 1>(w, s, xp, nullptr, nullptr, M, K / (int)gs, gsh, e); \
    return;                                                                           \
  }
  if (act == 0) { CD(10240, 64, 4, 1) CD(10240, 64, 4, 0) CD(10240, 64, 1, 0) }
#undef CD
#define CB(KK, L, W, MO)                                                                 \
  if (K == KK && lpr == L && wpb == W && mode == MO) {                                   \
    launch_slab<KK, L, W, 2, MO, 1>(w, s, xp, nullptr, nullptr, M, K / (int)gs, gsh, e);  \
    return;                                                                              \
  }
  if (act == 2) { CB(10240, 64, 4, 1) }
#undef CB
  TORCH_CHECK(false, "w4lt_hc_down: no instantiation for K/lpr/mode/wpb ", K, "/", lpr, "/", mode,
              "/", wpb);
}

// ---- EPI 2: hyper-connection mix_up + gate-mix ------------------------------------------------
// w [HC*HD, K/8] with rows permuted (row i*HC + s = original row s*HD + i),
// xh fp16 [M, K=R] permuted, xn bf16 [M, HC*HD], y bf16 [M, HD].
void w4lt_hc_up(at::Tensor w, at::Tensor s, at::Tensor xh, at::Tensor xn, at::Tensor y,
                int64_t gs, int64_t hc, int64_t lpr, int64_t mode, int64_t wpb) {
  const int M = xh.size(0), K = xh.size(1), HD = y.size(1), N = w.size(0);
  TORCH_CHECK(xh.scalar_type() == at::kHalf && xh.is_contiguous());
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous());
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.size(0) == M);
  TORCH_CHECK(hc == 4, "w4lt_hc_up: the gate-mix epilogue is instantiated for HC = 4");
  TORCH_CHECK(N == hc * HD, "w4lt_hc_up: weight rows must be HC*HD");
  check_w(w, s, N, K, (int)gs);
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.out = reinterpret_cast<uint16_t*>(y.data_ptr());
  e.xn = reinterpret_cast<const uint16_t*>(xn.data_ptr());
  e.NT = N;
  e.HC = (int)hc;
  e.HD = HD;
  const void* xp = xh.data_ptr();
#define CU(KK, L, W, MO)                                                                 \
  if (K == KK && lpr == L && wpb == W && mode == MO) {                                   \
    launch_slab<KK, L, W, 0, MO, 2>(w, s, xp, nullptr, nullptr, M, K / (int)gs, gsh, e); \
    return;                                                                              \
  }
  CU(320, 16, 4, 1) CU(320, 16, 4, 0) CU(320, 16, 1, 0)
#undef CU
  TORCH_CHECK(false, "w4lt_hc_up: no instantiation for K/lpr/mode/wpb ", K, "/", lpr, "/", mode,
              "/", wpb);
}

template <int K, int LPR, int WPB, int MMAX, int EPI>
static void launch_slab_m(const at::Tensor& w, const at::Tensor& s, const uint16_t* xp, int M,
                          int Gs, int gshift, const EpiArgs& e) {
  constexpr int ROWS = 64 / LPR;
  TORCH_CHECK(M <= MMAX, "w4lt slab_m: M ", M, " exceeds MMAX ", MMAX);
  TORCH_CHECK(e.NT % ROWS == 0, "w4lt: NT must be a multiple of ", ROWS);
  const int tiles = e.NT / ROWS;
  auto st = c10::hip::getCurrentHIPStream().stream();
  w4lt_slab_m<K, LPR, WPB, MMAX, EPI><<<(tiles + WPB - 1) / WPB, 64 * WPB, 0, st>>>(
      reinterpret_cast<const uint32_t*>(w.data_ptr()),
      reinterpret_cast<const uint16_t*>(s.data_ptr()), xp, M, Gs, gshift, e);
}

// Multi-token dense GEMV: bf16 [M, K] in (natural order), bf16 [M, N] out, one launch, and the
// weight is read once for all M.
void w4lt_gemv_m(at::Tensor w, at::Tensor s, at::Tensor x, at::Tensor out, int64_t gs,
                 int64_t lpr, int64_t mmax) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M);
  check_w(w, s, N, K, (int)gs);
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  e.NT = N;
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
#define CM(KK, L, MM)                                                              \
  if (K == KK && lpr == L && mmax == MM) {                                         \
    launch_slab_m<KK, L, 4, MM, 0>(w, s, xp, M, K / (int)gs, gsh, e); return;       \
  }
  CM(2560, 16, 2) CM(2560, 16, 4) CM(2560, 16, 8)
  CM(2560, 32, 4) CM(2560, 32, 8)
  CM(1536, 16, 2) CM(1536, 16, 4) CM(1536, 16, 8)
  CM(1536, 32, 4) CM(1536, 32, 8)
#undef CM
  TORCH_CHECK(false, "w4lt_gemv_m: no instantiation for K/lpr/mmax ", K, "/", lpr, "/", mmax);
}

// Multi-token mix_up + gate-mix (K = 320, so MMAX = 8 costs only 5 KB of LDS).
void w4lt_hc_up_m(at::Tensor w, at::Tensor s, at::Tensor x, at::Tensor xn, at::Tensor y,
                  int64_t gs, int64_t hc, int64_t mmax) {
  const int M = x.size(0), K = x.size(1), HD = y.size(1), N = w.size(0);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous());
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.size(0) == M);
  TORCH_CHECK(hc == 4 && N == hc * HD);
  check_w(w, s, N, K, (int)gs);
  const int gsh = gshift_of((int)gs);
  EpiArgs e{};
  e.out = reinterpret_cast<uint16_t*>(y.data_ptr());
  e.xn = reinterpret_cast<const uint16_t*>(xn.data_ptr());
  e.NT = N;
  e.HC = (int)hc;
  e.HD = HD;
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
#define CMU(MM)                                                                    \
  if (K == 320 && mmax == MM) {                                                    \
    launch_slab_m<320, 16, 4, MM, 2>(w, s, xp, M, K / (int)gs, gsh, e); return;    \
  }
  CMU(2) CMU(4) CMU(8)
#undef CMU
  TORCH_CHECK(false, "w4lt_hc_up_m: no instantiation for K/mmax ", K, "/", mmax);
}

// ---- dequant ---------------------------------------------------------------------------------
// out bf16 [rows, K] <- weight rows [r0, r0 + rows) of the W4 pair (w, s).  Generic over K.
void w4lt_dequant(at::Tensor w, at::Tensor s, at::Tensor out, int64_t r0, int64_t gs) {
  TORCH_CHECK(w.scalar_type() == at::kInt && w.is_contiguous() && w.dim() == 2);
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.is_contiguous() && s.dim() == 2);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 2);
  const int Kw = w.size(1), K = Kw * 8, rows = out.size(0);
  TORCH_CHECK(out.size(1) == K, "w4lt_dequant: out K mismatch");
  TORCH_CHECK(r0 >= 0 && r0 + rows <= w.size(0), "w4lt_dequant: row range out of bounds");
  const int gsh = gshift_of((int)gs);
  TORCH_CHECK(s.size(1) == K / (int)gs, "w4lt_dequant: scale K/gs mismatch");
  const long total = (long)rows * Kw;
  if (total <= 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  w4lt_dequant_kernel<<<(total + 255) / 256, 256, 0, st>>>(
      reinterpret_cast<const uint32_t*>(w.data_ptr()),
      reinterpret_cast<const uint16_t*>(s.data_ptr()),
      reinterpret_cast<uint16_t*>(out.data_ptr()), (int)r0, rows, Kw, K / (int)gs, gsh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4lt_gemv_f16", &w4lt_gemv_f16,
        "w [N,K/8] i32, s [N,K/gs] bf16, xh fp16 [M,K] permuted -> out bf16 [M,N]");
  m.def("w4lt_gemv_bf16", &w4lt_gemv_bf16,
        "w, s, x bf16 [M,K] natural order (cast+permute folded into LDS staging), lpr -> bf16");
  m.def("w4lt_gemv_i8", &w4lt_gemv_i8,
        "w, s, x8 int8 [M,K] + xs/xsum fp32 [M,K/32] -> out bf16 [M,N]");
  m.def("w4lt_hc_down", &w4lt_hc_down,
        "mix_down + silu (act 0 = fp16 in, 2 = bf16 in): -> xh_out fp16 [M,R], inj bf16 [M,HC]");
  m.def("w4lt_hc_up", &w4lt_hc_up, "mix_up + gate-mix (permuted rows) -> y bf16 [M,HD]");
  m.def("w4lt_gemv_m", &w4lt_gemv_m,
        "multi-token dense GEMV: w, s, x bf16 [M,K], out bf16 [M,N], gs, lpr, mmax");
  m.def("w4lt_hc_up_m", &w4lt_hc_up_m,
        "multi-token mix_up + gate-mix: w, s, x bf16 [M,320], xn bf16, y bf16 [M,HD], gs, hc, mmax");
  m.def("w4lt_dequant", &w4lt_dequant, "out bf16 [rows,K] <- W4 rows [r0, r0+rows)");
}
