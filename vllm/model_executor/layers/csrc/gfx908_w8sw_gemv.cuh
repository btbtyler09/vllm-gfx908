// SPDX-License-Identifier: Apache-2.0
// gfx908 / MI100 W8A16 skinny GEMV that reads the **MFMA-swizzled** int8 layout directly.
//
// Motivation: with VLLM_GFX908_W8A16_MFMA=1 the swizzled int8 tensor is the only resident
// copy of the weight, so M <= 4 (decode) currently has to go through the MFMA GEMM with the
// M-tile padded to 16 rows.  This kernel serves M <= 4 from the same bytes with a pure GEMV,
// so the MFMA flag costs nothing at c = 1.
//
//   C[M, N] (bf16) = X[M, K] (bf16) @ dequant(Wq[N, K] int8, S[N, K/gs] fp32)^T,  1 <= M <= 4
//
// Layout (identical to swizzle_w8_mfma(tile=16) in gfx908_w8a16.py):
//   Wsw[nt][kt][h][n_in][k_in]   nt = N/16, kt = K/64, h = 0..3, n_in = 0..15, k_in = 0..15
//   byte offset = nt*(16*K) + kt*1024 + h*256 + n_in*16 + k_in
//     -> n = nt*16 + n_in ,  k = kt*64 + h*16 + k_in
//   Ssw[K/gs][N] fp32 (= S.t().contiguous(), swizzle_s_mfma)
//
// Mapping: one wave64 owns NT consecutive n-tiles (16*NT output rows).  Lane l owns
// n_in = l & 15 and the 16-k run h = l >> 4, i.e. **exactly the 16 B that lane l reads in the
// MFMA kernel**, so a k-tile is consumed by one fully coalesced 1 KiB
// `global_load_dwordx4 glc slc` per wave per n-tile -- the same linear stream the MFMA kernel
// measures at 847 GB/s, with no cross-lane shuffles in the loop.
//
// Because gs >= 64 and a k-tile is 64 k wide, all four h slots of a k-tile share one scale
// group (g = (kt*64) >> gshift), so the scale is one broadcast dword per lane per k-tile.
//
// Reduction: the four h slots of an output are summed at the *end* of the row-group with two
// `ds_bpermute_b32` (lane ^ 16, lane ^ 32) -- O(1) per output, not per k-tile.  Optional
// intra-WG split-K (KS waves over disjoint k-slices of the same n-tiles) reduces through LDS
// in a fixed wave order, so the result is deterministic across cudagraph replays.
//
// Activations are staged into LDS as fp32 exactly as in gfx908_w8a16_kernel.cuh (same global
// load order / LDS barrier trick), so the fixed per-launch cost matches the row-major GEMV.
//
// Accumulation order (see REPORT.md "Numerics"): per output,
//   sum_h = sum_{kt in slice} ( sum_{i=0..15} Wq[n, kt*64+h*16+i] * x[...] ) * S[n, g(kt)]
//   out   = ((sum_0 + sum_1) + (sum_2 + sum_3))  [+ split-K partials in wave order]
// The row-major GEMV instead gives lane kl the chunks {kl*16 + 512*j} and reduces with a DPP
// tree, so the two differ in fp32 summation order only (both are exact-product fp32 FMA
// chains over the same 2^7-bounded terms).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define WSW_LDS_BYTES (64 * 1024)
#define WSW_RED_BYTES (8 * 1024)  // 16 waves x 16*NB*NT partials x 4 B (NB*NT <= 8)
#define WSW_X_BYTES (WSW_LDS_BYTES - WSW_RED_BYTES)  // 57344 B of X
#define WSW_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

typedef uint32_t wsw_u32x4 __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ T wsw_loadnt(const T* p) { return __builtin_nontemporal_load(p); }

__device__ __forceinline__ float wsw_bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float wsw_bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float wsw_i8f(uint32_t u, int b) { return (float)(int)(int8_t)(u >> (8 * b)); }

// sum over the 4 h-slots of a lane group: lanes {l, l^16, l^32, l^48}; order ((h0+h1)+(h2+h3))
__device__ __forceinline__ float wsw_h_reduce(float v, int lane) {
  v += __int_as_float(__builtin_amdgcn_ds_bpermute((lane ^ 16) << 2, __float_as_int(v)));
  v += __int_as_float(__builtin_amdgcn_ds_bpermute((lane ^ 32) << 2, __float_as_int(v)));
  return v;
}

// NT   : n-tiles (16 rows each) per wave
// UNRL : k-tiles prefetched in flight
// NB   : batch M (1..4), template
// F32LDS : stage X as fp32 (else bf16)
// KS   : intra-WG split-K waves per row group
template <int NT, int UNRL, int NB, bool F32LDS, int KS>
__global__ void __launch_bounds__(1024)
w8sw_gemv_kernel(const int K, const int N, const int8_t* __restrict__ W,
                 const float* __restrict__ S, const uint16_t* __restrict__ X,
                 uint16_t* __restrict__ C, const int wvRG, const int rounds,
                 const int NKT, const int ntiles, const int gshift) {
  constexpr int LDS_BF16 = WSW_X_BYTES / 2;
  constexpr int RSTRIDE = 16 * NB * NT;  // floats per wave in the split-K buffer
  static_assert(KS == 1 || RSTRIDE * 16 * 4 <= WSW_RED_BYTES, "split-K partial buffer");
  __shared__ __align__(16) unsigned char lds_raw[WSW_LDS_BYTES];
  float* sf = reinterpret_cast<float*>(lds_raw);
  uint16_t* sh = reinterpret_cast<uint16_t*>(lds_raw);
  float* red = reinterpret_cast<float*>(lds_raw + WSW_X_BYTES);

  const int lane = threadIdx.x, wave = threadIdx.y;
  const int tid = wave * 64 + lane;
  const int nin = lane & 15, hh = lane >> 4;
  const int rg = wave / KS, ks = wave % KS;
  const int kslice = NKT / KS;
  const int kt0 = ks * kslice;
  const long tstride = (long)NKT * 1024;  // bytes per n-tile
  const int total = NB * K;

  // Row-group slot -> n-tile.  rg is the OUTER index so that a partial pass spreads over all
  // the CUs (blockIdx-major would leave whole workgroups with no tile at all: e.g. 160 n-tiles
  // over 120 WGs x RG=2 gives tiles 0..159 to WGs 0..79 and nothing to WGs 80..119).
  int ntb = (rg * gridDim.x + blockIdx.x) * NT;
  const int ntstep = gridDim.x * wvRG * NT;

  const int8_t* wp[NT];
  const float* sp[NT];
  auto setptr = [&]() {
#pragma unroll
    for (int y = 0; y < NT; y++) {
      const int nt = min(ntb + y, ntiles - 1);
      wp[y] = W + (long)nt * tstride + (long)kt0 * 1024 + hh * 256 + nin * 16;
      sp[y] = S + nt * 16 + nin;
    }
  };

  wsw_u32x4 bw[NT][UNRL];
  float sc[NT][UNRL];
  auto issue = [&](int kk) {
#pragma unroll
    for (int u = 0; u < UNRL; u++) {
      const int ktl = min(kk + u, kslice - 1);
      const int g = ((kt0 + ktl) * 64) >> gshift;
#pragma unroll
      for (int y = 0; y < NT; y++) {
        bw[y][u] = wsw_loadnt(reinterpret_cast<const wsw_u32x4*>(wp[y] + (long)ktl * 1024));
        sc[y][u] = sp[y][(long)g * N];
      }
    }
  };

  setptr();

  // ---- X staging.  Global X loads are issued first, then the first weight loads, then the
  // LDS stores: vmcnt is in-order on gfx9, so consuming X leaves the HBM weight loads in
  // flight across the staging + (LDS-only) barrier.
  {
    const int lim = F32LDS ? total : (total < LDS_BF16 ? total : LDS_BF16);
    wsw_u32x4 v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int k = tid * 8 + i * 8192;
      if (k < lim) v[i] = *reinterpret_cast<const wsw_u32x4*>(X + k);
    }
    __builtin_amdgcn_sched_barrier(0);
    issue(0);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int k = tid * 8 + i * 8192;
      if (k < lim) {
        if constexpr (F32LDS) {
          float4 a = make_float4(wsw_bf16lo(v[i].x), wsw_bf16hi(v[i].x), wsw_bf16lo(v[i].y), wsw_bf16hi(v[i].y));
          float4 b = make_float4(wsw_bf16lo(v[i].z), wsw_bf16hi(v[i].z), wsw_bf16lo(v[i].w), wsw_bf16hi(v[i].w));
          *reinterpret_cast<float4*>(sf + k) = a;
          *reinterpret_cast<float4*>(sf + k + 4) = b;
        } else {
          *reinterpret_cast<wsw_u32x4*>(sh + k) = v[i];
        }
      }
    }
  }
  __builtin_amdgcn_sched_barrier(0);
  WSW_LDS_BARRIER();
  __builtin_amdgcn_sched_barrier(0);
  if constexpr (KS == 1) { if (rg >= wvRG) return; }

  for (int r = 0; r < rounds; r++) {
    const bool act = (rg < wvRG) && (ntb < ntiles);  // wave-uniform
    float sum[NB][NT];
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < NT; y++) sum[n][y] = 0.f;

    if (act) {
      if (r > 0) issue(0);
      for (int kk = 0; kk < kslice; kk += UNRL) {
#pragma unroll
        for (int u = 0; u < UNRL; u++) {
          const int ktl = kk + u;
          if (ktl < kslice) {
            const int kbase = (kt0 + ktl) * 64 + hh * 16;
            float t[NB][NT];
#pragma unroll
            for (int n = 0; n < NB; n++)
#pragma unroll
              for (int y = 0; y < NT; y++) t[n][y] = 0.f;
            constexpr int XW = F32LDS ? 4 : 8, XD = XW / 4;
#pragma unroll
            for (int jj = 0; jj < 16 / XW; jj++) {
              float wf[NT][XW];
#pragma unroll
              for (int y = 0; y < NT; y++)
#pragma unroll
                for (int d = 0; d < XD; d++) {
                  const uint32_t dw = bw[y][u][XD * jj + d];
#pragma unroll
                  for (int b = 0; b < 4; b++) wf[y][4 * d + b] = wsw_i8f(dw, b);
                }
#pragma unroll
              for (int n = 0; n < NB; n++) {
                float xv[XW];
                if constexpr (F32LDS) {
                  const float4 a = *reinterpret_cast<const float4*>(sf + n * K + kbase + XW * jj);
                  xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
                } else {
                  const int idx = n * K + kbase + XW * jj;
                  const wsw_u32x4 u2 = (idx < LDS_BF16) ? *reinterpret_cast<const wsw_u32x4*>(sh + idx)
                                                        : *reinterpret_cast<const wsw_u32x4*>(X + idx);
                  xv[0] = wsw_bf16lo(u2.x); xv[1] = wsw_bf16hi(u2.x);
                  xv[2] = wsw_bf16lo(u2.y); xv[3] = wsw_bf16hi(u2.y);
                  xv[4] = wsw_bf16lo(u2.z); xv[5] = wsw_bf16hi(u2.z);
                  xv[6] = wsw_bf16lo(u2.w); xv[7] = wsw_bf16hi(u2.w);
                }
#pragma unroll
                for (int y = 0; y < NT; y++)
#pragma unroll
                  for (int i = 0; i < XW; i++) t[n][y] = __builtin_fmaf(wf[y][i], xv[i], t[n][y]);
              }
            }
#pragma unroll
            for (int n = 0; n < NB; n++)
#pragma unroll
              for (int y = 0; y < NT; y++) sum[n][y] = __builtin_fmaf(t[n][y], sc[y][u], sum[n][y]);
          }
        }
        const int knext = kk + UNRL;
        if (knext < kslice) issue(knext);
      }
    }
    __builtin_amdgcn_sched_barrier(0);

#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < NT; y++) sum[n][y] = wsw_h_reduce(sum[n][y], lane);

    if constexpr (KS == 1) {
      if (hh == 0 && act) {
#pragma unroll
        for (int y = 0; y < NT; y++) {
          const int nt = ntb + y;
          if (nt < ntiles) {
#pragma unroll
            for (int n = 0; n < NB; n++) {
              __hip_bfloat16 o = __float2bfloat16(sum[n][y]);
              C[(size_t)n * N + nt * 16 + nin] = *reinterpret_cast<uint16_t*>(&o);
            }
          }
        }
      }
    } else {
      if (hh == 0) {
#pragma unroll
        for (int n = 0; n < NB; n++)
#pragma unroll
          for (int y = 0; y < NT; y++) red[wave * RSTRIDE + (n * NT + y) * 16 + nin] = sum[n][y];
      }
      __syncthreads();
      if (ks == 0 && act) {
        for (int i = lane; i < RSTRIDE; i += 64) {
          float v = 0.f;
#pragma unroll
          for (int q = 0; q < KS; q++) v += red[(rg * KS + q) * RSTRIDE + i];
          const int nn = i & 15, y = (i >> 4) % NT, n = i / (16 * NT);
          const int nt = ntb + y;
          if (nt < ntiles) {
            __hip_bfloat16 o = __float2bfloat16(v);
            C[(size_t)n * N + nt * 16 + nn] = *reinterpret_cast<uint16_t*>(&o);
          }
        }
      }
      __syncthreads();
    }
    ntb += ntstep;
    setptr();
  }
}
