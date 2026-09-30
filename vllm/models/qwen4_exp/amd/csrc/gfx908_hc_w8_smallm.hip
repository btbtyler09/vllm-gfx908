// SPDX-License-Identifier: Apache-2.0
// gfx908 small-M (MTP verify, M = 2..4) W8 hyper-connection mixes, ONE-ROUND-TRIP layout.
//
// Separate extension on purpose (gfx908_hc_w8_smallm_ext): code-object size is a per-dispatch cost
// on gfx908, so none of these kernels are linked into the hot M = 1 module (gfx908_wv_fused_w8_ext).
// Loaded only under VLLM_GFX908_SMALLM_HC_W8=1.  Design notes: docs/mi100_decode_opt/
// mtp_smallm_gemv_gfx908.md.
//
// Why the shipping hc_w8_gemv_kernel scales badly in M (the "per extra row" cost):
//   * X is staged into LDS by every one of the 120 workgroups (including the ones with no rows),
//     through a __syncthreads, BEFORE the first weight load is issued -> the X round trip and the
//     weight round trip are serialised, and the staged bytes grow with M (120 x M x 20 KB of L2
//     reads for mix_down: 2.4 MB at M = 1, 4.9 MB at M = 2).
//   * mix_down (336 rows) keeps only 4 waves busy per CU (1 per SIMD on 84 CUs); each wave runs
//     3 dependent load -> compute iterations, so the per-row FMA work sits on the critical path
//     with nothing to hide it.  At M = 2 the fp32-LDS mode also flips to bf16 (NB*K*4 > 62 KB).
//   * mix_up (K = 320) runs 20 of 32 lanes per half-wave (62.5 %), and the gate-mix epilogue's XN
//     loads are issued after the reduction (sched_barrier), exposing one more L2 round trip.
//   * M >= 2 also pays a separate `.contiguous()` launch for the injection columns (the [M, 4]
//     slice of the [M, 336] down output is no longer contiguous), 98 launches per step at K=1.
//
// What these kernels do instead (both): every weight / scale / activation load a lane will ever need
// is issued up front, before any LDS staging or barrier, so the whole kernel is ONE global round
// trip plus a short tail; only workgroups with rows are launched; all lanes do useful work.
//
// hcsm_down_kernel<NB, YT, SPW>   mix_down [N, K] (336 x 10240), EPI "silu on lora rows"
//   WG = YT output rows x the full K; wave ks of KS owns 1024-k steps ks, ks+KS, ... (SPW of them),
//   lane l owns the 16 k at step*1024 + 16*l for all YT rows (so one X read feeds YT rows).
//   Activation is read straight from global into registers (no LDS, no barrier before compute).
//   Reduction: DPP inside the wave, then the KS wave partials through LDS in fixed order.
//   The injection columns [R, R + HC) are ALSO written to a compact [M, HC] output (no copy launch).
//
// hcsm_up_kernel<NB>              mix_up over the row-permuted weight [HC*HD, K] (10240 x 320),
//                                 gate-mix epilogue: Y[n, i] = mean_s sigmoid(bf16(g_s)) * XN[n, s*HD+i]
//   A "unit" is (hidden index i, 16-k chunk c): the lane loads that chunk for all HC = 4 permuted rows
//   4i..4i+3 (one X read feeds 4 rows).  A WG holds IPB indices x CPR chunks with IPB*CPR a multiple
//   of 64, so every lane is busy (K = 320: CPR = 20, IPB = 16, 320 threads = 5 waves, 100 % lanes).
//   Partials go to LDS as red[unit][n][s]; thread (i, n, s) sums its CPR chunks in order, the quad
//   (s = 0..3) is gathered with DPP and lane s = 0 applies the gate-mix in the shipping order.
//
// Numerics: same per-lane math as hc_w8_gemv_kernel (exact int8 x bf16 products, fp32 FMA chain over
// the lane's 16 k in order, one fma with the fp32 group scale per chunk), same epilogue expressions.
// Only the order in which the per-chunk partials are summed differs (a different reduction tree), so
// results match the shipping kernel within fp32 accumulation rounding (occasional 1-ulp bf16 flips),
// not bit-for-bit.  tests/gfx908/test_hc_w8_smallm_ref.py emulates both orders on CPU.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace {

typedef uint32_t hs_u32x4 __attribute__((ext_vector_type(4)));

constexpr int HS_HC = 4;            // hyper-connection streams (the permuted mix_up layout needs 4)

__device__ __forceinline__ float hs_bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hs_bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float hs_i8f(uint32_t u, int b) { return (float)(int)(int8_t)(u >> (8 * b)); }
__device__ __forceinline__ float hs_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float hs_bf16u(uint16_t r) { return __uint_as_float((uint32_t)r << 16); }
__device__ __forceinline__ uint16_t hs_tobf16(float v) {
  __hip_bfloat16 o = __float2bfloat16(v);
  return *reinterpret_cast<uint16_t*>(&o);
}

// Full wave64 reduction, identical DPP tree to hw8_wave_reduce<64>: lane 63 holds the total.
__device__ __forceinline__ float hs_wave_reduce(float v) {
  v += __builtin_amdgcn_mov_dpp(v, 0x118, 0xf, 0xf, 1);  // row_shr8
  v += __builtin_amdgcn_mov_dpp(v, 0x114, 0xf, 0xf, 1);  // row_shr4
  v += __builtin_amdgcn_mov_dpp(v, 0x112, 0xf, 0xf, 1);  // row_shr2
  v += __builtin_amdgcn_mov_dpp(v, 0x111, 0xf, 0xf, 1);  // row_shr1
  v += __builtin_amdgcn_mov_dpp(v, 0x142, 0xf, 0xf, 1);  // row_bcast15
  v += __builtin_amdgcn_mov_dpp(v, 0x143, 0xf, 0xf, 1);  // row_bcast31
  return v;
}

// 16 int8 weights x 16 bf16 activations for NB rows, fp32 fma chain in k order (0..15), done in
// 4-k sub-chunks so the live temporaries stay at 4*(YT + NB) VGPRs.
template <int NB, int YT>
__device__ __forceinline__ void hs_dot16(const hs_u32x4 (&w)[YT], const hs_u32x4 (&x)[NB][2],
                                         float (&t)[NB][YT]) {
#pragma unroll
  for (int n = 0; n < NB; n++)
#pragma unroll
    for (int y = 0; y < YT; y++) t[n][y] = 0.f;
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    float wf[YT][4];
#pragma unroll
    for (int y = 0; y < YT; y++)
#pragma unroll
      for (int b = 0; b < 4; b++) wf[y][b] = hs_i8f(w[y][jj], b);
#pragma unroll
    for (int n = 0; n < NB; n++) {
      // bf16 pair dwords: k = 4jj..4jj+3 live in dwords 2jj, 2jj+1 of the 16 k (x[n][0] = k 0..7)
      const uint32_t d0 = x[n][jj >> 1][(jj & 1) * 2 + 0];
      const uint32_t d1 = x[n][jj >> 1][(jj & 1) * 2 + 1];
      const float xf[4] = {hs_bf16lo(d0), hs_bf16hi(d0), hs_bf16lo(d1), hs_bf16hi(d1)};
#pragma unroll
      for (int y = 0; y < YT; y++)
#pragma unroll
        for (int b = 0; b < 4; b++) t[n][y] = __builtin_fmaf(wf[y][b], xf[b], t[n][y]);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// mix_down.  grid = ceil(N / YT), block = (64, KS), KS = ceil(ceil(K / 1024) / SPW) <= 10.
//   W int8 [N, K], S fp32 [N, K >> gshift], X bf16 [NB, K] (row stride Xs), C bf16 [NB, N],
//   Cinj bf16 [NB, HCn] or nullptr.  Rows < R get t = bf16(v)/HCn ; v = t * sigmoid(t).
// ------------------------------------------------------------------------------------------------
// launch bounds 640 (KS <= 10 waves): 3 waves/SIMD -> an 84-VGPR budget instead of the 64 that
// 1024 would force (which made the compiler park live values in AGPRs at NB >= 2).
constexpr int HS_DOWN_MAXKS = 10;
template <int NB, int YT, int SPW>
__global__ void __launch_bounds__(HS_DOWN_MAXKS * 64)
hcsm_down_kernel(const int K, const int N, const int8_t* __restrict__ W, const float* __restrict__ S,
                 const uint16_t* __restrict__ X, const int Xs, uint16_t* __restrict__ C,
                 uint16_t* __restrict__ Cinj, const int G, const int gshift, const int R,
                 const int HCn, const int KS) {
  __shared__ float red[HS_DOWN_MAXKS * NB * YT];
  const int lane = threadIdx.x, ks = threadIdx.y;
  const int row0 = blockIdx.x * YT;

  hs_u32x4 w[SPW][YT];
  float sc[SPW][YT];
  hs_u32x4 xv[SPW][NB][2];
  // ---- 1. every load up front: weights (nontemporal), group scales, activation
#pragma unroll
  for (int s = 0; s < SPW; s++) {
    const int k = (ks + s * KS) * 1024 + lane * 16;
    const int kc = k < K ? k : K - 16;
#pragma unroll
    for (int y = 0; y < YT; y++) {
      const int row = (row0 + y < N) ? row0 + y : N - 1;
      w[s][y] = __builtin_nontemporal_load(reinterpret_cast<const hs_u32x4*>(W + (uint32_t)(row * K + kc)));
    }
  }
#pragma unroll
  for (int s = 0; s < SPW; s++) {
    const int k = (ks + s * KS) * 1024 + lane * 16;
    const int kc = k < K ? k : K - 16;
#pragma unroll
    for (int y = 0; y < YT; y++) {
      const int row = (row0 + y < N) ? row0 + y : N - 1;
      sc[s][y] = S[(uint32_t)(row * G + (kc >> gshift))];
    }
#pragma unroll
    for (int n = 0; n < NB; n++) {
      const hs_u32x4* xp = reinterpret_cast<const hs_u32x4*>(X + (uint32_t)(n * Xs + kc));
      xv[s][n][0] = xp[0];
      xv[s][n][1] = xp[1];
    }
  }

  // Keep every load above this point (checked in the gfx908 ISA: all 2*YT + 2*NB loads are issued
  // before the first s_waitcnt vmcnt for every instantiated config).
  __builtin_amdgcn_sched_barrier(0);

  // ---- 2. dot products (per lane: YT rows x 16 k per step, one group scale per chunk)
  float sum[NB][YT];
#pragma unroll
  for (int n = 0; n < NB; n++)
#pragma unroll
    for (int y = 0; y < YT; y++) sum[n][y] = 0.f;
#pragma unroll
  for (int s = 0; s < SPW; s++) {
    // Out-of-range lanes (k >= K) contribute exact zeros.  Branch-free on purpose: a use of the
    // loaded registers inside an `if` lets LLVM sink the loads into that block, i.e. below the
    // first wait, which re-serialises the round trips this kernel exists to overlap.
    const bool valid = (ks + s * KS) * 1024 + lane * 16 < K;
    hs_u32x4 ws[YT];
    hs_u32x4 xs[NB][2];
#pragma unroll
    for (int y = 0; y < YT; y++) ws[y] = valid ? w[s][y] : (hs_u32x4){0u, 0u, 0u, 0u};
#pragma unroll
    for (int n = 0; n < NB; n++) {
      xs[n][0] = valid ? xv[s][n][0] : (hs_u32x4){0u, 0u, 0u, 0u};
      xs[n][1] = valid ? xv[s][n][1] : (hs_u32x4){0u, 0u, 0u, 0u};
    }
    float t[NB][YT];
    hs_dot16<NB, YT>(ws, xs, t);
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < YT; y++)
        sum[n][y] = __builtin_fmaf(t[n][y], valid ? sc[s][y] : 0.f, sum[n][y]);
  }

  // ---- 3. wave reduction, then the KS wave partials through LDS in fixed (ks) order
#pragma unroll
  for (int n = 0; n < NB; n++)
#pragma unroll
    for (int y = 0; y < YT; y++) sum[n][y] = hs_wave_reduce(sum[n][y]);
  if (lane == 63) {
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < YT; y++) red[ks * (NB * YT) + n * YT + y] = sum[n][y];
  }
  __syncthreads();
  if (ks == 0 && lane < NB * YT) {
    const int n = lane / YT, y = lane % YT;
    const int row = row0 + y;
    float v = 0.f;
    for (int q = 0; q < KS; q++) v += red[q * (NB * YT) + lane];
    if (row < N) {
      if (row < R) {
        const float t = hs_bf16(v) / HCn;
        v = t / (1.f + __expf(-t));
      }
      const uint16_t o = hs_tobf16(v);
      C[(size_t)n * N + row] = o;
      if (Cinj != nullptr && row >= R && row < R + HCn) Cinj[(size_t)n * HCn + (row - R)] = o;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// mix_up + gate-mix.  grid = ceil(HD / IPB), block = IPB * CPR threads (a multiple of 64).
//   W int8 [HC*HD, K] (row i*HC + s = original row s*HD + i), S fp32 [HC*HD, K >> gshift],
//   X bf16 [NB, K] (row stride Xs), XN bf16 [NB, HC*HD], Y bf16 [NB, HD].
//   Dynamic LDS: xs[NB*K] floats + red[IPB*CPR*NB*HC] floats.
// ------------------------------------------------------------------------------------------------
template <int NB>
__global__ void __launch_bounds__(1024)
hcsm_up_kernel(const int K, const int HD, const int8_t* __restrict__ W, const float* __restrict__ S,
               const uint16_t* __restrict__ X, const int Xs, const uint16_t* __restrict__ XN,
               uint16_t* __restrict__ Y, const int G, const int gshift, const int IPB) {
  extern __shared__ __align__(16) float hs_lds[];
  const int CPR = K >> 4;                 // 16-k chunks per row
  // units per WG = IPB * CPR == blockDim.x (host-guaranteed)
  float* xs = hs_lds;                     // [NB][K]
  float* red = hs_lds + NB * K;           // [U][NB][HC]
  const int tid = threadIdx.x;
  const int i0 = blockIdx.x * IPB;
  const int il = tid / CPR, c = tid - il * CPR;
  const int i = i0 + il;
  const bool uact = i < HD;
  const int ic = (i < HD) ? i : HD - 1;

  // ---- 0. the (tiny) activation first: thread t < K/8 loads 8 bf16 of every row (L2 hits, so they
  //         land before the weights and the LDS staging below does not wait on the weight stream;
  //         vmcnt is in-order).  Clamped + unconditional so the load stays in the entry block.
  const int K8 = K >> 3;
  const int xt = (tid < K8) ? tid : 0;
  hs_u32x4 xv[NB];
#pragma unroll
  for (int n = 0; n < NB; n++) xv[n] = *reinterpret_cast<const hs_u32x4*>(X + (uint32_t)(n * Xs + xt * 8));
  __builtin_amdgcn_sched_barrier(0);   // X strictly before the weights (the scheduler reorders otherwise)

  // ---- 1. weights for the 4 permuted rows of index i, chunk c; their group scales
  hs_u32x4 w[HS_HC];
  float sc[HS_HC];
#pragma unroll
  for (int s = 0; s < HS_HC; s++) {
    const int row = ic * HS_HC + s;
    w[s] = __builtin_nontemporal_load(reinterpret_cast<const hs_u32x4*>(W + (uint32_t)(row * K + c * 16)));
  }
#pragma unroll
  for (int s = 0; s < HS_HC; s++) {
    const int row = ic * HS_HC + s;
    sc[s] = S[(uint32_t)(row * G + ((c * 16) >> gshift))];
  }
  // gate-phase thread (il_g, n_g, s_g) = tid: prefetch its XN value now (no dependence on anything)
  const int gthreads = IPB * NB * HS_HC;
  const int s_g = tid & 3, n_g = (tid >> 2) % NB, il_g = (tid >> 2) / NB;
  const int i_g = i0 + il_g;
  // unconditional (clamped) load + select: a load under an `if` would sit in its own block
  const bool gact = (tid < gthreads) && (i_g < HD);
  const int n_gc = gact ? n_g : 0, i_gc = gact ? i_g : 0;
  const float xn_raw = hs_bf16u(XN[(uint32_t)(n_gc * (HS_HC * HD) + s_g * HD + i_gc)]);

  __builtin_amdgcn_sched_barrier(0);   // keep the X / weight / scale / XN loads ahead of the staging
  // Anchor the X registers here: without a use in the entry block LLVM sinks the X loads into the
  // `if (tid < K8)` staging block, i.e. behind the whole weight stream.  X was issued first, so the
  // wait this implies is vmcnt(#weight/scale/XN loads) -- the weight stream stays in flight.
#pragma unroll
  for (int n = 0; n < NB; n++) asm volatile("" : "+v"(xv[n]));

  // ---- 2. stage the activation as fp32 in LDS (host guarantees K/8 <= blockDim); the weight
  //         loads stay in flight across this and the barrier
  if (tid < K8) {
#pragma unroll
    for (int n = 0; n < NB; n++) {
      const hs_u32x4 v = xv[n];
      float* d = xs + n * K + tid * 8;
      d[0] = hs_bf16lo(v.x); d[1] = hs_bf16hi(v.x); d[2] = hs_bf16lo(v.y); d[3] = hs_bf16hi(v.y);
      d[4] = hs_bf16lo(v.z); d[5] = hs_bf16hi(v.z); d[6] = hs_bf16lo(v.w); d[7] = hs_bf16hi(v.w);
    }
  }
  __syncthreads();

  // ---- 3. per-unit partials: p[n][s] = scale * sum_16 w * x   ->  red[tid][n][s]
  //         (blockDim == U, so every thread owns a unit; units past HD compute exact zeros --
  //         branch-free so the weight loads cannot be sunk below the staging barrier)
  {
    float t[NB][HS_HC];
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int s = 0; s < HS_HC; s++) t[n][s] = 0.f;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
      float wf[HS_HC][4];
#pragma unroll
      for (int s = 0; s < HS_HC; s++) {
        const uint32_t d = uact ? w[s][jj] : 0u;
#pragma unroll
        for (int b = 0; b < 4; b++) wf[s][b] = hs_i8f(d, b);
      }
#pragma unroll
      for (int n = 0; n < NB; n++) {
        const float4 a = *reinterpret_cast<const float4*>(xs + n * K + c * 16 + jj * 4);
        const float xf[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (int s = 0; s < HS_HC; s++)
#pragma unroll
          for (int b = 0; b < 4; b++) t[n][s] = __builtin_fmaf(wf[s][b], xf[b], t[n][s]);
      }
    }
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int s = 0; s < HS_HC; s++)
        red[(tid * NB + n) * HS_HC + s] = __builtin_fmaf(t[n][s], uact ? sc[s] : 0.f, 0.f);
  }
  __syncthreads();

  // ---- 4. gate phase: thread (il_g, n_g, s_g) sums its row's CPR chunk partials in chunk order,
  //         the quad s = 0..3 is gathered by DPP and lane s = 0 applies the shipping gate-mix.
  //         gthreads is a multiple of 4, so every quad is either fully inside or fully outside.
  if (tid < gthreads) {
    float g = 0.f;
    const float* rp = red + ((size_t)(il_g * CPR) * NB + n_g) * HS_HC + s_g;
    for (int cc = 0; cc < CPR; cc++) g += rp[(size_t)cc * NB * HS_HC];
    const float sg = 1.f / (1.f + __expf(-hs_bf16(g)));
    // quad_perm broadcast of quad-lane q: all four selectors = q
    const float sg1 = __builtin_amdgcn_mov_dpp(sg, 0x55, 0xf, 0xf, 0);
    const float sg2 = __builtin_amdgcn_mov_dpp(sg, 0xaa, 0xf, 0xf, 0);
    const float sg3 = __builtin_amdgcn_mov_dpp(sg, 0xff, 0xf, 0xf, 0);
    const float xn_g = xn_raw;
    const float xn1 = __builtin_amdgcn_mov_dpp(xn_g, 0x55, 0xf, 0xf, 0);
    const float xn2 = __builtin_amdgcn_mov_dpp(xn_g, 0xaa, 0xf, 0xf, 0);
    const float xn3 = __builtin_amdgcn_mov_dpp(xn_g, 0xff, 0xf, 0xf, 0);
    if (s_g == 0 && i_g < HD) {
      float acc = 0.f;
      acc = __builtin_fmaf(sg, xn_g, acc);
      acc = __builtin_fmaf(sg1, xn1, acc);
      acc = __builtin_fmaf(sg2, xn2, acc);
      acc = __builtin_fmaf(sg3, xn3, acc);
      Y[(size_t)n_g * HD + i_g] = hs_tobf16(acc / HS_HC);
    }
  }
}

template <int NB, int YT, int SPW>
bool hcsm_down_launch(int K, int N, const int8_t* W, const float* S, const uint16_t* X, int Xs,
                      uint16_t* C, uint16_t* Cinj, int G, int gshift, int R, int HCn,
                      hipStream_t st) {
  const int nsteps = (K + 1023) / 1024;
  const int KS = (nsteps + SPW - 1) / SPW;
  if (KS < 1 || KS > HS_DOWN_MAXKS) return false;
  dim3 grid((N + YT - 1) / YT), block(64, KS);
  hcsm_down_kernel<NB, YT, SPW><<<grid, block, 0, st>>>(K, N, W, S, X, Xs, C, Cinj, G, gshift, R,
                                                        HCn, KS);
  return true;
}

template <int NB>
bool hcsm_down_cfg(int yt, int spw, int K, int N, const int8_t* W, const float* S,
                   const uint16_t* X, int Xs, uint16_t* C, uint16_t* Cinj, int G, int gshift,
                   int R, int HCn, hipStream_t st) {
#define HCSM_DOWN(YTv, SPWv)                                                               \
  if (yt == YTv && spw == SPWv) {                                                          \
    return hcsm_down_launch<NB, YTv, SPWv>(K, N, W, S, X, Xs, C, Cinj, G, gshift, R, HCn, st); \
  }
  // SPW = 1 only: K = 10240 is 10 wave-steps -> 10 waves.  SPW = 2 (5 waves, 2 steps each) spilled
  // to AGPRs at NB >= 3 and serialised its loads, so it is not instantiated.
  HCSM_DOWN(2, 1) HCSM_DOWN(3, 1) HCSM_DOWN(4, 1)
#undef HCSM_DOWN
  return false;
}

static int hs_gcd(int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; }

// Workgroup geometry of hcsm_up: IPB hidden indices x CPR chunks, IPB*CPR % 64 == 0.
bool hcsm_up_geometry(int K, int NB, int& IPB, int& U, size_t& lds) {
  const int CPR = K / 16;
  IPB = 64 / hs_gcd(CPR, 64);   // smallest index count with IPB*CPR % 64 == 0
  U = IPB * CPR;
  lds = sizeof(float) * ((size_t)NB * K + (size_t)U * NB * HS_HC);
  return U <= 1024 && IPB * NB * HS_HC <= U && K / 8 <= U && lds <= 64 * 1024;
}

template <int NB>
bool hcsm_up_launch(int K, int HD, const int8_t* W, const float* S, const uint16_t* X, int Xs,
                    const uint16_t* XN, uint16_t* Y, int G, int gshift, hipStream_t st) {
  int IPB, U;
  size_t lds;
  if (!hcsm_up_geometry(K, NB, IPB, U, lds)) return false;
  dim3 grid((HD + IPB - 1) / IPB), block(U);
  hcsm_up_kernel<NB><<<grid, block, lds, st>>>(K, HD, W, S, X, Xs, XN, Y, G, gshift, IPB);
  return true;
}

int hs_gs_shift(int64_t gs) {
  int sh = 0;
  while ((1 << sh) < gs) sh++;
  return sh;
}

}  // namespace

// hcsm_down(W int8 [N,K], S fp32 [N,K/gs], X bf16 [M,K] (unit inner stride), out bf16 [M,N],
//           inj bf16 [M,hc] or empty, gs, ytile, spw, hc, r) -> bool
bool hcsm_down(at::Tensor W, at::Tensor S, at::Tensor X, at::Tensor out, at::Tensor inj, int64_t gs,
               int64_t ytile, int64_t spw, int64_t hc, int64_t r) {
  TORCH_CHECK(W.dim() == 2 && W.dtype() == torch::kInt8 && W.is_contiguous(), "W: int8 [N, K]");
  const int N = (int)W.size(0), K = (int)W.size(1), M = (int)X.size(0);
  TORCH_CHECK(S.dtype() == torch::kFloat32 && S.is_contiguous(), "S: fp32 contiguous");
  TORCH_CHECK(X.dim() == 2 && X.dtype() == torch::kBFloat16 && X.stride(1) == 1 && X.size(1) == K,
              "X: bf16 [M, K], unit inner stride");
  TORCH_CHECK(out.dim() == 2 && out.dtype() == torch::kBFloat16 && out.is_contiguous() &&
                  out.size(0) == M && out.size(1) == N, "out: bf16 [M, N] contiguous");
  // all device offsets are 32-bit (lets the compiler use the SGPR-base + VGPR-offset load form)
  if ((long)N * K >= (1L << 31) || (long)M * X.stride(0) >= (1L << 31)) return false;
  if (M < 1 || M > 4 || K % 16 != 0 || K < 16 || X.stride(0) % 8 != 0 || gs < 16 || (gs & (gs - 1)) != 0 ||
      K % gs != 0 || S.numel() != (long)N * (K / gs))
    return false;
  uint16_t* injp = nullptr;
  if (inj.numel() > 0) {
    TORCH_CHECK(inj.dtype() == torch::kBFloat16 && inj.is_contiguous() && inj.size(0) == M &&
                    inj.size(1) == hc, "inj: bf16 [M, hc] contiguous");
    injp = reinterpret_cast<uint16_t*>(inj.data_ptr());
  }
  // instantiated (ytile, spw) set, checked before anything touches the device
  if (!((ytile == 2 || ytile == 3 || ytile == 4) && spw == 1)) return false;
  if ((K + 1023) / 1024 > HS_DOWN_MAXKS) return false;
  const int G = K / (int)gs, sh = hs_gs_shift(gs);
  auto* Wp = reinterpret_cast<const int8_t*>(W.data_ptr());
  auto* Sp = reinterpret_cast<const float*>(S.data_ptr());
  auto* Xp = reinterpret_cast<const uint16_t*>(X.data_ptr());
  auto* Cp = reinterpret_cast<uint16_t*>(out.data_ptr());
  const int Xs = (int)X.stride(0);
  hipStream_t st = c10::hip::getCurrentHIPStream().stream();
  const int yt = (int)ytile, sp = (int)spw, R = (int)r, H = (int)hc;
  switch (M) {
    case 1: return hcsm_down_cfg<1>(yt, sp, K, N, Wp, Sp, Xp, Xs, Cp, injp, G, sh, R, H, st);
    case 2: return hcsm_down_cfg<2>(yt, sp, K, N, Wp, Sp, Xp, Xs, Cp, injp, G, sh, R, H, st);
    case 3: return hcsm_down_cfg<3>(yt, sp, K, N, Wp, Sp, Xp, Xs, Cp, injp, G, sh, R, H, st);
    default: return hcsm_down_cfg<4>(yt, sp, K, N, Wp, Sp, Xp, Xs, Cp, injp, G, sh, R, H, st);
  }
}

// hcsm_up(W int8 [hc*HD,K] permuted, S fp32 [hc*HD,K/gs], X bf16 [M,K] (unit inner stride, row stride
//         a multiple of 8), xn bf16 [M,hc*HD], out bf16 [M,HD], gs, hc) -> bool
bool hcsm_up(at::Tensor W, at::Tensor S, at::Tensor X, at::Tensor xn, at::Tensor out, int64_t gs,
             int64_t hc) {
  TORCH_CHECK(W.dim() == 2 && W.dtype() == torch::kInt8 && W.is_contiguous(), "W: int8 [hc*HD, K]");
  const int NR = (int)W.size(0), K = (int)W.size(1), M = (int)X.size(0);
  TORCH_CHECK(S.dtype() == torch::kFloat32 && S.is_contiguous(), "S: fp32 contiguous");
  TORCH_CHECK(X.dim() == 2 && X.dtype() == torch::kBFloat16 && X.stride(1) == 1 && X.size(1) == K,
              "X: bf16 [M, K], unit inner stride");
  TORCH_CHECK(out.dim() == 2 && out.dtype() == torch::kBFloat16 && out.is_contiguous() &&
                  out.size(0) == M, "out: bf16 [M, HD] contiguous");
  const int HD = (int)out.size(1);
  TORCH_CHECK(xn.dtype() == torch::kBFloat16 && xn.is_contiguous() && xn.size(0) == M &&
                  xn.size(1) == (long)hc * HD, "xn: bf16 [M, hc*HD] contiguous");
  if ((long)NR * K >= (1L << 31) || (long)M * X.stride(0) >= (1L << 31) || (long)M * NR >= (1L << 31))
    return false;
  if (hc != HS_HC || NR != HS_HC * HD || M < 1 || M > 4 || K % 16 != 0 || K < 16 ||
      (M > 1 && X.stride(0) % 8 != 0) || gs < 16 || (gs & (gs - 1)) != 0 || K % gs != 0 ||
      S.numel() != (long)NR * (K / gs))
    return false;
  {
    int IPB, U;
    size_t lds;
    if (!hcsm_up_geometry(K, M, IPB, U, lds)) return false;   // before anything touches the device
  }
  const int G = K / (int)gs, sh = hs_gs_shift(gs);
  auto* Wp = reinterpret_cast<const int8_t*>(W.data_ptr());
  auto* Sp = reinterpret_cast<const float*>(S.data_ptr());
  auto* Xp = reinterpret_cast<const uint16_t*>(X.data_ptr());
  auto* XNp = reinterpret_cast<const uint16_t*>(xn.data_ptr());
  auto* Yp = reinterpret_cast<uint16_t*>(out.data_ptr());
  const int Xs = (int)X.stride(0);
  hipStream_t st = c10::hip::getCurrentHIPStream().stream();
  switch (M) {
    case 1: return hcsm_up_launch<1>(K, HD, Wp, Sp, Xp, Xs, XNp, Yp, G, sh, st);
    case 2: return hcsm_up_launch<2>(K, HD, Wp, Sp, Xp, Xs, XNp, Yp, G, sh, st);
    case 3: return hcsm_up_launch<3>(K, HD, Wp, Sp, Xp, Xs, XNp, Yp, G, sh, st);
    default: return hcsm_up_launch<4>(K, HD, Wp, Sp, Xp, Xs, XNp, Yp, G, sh, st);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hcsm_down", &hcsm_down);
  m.def("hcsm_up", &hcsm_up);
}
