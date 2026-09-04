// W8A16 GEMV for gfx908 (design D1 of bf16_skinny_gemm_and_int8.md), cloned from vLLM's
// wvSplitK_hf_sml_ (csrc/rocm/skinny_gemms.cu) / gfx908_wv_fused.hip, plus two extensions:
//   KS  > 1 : intra-WG split-K (D4 of the report): the 16 waves of a WG are split into KS k-slices of the same
//             rows, partial sums are reduced through LDS, one bf16 store (no atomics, no second launch).
//   LPR = 32: two rows per wave-step (lanes 0-31 / 32-63), so short rows (K = 320) keep 62.5% of the lanes busy.
//
// Weight  W : int8  [Nrows, K] row-major (K contiguous), K % 16 == 0
// Scales  S : fp32  [Nrows]           (GRP = false, per-channel)  y = (sum_k Wq*x) * S[row]
//             fp32  [Nrows, K/gs]     (GRP = true,  group-wise, gs = 1<<gshift >= 16, gs | K/KS)
//                                                                y = sum_g S[row,g] * (sum_{k in g} Wq*x)
// Act     X : bf16  [NB, K] contiguous (NB = batch, template, 1..4)
// Out     C : bf16  [NB, Nrows]
//
// grid = CuCount WGs of 16 wave64 (1 WG / CU; 62 KB of LDS hold X as fp32 or bf16, 2 KB the split-K partials).
// Wave w: row-group rg = w / KS, k-slice ks = w % KS. Each wave streams RPW = YTILE*(64/LPR) rows of its slice;
// each lane loads 16 int8 (dwordx4, nontemporal) per row per step -> LPR*16 B contiguous per row per step,
// UNRL steps in flight. int8 -> fp32 via v_cvt_f32_i32 SDWA byte-select + sext (compiler folds the bfe),
// fp32 fma against X from LDS, DPP row_shr/row_bcast reduction, scale, bf16 store.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define W8_LDS_BYTES (64 * 1024)
#define W8_RED_BYTES (2 * 1024)                      // 16 waves x 32 partials x 4 B
#define W8_X_BYTES (W8_LDS_BYTES - W8_RED_BYTES)     // 63488 B of X
// WG barrier that waits for LDS traffic only (keeps the prefetched global loads in flight)
#define W8_LDS_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")

typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ T w8_loadnt(const T* p) { return __builtin_nontemporal_load(p); }

__device__ __forceinline__ float w8_bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float w8_bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
// signed byte b of a dword -> fp32 (v_cvt_f32_i32_sdwa src0_sel:BYTE_b sext)
__device__ __forceinline__ float w8_i8f(uint32_t u, int b) { return (float)(int)(int8_t)(u >> (8 * b)); }

// wave reduction; LPR == 64: lane 63 holds the total; LPR == 32: lane 31 = sum(lanes 0-31), lane 63 = sum(32-63)
template <int LPR>
__device__ __forceinline__ float w8_wave_reduce(float v) {
  v += __builtin_amdgcn_mov_dpp(v, 0x118, 0xf, 0xf, 1);  // row_shr8
  v += __builtin_amdgcn_mov_dpp(v, 0x114, 0xf, 0xf, 1);  // row_shr4
  v += __builtin_amdgcn_mov_dpp(v, 0x112, 0xf, 0xf, 1);  // row_shr2
  v += __builtin_amdgcn_mov_dpp(v, 0x111, 0xf, 0xf, 1);  // row_shr1
  v += __builtin_amdgcn_mov_dpp(v, 0x142, 0xf, 0xf, 1);  // row_bcast15
  if constexpr (LPR == 64) v += __builtin_amdgcn_mov_dpp(v, 0x143, 0xf, 0xf, 1);  // row_bcast31
  return v;
}

// PF: issue the first weight loads before X is staged (LDS-only barrier); false = stock order (stage, __syncthreads, load)
template <int YTILE, int UNRL, int NB, bool GRP, bool F32LDS, int LPR, int KS, bool PF>
__global__ void __launch_bounds__(1024)
w8a16_gemv_kernel(const int K, const int Nrows, const int8_t* __restrict__ W,
                  const float* __restrict__ S, const uint16_t* __restrict__ X,
                  uint16_t* __restrict__ C, const int wvPrGrp, const int rounds, const int CuCount,
                  const int G, const int gshift) {
  constexpr int THRDS = 64, A_CHUNK = 16, KSTEP = LPR * A_CHUNK;  // k per wave-step
  constexpr int SUBS = THRDS / LPR, RPW = YTILE * SUBS;            // rows per wave
  constexpr int RG = 16 / KS;                                      // row groups per WG
  constexpr int LDS_BF16 = W8_X_BYTES / 2;
  static_assert(KS == 1 || (NB * RPW <= 32 && NB * RPW <= THRDS), "split-K partial buffer");
  static_assert(W8_X_BYTES / 2 <= 4 * 8192, "single-pass staging");
  __shared__ __align__(16) unsigned char lds_raw[W8_LDS_BYTES];
  float* sf = reinterpret_cast<float*>(lds_raw);
  uint16_t* sh = reinterpret_cast<uint16_t*>(lds_raw);
  float* red = reinterpret_cast<float*>(lds_raw + W8_X_BYTES);

  const int tid = threadIdx.y * THRDS + threadIdx.x;
  const int total = NB * K;
  const int lane = threadIdx.x, wave = threadIdx.y;
  const int rg = wave / KS, ks = wave % KS;
  const int sub = lane / LPR, kl = lane % LPR;
  const int Kslice = K / KS, kbeg = ks * Kslice, kend = kbeg + Kslice;
  const uint32_t mstride = CuCount * wvPrGrp * RPW;
  uint32_t m = (blockIdx.x * wvPrGrp + rg) * RPW;

  u32x4_t bw[YTILE][UNRL];
  float sc[YTILE][UNRL];
  // ---- weight stream: YTILE*UNRL dwordx4 nontemporal loads in flight per lane
  auto issue = [&](uint32_t k1, uint32_t mm) {
#pragma unroll
    for (int k2 = 0; k2 < UNRL; k2++) {
      uint32_t k_ = k1 + k2 * KSTEP + kl * A_CHUNK;
      uint32_t kc = min(k_, (uint32_t)(kend - A_CHUNK));
#pragma unroll
      for (int y = 0; y < YTILE; y++) {
        uint32_t row = min(mm + sub * YTILE + y, (uint32_t)(Nrows - 1));
        bw[y][k2] = w8_loadnt(reinterpret_cast<const u32x4_t*>(W + (size_t)row * K + kc));
        if constexpr (GRP) sc[y][k2] = S[row * G + (kc >> gshift)];
      }
    }
  };

  // X staging: global loads first, then the first weight loads, then the LDS stores. vmcnt is in-order on
  // gfx9, so this order lets the x loads be consumed with vmcnt(YTILE*UNRL) while the HBM weight loads stay
  // in flight across the staging + barrier. (__syncthreads would force vmcnt(0); the inline-asm barrier
  // below waits on LDS only.) NB*K fits 4 loads per thread in both LDS modes, so this is a single pass.
  {
    const int lim = F32LDS ? total : (total < LDS_BF16 ? total : LDS_BF16);
    u32x4_t v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) { const int k = tid * 8 + i * 8192; if (k < lim) v[i] = *reinterpret_cast<const u32x4_t*>(X + k); }
    // unconditional (rows/k are clamped, so inactive waves just re-read valid lines); the sched barriers keep
    // the machine scheduler from sinking these invariant loads past the LDS barrier
    if constexpr (PF) {
      __builtin_amdgcn_sched_barrier(0);
      issue(kbeg, m);
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int k = tid * 8 + i * 8192;
      if (k < lim) {
        if constexpr (F32LDS) {
          float4 a = make_float4(w8_bf16lo(v[i].x), w8_bf16hi(v[i].x), w8_bf16lo(v[i].y), w8_bf16hi(v[i].y));
          float4 b = make_float4(w8_bf16lo(v[i].z), w8_bf16hi(v[i].z), w8_bf16lo(v[i].w), w8_bf16hi(v[i].w));
          *reinterpret_cast<float4*>(sf + k) = a;
          *reinterpret_cast<float4*>(sf + k + 4) = b;
        } else {
          *reinterpret_cast<u32x4_t*>(sh + k) = v[i];
        }
      }
    }
  }
  if constexpr (PF) {
    __builtin_amdgcn_sched_barrier(0);
    W8_LDS_BARRIER();
    __builtin_amdgcn_sched_barrier(0);
  } else {
    __syncthreads();
  }
  if constexpr (KS == 1) { if (rg >= wvPrGrp) return; }

  for (int r = 0; r < rounds; r++, m += mstride) {
    const bool active = (rg < wvPrGrp) && (m < (uint32_t)Nrows);  // wave-uniform
    float sum[NB][YTILE];
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < YTILE; y++) sum[n][y] = 0.f;

    if (active) {
      if (!PF || r > 0) issue(kbeg, m);
      for (uint32_t k1 = kbeg; k1 < (uint32_t)kend; k1 += KSTEP * UNRL) {
        // ---- dequant + fma in small k chunks to keep VGPR pressure low (64 VGPR budget at 4 waves/SIMD)
#pragma unroll
        for (int k2 = 0; k2 < UNRL; k2++) {
          uint32_t k_ = k1 + k2 * KSTEP + kl * A_CHUNK;
          if (k_ < (uint32_t)kend) {
            float t[NB][YTILE];
            if constexpr (GRP) {
#pragma unroll
              for (int n = 0; n < NB; n++)
#pragma unroll
                for (int y = 0; y < YTILE; y++) t[n][y] = 0.f;
            }
            // XW k per step: 4 (fp32 LDS: one float4 per n, lowest VGPR pressure) or 8 (bf16 LDS: one 16 B load per n)
            constexpr int XW = F32LDS ? 4 : 8, XD = XW / 4;
#pragma unroll
            for (int jj = 0; jj < A_CHUNK / XW; jj++) {
              float wf[YTILE][XW];
#pragma unroll
              for (int y = 0; y < YTILE; y++) {
#pragma unroll
                for (int h = 0; h < XD; h++) {
                  const uint32_t d = bw[y][k2][XD * jj + h];
#pragma unroll
                  for (int b = 0; b < 4; b++) wf[y][4 * h + b] = w8_i8f(d, b);
                }
              }
#pragma unroll
              for (int n = 0; n < NB; n++) {
                float xv[XW];
                if constexpr (F32LDS) {
                  const float4 a = *reinterpret_cast<const float4*>(sf + n * K + k_ + XW * jj);
                  xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
                } else {
                  const int idx = n * K + k_ + XW * jj;
                  const u32x4_t u = (idx < LDS_BF16) ? *reinterpret_cast<const u32x4_t*>(sh + idx)
                                                     : *reinterpret_cast<const u32x4_t*>(X + idx);
                  xv[0] = w8_bf16lo(u.x); xv[1] = w8_bf16hi(u.x); xv[2] = w8_bf16lo(u.y); xv[3] = w8_bf16hi(u.y);
                  xv[4] = w8_bf16lo(u.z); xv[5] = w8_bf16hi(u.z); xv[6] = w8_bf16lo(u.w); xv[7] = w8_bf16hi(u.w);
                }
#pragma unroll
                for (int y = 0; y < YTILE; y++) {
                  float& acc = GRP ? t[n][y] : sum[n][y];
#pragma unroll
                  for (int i = 0; i < XW; i++) acc = __builtin_fmaf(wf[y][i], xv[i], acc);
                }
              }
            }
            if constexpr (GRP) {
#pragma unroll
              for (int n = 0; n < NB; n++)
#pragma unroll
                for (int y = 0; y < YTILE; y++) sum[n][y] = __builtin_fmaf(t[n][y], sc[y][k2], sum[n][y]);
            }
          }
        }
        const uint32_t knext = k1 + KSTEP * UNRL;
        if (knext < (uint32_t)kend) issue(knext, m);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    // ---- cross-lane reduction
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < YTILE; y++) sum[n][y] = w8_wave_reduce<LPR>(sum[n][y]);

    if constexpr (KS == 1) {
      if (kl == LPR - 1) {  // lane 63 (and lane 31 when LPR == 32) writes its sub-rows
#pragma unroll
        for (int y = 0; y < YTILE; y++) {
          const uint32_t row = m + sub * YTILE + y;
          if (row < (uint32_t)Nrows) {
            const float s = GRP ? 1.f : S[row];
#pragma unroll
            for (int n = 0; n < NB; n++) {
              __hip_bfloat16 o = __float2bfloat16(sum[n][y] * s);
              C[(size_t)n * Nrows + row] = *reinterpret_cast<uint16_t*>(&o);
            }
          }
        }
      }
    } else {
      // split-K: partials -> LDS, barrier, wave ks==0 sums KS partials and stores
      if (kl == LPR - 1) {
#pragma unroll
        for (int y = 0; y < YTILE; y++)
#pragma unroll
          for (int n = 0; n < NB; n++) red[wave * 32 + (n * RPW + sub * YTILE + y)] = sum[n][y];
      }
      __syncthreads();
      if (ks == 0 && active && lane < NB * RPW) {
        const int n = lane / RPW, rr = lane % RPW;
        const uint32_t row = m + rr;
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < KS; q++) v += red[(rg * KS + q) * 32 + lane];
        if (row < (uint32_t)Nrows) {
          const float s = GRP ? 1.f : S[row];
          __hip_bfloat16 o = __float2bfloat16(v * s);
          C[(size_t)n * Nrows + row] = *reinterpret_cast<uint16_t*>(&o);
        }
      }
      __syncthreads();
    }
  }
}

// peak HBM read probe: every WG streams a contiguous chunk with dwordx4 loads and xors it into one dword
template <bool NT>
__global__ void __launch_bounds__(256) w8_stream_probe(const u32x4_t* __restrict__ p, const long n16, uint32_t* __restrict__ out) {
  const long per = (n16 + gridDim.x - 1) / gridDim.x;
  const long beg = (long)blockIdx.x * per, end = min(beg + per, n16);
  uint32_t acc = 0;
  for (long i = beg + threadIdx.x; i < end; i += 256 * 4) {
    u32x4_t a = {0, 0, 0, 0}, b = a, c = a, d = a;
    if constexpr (NT) {
      a = w8_loadnt(p + i);
      if (i + 256 < end) b = w8_loadnt(p + i + 256);
      if (i + 512 < end) c = w8_loadnt(p + i + 512);
      if (i + 768 < end) d = w8_loadnt(p + i + 768);
    } else {
      a = p[i];
      if (i + 256 < end) b = p[i + 256];
      if (i + 512 < end) c = p[i + 512];
      if (i + 768 < end) d = p[i + 768];
    }
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  if (acc == 0xdeadbeef) out[blockIdx.x] = acc;  // practically never; keeps the loads alive
}

