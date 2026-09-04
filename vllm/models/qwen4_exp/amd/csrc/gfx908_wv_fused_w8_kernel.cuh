// W8 (int8 weight / bf16 activation) fused-epilogue GEMV for the gfx908 hyper-connection mixes.
//
// Kernel body is the W8A16 GEMV of agents/w8a16 (a wvSplitK_hf_sml_ clone whose weight stream is int8,
// with two structural extensions: LPR=32 "two rows per wave-step" for short K, and KS>1 intra-WG
// split-K), with the two fused epilogues of gfx908_wv_fused.hip grafted onto its writer:
//
//   EPI 0 : plain      C[n * Nrows + row] = bf16(sum * scale)
//   EPI 1 : mix_down   as EPI 0, but for row < R:  t = bf16(v) / HCn ; v = t * sigmoid(t)   (== hc_silu)
//   EPI 2 : mix_up     weight rows are permuted so row i*HC + s is the original row s*HD + i, YTILE == HC,
//                      so one wave-step's YTILE rows are the HC gate values of hidden index i:
//                      Y[n * HD + i] = bf16( mean_s sigmoid(bf16(g_s)) * XN[n*(HC*HD) + s*HD + i] )
//
// Weight  W : int8  [Nrows, K] row-major (K contiguous), K % 16 == 0
// Scales  S : fp32  [Nrows]        (GRP=false, per output channel)  or  fp32 [Nrows, K/gs] (GRP=true)
// Act     X : bf16  [NB, K], row stride Xs (inner stride 1); NB = 1..3
// Out     C : bf16  [NB, Nrows]   (EPI 0/1)     Y : bf16 [NB, HD]  (EPI 2)
// XN        : bf16  [NB, HC*HD]   (EPI 2 only, the RMSNorm'ed hidden state)
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define HW8_LDS_BYTES (64 * 1024)
#define HW8_RED_BYTES (2 * 1024)                        // 16 waves x 32 partials x 4 B
#define HW8_X_BYTES (HW8_LDS_BYTES - HW8_RED_BYTES)     // 63488 B of X

typedef uint32_t hw8_u32x4_t __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ T hw8_loadnt(const T* p) { return __builtin_nontemporal_load(p); }

__device__ __forceinline__ float hw8_bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hw8_bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
// signed byte b of a dword -> fp32 (v_cvt_f32_i32_sdwa src0_sel:BYTE_b sext)
__device__ __forceinline__ float hw8_i8f(uint32_t u, int b) { return (float)(int)(int8_t)(u >> (8 * b)); }
__device__ __forceinline__ float hw8_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float hw8_bf16u(uint16_t r) { return __uint_as_float((uint32_t)r << 16); }

// wave reduction; LPR == 64: lane 63 holds the total; LPR == 32: lane 31 = sum(0-31), lane 63 = sum(32-63)
template <int LPR>
__device__ __forceinline__ float hw8_wave_reduce(float v) {
  v += __builtin_amdgcn_mov_dpp(v, 0x118, 0xf, 0xf, 1);  // row_shr8
  v += __builtin_amdgcn_mov_dpp(v, 0x114, 0xf, 0xf, 1);  // row_shr4
  v += __builtin_amdgcn_mov_dpp(v, 0x112, 0xf, 0xf, 1);  // row_shr2
  v += __builtin_amdgcn_mov_dpp(v, 0x111, 0xf, 0xf, 1);  // row_shr1
  v += __builtin_amdgcn_mov_dpp(v, 0x142, 0xf, 0xf, 1);  // row_bcast15
  if constexpr (LPR == 64) v += __builtin_amdgcn_mov_dpp(v, 0x143, 0xf, 0xf, 1);  // row_bcast31
  return v;
}

template <int YTILE, int UNRL, int NB, bool GRP, bool F32LDS, int LPR, int KS, int EPI>
__global__ void __launch_bounds__(1024)
hc_w8_gemv_kernel(const int K, const int Nrows, const int8_t* __restrict__ W,
                  const float* __restrict__ S, const uint16_t* __restrict__ X,
                  uint16_t* __restrict__ C, const int wvPrGrp, const int rounds, const int CuCount,
                  const int G, const int gshift,
                  const uint16_t* __restrict__ XN, const int HD, const int R, const int HCn,
                  const int Xs) {
  constexpr int THRDS = 64, A_CHUNK = 16, KSTEP = LPR * A_CHUNK;  // k per wave-step
  constexpr int SUBS = THRDS / LPR, RPW = YTILE * SUBS;            // rows per wave
  constexpr int RG = 16 / KS;                                      // row groups per WG
  constexpr int LDS_BF16 = HW8_X_BYTES / 2;
  static_assert(KS == 1 || (NB * RPW <= 32 && NB * RPW <= THRDS), "split-K partial buffer");
  static_assert(EPI != 2 || KS == 1, "gate-mix epilogue needs the full row in one wave");
  __shared__ __align__(16) unsigned char lds_raw[HW8_LDS_BYTES];
  float* sf = reinterpret_cast<float*>(lds_raw);
  uint16_t* sh = reinterpret_cast<uint16_t*>(lds_raw);
  float* red = reinterpret_cast<float*>(lds_raw + HW8_X_BYTES);

  const int tid = threadIdx.y * THRDS + threadIdx.x;
  const int total = NB * K;
  const int lane = threadIdx.x, wave = threadIdx.y;
  const int rg = wave / KS, ks = wave % KS;
  const int sub = lane / LPR, kl = lane % LPR;
  const int Kslice = K / KS, kbeg = ks * Kslice, kend = kbeg + Kslice;
  const uint32_t mstride = CuCount * wvPrGrp * RPW;
  uint32_t m = (blockIdx.x * wvPrGrp + rg) * RPW;

  hw8_u32x4_t bw[YTILE][UNRL];
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
        bw[y][k2] = hw8_loadnt(reinterpret_cast<const hw8_u32x4_t*>(W + (size_t)row * K + kc));
        if constexpr (GRP) sc[y][k2] = S[row * G + (kc >> gshift)];
      }
    }
  };

  // X staging (single pass: NB*K fits 4 x 16 B per thread in both LDS modes).
  // LDS holds row n at offset n*K; X itself may have a row stride Xs != K (the mix_up input is a
  // column slice of the mix_down output), so the source index is unpacked into (row, k).
  {
    const int lim = F32LDS ? total : (total < LDS_BF16 ? total : LDS_BF16);
    hw8_u32x4_t v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int k = tid * 8 + i * 8192;
      if (k < lim) {
        const int nn = (NB == 1) ? 0 : (k / K);
        v[i] = *reinterpret_cast<const hw8_u32x4_t*>(X + (size_t)nn * Xs + (k - nn * K));
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int k = tid * 8 + i * 8192;
      if (k < lim) {
        if constexpr (F32LDS) {
          float4 a = make_float4(hw8_bf16lo(v[i].x), hw8_bf16hi(v[i].x), hw8_bf16lo(v[i].y), hw8_bf16hi(v[i].y));
          float4 b = make_float4(hw8_bf16lo(v[i].z), hw8_bf16hi(v[i].z), hw8_bf16lo(v[i].w), hw8_bf16hi(v[i].w));
          *reinterpret_cast<float4*>(sf + k) = a;
          *reinterpret_cast<float4*>(sf + k + 4) = b;
        } else {
          *reinterpret_cast<hw8_u32x4_t*>(sh + k) = v[i];
        }
      }
    }
  }
  __syncthreads();
  if constexpr (KS == 1) { if (rg >= wvPrGrp) return; }

  for (int r = 0; r < rounds; r++, m += mstride) {
    const bool active = (rg < wvPrGrp) && (m < (uint32_t)Nrows);  // wave-uniform
    float sum[NB][YTILE];
#pragma unroll
    for (int n = 0; n < NB; n++)
#pragma unroll
      for (int y = 0; y < YTILE; y++) sum[n][y] = 0.f;

    if (active) {
      issue(kbeg, m);
      for (uint32_t k1 = kbeg; k1 < (uint32_t)kend; k1 += KSTEP * UNRL) {
        // dequant + fma in small k chunks to keep VGPR pressure low (64 VGPR budget at 4 waves/SIMD)
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
                  for (int b = 0; b < 4; b++) wf[y][4 * h + b] = hw8_i8f(d, b);
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
                  const hw8_u32x4_t u =
                      (idx < LDS_BF16)
                          ? *reinterpret_cast<const hw8_u32x4_t*>(sh + idx)
                          : *reinterpret_cast<const hw8_u32x4_t*>(X + (size_t)n * Xs + (k_ + XW * jj));
                  xv[0] = hw8_bf16lo(u.x); xv[1] = hw8_bf16hi(u.x); xv[2] = hw8_bf16lo(u.y); xv[3] = hw8_bf16hi(u.y);
                  xv[4] = hw8_bf16lo(u.z); xv[5] = hw8_bf16hi(u.z); xv[6] = hw8_bf16lo(u.w); xv[7] = hw8_bf16hi(u.w);
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
      for (int y = 0; y < YTILE; y++) sum[n][y] = hw8_wave_reduce<LPR>(sum[n][y]);

    if constexpr (KS == 1) {
      if (kl == LPR - 1) {  // lane 63 (and lane 31 when LPR == 32) writes its sub-rows
        const uint32_t row0 = m + sub * YTILE;
        if constexpr (EPI == 2) {
          // gate-mix over the YTILE(=HC) permuted rows of hidden index i
          if (row0 + YTILE <= (uint32_t)Nrows) {
            const int i = (int)(row0 / YTILE);
#pragma unroll
            for (int n = 0; n < NB; n++) {
              float acc = 0.f;
#pragma unroll
              for (int y = 0; y < YTILE; y++) {
                const float sgl = GRP ? 1.f : S[row0 + y];
                const float g = hw8_bf16(sum[n][y] * sgl);
                acc += (1.f / (1.f + __expf(-g))) * hw8_bf16u(XN[(size_t)n * (HCn * HD) + y * HD + i]);
              }
              __hip_bfloat16 o = __float2bfloat16(acc / HCn);
              C[(size_t)n * HD + i] = *reinterpret_cast<uint16_t*>(&o);
            }
          }
        } else {
#pragma unroll
          for (int y = 0; y < YTILE; y++) {
            const uint32_t row = row0 + y;
            if (row < (uint32_t)Nrows) {
              const float sgl = GRP ? 1.f : S[row];
#pragma unroll
              for (int n = 0; n < NB; n++) {
                float v = sum[n][y] * sgl;
                if constexpr (EPI == 1) {
                  if (row < (uint32_t)R) { float t = hw8_bf16(v) / HCn; v = t / (1.f + __expf(-t)); }
                }
                __hip_bfloat16 o = __float2bfloat16(v);
                C[(size_t)n * Nrows + row] = *reinterpret_cast<uint16_t*>(&o);
              }
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
          v *= (GRP ? 1.f : S[row]);
          if constexpr (EPI == 1) {
            if (row < (uint32_t)R) { float t = hw8_bf16(v) / HCn; v = t / (1.f + __expf(-t)); }
          }
          __hip_bfloat16 o = __float2bfloat16(v);
          C[(size_t)n * Nrows + row] = *reinterpret_cast<uint16_t*>(&o);
        }
      }
      __syncthreads();
    }
  }
}
