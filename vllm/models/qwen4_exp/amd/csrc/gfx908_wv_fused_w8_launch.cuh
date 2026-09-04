// Per-(NB, EPI) launcher for the fused W8 hyper-connection GEMVs.
// Instantiated in gfx908_wv_fused_w8_nb{1,2,3}.hip so the batch sizes compile in parallel.
#pragma once
#include "gfx908_wv_fused_w8_kernel.cuh"

struct HW8Args {
  int K, Nrows;
  const int8_t* W;
  const float* S;
  const uint16_t* X;
  uint16_t* C;
  int cu;
  hipStream_t st;
  int G, gshift, gs;
  const uint16_t* XN;
  int HD, R, HCn;
  int Xs;  // X row stride in elements
};

static inline int hw8_ceil_div(long a, long b) { return (int)((a + b - 1) / b); }

template <int YT, int UN, int NB, bool GRP, bool F32, int LPR, int KS, int EPI>
static bool hw8_launch_one(const HW8Args& a) {
  constexpr int RPW = YT * (64 / LPR), RGmax = 16 / KS;
  const int Kslice = a.K / KS;
  if (a.K % KS != 0 || Kslice % 16 != 0) return false;
  if (GRP && (Kslice % a.gs != 0)) return false;
  if constexpr (EPI == 2) { if (YT != a.HCn || a.Nrows % YT != 0) return false; }
  // waves (row groups) per WG: smallest count that keeps the round count of the full WG (stock mindiv idea)
  const int rounds_full = hw8_ceil_div(a.Nrows, (long)a.cu * RGmax * RPW);
  int wv = RGmax;
  for (int c = RGmax - 1; c >= 1; c--) {
    if (hw8_ceil_div(a.Nrows, (long)a.cu * c * RPW) == rounds_full) wv = c; else break;
  }
  const int rounds = hw8_ceil_div(a.Nrows, (long)a.cu * wv * RPW);
  dim3 grid(a.cu), block(64, 16);
  {
    hc_w8_gemv_kernel<YT, UN, NB, GRP, F32, LPR, KS, EPI><<<grid, block, 0, a.st>>>(
        a.K, a.Nrows, a.W, a.S, a.X, a.C, wv, rounds, a.cu, a.G, a.gshift, a.XN, a.HD, a.R, a.HCn,
        a.Xs);
  }
  return true;
}

#define HW8_CFG(YT, UN, LPRv, KSv) \
  if (yt == YT && un == UN && lpr == LPRv && ks == KSv) return hw8_launch_one<YT, UN, NB, GRP, F32, LPRv, KSv, EPI>(a);

template <int NB, int EPI, bool GRP, bool F32>
static bool hw8_launch_cfg(const HW8Args& a, int yt, int un, int lpr, int ks) {
  if constexpr (EPI == 2) {
    // gate-mix: YTILE must equal HC (4); the 4 permuted rows of one hidden index live in one wave-step
    if constexpr (F32) { HW8_CFG(4, 1, 32, 1) HW8_CFG(4, 2, 32, 1) HW8_CFG(4, 1, 64, 1) }
    return false;
  } else {
    // long-K / few-row configs (mix_down 336 x 10240): plain + intra-WG split-K
    HW8_CFG(1, 4, 64, 1) HW8_CFG(2, 2, 64, 1)
    HW8_CFG(1, 2, 64, 2) HW8_CFG(2, 2, 64, 2)
    HW8_CFG(1, 3, 64, 4) HW8_CFG(2, 2, 64, 4)
    return false;
  }
}

template <int NB, int EPI>
bool hw8_launch_nb(const HW8Args& a, int yt, int un, int lpr, int ks) {
  const bool f32 = (long)NB * a.K * 4 <= HW8_X_BYTES;
  if (a.gs == 0)
    return f32 ? hw8_launch_cfg<NB, EPI, false, true>(a, yt, un, lpr, ks)
               : hw8_launch_cfg<NB, EPI, false, false>(a, yt, un, lpr, ks);
  return f32 ? hw8_launch_cfg<NB, EPI, true, true>(a, yt, un, lpr, ks)
             : hw8_launch_cfg<NB, EPI, true, false>(a, yt, un, lpr, ks);
}
