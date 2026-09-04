// Per-NB launcher (instantiated in w8a16_nbX.hip so the 4 batch sizes compile in parallel).
#pragma once
#include "gfx908_w8a16_kernel.cuh"

struct W8Args {
  int K, Nrows; const int8_t* W; const float* S; const uint16_t* X; uint16_t* C; int cu; hipStream_t st; int G, gshift, gs;
};

static inline int w8_ceil_div(long a, long b) { return (int)((a + b - 1) / b); }

template <int YT, int UN, int NB, bool GRP, bool F32, int LPR, int KS, bool PF>
static bool w8_launch_one(const W8Args& a) {
  constexpr int RPW = YT * (64 / LPR), RGmax = 16 / KS;
  const int Kslice = a.K / KS;
  if (a.K % KS != 0 || Kslice % 16 != 0) return false;
  if (GRP && (Kslice % a.gs != 0)) return false;
  // waves (row groups) per WG: smallest count that keeps the round count of the full WG (stock mindiv idea)
  const int rounds_full = w8_ceil_div(a.Nrows, (long)a.cu * RGmax * RPW);
  int wv = RGmax;
  for (int c = RGmax - 1; c >= 1; c--) { if (w8_ceil_div(a.Nrows, (long)a.cu * c * RPW) == rounds_full) wv = c; else break; }
  const int rounds = w8_ceil_div(a.Nrows, (long)a.cu * wv * RPW);
  dim3 grid(a.cu), block(64, 16);
  w8a16_gemv_kernel<YT, UN, NB, GRP, F32, LPR, KS, PF><<<grid, block, 0, a.st>>>(a.K, a.Nrows, a.W, a.S, a.X, a.C, wv, rounds, a.cu, a.G, a.gshift);
  return true;
}

// config table: (YTILE, UNRL, LPR, KS). Returned false = not instantiated / not applicable.
#define W8_CFG(YT, UN, LPR, KS) \
  if (yt == YT && un == UN && lpr == LPR && ks == KS) return w8_launch_one<YT, UN, NB, GRP, F32, LPR, KS, PF>(a);

template <int NB, bool GRP, bool F32, bool PF>
static bool w8_launch_cfg(const W8Args& a, int yt, int un, int lpr, int ks) {
  // plain (stock-like) configs
  W8_CFG(1, 4, 64, 1) W8_CFG(2, 2, 64, 1) W8_CFG(2, 4, 64, 1) W8_CFG(4, 1, 64, 1)
  // intra-WG split-K
  W8_CFG(1, 2, 64, 2) W8_CFG(2, 2, 64, 2) W8_CFG(1, 3, 64, 4) W8_CFG(2, 2, 64, 4)
  if constexpr (F32) {
    // two rows per wave-step (short K / K not a multiple of 1024)
    W8_CFG(2, 1, 32, 1) W8_CFG(4, 1, 32, 1) W8_CFG(8, 1, 32, 1) W8_CFG(2, 2, 32, 1) W8_CFG(1, 2, 32, 1) W8_CFG(2, 1, 32, 2) W8_CFG(4, 1, 32, 2)
  }
  return false;
}

template <int NB>
bool w8_launch_nb(const W8Args& a, int yt, int un, int lpr, int ks, int pf) {
  const bool f32 = (long)NB * a.K * 4 <= W8_X_BYTES;
  if (pf) {
    if (a.gs == 0) return f32 ? w8_launch_cfg<NB, false, true, true>(a, yt, un, lpr, ks) : w8_launch_cfg<NB, false, false, true>(a, yt, un, lpr, ks);
    else           return f32 ? w8_launch_cfg<NB, true, true, true>(a, yt, un, lpr, ks) : w8_launch_cfg<NB, true, false, true>(a, yt, un, lpr, ks);
  } else {
    if (a.gs == 0) return f32 ? w8_launch_cfg<NB, false, true, false>(a, yt, un, lpr, ks) : w8_launch_cfg<NB, false, false, false>(a, yt, un, lpr, ks);
    else           return f32 ? w8_launch_cfg<NB, true, true, false>(a, yt, un, lpr, ks) : w8_launch_cfg<NB, true, false, false>(a, yt, un, lpr, ks);
  }
}
