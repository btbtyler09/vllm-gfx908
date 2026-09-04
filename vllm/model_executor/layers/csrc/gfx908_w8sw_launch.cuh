// Per-NB launcher for the swizzled-layout W8A16 GEMV (one TU per batch size).
#pragma once
#include "gfx908_w8sw_gemv.cuh"

struct WswArgs {
  int K, N;
  const int8_t* W;
  const float* S;
  const uint16_t* X;
  uint16_t* C;
  int cu;
  hipStream_t st;
  int gshift;
};

static inline int wsw_ceil_div(long a, long b) { return (int)((a + b - 1) / b); }

template <int NT, int UNRL, int NB, bool F32, int KS>
static bool wsw_launch_one(const WswArgs& a) {
  const int NKT = a.K / 64;
  const int ntiles = a.N / 16;
  if (NKT % KS != 0) return false;
  // X staging: 1024 threads x 8 floats x 4 passes
  if ((long)NB * a.K > 32768) return false;
  if (F32 && (long)NB * a.K * 4 > WSW_X_BYTES) return false;
  constexpr int RGmax = 16 / KS;
  const int rounds_full = wsw_ceil_div(ntiles, (long)a.cu * RGmax * NT);
  int wv = RGmax;
  for (int c = RGmax - 1; c >= 1; c--) {
    if (wsw_ceil_div(ntiles, (long)a.cu * c * NT) == rounds_full) wv = c; else break;
  }
  const int rounds = wsw_ceil_div(ntiles, (long)a.cu * wv * NT);
  dim3 grid(a.cu), block(64, 16);
  w8sw_gemv_kernel<NT, UNRL, NB, F32, KS><<<grid, block, 0, a.st>>>(
      a.K, a.N, a.W, a.S, a.X, a.C, wv, rounds, NKT, ntiles, a.gshift);
  return true;
}

#define WSW_CFG(NT_, UN_, KS_) \
  if (nt == NT_ && un == UN_ && ks == KS_) return wsw_launch_one<NT_, UN_, NB, true, KS_>(a);

template <int NB>
bool wsw_launch_nb(const WswArgs& a, int nt, int un, int ks) {
  // fp32 LDS staging only (all whitelisted shapes fit: NB*K*4 <= 57344)
  if ((long)NB * a.K * 4 > WSW_X_BYTES) return false;
  WSW_CFG(1, 1, 1) WSW_CFG(1, 2, 1) WSW_CFG(2, 1, 1) WSW_CFG(2, 2, 1)
  WSW_CFG(4, 1, 1) WSW_CFG(4, 2, 1)
  WSW_CFG(1, 1, 2) WSW_CFG(1, 2, 2) WSW_CFG(2, 1, 2) WSW_CFG(2, 2, 2)
  WSW_CFG(1, 1, 4) WSW_CFG(1, 2, 4) WSW_CFG(2, 1, 4) WSW_CFG(2, 2, 4)
  WSW_CFG(1, 1, 8) WSW_CFG(1, 2, 8) WSW_CFG(2, 1, 8)
  WSW_CFG(1, 4, 1) WSW_CFG(1, 4, 4)
  WSW_CFG(1, 1, 5) WSW_CFG(2, 1, 5) WSW_CFG(1, 1, 6) WSW_CFG(2, 1, 6)
  WSW_CFG(1, 1, 3) WSW_CFG(2, 1, 3) WSW_CFG(1, 1, 12) WSW_CFG(1, 2, 12)
  return false;
}
