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
  // LDS chunk: KCH activation columns per row per staging pass.  KCH == K is the single-pass case
  // (identical to the pre-chunking kernel); a smaller KCH is what carries M = 4..8 at K = 10240.
  // A multiple of KSTEP*UNRL keeps the per-lane k sequence -- and so the accumulation order --
  // exactly what it is without chunking.
  constexpr int STEP = LPR * 16 * UN;
  const int cap = F32 ? (HW8_X_BYTES / 4) : (HW8_X_BYTES / 2);
  int KCH = a.K;
  if (NB <= 3) {
    // pre-change contract: the whole activation must fit LDS in one staging pass
    if ((long)NB * a.K > cap) return false;
  } else if ((long)NB * a.K > cap) {
    if (KS != 1) return false;   // split-K + chunking would idle most waves per chunk
    KCH = (cap / NB / STEP) * STEP;
    if (KCH <= 0) return false;
  }
  // waves (row groups) per WG: smallest count that keeps the round count of the full WG (stock mindiv idea)
  const int rounds_full = hw8_ceil_div(a.Nrows, (long)a.cu * RGmax * RPW);
  int wv = RGmax;
  for (int c = RGmax - 1; c >= 1; c--) {
    if (hw8_ceil_div(a.Nrows, (long)a.cu * c * RPW) == rounds_full) wv = c; else break;
  }
  const int rounds = hw8_ceil_div(a.Nrows, (long)a.cu * wv * RPW);
  dim3 grid(a.cu), block(64, 16);
  // NB <= 3 launches the pre-change kernel with the pre-change argument list.  Sharing one kernel
  // via `if constexpr` was measured at +1.3..2.0 us per HC module at M = 1..3, so the two shapes
  // are two kernels.
  if constexpr (NB <= 3) {
    hc_w8_gemv_kernel<YT, UN, NB, GRP, F32, LPR, KS, EPI><<<grid, block, 0, a.st>>>(
        a.K, a.Nrows, a.W, a.S, a.X, a.C, wv, rounds, a.cu, a.G, a.gshift, a.XN, a.HD, a.R, a.HCn,
        a.Xs);
  } else {
    hc_w8_gemv_chunk_kernel<YT, UN, NB, GRP, F32, LPR, KS, EPI><<<grid, block, 0, a.st>>>(
        a.K, a.Nrows, a.W, a.S, a.X, a.C, wv, rounds, a.cu, a.G, a.gshift, a.XN, a.HD, a.R, a.HCn,
        a.Xs, KCH);
  }
  return true;
}

#define HW8_CFG(YT, UN, LPRv, KSv) \
  if (yt == YT && un == UN && lpr == LPRv && ks == KSv) return hw8_launch_one<YT, UN, NB, GRP, F32, LPRv, KSv, EPI>(a);

template <int NB, int EPI, bool GRP, bool F32>
static bool hw8_launch_cfg(const HW8Args& a, int yt, int un, int lpr, int ks) {
  if constexpr (EPI == 2) {
    // gate-mix: YTILE must equal HC (4); the 4 permuted rows of one hidden index live in one wave-step
    if constexpr (F32 && NB <= 3) { HW8_CFG(4, 1, 32, 1) HW8_CFG(4, 2, 32, 1) HW8_CFG(4, 1, 64, 1) }
    else if constexpr (F32) { HW8_CFG(4, 1, 32, 1) }
    return false;
  } else if constexpr (NB <= 3) {
    // long-K / few-row configs (mix_down 336 x 10240): plain + intra-WG split-K
    HW8_CFG(1, 4, 64, 1) HW8_CFG(2, 2, 64, 1)
    HW8_CFG(1, 2, 64, 2) HW8_CFG(2, 2, 64, 2)
    HW8_CFG(1, 3, 64, 4) HW8_CFG(2, 2, 64, 4)
    return false;
  } else if constexpr (EPI == 1) {
    // M = 4..8: KS == 1 only (chunked staging), and a trimmed config set -- every extra (NB, cfg)
    // pair is a kernel in the JIT build wall.
    HW8_CFG(1, 4, 64, 1) HW8_CFG(1, 2, 64, 1) HW8_CFG(2, 2, 64, 1)
    return false;
  }
  return false;   // EPI 0 above M = 3 is not instantiated (nothing calls it)
}

template <int NB, int EPI>
bool hw8_launch_nb(const HW8Args& a, int yt, int un, int lpr, int ks) {
  const bool f32 = (long)NB * a.K * 4 <= HW8_X_BYTES;
  if constexpr (NB > 3) {
    // per-output-channel scales (gs == 0) are not instantiated above M = 3; the HC mixes always
    // use group scales, and the caller falls back to the stock chain if this returns false.
    if (a.gs == 0) return false;
    return f32 ? hw8_launch_cfg<NB, EPI, true, true>(a, yt, un, lpr, ks)
               : hw8_launch_cfg<NB, EPI, true, false>(a, yt, un, lpr, ks);
  } else {
    if (a.gs == 0)
      return f32 ? hw8_launch_cfg<NB, EPI, false, true>(a, yt, un, lpr, ks)
                 : hw8_launch_cfg<NB, EPI, false, false>(a, yt, un, lpr, ks);
    return f32 ? hw8_launch_cfg<NB, EPI, true, true>(a, yt, un, lpr, ks)
               : hw8_launch_cfg<NB, EPI, true, false>(a, yt, un, lpr, ks);
  }
}
