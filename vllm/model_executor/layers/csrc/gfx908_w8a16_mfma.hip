// SPDX-License-Identifier: Apache-2.0
// gfx908 / MI100 hand-written W8A16 MFMA GEMM (design D2), 1 <= M <= 64.
//
// Bound into the gfx908_w8a16 extension; opt-in from Python via
// VLLM_GFX908_W8A16_MFMA=1.  Microbenchmarks and the derivation are in
// docs/mi100_decode_opt/research/ / the agent report (agents/mfma_w8/REPORT.md).
//
//   C[M, N] (bf16) = X[M, K] (bf16) @ dequant(Wq[N, K] int8, S[N, K/128] fp32)^T
//
// Weights are pre-swizzled offline into MFMA-B fragments so the inner loop is a pure
//   buffer_load_dwordx4 -> v_cvt_f32_i32(sdwa) -> v_perm_b32 -> v_mfma_*bf16
// stream: B never touches LDS and there are no cross-lane shuffles.
//
// Two tile shapes share one code path (TILE = the MFMA's M == N):
//   TILE = 16 -> v_mfma_f32_16x16x8bf16, weight tile = 16 n x 64 k = 1 KiB, 4 acc VGPRs
//   TILE = 32 -> v_mfma_f32_32x32x4bf16, weight tile = 32 n x 32 k = 1 KiB, 16 acc VGPRs
// In both cases lane l of the wave64 owns the 16 B at tile_base + 16*l, which is
//   n = n_tile*TILE + l%TILE ,  k = k_tile*KPT + 16*(l/TILE) + 0..15   (KPT = 1024/TILE)
// and MFMA step t (t = 0..7) consumes the lane's operand pair (2t, 2t+1).  The K order
// inside a tile is permuted relative to memory; that is legal because A uses the same
// lane rule, so A and B are permuted identically.
//
// Swizzles (see swizzle_w8()/swizzle_s() in bench.py):
//   TILE=16: Wsw = Wq.view(N/16, 16, K/64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()
//   TILE=32: Wsw = Wq.view(N/32, 32, K/32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()
//   scales : Ssw = S.t().contiguous()            -> [K/128, N], tile-shape independent
//
// Register layout constants verified on-device (probe_layout.hip):
//   src0 = A (rows of D), src1 = B (cols of D), D col j = lane%TILE,
//   D row i = (reg%4) + 4*(lane/32) + 8*(reg/4)   for 32x32x4bf16
//   D row i = reg + 4*(lane/16)                   for 16x16x8bf16
//
// int8 -> bf16 is EXACT: |v| <= 127 needs <= 8 significant bits, v_cvt_f32_i32 is exact
// and the fp32 result's low 16 mantissa bits are zero, so keeping the high half
// (v_perm_b32) loses nothing.  Group scales multiply an fp32 partial accumulator once
// per 128-k group, so the MFMA chain itself stays scale-free and fp32-accumulated.
//
// Split-K lives *inside* the workgroup: the SK waves of a block own disjoint k-tile
// runs over the same n-tiles and reduce through LDS (ds_add_f32) -> a single bf16
// store, no fp32 partial traffic and no second launch.
//
// X is read through a buffer resource whose num_records is exactly M*K*2, so the
// padding rows of the M-tile (m >= M) return hardware zeros with no branch and no
// memory traffic.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

typedef short s16x2 __attribute__((ext_vector_type(2)));
typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef int i32x4 __attribute__((ext_vector_type(4)));

#define DEVI __device__ __forceinline__
static constexpr unsigned BUF_DWORD3 = 0x00020000u;  // GFX9 buffer descriptor word 3

// 4 packed int8 -> 2 dwords, each holding 2 bf16 (low half = lower k).
DEVI void i8x4_to_bf16(unsigned int w, unsigned int& o0, unsigned int& o1) {
  const int b0 = (int)(signed char)(w & 0xffu);
  const int b1 = (int)(signed char)((w >> 8) & 0xffu);
  const int b2 = (int)(signed char)((w >> 16) & 0xffu);
  const int b3 = (int)(signed char)((w >> 24) & 0xffu);
  const unsigned int f0 = __float_as_uint((float)b0);
  const unsigned int f1 = __float_as_uint((float)b1);
  const unsigned int f2 = __float_as_uint((float)b2);
  const unsigned int f3 = __float_as_uint((float)b3);
  o0 = __builtin_amdgcn_perm(f1, f0, 0x07060302u);  // {bf16(b1), bf16(b0)}
  o1 = __builtin_amdgcn_perm(f3, f2, 0x07060302u);
}

DEVI unsigned short f2bf16_rne(float f) {
  const unsigned int u = __float_as_uint(f);
  const unsigned int r = ((u >> 16) & 1u) + 0x7fffu;
  return (unsigned short)((u + r) >> 16);
}

template <int TILE>
struct Mfma;
template <>
struct Mfma<32> {
  using acc_t = f32x16;
  static constexpr int NACC = 16;
  DEVI static acc_t zero() {
    return (f32x16){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  }
  DEVI static acc_t op(s16x2 a, s16x2 b, acc_t c) {
    return __builtin_amdgcn_mfma_f32_32x32x4bf16(a, b, c, 0, 0, 0);
  }
  DEVI static int row(int r, int hi) { return (r & 3) + 4 * hi + 8 * (r >> 2); }
};
template <>
struct Mfma<16> {
  using acc_t = f32x4;
  static constexpr int NACC = 4;
  DEVI static acc_t zero() { return (f32x4){0.f, 0.f, 0.f, 0.f}; }
  DEVI static acc_t op(s16x2 a, s16x2 b, acc_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x8bf16(a, b, c, 0, 0, 0);
  }
  DEVI static int row(int r, int hi) { return r + 4 * hi; }
};

template <int AUX>
DEVI i32x4 bload(__amdgpu_buffer_rsrc_t r, int voff) {
  return __builtin_amdgcn_raw_buffer_load_b128(r, voff, 0, AUX);
}

// ---------------------------------------------------------------------------
// TILE : MFMA tile (16 or 32); n-tile width and m-tile height
// MT   : m-tiles per wave (M_pad = TILE*MT)
// NW   : n-tiles per wave
// UG   : scale groups per pipeline stage (UG*GKT k-tiles = UG KiB of weights in flight
//        per n-tile per stage; the loop always double-buffers one stage ahead)
// ---------------------------------------------------------------------------
template <int TILE, int MT, int NW, int UG>
__global__ void __launch_bounds__(512, 1) w8mfma_kernel(
    const int8_t* __restrict__ Wsw, const float* __restrict__ Ssw,
    const unsigned short* __restrict__ X, unsigned short* __restrict__ C,
    const int M, const int N, const int K, const int NKT, const int NG, const int SK,
    const int ntiles, const int xrec) {
  using MM = Mfma<TILE>;
  using acc_t = typename MM::acc_t;
  constexpr int KPT = 1024 / TILE;  // k per weight tile
  constexpr int NACC = MM::NACC;
  constexpr int GKT = 128 / KPT;    // k-tiles per 128-k scale group

  extern __shared__ float red[];
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wave = tid >> 6;
  const int lo = lane % TILE;  // n inside the tile == D column
  const int hi = lane / TILE;  // which 16-k slot of the tile this lane owns
  const int nt0 = blockIdx.x * NW;

  // scale-group slice of this wave (host guarantees NG % SK == 0)
  const int perg = NG / SK;
  const int g0 = wave * perg;
  const int g1 = g0 + perg;

  __amdgpu_buffer_rsrc_t wr =
      __builtin_amdgcn_make_buffer_rsrc((void*)Wsw, 0, (int)((size_t)N * K), BUF_DWORD3);
  __amdgpu_buffer_rsrc_t xr =
      __builtin_amdgcn_make_buffer_rsrc((void*)X, 0, xrec, BUF_DWORD3);

  float acc[MT][NW][NACC];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nw = 0; nw < NW; ++nw)
#pragma unroll
      for (int r = 0; r < NACC; ++r) acc[mt][nw][r] = 0.f;

  int xoff[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
    xoff[mt] = (mt * TILE + lo) * K * 2 + g0 * GKT * KPT * 2 + hi * 32;
  int woff[NW];
#pragma unroll
  for (int nw = 0; nw < NW; ++nw)
    woff[nw] = ((nt0 + nw) * NKT + g0 * GKT) * 1024 + lane * 16;
  const size_t soff = (size_t)nt0 * TILE + lo;  // scale column of this lane

  constexpr int CH = UG * GKT;           // k-tiles per pipeline stage
  constexpr int wstep = CH * 1024;       // bytes per stage per n-tile
  constexpr int xstep = CH * KPT * 2;    // bytes per stage per row

// Prefetches are unconditional: when there is no next stage the byte offset is set
// past the buffer's num_records, so the hardware returns zeros with no memory traffic
// and no branch -- a branch here would force the compiler to use s_waitcnt vmcnt(0)
// (unknown number of loads in flight), which destroys the software pipeline.
static constexpr int OOB = 0x20000000;

#define LOADW(dst, sdst, boff, gbase)                                           \
  {                                                                             \
    _Pragma("unroll") for (int kk = 0; kk < CH; ++kk)                           \
        _Pragma("unroll") for (int nw = 0; nw < NW; ++nw) dst[kk][nw] =         \
            bload<3>(wr, woff[nw] + (boff) + kk * 1024);                        \
    _Pragma("unroll") for (int ug = 0; ug < UG; ++ug)                           \
        _Pragma("unroll") for (int nw = 0; nw < NW; ++nw) sdst[ug][nw] =        \
            Ssw[soff + (size_t)((gbase) + ug) * N + nw * TILE];                 \
  }

// X for a whole stage is issued *before* the next stage's weight prefetch, so the
// s_waitcnt that consumes X leaves the prefetched weights outstanding (issuing the X
// loads inside the MFMA loop instead forces a vmcnt(0) drain every k-tile).
#define LOADX(dst, boff)                                                        \
  {                                                                             \
    _Pragma("unroll") for (int kc = 0; kc < CH; ++kc)                           \
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) {                     \
      const int xb = xoff[mt] + (boff) + kc * (KPT * 2);                        \
      dst[kc][mt][0] = bload<0>(xr, xb);                                        \
      dst[kc][mt][1] = bload<0>(xr, xb + 16);                                   \
    }                                                                           \
  }

#define COMPUTE(src, xsrc, ssrc)                                               \
  {                                                                             \
    _Pragma("unroll") for (int ug = 0; ug < UG; ++ug) {                         \
      acc_t tmp[MT][NW];                                                        \
      _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)                         \
          _Pragma("unroll") for (int nw = 0; nw < NW; ++nw) tmp[mt][nw] =       \
              MM::zero();                                                       \
      _Pragma("unroll") for (int kk = 0; kk < GKT; ++kk) {                      \
        const int kc = ug * GKT + kk;                                           \
        unsigned int wb[NW][8];                                                 \
        _Pragma("unroll") for (int nw = 0; nw < NW; ++nw) {                     \
          const unsigned int* wd = (const unsigned int*)&src[kc][nw];           \
          _Pragma("unroll") for (int d = 0; d < 4; ++d)                         \
              i8x4_to_bf16(wd[d], wb[nw][2 * d], wb[nw][2 * d + 1]);            \
        }                                                                       \
        _Pragma("unroll") for (int t = 0; t < 8; ++t) {                         \
          _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) {                   \
            const unsigned int* ad = (const unsigned int*)&xsrc[kc][mt][t >> 2]; \
            const s16x2 av = *(const s16x2*)&ad[t & 3];                         \
            _Pragma("unroll") for (int nw = 0; nw < NW; ++nw)                   \
                tmp[mt][nw] = MM::op(av, *(const s16x2*)&wb[nw][t], tmp[mt][nw]); \
          }                                                                     \
        }                                                                       \
      }                                                                         \
      _Pragma("unroll") for (int nw = 0; nw < NW; ++nw) {                       \
        const float sc = ssrc[ug][nw];                                          \
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)                       \
            _Pragma("unroll") for (int r = 0; r < NACC; ++r)                    \
                acc[mt][nw][r] += sc * tmp[mt][nw][r];                          \
      }                                                                         \
    }                                                                           \
  }

  {
    const int nstage = (g1 - g0) / UG;
    i32x4 wa[CH][NW];
    i32x4 wb2[CH][NW];
    i32x4 xa[CH][MT][2];
    i32x4 xb2[CH][MT][2];
    float sa[UG][NW];
    float sb[UG][NW];
    LOADW(wa, sa, 0, g0);
    LOADX(xa, 0);
    int st = 0;
    while (st < nstage) {
      int nb = (st + 1 < nstage) ? (st + 1) * wstep : OOB;
      int nx = (st + 1 < nstage) ? (st + 1) * xstep : OOB;
      int ng = (st + 1 < nstage) ? (g0 + (st + 1) * UG) : g0;
      LOADX(xb2, nx);
      LOADW(wb2, sb, nb, ng);
      COMPUTE(wa, xa, sa);
      ++st;
      if (st >= nstage) break;
      nb = (st + 1 < nstage) ? (st + 1) * wstep : OOB;
      nx = (st + 1 < nstage) ? (st + 1) * xstep : OOB;
      ng = (st + 1 < nstage) ? (g0 + (st + 1) * UG) : g0;
      LOADX(xa, nx);
      LOADW(wa, sa, nb, ng);
      COMPUTE(wb2, xb2, sb);
      ++st;
    }
  }
#undef LOADW
#undef COMPUTE

  // ---------------- reduce the SK k-slices, then store ----------------
  if (SK == 1) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nw = 0; nw < NW; ++nw) {
        const int n = (nt0 + nw) * TILE + lo;
#pragma unroll
        for (int r = 0; r < NACC; ++r) {
          const int m = mt * TILE + MM::row(r, hi);
          if (m < M) C[(size_t)m * N + n] = f2bf16_rne(acc[mt][nw][r]);
        }
      }
    return;
  }

  // Deterministic split-K reduce: every wave writes its own LDS slice, then the
  // block sums the SK slices in a fixed order.  (ds_add_f32 atomics would be
  // cheaper in LDS but make the result depend on wave arrival order, i.e. the
  // same inputs would not always give the same bits across cudagraph replays.)
  constexpr int TOT = MT * NW * TILE * TILE;
  float* mine = red + (size_t)wave * TOT;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nw = 0; nw < NW; ++nw)
#pragma unroll
      for (int r = 0; r < NACC; ++r)
        mine[((mt * NW + nw) * TILE + MM::row(r, hi)) * TILE + lo] = acc[mt][nw][r];
  __syncthreads();
  for (int i = tid; i < TOT; i += blockDim.x) {
    float v = red[i];
    for (int w = 1; w < SK; ++w) v += red[(size_t)w * TOT + i];
    const int j = i % TILE;
    const int row = (i / TILE) % TILE;
    const int nw = (i / (TILE * TILE)) % NW;
    const int mt = i / (TILE * TILE * NW);
    const int m = mt * TILE + row;
    if (m < M) C[(size_t)m * N + (nt0 + nw) * TILE + j] = f2bf16_rne(v);
  }
}

// ---------------------------------------------------------------------------
#define LAUNCH(T_, MT_, NW_, UG_)                                                \
  {                                                                              \
    hipLaunchKernelGGL((w8mfma_kernel<T_, MT_, NW_, UG_>), dim3(grid),           \
                       dim3(64 * sk), shmem, stream, Wp, Sp, Xp, Cp, M, N, K,    \
                       NKT, NG, sk, ntiles, xrec);                               \
  }

#define D_UG(T_, MT_, NW_)                        \
  {                                               \
    if (ug == 1) { LAUNCH(T_, MT_, NW_, 1) }      \
    else { LAUNCH(T_, MT_, NW_, 2) }              \
  }
#define D_NW(T_, MT_)                        \
  {                                          \
    if (nw == 1) { D_UG(T_, MT_, 1) }        \
    else if (nw == 2) { D_UG(T_, MT_, 2) }   \
    else { D_UG(T_, MT_, 4) }                \
  }

bool mfma_w8_gemm(at::Tensor Wsw, at::Tensor Ssw, at::Tensor X, at::Tensor C,
                  int64_t tile_, int64_t nw_, int64_t sk_, int64_t ug_, int64_t nta_) {
  const int M = (int)X.size(0);
  const int K = (int)X.size(1);
  const int N = (int)C.size(1);
  TORCH_CHECK(X.dtype() == torch::kBFloat16 && X.is_contiguous());
  TORCH_CHECK(C.dtype() == torch::kBFloat16 && C.is_contiguous());
  TORCH_CHECK(Wsw.dtype() == torch::kInt8 && Wsw.is_contiguous());
  TORCH_CHECK(Ssw.dtype() == torch::kFloat32 && Ssw.is_contiguous());
  const int tile = (int)tile_;
  if (tile != 16 && tile != 32) return false;
  if (K % 128 != 0 || N % tile != 0 || M < 1 || M > 64) return false;
  const int KPT = 1024 / tile, GKT = 128 / KPT;
  const int NKT = K / KPT;
  const int NG = K / 128;
  const int ntiles = N / tile;
  const int MT = (M + tile - 1) / tile;
  const int nw = (int)nw_;
  const int sk = (int)sk_;
  const int ug = (int)ug_;
  const int xrec = (nta_ & 2) ? 0 : M * K * 2;  // bit1: debug, zero the X stream
  if (nw != 1 && nw != 2 && nw != 4) return false;
  if (ug != 1 && ug != 2) return false;
  if (sk < 1 || sk > 8) return false;
  if (NG % sk || (NG / sk) % ug) return false;
  (void)KPT;
  if (ntiles % nw) return false;
  if (tile == 32 && MT > 2) return false;
  const int grid = ntiles / nw;
  const size_t shmem =
      (sk == 1) ? 0 : (size_t)sk * MT * nw * tile * tile * sizeof(float);
  if (shmem > 64 * 1024) return false;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const int8_t* Wp = (const int8_t*)Wsw.data_ptr();
  const float* Sp = (const float*)Ssw.data_ptr();
  const unsigned short* Xp = (const unsigned short*)X.data_ptr();
  unsigned short* Cp = (unsigned short*)C.data_ptr();

  if (tile == 16) {
    if (MT == 1) { D_NW(16, 1) }
    else if (MT == 2) { D_NW(16, 2) }
    else if (MT == 3) { D_NW(16, 3) }
    else { D_NW(16, 4) }
  } else {
#ifdef GFX908_W8A16_MFMA_TILE32
    if (MT == 1) { D_NW(32, 1) }
    else { D_NW(32, 2) }
#else
    return false;  // 32x32x4bf16 never won the sweep; not instantiated (build wall)
#endif
  }
  return true;
}


// ---------------------------------------------------------------------------
// Swizzled int8 -> row-major bf16, for the M > 64 (prefill) fallback: the
// swizzled tensor is the only resident int8 copy, so `_dequant_gemm` has to
// rematerialise from it rather than from a row-major [N, K] buffer.
//
//   mfma_w8_dequant(Wsw int8 (swizzled, rows [r0, r0+R) of the full [N, K]),
//                   Ssw fp32 [K/128, N], O bf16 [R, K], r0, N)
//
// One thread per 16 contiguous k of one row: reads the 16 B a wave lane would
// read (contiguous), writes 32 B (contiguous).  Rounding is RNE on the fp32
// product, i.e. bit-identical to gfx908_w8a16_dequant on the same values.
__global__ void __launch_bounds__(256, 1)
mfma_w8_dequant_kernel(const int8_t* __restrict__ Wsw, const float* __restrict__ Ssw,
                       unsigned short* __restrict__ O, int K, int N, int r0,
                       long total, int nkt) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  for (; i < total; i += stride) {
    const int lane = (int)(i & 63);          // chunk inside the 1 KiB tile
    const long t = i >> 6;                   // tile index
    const int kt = (int)(t % nkt);
    const int nt = (int)(t / nkt);
    const int n = nt * 16 + (lane & 15);
    const int k0 = kt * 64 + (lane >> 4) * 16;
    const float sc = Ssw[(size_t)(k0 >> 7) * N + r0 + n];
    const i32x4 qv = *reinterpret_cast<const i32x4*>(Wsw + (t << 10) + lane * 16);
    const signed char* qb = reinterpret_cast<const signed char*>(&qv);
    unsigned short h[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = f2bf16_rne((float)(int)qb[j] * sc);
    i32x4* dst = reinterpret_cast<i32x4*>(O + (size_t)n * K + k0);
    dst[0] = *reinterpret_cast<const i32x4*>(h);
    dst[1] = *reinterpret_cast<const i32x4*>(h + 8);
  }
}

bool mfma_w8_dequant(at::Tensor Wsw, at::Tensor Ssw, at::Tensor O, int64_t r0, int64_t Nfull) {
  const int R = (int)O.size(0), K = (int)O.size(1);
  TORCH_CHECK(Wsw.dtype() == torch::kInt8 && Wsw.is_contiguous(), "Wsw contiguous int8");
  TORCH_CHECK(Ssw.dtype() == torch::kFloat32 && Ssw.is_contiguous(), "Ssw contiguous fp32");
  TORCH_CHECK(O.dtype() == torch::kBFloat16 && O.is_contiguous(), "O contiguous bf16");
  if (R % 16 != 0 || K % 128 != 0) return false;
  if (Wsw.numel() != (long)R * K) return false;
  const int nkt = K / 64;
  const long total = (long)(R / 16) * nkt * 64;
  if (total == 0) return true;
  const int block = 256;
  long need = (total + block - 1) / block;
  const int grid = (int)(need < 4096L ? need : 4096L);
  {
    mfma_w8_dequant_kernel<<<dim3(grid), dim3(block), 0,
                             c10::hip::getCurrentHIPStream().stream()>>>(
        reinterpret_cast<const int8_t*>(Wsw.data_ptr()),
        reinterpret_cast<const float*>(Ssw.data_ptr()),
        reinterpret_cast<unsigned short*>(O.data_ptr()), K, (int)Nfull, (int)r0, total, nkt);
  }
  return true;
}
