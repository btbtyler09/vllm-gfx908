// gfx908 W4A8 GEMV kernels for small M (MoE experts, shared expert, dense projections).
// Microbench + design: docs/mi100_decode_opt/research/w4_gemv_kernels.md (D1, D2),
// agents/w4a8/REPORT.md (2026-09-04).
//
// Weights: [E, N, K/8] uint32, 8 nibbles along K (nibble t of word wd = k 8*wd+t),
// symmetric zero point 8; scales [E, N, K/32] bf16 (group size 32; one uint4 = one group).
//
// Activations (Q8_1 style, block 32, produced by quant_q8 or the Triton silu*mul+quant kernel):
//   x8   int8 [R, K]    per 8-element block DE-INTERLEAVED: bytes [k0 k2 k4 k6 | k1 k3 k5 k7], so the
//                       dword 2*wd holds the even k of word wd (pairs with q & 0x0F0F0F0F) and dword
//                       2*wd+1 the odd k (pairs with (q >> 4) & 0x0F0F0F0F). No weight shuffle needed.
//   xs   fp32 [R, K/32] block scale d = absmax/127
//   xsum fp32 [R, K/32] d * sum(q8)   -> sum(w x) = ws * (d * sumi - 8 * xsum)  (zero point folded)
//
// Output rows are pairs: out[p, :] = W[row_exp[p]] . x[row_tok[p]] as fp32 [P, N].
// row_exp[p] outside [0, E) -> zero row (cudagraph capture routing).
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

__device__ __forceinline__ float bf2f(uint32_t v) { return __uint_as_float(v << 16); }

__device__ __forceinline__ int dot4(uint32_t a, int b, int c) {
#ifdef W4A8_ASM_DOT4
  int r;
  asm("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
#else
  return __builtin_amdgcn_sdot4((int)a, b, c, false);
#endif
}

template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
  return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true));
}

// Sum over LPR consecutive lanes; result in lane LPR-1 of each group (row_shr prefix sums).
template <int LPR>
__device__ __forceinline__ float reduce_lpr(float v) {
  if constexpr (LPR >= 2) v = dpp_add<0x111>(v);   // row_shr:1
  if constexpr (LPR >= 4) v = dpp_add<0x112>(v);   // row_shr:2
  if constexpr (LPR >= 8) v = dpp_add<0x114>(v);   // row_shr:4
  if constexpr (LPR >= 16) v = dpp_add<0x118>(v);  // row_shr:8
  if constexpr (LPR >= 32) v += __shfl_xor(v, 16, 64);
  if constexpr (LPR >= 64) v += __shfl_xor(v, 32, 64);
  return v;
}

// ---------------------------------------------------------------------------------------------
// Activation quant: one thread per (row, group of 32). x bf16 [R, K] -> x8 / xs / xsum.
__global__ void __launch_bounds__(256) quant_q8_kernel(
    const uint32_t* __restrict__ x, uint4* __restrict__ x8, float* __restrict__ xs,
    float* __restrict__ xsum, int total) {
  const int gid = blockIdx.x * 256 + threadIdx.x;
  if (gid >= total) return;
  const uint4* xp = reinterpret_cast<const uint4*>(x) + (long)gid * 4;  // 64 B = 32 bf16
  uint4 v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = xp[i];
  float f[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t d[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      f[i * 8 + j * 2] = __uint_as_float(d[j] << 16);
      f[i * 8 + j * 2 + 1] = __uint_as_float(d[j] & 0xFFFF0000u);
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
  const float d = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int q[32], qsum = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) { q[i] = __float2int_rn(f[i] * inv); qsum += q[i]; }
  uint32_t o[8];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int* qb = q + b * 8;
    o[2 * b] = (qb[0] & 0xFF) | ((qb[2] & 0xFF) << 8) | ((qb[4] & 0xFF) << 16) | ((qb[6] & 0xFF) << 24);
    o[2 * b + 1] = (qb[1] & 0xFF) | ((qb[3] & 0xFF) << 8) | ((qb[5] & 0xFF) << 16) | ((qb[7] & 0xFF) << 24);
  }
  x8[(long)gid * 2] = make_uint4(o[0], o[1], o[2], o[3]);
  x8[(long)gid * 2 + 1] = make_uint4(o[4], o[5], o[6], o[7]);
  xs[gid] = d;
  xsum[gid] = d * (float)qsum;
}

// ---------------------------------------------------------------------------------------------
// D1 "K-slab": LPR lanes per output row, 64/LPR rows per wave, lane l owns groups l + LPR*j
// (one uint4 = one GS32 group = one scale). Every load of the wave is issued before the first
// use (sched_barrier); int8 dot4 per group; DPP row reduction; direct fp32 store, no split-K.
template <int K, int LPR, int WPB>
__global__ void __launch_bounds__(64 * WPB) w4a8_slab_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ row_exp, float* __restrict__ out, int N, int P, int E, long stride_we,
    long stride_se) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR;
  const int tiles_per_p = N / ROWS;
  // readfirstlane: the tile (and so p, e, t) is wave-uniform -> s_load for the ids.
  const int tile = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  if (tile >= P * tiles_per_p) return;
  const int p = tile / tiles_per_p, n = (tile % tiles_per_p) * ROWS + r;
  const int e = row_exp[p], t = row_tok[p];
  if (e < 0 || e >= E) {
    if (l == LPR - 1) out[(long)p * N + n] = 0.f;
    return;
  }
  const uint4* wp = reinterpret_cast<const uint4*>(w + e * stride_we + (long)n * (K / 8)) + l;
  const uint16_t* sp = s + e * stride_se + (long)n * G + l;
  const uint4* xp = reinterpret_cast<const uint4*>(x8 + (long)t * K) + 2 * l;
  const float* xsp = xs + (long)t * G + l;
  const float* xsump = xsum + (long)t * G + l;
  uint4 q[GPL], xa[GPL], xb[GPL];
  uint32_t sc[GPL];
  float sx[GPL], sxs[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) { q[j] = wp[LPR * j]; sc[j] = sp[LPR * j]; }
    else { q[j] = make_uint4(0, 0, 0, 0); sc[j] = 0; }
  }
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    if (G % LPR == 0 || l + LPR * j < G) {
      xa[j] = xp[2 * LPR * j]; xb[j] = xp[2 * LPR * j + 1];
      sx[j] = xsp[LPR * j]; sxs[j] = xsump[LPR * j];
    } else { xa[j] = make_uint4(0, 0, 0, 0); xb[j] = xa[j]; sx[j] = 0.f; sxs[j] = 0.f; }
  }
  __builtin_amdgcn_sched_barrier(0);   // all loads above the first use
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    const int xd[8] = {(int)xa[j].x, (int)xa[j].y, (int)xa[j].z, (int)xa[j].w,
                       (int)xb[j].x, (int)xb[j].y, (int)xb[j].z, (int)xb[j].w};
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
      sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
    }
    acc += bf2f(sc[j]) * (sx[j] * (float)sumi - 8.f * sxs[j]);
  }
  acc = reduce_lpr<LPR>(acc);
  if (l == LPR - 1) out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// D2: thread-per-row for K=160 (5 groups = 80 B per row). One pair per wave; the pair's
// activation (40 dwords + 10 floats) is wave-uniform and lives in SGPRs.
template <int K, int WPB>
__global__ void __launch_bounds__(64 * WPB) w4a8_rowlane_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ row_exp, float* __restrict__ out, int N, int P, int E, long stride_we,
    long stride_se) {
  constexpr int G = K / 32;
  const int lane = threadIdx.x & 63;
  const int waves_per_p = N / 64;
  const int wave = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  if (wave >= P * waves_per_p) return;
  const int p = wave / waves_per_p, n = (wave % waves_per_p) * 64 + lane;
  const int e = row_exp[p], t = row_tok[p];
  if (e < 0 || e >= E) { out[(long)p * N + n] = 0.f; return; }
  const uint4* wp = reinterpret_cast<const uint4*>(w + e * stride_we + (long)n * (K / 8));
  const uint16_t* sp = s + e * stride_se + (long)n * G;
  uint4 q[G];
  uint32_t sc[G];
#pragma unroll
  for (int j = 0; j < G; ++j) { q[j] = wp[j]; sc[j] = sp[j]; }
  const int* xu = reinterpret_cast<const int*>(x8 + (long)t * K);  // wave-uniform -> SGPRs
  const float* xsu = xs + (long)t * G;
  const float* xsumu = xsum + (long)t * G;
  int xv[8 * G]; float xsv[G], xsumv[G];
#pragma unroll
  for (int i = 0; i < 8 * G; ++i) xv[i] = xu[i];
#pragma unroll
  for (int j = 0; j < G; ++j) { xsv[j] = xsu[j]; xsumv[j] = xsumu[j]; }
  __builtin_amdgcn_sched_barrier(0);
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < G; ++j) {
    const uint32_t qw[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
    int sumi = 0;
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xv[8 * j + 2 * wd], sumi);
      sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xv[8 * j + 2 * wd + 1], sumi);
    }
    acc += bf2f(sc[j]) * (xsv[j] * (float)sumi - 8.f * xsumv[j]);
  }
  out[(long)p * N + n] = acc;
}

// ---------------------------------------------------------------------------------------------
// Host wrappers
static void check_act(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& xsum, int K) {
  TORCH_CHECK(x8.scalar_type() == at::kChar && x8.is_contiguous() && x8.size(1) == K);
  TORCH_CHECK(xs.scalar_type() == at::kFloat && xsum.scalar_type() == at::kFloat);
  TORCH_CHECK(xs.is_contiguous() && xsum.is_contiguous() && xs.size(1) == K / 32);
}

// x bf16 [R, K] (K % 32 == 0) -> x8 int8 [R, K], xs fp32 [R, K/32], xsum fp32 [R, K/32]
void quant_q8(at::Tensor x, at::Tensor x8, at::Tensor xs, at::Tensor xsum) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  const int R = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 32 == 0);
  check_act(x8, xs, xsum, K);
  const int total = R * (K / 32);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (total > 0) {
    quant_q8_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(x.data_ptr()), reinterpret_cast<uint4*>(x8.data_ptr()),
        xs.data_ptr<float>(), xsum.data_ptr<float>(), total);
  }
}

static void check_gemv(const at::Tensor& w, const at::Tensor& s, const at::Tensor& row_tok,
                       const at::Tensor& row_exp, const at::Tensor& out, int K) {
  TORCH_CHECK(w.is_contiguous() && s.is_contiguous() && s.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w.scalar_type() == at::kInt || w.scalar_type() == at::kByte);
  TORCH_CHECK(row_tok.scalar_type() == at::kInt && row_exp.scalar_type() == at::kInt);
  TORCH_CHECK(row_tok.is_contiguous() && row_exp.is_contiguous() && row_tok.numel() == row_exp.numel());
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous());
  TORCH_CHECK(w.dim() == 3 && s.dim() == 3 && s.size(1) == w.size(1) && s.size(2) == K / 32);
  TORCH_CHECK(out.size(0) == row_tok.numel() && out.size(1) == w.size(1));
}

template <int K, int LPR, int WPB>
static void launch_slab(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x8,
                        const at::Tensor& xs, const at::Tensor& xsum, const at::Tensor& row_tok,
                        const at::Tensor& row_exp, at::Tensor& out) {
  constexpr int ROWS = 64 / LPR;
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0, "gemv_slab: N must be a multiple of ", ROWS);
  const long tiles = (long)P * (N / ROWS);
  const int blocks = (int)((tiles + WPB - 1) / WPB);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (blocks > 0) {
    w4a8_slab_kernel<K, LPR, WPB><<<blocks, 64 * WPB, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
        reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(),
        row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E,
        (long)N * (K / 8), (long)N * (K / 32));
  }
}

// D1. w [E, N, K/8] int32 (or uint8 [E, N, K/2] viewed), s [E, N, K/32] bf16, out fp32 [P, N].
// K in {2560, 1536, 160}; lpr = 16; wpb (waves per block) in {1, 4}.
void gemv_slab(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
               at::Tensor row_tok, at::Tensor row_exp, at::Tensor out, int64_t lpr, int64_t wpb) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) * 8 == K) || (w.scalar_type() == at::kByte && w.size(2) * 2 == K),
              "gemv_slab: w K mismatch");
#define SLAB_CASE(KK, L, W) \
  if (K == KK && lpr == L && wpb == W) { launch_slab<KK, L, W>(w, s, x8, xs, xsum, row_tok, row_exp, out); return; }
  SLAB_CASE(2560, 16, 4) SLAB_CASE(2560, 16, 1)
  SLAB_CASE(1536, 16, 4) SLAB_CASE(1536, 16, 1)
  SLAB_CASE(160, 16, 4) SLAB_CASE(160, 16, 1)
#undef SLAB_CASE
  TORCH_CHECK(false, "gemv_slab: unsupported K/lpr/wpb ", K, "/", lpr, "/", wpb);
}

// D2. K == 160 only; N % 64 == 0; out fp32 [P, N]; wpb in {1, 4}.
void gemv_rowlane(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
                  at::Tensor row_tok, at::Tensor row_exp, at::Tensor out, int64_t wpb) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  check_gemv(w, s, row_tok, row_exp, out, K);
  TORCH_CHECK(K == 160, "gemv_rowlane: K must be 160");
  TORCH_CHECK((w.scalar_type() == at::kInt && w.size(2) == 20) || (w.scalar_type() == at::kByte && w.size(2) == 80));
  const int P = row_tok.numel(), N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % 64 == 0, "gemv_rowlane: N % 64 != 0");
  const long waves = (long)P * (N / 64);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  if (waves == 0) return;
#define ROWLANE_CASE(W) \
  if (wpb == W) { \
    w4a8_rowlane_kernel<160, W><<<(waves + W - 1) / W, 64 * W, 0, stream>>>( \
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()), \
        reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(), \
        row_tok.data_ptr<int32_t>(), row_exp.data_ptr<int32_t>(), out.data_ptr<float>(), N, P, E, \
        (long)N * (160 / 8), (long)N * 5); \
    return; \
  }
  ROWLANE_CASE(1) ROWLANE_CASE(4)
#undef ROWLANE_CASE
  TORCH_CHECK(false, "gemv_rowlane: wpb must be 1 or 4");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_q8", &quant_q8, "x bf16 [R,K] -> x8 int8 [R,K] (per-8 de-interleaved), xs/xsum fp32 [R,K/32]");
  m.def("gemv_slab", &gemv_slab, "w, s, x8, xs, xsum, row_tok, row_exp, out fp32 [P,N], lpr(16), wpb(1|4)");
  m.def("gemv_rowlane", &gemv_rowlane, "w, s, x8, xs, xsum, row_tok, row_exp, out fp32 [P,N], wpb(1|4)  (K=160)");
}
