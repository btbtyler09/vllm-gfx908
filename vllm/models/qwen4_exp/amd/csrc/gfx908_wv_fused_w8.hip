// SPDX-License-Identifier: Apache-2.0
// gfx908 fused W8 hyper-connection GEMVs (torch binding).
//
//   hc_w8_gemv(W, S, X, out, xn, epi, gs, ytile, unrl, lpr, ks, hc, r, cu) -> bool
//     W    int8  [Nrows, K]  contiguous, K % 16 == 0
//     S    fp32  [Nrows]              (gs == 0, per output channel)
//          fp32  [Nrows, K/gs]        (gs a power of two >= 16 dividing K and K/ks)
//     X    bf16  [M, K]      unit inner stride (row stride a multiple of 8), 1 <= M <= 3
//     out  bf16  [M, Nrows]  (epi 0/1)   or  [M, HD] (epi 2)
//     xn   bf16  [M, HC*HD]  (epi 2; ignored otherwise - pass any bf16 tensor)
//     epi  0 plain / 1 mix_down silu on rows < r / 2 mix_up gate-mix (needs ytile == hc)
//   returns false when the (ytile, unrl, lpr, ks) config is not instantiated or not applicable
//   to this shape; nothing is launched in that case.
//
//   hc_w8_dequant(W int8 [R,K], S fp32, O bf16 [R,K], gs) -> bool
//     RNE dequantization into a scratch buffer for the M > 3 fallback (bit-identical to
//     (W.float() * S).to(torch.bfloat16)).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include "gfx908_wv_fused_w8_launch.cuh"

extern template bool hw8_launch_nb<1, 1>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<2, 1>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<3, 1>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<1, 2>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<2, 2>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<3, 2>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<1, 0>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<2, 0>(const HW8Args&, int, int, int, int);
extern template bool hw8_launch_nb<3, 0>(const HW8Args&, int, int, int, int);

bool hc_w8_gemv(at::Tensor W, at::Tensor S, at::Tensor X, at::Tensor out, at::Tensor xn,
                int64_t epi, int64_t gs, int64_t ytile, int64_t unrl, int64_t lpr, int64_t ks,
                int64_t hc, int64_t r, int64_t cu) {
  const int Nrows = (int)W.size(0), K = (int)W.size(1), NB = (int)X.size(0);
  TORCH_CHECK(W.dim() == 2 && W.dtype() == torch::kInt8 && W.is_contiguous(),
              "W must be contiguous int8 [Nrows, K]");
  TORCH_CHECK(S.dtype() == torch::kFloat32 && S.is_contiguous(), "S must be contiguous fp32");
  TORCH_CHECK(X.dim() == 2 && X.dtype() == torch::kBFloat16 && X.stride(1) == 1 && X.size(1) == K,
              "X must be bf16 [M, K] with unit inner stride");
  TORCH_CHECK(X.size(0) == 1 || X.stride(0) % 8 == 0, "X row stride must be a multiple of 8");
  TORCH_CHECK(out.dim() == 2 && out.dtype() == torch::kBFloat16 && out.is_contiguous() &&
                  out.size(0) == NB,
              "out must be contiguous bf16 [M, *]");
  TORCH_CHECK(K % 16 == 0, "K % 16 == 0");
  TORCH_CHECK(NB >= 1 && NB <= 3, "M (batch) must be 1..3");
  int G = 1, gshift = 0;
  if (gs == 0) {
    TORCH_CHECK(S.numel() == Nrows, "S must be [Nrows] for gs == 0");
  } else {
    TORCH_CHECK(gs >= 16 && (gs & (gs - 1)) == 0 && K % gs == 0,
                "gs must be a power of two >= 16 dividing K");
    G = K / (int)gs;
    while ((1 << gshift) < gs) gshift++;
    TORCH_CHECK(S.numel() == (long)Nrows * G, "S must be [Nrows, K/gs]");
  }
  int HD = 0;
  if (epi == 2) {
    HD = (int)out.size(1);
    TORCH_CHECK(xn.dtype() == torch::kBFloat16 && xn.is_contiguous() && xn.size(0) == NB &&
                    xn.size(1) == (long)hc * HD && Nrows == (int)hc * HD,
                "epi 2: xn must be bf16 [M, HC*HD] and W must have HC*HD rows");
  } else {
    TORCH_CHECK(out.size(1) == Nrows, "epi 0/1: out must be [M, Nrows]");
  }
  HW8Args a{K,
            Nrows,
            reinterpret_cast<const int8_t*>(W.data_ptr()),
            reinterpret_cast<const float*>(S.data_ptr()),
            reinterpret_cast<const uint16_t*>(X.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            (int)cu,
            c10::hip::getCurrentHIPStream().stream(),
            G,
            gshift,
            (int)gs,
            reinterpret_cast<const uint16_t*>(xn.data_ptr()),
            HD,
            (int)r,
            (int)hc,
            (int)X.stride(0)};
  const int yt = (int)ytile, un = (int)unrl, lp = (int)lpr, k = (int)ks;
  if (epi == 2) {
    switch (NB) {
      case 1: return hw8_launch_nb<1, 2>(a, yt, un, lp, k);
      case 2: return hw8_launch_nb<2, 2>(a, yt, un, lp, k);
      default: return hw8_launch_nb<3, 2>(a, yt, un, lp, k);
    }
  } else if (epi == 1) {
    switch (NB) {
      case 1: return hw8_launch_nb<1, 1>(a, yt, un, lp, k);
      case 2: return hw8_launch_nb<2, 1>(a, yt, un, lp, k);
      default: return hw8_launch_nb<3, 1>(a, yt, un, lp, k);
    }
  }
  switch (NB) {
    case 1: return hw8_launch_nb<1, 0>(a, yt, un, lp, k);
    case 2: return hw8_launch_nb<2, 0>(a, yt, un, lp, k);
    default: return hw8_launch_nb<3, 0>(a, yt, un, lp, k);
  }
}

// ---------------------------------------------------------------------------
// int8 + fp32 scales -> bf16 (RNE), for the M > 3 fallback on a freed weight.
// ---------------------------------------------------------------------------
namespace {

__device__ __forceinline__ unsigned short hw8_f2bf16_rne(float f) {
  unsigned int u = __float_as_uint(f);
  unsigned int rr = ((u >> 16) & 1u) + 0x7fffu;
  return (unsigned short)((u + rr) >> 16);
}

__global__ void hc_w8_dequant_kernel(const int8_t* __restrict__ W, const float* __restrict__ S,
                                     unsigned short* __restrict__ O, int K, int KV, long total,
                                     int sK, int gshift) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  for (; i < total; i += stride) {
    const int row = (int)(i / KV);
    const int k0 = (int)(i - (long)row * KV) * 16;
    const float sc = S[(long)row * sK + (k0 >> gshift)];
    const long off = (long)row * K + k0;
    const int4 qv = *reinterpret_cast<const int4*>(W + off);
    const int8_t* qb = reinterpret_cast<const int8_t*>(&qv);
    unsigned short h[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) h[j] = hw8_f2bf16_rne((float)qb[j] * sc);
    float4* dst = reinterpret_cast<float4*>(O + off);
    dst[0] = *reinterpret_cast<const float4*>(h);
    dst[1] = *reinterpret_cast<const float4*>(h + 8);
  }
}

}  // namespace

bool hc_w8_dequant(at::Tensor W, at::Tensor S, at::Tensor O, int64_t gs) {
  const int R = (int)W.size(0), K = (int)W.size(1);
  TORCH_CHECK(W.dim() == 2 && W.dtype() == torch::kInt8 && W.is_contiguous(),
              "W must be contiguous int8 [R, K]");
  TORCH_CHECK(S.dtype() == torch::kFloat32 && S.is_contiguous(), "S must be contiguous fp32");
  TORCH_CHECK(O.dim() == 2 && O.dtype() == torch::kBFloat16 && O.is_contiguous() &&
                  O.size(0) == R && O.size(1) == K,
              "O must be contiguous bf16 [R, K]");
  if (K % 16 != 0) return false;
  int sK = 1, gshift = 31;
  if (gs == 0) {
    TORCH_CHECK(S.numel() == R, "S must be [R] for gs == 0");
  } else {
    TORCH_CHECK(gs >= 16 && (gs & (gs - 1)) == 0 && K % gs == 0,
                "gs must be a power of two >= 16 dividing K");
    sK = K / (int)gs;
    gshift = 0;
    while ((1 << gshift) < gs) gshift++;
    TORCH_CHECK(S.numel() == (long)R * sK, "S must be [R, K/gs]");
  }
  const int KV = K / 16;
  const long total = (long)R * KV;
  if (total == 0) return true;
  const int block = 256;
  long need = (total + block - 1) / block;
  const int grid = (int)(need < 4096L ? need : 4096L);
  {
    hc_w8_dequant_kernel<<<dim3(grid), dim3(block), 0, c10::hip::getCurrentHIPStream().stream()>>>(
        reinterpret_cast<const int8_t*>(W.data_ptr()), reinterpret_cast<const float*>(S.data_ptr()),
        reinterpret_cast<unsigned short*>(O.data_ptr()), K, KV, total, sK, gshift);
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hc_w8_gemv", &hc_w8_gemv);
  m.def("hc_w8_dequant", &hc_w8_dequant);
}
