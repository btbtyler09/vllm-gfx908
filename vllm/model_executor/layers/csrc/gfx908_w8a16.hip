// SPDX-License-Identifier: Apache-2.0
// gfx908 W8A16 skinny GEMV — torch binding.
//
//   w8a16_gemv(W int8 [N, K], S fp32 [N] (gs == 0) or [N, K/gs] (gs power of two >= 16, gs | K/ks),
//              X bf16 [M, K], C bf16 [M, N], gs, ytile, unrl, lpr, ks, pf, cu) -> bool
//
// Returns false (and launches nothing) when the requested config is not instantiated or does not
// apply to this shape, so the caller can fall back to the bf16 path.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include "gfx908_w8a16_launch.cuh"

bool w8a16_dequant(at::Tensor W, at::Tensor S, at::Tensor O, int64_t gs);
bool mfma_w8_gemm(at::Tensor Wsw, at::Tensor Ssw, at::Tensor X, at::Tensor C, int64_t tile,
                  int64_t nw, int64_t sk, int64_t ug, int64_t dbg);
bool mfma_w8_dequant(at::Tensor Wsw, at::Tensor Ssw, at::Tensor O, int64_t r0, int64_t Nfull);
void wsw_stream(at::Tensor W, at::Tensor out, int64_t cu);
bool w8sw_gemv(at::Tensor Wsw, at::Tensor Ssw, at::Tensor X, at::Tensor C, int64_t gs,
               int64_t nt, int64_t unrl, int64_t ks, int64_t cu);

extern template bool w8_launch_nb<1>(const W8Args&, int, int, int, int, int);
extern template bool w8_launch_nb<2>(const W8Args&, int, int, int, int, int);
extern template bool w8_launch_nb<3>(const W8Args&, int, int, int, int, int);
extern template bool w8_launch_nb<4>(const W8Args&, int, int, int, int, int);

bool w8a16_gemv(at::Tensor W, at::Tensor S, at::Tensor X, at::Tensor C, int64_t gs,
                int64_t ytile, int64_t unrl, int64_t lpr, int64_t ks, int64_t pf, int64_t cu) {
  const int Nrows = W.size(0), K = W.size(1), NB = X.size(0);
  TORCH_CHECK(W.dtype() == torch::kInt8 && W.is_contiguous(), "W must be contiguous int8 [N, K]");
  TORCH_CHECK(S.dtype() == torch::kFloat32 && S.is_contiguous(), "S must be contiguous fp32");
  TORCH_CHECK(X.dtype() == torch::kBFloat16 && X.is_contiguous() && X.size(1) == K,
              "X must be contiguous bf16 [M, K]");
  TORCH_CHECK(C.dtype() == torch::kBFloat16 && C.is_contiguous() && C.size(0) == NB && C.size(1) == Nrows,
              "C must be contiguous bf16 [M, N]");
  TORCH_CHECK(K % 16 == 0, "K % 16 == 0");
  TORCH_CHECK(NB >= 1 && NB <= 4, "M (batch) must be 1..4");
  int G = 1, gshift = 0;
  if (gs == 0) {
    TORCH_CHECK(S.numel() == Nrows, "S must be [N]");
  } else {
    TORCH_CHECK(gs >= 16 && (gs & (gs - 1)) == 0 && K % gs == 0,
                "gs must be a power of two >= 16 dividing K");
    G = K / (int)gs;
    while ((1 << gshift) < gs) gshift++;
    TORCH_CHECK(S.numel() == (long)Nrows * G, "S must be [N, K/gs]");
  }
  W8Args a{K, Nrows, reinterpret_cast<const int8_t*>(W.data_ptr()),
           reinterpret_cast<const float*>(S.data_ptr()),
           reinterpret_cast<const uint16_t*>(X.data_ptr()),
           reinterpret_cast<uint16_t*>(C.data_ptr()), (int)cu,
           c10::hip::getCurrentHIPStream().stream(), G, gshift, (int)gs};
  switch (NB) {
    case 1: return w8_launch_nb<1>(a, (int)ytile, (int)unrl, (int)lpr, (int)ks, (int)pf);
    case 2: return w8_launch_nb<2>(a, (int)ytile, (int)unrl, (int)lpr, (int)ks, (int)pf);
    case 3: return w8_launch_nb<3>(a, (int)ytile, (int)unrl, (int)lpr, (int)ks, (int)pf);
    default: return w8_launch_nb<4>(a, (int)ytile, (int)unrl, (int)lpr, (int)ks, (int)pf);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_gemv", &w8a16_gemv);
  m.def("w8a16_dequant", &w8a16_dequant);
  m.def("mfma_w8_gemm", &mfma_w8_gemm);
  m.def("mfma_w8_dequant", &mfma_w8_dequant);
  m.def("w8sw_gemv", &w8sw_gemv);
  m.def("wsw_stream", &wsw_stream);
}
