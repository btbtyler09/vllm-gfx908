// SPDX-License-Identifier: Apache-2.0
// Torch binding for the swizzled-layout W8A16 GEMV (M <= 4).
//
//   w8sw_gemv(Wsw int8 (MFMA-swizzled, tile 16, N*K bytes),
//             Ssw fp32 [K/gs, N],
//             X   bf16 [M, K], C bf16 [M, N], N, gs, nt, unrl, ks, cu) -> bool
//
// Returns false (and launches nothing) when the config is not instantiated / not applicable.
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include "gfx908_w8sw_launch.cuh"

// Practical HBM read ceiling for the same rotation protocol: every WG streams a contiguous
// slice with nontemporal dwordx4 loads and xors it into one dword (never stored).
__global__ void __launch_bounds__(256)
wsw_stream_probe(const wsw_u32x4* __restrict__ p, const long n16, uint32_t* __restrict__ out) {
  const long per = (n16 + gridDim.x - 1) / gridDim.x;
  const long beg = (long)blockIdx.x * per, end = min(beg + per, n16);
  uint32_t acc = 0;
  for (long i = beg + threadIdx.x; i < end; i += 256 * 4) {
    wsw_u32x4 a = {0, 0, 0, 0}, b = a, c = a, d = a;
    a = wsw_loadnt(p + i);
    if (i + 256 < end) b = wsw_loadnt(p + i + 256);
    if (i + 512 < end) c = wsw_loadnt(p + i + 512);
    if (i + 768 < end) d = wsw_loadnt(p + i + 768);
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  if (acc == 0xdeadbeef) out[blockIdx.x] = acc;
}

void wsw_stream(at::Tensor W, at::Tensor out, int64_t cu) {
  const long n16 = W.numel() / 16;
  wsw_stream_probe<<<dim3((int)cu), dim3(256), 0, c10::hip::getCurrentHIPStream().stream()>>>(
      reinterpret_cast<const wsw_u32x4*>(W.data_ptr()), n16,
      reinterpret_cast<uint32_t*>(out.data_ptr()));
}

extern template bool wsw_launch_nb<1>(const WswArgs&, int, int, int);
extern template bool wsw_launch_nb<2>(const WswArgs&, int, int, int);
extern template bool wsw_launch_nb<3>(const WswArgs&, int, int, int);
extern template bool wsw_launch_nb<4>(const WswArgs&, int, int, int);

bool w8sw_gemv(at::Tensor Wsw, at::Tensor Ssw, at::Tensor X, at::Tensor C, int64_t gs,
               int64_t nt, int64_t unrl, int64_t ks, int64_t cu) {
  const int NB = (int)X.size(0), K = (int)X.size(1), N = (int)C.size(1);
  TORCH_CHECK(Wsw.dtype() == torch::kInt8 && Wsw.is_contiguous(), "Wsw contiguous int8");
  TORCH_CHECK(Ssw.dtype() == torch::kFloat32 && Ssw.is_contiguous(), "Ssw contiguous fp32");
  TORCH_CHECK(X.dtype() == torch::kBFloat16 && X.is_contiguous(), "X contiguous bf16 [M, K]");
  TORCH_CHECK(C.dtype() == torch::kBFloat16 && C.is_contiguous() && C.size(0) == NB,
              "C contiguous bf16 [M, N]");
  TORCH_CHECK(NB >= 1 && NB <= 4, "M must be 1..4");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "K % 64 == 0, N % 16 == 0");
  TORCH_CHECK(Wsw.numel() == (long)N * K, "Wsw must hold N*K bytes");
  TORCH_CHECK(gs >= 64 && (gs & (gs - 1)) == 0 && K % gs == 0,
              "gs must be a power of two >= 64 dividing K");
  TORCH_CHECK(Ssw.numel() == (long)N * (K / (int)gs), "Ssw must be [K/gs, N]");
  int gshift = 0;
  while ((1 << gshift) < (int)gs) gshift++;
  WswArgs a{K, N, reinterpret_cast<const int8_t*>(Wsw.data_ptr()),
            reinterpret_cast<const float*>(Ssw.data_ptr()),
            reinterpret_cast<const uint16_t*>(X.data_ptr()),
            reinterpret_cast<uint16_t*>(C.data_ptr()), (int)cu,
            c10::hip::getCurrentHIPStream().stream(), gshift};
  switch (NB) {
    case 1: return wsw_launch_nb<1>(a, (int)nt, (int)unrl, (int)ks);
    case 2: return wsw_launch_nb<2>(a, (int)nt, (int)unrl, (int)ks);
    case 3: return wsw_launch_nb<3>(a, (int)nt, (int)unrl, (int)ks);
    default: return wsw_launch_nb<4>(a, (int)nt, (int)unrl, (int)ks);
  }
}
