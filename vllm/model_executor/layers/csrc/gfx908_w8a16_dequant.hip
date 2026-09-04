// SPDX-License-Identifier: Apache-2.0
// gfx908 W8A16 — int8 + fp32 group scales -> bf16 dequantization into a scratch buffer.
//
//   w8a16_dequant(W int8 [R, K], S fp32 [R] (gs == 0) or [R, K/gs], O bf16 [R, K], gs)
//
// Used by the M > 4 (prefill / batched decode) path once the bf16 master copy of a
// whitelisted weight has been released: the weight is rematerialised in N-row chunks
// into a reusable bf16 scratch and handed to the stock bf16 GEMM dispatch.
//
// Rounding is round-to-nearest-even on the fp32 product, i.e. bit-identical to
// torch's  (W.float() * S).to(torch.bfloat16).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ unsigned short f2bf16_rne(float f) {
  unsigned int u = __float_as_uint(f);
  unsigned int r = ((u >> 16) & 1u) + 0x7fffu;
  return (unsigned short)((u + r) >> 16);
}

// One thread handles 16 contiguous k of one row (16 B int8 in, 32 B bf16 out).
// gs is a power of two >= 16, so those 16 lanes always share a single scale.
__global__ void w8a16_dequant_kernel(const int8_t* __restrict__ W,
                                     const float* __restrict__ S,
                                     unsigned short* __restrict__ O, int K, int KV,
                                     long total, int sK, int gshift) {
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
    for (int j = 0; j < 16; ++j) h[j] = f2bf16_rne((float)qb[j] * sc);
    float4* dst = reinterpret_cast<float4*>(O + off);
    dst[0] = *reinterpret_cast<const float4*>(h);
    dst[1] = *reinterpret_cast<const float4*>(h + 8);
  }
}

}  // namespace

bool w8a16_dequant(at::Tensor W, at::Tensor S, at::Tensor O, int64_t gs) {
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
    w8a16_dequant_kernel<<<dim3(grid), dim3(block), 0, c10::hip::getCurrentHIPStream().stream()>>>(
        reinterpret_cast<const int8_t*>(W.data_ptr()),
        reinterpret_cast<const float*>(S.data_ptr()),
        reinterpret_cast<unsigned short*>(O.data_ptr()), K, KV, total, sK, gshift);
  }
  return true;
}
