// gfx908 W4A16 GEMV for small M: dense projections and MoE experts.
//
// Weights: [E, N, K/8] uint32, 8 nibbles along K (low nibble = lowest k),
// symmetric zero point 8; scales [E, N, K/gs] bf16.  This is the layout of
// vLLM's TRITON WNA16 MoE backend ([E, N, K/2] uint8 viewed as uint32) and,
// transposed, of triton_w4a16's [K, N/8] int32 (handled on the Python side by
// a one-time repack).
//
// Rows of the output are "pairs": row r reads token row_token[r] of x and
// expert row_expert[r]. Consecutive rows in groups of MT share one expert
// (dense: one group of M rows, expert 0; MoE: MT=1, one row per pair).
// Grid: (N / BN, SPLIT_K, num_groups). Block: BN threads, thread = one n.
// Output: fp32 partials part[SPLIT_K, P, N], reduced by the caller.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>


__device__ __forceinline__ float bf2f(uint16_t v) {
  return __uint_as_float(((uint32_t)v) << 16);
}

template <int MT, int BN>
__global__ void __launch_bounds__(BN) w4gemv_kernel(
    const uint16_t* __restrict__ x, const uint32_t* __restrict__ w,
    const uint16_t* __restrict__ s, const int32_t* __restrict__ row_token,
    const int32_t* __restrict__ row_expert, float* __restrict__ part, int P,
    int N, int K, int gs, int k_per_split, long stride_we, long stride_se,
    int E, int M) {
  const int n = blockIdx.x * BN + threadIdx.x;
  const int z = blockIdx.y;
  const int g = blockIdx.z;
  const int r0 = g * MT;
  const int nr = min(MT, P - r0);
  const int e = row_expert[r0];
  const int k_begin = z * k_per_split;
  const int k_end = min(K, k_begin + k_per_split);
  const int klen = k_end - k_begin;
  if (klen <= 0) return;
  if (e < 0 || e >= E) {
    // invalid expert (dummy/capture routing): emit zeros, touch no weights
    if (n < N) {
      float* out0 = part + ((long)z * P + r0) * N + n;
      for (int m = 0; m < nr; ++m) out0[(long)m * N] = 0.f;
    }
    return;
  }

  extern __shared__ float xs_flat[];
  const int kstride = k_per_split;
  for (int m = 0; m < nr; ++m) {
    const int tok = min(max(row_token[r0 + m], 0), M - 1);
    const uint16_t* xr = x + (long)tok * K + k_begin;
    for (int k = threadIdx.x; k < klen; k += BN) xs_flat[m * kstride + k] = bf2f(xr[k]);
  }
  __syncthreads();
  if (n >= N) return;

  const uint32_t* wp = w + e * stride_we + (long)n * (K / 8) + k_begin / 8;
  const uint16_t* sp = s + e * stride_se + (long)n * (K / gs);
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;

  // 16-byte weight loads: 4 x uint32 = 32 k values.
  const int nvec = klen / 32;
  const uint4* wv4 = reinterpret_cast<const uint4*>(wp);
  uint4 q4 = nvec > 0 ? wv4[0] : make_uint4(0, 0, 0, 0);
  for (int v = 0; v < nvec; ++v) {
    uint4 q4n = (v + 1 < nvec) ? wv4[v + 1] : make_uint4(0, 0, 0, 0);
    const uint32_t qs[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int kk = v * 32 + j * 8;  // offset within chunk
      const float sc = bf2f(sp[(k_begin + kk) / gs]);
      const uint32_t q = qs[j];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const float wf = (float)((q >> (4 * t)) & 0xF) - 8.f;
        const float wsc = wf * sc;
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] += wsc * xs_flat[m * kstride + kk + t];
      }
    }
    q4 = q4n;
  }
  // tail (klen % 32) in uint32 steps
  for (int kk = nvec * 32; kk < klen; kk += 8) {
    const uint32_t q = wp[kk / 8];
    const float sc = bf2f(sp[(k_begin + kk) / gs]);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const float wf = (float)((q >> (4 * t)) & 0xF) - 8.f;
      const float wsc = wf * sc;
#pragma unroll
      for (int m = 0; m < MT; ++m) acc[m] += wsc * xs_flat[m * kstride + kk + t];
    }
  }
  float* out = part + ((long)z * P + r0) * N + n;
  for (int m = 0; m < nr; ++m) out[(long)m * N] = acc[m];
}

template <int MT, int BN>
static void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s,
                   const at::Tensor& row_token, const at::Tensor& row_expert,
                   at::Tensor& part, int P, int N, int K, int gs, int split_k,
                   int k_per_split, long stride_we, long stride_se) {
  const int groups = (P + MT - 1) / MT;
  dim3 grid((N + BN - 1) / BN, split_k, groups);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const size_t smem = (size_t)MT * k_per_split * sizeof(float);
  w4gemv_kernel<MT, BN><<<grid, BN, smem, stream>>>(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      reinterpret_cast<const uint32_t*>(w.data_ptr()),
      reinterpret_cast<const uint16_t*>(s.data_ptr()),
      row_token.data_ptr<int32_t>(), row_expert.data_ptr<int32_t>(),
      part.data_ptr<float>(), P, N, K, gs, k_per_split, stride_we, stride_se,
      (int)w.size(0), (int)x.size(0));
}

// part: [split_k, P, N] fp32 (preallocated). mt: rows per group (1 or 16).
void w4gemv(at::Tensor x, at::Tensor w, at::Tensor s, at::Tensor row_token,
            at::Tensor row_expert, at::Tensor part, int64_t gs, int64_t mt,
            int64_t bn) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && s.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w.scalar_type() == at::kInt || w.scalar_type() == at::kByte);
  const int P = row_token.numel();
  const int K = x.size(1);
  const int N = w.size(1);
  const int split_k = part.size(0);
  const int k_per_split = (K + split_k - 1) / split_k;
  TORCH_CHECK(k_per_split % 8 == 0 && (size_t)mt * k_per_split * sizeof(float) <= 60000);
  const long stride_we = (long)N * K / 8;   // uint32 units
  const long stride_se = (long)N * (K / gs);
  if (mt == 1) {
    if (bn == 64) launch<1, 64>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
    else if (bn == 128) launch<1, 128>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
    else launch<1, 256>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
  } else if (mt == 4) {
    if (bn == 64) launch<4, 64>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
    else if (bn == 128) launch<4, 128>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
    else launch<4, 256>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
  } else {
    if (bn == 64) launch<16, 64>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
    else if (bn == 128) launch<16, 128>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
    else launch<16, 256>(x, w, s, row_token, row_expert, part, P, N, K, gs, split_k, k_per_split, stride_we, stride_se);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("w4gemv", &w4gemv); }
