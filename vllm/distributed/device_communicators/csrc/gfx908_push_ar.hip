// SPDX-License-Identifier: Apache-2.0
// gfx908 (MI100 / CDNA1) sentinel-validated push all-reduce over XGMI.
//
// Design + measurements: docs/mi100_decode_opt/research/allreduce_fusion.md (Step C) and the
// microbenchmark report in agents/ar_push/REPORT.md.  Summary of the scheme:
//
//   * every rank owns one uncached (hipDeviceMallocUncached -> MTYPE_UC locally, MTYPE_UC over
//     XGMI) receive buffer, IPC-shared with the other ranks and pre-armed with the bf16 -0.0
//     sentinel (0x8000);
//   * `push` writes this rank's partial into slot (site, my_rank) of ALL four ranks' buffers with
//     nontemporal 16 B stores -- fire and forget, no fences, no flags, no atomics.  Any element
//     that is exactly -0.0 is sanitized to +0.0 so it can never be mistaken for the sentinel;
//   * `consume` spins on its own 16 B of each of the four source rows until none of them holds
//     the sentinel any more, sums them in fixed rank order 0..3 in fp32 (bitwise identical to
//     vLLM's `packed_reduce`), rounds once to bf16, writes `out`, and re-arms its own 16 B of all
//     four rows with the sentinel for the next use of the slot.
//
// Because every element is owned by exactly one thread, no block barrier is needed between the
// reads and the re-arm, and no end-of-collective barrier is needed at all: correctness relies on
// a slot not being pushed again before every rank consumed it, which holds as long as >= 2 sites
// are in flight per cycle (the Python side allocates 128 + 8).
//
// Slot layout on every rank (bf16 elements): slot[site][src_rank][slot_elems]
//   push offset (elements) = (site * 4 + my_rank) * slot_elems
//   consume base (bytes)   = own_buffer + site * 4 * slot_elems * 2
//   source stride          = slot_elems  (passed to the kernel as src_stride16 = slot_elems / 8)
//
// The poll loop MUST use inline asm: ROCm 7.x LLVM hoists __builtin_nontemporal_load and
// __builtin_amdgcn_raw_buffer_load (even with the VOLATILE aux bit) out of a loop whose body
// writes no memory.  Verified in the disassembly; see REPORT.md section 1.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>
#include <algorithm>
#include <vector>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
struct PushPtrs { void* base[4]; };

static constexpr unsigned SENT16 = 0x8000u;      // bf16 -0.0
static constexpr unsigned SENT32 = 0x80008000u;

#define HIP_CHECK(x)                                                                \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    TORCH_CHECK(e_ == hipSuccess, #x, " failed: ", hipGetErrorString(e_));          \
  } while (0)

static inline hipStream_t cur_stream() { return c10::hip::getCurrentHIPStream().stream(); }

// ------------------------------------------------------------------ bf16 helpers (torch RNE)
__device__ __forceinline__ unsigned f2bf_bits(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;   // NaN -> canonical qNaN (c10::BFloat16)
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ float bf_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ unsigned sanitize_word(unsigned w) {
  w = ((w & 0xffffu) == SENT16) ? (w & 0xffff0000u) : w;
  w = ((w >> 16) == SENT16) ? (w & 0x0000ffffu) : w;
  return w;
}
__device__ __forceinline__ bool word_has_sentinel(unsigned w) {
  return ((w & 0xffffu) == SENT16) | ((w >> 16) == SENT16);
}

// ------------------------------------------------------------------ arming
__global__ void fill_u32_k(unsigned* p, long n, unsigned v) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    __builtin_nontemporal_store(v, p + i);
  }
}
// Fill `nbytes` at `ptr` with the 32-bit pattern `v` (hipMemset cannot do 32-bit patterns).
void fill_u32(int64_t ptr, int64_t nbytes, int64_t v) {
  TORCH_CHECK(nbytes % 4 == 0);
  long n = nbytes / 4;
  int grid = (int)std::min<long>((n + 255) / 256, 1024);
  {
    fill_u32_k<<<grid, 256, 0, cur_stream()>>>((unsigned*)ptr, n, (unsigned)v);
  }
}

// ------------------------------------------------------------------ producer
__global__ void __launch_bounds__(256)
push_k(const u32x4* __restrict__ in, PushPtrs pp, long off16, int n16) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n16) return;
  u32x4 v = in[i];
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] = sanitize_word(v[j]);
#pragma unroll
  for (int r = 0; r < 4; r++) {   // fixed order, self included; no fences, no flags
    __builtin_nontemporal_store(v, ((u32x4*)pp.base[r]) + off16 + i);
  }
}

// x: [T, N] bf16 contiguous.  bases: this rank's view of all 4 ranks' receive buffers.
// off_elems = (site * 4 + my_rank) * slot_elems.
void push(at::Tensor x, std::vector<int64_t> bases, int64_t off_elems) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16, "push: contiguous bf16 only");
  TORCH_CHECK(bases.size() == 4, "push: 4 ranks only");
  TORCH_CHECK(x.numel() % 8 == 0 && off_elems % 8 == 0, "push: 16 B alignment");
  PushPtrs pp;
  for (int r = 0; r < 4; r++) pp.base[r] = (void*)bases[r];
  int n16 = (int)(x.numel() / 8);
  {
    push_k<<<(n16 + 255) / 256, 256, 0, cur_stream()>>>((const u32x4*)x.data_ptr(), pp, off_elems / 8, n16);
  }
}

// ------------------------------------------------------------------ consumer
__device__ __forceinline__ void poll_load4(u32x4& a, u32x4& b, u32x4& c, u32x4& d,
                                           const void* p0, const void* p1,
                                           const void* p2, const void* p3) {
  asm volatile(
      "global_load_dwordx4 %0, %4, off glc slc\n\t"
      "global_load_dwordx4 %1, %5, off glc slc\n\t"
      "global_load_dwordx4 %2, %6, off glc slc\n\t"
      "global_load_dwordx4 %3, %7, off glc slc\n\t"
      "s_waitcnt vmcnt(0)"
      : "=&v"(a), "=&v"(b), "=&v"(c), "=&v"(d)
      : "v"(p0), "v"(p1), "v"(p2), "v"(p3)
      : "memory");
}

__device__ __forceinline__ unsigned wave_max(unsigned v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v = max(v, (unsigned)__shfl_xor((int)v, o));
  return v;
}

// grid = T blocks (one row per block), block = N / 8 threads (16 B per thread per source).
// RES: out = bf16(res + bf16(sum)) -- the same double rounding as today's AR-then-add path.
template <bool RES>
__global__ void __launch_bounds__(1024)
consume_k(u32x4* __restrict__ out, const u32x4* __restrict__ res, u32x4* site_ptr,
          int src_stride16, int n16, unsigned* stats, unsigned max_spin, int site, int spin_stats) {
  const int t = blockIdx.x, c = threadIdx.x;
  if (c >= n16) return;
  u32x4* rowp[4];
#pragma unroll
  for (int r = 0; r < 4; r++) rowp[r] = site_ptr + (long)r * src_stride16 + (long)t * n16;
  u32x4 v[4];
  unsigned spins = 0;
  bool ready;
  do {
    poll_load4(v[0], v[1], v[2], v[3], rowp[0] + c, rowp[1] + c, rowp[2] + c, rowp[3] + c);
    ready = true;
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
      for (int j = 0; j < 4; j++) ready &= !word_has_sentinel(v[r][j]);
    }
    if (!ready) {
      if (++spins >= max_spin) {   // bounded spin: flag, record, proceed (never hangs)
        if (stats) {
          atomicAdd(stats + 0, 1u);
          stats[3] = (unsigned)site;
          stats[4] = (unsigned)t;
          stats[5] = (unsigned)c * 8;
          for (int r = 0; r < 4; r++) stats[6 + r] = v[r][0];
        }
        break;
      }
    }
  } while (!ready);
  u32x4 o;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float lo = ((bf_lo(v[0][j]) + bf_lo(v[1][j])) + bf_lo(v[2][j])) + bf_lo(v[3][j]);
    float hi = ((bf_hi(v[0][j]) + bf_hi(v[1][j])) + bf_hi(v[2][j])) + bf_hi(v[3][j]);
    unsigned blo = f2bf_bits(lo), bhi = f2bf_bits(hi);
    if constexpr (RES) {
      unsigned rw = res[(long)t * n16 + c][j];
      blo = f2bf_bits(bf_lo(rw) + __uint_as_float(blo << 16));
      bhi = f2bf_bits(bf_hi(rw) + __uint_as_float(bhi << 16));
    }
    o[j] = blo | (bhi << 16);
  }
  out[(long)t * n16 + c] = o;
  const u32x4 s4 = {SENT32, SENT32, SENT32, SENT32};
#pragma unroll
  for (int r = 0; r < 4; r++) __builtin_nontemporal_store(s4, rowp[r] + c);   // re-arm (local UC)
  if (stats && spin_stats) {
    unsigned m = wave_max(spins);
    if ((threadIdx.x & 63) == 0) {
      atomicMax(stats + 1, m);
      if (m) atomicAdd(stats + 2, 1u);
    }
  }
}

static void consume_impl(at::Tensor out, const at::Tensor* res, int64_t site_ptr,
                         int64_t src_stride_elems, at::Tensor stats, int64_t max_spin,
                         int64_t site, int64_t spin_stats) {
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2,
              "consume: contiguous 2D bf16 out");
  const int T = (int)out.size(0), N = (int)out.size(1);
  TORCH_CHECK(N % 8 == 0 && N / 8 <= 1024, "consume: N must be a multiple of 8 and <= 8192");
  TORCH_CHECK(src_stride_elems % 8 == 0 && (long)T * N <= src_stride_elems,
              "consume: message does not fit the slot");
  unsigned* st = (stats.defined() && stats.numel() > 0) ? (unsigned*)stats.data_ptr() : nullptr;
  auto o = (u32x4*)out.data_ptr();
  auto sp = (u32x4*)site_ptr;
  const int stride16 = (int)(src_stride_elems / 8);
  if (res) {
    TORCH_CHECK(res->is_contiguous() && res->sizes() == out.sizes() &&
                res->scalar_type() == at::kBFloat16);
    consume_k<true><<<T, N / 8, 0, cur_stream()>>>(o, (const u32x4*)res->data_ptr(), sp, stride16,
                                                   N / 8, st, (unsigned)max_spin, (int)site,
                                                   (int)spin_stats);
  } else {
    consume_k<false><<<T, N / 8, 0, cur_stream()>>>(o, nullptr, sp, stride16, N / 8, st,
                                                    (unsigned)max_spin, (int)site, (int)spin_stats);
  }
}

void consume(at::Tensor out, int64_t site_ptr, int64_t src_stride_elems, at::Tensor stats,
             int64_t max_spin, int64_t site, int64_t spin_stats) {
  consume_impl(out, nullptr, site_ptr, src_stride_elems, stats, max_spin, site, spin_stats);
}
void consume_res(at::Tensor out, at::Tensor res, int64_t site_ptr, int64_t src_stride_elems,
                 at::Tensor stats, int64_t max_spin, int64_t site, int64_t spin_stats) {
  consume_impl(out, &res, site_ptr, src_stride_elems, stats, max_spin, site, spin_stats);
}

// ------------------------------------------------------------------ capture identity
// Unique id of the stream capture the current stream is part of, 0 when not capturing.  The Python
// side uses it to reset the per-graph site counter exactly once per captured graph.
int64_t capture_id() {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long id = 0;
  hipError_t e = hipStreamGetCaptureInfo(cur_stream(), &status, &id);
  if (e != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  return (status == hipStreamCaptureStatusActive) ? (int64_t)id : 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_u32", &fill_u32);
  m.def("push", &push);
  m.def("consume", &consume);
  m.def("consume_res", &consume_res);
  m.def("capture_id", &capture_id);
}
