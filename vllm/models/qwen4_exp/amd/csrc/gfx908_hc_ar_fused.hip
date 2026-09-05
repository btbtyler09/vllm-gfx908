// SPDX-License-Identifier: Apache-2.0
// gfx908 (MI100) push-AR consumer fused with the hyper-connection combine (+ grouped Gemma RMSNorm).
//
// Today (VLLM_GFX908_PUSH_AR=1) every TP all-reduce of the decode step is `push_k` (producer,
// fire-and-forget to all 4 ranks) + `consume_k` (poll the 4 source rows of the slot, sum in rank
// order, write the reduced bf16 tensor, re-arm), and the reduced tensor is then read by the Triton
// `_hc_combine_norm_kernel` (block = the reduced tensor, res = the multi-stream residual):
//
//     out[s] = bf16(res[s] + block * 2*sigmoid(inj[s]/HC))          s = 0..HC-1   (new residual)
//     y[s]   = bf16(out[s] * rsqrt(mean(out[s]^2) + eps) * (1 + w[s]))            (norm output)
//
// This kernel does consume + combine + norm in ONE launch: grid = T rows, block = HD/8 threads,
// thread c owns the 16 B at columns [8c, 8c+8) of *every* stream, i.e. exactly the 16 B of each of
// the four source rows that `consume_k` gave it, so
//   * the poll / sum / re-arm is per-thread ownership, no cross-workgroup wait beyond the AR's own
//     flag wait (the sentinel poll), no atomics, no LDS in the AR part;
//   * the reduced value never round-trips through HBM: it is rounded to bf16 in registers (the same
//     rounding the stock path applies when it stores the AR output) and consumed directly;
//   * the HC streams of a row are all in the same workgroup, so the HC sum-of-squares reductions
//     are one LDS pass (5 waves x HC floats).
// The residual / injection / norm-weight loads are issued *before* the poll so their latency
// overlaps the (local UC) flag wait.
//
// Numerics: the consumer sum is ((r0+r1)+r2)+r3 in fp32, one RNE to bf16 -- identical to
// `consume_k` / `packed_reduce`.  The combine is res + block*inj in fp32, one RNE -- the Triton
// kernel contracts this to an FMA (mode bit 1 selects fma / mul+add; the test picks the bit-exact
// one).  The sum of squares is a different reduction tree than Triton's [8, 512] tile sum, so y may
// differ by one bf16 ulp on a small fraction of elements; measured and reported by the test.
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
__device__ __forceinline__ float bfbits2f(unsigned b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ unsigned sanitize_word(unsigned w) {
  w = ((w & 0xffffu) == SENT16) ? (w & 0xffff0000u) : w;
  w = ((w >> 16) == SENT16) ? (w & 0x0000ffffu) : w;
  return w;
}
__device__ __forceinline__ bool word_has_sentinel(unsigned w) {
  return ((w & 0xffffu) == SENT16) | ((w >> 16) == SENT16);
}

// ------------------------------------------------------------------ arming / buffers
__global__ void fill_u32_k(unsigned* p, long n, unsigned v) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    __builtin_nontemporal_store(v, p + i);
  }
}
void fill_u32(int64_t ptr, int64_t nbytes, int64_t v) {
  TORCH_CHECK(nbytes % 4 == 0);
  long n = nbytes / 4;
  int grid = (int)std::min<long>((n + 255) / 256, 1024);
  {
    fill_u32_k<<<grid, 256, 0, cur_stream()>>>((unsigned*)ptr, n, (unsigned)v);
  }
}
// Uncached device allocation (same flag vLLM's custom-AR / push-AR buffers use).
int64_t alloc_uncached(int64_t nbytes) {
  void* p = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(&p, (size_t)nbytes, hipDeviceMallocUncached));
  return (int64_t)p;
}
void free_ptr(int64_t p) { HIP_CHECK(hipFree((void*)p)); }
void memcpy_to(int64_t dst, at::Tensor src) {
  TORCH_CHECK(src.is_contiguous());
  HIP_CHECK(hipMemcpyAsync((void*)dst, src.data_ptr(), src.numel() * src.element_size(),
                           hipMemcpyDeviceToDevice, cur_stream()));
}
void memcpy_from(at::Tensor dst, int64_t src) {
  TORCH_CHECK(dst.is_contiguous());
  HIP_CHECK(hipMemcpyAsync(dst.data_ptr(), (const void*)src, dst.numel() * dst.element_size(),
                           hipMemcpyDeviceToDevice, cur_stream()));
}

// ------------------------------------------------------------------ producer (= gfx908_push_ar.hip)
__global__ void __launch_bounds__(256)
push_k(const u32x4* __restrict__ in, PushPtrs pp, long off16, int n16) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n16) return;
  u32x4 v = in[i];
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] = sanitize_word(v[j]);
#pragma unroll
  for (int r = 0; r < 4; r++) {
    __builtin_nontemporal_store(v, ((u32x4*)pp.base[r]) + off16 + i);
  }
}
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

// ------------------------------------------------------------------ poll (= gfx908_push_ar.hip)
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
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

// Poll the four source rows for this thread's 16 B, bounded spin, stats as consume_k.
__device__ __forceinline__ void ar_poll(u32x4 (&v)[4], u32x4* const (&rowp)[4], int c, int t,
                                        unsigned* stats, unsigned max_spin, int site,
                                        unsigned& spins) {
  spins = 0;
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
      if (++spins >= max_spin) {
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
}

// ------------------------------------------------------------------ stock consumer (baseline arm)
__global__ void __launch_bounds__(1024)
consume_k(u32x4* __restrict__ out, u32x4* site_ptr, int src_stride16, int n16, unsigned* stats,
          unsigned max_spin, int site, int spin_stats) {
  const int t = blockIdx.x, c = threadIdx.x;
  if (c >= n16) return;
  u32x4* rowp[4];
#pragma unroll
  for (int r = 0; r < 4; r++) rowp[r] = site_ptr + (long)r * src_stride16 + (long)t * n16;
  u32x4 v[4];
  unsigned spins;
  ar_poll(v, rowp, c, t, stats, max_spin, site, spins);
  u32x4 o;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float lo = ((bf_lo(v[0][j]) + bf_lo(v[1][j])) + bf_lo(v[2][j])) + bf_lo(v[3][j]);
    float hi = ((bf_hi(v[0][j]) + bf_hi(v[1][j])) + bf_hi(v[2][j])) + bf_hi(v[3][j]);
    o[j] = f2bf_bits(lo) | (f2bf_bits(hi) << 16);
  }
  out[(long)t * n16 + c] = o;
  const u32x4 s4 = {SENT32, SENT32, SENT32, SENT32};
#pragma unroll
  for (int r = 0; r < 4; r++) __builtin_nontemporal_store(s4, rowp[r] + c);
  if (stats && spin_stats) {
    unsigned m = wave_max(spins);
    if ((threadIdx.x & 63) == 0) {
      atomicMax(stats + 1, m);
      if (m) atomicAdd(stats + 2, 1u);
    }
  }
}

// ------------------------------------------------------------------ fused consumer + HC combine (+ norm)
// modes: bit0 exp2-based sigmoid (Triton's exp lowering) vs expf; bit1 fma in the combine;
//        bit2 fma in the affine; bit3 rsqrt via 1/sqrtf instead of rsqrtf.
__device__ __forceinline__ float sigmoid_m(float x, int modes) {
  const float e = (modes & 1) ? exp2f(-x * 1.4426950408889634f) : expf(-x);
  return 1.f / (1.f + e);
}

template <int HC, bool NORM>
__global__ void __launch_bounds__(1024)
hc_ar_fused_k(const u32x4* __restrict__ res, long s_res16,
              const uint16_t* __restrict__ inj, long s_inj,
              const u32x4* __restrict__ w, int w_shared,
              u32x4* __restrict__ out, long s_out16,
              u32x4* __restrict__ y, long s_y16,
              float eps, int hd,
              u32x4* site_ptr, int src_stride16, int n16,
              unsigned* stats, unsigned max_spin, int site, int spin_stats, int modes) {
  const int t = blockIdx.x, c = threadIdx.x;
  const int wave = c >> 6, lane = c & 63, nwaves = (n16 + 63) >> 6;
  __shared__ float red[HC][16];
  if (c >= n16) return;   // block == n16 in practice; kept for safety (no barrier reached below then)

  // ---- (A) issue the AR-independent loads first: residual, injection logits, norm weight ----
  u32x4 rw[HC], ww[HC];
  const u32x4 z4 = {0u, 0u, 0u, 0u};
  if (modes & 16) {
#pragma unroll
    for (int s = 0; s < HC; s++) { rw[s] = z4; ww[s] = z4; }
  } else {
#pragma unroll
    for (int s = 0; s < HC; s++) rw[s] = res[t * s_res16 + (long)s * n16 + c];
    if constexpr (NORM) {
#pragma unroll
      for (int s = 0; s < HC; s++) ww[s] = w_shared ? w[c] : w[(long)s * n16 + c];
    }
  }
  float injf[HC];
  {
    const uint16_t* ip = inj + (long)t * s_inj;
#pragma unroll
    for (int s = 0; s < HC; s++) injf[s] = bfbits2f(ip[s]);
  }

  // ---- (B) the AR: poll the four source rows, sum in rank order, one RNE to bf16 ----
  u32x4* rowp[4];
#pragma unroll
  for (int r = 0; r < 4; r++) rowp[r] = site_ptr + (long)r * src_stride16 + (long)t * n16;
  u32x4 v[4];
  unsigned spins = 0;
  if (modes & 128) {
#pragma unroll
    for (int r = 0; r < 4; r++) v[r] = z4;
  } else {
    ar_poll(v, rowp, c, t, stats, max_spin, site, spins);
  }
  float blk[8];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float lo = ((bf_lo(v[0][j]) + bf_lo(v[1][j])) + bf_lo(v[2][j])) + bf_lo(v[3][j]);
    float hi = ((bf_hi(v[0][j]) + bf_hi(v[1][j])) + bf_hi(v[2][j])) + bf_hi(v[3][j]);
    blk[2 * j] = bfbits2f(f2bf_bits(lo));      // the stock path stores the AR output as bf16
    blk[2 * j + 1] = bfbits2f(f2bf_bits(hi));
  }
  const u32x4 s4 = {SENT32, SENT32, SENT32, SENT32};
#pragma unroll
  for (int r = 0; r < 4; r++) __builtin_nontemporal_store(s4, rowp[r] + c);   // re-arm (own 16 B)
  if (stats && spin_stats) {
    unsigned m = wave_max(spins);
    if (lane == 0) {
      atomicMax(stats + 1, m);
      if (m) atomicAdd(stats + 2, 1u);
    }
  }

  // ---- (C) combine: out[s] = bf16(res[s] + blk * 2*sigmoid(inj[s]/HC)); partial sum of squares ----
  float sq[HC];
  u32x4 ob[HC];
#pragma unroll
  for (int s = 0; s < HC; s++) {
    const float g = 2.0f * sigmoid_m(injf[s] / (float)HC, modes);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float rlo = bf_lo(rw[s][j]), rhi = bf_hi(rw[s][j]);
      float olo, ohi;
      if (modes & 2) { olo = __builtin_fmaf(blk[2 * j], g, rlo); ohi = __builtin_fmaf(blk[2 * j + 1], g, rhi); }
      else { olo = rlo + blk[2 * j] * g; ohi = rhi + blk[2 * j + 1] * g; }
      const unsigned blo = f2bf_bits(olo), bhi = f2bf_bits(ohi);
      ob[s][j] = blo | (bhi << 16);
      if constexpr (NORM) {
        const float flo = bfbits2f(blo), fhi = bfbits2f(bhi);
        acc += flo * flo;
        acc += fhi * fhi;
      }
    }
    sq[s] = acc;
    out[t * s_out16 + (long)s * n16 + c] = ob[s];
  }
  if constexpr (!NORM) return;
  if (modes & 32) return;

  // ---- (D) HC sum-of-squares across the block: wave shuffle tree, then LDS across waves ----
  if (!(modes & 64)) {
#pragma unroll
    for (int s = 0; s < HC; s++) sq[s] = wave_sum(sq[s]);
  }
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < HC; s++) red[s][wave] = sq[s];
  }
  __syncthreads();
  float rrms[HC];
#pragma unroll
  for (int s = 0; s < HC; s++) {
    float tot = 0.f;
    for (int q = 0; q < nwaves; q++) tot += red[s][q];
    const float ms = tot / (float)hd + eps;
    rrms[s] = (modes & 8) ? (1.0f / sqrtf(ms)) : rsqrtf(ms);
  }

  // ---- (E) y[s] = bf16(out[s] * rrms[s] * (1 + w[s])) ----
#pragma unroll
  for (int s = 0; s < HC; s++) {
    u32x4 yo;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float olo = bf_lo(ob[s][j]), ohi = bf_hi(ob[s][j]);
      const float wlo = bf_lo(ww[s][j]), whi = bf_hi(ww[s][j]);
      float ylo = olo * rrms[s], yhi = ohi * rrms[s];
      if (modes & 4) { ylo = __builtin_fmaf(ylo, wlo, ylo); yhi = __builtin_fmaf(yhi, whi, yhi); }
      else { ylo = ylo + ylo * wlo; yhi = yhi + yhi * whi; }
      yo[j] = f2bf_bits(ylo) | (f2bf_bits(yhi) << 16);
    }
    y[t * s_y16 + (long)s * n16 + c] = yo;
  }
}

// ------------------------------------------------------------------ split variant: grid (T, HC)
// One workgroup per (row, stream), like the Triton kernel's decomposition, so at T = 1 the four
// streams' combine/norm chains run on four CUs instead of serially on one.  Every WG of a row polls
// the full row's four source rows (redundant 4 x 20 KB of local UC reads -- negligible), and the
// slot is re-armed by the LAST workgroup of the row to finish reading it: a self-resetting
// per-(site, row) counter (atomicAdd after the poll; the last arriver rewrites the sentinels and
// zeroes the counter).  No workgroup ever waits on another.
template <int HC, bool NORM>
__global__ void __launch_bounds__(1024)
hc_ar_fused_split_k(const u32x4* __restrict__ res, long s_res16,
                    const uint16_t* __restrict__ inj, long s_inj,
                    const u32x4* __restrict__ w, int w_shared,
                    u32x4* __restrict__ out, long s_out16,
                    u32x4* __restrict__ y, long s_y16,
                    float eps, int hd,
                    u32x4* site_ptr, int src_stride16, int n16,
                    unsigned* stats, unsigned max_spin, int site, int spin_stats, int modes,
                    int* __restrict__ cnt, int cnt_stride) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x;
  const int wave = c >> 6, lane = c & 63, nwaves = (n16 + 63) >> 6;
  __shared__ float red[16];
  __shared__ int last_s;
  if (c >= n16) return;

  // (A) AR-independent loads first
  const u32x4 rw = res[t * s_res16 + (long)s * n16 + c];
  const float injf = bfbits2f(inj[(long)t * s_inj + s]);
  u32x4 ww = {0u, 0u, 0u, 0u};
  if constexpr (NORM) ww = w_shared ? w[c] : w[(long)s * n16 + c];

  // (B) poll the four source rows of row t
  u32x4* rowp[4];
#pragma unroll
  for (int r = 0; r < 4; r++) rowp[r] = site_ptr + (long)r * src_stride16 + (long)t * n16;
  u32x4 v[4];
  unsigned spins = 0;
  ar_poll(v, rowp, c, t, stats, max_spin, site, spins);
  float blk[8];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float lo = ((bf_lo(v[0][j]) + bf_lo(v[1][j])) + bf_lo(v[2][j])) + bf_lo(v[3][j]);
    float hi = ((bf_hi(v[0][j]) + bf_hi(v[1][j])) + bf_hi(v[2][j])) + bf_hi(v[3][j]);
    blk[2 * j] = bfbits2f(f2bf_bits(lo));
    blk[2 * j + 1] = bfbits2f(f2bf_bits(hi));
  }
  // (C) arrival counter.  The atomic is *issued* here (this WG's poll data is in registers, so
  // signalling is safe) but its result is only read after the barrier that the norm reduction
  // needs anyway, so the L2 round trip hides behind the combine.  The last-arriving stream WG of
  // the row re-arms the slot, at the very end of the kernel.
  __syncthreads();
  if (c == 0) last_s = atomicAdd(cnt + (long)site * cnt_stride + t, 1);
  if (stats && spin_stats) {
    unsigned m = wave_max(spins);
    if (lane == 0) {
      atomicMax(stats + 1, m);
      if (m) atomicAdd(stats + 2, 1u);
    }
  }

  // (D) combine for stream s
  const float g = 2.0f * sigmoid_m(injf / (float)HC, modes);
  float sq = 0.f;
  u32x4 ob;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float rlo = bf_lo(rw[j]), rhi = bf_hi(rw[j]);
    float olo, ohi;
    if (modes & 2) { olo = __builtin_fmaf(blk[2 * j], g, rlo); ohi = __builtin_fmaf(blk[2 * j + 1], g, rhi); }
    else { olo = rlo + blk[2 * j] * g; ohi = rhi + blk[2 * j + 1] * g; }
    const unsigned blo = f2bf_bits(olo), bhi = f2bf_bits(ohi);
    ob[j] = blo | (bhi << 16);
    if constexpr (NORM) {
      const float flo = bfbits2f(blo), fhi = bfbits2f(bhi);
      sq += flo * flo;
      sq += fhi * fhi;
    }
  }
  out[t * s_out16 + (long)s * n16 + c] = ob;
  if constexpr (!NORM) {
    __syncthreads();          // publishes last_s
    if (last_s == HC - 1) {
      const u32x4 s4 = {SENT32, SENT32, SENT32, SENT32};
#pragma unroll
      for (int r = 0; r < 4; r++) __builtin_nontemporal_store(s4, rowp[r] + c);
      if (c == 0) cnt[(long)site * cnt_stride + t] = 0;
    }
    return;
  }

  // (E) sum of squares over the stream, rrms, affine
  sq = wave_sum(sq);
  if (lane == 0) red[wave] = sq;
  __syncthreads();            // also publishes last_s
  float tot = 0.f;
  for (int q = 0; q < nwaves; q++) tot += red[q];
  const float ms = tot / (float)hd + eps;
  const float rrms = (modes & 8) ? (1.0f / sqrtf(ms)) : rsqrtf(ms);
  u32x4 yo;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float olo = bf_lo(ob[j]), ohi = bf_hi(ob[j]);
    const float wlo = bf_lo(ww[j]), whi = bf_hi(ww[j]);
    float ylo = olo * rrms, yhi = ohi * rrms;
    if (modes & 4) { ylo = __builtin_fmaf(ylo, wlo, ylo); yhi = __builtin_fmaf(yhi, whi, yhi); }
    else { ylo = ylo + ylo * wlo; yhi = yhi + yhi * whi; }
    yo[j] = f2bf_bits(ylo) | (f2bf_bits(yhi) << 16);
  }
  y[t * s_y16 + (long)s * n16 + c] = yo;
  if (last_s == HC - 1) {     // last stream WG of this row re-arms the slot and resets the counter
    const u32x4 s4 = {SENT32, SENT32, SENT32, SENT32};
#pragma unroll
    for (int r = 0; r < 4; r++) __builtin_nontemporal_store(s4, rowp[r] + c);
    if (c == 0) cnt[(long)site * cnt_stride + t] = 0;
  }
}

// ------------------------------------------------------------------ bindings
void consume(at::Tensor out, int64_t site_ptr, int64_t src_stride_elems, at::Tensor stats,
             int64_t max_spin, int64_t site, int64_t spin_stats) {
  TORCH_CHECK(out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.dim() == 2);
  const int T = (int)out.size(0), N = (int)out.size(1);
  TORCH_CHECK(N % 8 == 0 && N / 8 <= 1024 && src_stride_elems % 8 == 0 && (long)T * N <= src_stride_elems);
  unsigned* st = (stats.defined() && stats.numel() > 0) ? (unsigned*)stats.data_ptr() : nullptr;
  {
    consume_k<<<T, N / 8, 0, cur_stream()>>>((u32x4*)out.data_ptr(), (u32x4*)site_ptr,
                                             (int)(src_stride_elems / 8), N / 8, st,
                                             (unsigned)max_spin, (int)site, (int)spin_stats);
  }
}

// res [T, HC*HD] bf16 (row stride free), inj [T, HC] bf16, w [HC*HD] or [HD] bf16, out/y [T, HC*HD]
// bf16 (row strides free).  The AR message is [T, HD] (the reduced block output); site_ptr /
// src_stride_elems / stats / max_spin / site / spin_stats exactly as gfx908_push_ar.consume.
static void fused_impl(at::Tensor res, at::Tensor inj, const at::Tensor* w, at::Tensor out,
                       const at::Tensor* y, double eps, int64_t hc, int64_t site_ptr,
                       int64_t src_stride_elems, at::Tensor stats, int64_t max_spin, int64_t site,
                       int64_t spin_stats, int64_t modes, const at::Tensor* cnt) {
  TORCH_CHECK(hc == 4, "hc_ar_fused: HC == 4 only (template)");
  TORCH_CHECK(res.dim() == 2 && res.scalar_type() == at::kBFloat16 && res.stride(1) == 1);
  const int T = (int)res.size(0), DIM = (int)res.size(1), HD = DIM / (int)hc;
  TORCH_CHECK(DIM % hc == 0 && HD % 8 == 0 && HD / 8 <= 1024, "hc_ar_fused: HD must be a multiple of 8, <= 8192");
  TORCH_CHECK(inj.dim() == 2 && inj.size(0) == T && inj.size(1) == hc && inj.stride(1) == 1 &&
              inj.scalar_type() == at::kBFloat16);
  TORCH_CHECK(out.sizes() == res.sizes() && out.stride(1) == 1 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(res.stride(0) % 8 == 0 && out.stride(0) % 8 == 0, "row strides must keep 16 B alignment");
  TORCH_CHECK(src_stride_elems % 8 == 0 && (long)T * HD <= src_stride_elems, "message does not fit the slot");
  unsigned* st = (stats.defined() && stats.numel() > 0) ? (unsigned*)stats.data_ptr() : nullptr;
  const int n16 = HD / 8;
  auto strm = cur_stream();
  if (cnt) {   // split variant: grid (T, HC), per-(site,row) self-resetting counters
    TORCH_CHECK(cnt->scalar_type() == at::kInt && cnt->is_contiguous() && cnt->dim() == 2 &&
                cnt->size(1) >= T && site < cnt->size(0), "cnt must be int32 [sites, rows_max]");
    int* cp = (int*)cnt->data_ptr();
    const int cs = (int)cnt->size(1);
    dim3 grid(T, 4);
    if (y) {
      TORCH_CHECK(w && w->is_contiguous() && w->scalar_type() == at::kBFloat16 &&
                  (w->numel() == DIM || w->numel() == HD));
      TORCH_CHECK(y->sizes() == res.sizes() && y->stride(1) == 1 && y->stride(0) % 8 == 0 &&
                  y->scalar_type() == at::kBFloat16);
      hc_ar_fused_split_k<4, true><<<grid, n16, 0, strm>>>(
          (const u32x4*)res.data_ptr(), res.stride(0) / 8, (const uint16_t*)inj.data_ptr(), inj.stride(0),
          (const u32x4*)w->data_ptr(), w->numel() == HD ? 1 : 0,
          (u32x4*)out.data_ptr(), out.stride(0) / 8, (u32x4*)y->data_ptr(), y->stride(0) / 8,
          (float)eps, HD, (u32x4*)site_ptr, (int)(src_stride_elems / 8), n16, st,
          (unsigned)max_spin, (int)site, (int)spin_stats, (int)modes, cp, cs);
    } else {
      hc_ar_fused_split_k<4, false><<<grid, n16, 0, strm>>>(
          (const u32x4*)res.data_ptr(), res.stride(0) / 8, (const uint16_t*)inj.data_ptr(), inj.stride(0),
          nullptr, 0, (u32x4*)out.data_ptr(), out.stride(0) / 8, nullptr, 0,
          (float)eps, HD, (u32x4*)site_ptr, (int)(src_stride_elems / 8), n16, st,
          (unsigned)max_spin, (int)site, (int)spin_stats, (int)modes, cp, cs);
    }
    return;
  }
  if (y) {
    TORCH_CHECK(w && w->is_contiguous() && w->scalar_type() == at::kBFloat16 &&
                (w->numel() == DIM || w->numel() == HD));
    TORCH_CHECK(y->sizes() == res.sizes() && y->stride(1) == 1 && y->stride(0) % 8 == 0 &&
                y->scalar_type() == at::kBFloat16);
    hc_ar_fused_k<4, true><<<T, n16, 0, strm>>>(
        (const u32x4*)res.data_ptr(), res.stride(0) / 8, (const uint16_t*)inj.data_ptr(), inj.stride(0),
        (const u32x4*)w->data_ptr(), w->numel() == HD ? 1 : 0,
        (u32x4*)out.data_ptr(), out.stride(0) / 8, (u32x4*)y->data_ptr(), y->stride(0) / 8,
        (float)eps, HD, (u32x4*)site_ptr, (int)(src_stride_elems / 8), n16, st,
        (unsigned)max_spin, (int)site, (int)spin_stats, (int)modes);
  } else {
    hc_ar_fused_k<4, false><<<T, n16, 0, strm>>>(
        (const u32x4*)res.data_ptr(), res.stride(0) / 8, (const uint16_t*)inj.data_ptr(), inj.stride(0),
        nullptr, 0, (u32x4*)out.data_ptr(), out.stride(0) / 8, nullptr, 0,
        (float)eps, HD, (u32x4*)site_ptr, (int)(src_stride_elems / 8), n16, st,
        (unsigned)max_spin, (int)site, (int)spin_stats, (int)modes);
  }
}

void hc_ar_combine_norm(at::Tensor res, at::Tensor inj, at::Tensor w, at::Tensor out, at::Tensor y,
                        double eps, int64_t hc, int64_t site_ptr, int64_t src_stride_elems,
                        at::Tensor stats, int64_t max_spin, int64_t site, int64_t spin_stats,
                        int64_t modes) {
  fused_impl(res, inj, &w, out, &y, eps, hc, site_ptr, src_stride_elems, stats, max_spin, site,
             spin_stats, modes, nullptr);
}
void hc_ar_combine(at::Tensor res, at::Tensor inj, at::Tensor out, int64_t hc, int64_t site_ptr,
                   int64_t src_stride_elems, at::Tensor stats, int64_t max_spin, int64_t site,
                   int64_t spin_stats, int64_t modes) {
  fused_impl(res, inj, nullptr, out, nullptr, 0.0, hc, site_ptr, src_stride_elems, stats, max_spin,
             site, spin_stats, modes, nullptr);
}
// Split variants (grid (T, HC)); cnt: int32 [sites, rows_max] zero-initialised, self-resetting.
void hc_ar_combine_norm_split(at::Tensor res, at::Tensor inj, at::Tensor w, at::Tensor out,
                              at::Tensor y, double eps, int64_t hc, int64_t site_ptr,
                              int64_t src_stride_elems, at::Tensor stats, int64_t max_spin,
                              int64_t site, int64_t spin_stats, int64_t modes, at::Tensor cnt) {
  fused_impl(res, inj, &w, out, &y, eps, hc, site_ptr, src_stride_elems, stats, max_spin, site,
             spin_stats, modes, &cnt);
}
void hc_ar_combine_split(at::Tensor res, at::Tensor inj, at::Tensor out, int64_t hc, int64_t site_ptr,
                         int64_t src_stride_elems, at::Tensor stats, int64_t max_spin, int64_t site,
                         int64_t spin_stats, int64_t modes, at::Tensor cnt) {
  fused_impl(res, inj, nullptr, out, nullptr, 0.0, hc, site_ptr, src_stride_elems, stats, max_spin,
             site, spin_stats, modes, &cnt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_u32", &fill_u32);
  m.def("alloc_uncached", &alloc_uncached);
  m.def("free_ptr", &free_ptr);
  m.def("memcpy_to", &memcpy_to);
  m.def("memcpy_from", &memcpy_from);
  m.def("push", &push);
  m.def("consume", &consume);
  m.def("hc_ar_combine_norm", &hc_ar_combine_norm);
  m.def("hc_ar_combine", &hc_ar_combine);
  m.def("hc_ar_combine_norm_split", &hc_ar_combine_norm_split);
  m.def("hc_ar_combine_split", &hc_ar_combine_split);
}
