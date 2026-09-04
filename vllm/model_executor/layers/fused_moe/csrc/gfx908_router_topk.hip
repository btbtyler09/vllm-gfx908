// Fused MoE router for gfx908 (MI100): bf16 GEMV (x[M,K] . W[E,K]^T) + fp32 softmax over E=512 experts
// + top-10 selection, reproducing vLLM's ops.topk_softmax (vllm::moe::topkGating) semantics:
//   * softmax in fp32: max, expf(x-max), sum, multiply by 1/sum; NaN/Inf -> 0
//   * k passes of argmax, ties -> lowest expert index, selected slot masked to -10000
//   * renormalize: w *= 1/sum(selected) (sum>0 else 1)
//   * is_padding[m] -> expert ids -1 (weights still written)
//   * token_expert_indices[m][k] = k*M + m
//
// Kernel 1 (router_topk_fused): grid = 32 column tiles x KSO K-slices, WPG waves per WG; each wave computes a
// 16(row) x 16(col) tile over a K-slice with v_mfma_f32_16x16x8bf16 (no cross-lane reduction needed), the WG
// reduces its waves through LDS, writes fp32 partials, bumps an atomic arrival counter. The last R arriving WGs
// are the finalizers (R = ceil(M / WPG)): they spin until all WGs arrived, then each finalizer WG sums the KSO
// partials for WPG rows (one wave per row) and runs the wave-level softmax+top-k. Counters are reset in-kernel
// by the last finalizer, so the kernel is cudagraph-replay safe.
// Kernel 2 (topk_softmax standalone): one wave per row, same finalize, reads bf16 or fp32 logits.
//
// Wave-level top-k (single wave, ~60 instructions per pass, branch-free): 8 softmax values per lane
// (cols 4*lane+i and 256+4*lane+i, same layout as topkGating); per pass: v_max3 local max, 6-step DPP
// row_shr/row_bcast max (folded v_max_f32_dpp via inline asm) + v_readlane, per-lane lowest matching column via
// v_cmp/v_cndmask, 6-step DPP min on the column, then the winning column's slot is masked to -10000.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>
#include <cmath>

namespace {

constexpr int E = 512;      // experts (fixed: finalize holds 8 per lane)
constexpr int TOPK = 10;
constexpr int NT = 16;      // MFMA tile rows (max M per tile)

typedef short v2s __attribute__((ext_vector_type(2)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------- wave reductions (gfx9 DPP, folded VOP2 dpp)
// s_nop 1 supplies the 2 wait states a DPP read needs after the VALU write of its source.
#define DPP_STEP(OP, CTRL, EXTRA) \
  asm("s_nop 1\n\t" OP "_dpp %0, %0, %0 " CTRL " row_mask:0xf bank_mask:0xf" EXTRA : "+v"(v))
#define DPP_REDUCE(OP, EXTRA)                       \
  DPP_STEP(OP, "row_shr:1", EXTRA);                 \
  DPP_STEP(OP, "row_shr:2", EXTRA);                 \
  DPP_STEP(OP, "row_shr:4", EXTRA);                 \
  DPP_STEP(OP, "row_shr:8", EXTRA);                 \
  DPP_STEP(OP, "row_bcast:15", EXTRA);              \
  DPP_STEP(OP, "row_bcast:31", EXTRA);

__device__ __forceinline__ float readlane63(float v) {
  return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 63));
}
// full-wave max, result uniform. Invalid DPP source lanes are disabled (no bound_ctrl) -> dest keeps v.
__device__ __forceinline__ float wave_max(float v) {
  DPP_REDUCE("v_max_f32", "")
  return readlane63(v);
}
__device__ __forceinline__ int wave_min_u32(int v) {
  DPP_REDUCE("v_min_u32", "")
  return __builtin_amdgcn_readlane(v, 63);
}
__device__ __forceinline__ int wave_max_i32(int v) {
  DPP_REDUCE("v_max_i32", "")
  return __builtin_amdgcn_readlane(v, 63);
}
// sum: bound_ctrl:0 (LLVM syntax) = zero-fill invalid source lanes
__device__ __forceinline__ float wave_sum_dpp(float v) {
  DPP_REDUCE("v_add_f32", " bound_ctrl:0")
  return readlane63(v);
}
// bit-identical to topkGating's butterfly (xor 32,16,...,1 via ds_bpermute)
__device__ __forceinline__ float wave_sum_xor(float v) {
#pragma unroll
  for (int m = 32; m > 0; m >>= 1) v += __shfl_xor(v, m, 64);
  return v;
}

// ---------------------------------------------------------------- softmax + top-k for one row, one wave
// v[i]: lane holds experts 4*lane+i (i<4) and 256+4*lane+(i-4) (i>=4) -- identical to topkGating<VPT=8,512>.
__device__ __forceinline__ void softmax_topk_row(float (&v)[8], const int lane, const bool pad, const bool renorm,
                                                 const bool exact_sum, const bool fast_exp,
                                                 float* __restrict__ w_out, int* __restrict__ id_out,
                                                 int* __restrict__ src_out, const int m, const int num_rows) {
  float lm = fmaxf(fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])), fmaxf(fmaxf(v[4], v[5]), fmaxf(v[6], v[7])));
  const float mx = wave_max(lm);
  float s = 0.f;
  if (fast_exp) {
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = __expf(v[i] - mx); s += v[i]; }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = expf(v[i] - mx); s += v[i]; }
  }
  s = exact_sum ? wave_sum_xor(s) : wave_sum_dpp(s);
  const float inv = 1.f / s;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] *= inv;
    if (isnan(v[i]) || isinf(v[i])) v[i] = 0.f;
  }
  int col[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) col[i] = (i < 4) ? (4 * lane + i) : (256 + 4 * lane + (i - 4));
  // Selection on the fp32 bit patterns as int32: softmax values are >= 0 (no -0.0, NaN/Inf clamped), so int order
  // == float order, v_max3_i32 needs no NaN canonicalisation, and a masked slot is -1 (< every valid value).
  int iv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) iv[i] = __float_as_int(v[i]);
  float sel_v[TOPK];
  int sel_i[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const int lm = max(max(max(iv[0], iv[1]), max(iv[2], iv[3])), max(max(iv[4], iv[5]), max(iv[6], iv[7])));
    const int wm = wave_max_i32(lm);
    int cand = 0x7fffffff;
#pragma unroll
    for (int i = 7; i >= 0; --i) cand = (iv[i] == wm) ? col[i] : cand;  // lowest matching column in this lane
    const int widx = wave_min_u32(cand);                                   // lowest matching column in the wave
    sel_v[k] = __int_as_float(wm);
    sel_i[k] = widx;
#pragma unroll
    for (int i = 0; i < 8; ++i) iv[i] = (col[i] == widx) ? -1 : iv[i];
  }
  float scale = 1.f;
  if (renorm) {
    float ssum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) ssum += sel_v[k];
    scale = 1.f / (ssum > 0.f ? ssum : 1.f);
  }
  // lanes 0..9 each write one (weight, id, source row) -- 3 store instructions instead of 30
  float wv = sel_v[0];
  int wi = sel_i[0];
#pragma unroll
  for (int k = 1; k < TOPK; ++k) { wv = (lane == k) ? sel_v[k] : wv; wi = (lane == k) ? sel_i[k] : wi; }
  if (lane < TOPK) {
    w_out[lane] = wv * scale;
    id_out[lane] = pad ? -1 : wi;
    if (src_out) src_out[lane] = lane * num_rows + m;
  }
}

__device__ __forceinline__ float bf16_round(float f) { return __bfloat162float(__float2bfloat16(f)); }

// ---------------------------------------------------------------- kernel 1: fused GEMV + finalize
// partials: [KSO][NT][E] fp32 ; counter: 2 x uint32 (zero-initialised once, reset in-kernel)
// flags: bit0 renormalize, bit1 round logits to bf16, bit2 exact (xor) sum, bit3 fast exp
template <int WPG, int KSO, int STEPS, bool DO_TOPK>
__global__ void __launch_bounds__(64 * WPG) router_topk_fused_kernel(
    const uint16_t* __restrict__ X, const int ldx, const uint16_t* __restrict__ W, const int K, const int M,
    float* __restrict__ partials, unsigned* __restrict__ counter, float* __restrict__ topk_w,
    int* __restrict__ topk_id, int* __restrict__ src_rows, const bool* __restrict__ is_pad, const int flags,
    const int R, float* __restrict__ logits_out) {
  __shared__ float red[WPG][NT][16];
  __shared__ int fin_s;
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int ctile = blockIdx.x % (E / 16), kso = blockIdx.x / (E / 16);
  const int r = lane & 15, q = lane >> 4;
  const int kbeg = (kso * WPG + wave) * (STEPS * 32);

  // ---- GEMV tile: rows = x rows (clamped to M-1), cols = ctile*16 + r, K-slice [kbeg, kbeg + STEPS*32)
  const v8s* wp = reinterpret_cast<const v8s*>(W + (size_t)(ctile * 16 + r) * K + kbeg + q * 8);
  const v8s* xp = reinterpret_cast<const v8s*>(X + (size_t)min(r, M - 1) * ldx + kbeg + q * 8);
  v8s wv[STEPS], xv[STEPS];
#pragma unroll
  for (int s = 0; s < STEPS; ++s) {
    wv[s] = __builtin_nontemporal_load(wp + s * 4);  // 64 B stride per step (32 bf16)
    xv[s] = xp[s * 4];
  }
  v4f acc0 = {0.f, 0.f, 0.f, 0.f}, acc1 = {0.f, 0.f, 0.f, 0.f};
#define PAIR(V, J) __builtin_shufflevector(V, V, 2 * (J), 2 * (J) + 1)
#pragma unroll
  for (int s = 0; s < STEPS; ++s) {
    // MFMA j consumes k-pair j of every lane: lane (r,q) supplies k = kbeg + s*32 + 8q + 2j + {0,1} for A and B
    acc0 = __builtin_amdgcn_mfma_f32_16x16x8bf16(PAIR(xv[s], 0), PAIR(wv[s], 0), acc0, 0, 0, 0);
    acc1 = __builtin_amdgcn_mfma_f32_16x16x8bf16(PAIR(xv[s], 1), PAIR(wv[s], 1), acc1, 0, 0, 0);
    acc0 = __builtin_amdgcn_mfma_f32_16x16x8bf16(PAIR(xv[s], 2), PAIR(wv[s], 2), acc0, 0, 0, 0);
    acc1 = __builtin_amdgcn_mfma_f32_16x16x8bf16(PAIR(xv[s], 3), PAIR(wv[s], 3), acc1, 0, 0, 0);
  }
#undef PAIR
  // D layout: lane (r=col, q) holds rows 4q+i, i=0..3
#pragma unroll
  for (int i = 0; i < 4; ++i) red[wave][4 * q + i][r] = acc0[i] + acc1[i];
  __syncthreads();
  // ---- reduce the WPG waves, write fp32 partial tile
  for (int t = threadIdx.x; t < NT * 16; t += 64 * WPG) {
    const int m = t >> 4, c = t & 15;
    if (m < M) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < WPG; ++w) s += red[w][m][c];
      partials[((size_t)kso * NT + m) * E + ctile * 16 + c] = s;
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) fin_s = (int)atomicAdd(counter, 1u) - (int)(gridDim.x - R);  // >= 0: finalizer index
  __syncthreads();
  const int f = fin_s;
  if (f < 0) return;
  if (threadIdx.x == 0) {  // wait for the remaining arrivals (at most R-1 WGs still running)
    while (__hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < gridDim.x) __builtin_amdgcn_s_sleep(1);
  }
  __syncthreads();
  __threadfence();

  // ---- finalize: finalizer f handles rows [f*WPG, f*WPG+WPG), one wave per row
  const int m = f * WPG + wave;
  if (m < M) {
    const float* row = partials + (size_t)m * E;
    v4f a = __builtin_nontemporal_load(reinterpret_cast<const v4f*>(row + 4 * lane));
    v4f b = __builtin_nontemporal_load(reinterpret_cast<const v4f*>(row + 256 + 4 * lane));
#pragma unroll
    for (int j = 1; j < KSO; ++j) {
      a += __builtin_nontemporal_load(reinterpret_cast<const v4f*>(row + (size_t)j * NT * E + 4 * lane));
      b += __builtin_nontemporal_load(reinterpret_cast<const v4f*>(row + (size_t)j * NT * E + 256 + 4 * lane));
    }
    float v[8] = {a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
    if (flags & 2) {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = bf16_round(v[i]);
    }
    if (logits_out) {
      *reinterpret_cast<v4f*>(logits_out + (size_t)m * E + 4 * lane) = v4f{v[0], v[1], v[2], v[3]};
      *reinterpret_cast<v4f*>(logits_out + (size_t)m * E + 256 + 4 * lane) = v4f{v[4], v[5], v[6], v[7]};
    }
    if constexpr (DO_TOPK) {
      const bool pad = (is_pad != nullptr) && is_pad[m];
      softmax_topk_row(v, lane, pad, flags & 1, flags & 4, flags & 8, topk_w + (size_t)m * TOPK,
                       topk_id + (size_t)m * TOPK, src_rows ? src_rows + (size_t)m * TOPK : nullptr, m, M);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(counter + 1, 1u) == (unsigned)(R - 1)) { counter[0] = 0u; counter[1] = 0u; }  // last finalizer resets
  }
}

// ---------------------------------------------------------------- kernel 2: standalone top-k (one wave per row)
template <typename T>
__global__ void __launch_bounds__(64) topk_softmax_kernel(const T* __restrict__ logits, const int ld, const int M,
                                                          float* __restrict__ topk_w, int* __restrict__ topk_id,
                                                          int* __restrict__ src_rows, const bool* __restrict__ is_pad,
                                                          const int flags) {
  const int m = blockIdx.x, lane = threadIdx.x;
  const T* row = logits + (size_t)m * ld;
  float v[8];
  if constexpr (sizeof(T) == 4) {
    const v4f a = *reinterpret_cast<const v4f*>(row + 4 * lane);
    const v4f b = *reinterpret_cast<const v4f*>(row + 256 + 4 * lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = a[i]; v[4 + i] = b[i]; }
  } else {  // bf16: two 8-byte loads
    typedef unsigned v2u __attribute__((ext_vector_type(2)));
    const v2u a = *reinterpret_cast<const v2u*>(row + 4 * lane);
    const v2u b = *reinterpret_cast<const v2u*>(row + 256 + 4 * lane);
    v[0] = __uint_as_float(a[0] << 16); v[1] = __uint_as_float(a[0] & 0xffff0000u);
    v[2] = __uint_as_float(a[1] << 16); v[3] = __uint_as_float(a[1] & 0xffff0000u);
    v[4] = __uint_as_float(b[0] << 16); v[5] = __uint_as_float(b[0] & 0xffff0000u);
    v[6] = __uint_as_float(b[1] << 16); v[7] = __uint_as_float(b[1] & 0xffff0000u);
  }
  const bool pad = (is_pad != nullptr) && is_pad[m];
  softmax_topk_row(v, lane, pad, flags & 1, flags & 4, flags & 8, topk_w + (size_t)m * TOPK,
                   topk_id + (size_t)m * TOPK, src_rows ? src_rows + (size_t)m * TOPK : nullptr, m, M);
}

__global__ void noop_kernel(int* p) { if (p == nullptr && threadIdx.x == 12345) *p = 0; }

// ---------------------------------------------------------------- host
template <int WPG, int KSO, int STEPS, bool DO_TOPK>
void launch_fused(const uint16_t* X, int ldx, const uint16_t* W, int K, int M, float* partials, unsigned* counter,
                  float* tw, int* ti, int* sr, const bool* pad, int flags, float* logits, hipStream_t st) {
  TORCH_CHECK(K == STEPS * 32 * WPG * KSO, "router_topk_fused: K must equal ", STEPS * 32 * WPG * KSO, " for cfg");
  dim3 grid((E / 16) * KSO), block(64 * WPG);
  const int R = min((M + WPG - 1) / WPG, (int)grid.x);
  router_topk_fused_kernel<WPG, KSO, STEPS, DO_TOPK><<<grid, block, 0, st>>>(
      X, ldx, W, K, M, partials, counter, tw, ti, sr, pad, flags, R, logits);
}

template <bool DO_TOPK>
void dispatch_fused(int cfg, const uint16_t* X, int ldx, const uint16_t* W, int K, int M, float* partials,
                    unsigned* counter, float* tw, int* ti, int* sr, const bool* pad, int flags, float* logits,
                    hipStream_t st) {
#define CFG(ID, WPG, KSO, STEPS) \
  if (cfg == ID) { launch_fused<WPG, KSO, STEPS, DO_TOPK>(X, ldx, W, K, M, partials, counter, tw, ti, sr, pad, flags, logits, st); return; }
  // K = 2560 = STEPS*32*WPG*KSO. grid = 32*KSO WGs of WPG waves.
  CFG(0, 4, 4, 5)    // 128 WGs x 4 waves, 5 steps
  CFG(1, 8, 2, 5)    //  64 WGs x 8 waves
  CFG(2, 16, 1, 5)   //  32 WGs x 16 waves
  CFG(3, 8, 1, 10)   //  32 WGs x 8 waves
  CFG(4, 4, 2, 10)   //  64 WGs x 4 waves
  CFG(5, 2, 4, 10)   // 128 WGs x 2 waves
  CFG(6, 1, 4, 20)   // 128 WGs x 1 wave
#undef CFG
  TORCH_CHECK(false, "router_topk_fused: unknown cfg ", cfg);
}

}  // namespace

// x [M, K] bf16 (row stride may exceed K), w [E=512, K] bf16 contiguous, topk_w [M,10] f32, topk_id [M,10] i32,
// src_rows optional [M,10] i32, is_pad optional [M] bool, partials f32 workspace >= 8*16*512 floats,
// counter i32[2] zero-initialised, logits_out optional [M,512] f32. cfg: see dispatch_fused. M <= 16.
// flags: bit0 renormalize, bit1 round logits to bf16 before softmax (matches the bf16-GEMV pipeline),
//        bit2 exact xor-butterfly sum (bit-identical to topkGating), bit3 fast exp (__expf)
void router_topk_fused(at::Tensor x, at::Tensor w, at::Tensor topk_w, at::Tensor topk_id,
                       c10::optional<at::Tensor> src_rows, c10::optional<at::Tensor> is_pad, at::Tensor partials,
                       at::Tensor counter, int64_t flags, c10::optional<at::Tensor> logits_out, int64_t cfg,
                       bool do_topk) {
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, "bf16 x / w");
  TORCH_CHECK(w.is_contiguous() && w.size(0) == E && w.size(1) == K, "w must be [512, K] contiguous");
  TORCH_CHECK(x.stride(1) == 1 && (x.stride(0) % 8) == 0, "x rows must be 16B aligned");
  TORCH_CHECK(M >= 1 && M <= NT, "M <= 16");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_id.scalar_type() == at::kInt && topk_id.size(1) == TOPK);
  TORCH_CHECK(partials.scalar_type() == at::kFloat && partials.numel() >= 8 * NT * E, "partials >= 8*16*512 floats");
  TORCH_CHECK(counter.scalar_type() == at::kInt && counter.numel() >= 2, "counter: int32[2]");
  auto st = c10::hip::getCurrentHIPStream().stream();
  const uint16_t* X = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* Wp = reinterpret_cast<const uint16_t*>(w.data_ptr());
  int* sr = src_rows.has_value() ? src_rows->data_ptr<int>() : nullptr;
  const bool* pad = is_pad.has_value() ? is_pad->data_ptr<bool>() : nullptr;
  float* lo = logits_out.has_value() ? logits_out->data_ptr<float>() : nullptr;
  unsigned* cnt = reinterpret_cast<unsigned*>(counter.data_ptr<int>());
  if (do_topk)
    dispatch_fused<true>((int)cfg, X, (int)x.stride(0), Wp, K, M, partials.data_ptr<float>(), cnt,
                         topk_w.data_ptr<float>(), topk_id.data_ptr<int>(), sr, pad, (int)flags, lo, st);
  else
    dispatch_fused<false>((int)cfg, X, (int)x.stride(0), Wp, K, M, partials.data_ptr<float>(), cnt,
                          topk_w.data_ptr<float>(), topk_id.data_ptr<int>(), sr, pad, (int)flags, lo, st);
}

// logits [M, 512] fp32 or bf16 (row stride arbitrary, 16B aligned), one wave per row. flags as above (bit1 unused).
void topk_softmax(at::Tensor logits, at::Tensor topk_w, at::Tensor topk_id, c10::optional<at::Tensor> src_rows,
                  c10::optional<at::Tensor> is_pad, int64_t flags) {
  const int M = logits.size(0);
  TORCH_CHECK(logits.size(1) == E && logits.stride(1) == 1, "logits must be [M, 512]");
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_id.scalar_type() == at::kInt && topk_id.size(1) == TOPK);
  auto st = c10::hip::getCurrentHIPStream().stream();
  int* sr = src_rows.has_value() ? src_rows->data_ptr<int>() : nullptr;
  const bool* pad = is_pad.has_value() ? is_pad->data_ptr<bool>() : nullptr;
  float* tw = topk_w.data_ptr<float>();
  int* ti = topk_id.data_ptr<int>();
  const int ld = (int)logits.stride(0);
  if (logits.scalar_type() == at::kFloat) {
    topk_softmax_kernel<float><<<M, 64, 0, st>>>(logits.data_ptr<float>(), ld, M, tw, ti, sr, pad, (int)flags);
  } else {
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be fp32 or bf16");
    topk_softmax_kernel<uint16_t><<<M, 64, 0, st>>>(reinterpret_cast<const uint16_t*>(logits.data_ptr()), ld, M, tw,
                                                    ti, sr, pad, (int)flags);
  }
}

void noop(at::Tensor t, int64_t blocks) {
  noop_kernel<<<blocks, 64, 0, c10::hip::getCurrentHIPStream().stream()>>>(t.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("noop", &noop);
  m.def("router_topk_fused", &router_topk_fused);
  m.def("topk_softmax", &topk_softmax);
}
