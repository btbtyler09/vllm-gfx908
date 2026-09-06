// gfx908 fused PLE decode glue (Qwen3.8-Flash-Next).
//
// One workgroup per (token, hyper-connection stream) does, for M <= a few
// tokens, everything the PLE layer body launches after key_proj / value_proj:
//
//   norm_key / norm_query (grouped RMS, group == hidden_size)
//   gate = sigmoid(sign(d) * sqrt(max(|d|, 1e-6))),  d = <nk, nq> / sqrt(H)
//   gated_value = gate * value            (broadcast over the 4 HC streams)
//   norm_conv (grouped RMS over gated_value) -> bf16 `normalized`
//   dilated depthwise short conv from the conv state + silu
//   conv state shift-and-write-back
//   out = hidden + (gated_value + conv_out)
//
// Numerics target: what inductor compiles the shipping python into with
// VLLM_MI100_TORCH_COMPILE=1 (see REPORT.md) -- one fp32 chain with a bf16
// round only where a buffer is materialised (`normalized`, the conv output,
// the layer output).  `1/sqrt(H)` is a multiply by 0.01976423537605237f, as
// inductor's reciprocal-folded constant.
//
// The three reductions of the compiled `triton_red_..._0` kernel (sum key^2,
// sum query^2, dot) are reproduced element-for-element: XBLOCK=1,
// R0_BLOCK=2048, 8 warps, so lane j of the [1, 2048] accumulator owns
// x[j] (+ x[2048 + j] when j < 512), each thread owns 4 consecutive lanes,
// and the tree is (intra-thread sequential) -> (64-lane butterfly 32..1) ->
// (8-warp butterfly 4..1).

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

#define NT 512          // threads per workgroup (8 wavefronts, as Triton used)
#define WAVE 64
#define NW (NT / WAVE)
#define RBLOCK 2048     // Triton R0_BLOCK of the fused reduction kernel
#define MAXK 8
#define MAXS 16     // max conv_state_len supported in registers

__device__ __forceinline__ float bf2f(unsigned short h) {
  union { unsigned int u; float f; } c;
  c.u = ((unsigned int)h) << 16;
  return c.f;
}

__device__ __forceinline__ unsigned short f2bf(float f) {
  union { float f; unsigned int u; } c;
  c.f = f;
  if ((c.u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((c.u >> 16) | 0x0040u);
  unsigned int r = c.u + 0x7fffu + ((c.u >> 16) & 1u);
  return (unsigned short)(r >> 16);
}

// 4 consecutive bf16 as one 8-byte load (the reduction's lane layout owns 4
// consecutive elements, so scalar loads would fetch 4x the bytes they use).
__device__ __forceinline__ void ld4(const unsigned short* p, int j, int n, float o[4]) {
  if (j + 3 < n && ((((size_t)p + 2 * (size_t)j) & 7) == 0)) {
    const ushort4 v = *reinterpret_cast<const ushort4*>(p + j);
    o[0] = bf2f(v.x); o[1] = bf2f(v.y); o[2] = bf2f(v.z); o[3] = bf2f(v.w);
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = (j + k < n) ? bf2f(p[j + k]) : 0.0f;
  }
}

// (intra-thread sequential) -> 64-lane butterfly -> NW-warp butterfly, the
// order the compiled reduction uses.  Each call gets its own LDS slot so one
// __syncthreads per reduction is enough (no trailing barrier to protect reuse).
__device__ __forceinline__ float wave_reduce(float s) {
#pragma unroll
  for (int off = WAVE / 2; off >= 1; off >>= 1) s += __shfl_xor(s, off, WAVE);
  return s;
}

__device__ __forceinline__ float cross_wave(const float* slot) {
  float p[NW];
#pragma unroll
  for (int i = 0; i < NW; ++i) p[i] = slot[i];
#pragma unroll
  for (int off = NW / 2; off >= 1; off >>= 1) {
    float q[NW];
#pragma unroll
    for (int i = 0; i < NW; ++i) q[i] = p[i] + p[i ^ off];
#pragma unroll
    for (int i = 0; i < NW; ++i) p[i] = q[i];
  }
  return p[0];
}

__device__ __forceinline__ float tri_red4(const float a[4], float* slot) {
  float s = wave_reduce(((a[0] + a[1]) + a[2]) + a[3]);
  if ((threadIdx.x & (WAVE - 1)) == 0) slot[threadIdx.x >> 6] = s;
  __syncthreads();
  return cross_wave(slot);
}

// two independent reductions sharing one barrier
__device__ __forceinline__ void tri_red4x2(const float a[4], const float b[4],
                                           float* slot, float* out0, float* out1) {
  const float sa = wave_reduce(((a[0] + a[1]) + a[2]) + a[3]);
  const float sb = wave_reduce(((b[0] + b[1]) + b[2]) + b[3]);
  if ((threadIdx.x & (WAVE - 1)) == 0) {
    slot[threadIdx.x >> 6] = sa;
    slot[NW + (threadIdx.x >> 6)] = sb;
  }
  __syncthreads();
  *out0 = cross_wave(slot);
  *out1 = cross_wave(slot + NW);
}

// gate chain, exactly as triton_poi_fused_..._5 emits it
__device__ __forceinline__ float ple_gate(float d, float inv_sqrt_h) {
  const float gd = d * inv_sqrt_h;
  const float sgn = (float)((int)(0.0f < gd) - (int)(gd < 0.0f));
  const float ab = fabsf(gd);
  const float mm = (ab != ab) ? ab : fmaxf(ab, 1e-6f);   // PropagateNan
  const float m = sgn * sqrtf(mm);
  return 1.0f / (1.0f + expf(-m));
}


// One workgroup per (token, hc stream, channel split).  Each workgroup redoes
// the (cheap, 20 KB) group reductions and owns HG/SPLIT of the channels, so a
// one-token step has SPLIT x HC workgroups instead of HC: the conv-state
// traffic (the bulk of the bytes) is spread over more CUs and, because the
// state and conv weights are loaded *before* the first __syncthreads, it
// overlaps the reduction's dependent round trips.
template <int SKSZ, int SDIL, int ITERS>
__global__ __launch_bounds__(NT) void ple_glue_k(
    const unsigned short* __restrict__ hidden,   // [T, C]  (query + residual)
    const unsigned short* __restrict__ key,      // [T, C]
    const unsigned short* __restrict__ value,    // [T, HG]
    const unsigned short* __restrict__ wq,       // [C]
    const unsigned short* __restrict__ wk,       // [C]
    const unsigned short* __restrict__ wc,       // [C]
    const unsigned short* __restrict__ conv_w,   // [C, K] contiguous
    unsigned short* __restrict__ cstate,         // [slots, C, cap] strided
    const int64_t* __restrict__ sidx64,
    const int32_t* __restrict__ sidx32,
    const bool* __restrict__ has_init,
    unsigned short* __restrict__ out,            // [T, C]
    unsigned short* __restrict__ normalized,     // [T, C] or nullptr (debug)
    float* __restrict__ dbg,                     // [T*HC, 4] or nullptr (debug)
    int HG, int HC, int64_t s_slot, int64_t s_ch, int64_t s_j,
    int slen, int ksz, int dil, int64_t null_id, int64_t nslots, float eps, float inv_sqrt_h,
    int fuse_res) {
  __shared__ float smem[4 * NW];
  constexpr int SL = (SKSZ - 1) * SDIL;      // conv_state_len, compile time
  const int SPLIT = HG / (NT * ITERS);

  const int blk = blockIdx.x;
  const int t = blk / (HC * SPLIT);
  const int rem = blk - t * HC * SPLIT;
  const int g = rem / SPLIT;
  const int sp = rem - g * SPLIT;
  const long C = (long)HG * HC;
  const int tid = threadIdx.x;
  const int w = tid >> 6, l = tid & (WAVE - 1);

  const unsigned short* kp = key + (long)t * C + (long)g * HG;
  const unsigned short* qp = hidden + (long)t * C + (long)g * HG;
  const unsigned short* wkp = wk + (long)g * HG;
  const unsigned short* wqp = wq + (long)g * HG;
  const unsigned short* wcp = wc + (long)g * HG;
  const unsigned short* vp = value + (long)t * HG;
  unsigned short* op = out + (long)t * C + (long)g * HG;

  // ---- the channels this workgroup owns, and everything they need ---------
  long slot = sidx64 ? sidx64[t] : (long)sidx32[t];
  // padded / dummy rows may carry -1 or out-of-range block ids: treat them as null
  const bool valid = (slot != null_id) && (slot >= 0) && (slot < nslots);
  const bool hinit = valid && (has_init ? has_init[t] : true);
  if (!valid) slot = 0;  // never index the state with an unvalidated slot

  int mi[ITERS];
  float stv[ITERS][SL > 0 ? SL : 1], cwv[ITERS][SKSZ];
  float vmain[ITERS], wcmain[ITERS], qmain[ITERS];
  unsigned short* spv[ITERS];
#pragma unroll
  for (int c = 0; c < ITERS; ++c) {
    const int i = sp * (NT * ITERS) + c * NT + tid;
    mi[c] = i;
    const int ch = g * HG + i;
    spv[c] = cstate + slot * s_slot + (long)ch * s_ch;
#pragma unroll
    for (int j = 0; j < SL; ++j) stv[c][j] = (valid && hinit) ? bf2f(spv[c][(long)j * s_j]) : 0.0f;
    ld4(conv_w + (long)ch * SKSZ, 0, SKSZ, cwv[c]);
    vmain[c] = bf2f(vp[i]);
    wcmain[c] = bf2f(wcp[i]);
    qmain[c] = fuse_res ? bf2f(qp[i]) : 0.0f;
  }

  // ---- reductions in the compiled kernel's lane layout --------------------
  // lane j = w*256 + l*4 + k ; the loop adds j + RBLOCK when still in range
  // every input of every reduction is loaded here, before the first barrier,
  // so the whole kernel takes a single dependent memory round trip
  float kacc[4], qacc[4], kv[4], qv[4], kv2[4], qv2[4];
  float wkv[4], wqv[4], wkv2[4], wqv2[4], vr[4], vr2[4];
  const int j0 = w * 256 + l * 4, j1 = j0 + RBLOCK;
  ld4(kp, j0, HG, kv);   ld4(qp, j0, HG, qv);
  ld4(wkp, j0, HG, wkv); ld4(wqp, j0, HG, wqv); ld4(vp, j0, HG, vr);
  ld4(kp, j1, HG, kv2);  ld4(qp, j1, HG, qv2);
  ld4(wkp, j1, HG, wkv2); ld4(wqp, j1, HG, wqv2); ld4(vp, j1, HG, vr2);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    kacc[k] = kv[k] * kv[k] + kv2[k] * kv2[k];
    qacc[k] = qv[k] * qv[k] + qv2[k] * qv2[k];
  }
  float sk, sq;
  tri_red4x2(kacc, qacc, smem, &sk, &sq);
  const float hf = (float)HG;
  const float rkn = rsqrtf(sk / hf + eps);
  const float rqn = rsqrtf(sq / hf + eps);

  float dacc[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int j = j0 + k, j2 = j1 + k;
    float a = 0.0f, b = 0.0f, a2 = 0.0f, b2 = 0.0f;
    if (j < HG) {
      a = (kv[k] * rkn) * (1.0f + wkv[k]);
      b = (qv[k] * rqn) * (1.0f + wqv[k]);
    }
    if (j2 < HG) {
      a2 = (kv2[k] * rkn) * (1.0f + wkv2[k]);
      b2 = (qv2[k] * rqn) * (1.0f + wqv2[k]);
    }
    dacc[k] = a * b + a2 * b2;
  }
  // eager `.sum(dim=-1)` on a bf16 tensor yields bf16; inductor keeps that
  // buffer in bf16 and reloads it as fp32 for the gate chain.
  const float dot = bf2f(f2bf(tri_red4(dacc, smem + 2 * NW)));
  const float gate = ple_gate(dot, inv_sqrt_h);
  if (dbg && tid == 0 && sp == 0) {
    dbg[(t * HC + g) * 4 + 0] = sk; dbg[(t * HC + g) * 4 + 1] = sq;
    dbg[(t * HC + g) * 4 + 2] = dot; dbg[(t * HC + g) * 4 + 3] = gate;
  }

  float vacc[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int j = j0 + k, j2 = j1 + k;
    const float g1 = (j < HG) ? gate * vr[k] : 0.0f;
    const float g2 = (j2 < HG) ? gate * vr2[k] : 0.0f;
    vacc[k] = g1 * g1 + g2 * g2;
  }
  const float rc = rsqrtf(tri_red4(vacc, smem + 3 * NW) / hf + eps);

  // ---- conv + state write-back + output -----------------------------------
#pragma unroll
  for (int c = 0; c < ITERS; ++c) {
    const float gv = gate * vmain[c];
    const unsigned short xb = f2bf((gv * rc) * (1.0f + wcmain[c]));
    if (normalized) normalized[(long)t * C + (long)g * HG + mi[c]] = xb;
    float co = 0.0f;
    if (valid) {
      float acc = 0.0f;
#pragma unroll
      for (int jk = 0; jk < SKSZ - 1; ++jk) acc += cwv[c][jk] * stv[c][jk * SDIL];
      acc += cwv[c][SKSZ - 1] * bf2f(xb);
      // F.conv1d materialises its bf16 output before F.silu reads it back
      const float accb = bf2f(f2bf(acc));
      co = bf2f(f2bf(accb / (1.0f + expf(-accb))));
#pragma unroll
      for (int j = 0; j + 1 < SL; ++j) spv[c][(long)j * s_j] = f2bf(stv[c][j + 1]);
      spv[c][(long)(SL - 1) * s_j] = xb;
    }
    op[mi[c]] = f2bf(qmain[c] + (gv + co));
  }
}

// ---------------------------------------------------------------------------
void ple_glue(at::Tensor hidden, at::Tensor key, at::Tensor value,
              at::Tensor wq, at::Tensor wk, at::Tensor wc, at::Tensor conv_w,
              at::Tensor conv_state, at::Tensor state_idx,
              c10::optional<at::Tensor> has_init, at::Tensor out,
              c10::optional<at::Tensor> normalized, c10::optional<at::Tensor> dbg,
              int64_t hc, int64_t dil, int64_t null_id, double eps, bool fuse_res) {
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16);
  TORCH_CHECK(value.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && conv_w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(hidden.is_contiguous() && key.is_contiguous() && value.is_contiguous());
  TORCH_CHECK(out.is_contiguous() && conv_w.is_contiguous());
  TORCH_CHECK(wq.is_contiguous() && wk.is_contiguous() && wc.is_contiguous());
  const int T = (int)hidden.size(0);
  const long C = hidden.size(1);
  TORCH_CHECK(C % hc == 0);
  const int HG = (int)(C / hc);
  TORCH_CHECK(value.size(1) == HG && key.size(1) == C && out.size(1) == C);
  TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == C);
  const int ksz = (int)conv_w.size(1);
  TORCH_CHECK(ksz >= 1 && ksz <= MAXK);
  const int slen = (ksz - 1) * (int)dil;
  TORCH_CHECK(conv_state.size(2) >= slen);
  TORCH_CHECK(slen >= 1 && slen <= MAXS, "conv_state_len out of range");
  TORCH_CHECK(HG % NT == 0, "hidden_size must be a multiple of ", NT);
  TORCH_CHECK(HG <= 2 * RBLOCK, "HG > 4096 not supported by the fused reduction layout");
  if (T == 0) return;

  const int64_t* s64 = nullptr;
  const int32_t* s32 = nullptr;
  if (state_idx.scalar_type() == at::kLong) {
    s64 = state_idx.data_ptr<int64_t>();
  } else {
    TORCH_CHECK(state_idx.scalar_type() == at::kInt);
    s32 = state_idx.data_ptr<int32_t>();
  }
  const bool* hi = nullptr;
  if (has_init.has_value() && has_init->defined()) {
    TORCH_CHECK(has_init->scalar_type() == at::kBool);
    hi = has_init->data_ptr<bool>();
  }
  unsigned short* np = nullptr;
  if (normalized.has_value() && normalized->defined()) {
    np = reinterpret_cast<unsigned short*>(normalized->data_ptr<at::BFloat16>());
  }
  float* dbgp = nullptr;
  if (dbg.has_value() && dbg->defined()) dbgp = dbg->data_ptr<float>();
  auto st = c10::hip::getCurrentHIPStream().stream();
  // channel split: enough workgroups to cover the machine at small T
  int split = 1;
  const char* env = std::getenv("VLLM_GFX908_PLE_GLUE_ITERS");
  const int force = env ? atoi(env) : 0;
  // Optional cap on the per-lane iteration count.  Default 0 = no cap:
  // -Rpass-analysis=kernel-resource-usage reports ScratchSize 0 bytes/lane and no
  // spills for ITERS 1/2/4/5, so there is nothing to work around here (the
  // 2026-09-06 HSA 0x1016 fault was this op being captured into the piecewise
  // cudagraph, not a scratch-backed dispatch).  The cap has to be applied *after*
  // the selection loop: skipping candidates inside it made the loop fall through
  // to split = 1, i.e. ITERS = HG/NT = 5, whenever T*hc*split > 240 (T >= 13) --
  // exactly the configuration the cap was meant to exclude.
  const char* env_max = std::getenv("VLLM_GFX908_PLE_GLUE_MAX_ITERS");
  const int max_it = env_max ? atoi(env_max) : 0;
  for (int cand : {1, 2, 4, 5}) {
    if (HG % (NT * cand)) continue;
    if (force ? (cand == force) : ((long)T * hc * (HG / (NT * cand)) <= 240)) {
      split = HG / (NT * cand);
      break;
    }
  }
  if (!force && max_it > 0 && HG / (NT * split) > max_it) {
    TORCH_CHECK(HG % (NT * max_it) == 0, "VLLM_GFX908_PLE_GLUE_MAX_ITERS=", max_it,
                " does not divide hidden_size / ", NT);
    split = HG / (NT * max_it);
  }
  const int iters = HG / (NT * split);
  const int grid = T * (int)hc * split;
#define PLE_LAUNCH(A, B, IT)                                          \
  {                                                                   \
    ple_glue_k<A, B, IT><<<grid, NT, 0, st>>>(                         \
        reinterpret_cast<const unsigned short*>(hidden.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const unsigned short*>(key.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const unsigned short*>(value.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const unsigned short*>(wq.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const unsigned short*>(wk.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const unsigned short*>(wc.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const unsigned short*>(conv_w.data_ptr<at::BFloat16>()), \
        reinterpret_cast<unsigned short*>(conv_state.data_ptr<at::BFloat16>()), \
        s64, s32, hi, \
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()), np, dbgp, \
        HG, (int)hc, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), \
        slen, ksz, (int)dil, null_id, (int64_t)conv_state.size(0), (float)eps, \
        (float)(1.0 / std::sqrt((double)HG)), fuse_res ? 1 : 0 \
        );                                                            \
  }
#define PLE_ITERS(A, B)                                               \
  {                                                                   \
    if (iters == 1) { PLE_LAUNCH(A, B, 1) }                           \
    else if (iters == 2) { PLE_LAUNCH(A, B, 2) }                      \
    else if (iters == 4) { PLE_LAUNCH(A, B, 4) }                      \
    else if (iters == 5) { PLE_LAUNCH(A, B, 5) }                      \
    else { TORCH_CHECK(false, "unsupported HG/NT split: ", iters); }   \
  }
  if (ksz == 4 && dil == 3) {
    PLE_ITERS(4, 3)
  } else if (ksz == 4 && dil == 1) {
    PLE_ITERS(4, 1)
  } else {
    TORCH_CHECK(false, "ple_glue: unsupported (ksz, dil) = (", ksz, ", ", dil, ")");
  }
#undef PLE_ITERS
#undef PLE_LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ple_glue", &ple_glue); }
