// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// gfx908 (MI100) fused GDN decode step: causal conv1d update (3-tap ring, silu)
// + gated delta-rule recurrence (fp32 state read-modify-write) + gated RMSNorm,
// one launch per layer for M <= 8 non-spec decode tokens, plus a speculative
// (MTP verify / draft) variant below that walks the n+1 tokens of each sequence.  Replaces the five
// launches zero_ / z copy / causal_conv1d_update / fused_recurrent_packed_decode
// / RMSNormGated between in_proj and out_proj.  JIT-built by gfx908_gdn_fused.py.
//
// Shapes are fixed for the Qwen3.8-Flash-Next rank layout: H=4 K-heads, HV=12
// V-heads, K=V=128, conv width 4, qkvz row = [q 512 | k 512 | v 1536 | z 1536].
//
// Work decomposition: grid (NV, HV, M), block ROWS*4 threads.  A block owns
// ROWS full V-rows (128 fp32 each) of one (token, V-head) state tile; lane
// (row, kseg) holds 32 K-elements of its row as 8 float4 (interleaved so that
// each wave-level dwordx4 touches 16 full 64 B lines).  Every block recomputes
// the 3-tap conv + silu + l2norm for its K-head's q/k (256 values) and its own
// v rows.  Cross-block hazards are handled with self-resetting counters:
//   * the q/k conv-state shift is written by the LAST block of the
//     (token, K-head) group to have read the old taps (3*NV blocks);
//   * with NV > 1 the gated RMSNorm (needs the full 128-wide o row) is done by
//     the LAST block of the (token, V-head) group, reading the other blocks'
//     bf16-rounded o values from a small fp32 scratch.
// No block ever waits on another, so there is no residency/deadlock concern.
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

namespace {

constexpr int H = 4, HV = 12, K = 128, V = 128;
constexpr int GROUP = HV / H;          // V-heads per K-head
constexpr int Q_OFF = 0, K_OFF = H * K, V_OFF = 2 * H * K, Z_OFF = 2 * H * K + HV * V;
constexpr int LPR = 4;                 // lanes per V-row
constexpr int NCH = (K / LPR) / 4;     // float4 chunks per lane = 8

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float(((uint32_t)u) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);   // NaN
  u += 0x7fffu + ((u >> 16) & 1u);                                            // RNE
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }

// EXPMODE 0: libm expf/logf.  EXPMODE 1: v_exp_f32 / v_log_f32 based (what the
// Triton AMD backend emits for tl.exp / tl.log on f32).
template <int EXPMODE> __device__ __forceinline__ float ex(float x) {
  if constexpr (EXPMODE == 0) return expf(x);
  else return exp2f(x * 1.4426950408889634f);
}
template <int EXPMODE> __device__ __forceinline__ float lg(float x) {
  if constexpr (EXPMODE == 0) return logf(x);
  else return log2f(x) * 0.6931471805599453f;
}
template <int EXPMODE> __device__ __forceinline__ float silu_(float x) { return x / (1.f + ex<EXPMODE>(-x)); }
template <int EXPMODE> __device__ __forceinline__ float sigmoid_(float x) { return 1.f / (1.f + ex<EXPMODE>(-x)); }

// Same op order as _causal_conv1d_update_kernel: fp32 acc, each bf16*bf16
// product rounded to bf16 before accumulation (Triton bf16 binary-op semantics).
__device__ __forceinline__ float conv4(float bias, float t0, float t1, float t2, float x,
                                       float w0, float w1, float w2, float w3) {
  float acc = bias;
  acc += bfround(t0 * w0);
  acc += bfround(t1 * w1);
  acc += bfround(t2 * w2);
  acc += bfround(x * w3);
  return acc;
}

// DPP-based reductions (VALU row ops, no LDS round trips).  quad_perm(1,0,3,2)
// = 0xB1 (xor 1), quad_perm(2,3,0,1) = 0x4E (xor 2), row_half_mirror = 0x141,
// row_mirror = 0x140; after those four every lane of a 16-lane row holds the row
// sum, and four v_readlane + three adds finish the wave.
template <int CTRL> __device__ __forceinline__ float dpp(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true));
}
__device__ __forceinline__ float quad_sum(float v) {
  v += dpp<0xB1>(v);
  v += dpp<0x4E>(v);
  return v;
}
__device__ __forceinline__ float wave_sum(float v) {
  v = quad_sum(v);
  v += dpp<0x141>(v);
  v += dpp<0x140>(v);
  const float r0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0));
  const float r1 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
  const float r2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32));
  const float r3 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
  return ((r0 + r1) + r2) + r3;
}

// LDS-only barrier (no vmcnt wait, so in-flight global loads keep flying).
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

template <int ROWS, bool GATE_SIGMOID, int EXPMODE>
__global__ void __launch_bounds__(ROWS * LPR)
gdn_fused_kernel(const uint16_t* __restrict__ qkvz, int64_t s_qkvz,
                 const uint16_t* __restrict__ ba, int64_t s_ba,
                 const uint16_t* __restrict__ cw,          // [2560, 4] contiguous
                 const uint16_t* __restrict__ cbias,
                 uint16_t* __restrict__ cstate, int64_t s_cs_slot, int64_t s_cs_dim, int64_t s_cs_tok,
                 float* __restrict__ hstate, int64_t s_h_slot,
                 const int32_t* __restrict__ idx,
                 const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                 const float* __restrict__ nw, float eps, float scale,
                 uint16_t* __restrict__ out, float* __restrict__ scratch, int32_t* __restrict__ cnt,
                 int M, int dbg) {
  constexpr int NT = ROWS * LPR;
  constexpr int NV = V / ROWS;
  constexpr int QKC = 2 * K;
  constexpr int PER = (QKC + NT - 1) / NT;   // q/k conv columns per thread
  static_assert(NV * ROWS == V && NCH * 4 * LPR == K, "layout");

  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  const int i_v = blockIdx.x, i_hv = blockIdx.y, n = blockIdx.z, i_h = i_hv / GROUP;
  const int row0 = i_v * ROWS;
  const int row = tid / LPR, kseg = tid % LPR, grow = row0 + row;

  __shared__ __align__(16) float q_s[K];
  __shared__ __align__(16) float k_s[K];
  __shared__ float v_s[ROWS];
  __shared__ __align__(16) float o_s[V];
  __shared__ int flag_s[2];

  uint16_t* out_row = out + ((size_t)n * HV + i_hv) * V;
  const int slot = idx[n];
  if (slot <= 0) {  // PAD / null block: zero output, leave both states alone
    for (int c = tid; c < ROWS; c += NT) out_row[row0 + c] = 0;
    return;
  }
  const size_t cs_base = (size_t)slot * s_cs_slot;
  const uint16_t* xrow = qkvz + (size_t)n * s_qkvz;

  // ---- (B) prefetch this lane's 32 fp32 state elements (overlaps the conv prologue) ----
  float* hp = hstate + (size_t)slot * s_h_slot + ((size_t)i_hv * V + grow) * K + kseg * 4;
  float4 h[NCH];
  if (!(dbg & 2)) {
#pragma unroll
    for (int j = 0; j < NCH; ++j) h[j] = *reinterpret_cast<const float4*>(hp + j * 16);
  }
  // gate scalars (independent small loads)
  const float a_val = bf2f(ba[(size_t)n * s_ba + HV + i_hv]);
  const float b_val = bf2f(ba[(size_t)n * s_ba + i_hv]);
  const float dtb = dt_bias[i_hv], alog = A_log[i_hv];
  // norm inputs for this lane's two output columns (only wave 0 uses them; cheap to hoist)
  const uint16_t* zrow = xrow + Z_OFF + i_hv * V;
  const uint32_t zpk = *reinterpret_cast<const uint32_t*>(zrow + 2 * lane);
  const float2 nw2 = *reinterpret_cast<const float2*>(nw + 2 * lane);
  if (dbg & 2) {
#pragma unroll
    for (int j = 0; j < NCH; ++j) h[j] = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // ---- (A) issue every conv input load: old taps, x, packed weights ----
  uint16_t u0[PER], u1[PER], u2[PER], ux[PER];
  uint2 wq[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int c = tid + i * NT;
    u0[i] = u1[i] = u2[i] = ux[i] = 0; wq[i] = make_uint2(0, 0);
    if (c < QKC) {
      const int dim = (c < K) ? (Q_OFF + i_h * K + c) : (K_OFF + i_h * K + (c - K));
      const uint16_t* cs = cstate + cs_base + (size_t)dim * s_cs_dim;
      u0[i] = cs[0]; u1[i] = cs[s_cs_tok]; u2[i] = cs[2 * s_cs_tok]; ux[i] = xrow[dim];
      wq[i] = *reinterpret_cast<const uint2*>(cw + (size_t)dim * 4);
    }
  }
  uint16_t v0 = 0, v1 = 0, v2 = 0, vx = 0; uint2 wv = make_uint2(0, 0);
  uint16_t* vcs = nullptr;
  if (tid < ROWS) {
    const int dim = V_OFF + i_hv * V + row0 + tid;
    vcs = cstate + cs_base + (size_t)dim * s_cs_dim;
    v0 = vcs[0]; v1 = vcs[s_cs_tok]; v2 = vcs[2 * s_cs_tok]; vx = xrow[dim];
    wv = *reinterpret_cast<const uint2*>(cw + (size_t)dim * 4);
  }
  // (D) every old q/k tap this block will ever read is now in registers (the
  // fence waited for all loads, including the state prefetch that was issued
  // first, so it costs one latency, not two): count this block as done reading.
  // The atomic's return value is only consumed in the tail.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  int conv_old = 0;
  if (tid == 0) conv_old = atomicAdd(cnt + M * HV + n * H + i_h, 1);
  // ---- (C) conv + silu for q/k (256 columns) and this block's v rows ----
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int c = tid + i * NT;
    if (c < QKC) {
      const int dim = (c < K) ? (Q_OFF + i_h * K + c) : (K_OFF + i_h * K + (c - K));
      const float bias = cbias ? bf2f(cbias[dim]) : 0.f;
      float acc = conv4(bias, bf2f(u0[i]), bf2f(u1[i]), bf2f(u2[i]), bf2f(ux[i]),
                        bf2f(wq[i].x & 0xffffu), bf2f(wq[i].x >> 16), bf2f(wq[i].y & 0xffffu), bf2f(wq[i].y >> 16));
      acc = bfround(silu_<EXPMODE>(acc));   // stock stores the conv output as bf16
      if (c < K) q_s[c] = acc; else k_s[c - K] = acc;
    }
  }
  if (tid < ROWS) {
    const int dim = V_OFF + i_hv * V + row0 + tid;
    const float bias = cbias ? bf2f(cbias[dim]) : 0.f;
    float acc = conv4(bias, bf2f(v0), bf2f(v1), bf2f(v2), bf2f(vx),
                      bf2f(wv.x & 0xffffu), bf2f(wv.x >> 16), bf2f(wv.y & 0xffffu), bf2f(wv.y >> 16));
    v_s[tid] = bfround(silu_<EXPMODE>(acc));
    vcs[0] = v1; vcs[s_cs_tok] = v2; vcs[2 * s_cs_tok] = vx;   // own columns: shift now
  }
  lds_barrier();

  // ---- (E) l2norm(q), l2norm(k) once, in place in LDS (wave 0); gate ----
  if (wave == 0) {
    const float q0 = q_s[2 * lane], q1 = q_s[2 * lane + 1], k0 = k_s[2 * lane], k1 = k_s[2 * lane + 1];
    const float sq = wave_sum(q0 * q0 + q1 * q1), sk = wave_sum(k0 * k0 + k1 * k1);
    const float dq = sqrtf(sq + 1e-6f), dk = sqrtf(sk + 1e-6f);
    q_s[2 * lane] = (q0 / dq) * scale; q_s[2 * lane + 1] = (q1 / dq) * scale;   // same ops as Triton: (q / sqrt) * scale
    k_s[2 * lane] = k0 / dk;           k_s[2 * lane + 1] = k1 / dk;
  }
  lds_barrier();
  const float xg = a_val + dtb;
  const float sp = (xg <= 20.f) ? lg<EXPMODE>(1.f + ex<EXPMODE>(xg)) : xg;
  const float g = -ex<EXPMODE>(alog) * sp;
  const float beta = sigmoid_<EXPMODE>(b_val);
  const float eg = ex<EXPMODE>(g);
  const float vin = v_s[row];

  // ---- (F) recurrence; q/k slices re-read from LDS (keeps VGPRs low) ----
  float d = 0.f;
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const float4 k4 = *reinterpret_cast<const float4*>(k_s + j * 16 + kseg * 4);
    h[j].x *= eg; h[j].y *= eg; h[j].z *= eg; h[j].w *= eg;
    d += h[j].x * k4.x + h[j].y * k4.y + h[j].z * k4.z + h[j].w * k4.w;
  }
  d = quad_sum(d);
  const float vb = (vin - d) * beta;
  float o = 0.f;
  asm volatile("" ::: "memory");
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const float4 k4 = *reinterpret_cast<const float4*>(k_s + j * 16 + kseg * 4);
    const float4 q4 = *reinterpret_cast<const float4*>(q_s + j * 16 + kseg * 4);
    h[j].x += vb * k4.x; h[j].y += vb * k4.y; h[j].z += vb * k4.z; h[j].w += vb * k4.w;
    o += h[j].x * q4.x + h[j].y * q4.y + h[j].z * q4.z + h[j].w * q4.w;
  }
  o = quad_sum(o);
  const float ob = bfround(o);   // stock stores core_attn_out as bf16 before the norm

  // ---- (G) publish o and count BEFORE the 32 KB state store so the release
  //          fence only drains one dword per lane; then store the state ----
  if constexpr (NV == 1) {
    if (kseg == 0) o_s[row] = ob;
    if (tid == 0) { flag_s[0] = conv_old; }
  } else {
    if (kseg == 0) scratch[((size_t)n * HV + i_hv) * V + grow] = ob;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // scratch store visible before the count
    if (tid == 0) { flag_s[0] = conv_old; flag_s[1] = atomicAdd(cnt + n * HV + i_hv, 1); }
  }
  if (!(dbg & 2)) {
#pragma unroll
    for (int j = 0; j < NCH; ++j) *reinterpret_cast<float4*>(hp + j * 16) = h[j];
  }
  if (dbg & 1) return;   // ablation: no conv-state shift, no norm
  lds_barrier();
  const bool last_conv_reader = (flag_s[0] == GROUP * NV - 1);

  // ---- (H) deferred q/k conv-state shift (last reader of the K-head group) ----
  if (last_conv_reader) {
    if (tid == 0) cnt[M * HV + n * H + i_h] = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int c = tid + i * NT;
      if (c < QKC) {
        const int dim = (c < K) ? (Q_OFF + i_h * K + c) : (K_OFF + i_h * K + (c - K));
        uint16_t* cs = cstate + cs_base + (size_t)dim * s_cs_dim;
        cs[0] = u1[i]; cs[s_cs_tok] = u2[i]; cs[2 * s_cs_tok] = ux[i];
      }
    }
  }

  // ---- (I) gated RMSNorm: out = rmsnorm(o) * act(z) ----
  float ov[2];
  if constexpr (NV == 1) {
    if (wave != 0) return;
    ov[0] = o_s[2 * lane]; ov[1] = o_s[2 * lane + 1];
  } else {
    if (flag_s[1] != NV - 1 || wave != 0) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // invalidate L1: see the other blocks' scratch
    if (tid == 0) cnt[n * HV + i_hv] = 0;
    const float* sc = scratch + ((size_t)n * HV + i_hv) * V;
    ov[0] = __hip_atomic_load(sc + 2 * lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    ov[1] = __hip_atomic_load(sc + 2 * lane + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  float ss = ov[0] * ov[0] + ov[1] * ov[1];
  ss = wave_sum(ss);
  const float rstd = rsqrtf(ss / (float)V + eps);
  const float zz[2] = {bf2f((uint16_t)(zpk & 0xffffu)), bf2f((uint16_t)(zpk >> 16))};
  const float ww[2] = {nw2.x, nw2.y};
  uint32_t pk = 0;
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const float act = GATE_SIGMOID ? sigmoid_<EXPMODE>(zz[e]) : silu_<EXPMODE>(zz[e]);
    const float y = ((ov[e] * rstd) * ww[e]) * act;
    pk |= (uint32_t)f2bf(y) << (16 * e);
  }
  *reinterpret_cast<uint32_t*>(out_row + 2 * lane) = pk;
}


// ---------------------------------------------------------------------------
// Speculative-decode (MTP verify / draft) variant.  Grid (NV, HV, B) over the
// B sequences of the batch; a block walks the T_b = cu[b+1] - cu[b] tokens of
// its sequence in order (T_b <= TMAX, 0 => padded sequence, nothing to do).
// Semantics copied from the stock spec path (causal_conv1d_update with
// IS_SPEC_DECODING + fused_sigmoid_gating_delta_rule_update with
// INPLACE_FINAL_STATE), acc = num_accepted[b] (>= 1), n = TMAX - 1 drafts:
//   conv : block idx[b, 0] holds n + 3 taps; the 3 taps preceding x_0 sit at
//          columns [acc-1, acc+1]; the new state [old[acc], old[acc+1],
//          x_0 .. x_{T-1}] is written to columns [0, T+2) of the same block.
//   ssm  : h = state[idx[b, acc-1]] (rollback to the accepted position); after
//          every token t the state is stored to block idx[b, t] so the next
//          verify step can resume from whichever position was accepted.
//   norm : gated RMSNorm per token row; output rows in [cu[B], M) (cudagraph
//          padding) are zeroed by the blocks of the last sequence.
// q/k l2norm uses the rsqrt-multiply form of that Triton kernel.  Cross-block
// hazards use the same self-resetting last-arriver counters as the non-spec
// kernel: cnt[b*HV + hv] for the norm tail (NV > 1), cnt[B*HV + b*H + h] for
// the deferred q/k conv-state shift.  No block waits on another.
template <int ROWS, int TMAX, bool GATE_SIGMOID, int EXPMODE>
__global__ void __launch_bounds__(ROWS * LPR)
gdn_fused_spec_kernel(const uint16_t* __restrict__ qkvz, int64_t s_qkvz,
                      const uint16_t* __restrict__ ba, int64_t s_ba,
                      const uint16_t* __restrict__ cw, const uint16_t* __restrict__ cbias,
                      uint16_t* __restrict__ cstate, int64_t s_cs_slot, int64_t s_cs_dim, int64_t s_cs_tok,
                      float* __restrict__ hstate, int64_t s_h_slot,
                      const int32_t* __restrict__ idx, int64_t s_idx_seq, int64_t s_idx_tok,
                      const int32_t* __restrict__ cu, const int32_t* __restrict__ nacc,
                      const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                      const float* __restrict__ nw, float eps, float scale,
                      uint16_t* __restrict__ out, float* __restrict__ scratch, int32_t* __restrict__ cnt,
                      int M) {
  constexpr int NT = ROWS * LPR;
  constexpr int NV = V / ROWS;
  constexpr int NW = NT / 64;                // waves per block
  constexpr int QKC = 2 * K;
  constexpr int PER = (QKC + NT - 1) / NT;
  static_assert(NV * ROWS == V && NCH * 4 * LPR == K, "layout");

  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  const int i_v = blockIdx.x, i_hv = blockIdx.y, b = blockIdx.z, i_h = i_hv / GROUP;
  const int B = gridDim.z;
  const int row0 = i_v * ROWS;
  const int row = tid / LPR, kseg = tid % LPR, grow = row0 + row;

  __shared__ __align__(16) float q_s[TMAX][K];
  __shared__ __align__(16) float k_s[TMAX][K];
  __shared__ float v_s[TMAX][ROWS];
  __shared__ __align__(16) float o_s[NV == 1 ? TMAX : 1][V];
  __shared__ int flag_s[2];

  const int bos = cu[b];
  const int T = cu[b + 1] - bos;
  // (Z) cudagraph padding rows [cu[B], M): zero this block's slice of them.
  if (b == B - 1) {
    const int start = cu[B];
    for (int e = tid; e < (M - start) * ROWS; e += NT) {
      const int r = start + e / ROWS, c = e % ROWS;
      out[((size_t)r * HV + i_hv) * V + row0 + c] = 0;
    }
  }
  if (T <= 0 || T > TMAX) return;

  const int acc = nacc[b];
  const int off = acc - 1;                                   // 0 .. n
  const int slot0 = idx[(size_t)b * s_idx_seq];              // conv block (spec path uses column 0)
  const int slotH = (off >= 0 && off < TMAX) ? idx[(size_t)b * s_idx_seq + (size_t)off * s_idx_tok] : 0;
  uint16_t* out_seq = out + ((size_t)bos * HV + i_hv) * V + row0;   // token t row at + t*HV*V
  if (slot0 <= 0 || slotH <= 0) {   // not a live sequence (stock: conv skips null block, recurrent returns)
    for (int t = 0; t < T; ++t)
      for (int c = tid; c < ROWS; c += NT) out_seq[(size_t)t * HV * V + c] = 0;
    return;
  }
  const size_t cs_base = (size_t)slot0 * s_cs_slot;
  const uint16_t* xseq = qkvz + (size_t)bos * s_qkvz;        // token t row at + t*s_qkvz

  // ---- (B) prefetch this lane's 32 fp32 state elements from the accepted block ----
  const float* hp0 = hstate + (size_t)slotH * s_h_slot + ((size_t)i_hv * V + grow) * K + kseg * 4;
  float4 h[NCH];
#pragma unroll
  for (int j = 0; j < NCH; ++j) h[j] = *reinterpret_cast<const float4*>(hp0 + j * 16);
  const float dtb = dt_bias[i_hv], alog = A_log[i_hv];
  const float2 nw2 = *reinterpret_cast<const float2*>(nw + 2 * lane);
  // z (norm gate) for this lane's two output columns, every token
  uint32_t zpk[TMAX];
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    zpk[t] = 0;
    if (t < T) zpk[t] = *reinterpret_cast<const uint32_t*>(xseq + (size_t)t * s_qkvz + Z_OFF + i_hv * V + 2 * lane);
  }

  // ---- (A) conv inputs: 3 old taps at [off, off+2], x_0..x_{T-1}, packed weights ----
  uint16_t u0[PER], u1[PER], u2[PER], ux[PER][TMAX];
  uint2 wq[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int c = tid + i * NT;
    u0[i] = u1[i] = u2[i] = 0; wq[i] = make_uint2(0, 0);
#pragma unroll
    for (int t = 0; t < TMAX; ++t) ux[i][t] = 0;
    if (c < QKC) {
      const int dim = (c < K) ? (Q_OFF + i_h * K + c) : (K_OFF + i_h * K + (c - K));
      const uint16_t* cs = cstate + cs_base + (size_t)dim * s_cs_dim + (size_t)off * s_cs_tok;
      u0[i] = cs[0]; u1[i] = cs[s_cs_tok]; u2[i] = cs[2 * s_cs_tok];
#pragma unroll
      for (int t = 0; t < TMAX; ++t) if (t < T) ux[i][t] = xseq[(size_t)t * s_qkvz + dim];
      wq[i] = *reinterpret_cast<const uint2*>(cw + (size_t)dim * 4);
    }
  }
  uint16_t v0 = 0, v1 = 0, v2 = 0, vx[TMAX]; uint2 wv = make_uint2(0, 0);
  uint16_t* vcs = nullptr;
#pragma unroll
  for (int t = 0; t < TMAX; ++t) vx[t] = 0;
  if (tid < ROWS) {
    const int dim = V_OFF + i_hv * V + row0 + tid;
    vcs = cstate + cs_base + (size_t)dim * s_cs_dim;
    const uint16_t* cs = vcs + (size_t)off * s_cs_tok;
    v0 = cs[0]; v1 = cs[s_cs_tok]; v2 = cs[2 * s_cs_tok];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) if (t < T) vx[t] = xseq[(size_t)t * s_qkvz + dim];
    wv = *reinterpret_cast<const uint2*>(cw + (size_t)dim * 4);
  }
  // (D) all old q/k taps this block reads are in registers: count as done reading.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  int conv_old = 0;
  if (tid == 0) conv_old = atomicAdd(cnt + B * HV + b * H + i_h, 1);

  // ---- (C) sliding-window conv + silu per token (stock: taps shift by one each token) ----
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int c = tid + i * NT;
    if (c < QKC) {
      const int dim = (c < K) ? (Q_OFF + i_h * K + c) : (K_OFF + i_h * K + (c - K));
      const float bias = cbias ? bf2f(cbias[dim]) : 0.f;
      const float w0 = bf2f(wq[i].x & 0xffffu), w1 = bf2f(wq[i].x >> 16), w2 = bf2f(wq[i].y & 0xffffu), w3 = bf2f(wq[i].y >> 16);
      float s0 = bf2f(u0[i]), s1 = bf2f(u1[i]), s2 = bf2f(u2[i]);
#pragma unroll
      for (int t = 0; t < TMAX; ++t) {
        if (t < T) {
          const float x = bf2f(ux[i][t]);
          float acc_ = conv4(bias, s0, s1, s2, x, w0, w1, w2, w3);
          acc_ = bfround(silu_<EXPMODE>(acc_));
          if (c < K) q_s[t][c] = acc_; else k_s[t][c - K] = acc_;
          s0 = s1; s1 = s2; s2 = x;
        }
      }
    }
  }
  if (tid < ROWS) {
    const int dim = V_OFF + i_hv * V + row0 + tid;
    const float bias = cbias ? bf2f(cbias[dim]) : 0.f;
    const float w0 = bf2f(wv.x & 0xffffu), w1 = bf2f(wv.x >> 16), w2 = bf2f(wv.y & 0xffffu), w3 = bf2f(wv.y >> 16);
    float s0 = bf2f(v0), s1 = bf2f(v1), s2 = bf2f(v2);
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
      if (t < T) {
        const float x = bf2f(vx[t]);
        float acc_ = conv4(bias, s0, s1, s2, x, w0, w1, w2, w3);
        v_s[t][tid] = bfround(silu_<EXPMODE>(acc_));
        s0 = s1; s1 = s2; s2 = x;
      }
    }
    // own v columns: new state [old[off+1], old[off+2], x_0 .. x_{T-1}] at columns [0, T+2)
    vcs[0] = v1; vcs[s_cs_tok] = v2;
#pragma unroll
    for (int t = 0; t < TMAX; ++t) if (t < T) vcs[(size_t)(2 + t) * s_cs_tok] = vx[t];
  }
  lds_barrier();

  // ---- (E) l2norm(q), l2norm(k) per token, wave t handles token t (mod NW) ----
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    if (t < T && (t % NW) == wave) {
      const float q0 = q_s[t][2 * lane], q1 = q_s[t][2 * lane + 1], k0 = k_s[t][2 * lane], k1 = k_s[t][2 * lane + 1];
      const float sq = wave_sum(q0 * q0 + q1 * q1), sk = wave_sum(k0 * k0 + k1 * k1);
      const float rq = rsqrtf(sq + 1e-6f), rk = rsqrtf(sk + 1e-6f);
      q_s[t][2 * lane] = (q0 * rq) * scale; q_s[t][2 * lane + 1] = (q1 * rq) * scale;   // Triton: (q * rsqrt) * scale
      k_s[t][2 * lane] = k0 * rk;           k_s[t][2 * lane + 1] = k1 * rk;
    }
  }
  lds_barrier();

  // ---- (F) recurrence over the T tokens; state after each token -> its own block ----
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    if (t >= T) break;
    const float a_val = bf2f(ba[(size_t)(bos + t) * s_ba + HV + i_hv]);
    const float b_val = bf2f(ba[(size_t)(bos + t) * s_ba + i_hv]);
    const float xg = a_val + dtb;
    const float sp = (xg <= 20.f) ? lg<EXPMODE>(1.f + ex<EXPMODE>(xg)) : xg;
    const float g = -ex<EXPMODE>(alog) * sp;
    const float beta = sigmoid_<EXPMODE>(b_val);
    const float eg = ex<EXPMODE>(g);
    const float vin = v_s[t][row];
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      const float4 k4 = *reinterpret_cast<const float4*>(&k_s[t][j * 16 + kseg * 4]);
      h[j].x *= eg; h[j].y *= eg; h[j].z *= eg; h[j].w *= eg;
      d += h[j].x * k4.x + h[j].y * k4.y + h[j].z * k4.z + h[j].w * k4.w;
    }
    d = quad_sum(d);
    const float vb = (vin - d) * beta;
    float o = 0.f;
    asm volatile("" ::: "memory");
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
      const float4 k4 = *reinterpret_cast<const float4*>(&k_s[t][j * 16 + kseg * 4]);
      const float4 q4 = *reinterpret_cast<const float4*>(&q_s[t][j * 16 + kseg * 4]);
      h[j].x += vb * k4.x; h[j].y += vb * k4.y; h[j].z += vb * k4.z; h[j].w += vb * k4.w;
      o += h[j].x * q4.x + h[j].y * q4.y + h[j].z * q4.z + h[j].w * q4.w;
    }
    o = quad_sum(o);
    const float ob = bfround(o);   // stock stores core_attn_out as bf16 before the norm
    if constexpr (NV == 1) {
      if (kseg == 0) o_s[t][row] = ob;
    } else {
      if (kseg == 0) scratch[((size_t)(bos + t) * HV + i_hv) * V + grow] = ob;
    }
    const int slot_t = idx[(size_t)b * s_idx_seq + (size_t)t * s_idx_tok];
    if (slot_t > 0) {
      float* hp = hstate + (size_t)slot_t * s_h_slot + ((size_t)i_hv * V + grow) * K + kseg * 4;
#pragma unroll
      for (int j = 0; j < NCH; ++j) *reinterpret_cast<float4*>(hp + j * 16) = h[j];
    }
  }

  // ---- (G) publish the o rows / count (NV > 1) ----
  if constexpr (NV == 1) {
    if (tid == 0) { flag_s[0] = conv_old; }
  } else {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // scratch (and state) stores visible before the count
    if (tid == 0) { flag_s[0] = conv_old; flag_s[1] = atomicAdd(cnt + b * HV + i_hv, 1); }
  }
  lds_barrier();
  const bool last_conv_reader = (flag_s[0] == GROUP * NV - 1);

  // ---- (H) deferred q/k conv-state shift (last reader of the (b, h) group) ----
  if (last_conv_reader) {
    if (tid == 0) cnt[B * HV + b * H + i_h] = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int c = tid + i * NT;
      if (c < QKC) {
        const int dim = (c < K) ? (Q_OFF + i_h * K + c) : (K_OFF + i_h * K + (c - K));
        uint16_t* cs = cstate + cs_base + (size_t)dim * s_cs_dim;
        cs[0] = u1[i]; cs[s_cs_tok] = u2[i];
#pragma unroll
        for (int t = 0; t < TMAX; ++t) if (t < T) cs[(size_t)(2 + t) * s_cs_tok] = ux[i][t];
      }
    }
  }

  // ---- (I) gated RMSNorm per token row: out = rmsnorm(o) * act(z) ----
  const float* sc = nullptr;
  if constexpr (NV > 1) {
    if (flag_s[1] != NV - 1) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // invalidate L1: see the other blocks' scratch
    if (tid == 0) cnt[b * HV + i_hv] = 0;
    sc = scratch + ((size_t)bos * HV + i_hv) * V;
  }
#pragma unroll
  for (int t = 0; t < TMAX; ++t) {
    if (t < T && (t % NW) == wave) {
      float ov[2];
      if constexpr (NV == 1) {
        ov[0] = o_s[t][2 * lane]; ov[1] = o_s[t][2 * lane + 1];
      } else {
        const float* st = sc + (size_t)t * HV * V;
        ov[0] = __hip_atomic_load(st + 2 * lane, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        ov[1] = __hip_atomic_load(st + 2 * lane + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      float ss = ov[0] * ov[0] + ov[1] * ov[1];
      ss = wave_sum(ss);
      const float rstd = rsqrtf(ss / (float)V + eps);
      const float zz[2] = {bf2f((uint16_t)(zpk[t] & 0xffffu)), bf2f((uint16_t)(zpk[t] >> 16))};
      const float ww[2] = {nw2.x, nw2.y};
      uint32_t pk = 0;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float act = GATE_SIGMOID ? sigmoid_<EXPMODE>(zz[e]) : silu_<EXPMODE>(zz[e]);
        const float y = ((ov[e] * rstd) * ww[e]) * act;
        pk |= (uint32_t)f2bf(y) << (16 * e);
      }
      *reinterpret_cast<uint32_t*>(out + ((size_t)(bos + t) * HV + i_hv) * V + 2 * lane) = pk;
    }
  }
}

template <int ROWS, int TMAX, bool SIG, int EXPMODE>
void launch_spec(dim3 grid, hipStream_t st, const uint16_t* qkvz, int64_t s_qkvz, const uint16_t* ba, int64_t s_ba,
                 const uint16_t* cw, const uint16_t* cbias,
                 uint16_t* cstate, int64_t s_cs_slot, int64_t s_cs_dim, int64_t s_cs_tok,
                 float* hstate, int64_t s_h_slot, const int32_t* idx, int64_t s_idx_seq, int64_t s_idx_tok,
                 const int32_t* cu, const int32_t* nacc, const float* A_log, const float* dt_bias,
                 const float* nw, float eps, float scale, uint16_t* out, float* scratch, int32_t* cnt, int M) {
  hipLaunchKernelGGL((gdn_fused_spec_kernel<ROWS, TMAX, SIG, EXPMODE>), grid, dim3(ROWS * LPR), 0, st,
                     qkvz, s_qkvz, ba, s_ba, cw, cbias, cstate, s_cs_slot, s_cs_dim, s_cs_tok,
                     hstate, s_h_slot, idx, s_idx_seq, s_idx_tok, cu, nacc, A_log, dt_bias, nw, eps, scale,
                     out, scratch, cnt, M);
}

template <int ROWS, bool SIG, int EXPMODE>
void launch(dim3 grid, hipStream_t st, const uint16_t* qkvz, int64_t s_qkvz, const uint16_t* ba, int64_t s_ba,
            const uint16_t* cw, const uint16_t* cbias,
            uint16_t* cstate, int64_t s_cs_slot, int64_t s_cs_dim, int64_t s_cs_tok,
            float* hstate, int64_t s_h_slot, const int32_t* idx, const float* A_log, const float* dt_bias,
            const float* nw, float eps, float scale, uint16_t* out, float* scratch, int32_t* cnt, int M, int dbg) {
  hipLaunchKernelGGL((gdn_fused_kernel<ROWS, SIG, EXPMODE>), grid, dim3(ROWS * LPR), 0, st,
                     qkvz, s_qkvz, ba, s_ba, cw, cbias, cstate, s_cs_slot, s_cs_dim, s_cs_tok,
                     hstate, s_h_slot, idx, A_log, dt_bias, nw, eps, scale, out, scratch, cnt, M, dbg);
}

}  // namespace

// qkvz [M, >=4096] bf16 (row stride free); ba [M, 24] bf16 (b = [:, :12], a = [:, 12:]);
// conv_w [2560, 4] bf16; conv_bias optional [2560] bf16; conv_state bf16, any strides
// (slot, dim, tok) with 3 taps; ssm_state [slots, 12, 128, 128] fp32; idx [M] int32
// (<= 0 => padded token: output zeroed, states untouched); A_log/dt_bias [12] fp32;
// norm_w [128] fp32; out [M, 12, 128] bf16 contiguous; scratch [M, 12, 128] fp32;
// cnt int32 [M*12 + M*4], zero-initialised once (self-resetting).
void gdn_fused_decode(torch::Tensor qkvz, torch::Tensor ba, torch::Tensor conv_w,
                      c10::optional<torch::Tensor> conv_bias, torch::Tensor conv_state, torch::Tensor ssm_state,
                      torch::Tensor idx, torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor norm_w,
                      double eps, double scale, torch::Tensor out, torch::Tensor scratch, torch::Tensor cnt,
                      int64_t rows, int64_t exp_mode, bool gate_sigmoid, int64_t dbg) {
  const int M = (int)qkvz.size(0);
  TORCH_CHECK(qkvz.dtype() == torch::kBFloat16 && qkvz.stride(1) == 1 && qkvz.size(1) >= Z_OFF + HV * V);
  TORCH_CHECK(ba.dtype() == torch::kBFloat16 && ba.stride(1) == 1 && ba.size(0) == M && ba.size(1) == 2 * HV);
  TORCH_CHECK(conv_w.dtype() == torch::kBFloat16 && conv_w.dim() == 2 && conv_w.size(0) == V_OFF + HV * V &&
              conv_w.size(1) == 4 && conv_w.is_contiguous());
  TORCH_CHECK(conv_state.dtype() == torch::kBFloat16 && conv_state.dim() == 3 && conv_state.size(1) == V_OFF + HV * V && conv_state.size(2) >= 3);
  TORCH_CHECK(ssm_state.dtype() == torch::kFloat32 && ssm_state.dim() == 4 && ssm_state.size(1) == HV &&
              ssm_state.size(2) == V && ssm_state.size(3) == K && ssm_state.stride(3) == 1 &&
              ssm_state.stride(2) == K && ssm_state.stride(1) == V * K);
  TORCH_CHECK(idx.dtype() == torch::kInt32 && idx.is_contiguous() && idx.numel() >= M);
  TORCH_CHECK(A_log.dtype() == torch::kFloat32 && A_log.is_contiguous() && A_log.numel() == HV);
  TORCH_CHECK(dt_bias.dtype() == torch::kFloat32 && dt_bias.is_contiguous() && dt_bias.numel() == HV);
  TORCH_CHECK(norm_w.dtype() == torch::kFloat32 && norm_w.is_contiguous() && norm_w.numel() == V);
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)M * HV * V);
  TORCH_CHECK(scratch.dtype() == torch::kFloat32 && scratch.is_contiguous() && scratch.numel() >= (int64_t)M * HV * V);
  TORCH_CHECK(cnt.dtype() == torch::kInt32 && cnt.is_contiguous() && cnt.numel() >= (int64_t)M * (HV + H));
  const uint16_t* cb = nullptr;
  if (conv_bias.has_value()) {
    TORCH_CHECK(conv_bias->dtype() == torch::kBFloat16 && conv_bias->is_contiguous());
    cb = (const uint16_t*)conv_bias->data_ptr();
  }
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int NV = V / (int)rows;
  dim3 grid(NV, HV, M);
#define ARGS                                                                                            \
  grid, st, (const uint16_t*)qkvz.data_ptr(), qkvz.stride(0), (const uint16_t*)ba.data_ptr(), ba.stride(0), \
      (const uint16_t*)conv_w.data_ptr(), cb, (uint16_t*)conv_state.data_ptr(),                         \
      conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), (float*)ssm_state.data_ptr(),   \
      ssm_state.stride(0), (const int32_t*)idx.data_ptr(), (const float*)A_log.data_ptr(),               \
      (const float*)dt_bias.data_ptr(), (const float*)norm_w.data_ptr(), (float)eps, (float)scale,     \
      (uint16_t*)out.data_ptr(), (float*)scratch.data_ptr(), (int32_t*)cnt.data_ptr(), M, (int)dbg
#define DISPATCH_E(R, S)                                              \
  if (exp_mode == 0) { launch<R, S, 0>(ARGS); } else { launch<R, S, 1>(ARGS); }
#define DISPATCH_S(R)                                                 \
  if (gate_sigmoid) { DISPATCH_E(R, true) } else { DISPATCH_E(R, false) }
  if (rows == 16) { DISPATCH_S(16) }
  else if (rows == 32) { DISPATCH_S(32) }
  else if (rows == 64) { DISPATCH_S(64) }
  else if (rows == 128) { DISPATCH_S(128) }
  else { TORCH_CHECK(false, "rows must be 16/32/64/128"); }
#undef ARGS
#undef DISPATCH_E
#undef DISPATCH_S
}

// Spec variant.  qkvz [M, >=4096] bf16 (M = padded token count, sequences
// contiguous at cu[b]..cu[b+1]); ba [M, 24]; idx [B, n+1] int32 (any strides;
// <= 0 => padded/null); cu [>= B+1] int32; nacc [>= B] int32 (>= 1);
// conv_state (slots, dim, >= n+3) bf16; out [M, 12, 128] bf16; scratch fp32
// [>= M, 12, 128]; cnt int32 [>= B*16] zeroed once (self-resetting).
// rows: 64 | 128, tmax: 4 | 8 (>= n+1), exp_mode 1 only.
void gdn_fused_spec_decode(torch::Tensor qkvz, torch::Tensor ba, torch::Tensor conv_w,
                           c10::optional<torch::Tensor> conv_bias, torch::Tensor conv_state, torch::Tensor ssm_state,
                           torch::Tensor idx, torch::Tensor cu, torch::Tensor nacc,
                           torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor norm_w,
                           double eps, double scale, torch::Tensor out, torch::Tensor scratch, torch::Tensor cnt,
                           int64_t rows, int64_t tmax, bool gate_sigmoid) {
  const int M = (int)qkvz.size(0);
  TORCH_CHECK(qkvz.dtype() == torch::kBFloat16 && qkvz.stride(1) == 1 && qkvz.size(1) >= Z_OFF + HV * V);
  TORCH_CHECK(ba.dtype() == torch::kBFloat16 && ba.stride(1) == 1 && ba.size(0) == M && ba.size(1) == 2 * HV);
  TORCH_CHECK(conv_w.dtype() == torch::kBFloat16 && conv_w.dim() == 2 && conv_w.size(0) == V_OFF + HV * V &&
              conv_w.size(1) == 4 && conv_w.is_contiguous());
  TORCH_CHECK(idx.dtype() == torch::kInt32 && idx.dim() == 2 && idx.stride(1) >= 1);
  const int B = (int)idx.size(0), NS1 = (int)idx.size(1);
  TORCH_CHECK(NS1 <= tmax && (tmax == 4 || tmax == 8), "tmax must be 4 or 8 and >= idx.size(1)");
  TORCH_CHECK(conv_state.dtype() == torch::kBFloat16 && conv_state.dim() == 3 && conv_state.size(1) == V_OFF + HV * V &&
              conv_state.size(2) >= NS1 + 2, "conv state needs n+3 taps for the spec path");
  TORCH_CHECK(ssm_state.dtype() == torch::kFloat32 && ssm_state.dim() == 4 && ssm_state.size(1) == HV &&
              ssm_state.size(2) == V && ssm_state.size(3) == K && ssm_state.stride(3) == 1 &&
              ssm_state.stride(2) == K && ssm_state.stride(1) == V * K);
  TORCH_CHECK(cu.dtype() == torch::kInt32 && cu.is_contiguous() && cu.numel() >= B + 1);
  TORCH_CHECK(nacc.dtype() == torch::kInt32 && nacc.is_contiguous() && nacc.numel() >= B);
  TORCH_CHECK(A_log.dtype() == torch::kFloat32 && A_log.is_contiguous() && A_log.numel() == HV);
  TORCH_CHECK(dt_bias.dtype() == torch::kFloat32 && dt_bias.is_contiguous() && dt_bias.numel() == HV);
  TORCH_CHECK(norm_w.dtype() == torch::kFloat32 && norm_w.is_contiguous() && norm_w.numel() == V);
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == (int64_t)M * HV * V);
  TORCH_CHECK(scratch.dtype() == torch::kFloat32 && scratch.is_contiguous() && scratch.numel() >= (int64_t)M * HV * V);
  TORCH_CHECK(cnt.dtype() == torch::kInt32 && cnt.is_contiguous() && cnt.numel() >= (int64_t)B * (HV + H));
  const uint16_t* cb = nullptr;
  if (conv_bias.has_value()) {
    TORCH_CHECK(conv_bias->dtype() == torch::kBFloat16 && conv_bias->is_contiguous());
    cb = (const uint16_t*)conv_bias->data_ptr();
  }
  if (B == 0) return;
  auto st = c10::hip::getCurrentHIPStream().stream();
  const int NV = V / (int)rows;
  dim3 grid(NV, HV, B);
#define SARGS                                                                                            \
  grid, st, (const uint16_t*)qkvz.data_ptr(), qkvz.stride(0), (const uint16_t*)ba.data_ptr(), ba.stride(0), \
      (const uint16_t*)conv_w.data_ptr(), cb, (uint16_t*)conv_state.data_ptr(),                          \
      conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), (float*)ssm_state.data_ptr(),    \
      ssm_state.stride(0), (const int32_t*)idx.data_ptr(), idx.stride(0), idx.stride(1),                 \
      (const int32_t*)cu.data_ptr(), (const int32_t*)nacc.data_ptr(), (const float*)A_log.data_ptr(),    \
      (const float*)dt_bias.data_ptr(), (const float*)norm_w.data_ptr(), (float)eps, (float)scale,      \
      (uint16_t*)out.data_ptr(), (float*)scratch.data_ptr(), (int32_t*)cnt.data_ptr(), M
#define SDISPATCH_S(R, TM)                                                                     \
  if (gate_sigmoid) { launch_spec<R, TM, true, 1>(SARGS); } else { launch_spec<R, TM, false, 1>(SARGS); }
#define SDISPATCH_T(R)                                                                         \
  if (tmax == 4) { SDISPATCH_S(R, 4) } else { SDISPATCH_S(R, 8) }
  if (rows == 64) { SDISPATCH_T(64) }
  else if (rows == 128) { SDISPATCH_T(128) }
  else { TORCH_CHECK(false, "rows must be 64/128 for the spec kernel"); }
#undef SARGS
#undef SDISPATCH_S
#undef SDISPATCH_T
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_fused_decode", &gdn_fused_decode);
  m.def("gdn_fused_spec_decode", &gdn_fused_spec_decode);
}
