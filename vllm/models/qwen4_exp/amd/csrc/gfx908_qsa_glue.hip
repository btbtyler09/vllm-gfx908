// gfx908 QSA decode glue for Qwen4Exp (Qwen3.8-Flash-Next), one launch each:
//
//  qsa_glue_pre   (grid = rows, 256 threads, one row per workgroup)
//    main q  : GemmaRMSNorm (ir.ops.rms_norm form) -> bf16 -> MRoPE(neox, interleaved) -> query[rows, 6, 256]
//    main k  : same -> paged KV cache (reshape_and_cache_flash semantics)      v : raw copy -> KV cache
//    idx q   : grouped Gemma RMSNorm (Triton form) -> bf16 -> triton_mrope (bf16 per-op) -> iq[rows, 4, 128]
//    idx k   : raw key -> compressor ring (+ int64 MRoPE positions), group pool over ring + this row
//              -> bf16 -> grouped norm -> bf16 -> mrope(first position) -> compressed-key cache
//    Replaces 18 launches (3 inductor norm/rope kernels, reshape_and_cache_flash, grouped norm x2,
//    cos/sin gather + 2 contiguous copies + mrope x2, compress, store x3).
//    Every cross-row hazard is the compressor ring: row B's ring store may alias a slot row A of the
//    same request still needs.  Rows of different requests use different ring blocks, and within a
//    request the ring reads are at positions < chunk_start while the stores are at positions
//    chunk_start .. chunk_start + T - 1, so the per-row grid (TQ = 1) is hazard-free whenever
//    ring_size >= T + 3 (uniform decode T = 1; every spec-verify batch, whose ring is sized
//    4 * ceil((4 + n) / 4) >= n + 4 for T = n + 1 rows).  Narrower rings with T > 1 rows use the
//    per-request grid (TQ = T, template below): all rows of a request in one workgroup, ring stores
//    after a barrier.  No cross-workgroup dependency inside the kernel either way.
//
//  qsa_topk_expand (grid = rows, 512 threads)
//    deterministic top-512 of logits[row, :visible] (32-bit radix select, 11/11/10-bit digits) with the
//    stable rule of qsa_stable_topk_ ({c: l > t} U first (k - |>t|) columns with l == t, ascending), then
//    the token expansion of _expand_qsa_indices_tile.  Replaces topKPerRowDecode + stable repair + expand.
//    Only workgroup-local LDS atomics (histogram counts, order-independent).
//
// Numerics: fp32 everywhere, RNE bf16 rounding at exactly the points the stock sequence rounds.
// Reduction order inside a warp differs from Triton's tree, so norms are within 1 fp32 ulp of the
// stock kernels' rrms (bf16 outputs identical except on rounding-boundary elements; measured in the test).
#pragma clang fp contract(off)

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

__device__ __forceinline__ float warp_sum64(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

// interleaved MRoPE axis of rotary channel i (0..rd/2): T unless (i%3==1 && i<3*sec_h) -> H,
// (i%3==2 && i<3*sec_w) -> W   (vllm apply_interleaved_rope / _triton_mrope_forward)
__device__ __forceinline__ int mrope_axis(int i, int sec_h3, int sec_w3) {
  const int m = i % 3;
  if (m == 1 && i < sec_h3) return 1;
  if (m == 2 && i < sec_w3) return 2;
  return 0;
}

struct GlueArgs {
  const uint16_t* qkv; long s_qkv;    // [rows, 3584] bf16
  const uint16_t* iqk; long s_iqk;    // [rows, 640] bf16
  const long* pos; long s_pos;        // positions[axis * s_pos + row]; s_pos == 0 for 1-D positions
  const uint16_t* cs; long max_pos;   // cos_sin cache [max_pos, 64] bf16: cos[32] | sin[32]
  const uint16_t* qw; const uint16_t* kw; const uint16_t* iqw; const uint16_t* ikw;
  float eps_main, eps_idx;
  uint16_t* query;                    // [rows, 6, 256] bf16 out
  uint16_t* iq;                       // [rows, 4, 128] bf16 out
  uint16_t* kc; uint16_t* vc; long kv_bs, kv_ps, kv_hs; const long* main_slots; long main_block;
  uint16_t* ring; long ring_bs, ring_ts; const long* raw_slots; long ring_size, ring_blocks;
  long* rope; long rope_bs, rope_ts, rope_ds;                  // nullable
  const int* raw_table; long s_raw_table;
  const int* tok2req; const int* qsl; const long* logpos; const long* cmp_slots;
  uint16_t* cmp; long cmp_bs, cmp_ts; long cmp_page, cmp_blocks;
  int rows; int num_reqs; int sec_h3; int sec_w3; int rope_mode; int mode;
};

// cos/sin of rotary channel (lane & 31) for a row's three positions (same for every head of the row).
__device__ __forceinline__ void rope_cs(const uint16_t* __restrict__ cs, const long* p3, long max_pos, int lane,
                                        int sec_h3, int sec_w3, float& c, float& s) {
  const int i = lane & 31;
  long p = p3[mrope_axis(i, sec_h3, sec_w3)];
  p = p < 0 ? 0 : (p >= max_pos ? max_pos - 1 : p);
  c = bf2f(cs[p * 64 + i]);
  s = bf2f(cs[p * 64 + 32 + i]);
}

template <int D>
__device__ __forceinline__ void load_head(const uint16_t* __restrict__ src, int lane, float* x) {
#pragma unroll
  for (int j = 0; j < D / 64; ++j) x[j] = bf2f(src[lane + 64 * j]);
}

// One head (values already in registers, lane holds dims lane + 64 j): RMS-normalize, rotate the
// first 64 dims (neox pairs (i, i+32) live on lanes l and l^32 of element 0).  y[E] in fp32; the
// caller rounds on store.
//   TRITON_FORM  y = x*rrms; y += y*w (fma)          else  y = (x*rrms) * (1 + w)   (ir.ops.rms_norm)
//   ROPE_BF16    products/sums round to bf16 like triton_mrope on bf16 tiles (indexer path)
//   rope_mode    bit 2: norm output stays fp32 into the rope (inductor fuses norm-apply + RoPE into
//                one kernel and, with emulate_precision_casts=False, drops the bf16 round trip);
//                bits 0-1: fp-contraction form of x1*c -/+ x2*s (0 none, 1 fma(x1,c,.), 2 fma(x2,s,.)).
template <int D, bool TRITON_FORM, bool ROPE_BF16>
__device__ __forceinline__ void norm_rope_vals(const float* x, const uint16_t* __restrict__ w, float eps,
                                               float c, float s, int lane, int rope_mode, float* y) {
  constexpr int E = D / 64;
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < E; ++j) ss = ss + x[j] * x[j];
  ss = warp_sum64(ss);
  const float rrms = rsqrtf(ss / (float)D + eps);
  const bool keep_fp32 = (rope_mode & 4) != 0;
  const int contract = rope_mode & 3;
#pragma unroll
  for (int j = 0; j < E; ++j) {
    const float wj = bf2f(w[lane + 64 * j]);
    float v;
    if (TRITON_FORM) {
      const float t = x[j] * rrms;
      v = fmaf(t, wj, t);
    } else {
      v = (x[j] * rrms) * (1.0f + wj);
    }
    y[j] = keep_fp32 ? v : rbf(v);
  }
  const float own = y[0];
  const float other = __shfl_xor(own, 32, 64);
  float o;
  if (ROPE_BF16) {
    const float pa = rbf(own * c), pb = rbf(other * s);
    o = (lane < 32) ? (pa - pb) : (pa + pb);
  } else if (contract == 1) {
    o = (lane < 32) ? fmaf(own, c, -(other * s)) : fmaf(own, c, other * s);
  } else if (contract == 2) {
    o = (lane < 32) ? fmaf(-other, s, own * c) : fmaf(other, s, own * c);
  } else {
    o = (lane < 32) ? (own * c - other * s) : (own * c + other * s);
  }
  y[0] = o;
}

// main q head h of the row -> query[row, h, :]
__device__ __forceinline__ void main_q_head(const GlueArgs& a, const float* x, float c, float s, int lane,
                                            uint16_t* dst) {
  float y[4];
  norm_rope_vals<256, false, false>(x, a.qw, a.eps_main, c, s, lane, a.rope_mode, y);
#pragma unroll
  for (int j = 0; j < 4; ++j) dst[lane + 64 * j] = f2bf(y[j]);
}

// indexer q head -> iq[row, h, :]
__device__ __forceinline__ void idx_q_head(const GlueArgs& a, const float* x, float c, float s, int lane,
                                           uint16_t* dst) {
  float y[2];
  norm_rope_vals<128, true, true>(x, a.iqw, a.eps_idx, c, s, lane, 0, y);
  dst[lane] = f2bf(y[0]);
  dst[lane + 64] = f2bf(y[1]);
}

// pool the closing group (ring + this row), norm, rope at the group's first position, store
__device__ __forceinline__ void compress_chain(const GlueArgs& a, int row, int lane, const uint16_t* irow) {
  const long cslot = a.cmp_slots[row];
  if (cslot < 0) return;
  const int req = a.tok2req[row];
  const bool vreq = req >= 0 && req < a.num_reqs;
  const int sreq = req < 0 ? 0 : (req >= a.num_reqs ? a.num_reqs - 1 : req);
  const long endp = a.logpos[row];
  const int qrs = vreq ? a.qsl[sreq] : 0;
  const int qre = vreq ? a.qsl[sreq + 1] : 0;
  const long chunk_start = endp - (row - qrs);
  const int rblock = vreq ? a.raw_table[(long)sreq * a.s_raw_table] : -1;
  const bool vblock = rblock >= 0 && rblock < a.ring_blocks;
  const bool vrow = (row < a.rows) && vreq && (row >= qrs) && (row < qre) && (endp >= 3);
  // issue every ring / raw / rope-position load before touching any of them
  float r[4][2], sv[4][2];
  long rp[3] = {0, 0, 0}, sp[3] = {0, 0, 0};
  const long first_position = endp - 3;
  const bool first_from_raw = first_position >= chunk_start;
  const long raw_first_row = qrs + first_position - chunk_start;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const long position = endp - (3 - g);
    const bool use_raw = position >= chunk_start;
    const long raw_row = qrs + position - chunk_start;
    r[g][0] = r[g][1] = sv[g][0] = sv[g][1] = 0.f;
    if (vrow && use_raw && raw_row >= qrs && raw_row < qre && raw_row < a.rows) {
      const uint16_t* rr = a.iqk + raw_row * a.s_iqk + 512;
      r[g][0] = bf2f(rr[lane]);
      r[g][1] = bf2f(rr[lane + 64]);
    }
    if (vrow && !use_raw && vblock) {
      const uint16_t* sr = a.ring + (long)rblock * a.ring_bs + (position % a.ring_size) * a.ring_ts;
      sv[g][0] = bf2f(sr[lane]);
      sv[g][1] = bf2f(sr[lane + 64]);
    }
  }
  if (a.rope != nullptr) {
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
      if (vrow && first_from_raw && raw_first_row >= qrs && raw_first_row < qre && raw_first_row < a.rows)
        rp[ax] = a.pos[ax * a.s_pos + raw_first_row];
      if (vrow && !first_from_raw && vblock)
        sp[ax] = a.rope[(long)rblock * a.rope_bs + (first_position % a.ring_size) * a.rope_ts + ax * a.rope_ds];
    }
  }
  float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const bool use_raw = (endp - (3 - g)) >= chunk_start;
    acc0 += use_raw ? r[g][0] : sv[g][0];
    acc1 += use_raw ? r[g][1] : sv[g][1];
  }
  const float pooled[2] = {rbf(acc0 / 4.0f), rbf(acc1 / 4.0f)};
  long fp3[3];
#pragma unroll
  for (int ax = 0; ax < 3; ++ax)
    fp3[ax] = (a.rope != nullptr) ? (first_from_raw ? rp[ax] : sp[ax]) : (vrow ? first_position : 0);
  float c, s;
  rope_cs(a.cs, fp3, a.max_pos, lane, a.sec_h3, a.sec_w3, c, s);
  float y[2];
  norm_rope_vals<128, true, true>(pooled, a.ikw, a.eps_idx, c, s, lane, 0, y);
  if (cslot < a.cmp_blocks * a.cmp_page) {
    const long blk = cslot / a.cmp_page, tok = cslot % a.cmp_page;
    uint16_t* dst = a.cmp + blk * a.cmp_bs + tok * a.cmp_ts;
    dst[lane] = f2bf(y[0]);
    dst[lane + 64] = f2bf(y[1]);
  }
  (void)irow;
}

// raw key + int64 MRoPE positions of one row into the compressor ring (one warp)
__device__ __forceinline__ void ring_store(const GlueArgs& a, int row, int lane, const long* p3) {
  const long rslot = a.raw_slots[row];
  if (rslot >= 0 && rslot < a.ring_blocks * a.ring_size) {
    const uint16_t* irow = a.iqk + (long)row * a.s_iqk;
    const long blk = rslot / a.ring_size, tok = rslot % a.ring_size;
    uint16_t* dst = a.ring + blk * a.ring_bs + tok * a.ring_ts;
    dst[lane] = irow[512 + lane];
    dst[lane + 64] = irow[512 + lane + 64];
    if (a.rope != nullptr && lane < 3) a.rope[blk * a.rope_bs + tok * a.rope_ts + lane * a.rope_ds] = p3[lane];
  }
}

// TQ = rows per workgroup (4 warps per row, warp = 4 * local_row + job).
//   TQ == 1: grid = rows, one workgroup per row (uniform decode, or any batch whose ring is wide
//            enough that no row's ring store aliases a slot another row of the same request pools
//            from: ring_size >= max_query_len + 3 -- the spec-decode ring, 4 * ceil((4 + n) / 4),
//            guarantees that for every verify batch of n + 1 rows).
//   TQ > 1:  grid = num_reqs + ceil(rows / TQ); workgroup g < num_reqs owns request g's rows
//            [qsl[g], min(qsl[g+1], qsl[g] + TQ)) (host gate: max_query_len <= TQ), the rest cover
//            the unmapped padded rows [qsl[num_reqs], rows) in slices of TQ (main projection only:
//            their slots are all -1).  The ring / RoPE-position stores of every row run after a
//            workgroup barrier, i.e. after every row of the request has pooled what it needs from
//            the ring, which is the stock order (compress, then store).  The only cross-row hazard
//            of this kernel is therefore closed inside one workgroup; workgroups never wait on
//            each other.
template <int TQ>
__global__ void __launch_bounds__(256 * TQ) qsa_glue_pre_kernel(const GlueArgs a) {
  const int tid = threadIdx.x, lane = tid & 63, warp = tid >> 6;
  const int lrow = warp >> 2, job = warp & 3;
  int row;
  bool active;
  if constexpr (TQ == 1) {
    row = blockIdx.x;
    active = true;
  } else {
    const int g = blockIdx.x;
    int base, count;
    if (g < a.num_reqs) {
      const int s = a.qsl[g], e = a.qsl[g + 1];
      base = s;
      count = e - s < TQ ? e - s : TQ;
    } else {
      base = a.qsl[a.num_reqs] + (g - a.num_reqs) * TQ;
      count = TQ;
    }
    row = base + lrow;
    active = (lrow < count) && (row >= 0) && (row < a.rows);
  }
  const bool do_main = (a.mode & 1) != 0, do_idx = (a.mode & 2) != 0;
  long p3[3] = {0, 0, 0};
  if (active) {
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) p3[ax] = a.pos[ax * a.s_pos + row];
    const uint16_t* qrow = a.qkv + (long)row * a.s_qkv;
    const uint16_t* irow = a.iqk + (long)row * a.s_iqk;
    uint16_t* qdst = a.query + (long)row * (6 * 256);
    uint16_t* idst = a.iq + (long)row * (4 * 128);
    // Job split (11 norm jobs + the compress chain over 4 warps; the chain has ~5 dependent
    // round trips so it shares a warp with only two short jobs):
    //   job 0: compress chain, main q h0,  idx q h0
    //   job 1: main q h1, h2,               idx q h1
    //   job 2: main q h3, h4,               idx q h2
    //   job 3: main q h5, main k (+KV write), v copy, idx q h3, raw ring + rope-position stores
    // Every warp issues all of its independent loads (x, weights, cos/sin) before the first reduce.
    float c = 0.f, s = 0.f;
    rope_cs(a.cs, p3, a.max_pos, lane, a.sec_h3, a.sec_w3, c, s);
    if (job == 0) {
      float xq[4], xi[2];
      if (do_main) load_head<256>(qrow, lane, xq);
      if (do_idx) load_head<128>(irow, lane, xi);
      if (do_idx) compress_chain(a, row, lane, irow);
      if (do_main) main_q_head(a, xq, c, s, lane, qdst);
      if (do_idx) idx_q_head(a, xi, c, s, lane, idst);
    } else if (job < 3) {
      const int h0 = (job == 1) ? 1 : 3;
      float xa[4], xb[4], xi[2];
      if (do_main) {
        load_head<256>(qrow + h0 * 512, lane, xa);
        load_head<256>(qrow + (h0 + 1) * 512, lane, xb);
      }
      if (do_idx) load_head<128>(irow + job * 128, lane, xi);
      if (do_main) {
        main_q_head(a, xa, c, s, lane, qdst + h0 * 256);
        main_q_head(a, xb, c, s, lane, qdst + (h0 + 1) * 256);
      }
      if (do_idx) idx_q_head(a, xi, c, s, lane, idst + job * 128);
    } else {
      float xq[4], xk[4], xi[2];
      if (do_main) {
        load_head<256>(qrow + 5 * 512, lane, xq);
        load_head<256>(qrow + 3072, lane, xk);
      }
      if (do_idx) load_head<128>(irow + 3 * 128, lane, xi);
      if (do_main) {
        main_q_head(a, xq, c, s, lane, qdst + 5 * 256);
        float y[4];
        norm_rope_vals<256, false, false>(xk, a.kw, a.eps_main, c, s, lane, a.rope_mode, y);
        const long mslot = a.main_slots[row];
        if (mslot >= 0) {
          const long blk = mslot / a.main_block, off = mslot % a.main_block;
          const long base = blk * a.kv_bs + off * a.kv_ps;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            a.kc[base + lane + 64 * j] = f2bf(y[j]);
            a.vc[base + lane + 64 * j] = qrow[3328 + lane + 64 * j];
          }
        }
      }
      if (do_idx) {
        idx_q_head(a, xi, c, s, lane, idst + 3 * 128);
        if constexpr (TQ == 1) ring_store(a, row, lane, p3);
      }
    }
  }
  if constexpr (TQ > 1) {
    // every ring / RoPE-position read of this request's rows (compress_chain) is complete
    __syncthreads();
    if (active && do_idx && job == 3) ring_store(a, row, lane, p3);
  }
}

// ---------------------------------------------------------------------------------------------
// deterministic top-k + expand
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned tokey(float f) {
  if (f == 0.0f) f = 0.0f;  // -0 == +0 for the stable rule's float comparisons
  const unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int NT>
__device__ __forceinline__ unsigned block_scan_incl(unsigned v, unsigned* wsum, unsigned& total, int lane, int warp) {
  // inclusive scan over the NT threads in thread order; wsum has NT/64 entries; two barriers.
  unsigned incl = v;
#pragma unroll
  for (int o = 1; o < 64; o <<= 1) {
    const unsigned n = __shfl_up(incl, o, 64);
    if (lane >= o) incl += n;
  }
  if (lane == 63) wsum[warp] = incl;
  __syncthreads();
  unsigned pre = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < NT / 64; ++w) {
    const unsigned x = wsum[w];
    if (w < warp) pre += x;
    total += x;
  }
  __syncthreads();
  return pre + incl;
}

template <int TOPK, int COMPRESS, int WIDTH, int NT, int LDS_KEYS>
__global__ void __launch_bounds__(NT) qsa_topk_expand_kernel(
    const float* __restrict__ logits, long s_logits, const int* __restrict__ visible, int* __restrict__ out,
    long s_out, int* __restrict__ blocks_out, long s_blocks, const long* __restrict__ logpos,
    const int* __restrict__ seq_lens, const int* __restrict__ tok2req, int num_reqs, int num_columns, int rows) {
  static_assert(NT * 4 == 2048, "the 11-bit histogram scan assumes 4 bins per thread");
  __shared__ int sblocks[TOPK];
  __shared__ unsigned hist[2048];
  __shared__ unsigned wsum[NT / 64];
  __shared__ unsigned found_bin, found_above;
  __shared__ unsigned skeys[LDS_KEYS];   // order-preserving keys of logits[row, :visible] (one global read)
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 63, warp = tid >> 6;
  int vis = visible[row];
  vis = vis < 0 ? 0 : (vis > num_columns ? num_columns : vis);
  const float* lrow = logits + (long)row * s_logits;

  if (vis <= TOPK) {
    for (int r = tid; r < TOPK; r += NT) sblocks[r] = r < vis ? r : -1;
  } else {
    const bool in_lds = vis <= LDS_KEYS;
    if (in_lds)
      for (int c = tid; c < vis; c += NT) skeys[c] = tokey(lrow[c]);
    unsigned prefix = 0;
    int pbits = 0, krem = TOPK, greater = 0;
    for (int pass = 0; pass < 3; ++pass) {
      const int dbits = (pass == 2) ? 10 : 11;
      const int shift = 32 - pbits - dbits;
      const unsigned mask = (1u << dbits) - 1u;
      for (int b = tid; b < 2048; b += NT) hist[b] = 0u;
      __syncthreads();
      for (int c = tid; c < vis; c += NT) {
        const unsigned key = in_lds ? skeys[c] : tokey(lrow[c]);
        if (pbits == 0 || (key >> (shift + dbits)) == prefix) atomicAdd(&hist[(key >> shift) & mask], 1u);
      }
      __syncthreads();
      const unsigned h0 = hist[4 * tid], h1 = hist[4 * tid + 1], h2 = hist[4 * tid + 2], h3 = hist[4 * tid + 3];
      unsigned total;
      const unsigned incl = block_scan_incl<NT>(h0 + h1 + h2 + h3, wsum, total, lane, warp);
      unsigned above = total - incl;  // count in bins > 4*tid+3
      const unsigned hs[4] = {h0, h1, h2, h3};
#pragma unroll
      for (int j = 3; j >= 0; --j) {
        const unsigned h = hs[j];
        if (above < (unsigned)krem && above + h >= (unsigned)krem) {
          found_bin = (unsigned)(4 * tid + j);
          found_above = above;
        }
        above += h;
      }
      __syncthreads();
      const unsigned bin = found_bin, ab = found_above;
      prefix = (prefix << dbits) | bin;
      pbits += dbits;
      krem -= (int)ab;
      greater += (int)ab;
      __syncthreads();
    }
    const unsigned tkey = prefix;
    const int needed = krem;
    // one ascending sequence of {l > t} U first `needed` ties (the stable rule).  One packed scan
    // of (above, tie) per chunk: ties are taken in ascending order until `needed` is reached, so the
    // number of taken ties before column c is min(run_e + pe, needed) - run_e.
    unsigned run_a = 0, run_e = 0;
    const unsigned needed_u = (unsigned)needed;
    for (int start = 0; start < vis; start += NT) {
      const int c = start + tid;
      const bool in = c < vis;
      const unsigned key = in ? (in_lds ? skeys[c] : tokey(lrow[c])) : 0u;
      const unsigned av = (in && key > tkey) ? 1u : 0u;
      const unsigned ev = (in && key == tkey) ? 1u : 0u;
      unsigned total;
      const unsigned excl = block_scan_incl<NT>((av << 16) | ev, wsum, total, lane, warp) - ((av << 16) | ev);
      const unsigned pa = excl >> 16, pe = excl & 0xffffu;
      const unsigned ties_before = (run_e + pe < needed_u ? run_e + pe : needed_u) - (run_e < needed_u ? run_e : needed_u);
      const unsigned taken_before = run_a + pa + (run_e < needed_u ? run_e : needed_u) + ties_before;
      const bool take = av || (ev && (run_e + pe) < needed_u);
      if (take && taken_before < (unsigned)TOPK) sblocks[taken_before] = c;
      run_a += total >> 16;
      run_e += total & 0xffffu;
    }
    (void)greater;
  }
  __syncthreads();
  if (blocks_out != nullptr)
    for (int r = tid; r < TOPK; r += NT) blocks_out[(long)row * s_blocks + r] = sblocks[r];

  // ---- expand (_expand_qsa_indices_tile, C integer semantics like Triton's divsi/remsi) ----
  const long qp = logpos[row];
  const int req = tok2req[row];
  const bool vreq = req >= 0 && req < num_reqs;
  const int sreq = req < 0 ? 0 : (req >= num_reqs ? num_reqs - 1 : req);
  const long seq_len = vreq ? (long)seq_lens[sreq] : 0;
  long complete = (qp + 1) / COMPRESS;
  const long s4 = seq_len / COMPRESS;
  if (s4 < complete) complete = s4;
  if ((long)TOPK < complete) complete = TOPK;
  const long expanded_count = complete * COMPRESS;
  const long tail_start = ((qp + 1) / COMPRESS) * COMPRESS;
  const long tail_count = (qp + 1) - tail_start;
  int* orow = out + (long)row * s_out;
  for (int c = tid; c < WIDTH; c += NT) {
    const bool is_expanded = (long)c < expanded_count;
    int rank = c / COMPRESS;
    const int off = c % COMPRESS;
    if (rank > TOPK - 1) rank = TOPK - 1;
    const long block = (row < rows && is_expanded) ? (long)sblocks[rank] : -1L;
    const long expanded = block * COMPRESS + off;
    const long tail_off = (long)c - expanded_count;
    const bool is_tail = ((long)c >= expanded_count) && (tail_off < tail_count) && (tail_off < COMPRESS - 1);
    const long token = is_expanded ? expanded : (tail_start + tail_off);
    const bool valid = (row < rows) && (is_expanded || is_tail) && (token >= 0) && (token < seq_len);
    orow[c] = valid ? (int)token : -1;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// torch bindings
// ---------------------------------------------------------------------------------------------
static const uint16_t* bfp(const torch::Tensor& t) { return reinterpret_cast<const uint16_t*>(t.data_ptr()); }
static uint16_t* bfpm(torch::Tensor& t) { return reinterpret_cast<uint16_t*>(t.data_ptr()); }

void qsa_glue_pre(torch::Tensor qkv, c10::optional<torch::Tensor> iqk_opt, torch::Tensor positions,
                  torch::Tensor cos_sin, torch::Tensor qw, torch::Tensor kw, torch::Tensor iqw, torch::Tensor ikw,
                  double eps_main, double eps_idx, torch::Tensor query, c10::optional<torch::Tensor> iq_opt,
                  torch::Tensor key_cache,
                  torch::Tensor value_cache, torch::Tensor main_slots, int64_t main_block, torch::Tensor ring,
                  torch::Tensor raw_slots, c10::optional<torch::Tensor> rope, torch::Tensor raw_table,
                  torch::Tensor tok2req, torch::Tensor qsl, torch::Tensor logpos, torch::Tensor cmp_slots,
                  torch::Tensor cmp, int64_t sec_h, int64_t sec_w, int64_t rope_mode, int64_t num_actual_rows, int64_t mode,
                  int64_t tq) {
  GlueArgs a;
  const int rows_padded = (int)num_actual_rows;  // grid: one workgroup per row
  const bool do_idx = (mode & 2) != 0;
  TORCH_CHECK(qkv.size(1) == 3584 && query.size(1) == 6 && query.size(2) == 256 && cos_sin.size(1) == 64);
  TORCH_CHECK(qkv.stride(1) == 1 && query.is_contiguous() && cos_sin.is_contiguous());
  TORCH_CHECK(ring.stride(3) == 1 && cmp.stride(3) == 1);
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 && key_cache.size(2) == 1);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0) && key_cache.stride(1) == value_cache.stride(1));
  TORCH_CHECK(query.size(0) >= rows_padded && qkv.size(0) >= rows_padded);
  TORCH_CHECK(main_slots.numel() >= rows_padded && raw_slots.numel() >= rows_padded &&
              cmp_slots.numel() >= rows_padded && tok2req.numel() >= rows_padded && logpos.numel() >= rows_padded);
  a.qkv = bfp(qkv); a.s_qkv = qkv.stride(0);
  if (do_idx) {
    TORCH_CHECK(iqk_opt.has_value() && iq_opt.has_value(), "indexer mode needs iqk and iq");
    torch::Tensor iqk = iqk_opt.value(), iq = iq_opt.value();
    TORCH_CHECK(iqk.size(1) == 640 && iqk.stride(1) == 1 && iqk.size(0) >= rows_padded);
    TORCH_CHECK(iq.size(1) == 4 && iq.size(2) == 128 && iq.is_contiguous() && iq.size(0) >= rows_padded);
    a.iqk = bfp(iqk); a.s_iqk = iqk.stride(0); a.iq = bfpm(iq);
  } else {
    a.iqk = nullptr; a.s_iqk = 0; a.iq = nullptr;
  }
  a.pos = positions.data_ptr<int64_t>();
  if (positions.dim() == 2) {
    TORCH_CHECK(positions.size(0) == 3 && positions.stride(1) == 1 && positions.size(1) >= rows_padded);
    a.s_pos = positions.stride(0);
  } else {
    TORCH_CHECK(positions.stride(0) == 1 && positions.size(0) >= rows_padded);
    a.s_pos = 0;
  }
  a.cs = bfp(cos_sin); a.max_pos = cos_sin.size(0);
  a.qw = bfp(qw); a.kw = bfp(kw); a.iqw = bfp(iqw); a.ikw = bfp(ikw);
  a.eps_main = (float)eps_main; a.eps_idx = (float)eps_idx;
  a.query = bfpm(query);
  a.kc = bfpm(key_cache); a.vc = bfpm(value_cache);
  a.kv_bs = key_cache.stride(0); a.kv_ps = key_cache.stride(1); a.kv_hs = key_cache.stride(2);
  a.main_slots = main_slots.data_ptr<int64_t>(); a.main_block = main_block;
  a.ring = bfpm(ring); a.ring_bs = ring.stride(0); a.ring_ts = ring.stride(1);
  a.raw_slots = raw_slots.data_ptr<int64_t>(); a.ring_size = ring.size(1); a.ring_blocks = ring.size(0);
  if (rope.has_value()) {
    torch::Tensor r = rope.value();
    TORCH_CHECK(r.dim() == 4 && r.size(3) == 3 && r.scalar_type() == torch::kInt64);
    a.rope = r.data_ptr<int64_t>(); a.rope_bs = r.stride(0); a.rope_ts = r.stride(1); a.rope_ds = r.stride(3);
  } else {
    a.rope = nullptr; a.rope_bs = a.rope_ts = a.rope_ds = 0;
  }
  a.raw_table = raw_table.data_ptr<int32_t>(); a.s_raw_table = raw_table.stride(0);
  a.tok2req = tok2req.data_ptr<int32_t>(); a.qsl = qsl.data_ptr<int32_t>();
  a.logpos = logpos.data_ptr<int64_t>(); a.cmp_slots = cmp_slots.data_ptr<int64_t>();
  a.cmp = bfpm(cmp); a.cmp_bs = cmp.stride(0); a.cmp_ts = cmp.stride(1);
  a.cmp_page = cmp.size(1); a.cmp_blocks = cmp.size(0);
  a.rows = (int)num_actual_rows; a.num_reqs = (int)(qsl.numel() - 1);
  a.sec_h3 = (int)(3 * sec_h); a.sec_w3 = (int)(3 * sec_w); a.rope_mode = (int)rope_mode; a.mode = (int)mode;
  auto st = c10::hip::getCurrentHIPStream().stream();
  TORCH_CHECK(tq >= 1 && tq <= 4, "qsa_glue_pre: tq (rows per workgroup) must be 1..4");
  if (rows_padded > 0) {
    if (tq == 1) {
      hipLaunchKernelGGL(qsa_glue_pre_kernel<1>, dim3(rows_padded), dim3(256), 0, st, a);
    } else {
      // one workgroup per (padded) request + enough TQ-row slices for the unmapped padded rows
      const unsigned grid = (unsigned)a.num_reqs + (unsigned)((rows_padded + tq - 1) / tq);
      if (tq == 2) {
        hipLaunchKernelGGL(qsa_glue_pre_kernel<2>, dim3(grid), dim3(512), 0, st, a);
      } else if (tq == 3) {
        hipLaunchKernelGGL(qsa_glue_pre_kernel<3>, dim3(grid), dim3(768), 0, st, a);
      } else {
        hipLaunchKernelGGL(qsa_glue_pre_kernel<4>, dim3(grid), dim3(1024), 0, st, a);
      }
    }
  }
}

void qsa_topk_expand(torch::Tensor logits, torch::Tensor visible, torch::Tensor out,
                     c10::optional<torch::Tensor> blocks_out, torch::Tensor logpos, torch::Tensor seq_lens,
                     torch::Tensor tok2req, int64_t rows) {
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.stride(1) == 1);
  TORCH_CHECK(out.scalar_type() == torch::kInt32 && out.stride(1) == 1 && out.size(1) == 2051);
  TORCH_CHECK(logpos.scalar_type() == torch::kInt64 && seq_lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(tok2req.scalar_type() == torch::kInt32 && visible.scalar_type() == torch::kInt32);
  TORCH_CHECK(logits.size(0) >= rows && out.size(0) >= rows && visible.numel() >= rows);
  int* bo = nullptr; long sb = 0;
  if (blocks_out.has_value()) {
    torch::Tensor b = blocks_out.value();
    TORCH_CHECK(b.size(1) == 512 && b.stride(1) == 1 && b.size(0) >= rows);
    bo = b.data_ptr<int32_t>(); sb = b.stride(0);
  }
  auto st = c10::hip::getCurrentHIPStream().stream();
  if (rows > 0) {
    hipLaunchKernelGGL((qsa_topk_expand_kernel<512, 4, 2051, 512, 8192>), dim3((unsigned)rows), dim3(512), 0, st,
                       logits.data_ptr<float>(), (long)logits.stride(0), visible.data_ptr<int32_t>(),
                       out.data_ptr<int32_t>(), (long)out.stride(0), bo, sb, logpos.data_ptr<int64_t>(),
                       seq_lens.data_ptr<int32_t>(), tok2req.data_ptr<int32_t>(), (int)seq_lens.numel(),
                       (int)logits.size(1), (int)rows);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qsa_glue_pre", &qsa_glue_pre);
  m.def("qsa_topk_expand", &qsa_topk_expand);
}
