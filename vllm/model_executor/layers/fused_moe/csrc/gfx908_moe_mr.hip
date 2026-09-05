// Expert-deduplicated ("multi-row") W4A8 MoE GEMVs for batched decode on gfx908.
//
// Same weights / activations / per-group arithmetic as the shipping
// vllm/model_executor/layers/fused_moe/csrc/gfx908_w4a8.hip (D1 slab, D2 rowlane), with one change
// in the work decomposition: pairs are first sorted by expert (expert_sort), and a work item is
// (expert row-chunk, n-tile) instead of (pair, n-tile). A wave loads its weight tile ONCE and loops
// over the expert's rows, so every active expert's slice is streamed from HBM once per layer no
// matter how many tokens it serves. The grid is persistent (fixed block count, work-list loop), so the
// data-dependent item count never changes the graph.
//
//   weights  [E, N, K/8] uint32 nibbles along K (nibble t of word wd = k 8*wd+t), zero point 8
//   scales   [E, N, K/32] bf16
//   x8/xs/xsum : the Q8_1 activation format of the shipping kernels (per-8 de-interleaved int8,
//                fp32 block scale d, fp32 d*sum(q))
//   out      fp32 [P, N] in ORIGINAL pair order (the existing reduce epilogues read it unchanged)
//
// Bit-exactness: lane->group assignment, dot4 order, per-group fp32 formula and the DPP reduction are
// copied verbatim from the shipping kernels, so out[p] is bit-identical to gemv_slab / gemv_rowlane.
#include <hip/hip_runtime.h>
#ifndef MOE_MR_NO_TORCH
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#endif
#include <cstdint>

__device__ __forceinline__ float bf2f(uint32_t v) { return __uint_as_float(v << 16); }

__device__ __forceinline__ int dot4(uint32_t a, int b, int c) {
  return __builtin_amdgcn_sdot4((int)a, b, c, false);
}

template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
  return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true));
}

// Sum over LPR consecutive lanes; result in lane LPR-1 of each group (row_shr prefix sums).
template <int LPR>
__device__ __forceinline__ float reduce_lpr(float v) {
  if constexpr (LPR >= 2) v = dpp_add<0x111>(v);   // row_shr:1
  if constexpr (LPR >= 4) v = dpp_add<0x112>(v);   // row_shr:2
  if constexpr (LPR >= 8) v = dpp_add<0x114>(v);   // row_shr:4
  if constexpr (LPR >= 16) v = dpp_add<0x118>(v);  // row_shr:8
  if constexpr (LPR >= 32) v += __shfl_xor(v, 16, 64);   // (was missing: LPR=32 configs summed half the groups)
  if constexpr (LPR >= 64) v += __shfl_xor(v, 32, 64);
  return v;
}

__device__ __forceinline__ bool pick_expert(
    int e, int E, const uint32_t* w, const uint16_t* s, long stride_we, long stride_se,
    const uint32_t* w_extra, const uint16_t* s_extra, const uint32_t*& wb, const uint16_t*& sb) {
  if (w_extra != nullptr && e == E) { wb = w_extra; sb = s_extra; return true; }
  if (e < 0 || e >= E) return false;
  wb = w + (long)e * stride_we;
  sb = s + (long)e * stride_se;
  return true;
}

// ---------------------------------------------------------------------------------------------
// expert_sort: one block of 1024 threads.
//   bins: 0..E-1 routed experts, E = the extra (shared) expert when has_extra, E+1 = invalid ids.
//   Item order: shared expert first (it has the most rows -> longest work items go out first),
//   then experts 0..E-1, then the invalid bin (its items carry e = -1 and produce zero rows).
//   Rows of one expert are cut into chunks of RC rows; one item = (expert, chunk).
// Outputs:
//   order[P]            pair ids grouped by bin (order inside a bin is not stable -- irrelevant,
//                       every output row depends only on its own (expert, token))
//   otok[P]             row_tok[order[pos]]: the token of the pair at each sorted position, so the
//                       GEMVs need one independent load per row instead of two dependent ones
//   wl_tok[j*RC + i]    the token of row i of item j (>= n_items*RC entries; slots i >= cnt are
//                       left unwritten): lets a kernel fetch an item's meta and its rows' tokens in
//                       ONE round trip (index known from j alone)
//   wl_e/wl_r0/wl_cnt   per item: expert id (-1 = zero rows), first index into order, row count
//   uniq_e[]            one entry per non-empty valid bin (for the stream floor)
//   counters[0] = n_items, counters[1] = n_uniq, counters[2] = n_valid_rows
// The three exclusive scans over the E+2 bins run in ONE wave (each lane owns ceil((E+2)/64)
// consecutive bins in item order, sequential prefix inside the lane, one shfl_up scan across the
// lanes), so the kernel has 4 block barriers instead of the 61 of the Hillis-Steele version
// (10.9 -> ~4 us at E = 512).
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) expert_sort_kernel(
    const int* __restrict__ row_exp, const int* __restrict__ row_tok, int P, int E, int has_extra,
    int RC, int* __restrict__ order, int* __restrict__ otok, int* __restrict__ wl_e,
    int* __restrict__ wl_r0, int* __restrict__ wl_cnt, int* __restrict__ wl_tok,
    int* __restrict__ uniq_e, int* __restrict__ counters) {
  __shared__ int hist[1026];
  __shared__ int start[1026];   // row start per bin (in "item order")
  __shared__ int istart[1026];  // item start per bin
  __shared__ int ustart[1026];  // uniq index per bin
  __shared__ int cursor[1026];
  const int tid = threadIdx.x;
  const int NB = E + 2;
  // position of bin b in the item order
  auto bin_at = [E](int pos) { return pos == 0 ? E : (pos <= E ? pos - 1 : E + 1); };
  for (int b = tid; b < NB; b += 1024) { hist[b] = 0; cursor[b] = 0; }
  __syncthreads();
  for (int p = tid; p < P; p += 1024) {
    const int e = row_exp[p];
    const int b = (e >= 0 && e < E) ? e : ((e == E && has_extra) ? E : E + 1);
    atomicAdd(&hist[b], 1);
  }
  __syncthreads();
  if (tid < 64) {
    const int lane = tid;
    const int CH = (NB + 63) / 64;
    int rows = 0, items = 0, uniq = 0;
    for (int c = 0; c < CH; ++c) {
      const int pos = lane * CH + c;
      if (pos < NB) {
        const int b = bin_at(pos);
        const int n = hist[b];
        rows += n;
        items += (n + RC - 1) / RC;
        uniq += (b != E + 1 && n > 0) ? 1 : 0;
      }
    }
    const int my_rows = rows, my_items = items, my_uniq = uniq;
#pragma unroll
    for (int o = 1; o < 64; o <<= 1) {
      const int vr = __shfl_up(rows, o, 64), vi = __shfl_up(items, o, 64), vu = __shfl_up(uniq, o, 64);
      if (lane >= o) { rows += vr; items += vi; uniq += vu; }
    }
    const int tot_rows = __shfl(rows, 63, 64), tot_items = __shfl(items, 63, 64), tot_uniq = __shfl(uniq, 63, 64);
    int r = rows - my_rows, it = items - my_items, u = uniq - my_uniq;   // exclusive prefixes
    for (int c = 0; c < CH; ++c) {
      const int pos = lane * CH + c;
      if (pos < NB) {
        const int b = bin_at(pos);
        const int n = hist[b];
        start[b] = r; istart[b] = it; ustart[b] = u;
        r += n;
        it += (n + RC - 1) / RC;
        u += (b != E + 1 && n > 0) ? 1 : 0;
      }
    }
    if (lane == 63) {
      counters[0] = tot_items;
      counters[1] = tot_uniq;
      counters[2] = tot_rows - hist[E + 1];
    }
  }
  __syncthreads();
  // scatter rows
  for (int p = tid; p < P; p += 1024) {
    const int e = row_exp[p];
    const int b = (e >= 0 && e < E) ? e : ((e == E && has_extra) ? E : E + 1);
    const int pos = start[b] + atomicAdd(&cursor[b], 1);
    order[pos] = p;
    otok[pos] = row_tok[p];
    // item slot of this row: item istart[b] + (pos - start[b]) / RC, row (pos - start[b]) % RC
    const int q = pos - start[b];
    wl_tok[(istart[b] + q / RC) * RC + (q - (q / RC) * RC)] = row_tok[p];
  }
  // items + uniq list
  for (int b = tid; b < NB; b += 1024) {
    const int n = hist[b];
    if (n == 0) continue;
    const int eid = (b == E + 1) ? -1 : b;
    const int ni = (n + RC - 1) / RC;
    for (int c = 0; c < ni; ++c) {
      const int j = istart[b] + c;
      wl_e[j] = eid;
      wl_r0[j] = start[b] + c * RC;
      wl_cnt[j] = min(RC, n - c * RC);
    }
    if (b != E + 1) uniq_e[ustart[b]] = b;
  }
}

// ---------------------------------------------------------------------------------------------
// D1 slab, multi-row, persistent. LPR lanes per output row, 64/LPR rows per wave, lane l owns
// groups l + LPR*j. Weight tile loaded once per item, then the item's rows are looped with the
// next row's activation prefetched (PF=1) one row ahead.
// ---------------------------------------------------------------------------------------------
template <int K, int LPR, int WPB, int PF>
__global__ void __launch_bounds__(64 * WPB) slab_mr_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ order, const int* __restrict__ wl_e, const int* __restrict__ wl_r0,
    const int* __restrict__ wl_cnt, const int* __restrict__ counters, float* __restrict__ out,
    int N, int E, long stride_we, long stride_se, const uint32_t* __restrict__ w_extra,
    const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = (G + LPR - 1) / LPR;
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR;
  const int tiles = N / ROWS;
  const int n_work = __builtin_amdgcn_readfirstlane(counters[0]) * tiles;
  const int total_waves = gridDim.x * WPB;
  int wv = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  for (; wv < n_work; wv += total_waves) {
    const int j = wv / tiles, tile = wv - j * tiles;
    const int e = __builtin_amdgcn_readfirstlane(wl_e[j]);
    const int r0 = __builtin_amdgcn_readfirstlane(wl_r0[j]);
    const int cnt = __builtin_amdgcn_readfirstlane(wl_cnt[j]);
    const int n = tile * ROWS + r;
    const uint32_t* wbase;
    const uint16_t* sbase;
    if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
      for (int i = 0; i < cnt; ++i) {
        const int p = order[r0 + i];
        if (l == LPR - 1) out[(long)p * N + n] = 0.f;
      }
      continue;
    }
    const uint4* wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8)) + l;
    const uint16_t* sp = sbase + (long)n * G + l;
    uint4 q[GPL];
    uint32_t sc[GPL];
#pragma unroll
    for (int jj = 0; jj < GPL; ++jj) {
      if (G % LPR == 0 || l + LPR * jj < G) { q[jj] = wp[LPR * jj]; sc[jj] = sp[LPR * jj]; }
      else { q[jj] = make_uint4(0, 0, 0, 0); sc[jj] = 0; }
    }
    uint4 xa0[GPL], xb0[GPL], xa1[GPL], xb1[GPL];
    float sx0[GPL], sxs0[GPL], sx1[GPL], sxs1[GPL];
    int p0 = 0, p1 = 0;
#define LOAD_ACT(XA, XB, SX, SXS, PP, i)                                                   \
    {                                                                                     \
      const int p_ = __builtin_amdgcn_readfirstlane(order[r0 + (i)]);                     \
      const int t_ = __builtin_amdgcn_readfirstlane(row_tok[p_]);                         \
      PP = p_;                                                                            \
      const uint4* xp = reinterpret_cast<const uint4*>(x8 + (long)t_ * K) + 2 * l;        \
      const float* xsp = xs + (long)t_ * G + l;                                           \
      const float* xsump = xsum + (long)t_ * G + l;                                       \
      _Pragma("unroll") for (int jj = 0; jj < GPL; ++jj) {                                \
        if (G % LPR == 0 || l + LPR * jj < G) {                                           \
          XA[jj] = xp[2 * LPR * jj]; XB[jj] = xp[2 * LPR * jj + 1];                       \
          SX[jj] = xsp[LPR * jj]; SXS[jj] = xsump[LPR * jj];                              \
        } else {                                                                          \
          XA[jj] = make_uint4(0, 0, 0, 0); XB[jj] = XA[jj]; SX[jj] = 0.f; SXS[jj] = 0.f;  \
        }                                                                                 \
      }                                                                                   \
    }
#define COMPUTE_ACT(XA, XB, SX, SXS, PP)                                                  \
    {                                                                                     \
      float acc = 0.f;                                                                    \
      _Pragma("unroll") for (int jj = 0; jj < GPL; ++jj) {                                \
        const uint32_t qw[4] = {q[jj].x, q[jj].y, q[jj].z, q[jj].w};                      \
        const int xd[8] = {(int)XA[jj].x, (int)XA[jj].y, (int)XA[jj].z, (int)XA[jj].w,    \
                           (int)XB[jj].x, (int)XB[jj].y, (int)XB[jj].z, (int)XB[jj].w};   \
        int sumi = 0;                                                                     \
        _Pragma("unroll") for (int wd = 0; wd < 4; ++wd) {                                \
          sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);                            \
          sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);                 \
        }                                                                                 \
        acc += bf2f(sc[jj]) * (SX[jj] * (float)sumi - 8.f * SXS[jj]);                     \
      }                                                                                   \
      acc = reduce_lpr<LPR>(acc);                                                         \
      if (l == LPR - 1) out[(long)PP * N + n] = acc;                                      \
    }
    if constexpr (PF == 0) {
      for (int i = 0; i < cnt; ++i) {
        LOAD_ACT(xa0, xb0, sx0, sxs0, p0, i);
        __builtin_amdgcn_sched_barrier(0);
        COMPUTE_ACT(xa0, xb0, sx0, sxs0, p0);
      }
    } else {
      LOAD_ACT(xa0, xb0, sx0, sxs0, p0, 0);
      __builtin_amdgcn_sched_barrier(0);
      for (int i = 0; i < cnt; i += 2) {
        if (i + 1 < cnt) LOAD_ACT(xa1, xb1, sx1, sxs1, p1, i + 1);
        __builtin_amdgcn_sched_barrier(0);
        COMPUTE_ACT(xa0, xb0, sx0, sxs0, p0);
        if (i + 1 >= cnt) break;
        if (i + 2 < cnt) LOAD_ACT(xa0, xb0, sx0, sxs0, p0, i + 2);
        __builtin_amdgcn_sched_barrier(0);
        COMPUTE_ACT(xa1, xb1, sx1, sxs1, p1);
      }
    }
#undef LOAD_ACT
#undef COMPUTE_ACT
  }
}

// ---------------------------------------------------------------------------------------------
// D2 rowlane, multi-row, persistent. K = 160 (5 groups = 80 B per output row), one output row per
// lane, 64 rows per wave; the row's activation (40 dwords + 10 floats) is wave-uniform -> SGPRs.
// ---------------------------------------------------------------------------------------------
template <int K, int WPB, int OCC = 1>
__global__ void __launch_bounds__(64 * WPB, OCC) rowlane_mr_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ order, const int* __restrict__ wl_e, const int* __restrict__ wl_r0,
    const int* __restrict__ wl_cnt, const int* __restrict__ counters, float* __restrict__ out,
    int N, int E, long stride_we, long stride_se, const uint32_t* __restrict__ w_extra,
    const uint16_t* __restrict__ s_extra) {
  constexpr int G = K / 32;
  const int lane = threadIdx.x & 63;
  const int tiles = N / 64;
  const int n_work = __builtin_amdgcn_readfirstlane(counters[0]) * tiles;
  const int total_waves = gridDim.x * WPB;
  int wv = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  for (; wv < n_work; wv += total_waves) {
    const int j = wv / tiles, tile = wv - j * tiles;
    const int e = __builtin_amdgcn_readfirstlane(wl_e[j]);
    const int r0 = __builtin_amdgcn_readfirstlane(wl_r0[j]);
    const int cnt = __builtin_amdgcn_readfirstlane(wl_cnt[j]);
    const int n = tile * 64 + lane;
    const uint32_t* wbase;
    const uint16_t* sbase;
    if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
      for (int i = 0; i < cnt; ++i) {
        const int p = order[r0 + i];
        out[(long)p * N + n] = 0.f;
      }
      continue;
    }
    const uint4* wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8));
    const uint16_t* sp = sbase + (long)n * G;
    uint4 q[G];
    uint32_t sc[G];
#pragma unroll
    for (int jj = 0; jj < G; ++jj) { q[jj] = wp[jj]; sc[jj] = sp[jj]; }
    for (int i = 0; i < cnt; ++i) {
      const int p = __builtin_amdgcn_readfirstlane(order[r0 + i]);
      const int t = __builtin_amdgcn_readfirstlane(row_tok[r0 + i]);   // row_tok == otok (sorted order)
      const int* xu = reinterpret_cast<const int*>(x8 + (long)t * K);
      const float* xsu = xs + (long)t * G;
      const float* xsumu = xsum + (long)t * G;
      int xv[8 * G]; float xsv[G], xsumv[G];
#pragma unroll
      for (int ii = 0; ii < 8 * G; ++ii) xv[ii] = xu[ii];
#pragma unroll
      for (int jj = 0; jj < G; ++jj) { xsv[jj] = xsu[jj]; xsumv[jj] = xsumu[jj]; }
      __builtin_amdgcn_sched_barrier(0);
      float acc = 0.f;
#pragma unroll
      for (int jj = 0; jj < G; ++jj) {
        const uint32_t qw[4] = {q[jj].x, q[jj].y, q[jj].z, q[jj].w};
        int sumi = 0;
#pragma unroll
        for (int wd = 0; wd < 4; ++wd) {
          sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xv[8 * jj + 2 * wd], sumi);
          sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xv[8 * jj + 2 * wd + 1], sumi);
        }
        acc += bf2f(sc[jj]) * (xsv[jj] * (float)sumi - 8.f * xsumv[jj]);
      }
      out[(long)p * N + n] = acc;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// D1 slab, multi-row, persistent, LDS-STAGED activations ("slab_mr_lds").
//
// Why: the register variant above re-fetches the token row through the texture path for every
// (row, n-tile): per row-tile 10 dwordx4 + 10 dword vector loads (4x redundant across the wave's 4
// output rows), i.e. ~20 VMEM instructions x 16 clk of address issue per row-tile.  At M = 128 that
// is ~1.1 GB of L2->L1 activation traffic against 0.22 GB of weights and the slab runs at ~55% of the
// weight-stream floor (392 us, sweep_48_128.log).  Here one block = WPB waves = WPB consecutive
// n-tiles of the SAME item; the block stages the item's rows (int8 payload + block scales) in LDS
// once and every wave reads them with ds_read_b128, so the activation goes through L1 once per
// block instead of once per wave-tile (16x less at WPB = 4) and the VMEM path carries only weights.
// The item's rows are consumed in sub-chunks of RCS rows (LDS capacity); the weight tile stays in
// registers across sub-chunks, so RC (the sort's chunk = weight reuse) is independent of RCS.
// Same lane->group mapping, dot4 order, per-group formula and DPP reduction as the shipping slab:
// out[p] is bit-identical to gemv_slab / slab_mr.
// ---------------------------------------------------------------------------------------------
// LDS layout: xla[i][g] / xlb[i][g] hold group g's even-k / odd-k dwords (one uint4 each), so lane
// l's reads (g = l + LPR*jj) are 16 consecutive 16 B chunks -- conflict-free.  The interleaved
// [2g | 2g+1] layout of the shipping prep-fold kernel puts a 32 B stride between lanes, which is a
// 2-way bank conflict on ds_read_b128 (measured here, see REPORT.md sec. 4).
// DBG (diagnostics only, via pf = 2/3): 1 = skip the row loop (stream + staging only), 2 = rows
// computed from the weight registers instead of LDS (no ds_read of x).
template <int K, int LPR, int WPB, int RCS, int DBG = 0, int OCC = 1>
__global__ void __launch_bounds__(64 * WPB, OCC) slab_mr_lds_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ order,
    const int* __restrict__ otok, const int* __restrict__ wl_e, const int* __restrict__ wl_r0,
    const int* __restrict__ wl_cnt, const int* __restrict__ counters, float* __restrict__ out,
    int N, int E, long stride_we, long stride_se, const uint32_t* __restrict__ w_extra,
    const uint16_t* __restrict__ s_extra) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = G / LPR, U4 = K / 16, NTHR = 64 * WPB;
  static_assert(G % LPR == 0, "slab_mr_lds: K/32 must be a multiple of LPR");
  __shared__ __align__(16) uint4 xla[RCS][G], xlb[RCS][G];
  __shared__ float xsl[RCS][G], xsuml[RCS][G];
  __shared__ int lp[RCS];
  const int lane = threadIdx.x & 63, r = lane / LPR, l = lane % LPR, wv = threadIdx.x >> 6;
  const int tiles = N / ROWS;                 // host checks tiles % WPB == 0
  const int bpi = tiles / WPB;                // blocks per item
  const int n_blocks = __builtin_amdgcn_readfirstlane(counters[0]) * bpi;
  for (int blk = blockIdx.x; blk < n_blocks; blk += gridDim.x) {
    const int j = blk / bpi, tb = blk - j * bpi;
    const int e = __builtin_amdgcn_readfirstlane(wl_e[j]);
    const int r0 = __builtin_amdgcn_readfirstlane(wl_r0[j]);
    const int cnt = __builtin_amdgcn_readfirstlane(wl_cnt[j]);
    const int n = (tb * WPB + wv) * ROWS + r;
    const uint32_t* wbase;
    const uint16_t* sbase;
    if (!pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase)) {
      // block-uniform branch (e is per item): no barrier is skipped by part of the block
      for (int i = 0; i < cnt; ++i) {
        const int p = order[r0 + i];
        if (l == LPR - 1) out[(long)p * N + n] = 0.f;
      }
      continue;
    }
    // the weight tile, issued before the staging so it is in flight across the barrier
    const uint4* wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8)) + l;
    const uint16_t* sp = sbase + (long)n * G + l;
    uint4 q[GPL];
    uint32_t sc[GPL];
#pragma unroll
    for (int jj = 0; jj < GPL; ++jj) { q[jj] = wp[LPR * jj]; sc[jj] = sp[LPR * jj]; }
    for (int c0 = 0; c0 < cnt; c0 += RCS) {
      const int nr = min(RCS, cnt - c0);
      __syncthreads();   // the previous sub-chunk (or the previous item) is fully consumed
      for (int idx = threadIdx.x; idx < nr * U4; idx += NTHR) {
        const int i = idx / U4, c = idx - i * U4;
        const int t = otok[r0 + c0 + i];
        const uint4 v = reinterpret_cast<const uint4*>(x8 + (long)t * K)[c];
        if (c & 1) xlb[i][c >> 1] = v; else xla[i][c >> 1] = v;
      }
      for (int idx = threadIdx.x; idx < nr * G; idx += NTHR) {
        const int i = idx / G, g = idx - i * G;
        const int t = otok[r0 + c0 + i];
        xsl[i][g] = xs[(long)t * G + g];
        xsuml[i][g] = xsum[(long)t * G + g];
      }
      if (threadIdx.x < nr) lp[threadIdx.x] = order[r0 + c0 + threadIdx.x];
      __syncthreads();
      if constexpr (DBG == 1) continue;
      for (int i = 0; i < nr; ++i) {
        const int p = lp[i];
        float acc = 0.f;
#pragma unroll
        for (int jj = 0; jj < GPL; ++jj) {
          const int g = l + LPR * jj;
          uint4 xa, xb;
          if constexpr (DBG == 2) { xa = q[jj]; xb = q[(jj + 1) % GPL]; }
          else { xa = xla[i][g]; xb = xlb[i][g]; }
          const uint32_t qw[4] = {q[jj].x, q[jj].y, q[jj].z, q[jj].w};
          const int xd[8] = {(int)xa.x, (int)xa.y, (int)xa.z, (int)xa.w,
                             (int)xb.x, (int)xb.y, (int)xb.z, (int)xb.w};
          int sumi = 0;
#pragma unroll
          for (int wd = 0; wd < 4; ++wd) {
            sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
            sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
          }
          acc += bf2f(sc[jj]) * (xsl[i][g] * (float)sumi - 8.f * xsuml[i][g]);
        }
        acc = reduce_lpr<LPR>(acc);
        if (l == LPR - 1) out[(long)p * N + n] = acc;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// slab_mr_lds, SOFTWARE-PIPELINED ("slab_mr_lds_pipe", pf = 1).
//
// What the ISA of the plain LDS kernel shows (isa/moe_mr.s, REPORT.md sec. 4): on gfx908 the
// compiler turns every __syncthreads() into `s_waitcnt vmcnt(0); s_barrier`, so both staging
// barriers drain every in-flight weight load; the staging loop is `global_load; vmcnt(0); ds_write`
// one chunk at a time; and every unit starts a THREE-deep dependent load chain (item meta -> row
// token -> activation / weight) whose round trips are several us each once HBM is saturated.  Per
// block-task the stream, the staging and the row loop therefore run one after the other, and the
// slab reaches ~0.65 TB/s at M = 128 (328-373 us for 216 MB).
//
// This variant:
//   * barrier = CK's `s_waitcnt lgkmcnt(0); s_barrier` (inline asm, no vmcnt drain);
//   * a unit's meta (wl_e/wl_r0/wl_cnt[j]) and its rows' tokens (wl_tok[j*RC + i], written by the
//     sort so they are indexed by j alone) are fetched ONE unit ahead with plain vector loads
//     (vmcnt-tracked, so no lgkmcnt(0) ever waits for them), i.e. when a unit is due its
//     activation and weight addresses are already known;
//   * after the row loop of unit u the activations of u+1 are issued, then its weights, then the
//     meta of u+2, and only then the barrier + LDS write-back: the write-back waits for the
//     activation loads only (vmcnt retires in order), the weights and the meta stay in flight
//     across both barriers and are consumed by the next row loop.
//   Per unit the exposed latency is one round trip (activations) instead of three, no register
//   holds staged activations across the row loop (single weight buffer, ~80 VGPRs, 3 waves/SIMD).
// Numerics identical to slab_mr_lds (same per-row code).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void lds_barrier() {
  // CK's block_sync_lds: LDS traffic of this wave complete, then the barrier; no vmcnt drain.
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <int K, int LPR, int WPB, int RCS>
__global__ void __launch_bounds__(64 * WPB) slab_mr_lds_pipe_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ order,
    const int* __restrict__ wl_tok, const int* __restrict__ wl_e, const int* __restrict__ wl_r0,
    const int* __restrict__ wl_cnt, const int* __restrict__ counters, float* __restrict__ out,
    int N, int E, long stride_we, long stride_se, const uint32_t* __restrict__ w_extra,
    const uint16_t* __restrict__ s_extra, int RC) {
  constexpr int ROWS = 64 / LPR, G = K / 32, GPL = G / LPR, U4 = K / 16;
  constexpr int RPW = (RCS + WPB - 1) / WPB;          // rows staged per wave
  constexpr int XR = (U4 + 63) / 64;                  // uint4 per lane per row
  constexpr int XSR = (G + 63) / 64;                  // scales per lane per row
  static_assert(G % LPR == 0, "slab_mr_lds_pipe: K/32 must be a multiple of LPR");
  __shared__ __align__(16) uint4 xla[RCS][G], xlb[RCS][G];
  __shared__ float xsl[RCS][G], xsuml[RCS][G];
  __shared__ int lp[RCS];
  const int tid = threadIdx.x;
  const int lane = tid & 63, r = lane / LPR, l = lane % LPR, wv = tid >> 6;
  const int tiles = N / ROWS;
  const int bpi = tiles / WPB;
  const int n_blocks = __builtin_amdgcn_readfirstlane(counters[0]) * bpi;
  const int stride = gridDim.x;

  // ---- meta of a unit, fetched with vector loads (all lanes the same address) ----
  struct Meta { int e, r0, cnt, tok[RPW]; };
  auto issue_meta = [&](int blk, int c0, Meta& m) {
    const int jb = (blk < n_blocks) ? blk / bpi : 0;         // clamp: always a valid index
    const int c0b = (blk < n_blocks) ? c0 : 0;
    m.e = wl_e[jb]; m.r0 = wl_r0[jb]; m.cnt = wl_cnt[jb];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) m.tok[rr] = wl_tok[jb * RC + c0b + min(wv + rr * WPB, RCS - 1)];
  };
  struct Unit { int blk, c0, nr, r0, n; bool valid; const uint4* wp; const uint16_t* sp; int tok[RPW]; int cnt; };
  auto finalize = [&](int blk, int c0, const Meta& m, Unit& u) {
    u.blk = blk; u.c0 = c0;
    const int e = __builtin_amdgcn_readfirstlane(m.e);
    u.r0 = __builtin_amdgcn_readfirstlane(m.r0);
    u.cnt = __builtin_amdgcn_readfirstlane(m.cnt);
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) u.tok[rr] = __builtin_amdgcn_readfirstlane(m.tok[rr]);
    const int j = blk / bpi, tb = blk - j * bpi;
    u.n = (tb * WPB + wv) * ROWS + r;
    if (blk < n_blocks) {
      u.nr = min(RCS, u.cnt - c0);
      const uint32_t* wbase;
      const uint16_t* sbase;
      u.valid = pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase);
      if (u.valid) {
        u.wp = reinterpret_cast<const uint4*>(wbase + (long)u.n * (K / 8)) + l;
        u.sp = sbase + (long)u.n * G + l;
      } else {
        u.wp = reinterpret_cast<const uint4*>(w) + l;
        u.sp = s + l;
      }
    } else {
      u.nr = 0; u.valid = false;
      u.wp = reinterpret_cast<const uint4*>(w) + l;
      u.sp = s + l;
    }
  };
  auto next_pos = [&](const Unit& u, int& blk, int& c0) {
    if (u.blk < n_blocks && u.c0 + RCS < u.cnt) { blk = u.blk; c0 = u.c0 + RCS; }
    else { blk = u.blk + stride; c0 = 0; }
  };
  // activations of a unit -> registers (unconditional; clamped rows / lanes)
  auto issue_x = [&](const Unit& u, uint4 (*xr)[XR], float (*xsr)[XSR], float (*xsumr)[XSR]) {
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
      const int t = (wv + rr * WPB < u.nr) ? u.tok[rr] : 0;   // rows past the unit: any valid row
      const uint4* xrow = reinterpret_cast<const uint4*>(x8 + (long)t * K);
#pragma unroll
      for (int k = 0; k < XR; ++k) xr[rr][k] = xrow[min(lane + 64 * k, U4 - 1)];
#pragma unroll
      for (int k = 0; k < XSR; ++k) {
        const int g = min(lane + 64 * k, G - 1);
        xsr[rr][k] = xs[(long)t * G + g];
        xsumr[rr][k] = xsum[(long)t * G + g];
      }
    }
  };
  auto write_x = [&](const Unit& u, const uint4 (*xr)[XR], const float (*xsr)[XSR],
                     const float (*xsumr)[XSR]) {
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
      const int i = wv + rr * WPB;
      if (i < u.nr) {
#pragma unroll
        for (int k = 0; k < XR; ++k) {
          const int c = lane + 64 * k;
          if (U4 % 64 == 0 || c < U4) { if (c & 1) xlb[i][c >> 1] = xr[rr][k]; else xla[i][c >> 1] = xr[rr][k]; }
        }
#pragma unroll
        for (int k = 0; k < XSR; ++k) {
          const int g = lane + 64 * k;
          if (G % 64 == 0 || g < G) { xsl[i][g] = xsr[rr][k]; xsuml[i][g] = xsumr[rr][k]; }
        }
        if (lane == 0) lp[i] = order[u.r0 + u.c0 + i];
      }
    }
  };
  auto issue_w = [&](const Unit& u, uint4* q, uint32_t* sc) {
#pragma unroll
    for (int jj = 0; jj < GPL; ++jj) { q[jj] = u.wp[LPR * jj]; sc[jj] = u.sp[LPR * jj]; }
  };

  uint4 q[GPL];
  uint32_t sc[GPL];
  Unit u;
  Meta m2;
  int blk2, c02;
  {
    // prologue: unit 0 staged, its weights issued, meta of unit 1 in flight
    Meta m0;
    issue_meta(blockIdx.x, 0, m0);
    finalize(blockIdx.x, 0, m0, u);
    uint4 xr[RPW][XR];
    float xsr[RPW][XSR], xsumr[RPW][XSR];
    issue_x(u, xr, xsr, xsumr);
    issue_w(u, q, sc);
    next_pos(u, blk2, c02);
    issue_meta(blk2, c02, m2);
    lds_barrier();
    write_x(u, xr, xsr, xsumr);
    lds_barrier();
  }
  while (u.blk < n_blocks) {
    // ---- row loop of u (weights in q, activations in LDS) ----
    if (u.valid) {
      for (int i = 0; i < u.nr; ++i) {
        const int p = lp[i];
        float acc = 0.f;
#pragma unroll
        for (int jj = 0; jj < GPL; ++jj) {
          const int g = l + LPR * jj;
          const uint4 xa = xla[i][g], xb = xlb[i][g];
          const uint32_t qw[4] = {q[jj].x, q[jj].y, q[jj].z, q[jj].w};
          const int xd[8] = {(int)xa.x, (int)xa.y, (int)xa.z, (int)xa.w,
                             (int)xb.x, (int)xb.y, (int)xb.z, (int)xb.w};
          int sumi = 0;
#pragma unroll
          for (int wd = 0; wd < 4; ++wd) {
            sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xd[2 * wd], sumi);
            sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xd[2 * wd + 1], sumi);
          }
          acc += bf2f(sc[jj]) * (xsl[i][g] * (float)sumi - 8.f * xsuml[i][g]);
        }
        acc = reduce_lpr<LPR>(acc);
        if (l == LPR - 1) out[(long)p * N + u.n] = acc;
      }
    } else {
      for (int i = 0; i < u.nr; ++i) {
        if (l == LPR - 1) out[(long)lp[i] * N + u.n] = 0.f;
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    // ---- next unit: activations, then weights, then the meta of the unit after it ----
    Unit u2;
    finalize(blk2, c02, m2, u2);
    uint4 xr[RPW][XR];
    float xsr[RPW][XSR], xsumr[RPW][XSR];
    issue_x(u2, xr, xsr, xsumr);
    __builtin_amdgcn_sched_barrier(0);
    issue_w(u2, q, sc);
    __builtin_amdgcn_sched_barrier(0);
    int blk3, c03;
    next_pos(u2, blk3, c03);
    issue_meta(blk3, c03, m2);
    __builtin_amdgcn_sched_barrier(0);
    lds_barrier();                   // every wave is done reading u's LDS
    write_x(u2, xr, xsr, xsumr);     // waits for the activation loads only (in-order vmcnt)
    lds_barrier();
    u = u2; blk2 = blk3; c02 = c03;
  }
}

// ---------------------------------------------------------------------------------------------
// rowlane_mr with the same cross-item weight prefetch ("rowlane_mr_pf"): the next (item, n-tile)'s
// 5 uint4 + 5 scales per lane are issued before the current item's rows are computed.
// ---------------------------------------------------------------------------------------------
template <int K, int WPB>
__global__ void __launch_bounds__(64 * WPB) rowlane_mr_pf_kernel(
    const uint32_t* __restrict__ w, const uint16_t* __restrict__ s, const int8_t* __restrict__ x8,
    const float* __restrict__ xs, const float* __restrict__ xsum, const int* __restrict__ row_tok,
    const int* __restrict__ order, const int* __restrict__ wl_e, const int* __restrict__ wl_r0,
    const int* __restrict__ wl_cnt, const int* __restrict__ counters, float* __restrict__ out,
    int N, int E, long stride_we, long stride_se, const uint32_t* __restrict__ w_extra,
    const uint16_t* __restrict__ s_extra) {
  constexpr int G = K / 32;
  const int lane = threadIdx.x & 63;
  const int tiles = N / 64;
  const int n_work = __builtin_amdgcn_readfirstlane(counters[0]) * tiles;
  const int total_waves = gridDim.x * WPB;
  auto tile_ptrs = [&](int wv, const uint4*& wp, const uint16_t*& sp, int& e, int& r0, int& cnt,
                       bool& valid) {
    if (wv < n_work) {
      const int j = wv / tiles, tile = wv - j * tiles;
      e = __builtin_amdgcn_readfirstlane(wl_e[j]);
      r0 = __builtin_amdgcn_readfirstlane(wl_r0[j]);
      cnt = __builtin_amdgcn_readfirstlane(wl_cnt[j]);
      const int n = tile * 64 + lane;
      const uint32_t* wbase;
      const uint16_t* sbase;
      valid = pick_expert(e, E, w, s, stride_we, stride_se, w_extra, s_extra, wbase, sbase);
      if (valid) {
        wp = reinterpret_cast<const uint4*>(wbase + (long)n * (K / 8));
        sp = sbase + (long)n * G;
      }
    } else {
      valid = false; e = -1; r0 = 0; cnt = 0;
    }
  };
  int wv = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 6));
  const uint4* wp = reinterpret_cast<const uint4*>(w) + lane;
  const uint16_t* sp = s + lane;
  int e, r0, cnt;
  bool valid;
  tile_ptrs(wv, wp, sp, e, r0, cnt, valid);
  uint4 q[G];
  uint32_t sc[G];
#pragma unroll
  for (int jj = 0; jj < G; ++jj) { q[jj] = wp[jj]; sc[jj] = sp[jj]; }
  for (; wv < n_work; wv += total_waves) {
    int e2, r02, cnt2;
    bool valid2;
    tile_ptrs(wv + total_waves, wp, sp, e2, r02, cnt2, valid2);
    uint4 q2[G];
    uint32_t sc2[G];
#pragma unroll
    for (int jj = 0; jj < G; ++jj) { q2[jj] = wp[jj]; sc2[jj] = sp[jj]; }
    const int n = (wv - (wv / tiles) * tiles) * 64 + lane;
    if (!valid) {
      for (int i = 0; i < cnt; ++i) out[(long)order[r0 + i] * N + n] = 0.f;
    } else {
      for (int i = 0; i < cnt; ++i) {
        const int p = __builtin_amdgcn_readfirstlane(order[r0 + i]);
        const int t = __builtin_amdgcn_readfirstlane(row_tok[r0 + i]);
        const int* xu = reinterpret_cast<const int*>(x8 + (long)t * K);
        const float* xsu = xs + (long)t * G;
        const float* xsumu = xsum + (long)t * G;
        int xv[8 * G]; float xsv[G], xsumv[G];
#pragma unroll
        for (int ii = 0; ii < 8 * G; ++ii) xv[ii] = xu[ii];
#pragma unroll
        for (int jj = 0; jj < G; ++jj) { xsv[jj] = xsu[jj]; xsumv[jj] = xsumu[jj]; }
        float acc = 0.f;
#pragma unroll
        for (int jj = 0; jj < G; ++jj) {
          const uint32_t qw[4] = {q[jj].x, q[jj].y, q[jj].z, q[jj].w};
          int sumi = 0;
#pragma unroll
          for (int wd = 0; wd < 4; ++wd) {
            sumi = dot4(qw[wd] & 0x0F0F0F0Fu, xv[8 * jj + 2 * wd], sumi);
            sumi = dot4((qw[wd] >> 4) & 0x0F0F0F0Fu, xv[8 * jj + 2 * wd + 1], sumi);
          }
          acc += bf2f(sc[jj]) * (xsv[jj] * (float)sumi - 8.f * xsumv[jj]);
        }
        out[(long)p * N + n] = acc;
      }
    }
#pragma unroll
    for (int jj = 0; jj < G; ++jj) { q[jj] = q2[jj]; sc[jj] = sc2[jj]; }
    e = e2; r0 = r02; cnt = cnt2; valid = valid2;
  }
}

// ---------------------------------------------------------------------------------------------
// Stream floor: read the weight + scale slabs of every unique active expert exactly once with
// coalesced dwordx4 loads (no math), persistent grid. bytes_w / bytes_s per expert (multiples of 16).
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) stream_experts_kernel(
    const uint4* __restrict__ w, const uint4* __restrict__ s, const int* __restrict__ uniq_e,
    const int* __restrict__ counters, int E, long w_u4_per_e, long s_u4_per_e, int chunks_w,
    int chunks_s, const uint4* __restrict__ w_extra, const uint4* __restrict__ s_extra,
    unsigned* __restrict__ sink) {
  // one work item = (uniq expert, chunk of 256 uint4 = 4 KiB) over w then s
  const int n_uniq = counters[1];
  const int per_e = chunks_w + chunks_s;
  const int n_work = n_uniq * per_e;
  unsigned acc = 0;
  for (int wi = blockIdx.x; wi < n_work; wi += gridDim.x) {
    const int u = wi / per_e, c = wi - u * per_e;
    const int e = uniq_e[u];
    const uint4* base;
    long off;
    if (c < chunks_w) {
      base = (e == E) ? w_extra : (w + (long)e * w_u4_per_e);
      off = (long)c * 256;
      if (off + threadIdx.x >= w_u4_per_e) continue;
    } else {
      base = (e == E) ? s_extra : (s + (long)e * s_u4_per_e);
      off = (long)(c - chunks_w) * 256;
      if (off + threadIdx.x >= s_u4_per_e) continue;
    }
    const uint4 v = base[off + threadIdx.x];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) sink[0] = acc;  // never true in practice; keeps the loads alive
}

#ifndef MOE_MR_NO_TORCH
// ---------------------------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------------------------
static void check_act(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& xsum, int K) {
  TORCH_CHECK(x8.scalar_type() == at::kChar && x8.is_contiguous() && x8.size(1) == K);
  TORCH_CHECK(xs.scalar_type() == at::kFloat && xsum.scalar_type() == at::kFloat);
  TORCH_CHECK(xs.is_contiguous() && xsum.is_contiguous() && xs.size(1) == K / 32);
}

// row_exp, row_tok int32 [P] -> order, otok, wl_e/wl_r0/wl_cnt [P], uniq_e [E+1], counters [4]
// wl_tok needs >= (P + (E + 2) * (rc - 1)) entries (every bin can leave one partial item).
void expert_sort(at::Tensor row_exp, at::Tensor row_tok, int64_t E, bool has_extra, int64_t rc,
                 at::Tensor order, at::Tensor otok, at::Tensor wl_e, at::Tensor wl_r0,
                 at::Tensor wl_cnt, at::Tensor wl_tok, at::Tensor uniq_e, at::Tensor counters) {
  TORCH_CHECK(row_exp.scalar_type() == at::kInt && row_exp.is_contiguous());
  TORCH_CHECK(row_tok.scalar_type() == at::kInt && row_tok.is_contiguous() && row_tok.numel() == row_exp.numel());
  TORCH_CHECK(E + 2 <= 1024, "expert_sort: E + 2 must be <= 1024");
  const int P = row_exp.numel();
  TORCH_CHECK(order.numel() >= P && otok.numel() >= P && wl_e.numel() >= P && wl_r0.numel() >= P && wl_cnt.numel() >= P);
  TORCH_CHECK(wl_tok.numel() >= P + (E + 2) * (rc - 1), "expert_sort: wl_tok too small");
  TORCH_CHECK(uniq_e.numel() >= E + 1 && counters.numel() >= 4);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  expert_sort_kernel<<<1, 1024, 0, stream>>>(
      row_exp.data_ptr<int32_t>(), row_tok.data_ptr<int32_t>(), P, (int)E, has_extra ? 1 : 0, (int)rc,
      order.data_ptr<int32_t>(), otok.data_ptr<int32_t>(), wl_e.data_ptr<int32_t>(),
      wl_r0.data_ptr<int32_t>(), wl_cnt.data_ptr<int32_t>(), wl_tok.data_ptr<int32_t>(),
      uniq_e.data_ptr<int32_t>(), counters.data_ptr<int32_t>());
}

template <int K, int LPR, int WPB, int RCS, int PF>
static void launch_slab_mr_lds(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x8,
                               const at::Tensor& xs, const at::Tensor& xsum, const at::Tensor& order,
                               const at::Tensor& otok, const at::Tensor& wl_e, const at::Tensor& wl_r0,
                               const at::Tensor& wl_cnt, const at::Tensor& counters, at::Tensor& out,
                               int blocks, const at::Tensor& w_extra, const at::Tensor& s_extra, int rc) {
  constexpr int ROWS = 64 / LPR;
  const int N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0 && (N / ROWS) % WPB == 0, "slab_mr_lds: N/ROWS must be a multiple of WPB");
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const uint32_t* wx = w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr;
  const uint16_t* sx = w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr;
  if (PF == 1) {
    // `otok` is the per-item-slot token table (wl_tok) for this variant, `rc` its row stride
    slab_mr_lds_pipe_kernel<K, LPR, WPB, RCS><<<blocks, 64 * WPB, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
        reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(),
        order.data_ptr<int32_t>(), otok.data_ptr<int32_t>(), wl_e.data_ptr<int32_t>(),
        wl_r0.data_ptr<int32_t>(), wl_cnt.data_ptr<int32_t>(), counters.data_ptr<int32_t>(),
        out.data_ptr<float>(), N, E, (long)N * (K / 8), (long)N * (K / 32), wx, sx, rc);
    return;
  }
  // pf 4 / 5: occupancy-forced variants (launch_bounds min 4 / 5 waves per EU -> <= 64 / 51 VGPRs)
  auto kern = PF == 2 ? slab_mr_lds_kernel<K, LPR, WPB, RCS, 1>
            : PF == 3 ? slab_mr_lds_kernel<K, LPR, WPB, RCS, 2>
            : PF == 4 ? slab_mr_lds_kernel<K, LPR, WPB, RCS, 0, 4>
            : PF == 5 ? slab_mr_lds_kernel<K, LPR, WPB, RCS, 0, 5>
                      : slab_mr_lds_kernel<K, LPR, WPB, RCS, 0>;
  kern<<<blocks, 64 * WPB, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
      reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(),
      order.data_ptr<int32_t>(), otok.data_ptr<int32_t>(), wl_e.data_ptr<int32_t>(),
      wl_r0.data_ptr<int32_t>(), wl_cnt.data_ptr<int32_t>(), counters.data_ptr<int32_t>(),
      out.data_ptr<float>(), N, E, (long)N * (K / 8), (long)N * (K / 32), wx, sx);
}

// LDS-staged multi-row slab: wpb in {4, 8} waves per block, rcs in {4, 8, 16} staged rows.
// pf = 0: plain; pf = 1: software-pipelined variant (pass wl_tok as `otok` and the sort's rc);
// pf = 2 / 3: diagnostics (no row loop / rows without LDS reads).
void gemv_slab_mr_lds(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
                      at::Tensor order, at::Tensor otok, at::Tensor wl_e, at::Tensor wl_r0,
                      at::Tensor wl_cnt, at::Tensor counters, at::Tensor out, int64_t wpb,
                      int64_t rcs, int64_t blocks, at::Tensor w_extra, at::Tensor s_extra,
                      int64_t pf, int64_t rc) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  TORCH_CHECK(w.scalar_type() == at::kInt && w.size(2) * 8 == K && w.is_contiguous());
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.is_contiguous() && s.size(2) == K / 32);
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous() && out.size(1) == w.size(1));
#define CASE(KK, W, R, P)                                                                               \
  if (K == KK && wpb == W && rcs == R && pf == P) {                                                     \
    launch_slab_mr_lds<KK, 16, W, R, P>(w, s, x8, xs, xsum, order, otok, wl_e, wl_r0, wl_cnt, counters, \
                                        out, (int)blocks, w_extra, s_extra, (int)rc);                  \
    return;                                                                                             \
  }
  CASE(2560, 4, 4, 0) CASE(2560, 4, 8, 0) CASE(2560, 8, 4, 0) CASE(2560, 8, 8, 0) CASE(2560, 4, 16, 0) CASE(2560, 8, 16, 0)
  CASE(2560, 4, 4, 1) CASE(2560, 4, 8, 1) CASE(2560, 8, 4, 1) CASE(2560, 8, 8, 1) CASE(2560, 4, 16, 1) CASE(2560, 8, 16, 1)
  CASE(2560, 4, 4, 2) CASE(2560, 4, 8, 2) CASE(2560, 4, 4, 3) CASE(2560, 4, 8, 3)
  CASE(2560, 4, 4, 4) CASE(2560, 4, 8, 4) CASE(2560, 8, 4, 4) CASE(2560, 8, 8, 4) CASE(2560, 4, 4, 5) CASE(2560, 4, 8, 5)
#undef CASE
  TORCH_CHECK(false, "gemv_slab_mr_lds: unsupported K/wpb/rcs ", K, "/", wpb, "/", rcs);
}

template <int K, int LPR, int WPB, int PF>
static void launch_slab_mr(const at::Tensor& w, const at::Tensor& s, const at::Tensor& x8,
                           const at::Tensor& xs, const at::Tensor& xsum, const at::Tensor& row_tok,
                           const at::Tensor& order, const at::Tensor& wl_e, const at::Tensor& wl_r0,
                           const at::Tensor& wl_cnt, const at::Tensor& counters, at::Tensor& out,
                           int blocks, const at::Tensor& w_extra, const at::Tensor& s_extra) {
  constexpr int ROWS = 64 / LPR;
  const int N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % ROWS == 0);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  slab_mr_kernel<K, LPR, WPB, PF><<<blocks, 64 * WPB, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()),
      reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(),
      row_tok.data_ptr<int32_t>(), order.data_ptr<int32_t>(), wl_e.data_ptr<int32_t>(),
      wl_r0.data_ptr<int32_t>(), wl_cnt.data_ptr<int32_t>(), counters.data_ptr<int32_t>(),
      out.data_ptr<float>(), N, E, (long)N * (K / 8), (long)N * (K / 32),
      w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr,
      w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr);
}

void gemv_slab_mr(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
                  at::Tensor row_tok, at::Tensor order, at::Tensor wl_e, at::Tensor wl_r0,
                  at::Tensor wl_cnt, at::Tensor counters, at::Tensor out, int64_t lpr, int64_t wpb,
                  int64_t pf, int64_t blocks, at::Tensor w_extra, at::Tensor s_extra) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  TORCH_CHECK(w.scalar_type() == at::kInt && w.size(2) * 8 == K && w.is_contiguous());
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.is_contiguous() && s.size(2) == K / 32);
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous() && out.size(1) == w.size(1));
#define CASE(KK, L, W, PFV)                                                                       \
  if (K == KK && lpr == L && wpb == W && pf == PFV) {                                              \
    launch_slab_mr<KK, L, W, PFV>(w, s, x8, xs, xsum, row_tok, order, wl_e, wl_r0, wl_cnt, counters, \
                                  out, (int)blocks, w_extra, s_extra);                             \
    return;                                                                                        \
  }
  CASE(2560, 16, 4, 0) CASE(2560, 16, 4, 1) CASE(2560, 16, 1, 0) CASE(2560, 16, 1, 1)
  CASE(2560, 32, 4, 0) CASE(2560, 32, 4, 1) CASE(2560, 32, 1, 0) CASE(2560, 32, 1, 1)
  CASE(2560, 8, 4, 0) CASE(2560, 8, 4, 1)
#undef CASE
  TORCH_CHECK(false, "gemv_slab_mr: unsupported K/lpr/wpb/pf ", K, "/", lpr, "/", wpb, "/", pf);
}

void gemv_rowlane_mr(at::Tensor w, at::Tensor s, at::Tensor x8, at::Tensor xs, at::Tensor xsum,
                     at::Tensor row_tok, at::Tensor order, at::Tensor wl_e, at::Tensor wl_r0,
                     at::Tensor wl_cnt, at::Tensor counters, at::Tensor out, int64_t wpb,
                     int64_t blocks, at::Tensor w_extra, at::Tensor s_extra, int64_t pf) {
  const int K = x8.size(1);
  check_act(x8, xs, xsum, K);
  TORCH_CHECK(K == 160 && w.scalar_type() == at::kInt && w.size(2) == 20 && w.is_contiguous());
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && s.is_contiguous() && s.size(2) == 5);
  const int N = w.size(1), E = w.size(0);
  TORCH_CHECK(N % 64 == 0 && out.scalar_type() == at::kFloat && out.is_contiguous() && out.size(1) == N);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const uint32_t* wx = w_extra.numel() ? reinterpret_cast<const uint32_t*>(w_extra.data_ptr()) : nullptr;
  const uint16_t* sx = w_extra.numel() ? reinterpret_cast<const uint16_t*>(s_extra.data_ptr()) : nullptr;
#define CASE(W)                                                                                      \
  if (wpb == W) {                                                                                    \
    auto kern = pf == 1 ? rowlane_mr_pf_kernel<160, W>                                                \
              : pf == 2 ? rowlane_mr_kernel<160, W, 6> : pf == 3 ? rowlane_mr_kernel<160, W, 8>        \
                        : rowlane_mr_kernel<160, W>;                                                  \
    kern<<<(int)blocks, 64 * W, 0, stream>>>(                                                        \
        reinterpret_cast<const uint32_t*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(s.data_ptr()), \
        reinterpret_cast<const int8_t*>(x8.data_ptr()), xs.data_ptr<float>(), xsum.data_ptr<float>(), \
        row_tok.data_ptr<int32_t>(), order.data_ptr<int32_t>(), wl_e.data_ptr<int32_t>(),           \
        wl_r0.data_ptr<int32_t>(), wl_cnt.data_ptr<int32_t>(), counters.data_ptr<int32_t>(),        \
        out.data_ptr<float>(), N, E, (long)N * 20, (long)N * 5, wx, sx);                             \
    return;                                                                                          \
  }
  CASE(1) CASE(2) CASE(4)
#undef CASE
  TORCH_CHECK(false, "gemv_rowlane_mr: wpb must be 1, 2 or 4");
}

// stream floor over the unique experts: w [E, N, K/8] int32, s [E, N, K/32] bf16 (both contiguous)
void stream_experts(at::Tensor w, at::Tensor s, at::Tensor uniq_e, at::Tensor counters,
                    int64_t blocks, at::Tensor w_extra, at::Tensor s_extra, at::Tensor sink) {
  const int E = w.size(0);
  const long wb = (long)w.size(1) * w.size(2) * 4, sb = (long)s.size(1) * s.size(2) * 2;
  TORCH_CHECK(wb % 16 == 0 && sb % 16 == 0);
  const long wu = wb / 16, su = sb / 16;
  const int cw = (int)((wu + 255) / 256), cs = (int)((su + 255) / 256);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  stream_experts_kernel<<<(int)blocks, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint4*>(s.data_ptr()),
      uniq_e.data_ptr<int32_t>(), counters.data_ptr<int32_t>(), E, wu, su, cw, cs,
      w_extra.numel() ? reinterpret_cast<const uint4*>(w_extra.data_ptr()) : nullptr,
      w_extra.numel() ? reinterpret_cast<const uint4*>(s_extra.data_ptr()) : nullptr,
      reinterpret_cast<unsigned*>(sink.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_sort", &expert_sort);
  m.def("gemv_slab_mr", &gemv_slab_mr);
  m.def("gemv_slab_mr_lds", &gemv_slab_mr_lds);
  m.def("gemv_rowlane_mr", &gemv_rowlane_mr);
  m.def("stream_experts", &stream_experts);
}
#else
// device-only build for resource-usage checks: instantiate the shipped configs
template __global__ void slab_mr_kernel<2560, 16, 4, 0>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_kernel<2560, 16, 4, 1>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_kernel<2560, 32, 4, 0>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_kernel<2560, 32, 4, 1>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_kernel<2560, 8, 4, 0>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_kernel<2560, 16, 4, 4>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_kernel<2560, 16, 4, 8>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_kernel<2560, 16, 8, 8>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_pipe_kernel<2560, 16, 4, 4>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*, int);
template __global__ void slab_mr_lds_pipe_kernel<2560, 16, 4, 8>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*, int);
template __global__ void rowlane_mr_pf_kernel<160, 4>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_kernel<2560, 16, 8, 8, 0, 4>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_kernel<2560, 16, 4, 4, 0, 4>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void slab_mr_lds_kernel<2560, 16, 4, 4, 0, 5>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void rowlane_mr_kernel<160, 4, 6>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void rowlane_mr_kernel<160, 4, 8>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
template __global__ void rowlane_mr_kernel<160, 4>(const uint32_t*, const uint16_t*, const int8_t*, const float*, const float*, const int*, const int*, const int*, const int*, const int*, const int*, float*, int, int, long, long, const uint32_t*, const uint16_t*);
#endif
