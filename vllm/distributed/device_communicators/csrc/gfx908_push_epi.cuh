// SPDX-License-Identifier: Apache-2.0
// gfx908 fused push-AR producer epilogue.
//
// The shipping sentinel push all-reduce (vllm/distributed/device_communicators/csrc/
// gfx908_push_ar.hip) has NO separate arrival flag: the payload IS the flag.  Every receive
// slot is pre-armed with the bf16 -0.0 sentinel (0x8000), the producer sanitizes -0.0 to +0.0
// so a real value can never look like the sentinel, and the consumer polls its own 16 B of each
// of the four source rows until none of them still reads as the sentinel.  Ownership is
// per element, so:
//
//   * there is nothing to order: no release fence, no arrival counter, no "last workgroup
//     flips the flag" step.  A producer workgroup that owns a slice can store that slice and
//     retire; the consumer thread that owns those elements sees them arrive on its own.
//   * a producer may therefore be spread over as many workgroups as it likes, and the four
//     ranks' workgroup counts / arrival order need not agree.
//
// That is what makes the *fused* producer cheap: the GEMV / reduce kernel that computes the
// partial simply stores its output tile into the three peers' slots (and its own) instead of
// into a local tensor, and the separate push_k launch disappears.
//
// Requirements on the fused epilogue (all enforced by the helpers below):
//   * 16 B granularity.  Per-dword atomicity over XGMI is what the sentinel test relies on
//     (the test is per half-word inside each dword), so a 16 B `global_store_dwordx4` is safe
//     and is 8x fewer store instructions than the natural 2 B scatter of a GEMV epilogue.
//     Sub-16 B stores are correct but pay a full fabric transaction per store; every epilogue
//     here gathers its lanes through LDS first so the fabric sees dwordx4.
//   * sanitize -0.0 -> +0.0 on every half-word.
//   * nontemporal (glc/slc) stores: the slot pages are MTYPE_UC, and the local copy must not
//     sit in L2 where the consumer's bypassing poll would not see it.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef unsigned int pe_u32x4 __attribute__((ext_vector_type(4)));

struct PushPtrs {
  void* base[4];
};

static constexpr unsigned PE_SENT16 = 0x8000u;
static constexpr unsigned PE_SENT32 = 0x80008000u;

// push mode of a fused producer kernel
#define PE_MODE_LOCAL 0  // stock: write the local output tensor only
#define PE_MODE_BOTH  1  // write the local output AND push (debug / soak only)
#define PE_MODE_PUSH  2  // push only: the local output tensor is not written at all

__device__ __forceinline__ unsigned pe_f2bf(float f) {  // torch RNE
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}
__device__ __forceinline__ unsigned pe_sanitize(unsigned w) {
  w = ((w & 0xffffu) == PE_SENT16) ? (w & 0xffff0000u) : w;
  w = ((w >> 16) == PE_SENT16) ? (w & 0x0000ffffu) : w;
  return w;
}
__device__ __forceinline__ unsigned pe_sanitize16(unsigned h) {  // single bf16 in the low half
  return (h == PE_SENT16) ? 0u : h;
}

// Store one 16 B chunk (8 bf16, already sanitized) into the same chunk index of all four
// ranks' slots.  `off16` = (site * 4 + my_rank) * slot_elems / 8; `i` = element_index / 8.
__device__ __forceinline__ void pe_push_chunk(const PushPtrs& pp, long off16, long i, pe_u32x4 v) {
#pragma unroll
  for (int r = 0; r < 4; r++) {
    __builtin_nontemporal_store(v, ((pe_u32x4*)pp.base[r]) + off16 + i);
  }
}

// Same, but only rank `r` (used when the epilogue spreads (chunk, rank) over lanes).
__device__ __forceinline__ void pe_push_chunk_r(const PushPtrs& pp, long off16, long i, pe_u32x4 v,
                                                int r) {
  __builtin_nontemporal_store(v, ((pe_u32x4*)pp.base[r]) + off16 + i);
}
