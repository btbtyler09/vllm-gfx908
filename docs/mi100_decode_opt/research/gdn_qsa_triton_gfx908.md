# GDN / QSA / glue Triton kernels on gfx908 — byte floors, codegen findings, ranked changes

Scope: Qwen3.8-Flash-Next (qwen4_exp) decode at c=1, TP4 on 4x MI100, the non-GEMM Triton
kernels in `_profile_c1_launch_map.txt`. Research only (2026-09-03): no GPU runs; all codegen
numbers below come from **offline** `triton.compile(..., target=GPUTarget("hip","gfx908",64))`
with Triton 3.7.0 on the host, which is the same major line as the serving image
(`docker/Dockerfile.mi100_base`: ROCm/triton `release/internal/3.7.x` @ f0b55c0, torch 2.12).
Probe scripts and .amdgcn dumps: `/tmp/gfx908_probe/` (qsa_probe.py, qsa_probe2.py, gdn_probe.py).

Framing: the whole GDN+QSA+glue family is ~1.6 ms of the 16.3 ms profiled step (~10%). The
levers here are second-order next to bf16 GDN projections (3.5 ms), W4 GEMV, MoE and all-reduce,
but they are also the only ones that are pure kernel/launch work with no numerics risk to the
W4 artifact. Realistic total from everything below: **~0.8–1.0 ms/token (5–6%, c=1 59.6 → ~63)**.

---

## 1. Shape facts per rank (TP4), from `/mnt/slow-storage/quant/Qwen3.8-Flash-Next-GPTQ-4bit/config.json`

| item | value |
|---|---|
| layers | 48 = 36 GDN (`linear_attention`) + 12 QSA (`full_attention`, interval 4) |
| GDN heads | H = 16 K-heads / 4 = **4**, HV = 48 V-heads / 4 = **12**, K = V = 128, conv width 4 |
| GDN state | `mamba_ssm_dtype: float32` → vLLM sets `mamba_ssm_cache_dtype=float32` (`vllm/model_executor/models/config.py:806-816`); state tensor `[slots, HV, V, K]` fp32 = **786,432 B/layer/rank** |
| GDN mixed_qkv row | (2·4·128 + 12·128)·2 B = 5 KB; conv state (2560 feats × 3 taps) |
| QSA heads | 24 Q / 2 KV → per rank **6 Q heads, 1 KV head**, head_dim 256, bf16 K/V, page 16 |
| QSA selection | `indexer_budget` 2048 tokens, `compress_ratio` 4 → index width 2048+3 = **2051** (`amd/indexer_qsa.py:159`) |
| Indexer | 4 heads × 128, 1 KV head, GemmaRMSNorm, MRoPE interleaved (section 11/11/10, partial rotary 0.25 → rotary_dim 64 of 256) |
| profiler | +2.4 us per kernel on gfx908 (launch-map note); real ≈ profiled − 2.4 |

---

## 2. Byte floor vs measured, per launch (c=1, one rank)

Floor = bytes / 1.2 TB/s (HBM2 peak); "realistic" = /0.8 TB/s. Real time = profiled − 2.4 us.

| kernel (count/token) | bytes moved | floor | realistic | profiled → real | headroom/launch | ×count |
|---|---|---|---|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` (36) | state R+W 1.57 MB + 5 KB in | 1.3 us | 2.0 us | 14.2 → **11.8** | ~9.5 us | **~340 us** |
| `_causal_conv1d_update_kernel` (36) | conv state 2560×3×2 B R+W ≈ 31 KB + x 5 KB | <0.1 us | launch floor ~1.3 us | 4.4 → 2.0 | ~0.7 (fusable) | ~70 us |
| `_qsa_sparse_paged_gqa_splitk_kernel` (12) | K+V gather 2×2051×256×2 B = 2.1 MB; partials write 64×6×256×4 = 393 KB | 2.1 us | 3.1 us | 16.4 → **14.0** | ~10 us | ~120 us |
| `_qsa_merge_splitk_kernel` (12) | 393 KB read, 3 KB write | 0.33 us | 0.5 us | 11.1 → **8.7** | ~8 us | ~100 us |
| `_qsa_mqa_paged_kernel` (12) | compressed K: (seq/4)×128×2 B ×4 heads scored; ≤ 0.5 MB @ 4k ctx | <0.5 us | | 7.2 → 4.8 | ~3.5 | ~40 us |
| `_compress_qsa_groups` (12) | 4×128×2 B ring reads | ~0 | launch floor | 7.3 → 4.9 | ~3.5 | ~40 us |
| `_store_qsa_rows` (36) | 256–1 KB | ~0 | launch floor | 3.9 → 1.5 | fusable | ~55 us |
| `_grouped_gemma_rmsnorm` (26) | ≤ 2 KB | ~0 | launch floor | 4.1 → 1.7 | fusable | ~45 us |
| `_triton_mrope_forward` (24) + `index_elementwise` gathers (16) | ≤ 4 KB | ~0 | launch floor | 3.9/6.8 → 1.5/4.4 | fusable | ~100 us |
| `reshape_and_cache_flash` (12) | 2×256×2 B | ~0 | launch floor | 7.3 → 4.9 | fusable into norm+rope | ~45 us |
| `topKPerRowDecode` (12), `_expand_qsa_indices` (12) | 8 KB | ~0 | launch floor | 4.0/3.9 → ~1.5 | expand fusable | ~20 us |

Totals: GDN family ≈ 0.51+0.16 = 0.67 ms profiled; QSA attention ≈ 0.33 ms; QSA index/glue ≈
0.5 ms. Nothing in this table is bandwidth-bound at c=1. The GDN kernel and the QSA split-K pair
are **latency-bound by their own structure** (too few waves, serialized reductions, dependent
load chains), the rest is launch floor and only fusion helps.

---

## 3. What the offline gfx908 codegen shows

### 3.1 QSA split-K kernel (`vllm/models/qwen4_exp/amd/ops/qsa.py:191`)

Launch at c=1 (`qsa_sparse_paged_attention`, lines 873-905): `base_programs = rows × kv_heads = 1`
→ `block_n=16, target_splits=64, num_warps=4, num_stages=1`; `num_tiles = cdiv(2051,16)=129`,
`num_splits=64` → grid (1,1,64), each program does ~2 tiles of 16 keys. Merge grid (1, 6 heads),
2 warps, `BLOCK_SPLITS=64`.

Compiled for gfx908 (Triton 3.7, GROUP_SIZE=6, HEAD_DIM=256):

| config | VGPR / AGPR | sgpr spills | occ (waves/SIMD) | MFMA used | notes |
|---|---|---|---|---|---|
| BLOCK_M=8 (the pre-fix TP4 case) | 256 / 256 | **39 vgpr + 139 sgpr spills** | 1 | `v_mfma_f32_4x4x2bf16 … cbsz:4 abid:N` ×64 for P·V; Q·K^T falls back to 272 `v_fmac_f32` | see 3.3 |
| BLOCK_M=16, BN=16, 4 warps (current) | 129 / 17 | 150 sgpr | 1 | `v_mfma_f32_16x16x8bf16` ×40 | loop body has **145 v_readlane + 125 v_writelane** |
| BLOCK_M=16, BN=32, 4 warps | 175 / 20 | 423 sgpr | 1 | 16x16x8bf16 ×80 | 475/396 readlane/writelane in loop |
| BLOCK_M=16, BN=64, 2 warps | 256 / 129 | 1079 sgpr | 1 | | unusable |
| BLOCK_M=16, BN=32, **8 warps** | 128 / 63 | 32 sgpr | 2 | 16x16x8bf16 ×72 | 284 readlane |
| BLOCK_M=16, BN=32, 8 warps, `AMDGCN_USE_BUFFER_OPS=0` | 128 / 29 | **19 sgpr** | 2 | ×72 | **23 readlane / 19 writelane** |
| BLOCK_M=16, BN=16, 4 warps, `AMDGCN_USE_BUFFER_OPS=0` | 122 / 17 | 81 sgpr | 2 | ×40 | 82/81 |
| BLOCK_M=32, BN=16 | 147 / 36 | 146 sgpr | 1 | 32x32x4bf16 ×8 + 16x16x8 ×32 | no benefit |

Observations:
- **Buffer ops are the readlane/writelane storm.** The default `AMDGCN_USE_BUFFER_OPS=1`
  converts the paged K/V gathers to `buffer_load_dword`, which needs a wave-uniform base per
  page; with 16 distinct pages per tile the backend waterfalls through `v_readlane_b32` and
  spills SGPRs (gfx908 has 102 SGPRs). Turning buffer ops off halves that at 4 warps and
  nearly eliminates it at 8 warps. Making strides `constexpr` does *not* help (127 spills).
- **V goes through LDS with 16-bit accesses** (`global_load_ushort` ×32–38, `ds_write_b16`
  ×32–48, `ds_read_u16`) because P·V needs V K-contiguous per lane but V is row-major
  [token, dim]. On gfx942 Triton fixes this with in-thread transpose (register `v_perm`),
  which is **off by default for every arch except gfx942** (`compiler.py:27`,
  `TRITON_HIP_USE_IN_THREAD_TRANSPOSE`). It uses only VALU perms, so it should work on gfx908 —
  worth an A/B. K loads are already `buffer_load_dword`/`global_load_dword` (kBase=2 bf16 → 4 B
  per lane per MFMA operand); `kpack=2` would double that to 64-bit loads.
- MFMA count is irrelevant: 40 × `16x16x8bf16` per tile ≈ 0.2 us. The kernel's time is
  the 3-deep dependent load chain per tile (indices → block_table → K/V), executed serially
  over ~2 tiles per program with `num_stages=1`, plus the spill/readlane VALU work, plus the
  merge kernel that does 64 KB per program on 6 programs of 2 warps.
- Occupancy is 1 wave/SIMD at 129 VGPRs (gfx908 VGPR file = 256 per lane, no unified AGPR
  pool). Irrelevant at c=1 (64 programs), relevant at c=48 (2,304 programs).

### 3.2 GDN packed decode kernel (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:256`)

Launch (line 436-483): `BK=128, BV=32, num_warps=1, num_stages=3`, grid `(NV=4, B·HV=12)` →
**48 single-wave programs on a 120-CU / 480-SIMD part**. That is the whole story at c=1.

| BV | warps | stages | state | VGPR | occ | loads/lane | cross-lane ops |
|---|---|---|---|---|---|---|---|
| 32 (current) | 1 | 3 | fp32 | 145 | 1 | 64 `global_load_dword` + 64 `global_store_dword` | 66 `ds_bpermute_b32` |
| 8 | 1 | 3 | fp32 | 47 | 5 | 16 + 16 dword | 18 ds_bpermute |
| 16 | 1 | 3 | fp32 | 81 | 3 | 32 + 32 | 34 |
| 32 | 4 | 3 or 1 | fp32 | 115 | 2 | 18 + 16 dword, plus 38 ds_read/22 ds_write + 13 `s_barrier` | 34 |
| 128 | 8 | 1 | fp32 | 127 | 2 | 34 + 32 | 66 + 13 barriers |
| 32 | 1 | 3 | bf16 state | 145 | 1 | 98 `global_load_ushort` / 65 `global_store_short` | 66 |

Observations:
- `num_stages` is a no-op here (no loop; identical code for 1 vs 3).
- State loads are **scalar dwords, never `dwordx4`**: the reduction over K (`tl.sum(..., 1)`)
  makes Triton pick a lane-along-K layout for the [BV,BK] state, so each lane holds one
  element per row. 64 lanes × 4 B is still a full 256 B line per instruction, so HBM
  efficiency is fine; the cost is instruction count and the fact that every `tl.sum` over K is
  a 6-step `ds_bpermute` butterfly (LDS round trip, ~100+ cycles each, serialized). 66 of them
  ≈ 4–5 us at 1.5 GHz — that, plus one wave's worth of memory parallelism for 32 KB, plus
  ~2 us of ramp, is the 11.8 us.
- **Do not switch the state to bf16 for speed**: it just turns the dword traffic into ushort
  traffic (98 loads/lane) and loses the fp32 recurrence the checkpoint was trained with.
- AITER's decode kernel (`aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_rearrange_sigmoid_gdr.py`,
  `BV=32, num_warps=4`) is the same algorithm with the same 48-program grid; at 4 warps the
  K-reduction crosses warps through LDS + `s_barrier`. Not a win by construction. The AITER
  path is not active on gfx908 anyway (profile shows the vLLM kernels; `GDN_AITER_TRITON_AVAILABLE`
  in `qwen_gdn_linear_attn.py:74` requires `rocm_aiter_ops.are_gdn_triton_kernels_available()`).
- Upstream FLA uses `BV=8, num_warps=1` for exactly this reason (more programs, fewer rows per
  reduction); vLLM's copy raised it to 32. AITER's `fused_sigmoid_gating_recurrent.py` HIP
  config is `BV=64, num_warps=4` — tuned for MI300 (304 CUs, 8 XCDs) not for a 48-task grid.

### 3.3 The BLOCK_M=8 miscompile — root cause in the compiler, not the kernel

`AccelerateAMDMatmul.cpp::chooseMfmaInstruction` (Triton main / 3.7 line) has a small-M path:

```cpp
} else if (minSize >= 4) {
  if (M >= 64)      { mDim = 64; nDim = 4; }
  else if (N >= 64) { mDim = 4;  nDim = 64; }
}
```

For `tl.dot(P[8,16], V[16,256])` this selects the **4x64 broadcast MFMA layout**, which on
gfx908 (mfma version 1) lowers to `v_mfma_f32_4x4x2bf16 … cbsz:4 abid:k` (64 per tile in the
dump), with `kWidth = kDim` for that case (`BlockedToMFMA::matchAndRewrite`). `Q·K^T` ([8,256]
x [256,16], N=16 < 64) fails MFMA selection and falls back to the FMA path
(`AccelerateBlocked::tryLegalizeFMA`, bf16 → f32 casts, 272 `v_fmac_f32`). The kernel also hits
256 VGPR + 256 AGPR with 39 VGPR spills at this shape. Release 3.4 did *not* have the
`minSize >= 4` branch (M<16 → FMA everywhere), so the garbage is specific to the 4x64 MFMA
layout on CDNA1 bf16 — a path nothing upstream tests on gfx908 (there is no gfx908 CI;
the newest CDNA1-adjacent work is the gfx906 target PR #9628). vLLM's `BLOCK_M >= 16` clamp
(`004b1ca779`) is the right fix: it routes both dots to `16x16x8bf16`, the tile the MFMA path
is validated on. Rule for gfx908: **never let a `tl.dot` see M or N < 16** — pad rows and mask.
AITER's `pa_decode.py:220-223` already does this (`query_grp_sz_pow2 = 16` when ≤ 16) and vLLM's
unified attention does the same (`triton_unified_attention.py:944`: `BLOCK_M = 16 if
num_queries_per_kv <= 16`).

The upstream issue does not exist yet. A minimal repro is the probe kernel with
`BLOCK_M=8, BLOCK_N=16, HEAD_DIM=256, bf16` on gfx908 vs the BLOCK_M=16 reference; file it against
triton-lang/triton with the `v_mfma_f32_4x4x2bf16 cbsz:4` dump attached.

---

## 4. Ranked changes (expected real us saved per token at c=1, 16.7 ms step)

Ranking is by (expected gain × confidence) / effort. Every item must pass the gate in §5.3
before a perf number is quoted; probe numbers are not results.

| # | change | where | expected | effort | risk |
|---|---|---|---|---|---|
| 1 | **GDN decode: more waves.** Set `BV=8` (grid 16×12 = 192 waves) or `BV=16` (96 waves), keep `num_warps=1`; drop `num_stages` to 1. Add `tl.max_contiguous(tl.multiple_of(o_k, 16), 16)` on `o_k` to let the state load vectorize. | `fused_recurrent.py:441-444` | 36 × (11.8 → ~4–5) ≈ **250 us** (1.5%) | 1 line + tune | none (same math, fp32) |
| 2 | **GDN decode: hand HIP kernel** (only if #1 stalls at ≥ 5 us). Layout: 4 lanes per V-row × 32 K-columns each, 16 rows/wave, state slice = 8 × `global_load_dwordx4` per lane; K-reduction = 2 DPP `quad_perm` steps (gfx908 has DPP row ops; no `ds_bpermute`); q, k streamed as wave-uniform from LDS `ds_read_b128`; 8 waves per head → 96 waves/layer. Fold the causal-conv update in: each wave recomputes the 3-tap conv for its head's q/k (3 FMA + silu on 256 values) and its own v slice, wave `(i_v==0, i_hv%3==0)` writes the q/k conv state, every wave writes its v slice. Same trick as `fused_moe/csrc/gfx908_w4gemv.hip` (JIT via `torch.utils.cpp_extension`). | new `vllm/models/qwen4_exp/amd/csrc/` | 36 × (11.8+2.0 → ~2.5) ≈ **400 us** (2.4%) | 2–3 days | fp32 recurrence must be bit-comparable to the Triton kernel (it is the same op order) |
| 3 | **QSA split-K: fewer, fatter, cheaper programs.** `BLOCK_N=32, num_warps=8, num_splits=32` (1–2 tiles each, occ 2, 19 SGPR spills), compile with `AMDGCN_USE_BUFFER_OPS=0` for this kernel (env is global — either set it for the whole worker and re-check the W4 GEMV/MoE kernels, or move the page gather out: see #4). Try `num_stages=2` so the index/page loads of tile i+1 overlap tile i. | `qsa.py:888-895` | 12 × (14 → ~8) ≈ **70 us** | tune | none; output bit-identical modulo split order (fp32 partials) |
| 4 | **Precompute physical slots.** Make `_expand_qsa_indices_kernel` (already knows request + block ids) emit `physical_page*16 + offset` int32 slots instead of logical tokens; the attention loop then has a 2-deep chain (slot → K/V) and no per-tile block-table gather, which is also what kills the buffer-op waterfall. | `qsa.py:117-190`, `:225-245` | part of #3, plus ~1–2 us/tile | ½ day | none |
| 5 | **QSA merge: parallelize or shrink.** Grid `(rows, heads, HEAD_DIM/64)` = 24 programs, 1 warp each, `BLOCK_SPLITS` = actual splits (32 after #3). Do **not** fuse the merge into the split kernel via a last-arriving atomic counter — that pattern lost 4% in-server for the MoE GEMV finalize (`qwen38_flash_next_gfx908.md`, "Fused split-K finalize"). | `qsa.py:357-398`, `:907-921` | 12 × (8.7 → ~2.5) ≈ **75 us** | ½ day | none |
| 6 | **Enable vLLM's Triton `fused_qk_rmsnorm_rope_gate` on ROCm** for the 12 main-attention layers: it is a Triton kernel (`vllm/model_executor/layers/fused_qk_norm_rope.py:16`) with `mrope_section` support, gated `current_platform.is_cuda()` at `qwen3_next.py:376-382` for no stated reason. Replaces q_norm + k_norm + cos/sin gather + `_triton_mrope_forward` (4–5 launches). Requires `is_neox_style` on this rope; if the model is GPT-J-style interleaved, port the `else` branch of `_triton_mrope_forward` into it. | `qwen3_next.py:379` | 12 × ~6 us ≈ **70 us** | small | must pass logprob parity at TP4 (new kernel path on gfx908) |
| 7 | **One indexer prologue kernel**: q GemmaRMSNorm + q MRoPE, and (compress ring pool → k GemmaRMSNorm → k MRoPE → store compressed row + raw row + rope-position row). All per-token, ≤ 4×128 + 128 values; today it is 2 rmsnorm + 2 rope + 2 gathers + 1 compress + 3 stores = 10 launches. Keep `_compress_qsa_groups_kernel` as the skeleton (it already walks the ring) and add the norm/rope/store epilogues; the q half is an independent program id. | `amd/indexer_qsa.py:170-270`, `qsa.py:401-583` | 12 × ~9 launches × ~1.5 us ≈ **160 us** | 1–2 days | parity gate; mrope interleaved math must be copied exactly from `mrope.py` |
| 8 | **Fold `_expand_qsa_indices` into the attention index load** (token = block·4 + offset, tail rows computed from `query_position`): −1 launch and −8 KB round trip per layer. Conflicts with #4's "emit physical slots" only in where the block-table lookup lives; pick #4. | `qsa.py` | 12 × 1.5 ≈ 20 us | small | none |
| 9 | **`reshape_and_cache_flash` into the fused norm/rope kernel** (AITER's `rope/fused_qkv_split_qk_norm_rope_cache.py` shows the pattern: split, norm, rope, write paged K/V in one Triton kernel; it is 1-D-RoPE/NeoX only, so extend #6 rather than adopt it). | `flash_attn.py:1219-1250` | 12 × ~4 us ≈ 50 us | with #6 | none |
| 10 | **In-thread transpose A/B**: `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` (global env) to replace the ushort/`ds_write_b16` V transpose in QSA (and in unified attention / AITER pa_decode which have the same P·V shape). Verify bit-exactness — this pass has only been validated on gfx942/gfx12. | env | unknown, maybe 1–2 us/tile | 0 | codegen path untested on gfx908 |
| 11 | `kpack=2` on the QSA dot (64-bit K operand loads). | `qsa.py` launch kwargs | small | 0 | check asm |

Not recommended:
- "GQA packing 6 heads × 3 into a 16/32-row tile": packing different KV tiles into the M
  dimension is invalid (one `tl.dot` shares the same K/V operand across all rows); packing
  different requests is invalid for the same reason; packing multiple query tokens of one
  request is only useful under MTP/spec, which is off. The 10 padding rows in BLOCK_M=16 cost
  nothing — the kernel is latency-bound, not MFMA-bound.
- bf16 SSM state, BV=128/8-warp GDN configs, BLOCK_M=32 for QSA (see tables above).
- Fusing the GDN in_proj GEMM into the recurrence: the projections are wvSplitK bf16 GEMMs at
  65–98% of HBM peak; the recurrence kernel cannot absorb a 0.4–4.6 MB weight read.

---

## 5. gfx908 Triton checklist

### 5.1 What the AMD backend does (Triton 3.7 line, `third_party/amd/backend/compiler.py`)

- `HIPOptions` defaults: `num_warps=4, waves_per_eu=0 (3.7; 1 in 3.4), num_stages=2, matrix_instr_nonkdim=0, kpack=1, schedule_hint='none', allow_flush_denorm=False`. `warp_size = 64` for gfx9.
- MFMA selection (`chooseMfmaInstruction`): min(M,N) ≥ 32 → 32x32, 16–31 → 16x16, 4–15 with the other dim ≥ 64 → **4x64 / 64x4 broadcast** (broken on gfx908, §3.3), else FMA fallback (bf16 upcast to f32). `matrix_instr_nonkdim=16|32` forces the tile. K must be a multiple of the intrinsic K (bf16: 8 for 16x16, 4 for 32x32; f16: 16 / 8; i8: 16 / 8) or it falls back to FMA.
- gfx908 (version 1) intrinsics Triton knows: `mfma_f32_{32x32x8,16x16x16,4x4x4}f16`, `mfma_f32_{32x32x4,16x16x8,4x4x2}bf16`, `mfma_i32_{32x32x8,16x16x16,4x4x4}i8`, `mfma_f32_{32x32x2,16x16x4,4x4x1}f32`. No TF32 (`allowXF32` needs version 3), no fp8, no `bf16_1k`. bf16 MFMA has half the K of f16 → half the throughput per instruction; if a kernel is genuinely MFMA-bound on gfx908, feeding f16 operands is 2x, but never for the QSA/GDN case.
- `kpack` multiplies kWidth (bf16 kBase=2 → 1 dword per lane per MFMA; `kpack=2` → 64-bit loads). Deprecated only on gfx950.
- Env knobs (`python/triton/knobs.py`, class `amd_knobs`): `AMDGCN_USE_BUFFER_OPS` (default **1** in 3.7; unset/0 in 3.4), `AMDGCN_USE_BUFFER_ATOMICS`, `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE`, `TRITON_HIP_USE_BLOCK_PINGPONG` (auto: gfx942 only; never gfx908), `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` (auto: gfx942 only), `TRITON_HIP_USE_ASYNC_COPY` (needs gfx950 LDS DMA; leave off), `TRITON_HIP_GLOBAL_PREFETCH` / `TRITON_HIP_LOCAL_PREFETCH` (3.4 stream-pipeliner distances; folded into `schedule_hint="local-prefetch"` in 3.7), `AMDGCN_SCALARIZE_PACKED_FOPS` (gfx950 perf knob; on gfx908 LLVM already scalarizes `v_pk_*_f32` since the ISA lacks them — this is why "v_pk_mul_f32" is a CK problem, not a Triton problem), `AMDGCN_ENABLE_DUMP=1` (print asm), `TRITON_OVERRIDE_ARCH`.
- `num_stages`: only affects kernels with a `for` loop containing loads; it drives the stream pipeliner (register double-buffering — gfx908 has no async global→LDS copy so "stages" cost VGPRs, not LDS). In a no-loop kernel like GDN decode it changes nothing (verified: identical asm for 1 vs 3). With 64 KB LDS and 256 VGPRs, stay at 1–2.
- `waves_per_eu=N` only adds `amdgpu-waves-per-eu` to make LLVM cap VGPRs; useful when you *want* occupancy (c≥16 QSA) and the kernel is at 129 VGPRs; it will spill if pushed too far.
- Buffer ops need every tensor < 2 GB (`HIPBackend.is_within_2gb`) or the load silently stays a global load; the KV cache exceeds 2 GB so the K/V gathers are already mixed (`global_load` for the cache, `buffer_load` for the index/table) — another reason the waterfall pattern appears.

### 5.2 Tuning procedure (offline first, GPU second)

1. **Compile offline on the host**, no GPU needed:
   `triton.compile(ASTSource(fn, signature, constexprs), target=GPUTarget("hip","gfx908",64), options={...}).asm["amdgcn"]` (see `/tmp/gfx908_probe/qsa_probe.py`). Read `.vgpr_count/.agpr_count/.sgpr_count/.vgpr_spill_count/.sgpr_spill_count`, `; Occupancy:`, and histogram the largest back-edge loop (`loop_an.py`). Reject any config with VGPR spills, or > ~50 SGPR spills in the hot loop, or `v_readlane/v_writelane` counts comparable to the load count.
2. Confirm the dot lowers to `v_mfma_f32_16x16x{16f16,8bf16}` or `32x32x*` — never `4x4x*` (broken) and never a pure `v_fmac` chain when MFMA was intended (that means M/N < 16 or K not a multiple of the intrinsic K).
3. Check load widths: want `global_load_dwordx4` / `buffer_load_dwordx4` on the byte-heavy operand. `_ushort` loads + `ds_write_b16` mean a transposed operand → try `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`, `kpack=2`, or restructure the load. Add `tl.max_contiguous(tl.multiple_of(offs, 16), 16)` on the contiguous index when the kernel is small (Triton can't prove alignment through int64 page math).
4. Count `ds_bpermute` / `s_barrier`: each is a serialized ~100-cycle latency; with 1 warp there are no barriers but `tl.sum` across lanes still costs 6 shuffle steps. For reduction-heavy decode kernels prefer layouts where the reduced axis is in-lane (small BV, many programs).
5. Grid size at the production shape must be ≥ ~240 waves (2 per CU) at c=1 or the kernel is at the ramp floor regardless of anything else; at c=48 check occupancy ≥ 2.
6. Then sweep on the GPU with `triton.testing.do_bench` under `HSA_OVERRIDE_GFX_VERSION=9.0.8`, in-graph (cudagraph capture) since profiler numbers inflate small kernels by 2.4 us; quote only in-server c=1 probes ≥ 3 runs, then the 12-tier bench.
7. Inductor-generated kernels (`triton_poi_fused_*`, `triton_per_fused_*`): same inspection via `TORCH_COMPILE_DEBUG=1` output dirs; the 2 slow ones (`triton_poi_fused_0` 14.7 us ×13, `triton_poi_fused_6` 11.7 us ×12) deserve the same asm look — they are not on this doc's list but they are 0.33 ms/token.

### 5.3 Validation gate (standing rule: kernel parity before perf)

- Unit reference in fp32 torch at the **production TP4 shapes**: GDN `(B=1..48, H=4, HV=12, K=V=128, fp32 state)`; QSA `(rows=1..48, 6 Q heads, 1 KV head, head_dim 256, TOPK width 2051, page 16, with -1 padding and out-of-range pages)`. Compare against the *current* kernel (bit-exact for GDN if op order is preserved; ≤ bf16 ulp for QSA since split order changes fp32 partial sums).
- Sweep every constexpr the launcher can pick (all `base_programs` profiles, `GROUP_SIZE ∈ {6,12,24}`, `NUM_SPLITS ∈ {1,…,64}`) — the BLOCK_M bug only appeared at TP4 because TP1/2 never compiled that instantiation.
- Per-layer logprob parity vs transformers on the 4-layer rehearsal artifact (TP1 ≤ 0.02 nats, TP4 ≤ 0.02 at the first QSA layer), then wikitext-2 PPL 3.136 ± 0.01 on the full model, then GSM8K as a coarse gate (±2% noise).
- Coherent text is not evidence; acceptance/PPL numbers are.

---

## 6. Fusion feasibility notes

**conv1d_update + GDN recurrence + glue**: feasible in one kernel (item #2). The conv is 3 taps
over a `[slots, 2560, 3]` ring (`causal_conv1d.py:763`, `BLOCK_N=256`, 10 programs); the GDN
program for `(i_v, i_hv)` needs conv'd q/k of head `i_hv//3` (256 values) and its v slice. Each
program recomputes q/k conv (3 FMAs + silu per value; ~0.2 us) — cheaper than a launch. The
only shared-write hazard is the conv-state shift for the q/k columns; assign it to one program
per K-head. The "in_proj activation glue" is already free: `a`/`b` are column slices of `ba`
with unit inner stride, so `.contiguous()` is a no-op and the kernel reads them strided. The
gated RMSNorm after the recurrence is inductor-fused (`triton_per_fused_*_rsqrt_sigmoid`) and
is bounded by the out_proj GEMM launch; fusing it into the recurrence kernel needs the full
V row per program (all 4 `i_v` slices) — do it only in the HIP kernel where one wave owns 16
full rows.

**QSA merge into split kernel**: technically easy (atomic tile counter, last CTA merges) but the
serialized tail was a measured in-server loss for the MoE finalize; prefer fewer splits + a
parallel merge (#3, #5). Revisit only with a real c=1 A/B.

**Norm + rope + cache write**: three separate fusions are available (#6, #7, #9); all are
per-token elementwise, LDS-free, and can be validated with a pure-torch reference in minutes.

---

## 7. References

Kernel sources (this tree, `/home/tyler/vllm-gfx908`):
- `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:256-483` (GDN packed decode + launcher)
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:74-85, 853-887, 1582-1700` (path selection, AITER gate, decode glue)
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py:763-1280` (conv update kernel, `BLOCK_N=256`)
- `vllm/models/qwen4_exp/amd/ops/qsa.py:20-115` (mqa scoring), `:117-190` (expand), `:191-355` (split-K attention), `:357-398` (merge), `:401-583` (store/compress), `:846-921` (launch heuristics, BLOCK_M clamp)
- `vllm/models/qwen4_exp/amd/indexer_qsa.py:20-100, 170-290` (rope/norm helpers, side-cache chain)
- `vllm/models/qwen4_exp/amd/qsa.py:117-165, 328-381` (attention impl, kv-cache update)
- `vllm/models/qwen4_exp/amd/ops/hc.py:13-80` (grouped Gemma RMSNorm)
- `vllm/model_executor/layers/rotary_embedding/mrope.py:16-225` (mrope kernel; ROCm single-wave path at :200)
- `vllm/model_executor/layers/fused_qk_norm_rope.py:16` (Triton fused qk-norm+rope+gate, CUDA-gated at `vllm/model_executor/models/qwen3_next.py:376-382`)
- `vllm/v1/attention/ops/triton_unified_attention.py:944-1000` (BLOCK_M=16 GQA packing; gfx908 prefill tune gate)
- `vllm/model_executor/models/config.py:806-816` (mamba_ssm_dtype → fp32 state)
- `docker/Dockerfile.mi100_base:2-5` (Triton `release/internal/3.7.x` f0b55c0, torch 2.12)

AITER (`/home/tyler/aiter/aiter/ops/triton/`):
- `_triton_kernels/gated_delta_rule/decode/fused_rearrange_sigmoid_gdr.py`, `gated_delta_net/fused_rearrange_sigmoid_gdr.py:60-64` (BV=32, 4 warps)
- `_triton_kernels/gated_delta_rule/decode/fused_sigmoid_gating_recurrent.py:16-24` (HIP config BV=64/4 warps, CUDA BV=8/1 warp)
- `conv/causal_conv1d_update_single_token.py` (fused reshape + conv update, `BLOCK_N=256`)
- `attention/pa_decode.py:183-253` (GQA padded to 16 rows for `tl.dot`)
- `rope/fused_qkv_split_qk_norm_rope_cache.py` (split + norm + rope + paged cache write in one kernel)
- gfx908 config JSONs present: only `configs/gfx908-{EXTEND_ATTENTION,GMM,LEANATTN-DEFAULT,MHA-DEFAULT,MLA_DECODE_ROPE-DEFAULT}.json` — nothing for GDN, pa_decode, or rope.

Triton compiler (release/3.4.x and main, github.com/triton-lang/triton):
- `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` — `chooseMfmaInstruction` (3.4: M/N<16 → FMA; main/3.7: `minSize >= 4` → 4x64/64x4), `AccelerateBlocked::tryLegalizeFMA` (bf16 → f32 upcast on FMA path), `kWidth`/`kPack` handling
- `third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp` — version-1 intrinsic table
- `third_party/amd/backend/compiler.py` — `HIPOptions`, pass pipeline, pingpong/in-thread-transpose arch gates
- `python/triton/knobs.py` — `amd_knobs` env names
- Upstream issues touching this area (none for gfx908 dot correctness): #9628 (gfx906 target, merged Mar 2026), #9175 (ROCm scalar-load → WMMA crash), #5306 (FMA path matmul failure), #9830 (fp16 dot wrong on sm86). File the 4x64-on-CDNA1 repro from §3.3.
- flash-linear-attention `fla/ops/gated_delta_rule/fused_recurrent.py` — `BV = min(8, next_pow2(V))`, `num_warps=1`, `num_stages=3`

Local context:
- `docs/mi100_decode_opt/qwen38_flash_next_gfx908.md` (BLOCK_M fix `004b1ca779`, fused-finalize loss, profiler inflation)
- `docs/mi100_decode_opt/research/_profile_c1_launch_map.txt`
- `~/.claude/projects/-home-tyler-aiter/memory/reference_gfx908_isa.md` (DPP row ops, MFMA inventory, 64 KB LDS / 32 banks)
