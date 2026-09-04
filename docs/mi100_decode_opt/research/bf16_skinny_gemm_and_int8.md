# bf16 skinny GEMMs on gfx908: shapes, int8 weight streaming, mid-M MFMA, fusion

Read-only research note, 2026-09-03. Target: the `wvSplitK_hf_sml_` bucket in the c=1 profile
(`_profile_c1_launch_map.txt`: 365 launches, 4.34 ms/token profiled, ~3.5 ms real — the
largest single GPU cost of the 16.7 ms step) and the Triton split-K bucket at 5 <= M <= 64.
Everything below is derived from the code paths listed in section 6 and from bytes / bandwidth;
nothing was run on the GPUs.

Hardware constants used: 120 CUs, 1.2 TB/s HBM2 peak (wvSplitK measured 1.10-1.17 TB/s on the
big shapes, so 1.1 TB/s is the practical ceiling), 64 KiB LDS, wave64, 1.5 GHz. MFMA peaks
(MI100 datasheet / AMD matrix calculator): fp16 184.6 TFLOPS, **bf16 92.3 TFLOPS** (gfx908 bf16
MFMA has half the K of fp16: `v_mfma_f32_32x32x4bf16`, `16x16x8bf16`, no `_1k` variants),
int8 184.6 TOPS (`v_mfma_i32_32x32x8i8`, `16x16x16i8`), fp32 matrix 46.1. VALU: ~11.5 T
lane-ops/s. Dot instructions present on gfx908: `v_dot2c_f32_f16`, `v_dot2_f32_f16`,
`v_dot4_i32_i8`, `v_dot4c_i32_i8`, `v_dot8_i32_i4`, `v_dot2_i32_i16`. **No bf16 dot**
(`v_dot2_f32_bf16` is gfx11+/gfx950), which is why wvSplitK's bf16 path is cvt+fma in fp32.

---

## 1. Exact bf16 GEMM shapes per rank (TP4) and bytes per token

Config (`/mnt/slow-storage/quant/Qwen3.8-Flash-Next-GPTQ-4bit/config.json`): hidden 2560,
hc_count 4, hc_lowrank 320, 36 GDN layers (`linear_num_key_heads 16 x 128`,
`linear_num_value_heads 48 x 128`), 12 QSA layers (indexer 4+1 heads x 128), 512 experts,
vocab 248320, `quantization_config.lm_head: false`. Everything not in the GPTQ module list
(GDN projections, HC, router, indexer, PLE projections, lm_head) is served bf16.

Sharding (verified in code):
- `in_proj_qkvz`: `MergedColumnParallelLinear([key_dim, key_dim, value_dim, value_dim])`
  = [2048, 2048, 6144, 6144] -> per rank 512+512+1536+1536 = **4096 x 2560**
  (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:550-575`).
- `in_proj_ba`: `MergedColumnParallelLinear([48, 48])`, `disable_tp` only on CUDA+Marlin, so on
  ROCm it is sharded: **24 x 2560** (`:577-616`).
- `out_proj`: `RowParallelLinear(value_dim=6144 -> 2560)` -> **2560 x 1536** (`:487`).
- HC (`vllm/models/qwen4_exp/amd/hyperconnection.py:88-118`): `input_mix_weight_down_block_inject`
  = `MergedColumnParallelLinear(10240, [320, 4, pad 12], disable_tp=True)` -> **336 x 10240**,
  `input_mix_weight_up` = `ReplicatedLinear(320 -> 10240)` -> **10240 x 320**. Two HC blocks
  per layer (`attn_hyper_connection`, `mlp_hyper_connection`, `amd/model.py:289-296`) plus the
  final `hyper_connection_mixer` (down only, 320 x 10240, `amd/model.py:465`). **Replicated on
  all four ranks** — every rank streams the full HC weights.
- Router `gate`: `ReplicatedLinear(2560 -> 512)` (`qwen3_next.py:170`) -> **512 x 2560**.
  `shared_expert_gate` (2560 -> 1) is already fused into `_reduce_gate_kernel`.
- QSA indexer `index_qk_proj`: `ReplicatedLinear(2560 -> (4+1)*128)` -> **640 x 2560**
  (`amd/indexer_qsa.py:122`). QSA `qkv_proj`/`o_proj` are W4 (not in this bucket).
- PLE (layer 2 only): `key_proj` 2560 -> 10240, `value_proj` 2560 -> 2560, replicated
  (`amd/ple_layer.py:702-714`).
- `lm_head`: `ParallelLMHead(248320, 2560)` -> **62080 x 2560** per rank, bf16, dispatched to
  wvSplitK (m > 8, n == 1) — the 338 us launch noted in `utils.py:641`.

| GEMM (per rank) | launches/token | N x K | MB/launch | MB/token | ideal us @1.1 TB/s | measured us (M=1) |
|---|---|---|---|---|---|---|
| GDN in_proj_qkvz | 36 | 4096 x 2560 | 20.97 | 755 | 19.1 | 19 |
| GDN in_proj_ba | 36 | 24 x 2560 | 0.12 | 4.4 | 0.1 (floor ~3) | ~3-4 |
| GDN out_proj | 36 | 2560 x 1536 | 7.86 | 283 | 7.1 | 6.7 |
| HC mix_down(+inject) | 97 | 336 x 10240 | 6.88 | 667 | 6.3 | 7.2 (fused silu) |
| HC mix_up | 96 | 10240 x 320 | 6.55 | 629 | 6.0 | 8.3 (fused gate-mix, YTILE=4) |
| MoE router | 48 | 512 x 2560 | 2.62 | 126 | 2.4 | 3.9 |
| QSA indexer qk | 12 | 640 x 2560 | 3.28 | 39 | 3.0 | ~4 |
| PLE key/value proj | 1+1 | 10240 x 2560, 2560 x 2560 | 52.4 + 13.1 | 65 | 60 | — |
| lm_head | 1 | 62080 x 2560 | 317.9 | 318 | 289 | 338 |
| **total** | **364** | | | **2,887 MB** | **2.62 ms** | **~3.5 ms real / 4.34 profiled** |

Count check: 48x4 HC + 36x3 GDN + 48 router + 12 indexer + 1 final mixer + 2 PLE + 1 lm_head = 364
(profile: 365). Effective streaming rate today is 2.89 GB / 3.5 ms = 0.83 TB/s, i.e. ~0.9 ms of
the 3.5 ms is per-launch floor (365 x ~2.5 us) and ~2.6 ms is bytes. **HC (1.30 GB) is the
largest bucket, not GDN (1.04 GB)**, and it is replicated x4 across ranks — sharding it would
need an extra all-gather per HC (97/token x ~8 us), which exceeds the saving, so bytes per weight
is the only HC lever.

Measured numbers: `utils.py:637-641` (mb_skinny.py graph-timed), `gfx908_hc_fused.py:16-18`,
campaign log `qwen38_flash_next_gfx908.md`.

At 5 <= M <= 64 (Triton `gfx908_midm_gemm.py` header + campaign log): mix_down 16-34 us
(6.9 MB -> 0.2-0.4 TB/s), router 7-15 us, indexer 8-17 us, in_proj_qkvz 33-37 us (0.6 TB/s);
`in_proj_ba` (N=24) is excluded from the Triton path and goes to rocBLAS. So the mid-M path
sits at 20-55% of HBM BW — there is a 2-3x kernel-side gap there, unlike M=1 where only bytes help.

---

## 2. Ranked designs

Savings are per token per rank at c=1 unless noted; step is 16.7 ms (59.6 tok/s).

### D1 — W8A16 per-channel int8 weights + `wvSplitK_w8` HIP GEMV (M <= 4)  [highest value, low risk]

Quantize the Tier-1 set (section 3): in_proj_qkvz, out_proj, HC down/up, PLE projections
(2,400 MB -> 1,200 MB) and optionally lm_head (318 -> 159 MB). Per-channel symmetric RTN
(`s[n] = absmax(W[n,:]) / 127`) computed **at load time** on gfx908 — no artifact change, no
calibration, seconds of work. Keep bf16 for in_proj_ba, router, indexer (section 3).

Kernel: a copy of `wvSplitK_hf_sml_` (`csrc/rocm/skinny_gemms.cu:350-570`; the fused-epilogue
copy in `amd/csrc/gfx908_wv_fused.hip` is the right starting point since it already carries the
HC epilogues) with the weight stream changed to int8. The row-contiguous `[N, K]` layout stays:
one wave streams YTILE rows, each lane loads 16 B = **16 k** (A_CHUNK 8 -> 16), 64 lanes = 1 KB
contiguous per row per step (full 64 B lines, `buffer/global_load_dwordx4`, nontemporal).

Inner loop sketch (per lane, per UNRL step, batch N <= 4, YTILE rows):

```
// LDS: x as fp32 when N*K*4 <= 64 KB (qkvz/out_proj at N<=4, HC down at N=1),
//      else bf16 (shift-to-f32 costs 1 op/elem).
uint4 w = loadnt((uint4*)&Wq[row*K + k0 + lane*16]);      // 16 int8 weights
float4 x0..x3 = *(float4*)&s_x[n*Kap + k0 + lane*16 + ...]; // 16 activations (LDS)
// per dword: 4 x (v_cvt_f32_i32 with SDWA sext BYTE_i) + 4 x v_fma_f32
#pragma unroll for i in 0..3:
    sum[n][y] = fma(cvt_f32_sext_byte0(w.x), x0.x, sum[n][y]); ...
// end of K loop: DPP row_shr/ROW_BCAST reduction as today, then
// C[row] = bf16(sum * scale[row])  (+ HC epilogues unchanged)
```

Cost: 2 VALU ops per weight element at N=1 (cvt + fma; SDWA byte-select makes the extract free),
3 with bf16 LDS. At 1.1 TB/s of int8 = 1.1e12 elem/s -> 2.2-3.3 T ops/s = 20-30% of VALU peak:
stays BW-bound up to N=4 (4 fma per element -> 5.5 T = 48%). Per-group scales (128 k) are
nearly free here: each lane's 16 k lie inside one group, so multiply the lane's 16-element
partial by `s[row][g]` before accumulating (1 extra fma per 16 elements).

Alternative dot path: `v_dot4_i32_i8` needs int8 activations (llama.cpp Q8_0 x Q8_1, block-32
scales); at M=1 the kernel is already BW-bound with the fp32 path, so the activation quant buys
nothing — reserve it for D6.

Expected: bytes halve on 2.4 GB -> **-1.05 to -1.1 ms/token** (qkvz 19 -> ~10 us, out_proj 6.7 ->
~3.8, HC down 7.2 -> ~4.3, HC up 8.3 -> ~5); lm_head 338 -> ~175 us adds **-0.15 ms**. Total
**~-1.2 ms = +7-8% c=1 (59.6 -> ~64 tok/s)**. VRAM: -1.2 GB/rank (more KV blocks).

### D2 — hand-written W8 MFMA kernel for 5 <= M <= 64 with pre-swizzled weights  [highest value at c>=8]

Replaces `_gfx908_midm_splitk_kernel` (Triton, 20-55% BW) for the Tier-1 shapes. Why a hand
kernel beats Triton here: Triton stages B through LDS and re-reads it in MFMA layout; with a
pre-swizzled weight the MFMA B fragment is loaded **directly into VGPRs** with one 16 B load per
lane, no LDS, no shuffles, full-line coalescing.

Layout (int8, 32-row groups; this is AITER's `shuffle_weight_gfx1250` pattern with 32-row tiles,
`aiter/ops/shuffle.py:68-89`):
```
W_sw = W_int8.view(N//32, 32, K//32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()
      # [N/32][K/32][khalf][n%32][16 k]  -> one (32 n x 32 k) tile = 1 KB contiguous
lane l loads 16 B at tile_base + 16*l : row n = l%32, k = 16*(l/32) .. +16
```
MFMA: `v_mfma_f32_32x32x4bf16` (B = weights as the 32-wide operand, A = activations padded to
32 rows) — 8 MFMAs consume one tile (MFMA j uses the lane's k-pair j: lanes 0-31 give k=2j,2j+1,
lanes 32-63 give k=16+2j,+1; K is consumed in a permuted order, which is legal as long as A uses
the same lane rule: lane l loads A[m=l%32][k0+16*(l/32) .. +16] = one 32 B load from the L2-resident
activation). For M <= 16 use `v_mfma_f32_16x16x8bf16` with 16-row tiles (`view(N//16,16,K//32,2,16)`
= AITER's exact gfx1250 layout) to halve the padding waste.

Dequant: int8 -> f32 (`v_cvt_f32_i32` SDWA) -> bf16 by taking the high 16 bits: **exact**, since
|v| <= 127 fits in bf16's 8-bit mantissa; pack pairs with `v_perm_b32`/SDWA `dst_sel:WORD_1`
(~2 VALU per element). Per-channel scale applied once to the f32 accumulator. Per-group-128
scales: keep the group's MFMA output in a temp accumulator and `acc += s_g * tmp` every 4 tiles
(16 f32 fma per lane per 128 k — cheap).

MFMA budget (bf16 92.3 TFLOPS): at int8 1.1e12 elem/s the required rate is 2*M_pad*1.1e12 =
35 TFLOPS at M_pad=16 (38%), 70 at 32 (76%), MFMA-bound above ~M=40. That is exactly the range
where rocBLAS/Triton already tie, so the kernel only needs M_pad in {16, 32}; hand M > 64 to
the existing path (or to CK a8w8 if D6 lands).

Parallelism: one WG = 16 waves on the same 32-row group, each wave a K-slice (split-K inside the
WG, LDS reduce, single bf16 store — no atomics, no second launch, no f32 partial traffic). qkvz:
128 WGs (~1 per CU), each wave 160 k = 5 tiles in flight; HC up (N=10240, K=320): 320 WGs with
10 active waves. 2-launch f32 reduce only for K=10240 x N=336 (mix_down: 11 WGs -> add a
grid-level split-K of ~10 with f32 atomics into a zeroed buffer + cast, or accept intra-WG only).

GEMV mode of the same kernel (M <= 4): lane l accumulates row l%32 over its 16 k per tile with
fp32 fma; final reduce is lane l + lane l+32 via LDS. Same 1 KB contiguous stream as wvSplitK,
so it should match D1's BW; if it does, the row layout of D1 can be dropped and one layout serves
all M (phase B in section 5).

Expected at M=16 (75% of 1.1 TB/s, int8): mix_down 3.4 MB -> ~6 us (vs 20-34), qkvz 10.5 MB ->
~13 us (vs 33-37), out_proj ~6 (vs ~12), HC up ~7 (vs ~15), router (bf16 kept) via D4 ~4 (vs
7-15). Per step at c=16: **~-3 to -3.5 ms** (HC 96 x 14 + 96 x 8, qkvz 36 x 22, out_proj 36 x 6,
router 48 x 6) on a ~40-60 ms step = **+6-9% throughput at c=16-48**, on top of D1's byte saving.

Interim (1-2 days, no HIP): AITER's `gemm_a16w8_blockscale` Triton kernel
(`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w8_blockscale.py:155-185`: `b.to(bf16)` +
`tl.dot` + per-block scale, split-K reduce) runs on gfx908 with a gfx908 config JSON (none exists
yet; the a8w8 blockscale ones from commit `9d07ca12a` show the tuner works on gfx908). Expect the
same 20-55% BW as the current Triton kernel but on half the bytes — a cheap 1.5-2x while D2 is
written. It takes the plain `[N, K]` int8 layout, so it shares D1's weights.

### D3 — fuse in_proj_qkvz + in_proj_ba into one launch (weight concat)  [free, small]

Same input x; ba is 0.12 MB and pays a full launch floor at M=1 and a rocBLAS N=24 kernel
(10-20 us) at 5 <= M <= 64. Build one `MergedColumnParallelLinear([2048, 2048, 6144, 6144, 48, 48])`
(TP shards each piece consistently -> per-rank 4120 x 2560) and split the output view; the GDN
forward already takes `mixed_qkvz` and `mixed_ba` separately (`fix_query_key_value_ordering`).
No kernel work. Saves 36 launches: **~-0.13 ms** at c=1, **~-0.5 ms/step** at c=16 (rocBLAS ba
gone). Note N=4120 breaks `M % YTILE == 0` for YTILE=3/4 in `WVSPLITK_CFG`; pad to 4128
(multiple of 32 also suits D2).

Other candidates checked and rejected: router + shared-expert gate_up share x but are bf16 vs W4
kernels (two code paths; the shared-expert gate itself is already fused). HC down -> silu -> up is a
true dependency chain (K=320 of `up` is `down`'s output) — a single-kernel version needs a
grid-wide barrier (~3-5 us on MI100 via L2 atomics) which cancels the ~2.5 us launch gap; a
generic "multi-descriptor wvSplitK" has no independent pairs left to batch in this model. 2.5 us
x 365 launches = the ~0.9 ms floor is therefore mostly structural (7 dependent GEMVs per layer).

### D4 — intra-WG split-K for the small-N bf16 shapes (router, indexer, mix_down)  [cheap, small]

`WVSPLIT_TILE_CFG` gives router 512 rows / YTILE 2 = 256 waves = 16 WGs -> 16 of 120 CUs, ~1 MB
in flight -> 0.67 TB/s (3.9 us vs 2.4 ideal). mix_down 336 rows -> 11 CUs (still 0.95 TB/s
thanks to K=10240 rows keeping 64 KB/CU in flight). Fix = D2's wave-level split-K inside the WG
(16 waves x K-slices of the same YTILE rows, LDS reduce, single store) applied to the bf16
wvSplitK clone. vLLM's `wvSplitKrc` does this with atomics but is `#if defined(__gfx950__)` only
(`skinny_gemms.cu:1330`) and gated `on_gfx950()` in `utils.py:320`. Expected: router 3.9 -> ~2.8,
indexer ~4 -> ~3, mix_down 7.2 -> ~6.3: **~-0.16 ms/token**. Do after D1 (int8 makes these shapes
floor-bound anyway).

### D5 — layout audit of wvSplitK (no change recommended for the GEMV path)

The current stream is already the optimal GEMV pattern: per wave YTILE rows x 1 KB contiguous per
step (64 lanes x dwordx4), UNRL 2-4 steps in flight, nontemporal loads, activations from LDS,
DPP reduction (`row_shr` + `ROW_BCAST15/31`, `skinny_gemms.cu:476-495`). Measured 92-98% of peak
on qkvz/out_proj confirms it. The bf16 `DOT2C` is 2 cvt (shift/and) + 2 fma per pair (no bf16 dot
on gfx908) — irrelevant at M <= 4. The only layout change that pays is D2's MFMA-native swizzle
(needed to make M >= 5 MFMA loads LDS-free), and since it can also run a GEMV mode it can replace
the row layout entirely.

### D6 — W8A8 with int8 MFMA (later, accuracy-gated)

For M >= 32 the bf16 MFMA rate (92 TFLOPS) becomes the D2 bottleneck; `v_mfma_i32_32x32x8i8`
(184.6 TOPS) with per-token dynamic activation scales removes it and also enables
`v_dot4_i32_i8` in the GEMV (0.3 ops/elem). The `mi100-a8w8-2026-08` branch already has CK int8
W8A8 building and tuned on gfx908 (`aiter/configs/a8w8_tuned_gemm.csv`, 299 shapes, 1.37x
geo-mean, commit `6200ca919`; per-token quant in `aiter/ops/quant.py:407/755`). Not for GDN inputs
without SmoothQuant-style calibration (section 3); reasonable for out_proj / lm_head at large M.
Value at c<=16: none (BW-bound with D1/D2). Park.

### Rejected

- TP-sharding HC weights (extra all-gather per HC > saving).
- exllama `gemm_half_q_half_gptq_8bit_kernel` for the int8 weights: fp16-only (this model must
  run bf16 for QSA), k-major `[K/4, N]` packing with per-thread 4 columns, `__int2half_rn` per
  element, 8 scalar `ds_read_u16` of `a` per 8 weights and a **half** running accumulator
  (`q_gemm.cu:137-150`, `qdq_8.cuh:14-24`) — ~10 VALU/LDS ops per weight element -> VALU/LDS-bound
  at ~1/3 of HBM BW, which is the "scalar" behaviour seen on the 27B project. Nothing to salvage.
- llama.cpp `mmvq` Q8_0 (`ggml-cuda/mmvq.cu`; `ggml_cuda_dp4a` -> `__builtin_amdgcn_sdot4` on
  CDNA, `common.cuh`): a good W8A8-block-32 GEMV reference (1-2 warps per row block on GCN, ncols_dst
  <= 8), but it quantizes activations to Q8_1 per 32-block and is HIP-generic; D1 reaches the same
  BW without touching activations.
- rocBLAS/hipBLASLt for mid-M: hipBLASLt has no real gfx908 logic (support was gfx90a+ until
  recently; ROCm 7.0 falls back to Tensile), and the Tensile MI100 library selects fixed macro
  tiles (128x128 / 64x64, MFMA 32x32x8) with no split-K/stream-K solutions for N <= 1024, so a
  M=16, N=512 GEMM runs on 4-8 CUs (the measured 38-112 GB/s) and N=1 hits a 508 us pathological
  pick. Stream-K exists only in hipBLASLt's Origami path (`TENSILE_SOLUTION_SELECTION_METHOD=2`),
  which is not available for gfx908. AITER's alternatives: CK (does not compile on gfx908,
  `v_pk_mul_f32`), FlyDSL `small_m_hgemm.py` (gfx1250 WMMA, `MAX_LDS_BYTES 163840` — MI300+),
  `custom_kernels.cu` wvSplitK (same lineage as vLLM's, gated `__gfx90a__/__gfx942__`). Nothing
  ports; D2 is the path.

---

## 3. Accuracy risk of int8 per projection

General evidence that **weight-only int8 per-channel RTN is near-lossless**: LLM.int8() shows 8-bit
vector-wise weights lossless to 175B (Dettmers et al. 2022, arXiv 2208.07339); llama.cpp Q8_0
(block-32 int8, no calibration) measures +0.02 PPL on Llama-3-8B and KL ~1e-3 vs fp16
([unified llama.cpp quant evaluation, arXiv 2601.14277](https://arxiv.org/html/2601.14277v1),
[dev.to summary](https://dev.to/kunal_d6a8fea2309e1571ee7/llm-quantization-levels-compared-q4km-vs-q80-vs-fp16-2026-3kg2));
vLLM/llm-compressor W8A8 docs put the *combined* int8 weight+activation loss at <1%
([int8_w8a8](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int8_w8a8/)). This
artifact already tolerates W4-GS32 on the body with PPL 3.1206 -> 3.1386.

Recurrent / linear-attention specifics: Mamba-PTQ ([arXiv 2407.12397](https://arxiv.org/pdf/2407.12397))
and Quamba ([arXiv 2410.13229](https://arxiv.org/html/2410.13229v1)) locate the quantization
difficulty of SSMs in **activation outlier channels at the SSM input** (needing percentile clipping /
Hadamard for A8), while W8 on in_proj/out_proj is benign; Quamba reports W8A8 Mamba2-2.7B with
negligible loss once the SSM input is handled. A 2026 GDN quantization study reports INT8 matching
INT16 on the delta-rule inversion path ([arXiv 2606.06034](https://arxiv.org/html/2606.06034)).
Community W8A16 Qwen3.8 checkpoints quantize `in_proj_qkv/z/b/a` and `out_proj`
([lued/Qwen3.8-27B-INT8-W8A16-MTP](https://huggingface.co/lued/Qwen3.8-27B-INT8-W8A16-MTP)). None
of these are this 180B model; the gate is the parity plan in section 5.

| Projection | MB/token | int8 gain | Risk | Verdict |
|---|---|---|---|---|
| in_proj_qkvz | 755 | 0.35 ms | low-medium: q/k errors enter `S += beta k^T (v - S k)`; bounded by the decay gate but can drift over long context -> test at 32K, not just 2K windows | **Tier 1** (per-channel; fall back to GS128 if 32K PPL moves) |
| in_proj_ba | 4.4 | ~0 | medium-high: `a` -> `g = -exp(A_log) softplus(a + dt_bias)` sets the per-step state decay; relative error compounds multiplicatively across steps; `b` -> beta. Zero bytes to gain | **keep bf16** |
| out_proj | 283 | 0.13 ms | low: ordinary output projection with fp32-accumulated recurrent output | **Tier 1** |
| HC mix_down / mix_up | 1,296 | 0.6 ms | medium: outputs are sigmoid gates + injection logits applied to 4 residual streams in 97 places; error compounds with depth, but sigmoid bounds it. Largest gain | **Tier 1, GS128 scales** (0.8% extra bytes) if per-channel fails the gate |
| MoE router | 126 | 0.06 ms | medium-high: top-10 selection flips on near-tie logits; MoE recipes keep routers fp32/bf16 (llm-compressor `ignore: gate`, DeepSeek-V3 FP8 keeps router fp32) | **keep bf16** |
| QSA indexer qk | 39 | 0.02 ms | medium: top-2048 key selection at the budget boundary | **keep bf16** |
| PLE key/value | 65 | 0.03 ms | low | Tier 1 (batch with HC) |
| lm_head | 318 | 0.15 ms | low-medium: direct logit error ~0.4% relative; widely done (Q8_0 `output.weight`) but gate on logprob parity | **Tier 2**, separate A/B |

Implementation note: per-channel symmetric RTN at load time; keep an env switch per tier
(`VLLM_GFX908_W8_TIERS=gdn,hc,lmhead`) so each can be A/B'd with the same image.

---

## 4. Expected savings summary

| Lever | c=1 ms/token | c=16 ms/step | effort |
|---|---|---|---|
| D1 int8 Tier 1 (GEMV M<=4) | -1.05 | (needs D2/interim for M>4) | HIP kernel clone (~2 days) + load-time quant |
| D1 + lm_head | -1.2 | | +A/B |
| D2 hand MFMA W8 (M 5-64) | 0 | -3 to -3.5 | HIP kernel (~1 week incl. layout + tests) |
| D2 interim (AITER Triton a16w8) | 0 | -1.5 to -2 | 1-2 days |
| D3 qkvz+ba concat | -0.13 | -0.5 | Python (hours) |
| D4 intra-WG split-K, small-N bf16 | -0.16 | -0.3 | small HIP change |
| all | **~-1.5 (+9-10% c=1)** | **~-4 (+8-10% at c=16-48)** | |

The remaining ~0.9 ms of launch floor is 7 dependency-chained GEMVs per layer and is not
addressable by GEMM work; it belongs to the glue-fusion / graph-launch campaign.

---

## 5. Microbenchmark and parity plan

Microbench (graph-captured, same harness style as `mb_skinny.py` / `mb_midm.py` from the campaign;
profiler numbers are inflated ~2.4 us on gfx908, so time by cudagraph replay of 100 launches):
- Shapes: the 9 rows of the table in section 1, M in {1, 2, 3, 4, 8, 16, 32, 48, 64}.
- Arms: wvSplitK bf16 (baseline M<=4), Triton midm bf16 (baseline 5-64), rocBLAS (in_proj_ba,
  M>64), D1 GEMV int8, AITER Triton a16w8 int8, D2 MFMA int8 (row and swizzled layouts), D4.
- Report per arm: us, GB/s actual, GB/s bf16-equivalent, % of 1.2 TB/s, and the launch floor
  (M=1 of a 64 KB weight) to separate bytes from floor.
- Acceptance to proceed: D1 >= 0.9 TB/s actual on qkvz/out_proj/HC; D2 >= 0.7 TB/s at M=16.

Kernel unit tests (`op_tests`-style, single GPU):
- Exactness test: weights = random int8 x power-of-two scales -> kernel output must equal the fp32
  reference to bf16 rounding (catches layout / permutation bugs, since dequant is exact).
- Random test vs `(Wq.float() * s) @ x.float()`: max |err| <= 2^-7 * max|y| + 1e-3, all M, all
  9 shapes, N <= 4 batch for the GEMV, non-multiple-of-tile N (336, 24) and K (320).
- Fused-epilogue tests for HC (silu / gate-mix) against the stock chain, bit-exact as today.
- Cudagraph capture + replay with dummy routing inputs (the W4 GEMV lesson).

Model parity (standing rule: parity before perf; each tier separately, same image, env-gated):
1. Per-layer logprob vs transformers on the 4-layer rehearsal artifact, TP1 and TP4 (target
   <= 0.02 nats, the level the QSA fix reached).
2. wikitext-2 PPL (64 win, seq 2048, stride 512): baseline 3.1362; accept <= 3.15 for Tier 1,
   <= 3.16 with lm_head. Add a 32K-context PPL run to catch recurrent drift from qkvz.
3. Greedy per-token logprob fingerprints, 16 GSM8K prompts x 160 tokens, c=1 and c=16: mean
   delta <= 0.01 nats vs the bf16 stack (noise floor 0.004-0.007).
4. GSM8K 500Q thinking mode (coarse gate, +-2% run-to-run): >= 96%.
5. In-server: >= 3 c=1 probes per lever before commit (fusion rule), then the full 12-tier
   BenchAndReport on every arm.

Order: D3 (free) -> D1 Tier 1 + unit tests + parity -> D2 interim Triton -> D2 hand kernel ->
lm_head A/B -> D4.

---

## 6. References (local paths and external)

Local:
- `/home/tyler/vllm-gfx908/csrc/rocm/skinny_gemms.cu` — `wvSplitK_hf_sml_` 350-570 (LDS activation
  prefetch, YTILE/UNRL stream, DPP reduce), dispatch macros 1214-1330 (`WVSPLITK_CFG`,
  `WVSPLIT_TILE_CFG`, `sYT`), `wvSplitKrc_` 1330+ (gfx950-only), `LLGemm1_kernel` 175.
- `/home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py` — `rocm_unquantized_gemm_gfx908_impl`
  606-708 (wvSplitK for m>8 & n<=4, LLMM1, Triton midm 5-64, einsum m<=8, rocBLAS), wvSplitKrc gating 318-335.
- `/home/tyler/vllm-gfx908/vllm/model_executor/layers/gfx908_midm_gemm.py` — Triton split-K + reduce, `_config`.
- `/home/tyler/vllm-gfx908/vllm/models/qwen4_exp/amd/{model.py,hyperconnection.py,gfx908_hc_fused.py,indexer_qsa.py,ple_layer.py}`,
  `amd/csrc/gfx908_wv_fused.hip` (fused-epilogue wvSplitK clone — D1 base).
- `/home/tyler/vllm-gfx908/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — GDN sharding.
- `/home/tyler/vllm-gfx908/csrc/libtorch_stable/quantization/gptq/{q_gemm.cu,qdq_8.cuh,qdq_util.cuh}` — exllama 8-bit.
- `/home/tyler/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w8_blockscale.py`,
  `/home/tyler/aiter/aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py` — W8A16 Triton (D2 interim).
- `/home/tyler/aiter/aiter/ops/shuffle.py` — `shuffle_weight_gfx1250` (D2 layout pattern), `shuffle_weight` (16x16 MFMA tiles).
- `/home/tyler/aiter/csrc/ck_gemm_a8w8/` + `aiter/configs/a8w8_tuned_gemm.csv` (gfx908 rows, commit `6200ca919`),
  `aiter/ops/triton/configs/gfx908-GEMM-A8W8_BLOCKSCALE-*.json` (commit `9d07ca12a`), `aiter/ops/quant.py` — D6.
- `/home/tyler/aiter/csrc/kernels/custom_kernels.cu` — AITER wvSplitK/LLGemm1 copies (gfx90a/942 gated).
- `/home/tyler/aiter/aiter/ops/flydsl/kernels/small_m_hgemm.py` — FlyDSL small-M (gfx1250/MI300, not portable).
- `/home/tyler/.claude/projects/-home-tyler-aiter/memory/reference_gfx908_isa.md` — ISA inventory.
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/qwen38_flash_next_gfx908.md`, `research/_profile_c1_launch_map.txt`.

External:
- llama.cpp `ggml/src/ggml-cuda/{common.cuh,mmvq.cu}` (`ggml_cuda_dp4a`: CDNA -> `__builtin_amdgcn_sdot4`).
- AMD Instinct MI100 CDNA1 ISA reference (VOP3P MFMA/dot opcodes); ROCm/amd_matrix_instruction_calculator
  (`--architecture cdna1 --register-layout` for the exact A/B/C lane maps needed by D2).
- hipBLASLt Stream-K/Origami docs (`TENSILE_SOLUTION_SELECTION_METHOD`), hipBLASLt issue #1299 (gfx908 support).
- Dettmers et al., LLM.int8() (2208.07339); Xiao et al., SmoothQuant (2211.10438); Mamba-PTQ (2407.12397);
  Quamba (2410.13229) / Quamba-SE (2601.09451); GDN INT8 inversion study (2606.06034);
  llama.cpp quantization evaluation (2601.14277).
