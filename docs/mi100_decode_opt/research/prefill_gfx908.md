# Prefill on 4x MI100 (gfx908): Qwen3.8-Flash-Next (qwen4_exp 180B), TP4

Read-only research note, 2026-09-04. Nothing here was run on a GPU; every number is derived from
the code in this tree, the model config, and the hardware constants below. The only measured
anchors are **2048-token prompt c=1: TTFT 850 ms** and **16K prompt c=4: TTFT 12.1 s** (65,536
prompt tokens => ~5.4k tok/s aggregate). Prefill has not moved through rounds 1-6.

Hardware constants (MI100, 1502 MHz, 120 CU, wave64):
fp16 MFMA **184.6 TFLOP/s**, **bf16 MFMA 92.3 TFLOP/s** (gfx908 bf16 MFMA has half the K of
fp16: `v_mfma_f32_32x32x4bf16` / `16x16x8bf16`, no `_1k`), int8 MFMA 184.6 TOPS, fp32 matrix
46.1, **fp32 VALU 23.1 TFLOP/s**, HBM2 1.2 TB/s peak (~1.1 practical), L2 8 MB, LDS 64 KB,
XGMI ~46 GB/s/direction/link, 3 links per GPU, 1-hop between every pair. Profiler caveat
(round 4): +2.4 us per kernel - irrelevant for the >30 us prefill kernels, but not for the ~200
sub-10 us glue launches per pass.

Model shape (`config.json`): 48 layers = 36 GDN + 12 QSA (interval 4); hidden 2560; `hc_count 4`
(hyper state 10240), `hc_lowrank 320`; 512 experts top-10, `moe_intermediate_size 640` = shared;
QSA 24 Q / 2 KV heads x 256, `indexer_budget 2048`, `compress_ratio 4`, indexer 4+1 heads x 128;
**`ple_layer_ids [2]` - PLE runs on exactly one layer**; vocab 248,320. Per rank at TP4: GDN 4
K-heads / 12 V-heads x 128; QSA 6 Q / 1 KV head x 256; MoE N=160.

## 0. The one-paragraph answer

At M=2048 the whole forward pass has a **~110 ms roofline** (65 ms of bf16 MFMA, 35 ms of W4
GEMM lowered through bf16 MFMA, 9 ms of fp32 VALU in the QSA indexer) against **850 ms measured
= 13% of achievable**. Prefill is not bandwidth-bound anywhere except the hyper-connection
activation stream and the MoE `ic3` round trip. Four things account for most of the gap, and
none of them were touched by rounds 1-6 (every gfx908 fast path in this tree is gated at
`M <= 8/16` and is therefore *off* during prefill - see the gate table in section 2):
1. **The QSA layers do decode-shaped work at prefill.** The sparse paged attention kernel is
   launched with one program per *token* and no query tiling, so it re-gathers 2.1 MB of K/V per
   token per layer and issues 5.5x the useful MFMA (BLOCK_M=16 for 6 real heads x a fixed
   33-tile loop over the 2051-wide index). And for **every token whose context <= 2048 the
   selection is a no-op** (it selects the whole causal prefix), so for a 2048-token prompt all
   12 indexer scoring + top-k + expand passes are pure waste.
2. **The indexer scorer is sized by `max_model_len`, not by the sequence.** It computes and
   writes `max_model_len/compress_ratio = 8192` fp32 columns per token per layer regardless of
   the real context - 16x waste at a 2048-token prompt - in a **fp32 VALU** kernel with no
   `tl.dot`.
3. **Hyper-connections are replicated on all four ranks and are the largest FLOP block in the
   model** (1.30 GFLOP/token/rank vs 1.42 for the whole MoE) plus ~12.8 MB/token of activation
   traffic. The decode-era decision not to shard them was a bytes argument; it inverts here.
4. **Every prefill all-reduce falls off the custom path onto RCCL, which is pinned to a
   decode-tuned `NCCL_ALGO=Tree, NCCL_PROTO=LL`.** 97 ARs per pass, 40 MB each at prefill.

## 1. (a) Per-layer FLOP and byte budgets, per rank, TP4

All MAC counts are per token per rank; FLOP = 2 x MAC. "Issued" includes tile padding actually
executed by the kernel; "useful" is the mathematically required work.

### 1.1 GEMM inventory per token per rank

| block | n/fwd | [K x N] per rank | kernel | MFLOP/tok | GFLOP/tok | est. eff | ms @M=8192 |
|---|---|---|---|---|---|---|---|
| HC `mix_down(+inject)` | 97 | 10240 x 336 | rocBLAS bf16, **replicated x4** | 6.88 | **0.667** | ~40% | 1.53 ea |
| HC `mix_up` | 96 | 320 x 10240 | rocBLAS bf16, **replicated x4** | 6.55 | **0.629** | ~40% | 1.45 ea |
| MoE routed top-10 | 48 | 10 x (2560x320 + 160x2560) | Triton `fused_moe_kernel_gptq_awq` W4 gs32 | 24.58 | **1.180** | ~18% | ~11 ea |
| MoE shared expert | 48 | 2560x320 + 160x2560 | `triton_w4a16_gemm` W4 | 2.46 | 0.118 | 15-20% | 1.2 ea |
| MoE router gate | 48 | 2560 x 512 | rocBLAS bf16 (replicated) | 2.62 | 0.126 | ~40% | 0.58 ea |
| GDN `in_proj_qkvz` | 36 | 2560 x 4096 | int8 -> dequant -> rocBLAS | 20.97 | **0.755** | ~55% | 3.38 (+40us) |
| GDN `in_proj_ba` | 36 | 2560 x **24** | rocBLAS bf16, **64 WGs on 120 CU** | 0.12 | 0.004 | ~10% | 0.11 ea |
| GDN `out_proj` | 36 | 1536 x 2560 | int8 -> dequant -> rocBLAS | 7.86 | 0.283 | ~52% | 1.34 (+15us) |
| QSA `qkv_proj` | 12 | 2560 x 3584 | `triton_w4a16_gemm` W4 | 18.35 | 0.220 | ~22% | **7.4 ea** |
| QSA `o_proj` | 12 | 1536 x 2560 | `triton_w4a16_gemm` W4 | 7.86 | 0.094 | ~20% | **3.5 ea** |
| QSA `index_qk_proj` | 12 | 2560 x 640 | rocBLAS bf16 (replicated) | 3.28 | 0.039 | ~40% | 0.73 ea |
| `shared_expert_gate` | 48 | 2560 x **1** | `torch.einsum` (`utils.py:734`) | 0.001 | ~0 | mem | ~0.04 ea (untested) |
| PLE `key_proj`+`value_proj` | 1 | 2560x10240 + 2560x2560 | rocBLAS bf16 (replicated) | 65.5 | 0.066 | ~45% | ~12 total |
| **GEMM subtotal** | | | | | **4.19** | | |

Efficiency figures are estimates (Tensile on CDNA1 at these aspect ratios; Triton W4 at
BLOCK_K=32 with no L2 swizzle) - the least reliable numbers here.
`qkv_proj` per rank is 3584 = (6 Q + 6 gate + 1 K + 1 V) x 256: `num_kv_heads = max(1, 2//4) = 1`,
so the 2 KV heads are replicated, not split (`qwen3_next.py:296-313`, `attn_output_gate=True`). HC shapes and the "replicated x4" fact are
from `research/bf16_skinny_gemm_and_int8.md` section 1 and
`vllm/models/qwen4_exp/amd/hyperconnection.py:88-125`.

### 1.2 Attention-like work per token per rank (issued vs useful)

| block | n | issued MFLOP/tok | useful MFLOP/tok | why the gap |
|---|---|---|---|---|
| QSA sparse paged attn | 12 | **34.60** (33 tiles x 2 dots x 16x64x256 MAC x2) | 12.6 @ctx2048, 6.3 @ctx1024 | BLOCK_M=16 for 6 real heads (2.67x) x fixed 33-tile loop |
| QSA indexer MQA scoring | 12 | **8.39 fp32 VALU** (8192 cols x 4 heads x 128) | 0.5 @ctx2048 | no `tl.dot`, fp32, width = `max_model_len/4` |
| GDN chunked delta rule | 36 | ~1.4 | ~1.4 | kkt, tril solve, WY/UT, delta_h, chunk_o at BT=64 |

### 1.3 Totals and roofline

| | M=2048 | M=8192 |
|---|---|---|
| bf16-MFMA-class FLOP/rank (HC + GDN proj + QSA attn + router + indexer proj + PLE) | 6.03 TFLOP | 24.13 TFLOP |
| W4-class FLOP/rank (MoE routed+shared, QSA qkv/o) | 3.27 TFLOP (5.2 issued w/ 1.6x MoE M-pad) | 13.08 TFLOP (20.9 issued) |
| fp32-VALU FLOP/rank (indexer scorer) | 0.21 TFLOP | 0.82 TFLOP |
| **time at 100% of the respective peaks** | **~110 ms** | **~440 ms** |
| measured / inferred | **850 ms** | ~1.4-1.5 s per 8192-chunk (inferred) |
| efficiency | **13%** | ~30% |

The 16K arm does **not** run 8192-token chunks: prefix caching is on by default and the hybrid
GDN layers force `mamba_cache_mode="align"`, inflating `cache_config.block_size` to **784** and
flooring every non-final chunk to a multiple of it (section 3). A 16,384-token prompt is
**7840 + 7840 + 704 = 3 passes**; at c=4 the WAITING loop breaks on the sub-block remainder so
the four prompts run near-serially in 9 passes (12.1 s / 9 = ~1.34 s, but passes are unequal -
treat the per-chunk figure as 1.3-1.7 s until traced).

### 1.4 Byte budgets

| stream | bytes/token/rank | GB at M=2048 | GB at M=8192 | ms at 1.2 TB/s (M=8192) |
|---|---|---|---|---|
| HC activations (96 x `combine_norm` 66.6 KB + `gate_mix` 46.1 KB + up-GEMM write 20.5 KB) | **12.78 MB** | 26.2 | 104.7 | **87** |
| QSA K/V gather, nominal (12 x 2051 x 256 x 2 B x 2) | 25.20 MB | 51.6 | 206.5 | 172 (heavily L2-mitigated below ~4k ctx) |
| MoE `ic3` write + `moe_sum` read (M x 10 x 2560 bf16 x 2) | 5.12 MB | 10.1 | 40.3 | 34 |
| MoE `ic1`/`ic2` (gate_up out + silu out) | 2.56 MB | 5.0 | 20.1 | 17 |
| QSA indexer logits write + top-k read | ~0.59 MB | 1.2 | 4.8 | 4 |
| GDN chunk kernels (94.7 KB/token/layer x 36, incl. the `h` buffer) | 3.41 MB | 7.0 | 27.9 | 23 |
| **M-independent per pass:** MoE expert weights 17.0 GB (14 ms); GDN W8A16 rematerialisation 1.50 GB (1.3 ms); HC weights 1.30 GB (1.1 ms); dense W4 ~0.6 GB | | | | |

Memory floor at M=2048 is ~60 ms, at M=8192 ~350 ms - i.e. prefill is **compute-bound at
M=2048 and roughly balanced at M=8192**, and the compute is bf16 MFMA at half rate.

## 2. (b) Kernel inventory of one prefill chunk

### 2.0 Every gfx908 fast path is off during prefill

Every gfx908 kernel written in rounds 1-6 is gated at a decode-sized M and is therefore **off**
during prefill: routed-MoE HIP W4 GEMV and `VLLM_GFX908_W4A8` and `SHARED_AS_EXPERT` at `M<=8`
(`experts/triton_moe.py:605-607`, `gfx908_moe_hip.py:36`, `gfx908_w4a8.py:54,135-149`);
shared-expert GEMV, dense W4 GEMV, fused router and fused top-k at `M<=16`
(`gfx908_shared_expert.py:36,157`, `triton_w4a16.py:253,355`, `gfx908_router_topk.py:59,225-240`,
`gfx908_topk.py:15`); HC fused mix at `M<=3` (`gfx908_hc_fused.py:47,391,424`);
`wvSplitK`/`LLMM1`/Triton mid-M at `M<=4`/`n==1`/`5<=M<=64` (`layers/utils.py:679-716`,
`gfx908_midm_gemm.py:89-90`); GDN fused decode glue (`qwen_gdn_linear_attn.py:1926`).

So a prefill step runs **stock upstream kernels plus rocBLAS**, except that the GDN projections
now pay a per-call int8->bf16 rematerialisation because `VLLM_GFX908_W8A16_FREE=gdn` released
their bf16 masters (`layers/utils.py:610-635`, `Dockerfile.q38fn:15`). HC is unaffected:
`VLLM_GFX908_HC_W8` defaults to `"0"` (`gfx908_hc_fused.py:114`) and is not set in the image, so
the HC weights stay bf16 and `_hc_mix_impl` takes the "stock chain" branch at
`gfx908_hc_fused.py:436-442`.

### 2.1 Per-layer launch sequences and cost estimates

Confidence: **H** arithmetic from the code, **M** derived with an efficiency guess, **L**
structural argument only.

#### Hyper-connections - 2 modules per layer, 96 total (+1 final mixer)

| kernel | file:line | launches/fwd | est. ms @M=2048 | @M=8192 | conf |
|---|---|---|---|---|---|
| `_hc_combine_norm_kernel` (grid `(M, 4)`) | `qwen4_exp/amd/ops/hc.py:266-330` | 95 | 11 | 45 | H (bytes) |
| `mix_down` GEMM 10240x336 bf16 | `gfx908_hc_fused.py:438` -> `utils.py:742` -> `F.linear` | 97 | **15-30** | **60-120** | M |
| `mix_up` GEMM 320x10240 bf16 | `gfx908_hc_fused.py:441` | 96 | **14-28** | **56-112** | M |
| `_hc_gate_mix_kernel` (grid `(M,5)`) | `ops/hc.py:125-186` | 97 | 8 | 31 | H (bytes) |
| **HC subtotal** | | **~484** | **~50-70 ms (6-8%)** | **~200-280 ms** | |

The two GEMMs are 1.30 GFLOP/token/rank of pure bf16 MFMA, **replicated on all 4 ranks**
(`MergedColumnParallelLinear(..., disable_tp=True)` `hyperconnection.py:96-107`; `ReplicatedLinear`
`:120-127`). At peak the pair is 65 ms at M=8192; a K=320 GEMM and an N=336 GEMM will not reach
peak on Tensile, hence the range.

#### MoE - all 48 layers (`decoder_sparse_step` makes every layer MoE)

Nine launches per layer, in order (grids from `fused_moe.py:704-708`): (0) router gate `F.linear` 2560x512 bf16
(`fused_moe/runner/moe_runner.py:913`); (1) `topk_softmax` plus a **fresh** fp32 `M x 512`
workspace every layer (`csrc/.../topk_softmax_kernels.cu:826`, `router/fused_topk_router.py:170`)
= 4.2 / 16.8 MB; (2) `moe_align_block_size_kernel` - **2 workgroups on 120 CUs**
(`csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:695`); (3)
`count_and_sort_expert_tokens_kernel` (`:707`), 80 / 320 WGs and 20,480 / 81,920 global atomics;
(4) `fused_moe_kernel_gptq_awq` w1 (`fused_moe.py:724`), grid 4,120 / 5,740; (5)
`torch.ops._C.silu_and_mul`; (6) `fused_moe_kernel_gptq_awq` w2 with `MUL_ROUTED_WEIGHT`, grid
32,960 / 45,920; (7) `moe_sum` -> `moe_sum_vec_dynamic_kernel` reading 105 / 419 MB - **topk=10
is not in the templated switch `{1,2,4,6,8,9}`** (`moe_align_sum_kernels.cu:896-914`) so it takes
the runtime-loop variant; (8) the shared expert (`triton_w4a16_gemm` x2 + `SiluAndMul` + gate
sigmoid, `qwen2_moe.py:116-133`). Workspaces `workspace13`/`workspace2` are `(M,10,2560)` bf16
each = **839 MB persistent at M=8192** (`experts/triton_moe.py:220-235`,
`modular_kernel.py:1169-1176`); there is no `VLLM_FUSED_MOE_CHUNK_SIZE` in this tree.

Config used (`fused_moe.py:1422-1454`, device string from `utils/platform_utils.py:69-74`):
`configs/E=512,N=160,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int4_w4a16.json`. It
**does** carry keys up to 4096, so M=2048 hits an exact key and M=8192 borrows the 4096 entry
(`configs[min(keys, key=lambda x: abs(x-M))]`, `:1448`).

| | key | BLOCK_M | BLOCK_N | BLOCK_K | GROUP_M | warps | stages | M-padding waste |
|---|---|---|---|---|---|---|---|---|
| M=2048 | 2048 | 64 | 64 | 32 | 1 | 4 | 2 | 1.61x (52,736 padded rows for 20,480) |
| M=8192 | 4096 (borrowed) | 128 | 64 | 32 | 4 | 4 | 2 | 1.79x (146,944 for 81,920) |

`BLOCK_K` is pinned at 32 not only by `group_size=32` but because the **second** GEMM has
K = N_moe = 160 and `b = tl.load(b_ptrs)` (`fused_moe.py:236`) is **unmasked** (`block_k_diviable`
at `:751` guards only A and the scales), so any BLOCK_K not dividing 160 reads past the expert
slab and faults. The MoE launcher has no clamp; the dense sibling does (`triton_w4a16.py:414-419`).
`SPLIT_K` (`:105`) is never read - inert.

Cost: real 2.42 TFLOP (M=2048) / 9.66 TFLOP (M=8192), issued 1.6-1.8x that, plus ~6.3e9 VALU
ops/layer of in-kernel nibble dequant (~26 ms/chunk at M=8192) and ~2.3 GB/layer of HBM.
**Estimate 250-350 ms at M=2048, 550-800 ms at M=8192; confidence M.**

#### QSA layers (12) - the decode-shaped block

| kernel | file:line | grid at M=8192 | est. ms/chunk (12 layers) | conf |
|---|---|---|---|---|
| indexer prologue: `index_qk_proj` + 2 norms + 2 MRoPE + compress + 3 stores | `amd/indexer_qsa.py:161-270`, `ops/qsa.py:401-583` | O(M) | 10-20 | L |
| **`_qsa_mqa_paged_kernel`** (indexer scoring) | `ops/qsa.py:19-114`, launch `:591-667` | **(rows, 8192/32=256)** = 2.1M WGs | **30-45** @M=8192, **9-14** @M=2048 | M |
| `topKPerRowDecode` (insertion-sort final, `numColumns=8192 < 12288`) | `csrc/libtorch_stable/sampler.cu:717-740, 574` | 1 block/row, 512 thr | 5-15 | L |
| `_expand_qsa_indices_kernel` + `reshape_and_cache_flash` | `ops/qsa.py:117-190` | O(M) | 4-6 | H |
| **`_qsa_sparse_paged_gqa_splitk_kernel`** | `ops/qsa.py:188-355`, launch `:869-953` | **(M, 1, 1)**, 2 warps, 33 tiles | **60-120** @M=8192, **20-40** @M=2048 | M |
| `o_proj` gate sigmoid + `o_proj` W4 | `amd/qsa.py:412-421` | | 5 | H |

Prefill launch profile (`ops/qsa.py:879-895`): `base_programs = M x 1 kv_head > 512` ->
`block_n=64, target_splits=1, partial_warps=2, num_stages=1`; `NUM_TILES = cdiv(2051,64) = 33`;
`num_splits=1` so the merge kernel is skipped. Consequences: **one program per token, two warps,
no query tiling** - every token independently gathers 2.1 MB of K/V (206 GB/chunk nominal at
M=8192 against an 8 MB L2 and a 16.8 MB per-layer KV working set at 16k ctx); `for tile in
range(0, 33)` (`:252`) runs unconditionally, so a token at position 5 does the same work as one
at 16,383; and `BLOCK_M = max(next_pow2(6), 16) = 16` (the gfx908 miscompile clamp, `:868-874`)
makes 10 of 16 MFMA rows padding. Issued/useful at a 2048-token prompt: **5.5x**.

`_qsa_mqa_paged_kernel` is worse structurally: `columns = page_table.shape[1] * page_size` =
`max_model_len / compress_ratio` = **8192**, used unconditionally (`:627-629`;
`qsa_select_paged_tokens` never passes the existing `num_columns` argument, `:750-771`). It is
an **fp32 VALU** reduction (`tl.sum(keys * query[None,:], axis=1)`, `:91` - no `tl.dot`), writes
`-inf` into every masked column, and allocates `[rows, 8192]` fp32 logits per Python chunk of
`_LOGITS_WORKSPACE_BYTES // (8192*4) = 4096` rows (`:15, 753`) - 2 host iterations of ~5 launches
per layer at M=8192, ~258 MB written and re-read per layer (~6.2 GB/chunk, ~76% masked padding
for chunk 1 of a 16K prompt). `output.zero_()` (`amd/qsa.py:139`) adds 289 MB/chunk of memset.

**For a 2048-token prompt the entire indexer is mathematically dead weight**: with
`indexer_budget = 2048`, any token whose context is <= 2048 selects its whole causal prefix, so
QSA output == dense causal attention, and all 12 project/norm/rope/compress/score/top-k/expand
chains produce the identity selection.

#### PLE - one layer only (`ple_layer_ids [2]`), fully replicated

`Qwen4ExpPLELayer.forward` (`amd/ple_layer.py:1250-1272`): n-gram gather (section 3), `key_proj`
2560->10240 + `value_proj` 2560->2560 (65.5 MFLOP/token replicated, ~12 ms at M=8192), three
grouped norms over `[M,4,2560]`, a `key*query` reduction, `gated_value`, `torch.zeros_like`, then
the short conv. `_short_conv_dilated_prefill_batched` (`:846-975`) is **eager PyTorch on 10240
channels** and materialises ~168 MB at M=8192 in each of `new_zeros` (`:900`), the scatter
(`:901`), `.transpose().contiguous()` (`:902`), `torch.cat` (`:934`), a **depthwise `F.conv1d`
with `groups=10240`, k=4, dilation=3** (`:937`), silu + `.contiguous()` (`:943`),
`masked_fill_` (`:949`) and the advanced-index gather (`:950`) - **~1.7 GB of HBM plus a grouped
conv MIOpen has no good gfx908 kernel for**. Estimate 15-40 ms/chunk at M=8192, confidence L.

#### GDN layers (36) - `_forward_core` (`mamba/gdn/qwen_gdn_linear_attn.py:1290`)

`gqa_interleaved_layout=False` (`amd/model.py:239`) so the AITER decode fast path in
`_forward_core_rocm:1263-1267` can never fire; it only adds `core_attn_out.zero_()` (`:1277`) and
`z_out[:] = z` (`:1282`). `ChunkGatedDeltaRule` resolves to `forward_native` on ROCm
(`:124-125, 255-256`) = FLA `chunk_gated_delta_rule`. Per layer per chunk, in order:

Eight Triton kernels per layer, in order (programs @T=2048 / 8192):
`_causal_conv1d_fwd_kernel` (`mamba/ops/causal_conv1d.py:717`) 2560/10240;
`_fused_post_conv_kernel` (`fused_gdn_prefill_post_conv.py:215`) 2048/8192;
`chunk_local_cumsum_scalar_kernel` (`fla/ops/cumsum.py:182`), `chunk_scaled_dot_kkt_fwd_kernel`
(`chunk_scaled_dot_kkt.py:162`), `merge_16x16_to_64x64_inverse_kernel` (`solve_tril.py:547`),
`recompute_w_u_fwd_kernel` (`wy_fast.py:139`) - all 384/1536;
**`chunk_..._fwd_kernel_h_blockdim64`** (`chunk_delta_h.py:362`) **24 (BV=64) or 48 (BV=32),
independent of T**; `chunk_fwd_kernel_o` (`chunk_o.py:173`) 768-1536 / 3072-6144.

BT = 64 fixed (`fla/ops/utils.py:31`). Budget per layer per rank: **4.47 GFLOP / 194 MB at
T=2048; 17.9 GFLOP / 776 MB at T=8192** (94.7 KB/token/layer, ~22 KB of it redundant re-reads of
`k`/`w`/`q` across V-blocks). AI = 23 FLOP/byte vs machine balance 75 -> **memory-bound by 3.3x**.
x36 layers: 6.99 GB (5.7-8.7 ms) at T=2048, 27.9 GB (23-35 ms) at T=8192.

Three structural problems, found offline (`triton.compile` for gfx908, no GPU):
- **`chunk_delta_h` is the serial recurrence and runs on 24-48 workgroups of a 120-CU part**,
  looping over all NT chunks; its grid `(cdiv(V,BV), N*H)` does not scale with T at all. At
  ~0.2-0.3 TB/s effective that is ~0.7-1.0 ms/layer at T=8192, **~25-36 ms for 36 layers -
  plausibly the whole GDN prefill cost on its own.** `BV=16` is *not* in the autotune space
  (`chunk_delta_h.py:39`) and is the only config reaching occupancy 2 with zero spills (VGPR 127
  at nw=8, 96 programs); `BV=64, nw=2` **is** in the space and spills 160-257 VGPRs; `BV=8` would
  hit the gfx908 4x64-broadcast miscompile (`v_mfma_f32_4x4x2bf16 cbsz:512`) - hard-floor 16.
- **Every FLA autotuner is trained on a `T=64` warmup and never re-tunes.**
  `_warmup_prefill_kernels` (`qwen_gdn_linear_attn.py:1097-1220`) runs the chain at
  `FLA_CHUNK_SIZE=64` with `cu_seqlens=[0,64]` (`:1178`), and no `key=` list contains `T` or `NT`
  (`chunk_delta_h.py:41`, `chunk_o.py:39`, `wy_fast.py:26`, `chunk_scaled_dot_kkt.py:44`). So
  kernel 7 is tuned against a 1-iteration loop and kernel 8 at 1/32-1/128 of production
  occupancy with a hot L2.
- **`chunk_scaled_dot_kkt` runs a full fp32 MFMA** (`:103`; `b_beta` is fp32 from
  `fused_gdn_prefill_post_conv.py:203`) = `v_mfma_f32_32x32x2f32` at 46.1 TFLOP/s. The
  `CAST_DOT_TO_K_DTYPE` bf16 path exists but is gated on `on_gfx1x()` (`:24-28`); enabling it on
  gfx9 halves the MFMA count, cuts VGPRs to 69, raises occupancy 2->3, and removes a
  252-SGPR-spill config. Numerics risk is real (beta*k rounded to bf16) - full PPL gate.

State carry ("align") costs ~5 full traversals of the 786 KB fp32 state per sequence/layer/chunk
(`:1548` gather, `:1549` masked zero, `h0` read, `ht` write, `:1567` scatter) of which 2 are
required; and `chunk_fwd_o` allocates its own output because `:1553-1566` never passes
`core_attn_out=` although the plumbing exists (`chunk_o.py:162-166`), forcing the `:1611` copy =
6.5% of GDN prefill traffic. The scheduler-level align copy (`v1/worker/mamba_utils.py:1433-1538`)
is one fused 72-region `batch_memcpy`, ~58 MB/chunk - negligible.

#### Dense GEMM dispatch at prefill M

`dispatch_unquantized_gemm` (`layers/utils.py:825-832`) -> opaque custom op
`rocm_unquantized_gemm_gfx908` (`:746, 819`) -> `_gfx908_gemm_bf16` (`:640`). At n = 2048/8192
every skinny predicate fails (`wvSplitK` n<=4 `:684`; `LLMM1` n==1 `:694`; Triton mid-M
`5<=M<=64` `gfx908_midm_gemm.py:89-90`; AITER `gemm_a16w16` shape whitelist `utils.py:250-272`
never matches this model) and the terminal path is **`F.linear` -> rocBLAS** (`:743`). Exceptions:
`m <= 8` takes `torch.einsum` (`:734`, i.e. `shared_expert_gate` N=1, never tested at large M),
and the two W8A16-freed GDN shapes take `gfx908_w8a16.py:643 -> _dequant_gemm(:607)`.
hipBLASLt is **off** (`platforms/rocm.py:297` sets `DISABLE_ADDMM_HIP_LT=1` as a gfx908 default);
TunableOp is not enabled, and `docs/mi100_decode_opt/tunableop_results/*.csv` are Qwen3.6-35B
shapes (K=2048/1024/1152) - **none of this model's shapes appear**.

W4 dense (QSA `qkv_proj`/`o_proj`, shared expert) selects `TritonW4A16LinearKernel`
(`kernels/linear/__init__.py:496-504`; the gfx908 default that would pin Exllama at
`platforms/rocm.py:281-283` is overridden because the serve line sets `VLLM_DISABLED_KERNELS`
explicitly, `:299-301`). Its gfx908 tile table is hardcoded (`triton_w4a16.py:385-397`):
`M > 64 -> BLOCK_M=128, BLOCK_N=64, BLOCK_K=32`, **no `GROUP_SIZE_M` L2 swizzle, `num_warps` /
`num_stages` never passed** (AMD defaults 4/2), one bucket for M = 65..8192, mirrored from the
MI300 branch and **never tuned on gfx908**. With `pid_m` fastest and a 64-wide N tile the A
matrix restreams per N-tile column: QSA `qkv_proj` at M=8192 has a worst-case A re-read of
2.24 GB against 150 GFLOP of MFMA. And unlike `exllama.py:64-71` (`if x.shape[0] >
dequant_mthresh: w = triton_w4a16_dequant(...); return torch.mm(x, w)`, default 512, comment
records *"wash at M=256, 1.4x win at M=2048"*), `TritonW4A16LinearKernel.apply_weights`
(`:728-763`) has **no large-M dequant escape**.

**GDN W8A16 rematerialisation:** `_dequant_gemm` (`gfx908_w8a16.py:607-641`) uses a single 32 MB
bf16 scratch per device (`:139, 400-409`), so nothing is cached across layers or chunks: per
forward pass it writes and immediately re-reads 36 x (20.97 + 7.86) MB = **1.50 GB**
(qkvz 10.00 int8 + 0.31 scales -> 20.00 bf16; out_proj 3.75 + 0.12 -> 7.50). That is 1.97 ms =
54.7 us/layer at ~0.8 TB/s, matching the documented "+54 us/layer" exactly. At M=8192 it is ~1%
of the layer's GEMM, but the cost is **per call**, so it grows as the chunk shrinks (~4x more
significant at M=512). Escape hatches: `VLLM_GFX908_W8A16_FREE=none` (pay 510 MB of duplicate
weights) or a per-layer dequant cache.

Roll-up at M=8192 per rank from the table in 1.1: GDN layers ~16.5 TFLOP (~455 ms), QSA layers
~5.5 TFLOP (~233 ms), routed MoE 9.7 TFLOP (~580 ms), **of which HC alone is 10.6 TFLOP (~285 ms)
and 4x redundant across ranks**. That already reaches the inferred chunk time before QSA
attention, GDN, PLE and the all-reduces, so either these efficiencies are 20-30% pessimistic or
something is wrong. **Resolve with a trace before acting on the ranking.**

## 3. Scheduler, host side, and the collectives

**Chunking.** `enable_prefix_caching` defaults True (`config/cache.py:138`); a hybrid model gets
`mamba_cache_mode="align"` (`model_executor/models/config.py:622-624`) and `mamba_block_size =
cache_config.block_size` (`:648-649`). `platforms/interface.py:908-924` inflates the attention
block size until one page covers a mamba page: GDN page/rank = conv `2560x3` bf16 (15,360 B) +
SSM `(12,128,128)` fp32 (786,432 B) = 801,792 B, QSA page 1024 B/token, so
`16 * cdiv(801792, 16384) = **784**` (cross-check: 784 x ~400 blocks = the 313.6k-token KV pool).
`scheduler.py:333-335` sets `need_mamba_block_aligned_split` (applied `:613-616` / `:1009-1017`)
and `_mamba_block_aligned_split` floors `end` to a multiple of 784 (`:443-445`) on every chunk
that is not the request's last.

| prompt | passes | tokens per pass |
|---|---|---|
| 2048, c=1 | 1 | 2048 |
| 16384, c=1 | 3 | 7840 / 7840 / 704 |
| 16384 x4, c=4 | **9** (not 8) | ~7840 each, ~350 tokens of budget stranded per step |

At c=4 the WAITING loop `break`s rather than `continue`s when the remaining budget is below one
block (`scheduler.py:1016-1017`), so prefills serialize per request.

**CUDA graphs: prefill is fully eager.** `max_cudagraph_capture_size = min(48*1*2, 512) = 96`
(`config/vllm.py:2001-2018, 2097-2098`), 8192 is never added to the buckets (`:2138-2143`), and
`cudagraph_dispatcher.py:274-281` returns `CUDAGraphMode.NONE` for `num_tokens > max_size`. The
GDN backend reports `UNIFORM_BATCH` (`gdn_attn.py:84`) so a requested `FULL` is rewritten to
`FULL_AND_PIECEWISE` (`config/compilation.py:1394-1419`) - "FULL graphs" means uniform decode
only. `splitting_ops` (`:765-784`) covers the GDN core, `qwen4_exp_qsa_with_output` and
`qwen4_exp_ple_short_conv`, so a pass is 49 split ops = 50 inductor subgraphs = ~99 Python-level
callables fanning out to ~2,000-2,400 HIP launches, none replayed.

**Metadata builds are cheap and sync-free** - 4 KV-cache groups (GDN `MambaSpec`, QSA main
`FullAttentionSpec`, QSA raw-key `CircularBufferSpec`, QSA compressed `MLAAttentionSpec`), all
built from `*_cpu` tensors; QSA metadata is one Triton kernel with an in-kernel binary search
(`qwen4_exp/common/qsa_cache.py:379-451`). The only Python loop is
`compute_causal_conv1d_metadata` (`v1/attention/backends/utils.py:1042-1088`): a ~980-element
`offsetlist` plus two uncached pinned allocations per chunk. **No `.item()`/`.cpu()` sync exists
in the per-chunk prefill path** - the round-4 PLE host stall is gone.

**PLE at prefill: zero-copy applies, no host gather.** `ple_layer.py:466-468` returns early from
`prepare_mmap_rows` whenever a ZC table is attached, with no token-count predicate, so the
memmap/threadpool path (and `VLLM_PLE_MMAP_SERIAL`) is dead at prefill. Per 7840-token chunk the
ZC gather (`gfx908_ple_zc.py:136-173`, `csrc/gfx908_ple_zc.hip:10-56`) launches
`<<<125440, 64>>>` - one 64-thread block per 320 B row with **only 20 of 64 lanes active** -
reads ~10 MB of scattered 320 B host rows (each rank owns `shard % 4 == rank`), writes 40.1 MB
of staging, then does a **full 40.1 MB TP all-reduce** (`:170`). `:149-153` runs `torch.cummax`
twice (copy-paste). `compute_ngram_ids` (`ple_layer.py:312-372`) is
O(num_reqs x `max_num_batched_tokens`), not O(num_tokens).

**lm_head and sampling are last-token only** - `logits_indices = query_start_loc[1:] - 1`
(`gpu_model_runner.py:2267-2276`), gathered before `compute_logits` (`:4561-4562`); non-final
chunks still run the sampler and discard it (`:3780-3822`) - a serial tail of tiny kernels.

**All-reduce: every prefill AR falls off the custom path onto RCCL, and RCCL is pinned to a
decode-tuned Tree+LL.** `CustomAllreduce.max_size = 8192*1024` = **8 MiB**
(`custom_all_reduce.py:125`; the per-capability override at `:208-222` is CUDA-only) and
`should_custom_ar` (`:413-427`) is a plain size test, so the crossover is
`8 MiB / (2560 x 2 B) = 1638 tokens`: **M=2048 (10.5 MB) and every prefill chunk (40-42 MB) use
pynccl/RCCL**. Meanwhile `platforms/rocm.py:290-291` sets, unconditionally for gfx908,
`NCCL_ALGO="Tree"` / `NCCL_PROTO="LL"`, chosen (comment `:285-290`) for "the small (~few KB)
per-step all-reduces decode produces". LL carries 4 B per 8 B flit (~2x wire bytes) and forcing
it removes RCCL's automatic LL->Simple switch at large sizes; Tree is not bandwidth-optimal on a
flat 4-GPU XGMI mesh.

Count: **97 ARs per pass** = 48 attention `RowParallelLinear` + 48 MoE + 1 PLE (HC adds none).
Payload per 7840-token chunk = 3.89 GB. Ring+Simple on XGMI is ~0.6-1.3 ms per 40 MB AR
(~60-125 ms/chunk); Tree+LL is estimated ~1.3-2.5 ms each, **~175 ms/chunk (10-13% of the pass)**
and ~2-3 s of the 12.1 s at 16K c=4; ~48 ms of the 850 ms at M=2048. **The cheapest large lever
in this report: two environment variables.**

## 4. (c) Ranked optimizations

Savings are per **2048-token prompt (one pass, 850 ms)** and per **7840-token chunk** of the 16K
case (~1.3-1.7 s, x3 for c=1 / x9 for c=4). Everything is modelled - nothing was run. Each item
needs the standing gate: unit test vs an fp32 reference at TP4 shapes, per-layer logprob parity,
then wikitext-2 PPL 3.141 +/- 0.01, then GSM8K.

| # | change | where | est. 2K | est. 7840-chunk | effort | risk |
|---|---|---|---|---|---|---|
| 1 | **Unset `NCCL_ALGO=Tree` / `NCCL_PROTO=LL`** (or `Tree,Ring` + `LL,Simple`) so RCCL picks Simple/Ring for the 10-42 MB prefill messages. Decode keeps its win via the size-based auto-switch. | env only; default set at `platforms/rocm.py:290-291` | **-20..35 ms** | **-60..140 ms** | **zero** (2 env vars) | low: re-gate decode c=1/c=48 |
| 2 | **Dense-causal fast path when `max_seq_len <= indexer_budget`**: skip the whole indexer (project/norm/rope/compress/score/top-k/expand) and run the 12 QSA layers through vLLM's Triton unified attention (`v1/attention/ops/triton_unified_attention.py`, which already has a gfx908 prefill tune gate at `:969-989` and does GQA query tiling). Mathematically identical, not an approximation. | `amd/qsa.py:328-381` + `amd/ops/qsa.py` | **-150..300 ms** | -0 (ctx > 2048) | 3-5 d | med: new attention path, full parity gate |
| 3 | **Pass `num_columns` to `qsa_mqa_paged`** = `cdiv(max_seq_len_in_batch, 4)` rounded to BLOCK_N (host-known in `CommonAttentionMetadata`). Kills 76-94% of the indexer scorer's work and its 6.2 GB/chunk of fp32 logits. | `amd/ops/qsa.py:750-771` (the arg already exists at `:598,627`) | -8..12 ms | **-7..10 ms** | **1 line** | none |
| 4 | **Query-tile the QSA prefill kernel**: process `BLOCK_Q` consecutive tokens per program against the union of their index sets (or simply re-block as `(cdiv(M,BQ), kv_head)` with a per-row index mask), and early-exit the tile loop on `visible`. Removes the 2.67x BLOCK_M padding and most of the 206 GB/chunk K/V re-gather. | `amd/ops/qsa.py:188-355, 869-953` | -10..25 ms | **-40..90 ms** | 1-2 wk | med-high: numerics identical but a new kernel |
| 5 | **Give `TritonW4A16LinearKernel` the large-M dequant escape** (`triton_w4a16_dequant` + `torch.mm`), mirroring `exllama.py:64-71`, threshold ~512. Affects QSA qkv/o_proj (12 layers) and the shared expert (48). Dequant is ~23 MB (~25 us) against a multi-ms GEMM. | `triton_w4a16.py:728-763` | -15..25 ms | **-50..70 ms** | 1 d | low: same weights, hgemm accumulate |
| 6 | **Shard HC `mix_down` over TP** (K-split by HC stream: hc_count 4 == TP 4) + one `[M,336]` all-reduce (5.5 MB at M=8192). Saves 3/4 of 56.4 GFLOP x 97 modules. The decode-era "don't shard HC" verdict (`research/bf16_skinny_gemm_and_int8.md:78-82`) is a bytes argument that inverts at prefill. | `amd/hyperconnection.py:96-107`, `gfx908_hc_fused.py:436-442` | -20..35 ms | **-70..90 ms** | 2-3 d | med: new collective in the HC path, needs an M gate so decode is unchanged |
| 7 | **Fix the FLA autotune warmup**: warm up at `T=2048` instead of `T=64`, and/or add `NT` to the autotune keys. Every GDN chunk kernel is currently tuned against a 1-iteration recurrence at 1/32-1/128 of production occupancy. | `qwen_gdn_linear_attn.py:1097-1220`, `chunk_delta_h.py:41`, `chunk_o.py:39`, `wy_fast.py:26` | -3..6 ms | **-5..12 ms** | 1 d | low |
| 8 | **`chunk_delta_h` grid**: add `BV=16` to the config list, drop `num_stages=3`, remove the spilling `BV=64,nw=2`, hard-floor `BV>=16`. 24-48 -> 96 programs on 120 CUs; trades bytes for memory-level parallelism, so sweep it. | `chunk_delta_h.py:21, 34-43` | -3..6 ms | **-10..25 ms** | 1 d + sweep | low (`BV>=16` avoids the 4x4-broadcast miscompile) |
| 9 | **Re-tune the MoE JSON at M>=2048 and add an `8192` key.** M=8192 borrows the 4096 entry; `BLOCK_M=128` costs 1.79x M-padding vs 1.20x at 64, and the whole table was tuned before this campaign. Also add the missing `BLOCK_SIZE_K` clamp in `invoke_fused_moe_wna16_triton_kernel` (mirroring `triton_w4a16.py:414-419`) so a bad JSON degrades instead of faulting. | `configs/E=512,N=160,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int4_w4a16.json` | -20..50 ms | **-60..150 ms** | 1 d tune | low |
| 10 | **Fuse `moe_sum` into GEMM2's epilogue** (or at minimum add `case 10` to the templated switch at `moe_align_sum_kernels.cu:896-914`, since topk=10 falls into the runtime-loop variant). Removes 10 GB (M=2048) / 40 GB (M=8192) of `ic3` round trip per pass. | `experts/triton_moe.py:794-814` | -8..15 ms | **-30..50 ms** | 2-3 d | med: touches the MoE epilogue |
| 11 | **fp16 MFMA for the bf16 GEMMs.** gfx908 fp16 MFMA is 184.6 TFLOP/s vs bf16 92.3. bf16->fp16 is exact for in-range values (fp16 has 3 more mantissa bits); the campaign already ships an exact fp16-dot2 W4 mode on this principle (`VLLM_GFX908_W4A8_MODE=f16`). Applies to HC (10.6 TFLOP/chunk), GDN projections, PLE, router. Needs overflow clamping and a full PPL gate. | `layers/utils.py:743` wrapper | -25..40 ms | **-90..150 ms** | 1 wk | **high**: range/denormal behaviour, needs per-tensor scaling to be safe |
| 12 | **Chunk-size / block-size hygiene.** `block_size` is forced to 784 by the mamba-align rule; setting `--block-size 1024` or `2048` (must stay >= 784, `interface.py:916-917`) makes a 16K prompt 2 passes instead of 3 and removes ~350 stranded tokens/step at c=4. Costs coarser prefix-cache granularity. | serve flag | n/a | **-1 whole pass per 16K prompt** | zero | low-med: verify the KV pool still fits and prefix-hit rate |
| 13 | Small, free: `output.zero_()` in `amd/qsa.py:139` (289 MB/chunk of memset); the duplicated `torch.cummax` in `gfx908_ple_zc.py:149-153`; pass `core_attn_out=` to `chunk_fwd_o` (`qwen_gdn_linear_attn.py:1553`, kills a 50 MB copy/layer at T=8192, 6.5% of GDN traffic); gate `GDN_AITER_TRITON_AVAILABLE` on `gqa_interleaved_layout` so qwen4_exp stops paying `zero_()`+`z_out[:]=z` per layer; concat `in_proj_ba` (N=24, 64 WGs) into `in_proj_qkvz`; cache the W8A16 dequant per layer or set `VLLM_GFX908_W8A16_FREE=none`. | various | -10..20 ms | -20..40 ms | 0.5-2 d each | low |
| 14 | **PLE short conv as one Triton kernel** (gather + depthwise dilated conv + silu + scatter, ~1.7 GB -> ~0.2 GB) and widen the ZC gather to use all 64 lanes (`csrc/gfx908_ple_zc.hip:22,25`). | `amd/ple_layer.py:846-975` | -5..10 ms | **-10..30 ms** | 3-4 d | low |
| 15 | `chunk_scaled_dot_kkt` bf16 dot on gfx9 (46.1 -> 92.3 TFLOP/s for that kernel) by extending `_CAST_DOT_TO_K_DTYPE` past `on_gfx1x()`; `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` A/B (process-global - validate the W4/MoE/QSA kernels too); `DISABLE_ADDMM_HIP_LT=0` A/B at large M (only ever measured at M=1). | `chunk_scaled_dot_kkt.py:24-28`; env | -2..5 ms | -5..15 ms | 0-1 d | med (numerics / untested backend paths) |

**If only three things get done**: #1 (free), #3 (one line), #5 (one branch) - plausibly
-50..70 ms on the 2K case and -120..220 ms per 16K chunk for ~two days. The big structural items
are #2/#4 (QSA does decode-shaped work at prefill) and #6/#11 (HC is 1.3 GFLOP/token of
replicated bf16).

## 5. (d) Profiling plan

Prefill kernels are >30 us so the round-4 inflation caveat does not apply to the big ones, but a
pass still has ~200 sub-10 us glue launches inflated by ~2.4 us each; always check the
kernel-table sum against wall time before attributing.

**Prompts.** Four arms, greedy, `max_tokens=1` so the trace is prefill only. **A: 2048 tokens
c=1** - one pass, no chunk boundary; the reference for every "2K" number here and the arm where
the whole indexer is dead weight. **B: 1600 tokens c=1** - below the 1638-token custom-AR
crossover, so the AR kernel should be `cross_device_reduce_1stage` not `ncclDevKernel`
(validates lever #1). **C: 16384 tokens c=1** - three passes 7840/7840/704, confirms the
block_size=784 split and gives long-context per-chunk cost. **D: 16384 x4, c=4** - reproduces
the 12.1 s anchor and the 9-pass / stranded-budget behaviour.

**Isolating chunks in the trace.** Steps are not separated by a graph replay at prefill (mode
`NONE`), so use these markers: (i) `ncclDevKernel_*` appears exactly 97 times per pass under the
current config - segment the trace on runs of 97; (ii) after lever #1 or in arm B, segment on
`cross_device_reduce_1stage`; (iii) the PLE ZC gather `<<<125440,64>>>` fires exactly once per
pass and is unmistakable by grid size; (iv) `moe_align_block_size_kernel` fires exactly 48 times
per pass. Cross-check the number of passes against the engine log's scheduled-token lines and
against `interface.py:916-921`'s "Setting attention block size to N tokens" (confirm N=784).

**Method.** Reuse `docs/mi100_decode_opt/scripts/parse_profile.py` on a `torch.profiler` trace,
grouping kernel names into the buckets of section 2.1 (HC GEMM / HC glue / MoE GEMM / MoE glue /
QSA attn / QSA indexer / GDN chunk / GDN proj / PLE / AR / other); report launches and ms per
bucket per pass plus the bucket sum vs wall time.

**Order of investigation** (each settles a question this note could not):
1. **Is there a large fixed per-pass term?** Compare arm A (2048 tokens) with arm C's 704-token
   tail chunk. If the 704-token pass costs a large fraction of 850 ms, the fixed term dominates
   short prompts; locate it in the bucket table (candidates: 17 GB of MoE expert weights, 1.5 GB
   of GDN dequant, the PLE conv chain, ~2,300 eager launches).
2. **HC vs MoE vs QSA split.** Section 2 predicts HC ~50-70 ms and MoE ~250-350 ms at M=2048.
   If the measured HC share is much larger than predicted, levers #6/#11 move to the top.
3. **The QSA pair.** `_qsa_sparse_paged_gqa_splitk_kernel` and `_qsa_mqa_paged_kernel` are the
   two kernels whose cost I am least able to bound (L2 hit rate decides the first, workgroup
   dispatch rate the second). Measure both in arm A and arm C - if the A/C ratio is much less
   than 4x, they are dispatch-bound and lever #4 matters more than lever #2.
4. **`chunk_delta_h` occupancy.** Dump `rocprof` MeanOccupancyPerCU for it in arm C; if it is
   near 20-40% as predicted, lever #8 is confirmed.
5. **All-reduce.** Time arm C with and without `NCCL_ALGO`/`NCCL_PROTO`, and log
   `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` once to record how many rings RCCL builds on
   this hive - the 60-175 ms/chunk range in section 3 collapses once that is known.
6. **Sweep Triton configs offline first** (`research/gdn_qsa_triton_gfx908.md` section 5.2):
   `triton.compile(..., GPUTarget("hip","gfx908",64))`, rejecting any config with VGPR spills or
   `v_mfma_f32_4x4x*`. That alone found the `BV=64,nw=2` spill and the fp32 kkt dot.

**Gate before any number is quoted**: 3 in-server probes, greedy logprob parity at c=1 and c=16
(noise floor ~0.006 nats), wikitext-2 PPL 3.141 +/- 0.01 (which for once is the *right* gate -
it is a prefill-only measurement at M >= 2048), GSM8K, and TTFT from a full BenchAndReport tier,
never a probe.
