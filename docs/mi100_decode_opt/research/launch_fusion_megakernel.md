# Launch-count and glue-kernel reduction for Qwen3.8-Flash-Next decode on gfx908

Research note, 2026-09-03. Read-only survey; nothing here has been run. Numbers are derived from
`_profile_c1_launch_map.txt` (1,896 launches / token, 16.2 ms profiled, ~14 ms real GPU step at c=1,
TP4) with the profiler-inflation correction: tiny-kernel real cost = profiled − 2.4 µs, and every
kernel boundary inside a FULL cudagraph costs ~1.33 µs of idle GPU.

Model facts used throughout (from `config.json` of the served artifact): 48 layers = 36 GDN + 12 QSA
(`GGGF` repeating), every layer is MoE (512 experts, top-10, one shared expert), hidden 2560,
hc=4 (multi-stream state 10240), hc_lowrank 320, MoE intermediate 640 (160/rank at TP4), shared
expert intermediate 640 (also 160/rank), GDN 16 k-heads / 48 v-heads × 128 (4/12 per rank),
QSA 24 q / 2 kv heads × 256 (6/… per rank), indexer 4 heads × 128, one PLE layer (`ple_layer_ids=[2]`).

## 0. Where the ~1,900 launches come from (per token, c=1)

| group | launches | profiled ms | real ms (est.) | code |
|---|---|---|---|---|
| bf16 GEMV `wvSplitK_hf_sml_` (HC mix_down/up ×194, GDN in_proj_qkvz/ba/out_proj ×108, router ×48, indexer ×12) | 365 | 4.34 | ~3.5 | `csrc/rocm/skinny_gemms.cu:1208` (`grid(CuCount)`, persistent WGs) |
| W4 MoE HIP GEMV `w4gemv3_kernel` (2/layer) | 96 | 2.30 | ~2.1 | `fused_moe/csrc/gfx908_w4gemv.hip`, `gfx908_moe_hip.py` |
| `topkGating` (512-way softmax + top-10) | 48 | 1.13 | ~1.0 | `csrc/libtorch_stable/moe/topk_softmax_kernels.cu:278-600` |
| custom all-reduce `cross_device_reduce_1stage` (2/layer) | 96 | 0.99 | ~0.75 | `csrc/libtorch_stable/custom_all_reduce.cu:70` |
| W4 dense GEMV partials (QSA qkv/o ×24, shared gate_up/down ×96) + split-K reduces | 120+24 | 1.08 | ~0.8 | `kernels/linear/mixed_precision/triton_w4a16.py:173,223`, `layers/gfx908_shared_expert.py` |
| `__amd_rocclr_copyBuffer` (hipMemcpyAsync D2D) | 187 | 0.74 | ~0.3 | see §4 |
| HC glue: `_hc_combine_norm` ×95, `_hc_gate_mix` ×97, `_hc_silu` ×97 | 289 | 1.39 | ~0.7 | `models/qwen4_exp/amd/ops/hc.py` |
| shared-expert reduces `_reduce_silu_mul` / `_reduce_gate` | 96 | 0.77 | ~0.55 | `layers/gfx908_shared_expert.py:31,51` |
| GDN: `causal_conv1d_update` ×36, `fused_recurrent…packed_decode` ×36, gated-RMSNorm (inductor `triton_per_fused…sigmoid` + `triton_poi_fused…rocm_unquantized_`) ×72 | 144 | 1.22 | ~0.9 | `mamba/gdn/qwen_gdn_linear_attn.py:1195-1260,1833` |
| QSA (12 layers × ~22): sparse attn splitk+merge, mqa_paged, compress, expand, topKPerRow, store_rows ×3, mrope ×2, grouped_gemma_rmsnorm ×2, reshape_and_cache, q/k-norm inductor kernels, sigmoid-gate mul | ~270 | 1.55 | ~0.9 | `models/qwen4_exp/amd/{qsa.py,indexer_qsa.py,ops/qsa.py}` |
| eager elementwise inside custom ops (`at::native::*elementwise*`, `index_elementwise`, gathers): fills (`zero_` ×48), casts, index_select | ~140 | 0.65 | ~0.25 | GDN `core_attn_out.zero_()`, QSA `output.zero_()`, PLE `ple_layer.py:787-846` |
| inductor glue (`triton_poi_fused_add_*` ×48 = `shared + routed`, router bf16→fp32 cast, q/k-norm pieces, etc.) | ~180 | 0.85 | ~0.35 | Inductor islands between custom ops |

Notes. (1) The launch map still shows `_hc_silu`/`_hc_gate_mix` ×97 each; with `VLLM_GFX908_HC_FUSED=1`
(`gfx908_hc_fused.py`, epilogues in the wvSplitK copy) those 194 launches are already gone, so the
live count is ~1,700. (2) Glue proper (everything except the GEMVs and the AR) is ~1,100 launches
≈ 1.5 ms of boundaries + ~2.5 ms of real kernel time, i.e. 4 ms of a 14 ms step. (3) The dense bf16
GEMVs are 8 µs latency-floor kernels for 0.4–4.6 MB of weight: ~365 × (8 − ~3 µs bandwidth time)
≈ 1.8 ms of "memory bubbles" that only cross-kernel prefetch (a megakernel) could recover.

## 1. Fusion candidates, ordered by per-token value

Value = launches saved × 1.33 µs + real kernel time saved, per token. "Sync" = the cross-workgroup
mechanism needed (none / last-arriving WG / dataflow counters).

| # | candidate | launches saved | est. ms/token | sync | confidence | code locations |
|---|---|---|---|---|---|---|
| 1 | **Router GEMV + softmax + top-10 (+ shared-expert gate logit) in one kernel**; even a standalone DPP top-k recovers most of it | 2–3 ×48 = 96–144 | **1.0–1.3** | last-arriving WG (finalize = 2 KB) | high (root cause is clear, §5) | `topk_softmax_kernels.cu:278-600`, `fused_moe/gfx908_topk.py`, `router/fused_topk_router.py:60-85`, `router/gate_linear.py:157-200` (tier 5 → bf16 wvSplitK + fp32 cast) |
| 2 | **Shared expert as expert #512 of the routed W4 GEMV** (same per-rank shape K=2560,N=160), weight = sigmoid(gate) from #1; removes gate_up/down partials, 2 reduces, `+ shared` add | 5 ×48 = 240 | **0.8–1.0** | none | high (kernel already takes row lists; needs weight repack + topk=11) | `gfx908_moe_hip.py:145-190`, `gfx908_w4gemv.hip`, `layers/gfx908_shared_expert.py`, `qwen2_moe.py:117`, `runner/moe_runner.py:780` (`shared_output + fused_output`) |
| 3 | **GDN decode chain: conv1d_update + recurrent + gated RMSNorm (+ zero_, z copy) → one kernel**; conv is per-channel, norm is per-head (128) so both are local to the recurrent kernel's head tile | 5 ×36 = 180 | **0.5–0.6** | none | med-high (AITER has the pieces: `gdr_decode_packed_bf16.cu`, `fused_split_gdr_update.cu`, Triton `fused_rearrange_sigmoid_gated_delta_rule`) | `qwen_gdn_linear_attn.py:1195-1257` (generic path taken because `gqa_interleaved_layout=False`), `:1833` (`layer_norm_fwd`), aiter `csrc/kernels/gdr_decode_packed_bf16.cu` |
| 4 | **All-reduce + HC combine_norm fused** (AR epilogue does the 4-stream residual combine + GemmaRMSNorm); mirrors AITER `fused_allreduce_rmsnorm` which already runs on gfx908 via the `fuse_allreduce_rms` pass | 95 | **0.35–0.45** | flag polling (already in the AR) | med (new HIP kernel variant in aiter `custom_all_reduce.cu`) | aiter `csrc/kernels/custom_all_reduce.cu:215,566`, vllm `compilation/passes/fusion/allreduce_rms_fusion.py:1568`, `ops/hc.py:_hc_combine_norm_kernel` |
| 5 | **Merge GDN `in_proj_qkvz` + `in_proj_ba` into one GEMV** (concat weights at load; ba is a 0.5 MB latency-floor kernel) | 36 | **0.3–0.35** | none | high (weight-loading change only) | `qwen_gdn_linear_attn.py:861-864`, weight loader for `in_proj_ba` |
| 6 | **Kill the 96 AR staging memcpys**: producer (out_proj GEMV / MoE reduce) writes straight into the pre-registered uncached AR buffer, or allocate those outputs from an uncached (fine-grained) pool so `registered=True` is coherent on gfx908 | 96 | **0.25–0.3** | none | med (the gfx908 IPC-coherence bug forced `registered=False`; curvedinf's int8-vllm has the uncached-pool fix) | `distributed/device_communicators/custom_all_reduce.py:441-460`, `csrc/libtorch_stable/custom_all_reduce.cu:84-86` |
| 7 | **QSA glue**: (a) compress_groups + k-norm + rope + 3× store_rows → 1; (b) q-norm + rope → 1 (indexer) ; (c) topKPerRow + expand_indices → 1; (d) merge_splitk + sigmoid-gate → o_proj partial prologue; (e) main q/k GemmaRMSNorm + mRoPE + gate split via `fused_qk_rmsnorm_rope_gate` (check it is actually taken on gfx908 — the map's `triton_red_fused_5/poi_4/poi_6` + `_triton_mrope_forward ×24` say the eager path runs) | ~9 ×12 = 108 | **0.4–0.5** | none | med | `indexer_qsa.py:160-330`, `ops/qsa.py:401,438,117,631-760`, `qwen3_next.py:384-440`, `qsa.py:383-420` |
| 8 | **HC chain (combine_norm → mix_down+silu → mix_up+gate_mix) as one persistent kernel** with two counter stages and weight prefetch across stages — the megakernel pilot | 2 ×97 = 194 | 0.3 (boundaries) + 0.4–0.6 (latency overlap) | dataflow counters (2 stages) | low-med until the barrier microbench (§2) is done | `gfx908_hc_fused.py`, `csrc/gfx908_wv_fused.hip`, `ops/hc.py` |
| 9 | Router bf16→fp32 cast (`GateLinear` tier 5 `.to(out_dtype)`) — folded into #1; standalone: emit fp32 from wvSplitK | 48 | 0.1–0.15 | none | high | `router/gate_linear.py:195-200` |
| 10 | GDN `core_attn_out.zero_()` + QSA `output.zero_()` fills: make the kernels write every row (padded rows included) instead of pre-zeroing | 48 | 0.1 | none | high | `qwen_gdn_linear_attn.py:1249`, `qsa.py:139` |
| 11 | PLE short-conv decode (1 layer): index_select / where ×4 / cat / conv1d / silu / index_copy → one Triton "dilated conv update" kernel (same shape as `causal_conv1d_update`) | ~25 | 0.05–0.1 | none | high | `ple_layer.py:780-846` |
| 12 | Inductor knobs (§3): combo kernels for q-norm ∥ k-norm, aggressive fusion inside islands | ~20–40 | 0.03–0.06 | — | low value | `config/compilation.py:983-994` |

Sum of #1–#7 and #9–#11 (no megakernel machinery): ~500 launches and **~3.5–4.5 ms/token
optimistic, 2.5–3 ms realistic** out of ~14 ms → c=1 59.6 → ~70–75 tok/s. #8 (and the full
megakernel, §2) is on top of that, at much higher engineering cost. Rule from the campaign log still
applies: a fusion ships only after ≥3 in-server c=1 probes, not a microbench (the split-K finalize
that was 25 % faster in isolation and 4 % slower in the server).

Per-layer picture after #1–#7 (GDN+MoE layer, c=1): combine_norm(+AR) → wv_fused ×2 → in_proj →
GDN-fused → out_proj → AR+combine_norm → wv_fused ×2 → router+topk → w4gemv ×2 (+reduces) → AR:
~14 launches vs ~33 today.

## 2. Persistent per-layer megakernel on gfx908 — feasibility verdict

### What the prior art actually does

- **Hazy "No Bubbles" (Llama-1B, H100)**: one persistent kernel, grid = #SMs, normal launch. Per-SM
  instruction streams scheduled ahead of time in Python. Dependencies via an array of integer
  counters in global memory: an instruction atomically increments its counter on completion, a
  dependent instruction spins until the counter reaches the expected value — **no grid barrier**.
  Weight prefetch across instruction boundaries via 13 × 16 KiB shared-memory pages. 7 instruction
  types for Llama-1B. Cites 1.3 µs per launch inside CUDA graphs (same as our number) and gets
  2.5× vs vLLM. The TP follow-up ("We bought the whole GPU", Llama-70B TP8) adds peer-memory
  stores from dedicated storer warps and replaces reduce-scatter by a distributed transpose; 9
  instruction types, +22 % over SGLang.
- **Mirage Persistent Kernel (MPK, arXiv 2512.22219)**: compiler + in-kernel runtime. SMs split
  into workers (one task queue each) and scheduler warps (event queues). Tasks/events are circular
  buffers in device memory driven by `atomicAdd`; a task completing triggers an event whose counter
  reaching its target enqueues dependents. In-kernel scheduler costs 0.28 % of runtime; Qwen3-8B
  A100 decode 14.5 → 12.5 ms. CUDA/NVSHMEM only, no AMD support.
- **TensorRT-LLM / FlashDecoding++**: not megakernels; per-op fusions (split routing kernels,
  fused decode GEMM epilogues, fused attention reductions). Their MoE routing does one block per
  token with warp cooperation; iterative argmax-with-masking for E=256.
- **AITER (gfx942/950)**: asm `fmoe` / `fmoe_2stages` .co kernels (`hsa/gfx942/`), persistent
  stage-2 grids, fused `allreduce_rmsnorm_N8192.co`. All asm is gfx942/950-only
  (`aiter/fused_moe.py:937` refuses other arches). The portable pieces are HIP:
  `csrc/kernels/custom_all_reduce.cu` (flag-polling AR, with a gfx908 graph-capture fix at :81-97),
  `gdr_decode_packed_bf16.cu` (GDN decode with `ds_swizzle` reductions), `fused_split_gdr_update.cu`.
- **Already persistent in our stack**: every bf16 GEMV is a `wvSplitK` launch with
  `dim3 grid(CuCount)` and a "top level loop that makes WGs persistent"
  (`skinny_gemms.cu:418,1208`); the custom AR is a flag-polling kernel (`RankSignals`). A layer
  megakernel is "these persistent kernels concatenated with counters between them".

### The sync primitive on gfx908

1. **`hipLaunchCooperativeKernel` / `grid.sync()` — do not use.** ROCm device-libs
   `ockl/src/cg.cl`: `__ockl_grid_sync()` uses the hardware GWS barrier (`ds_gws_barrier`) unless
   `AVOID_GWS()` (gfx90a, gfx950, gfx11+), which use `single_grid_sync()` on an `mg_info` atomic.
   gfx908 therefore takes the GWS path, which needs the cooperative queue with GWS allocation
   — the "order of magnitude slower even with one block" report in ROCm#3410 (gfx906, same
   generation) is consistent with that dispatch path. Whether a cooperative launch can be captured
   into a hipGraph on ROCm 7 is undocumented. `hipDeviceAttributeCooperativeLaunch` is reported
   true, but that only means the API exists.
2. **Normal launch + resident grid + software counters — this works and is what wvSplitK/Hazy do.**
   Grid ≤ 120 × (WGs/CU guaranteed by the kernel's VGPR/LDS budget); one WG per CU is the safe
   default. Counters: `__hip_atomic_fetch_add(p, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT)` on
   completion; waiters spin on `__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)`
   (agent-scope loads bypass the write-through L1, so no manual `buffer_wbinvl1`) with
   `__builtin_amdgcn_s_sleep(1)` in the loop. All in-kernel data handoff goes through the 8 MB L2,
   which is coherent device-wide on the single-die MI100 (no XCD partitioning).
3. **Cost model.** A full 120-WG barrier = 120 same-address atomics serialised in the L2 atomic
   unit (~1–2 µs) + one L2 round-trip poll after the last arrival (~0.5–1 µs) ≈ **2–3 µs, i.e.
   about 2× a kernel boundary (1.33 µs)**. Therefore a barrier-per-op megakernel *loses* on
   gfx908. Wins only come from (a) fine-grained counters — a consumer WG waits on the few producer
   WGs whose tiles it needs, not the grid; (b) prefetching the next instruction's weights into
   VGPRs/LDS while waiting (the ~4–5 µs bubble per latency-floor GEMV, ~1.8 ms/token); (c) dropping
   the pre-zero / staging copies that exist only because ops are separate kernels. Hazy hides the
   counter cost because its instructions are ≥100 µs; ours are 5–25 µs, so the counter path must be
   ≤1 µs (one atomic + one poll) to pay.
4. **Register/LDS budget is the real constraint, not the sync.** 64 KiB LDS (vs 228 KiB on H100) and
   512 VGPRs at 1 WG/CU; occupancy of the megakernel = min over its instructions. wvSplitK-class
   GEMV tiles and the W4 GEMV split-K (8–40 × 64-wide blocks) want many WGs for memory-level
   parallelism, so instructions must loop over tiles internally (MPK/Hazy style). No PDL
   (`launch_pdl` is CUDA-only) means the megakernel is the *only* way to overlap consecutive ops.
5. **All-reduce inside the megakernel** is natural: the custom AR is already a flag-polling
   instruction with per-rank signal buffers; the TP peers' pointers are known at capture time.

### Verdict

Feasible in principle (normal launch, resident grid, agent-scope atomic counters), **not as a
barrier-synchronised kernel** (barrier ≈ 2× a launch), and not before the cheaper fusions in §1
that need no cross-WG sync at all. Recommended staging:

1. Stage 1 (no cross-WG sync): #1, #2, #5, #6, #9, #10, #11.
2. Stage 2 (single-kernel fusions, last-arriving-WG finalize only where the finalize is ≤ 2 µs):
   #3, #4, #7.
3. Stage 3 (pilot): a 120-WG counter-barrier microbench (measure the 2–3 µs claim), then the HC
   chain (#8) as a 3-stage persistent kernel with per-stage prefetch of the 655 KB `mix_up`
   weight (5.5 KB/CU fits in VGPRs). If #8 does not beat the 3-launch chain by ≥ 4 µs/HC in the
   server, stop here.
4. Stage 4 (months): per-layer megakernel with a Python-side static schedule (Hazy model, not MPK's
   in-kernel scheduler — c ≤ 48 decode is static enough) covering the ~14 instruction types left
   after Stage 2. Ceiling ≈ boundaries (~1.5 ms) + GEMV bubbles (~1.8 ms) ≈ 3 ms/token on top of
   Stages 1–2.

## 3. torch.compile / Inductor on ROCm gfx908

Why glue still lands in separate `triton_poi_*` kernels: every `torch.ops.vllm.*` op registered via
`direct_register_custom_op` (all HC ops, the QSA/GDN cores, the gfx908 GEMV dispatch, the HIP MoE
path) is opaque to Inductor. A pointwise op between two custom ops cannot be fused into either, so
each becomes its own kernel: the router `.to(float32)`, `shared_output + fused_output`
(`moe_runner.py:780`), `flat_output * torch.sigmoid(gate)` (`qsa.py:418`), the RMSNormGated native
forward (one `per` reduction + one `poi` normalize/gate/cast per GDN layer), `torch.cat` in
`apply_qsa_rope`, and fills. Combo kernels (already on: `config/compilation.py:983-994` sets
`combo_kernels=True, benchmark_combo_kernel=True` for torch ≥ 2.9) only fuse *independent* kernels
inside one island, so they can merge q-norm ∥ k-norm, not much else here.

Knobs worth a try (all via `compilation_config.inductor_compile_config`; names verified against
torch 2.10 `_inductor/config.py`):

| knob | default | effect here |
|---|---|---|
| `combo_kernel_allow_mixed_sizes` | 1 | 2 = allow mixed-shape sub-kernels (q vs k norm, 6 vs 2 heads) |
| `combo_kernels_autotune` | 1 | 2 = also autotune the combined kernel; `combo_kernel_max_num_args` 250 |
| `aggressive_fusion` | False | fuse nodes with weaker memory-savings score (tiny islands, cheap) |
| `max_fusion_size` / `realize_opcount_threshold` | 64 / 30 | raise so long pointwise chains stay one kernel |
| `score_fusion_memory_threshold` | 10 | lower; the islands are register-resident at M=1 |
| `coordinate_descent_tuning` | False | picks smaller XBLOCK/num_warps for M=1 reductions (10240-wide rows) — small but free |
| `triton.persistent_reductions` | True | keep (single-WG row reductions) |
| `triton.unique_kernel_names` | env | set `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` so `triton_poi_fused_0/4/6` become attributable |
| `cpp_wrapper` | False | cuts host launch overhead for the *uncaptured* shapes only; irrelevant inside FULL graphs |
| `enable_auto_functionalized_v2` | vLLM sets | required for in-place custom ops (`mutates_args`) to not force clones; already handled by `FixFunctionalizationPass` |

Do not expect more than #12 in §1 from knobs: the graph is chopped into ~150 tiny islands by custom
ops, and Inductor cannot cross them. The lever is moving glue *inside* the custom-op boundaries
(#1–#7), which we can do at model level because we own `models/qwen4_exp/amd/*` and the gfx908 MoE
path — no pattern-matcher pass needed for those. A vLLM pass is only worth writing for shared layers
we do not own: `Gfx908ARCombineNormPass(VllmPatternMatcherPass)` matching
`all_reduce → qwen4_exp_hc_combine_norm` the way `RocmAiterAllReduceFusionPass`
(`allreduce_rms_fusion.py:1568`) matches `all_reduce → rms_norm`; register under a new
`pass_config` flag in `passes/pass_manager.py:143-227`. Sequence-parallel / async-TP passes
(`sequence_parallelism.py`, `collective_fusion.py`) do not apply: `Qwen4ExpDecoderLayer` raises on
`use_sequence_parallel_moe`, and at M=1 there is nothing to shard.

## 4. The 187 `copyBuffer` per token

`__amd_rocclr_copyBuffer` is the SDMA/blit path for `hipMemcpyAsync` D2D, i.e. `tensor.copy_()`
between two *contiguous* same-dtype tensors (non-contiguous copies show up as
`at::native::*elementwise*` instead). Attribution:

| count | source | fix |
|---|---|---|
| 96 | custom AR staging: `custom_all_reduce.py:448-458` forces `registered=False` on gfx908 → `custom_all_reduce.cu:84-86` `cudaMemcpyAsync(reg_buffer, inp, …)` before every AR (2/layer) | #6: producer writes into `buffer_ptrs[rank]` directly (pass `out=` to the GEMV/MoE reduce; in-graph addresses are static), or allocate the AR inputs from an uncached fine-grained pool so `registered=True` is coherent (curvedinf int8-vllm fix). Also subsumed by #4 (AR reads peer outputs directly). |
| 36 | GDN `z_out[:] = z` in `_forward_core_rocm` (`qwen_gdn_linear_attn.py:1251`) — the generic path runs because `gqa_interleaved_layout=False` skips `_forward_core_decode_aiter` | #3: fused GDN kernel reads `z` straight from the `qkvz` projection; or pass the `z` view to the norm instead of copying |
| ~48 | QSA, ~4/layer: `q_gate.view(...).chunk` + `reshape` (interleaved q/gate per head → materialised), k/v `.contiguous()` for `reshape_and_cache_flash`, `cos_sin_cache[positions]` gathers, `output.copy_` paths in the indexer | #7e (`fused_qk_rmsnorm_rope_gate` emits contiguous q/k/gate directly) and #7a |
| ~7 | per step: `hidden_states.repeat(1, hc)` at embed, final mixer, sampler, MTP buffer | ignore |

Confirm the split with `torch.profiler(with_stack=True)` filtered on `copyBuffer` (or rocprofv3
with `--kernel-trace` + Python call stacks) before building #6; the AR share is certain from the
code path, the QSA share is inferred.

## 5. Top-k routing: why 24 µs and what to build

**Root cause.** `topkGating<VPT=8, EXPERTS=512, WARPS_PER_TB=4, BYTES_PER_LDG=16, WARP_SIZE=64>`
(`topk_softmax_kernels.cu:598-660`, `LAUNCH_TOPK(512, …)` at :707) puts one 64-lane wave on each
token row (THREADS_PER_ROW = 512/8 = 64). The k loop (:500-580) does, per selected expert, 6
butterfly stages × 3 `VLLM_SHFL_XOR_SYNC_WIDTH` (value, value-for-choice, expert index) — on
AMD each is a `ds_bpermute` LDS round trip of ~100+ cycles and they are dependent: 10 × 18 ≈ 180
shuffles + 12 for the softmax max/sum ≈ 20k cycles ≈ 13 µs at 1.5 GHz, plus the renormalise loop
and the launch. That matches the 21 µs measured in-graph. The Triton `gfx908_topk.py` version is
the same latency chain in a different form (10 sequential `tl.max` + `tl.min` block reductions via
LDS + barriers, ~2 µs each → 21 µs, "neutral in-graph"). Neither is bandwidth- or ALU-bound.

**Design A — standalone, one wave per token, ~2–3 µs real (vs ~21).**
- Each lane holds 8 fp32 logits (`global_load_dwordx4` ×2 from the fp32 router output).
- Softmax max/sum: DPP row reductions (`row_shr:1,2,4,8` + `row_bcast:15` + `row_bcast:31`,
  all available on gfx908 per the ISA inventory; `__builtin_amdgcn_update_dpp`), 6 steps of
  ~8 cycles instead of 6 `ds_bpermute`.
- Top-10: 10 passes of {lane-local max over 8 (7 `v_max`), wave argmax via 6 DPP steps on a packed
  (value bits, 511 − index) 64-bit key so ties resolve to the lowest index, winning lane masks its
  slot}. ≈ 10 × ~120 cycles ≈ 1 µs. Renormalise from the 10 selected values in registers.
- Handle `is_padding` (write −1 ids) and `renormalize` exactly like the stock kernel (unit test
  against `ops.topk_softmax` bit-for-bit, then logprob parity per the standing rule).
- Alternative for larger M: two-pass radix select on 8-bit digits with LDS histograms — same cost
  class, more code; not needed at M ≤ 48.

**Design B — fused into the router GEMV (recommended, #1).** Router weight is replicated
(`ReplicatedLinear`, 2560×512 bf16 = 2.6 MB/rank) and today runs as a wvSplitK launch (7–15 µs)
followed by an fp32 cast kernel and the top-k. New kernel: grid = 120 WGs, WG *i* computes full-K
dot products for output columns {i, i+120, …} (~4–5 columns × 5 KB of weight each, prefetched to
VGPRs; the 5 KB activation row is L2-resident), writes fp32 logits, `atomicAdd` on a per-launch
counter; the last-arriving WG (one wave) reads the 512 logits (2 KB, L2) and runs Design A, writes
`topk_weights`/`topk_ids`. Add `shared_expert_gate.weight` (2560×1) as column 512 and emit
`sigmoid(logit_512)` as an 11th routing weight for #2. Finalize is ~1–2 µs on 2 KB, so the
"serialised finalize delays the dependent kernel" failure of the split-K experiment does not apply.
Expected: router + cast + top-k ≈ 7 + 3 + 21 (+ 3 boundaries) ≈ 35 µs → ~6 µs per layer,
≈ 1.3 ms/token at c=1; at c=16–48 the finalize becomes one wave per token, still ≤ 3 µs.
Keep the counter reset in-kernel (last WG zeroes it) so the kernel is cudagraph-replay safe.

## 6. References

Papers / blogs
- Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (2025-05-27),
  https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles ; code
  https://github.com/HazyResearch/Megakernels (per-SM instruction streams, global-memory counters,
  16 KiB shared-memory paging, 1.3 µs launch gap in CUDA graphs).
- Hazy Research, "We Bought the Whole GPU…" (TP Llama-70B megakernel, 2025-09-28),
  https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main (peer-memory stores from storer
  warps, distributed transpose instead of reduce-scatter, 9 instruction types).
- MPK: "A Compiler and Runtime for Mega-Kernelizing Tensor Programs", arXiv 2512.22219,
  https://arxiv.org/abs/2512.22219 ; code https://github.com/mirage-project/mirage/tree/mpk
  (worker/scheduler SMs, atomicAdd circular queues, Qwen3-8B A100 14.5 → 12.5 ms/token, CUDA only).
- PyTorch RFC "enablement of combo-kernels", https://github.com/pytorch/pytorch/issues/170268 ;
  knobs in `torch/_inductor/config.py` (`combo_kernels`, `benchmark_combo_kernel`,
  `combo_kernels_autotune`, `combo_kernel_allow_mixed_sizes`, `aggressive_fusion`, …).
- ROCm device-libs cooperative-groups implementation,
  https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/ockl/src/cg.cl
  (`__ockl_grid_sync` → `__ockl_gws_barrier` except `AVOID_GWS()` arches gfx90a/gfx950/gfx11+).
- ROCm#3410 "hipLaunchCooperativeKernel slowdown", https://github.com/ROCm/ROCm/issues/3410
  (10× slower than a normal launch even with one block, gfx906, unresolved).
- HIP cooperative groups docs,
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html

Local code (vLLM fork `/home/tyler/vllm-gfx908`, AITER `/home/tyler/aiter`)
- Launch map: `docs/mi100_decode_opt/research/_profile_c1_launch_map.txt`; campaign log
  `docs/mi100_decode_opt/qwen38_flash_next_gfx908.md`.
- Layer structure: `vllm/models/qwen4_exp/amd/model.py:180-330` (`Qwen4ExpDecoderLayer.forward`),
  `hyperconnection.py` (`GatedResidual.mix/combine_and_mix`), `ops/hc.py` (Triton HC kernels),
  `gfx908_hc_fused.py` + `csrc/gfx908_wv_fused.hip` (wvSplitK copy with silu / gate-mix epilogues).
- GDN: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (`forward_hip` :853,
  `_forward_core_rocm` :1195, `_forward_core_decode_non_spec` :1259+, `_rms_norm_gated_cuda` :1833).
- QSA: `vllm/models/qwen4_exp/amd/{qsa.py (:383 forward, :117 forward_qsa), indexer_qsa.py
  (:160 project_qk, :245 _update_and_compress, :292 forward), ops/qsa.py}`;
  `vllm/model_executor/models/qwen3_next.py:384` (`_project_qkv_gate`).
- MoE: `fused_moe/gfx908_moe_hip.py`, `fused_moe/csrc/gfx908_w4gemv.hip`, `fused_moe/gfx908_topk.py`,
  `fused_moe/router/{fused_topk_router.py,gate_linear.py}`, `fused_moe/experts/triton_moe.py:588-720`,
  `fused_moe/runner/moe_runner.py:583-610,780`, `layers/gfx908_shared_expert.py`,
  `csrc/libtorch_stable/moe/topk_softmax_kernels.cu`.
- GEMV / AR: `csrc/rocm/skinny_gemms.cu` (wvSplitK, persistent `grid(CuCount)`),
  `vllm/model_executor/layers/utils.py:606` (`rocm_unquantized_gemm_gfx908_impl`),
  `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py:173-300`,
  `vllm/distributed/device_communicators/custom_all_reduce.py:405-470`,
  `csrc/libtorch_stable/custom_all_reduce.cu:66-90`.
- Compile: `vllm/config/compilation.py:960-1000`, `vllm/compilation/passes/pass_manager.py:143-227`,
  `vllm/compilation/passes/fusion/{allreduce_rms_fusion.py,rocm_aiter_fusion.py}`,
  `vllm/platforms/rocm.py:1010-1105` (gfx908 compile defaults, `fuse_allreduce_rms` on).
- AITER portable HIP pieces: `csrc/kernels/custom_all_reduce.cu` (fused AR+RMSNorm, gfx908
  capture fix :81-97), `csrc/kernels/gdr_decode_packed_bf16.cu`, `csrc/kernels/fused_split_gdr_update.cu`,
  `csrc/kernels/topk_softmax_kernels.cu`; gfx942-only asm in `hsa/gfx942/` (`fmoe*`, `allreduce_rmsnorm_N8192.co`).
- gfx908 ISA inventory: `~/.claude/projects/-home-tyler-aiter/memory/reference_gfx908_isa.md`
  (DPP row ops, `ds_swizzle`, `global_atomic_add_f32`, 64 KiB LDS, no packed fp32, no PDL).
