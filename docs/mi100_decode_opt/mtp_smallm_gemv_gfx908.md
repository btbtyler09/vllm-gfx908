# Small-M verify projections on gfx908 (MTP workstream item #2)

Branch `gfx908-mtp-smallm` (base `gfx908-mtp-verify` = the rc10 Qwen3.8-Flash-Next serving tree).
Status 2026-09-30: **code + CPU gates done, not yet run on a GPU.** Everything below marked
*estimate* is a model, not a measurement; the GPU recipe at the end produces the real numbers.

## TL;DR

- The M=2 "per extra verify row" cost of the four kernel groups breaks down very differently:
  | group | +ms/step at M=2 (profiled) | what it really is | worth |
  |---|---|---|---|
  | HC W8 GEMVs (+ the injection copy it drags in) | +0.95 (+0.37 copy) | latency-chain / serialisation, NOT bytes: the weights are already read once for all rows | **yes, first** |
  | bf16 wvSplitK / Cijk | +0.62 | wvSplitK +0.25 (in_proj_ba x36, QSA 12, final mixer); Cijk +0.37 = 516-token prefill amortised over 68 vs 128 steps plus the unquantised MTP draft layer | small; config lever exists |
  | W8 GEMV -> MFMA swap | net +0.16 | the MFMA kernel at M=2 is *cheaper* than the GEMV at M=1 (-0.05 ms/step); the +0.18 is the draft pass's own extra 62080-row lm_head (171 us) | **no threshold change** |
  | W4A8 MoE | +0.49 | slab gate_up +0.23 and rowlane +0.09 are at the unique-expert byte floor; +0.13 is the bf16 Triton MoE of the MTP draft layer | last; nothing kernel-side |
- Implemented: `VLLM_GFX908_SMALLM_HC_W8=1` (default **off**): two new kernels in a **separate**
  extension (`gfx908_hc_w8_smallm_ext`) that serve the W8 hyper-connection chain at 2 <= M <= 4 as
  ONE global round trip each, use every lane, and write the injection compactly, so the chain is 2
  launches at every M (today 3 at M >= 2). The shipping M = 1 kernels and their code object are untouched.
- CPU gates: bit-faithful numpy emulation of both the new and the shipping kernels (same tiling, same
  summation order, same DPP trees): new vs shipping **0 bf16-ulp differences** on all outputs at
  M = 1..4 (the fp32 pre-rounding sums differ in 77% of elements at ~1e-7 relative, so the check is
  live), both inside the agents/hc_w8 tolerance vs fp64 (worst 0.14 / 0.23 of tol), injection
  bit-identical; hipcc gfx908 build clean, no spills, all loads issued before the first wait.
- Expected (*estimate*): HC chain at M=2 from 26.4 us (graph-cold, shipping incl. copy) to ~14-17 us
  per module -> **-0.7 to -1.1 ms per K=1 spec step** (97 modules). If the bench confirms the new
  kernels also beat the shipping ones at M = 1, `VLLM_GFX908_SMALLM_HC_W8_MIN_M=1` is a plain-decode
  lever too (*estimate* up to -0.5 ms/step) -- that needs its own A/B.

## 1. Where the verify step's extra row goes (K1 vs off, rank-0 traces)

Source: `/home/tyler/work/q38fn_mtp_retest/prof/{off,K1}` (off = 128 plain steps, K1 = 68 spec steps,
c=1, rc10). Per-launch medians are profiler-inflated (~+2.4 us on small kernels, see
feedback_profiler_inflation_gfx908); graph-cold microbench numbers are quoted where they exist.

| kernel (shape per rank) | off: /step, median us | K1: /step, median us | ms/step off -> K1 |
|---|---|---|---|
| hc_w8 mix_down 336x10240 int8 gs128 `<2,2,NB,..,64,2,1>` | 95, 9.6 | 98, 15.8 | 0.93 -> 1.44 |
| hc_w8 mix_up 10240x320 int8 gs64 `<4,1,NB,..,32,1,2>` | 95, 8.5 | 98, 12.2 | 0.81 -> 1.18 |
| injection `.contiguous()` copy (elementwise direct copy after mix_down) | 0 | 98, 3.8 | 0 -> 0.38 |
| wvSplitK `<bf16,64,1,16,8,4,N>` in_proj_ba 24x2560 bf16 | 37, 4.0 | 39, 8.2 | 0.17 -> 0.33 |
| wvSplitK `<..,2,..,2,N>` (QSA, 12 layers + MTP) | 15, 7.0 | 17, 8.6 | 0.17 -> 0.17 |
| wvSplitK `<..,4,..,2,2>` final-mixer mix_up (bf16 stock chain) | 0 | 3, 20.5 | 0 -> 0.10 |
| Cijk (rocBLAS) | prefill-only | prefill + MTP layer | 0.34 -> 0.71 |
| w8sw_gemv qkvz / out_proj (M=1) | 36+36, 14.2 / 9.8 | 0 | 0.88 -> 0 |
| w8mfma `<16,1,1,1>` qkvz / out_proj (M=2) | ~0 | 73, 14.7 / 8.2 | 0 -> 0.84 |
| lm_head: w8sw `<2,1,1,..>` M=1 / w8mfma `<16,1,2,1>` M=2 | 1 x 172 | 1.1 x 172 (draft) + 1 x 172 (verify) | 0.17 -> 0.36 |
| w4a8_slab_prep_gate (MoE gate_up) | 48, 11.5 | 48, 16.3 | 0.56 -> 0.79 |
| w4a8_rowlane (MoE down) | 48, 5.4 | 48, 7.2 | 0.27 -> 0.35 |
| fused_moe_kernel (bf16 MTP draft experts, Triton) | 0 | 2.1, 45 | 0 -> 0.13 |

Graph-cold microbench of the shipping HC W8 chain (agents/moe_prep_hc/bench_hc_cross.log, 2026-09-04):

| M | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| mix_down us | 10.42 | 13.12 | 14.45 | 18.74 |
| mix_up us | 9.73 | 12.31 | 14.80 | 17.23 |
| chain us (incl. injection copy) | 19.52 | 26.38 | 30.61 | 36.90 |

### 1.1 HC W8 -- why the extra row costs ~5 us per module

Bytes are not the problem: each kernel streams 3.55 / 3.49 MB (int8 + fp32 group scales), a
3.6 us floor at ~985 GB/s, and the NB loop already applies every weight tile to all M rows. The
M = 1 kernels already run at ~45% of the byte floor; the slope comes from structure:

1. **Serialised round trips.** Every one of the 120 workgroups (also the ~36 that own no rows) first
   stages the whole `[M, K]` activation into LDS, `__syncthreads`, and only then issues its first
   weight load. The X round trip (+ the barrier) and the weight round trip are in series, and the
   staged bytes grow with M: 120 x M x 20 KB of L2 reads for mix_down (2.4 MB at M=1, 4.9 MB at
   M=2, 9.8 MB at M=4 -- at ~3 TB/s of L2 that alone is 0.8 / 1.6 / 3.3 us).
2. **mix_down runs 4 waves per CU.** 336 rows / (YTILE 2 x KS 2) -> 84 active workgroups, 4
   streaming waves each (1 per SIMD), and each wave runs 3 dependent load -> compute iterations
   (the next load is issued after the current compute). Nothing hides the per-row FMA work, so it
   lands on the critical path: ~36 -> ~84 VALU instructions per 16 weights per lane from M=1 to M=2
   (at M=2 the LDS mode also flips from fp32 to bf16, `2*10240*4 B > 62 KB`, adding unpacks).
3. **mix_up uses 62.5% of its lanes** (K = 320 = 20 of 32 lanes per half-wave) and its gate-mix
   epilogue issues the XN loads after the reduction (a `sched_barrier` pins them there): one more L2
   round trip at the end, per row.
4. **The injection copy.** At M >= 2 the `[M, 4]` slice of the `[M, 336]` down output is no longer
   contiguous, so `_hc_mix_local` pays a `.contiguous()` launch: 98 launches/step (97 + the MTP layer), ~1.3 us of graph
   dispatch + ~1.5 us body each.

### 1.2 bf16 wvSplitK -- small, and a config lever already exists

- in_proj_ba is 24 x 2560 bf16 (123 KB): 24 waves on the whole GPU, pure latency. Its M=2 cost is
  +~0.16 ms/step profiled (upper bound; profiler-inflated). The existing
  `VLLM_GFX908_GDN_MERGED_PROJ=1` folds it into the qkvz W8 weight, i.e. into the MFMA kernel that
  is M-flat at M=2 -> it removes 36 launches/step at M=1 *and* M=2. Not bit-exact (int8 gs128 on the
  b/a rows, ~1.2% rel L2 on 24 columns) -> needs the PPL/GSM8K gate. Recommended as the A/B, not a
  new kernel.
- The Cijk +0.37 is not a verify-step cost: the 516-token prefill is amortised over 68 instead of
  128 steps, plus the unquantised MTP draft layer (`VLLM_GFX908_MTP_QUANT` exists).
- The final mixer runs the stock bf16 chain (wvSplitK + hc_silu + wvSplitK + gate_mix); moving it to
  the W8 fused path is worth ~20-30 us/step. Low priority.

### 1.3 W8 GEMV -> MFMA dispatch at M = 2: keep it

agents/mfma_gemv measured the MFMA kernel "nearly M-independent below 16 rows" (in_proj_qkvz
13.90 / 14.17 / 14.86 / 14.98 us at M = 1..4) and 0.88-0.95x of the swizzled GEMV at M = 2..4,
which scales ~linearly. The traces agree: at M=2 the MFMA path costs 0.84 ms/step for 73 launches
vs 0.88 ms for 72 GEMV launches at M=1. The "+0.16" is entirely the **draft pass's extra lm_head**
(62080 x 2560 int8 per rank = 159 MB, 171.5 us, once per draft token). Keeping the GEMV at M <= 4
would be *slower*; no threshold change was made. The lm_head per draft token is its own lever
(vocab-pruned draft head / reusing the verify logits), outside this item.

### 1.4 W4A8 MoE -- at the byte floor

At M=2 a token pair touches ~2x the unique experts: gate_up grows 11.5 -> 16.3 us (+4.8 us; the
extra ~10 expert slices of ~0.45 MB are +4.5 us at 985 GB/s) and rowlane 5.4 -> 7.2 us (+1.8 vs a
+2.3 us floor). The expert-major attempt (smallm_gemv, 2026-09-05) was negative for exactly this
reason. The remaining +0.13 is the bf16 Triton MoE of the MTP draft layer (config lever:
`VLLM_GFX908_MTP_QUANT`). Nothing to do here kernel-side.

## 2. What was implemented

`vllm/models/qwen4_exp/amd/csrc/gfx908_hc_w8_smallm.hip` -> extension `gfx908_hc_w8_smallm_ext`
(own code object, own build dir `$VLLM_GFX908_HIP_BUILD_DIR/w8smallm`, added to
`tools/prebuild_gfx908_exts.py`). Loaded only under the flag.

### hcsm_down<NB, YT, 1> (mix_down, EPI silu on lora rows)
- Workgroup = YT output rows x the full K; 10 waves (K = 10240 = 10 x 1024), wave `ks` owns the
  1024-k step `ks`, lane `l` owns k = 1024 ks + 16 l for all YT rows (one activation read feeds YT rows).
- Every load is issued up front, before any wait: YT weight dwordx4 (nontemporal), YT fp32 group
  scales, 2 NB activation dwordx4 **straight from global into registers** -- no LDS staging, no
  barrier before compute. Verified in the gfx908 ISA: for every instantiated config all
  `2 YT + 2 NB` loads precede the first `s_waitcnt vmcnt`.
- Per lane: fp32 FMA chain over its 16 k (k ascending), `sum = fma(t, scale, 0)`; DPP wave reduction
  (same tree as `hw8_wave_reduce<64>`); the 10 wave partials through LDS in fixed order; epilogue
  identical to EPI 1; rows [R, R+HC) are also written to a compact `[M, HC]` injection output.
- Grid 336 / YT: YT=3 -> 112 workgroups <= 120 CUs (default); YT=2 -> 168 (2 fit per CU at <= 67
  VGPRs); YT=4 -> 84 (least X re-read traffic). `VLLM_GFX908_SMALLM_HC_W8_DOWN_YT` selects.

### hcsm_up<NB> (mix_up over the row-permuted weight, gate-mix epilogue)
- A unit is (hidden index i, 16-k chunk c): one thread loads chunk c of the 4 permuted rows 4i..4i+3
  (one X read feeds 4 rows). A workgroup holds IPB x CPR units with IPB x CPR % 64 == 0: K = 320 ->
  CPR = 20, IPB = 16, 320 threads, **100% of lanes busy** (shipping: 62.5%). 160 workgroups.
- The tiny activation (M x 640 B) is loaded first, then weights, scales and the gate-phase XN value;
  the activation is staged to LDS as fp32 while the weight stream is still in flight (the staging
  waits only `vmcnt(#weight loads)`), one barrier, per-unit partials
  `fma(t, scale, 0)` to LDS, one barrier, thread (i, n, s) sums its 20 chunk partials in order, the
  quad s = 0..3 is gathered with `quad_perm` DPP and lane s = 0 applies the shipping gate-mix
  (`acc = fma(sigmoid(bf16(g_s)), xn_s, acc)`, s ascending, `/ HC`).

### Resources (hipcc 7.2.4, `-O3 --offload-arch=gfx908`, `-Rpass-analysis=kernel-resource-usage`)

| kernel | VGPR | AGPR | spill | waves/SIMD | LDS B |
|---|---|---|---|---|---|
| hcsm_down<1,{2,3,4},1> | 23 / 34 / 40 | 0 | 0 | 10 / 7 / 6 | 80-160 |
| hcsm_down<2,{2,3,4},1> | 40 / 47 / 58 | 0 | 0 | 6 / 5 / 4 | 160-320 |
| hcsm_down<3,{2,3,4},1> | 52 / 60 / 70 | 0 | 0 | 4 / 4 / 3 | 240-480 |
| hcsm_down<4,{2,3,4},1> | 67 / 79 / 81 | 0 | 0 | 3 | 320-640 |
| hcsm_up<1..4> | 39 / 57 / 60 / 57 | 0 | 0 | 6 / 4 / 4 / 4 | dynamic 6.4-25.6 KB |

All 16 kernels issue every global load before their first `s_waitcnt vmcnt` (checked in the
`-S` output). Three codegen traps were found and fixed on the way, each of which silently
re-serialised the round trips: (a) a use of the loaded registers inside `if (k < K)` let LLVM sink
the loads into that block, below the first wait -- the compute is now branch-free with zeroed
operands; (b) 64-bit `size_t` address arithmetic cost a VGPR pair per load and pushed NB >= 3 into
AGPR parking with one-load-at-a-time waits; 32-bit offsets let the compiler use the SGPR-base form
(host checks keep every offset < 2^31); (c) in `hcsm_up` the activation loads were sunk into the
`if (tid < K/8)` staging block, i.e. behind the whole weight stream -- an empty `asm volatile`
anchor after the weight loads keeps them in the entry block, issued first. `sched_barrier` alone
does not prevent (a) or (c): the sinking happens in the IR, before machine scheduling.

### Python / flags (`gfx908_hc_fused.py`)
- `VLLM_GFX908_SMALLM_HC_W8=1` (default 0). Requires `VLLM_GFX908_HC_W8=1`; under
  `VLLM_GFX908_STRICT_EXT=1` (default) a missing HC W8 path, a failed build, or a runtime decline
  **raises** instead of falling back silently. The flag is resolved at weight-load time and logs
  `gfx908: HC W8 small-M kernels ACTIVE for 2 <= M <= 4 (...)` at boot, and
  `gfx908: HC W8 small-M chain served M=2` on first use -- gate every arm on these lines.
- `VLLM_GFX908_SMALLM_HC_W8_MIN_M` / `_MAX_M` (default 2 / 4, clamp 1..4): the M range.
  `MIN_M=1` puts plain decode on the new kernels too (A/B only).
- `VLLM_GFX908_SMALLM_HC_W8_DOWN_YT` = 2 | 3 | 4 (default 3).
- W4 HC (`VLLM_GFX908_W4_LOADTIME=hc|all`) keeps precedence; per-channel scales (gs = 0) keep the
  shipping path.

## 3. Numerics and CPU tests

`tests/gfx908/test_hc_w8_smallm_ref.py` (numpy only) emulates, with the exact tiling and summation
order, the new kernels and the shipping configs they replace (`_CFG_DOWN` M=1,2: KS 2 UNRL 2;
M=3: KS 4 UNRL 3; M=4: KS 1 chunked; `_CFG_UP`: LPR 32). int8 x bf16 products are exact in fp32, so
`t + w*x` in fp32 is bit-identical to the device FMA; the DPP trees are replayed lane by lane. Data:
N(0, 0.02) weights with the agents/hc_w8 outlier pattern (1% of input channels x20, 0.1% of entries
x50), RTN int8 with the production group sizes, mix_up permuted before quantisation, activations
N(0,1) with 8 large columns, and the up input taken from the emulated down output (real chain).

| M | new vs fp64 (<= 1 passes): down / up | new vs shipping: max ulp, % elements | shipping vs fp64: down / up | injection == down cols |
|---|---|---|---|---|
| 1 | 0.124 / 0.226 | 0, 0.000% (both) | 0.124 / 0.226 | yes |
| 2 | 0.136 / 0.221 | 0, 0.000% (both) | 0.136 / 0.221 | yes |
| 3 | 0.136 / 0.219 | 0, 0.000% (both) | 0.136 / 0.219 | yes |
| 4 | 0.137 / 0.234 | 0, 0.000% (both) | 0.137 / 0.234 | yes |

The orders are genuinely different (77% of the fp32 pre-epilogue sums differ, max 9.3e-8
relative) but no bf16 output flipped on these draws; the test asserts <= 1 ulp and <= 1% of
elements, which is the honest contract (not bit-exact). A 1% error in one scale group is caught at
1.4x the output magnitude. Emulation limits: `__expf` replayed with float32 `np.exp`, and the
shipping kernel's multi-chunk `fma(t, scale, sum)` replayed in float64 (double rounding) -- both
shared by the two emulations, so they do not affect the order comparison.

`tests/gfx908/check_hc_w8_smallm_loader.py` (CPU container, no device): builds both W8 extensions
through the module's own loaders (hipcc, gfx908), then checks the flag logic (off / on / needs
HC_W8 -> raises under STRICT, warns otherwise), the env ranges, and that unsupported shapes /
configs decline before anything touches the device. All pass.

## 4. Expected numbers (*estimates*, graph-cold per launch, one MI100)

Model: ~1.4 us graph dispatch + ~1 us to first byte + 3.5-3.6 us byte stream + ~1 us tail, with the
per-row VALU work (~0.3-0.45 us per row when spread over all 480 SIMDs) mostly hidden under the
stream; mix_down at M >= 3 may become L2-bound on activation re-reads (112 x M x 20 KB).

| M | shipping down / up / chain (measured 09-04) | hcsm_down (est.) | hcsm_up (est.) | new chain (est.) | saving per module |
|---|---|---|---|---|---|
| 1 | 10.4 / 9.7 / 19.5 | 6.5-8 | 6-7.5 | 12.5-15.5 | 4-7 us |
| 2 | 13.1 / 12.3 / 26.4 | 7-8.5 | 6.5-8 | 13.5-16.5 | 10-13 us |
| 3 | 14.5 / 14.8 / 30.6 | 7.5-9 | 7-8.5 | 14.5-17.5 | 13-16 us |
| 4 | 18.7 / 17.2 / 36.9 | 8-9.5 | 7.5-9 | 15.5-18.5 | 18-21 us |

Per spec step, 97 HC modules on each rank's critical path: K=1 (M=2) **-0.7 to -1.1 ms**
(conservative end = new chain only matches today's M=1 chain), K=2 (M=3) -1.0 to -1.5 ms,
K=3 (M=4) -1.4 to -2.0 ms. In the profiled K1 anatomy that is most of the +1.3 ms (HC W8 +0.95 plus
the copy +0.37).

## 5. GPU validation recipe (tonight, ONE card, then the server A/B)

Pre-flight: the relabel job must be finished and its containers gone (check `docker ps`,
`rocm-smi --showuse`, power -- not just the low-power warning). One chain script, no pgrep waiters.

### 5.1 Microbench + device correctness (one card, ~3 min)
```bash
W=/home/tyler/vllm-gfx908-mtp-smallm; S=/home/tyler/work/mtp_smallm; mkdir -p $S
docker run --rm --name hcsm-mb --group-add=video --ipc=host --device=/dev/kfd \
  --device=/dev/dri/renderD128 --env HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  -v $W:/src:ro -v $S:/work btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn \
  python3 /src/tests/gfx908/bench_hc_w8_smallm.py --out /work/hcsm_bench.json 2>&1 | tee $S/hcsm_bench.log
```
Gates (exit code 0 = all pass), per M = 1..4:
- `inj_vs_emulator` max ulp **== 0** (proves the device index math is the emulated one);
  `down_vs_emulator`, `y_vs_emulator` <= 1 ulp (only `v_exp_f32` vs `np.exp`);
- new vs shipping `block_input` / injection <= 1 ulp on <= 1% of elements, for YT = 2, 3, 4.
Timing to read: `ship_chain` vs `new_chain_yt3` (the saving per module) and the YT sweep; pick the
YT default from it. Expected per the table in section 4. If `new_down` at M = 3/4 is well above
the estimate while M = 2 is on it, the activation re-read (L2) is the limit -> YT = 4 or the
cross-workgroup K split listed under follow-ups.

### 5.2 In-server A/B (4 cards, K=1, same boot pattern as q38fn_mtp_retest)
Overlay the two changed files onto the rc10 image (the image's copies are byte-identical to this
branch's base, checked):
```bash
P=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
OV="-v $W/vllm/models/qwen4_exp/amd/gfx908_hc_fused.py:$P/gfx908_hc_fused.py:ro \
    -v $W/vllm/models/qwen4_exp/amd/csrc/gfx908_hc_w8_smallm.hip:$P/csrc/gfx908_hc_w8_smallm.hip:ro"
cd /home/tyler/work/q38fn_mtp_retest
SPEC='--speculative-config {"method":"qwen4_exp_mtp","num_speculative_tokens":1,"draft_sample_method":"probabilistic"} --max-num-seqs 8'
# arm K1-base: EXTRA_ENV="$OV --env VLLM_GFX908_GDN_FUSED_SPEC=1"
# arm K1-sm  : EXTRA_ENV="$OV --env VLLM_GFX908_GDN_FUSED_SPEC=1 --env VLLM_GFX908_SMALLM_HC_W8=1"
```
(`serve_mtp.sh` passes `EXTRA_ENV` straight into `docker run`, so the `-v` overlays ride along.)
Per boot: (1) gate on the artifact: `docker logs | grep "HC W8 small-M kernels ACTIVE"` must be
present in K1-sm and absent in K1-base, and "`served M=2`" must appear after the first request;
(2) greedy coherence line (`screen.py`, T=0) vs K1-base -- hybrid+spec is not bit-exact, compare to
the off-vs-off self-floor recorded in RESULTS.md (2/8 exact, token-sim 0.59/0.47); (3) `screen.py`
step ms and tok/s; (4) one torch-profiler window as in `prof_window.py`: expect the two
`hc_w8_gemv_kernel<...,2,...>` variants replaced by `hcsm_down_kernel<2,3,1>` /
`hcsm_up_kernel<2>` (98/step each), the 98/step elementwise copy after mix_down gone, and HC
compute ms/step down by ~0.7-1.1 ms. Alternate arms (base, sm, base, sm) -- the expected effect is
above the ~0.3-0.4 ms boot scatter but only by ~2x. (5) Before any "ship": the parser-agnostic
GSM8K gate on the winning arm, and a 12-tier suite per the no-corner-cutting rule.
Optional third arm: `VLLM_GFX908_SMALLM_HC_W8_MIN_M=1` on the plain (no-spec) server -- only if 5.1
shows the new kernels win at M = 1.

## 6. Risks

- **Nothing here has run on a GPU.** The CPU emulation proves the math and the order the kernels are
  *meant* to execute; 5.1's emulator check (injection bit-exact) is what proves the device code does.
- Performance is modelled. The two biggest unknowns: L2 bandwidth for the activation re-reads of
  mix_down at M >= 3 (same traffic as the shipping kernel's staging, but now on the critical path
  of every wave), and whether 10-wave workgroups with 47-81 VGPRs get the latency hiding assumed.
- Not bit-exact vs the shipping chain (different fp32 summation order): 0 flips in the emulation at
  these draws, but occasional 1-ulp bf16 flips are expected on real activations; spec decode on this
  hybrid is already not bit-exact, so the gate is coherence + GSM8K, not token identity.
- Code-object size: the new kernels live in their own extension and are not linked into the hot
  M = 1 module; the only change on the default path is one cached-bool check per HC call (in the
  opaque custom op, i.e. at capture / eager time, not per graph replay).
- `hcsm_up` hard-requires HC = 4 and 16 | K; `hcsm_down` requires ceil(K/1024) <= 10. Other shapes
  decline -> raise under STRICT (they cannot occur for this model).

## 7. Follow-ups (not started)

1. If M >= 3 is L2-bound: split mix_down's K across 2 workgroups with a deterministic last-arriving
   finalize (as `router_topk_fused` does) to halve the activation re-reads.
2. `VLLM_GFX908_GDN_MERGED_PROJ=1` in the K1 A/B (removes in_proj_ba's bf16 launch at every M; PPL/GSM8K gate).
3. Draft-pass lm_head (171 us per draft token): vocab-pruned draft head or logits reuse.
4. Final mixer on the W8 fused path (~20-30 us/step).

## Files

| file | what |
|---|---|
| `vllm/models/qwen4_exp/amd/csrc/gfx908_hc_w8_smallm.hip` | kernels + torch binding (`hcsm_down`, `hcsm_up`) |
| `vllm/models/qwen4_exp/amd/gfx908_hc_fused.py` | loader `_ext_w8_smallm`, flags, dispatch in `_hc_mix_local` |
| `tools/prebuild_gfx908_exts.py` | prebuilds `_ext_w8_smallm` into the image |
| `tests/gfx908/test_hc_w8_smallm_ref.py` | CPU numerics gate (numpy emulation of new + shipping kernels) |
| `tests/gfx908/check_hc_w8_smallm_loader.py` | CPU build + flag / decline checks |
| `tests/gfx908/bench_hc_w8_smallm.py` | one-GPU device correctness + graph-timed microbench |
