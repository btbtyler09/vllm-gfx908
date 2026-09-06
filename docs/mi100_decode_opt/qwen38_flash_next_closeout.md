# Qwen3.8-Flash-Next on 4x MI100: campaign close-out (2026-09-02 .. 2026-09-06)

Final release: **rc9** (`btbtyler09/vllm-rocm-gfx908:v0.28.0rc9.dev-q38fn`;
numbers filled in below when its gates and 12-tier land). Model: qwen4_exp
180B MoE, GPTQ 4-bit experts + QSA, bf16 elsewhere in the artifact (int8 at
load for HC mixes, GDN projections, lm_head). TP=4 over xGMI, torch.compile +
HIP graphs (FULL_AND_PIECEWISE), V2 model runner. The artifact's quantization
was never changed: Tyler keeps quality first, so the calibrated 4-bit
re-quantization of HC/GDN (est. +6-7% at c=1) was declined.

## Result

| | bring-up (09-02) | rc7 (09-05) | rc8 (09-06) | rc9 final |
|---|---|---|---|---|
| c=1 decode (tok/s) | 17.5 | 100.3 | 105.7 | TBD |
| single-user TPOT | ~57 ms | 10.32 ms | 9.56 ms | TBD |
| step timer (ms/step) | ~57 | 9.57-9.61 | 9.17-9.20 | TBD |
| 16K c=4 (tok/s) | - | 132.9 | 141.2 | TBD |
| c=16 / c=64 | - | 532 / 582 | 544 / 582 | TBD |
| 290 W halo c=1 / c=64 | - | - | 107.3 / 619 | TBD |
| GSM8K (1319) / PPL | 1278 / 3.145 (rc5) | 1282 / 3.1451 | 1275 / 3.1407 | TBD |

Per-rank bytes per decode token: 1.91 GB (HC W8 636 MB, GDN int8 519,
experts W4 365, lm_head 159, router bf16 126, QSA W4 78). The step is
~900 graph nodes; GEMVs run at 78-97% of size-matched floors; the plain-decode
floor on this GPU is ~8.5 ms/step. What is left is spec decode (needs a better
drafter) and the artifact-level quantization that was declined.

## Levers that shipped (c=1 unless noted; measured in-server, step timer)

| lever | flag | effect |
|---|---|---|
| PLE zero-copy pinned-host gather + in-graph AR | VLLM_PLE_ZEROCOPY | removed a ~2.5 ms/token host chain (round 4) |
| W4A8 dot4 MoE GEMV (slab), prep fold, shared expert folded | VLLM_GFX908_W4A8, _MOE_PREP_FOLD | MoE decode at 78-97% of byte floor |
| fused router GEMV + softmax + top-k | VLLM_GFX908_ROUTER_FUSED | -4 launches/layer |
| W8A16 for GDN qkvz/out_proj + lm_head (M<=4), MFMA W8A16 5..64, swizzled M=1 GEMV | VLLM_GFX908_W8A16, _W8A16_MFMA | int8 traffic, no bf16 copy (KV pool back to 313k tokens) |
| GDN fused decode glue (conv1d + recurrence + gated norm + z copy) | VLLM_GFX908_GDN_FUSED | 5 -> 1 launches x 36 layers |
| HC mixes W8 with fused silu / gate-mix epilogues, fused range M<=4 | VLLM_GFX908_HC_W8, _HC_FUSED_MAX_M | 636 MB/token at int8, 3 launches/module |
| fp16 expert GEMMs for prefill / mid-M | VLLM_GFX908_MOE_FP16_COMPUTE | MI100 fp16 MFMA = 2x bf16 rate; 16K TTFT |
| MoE multi-row kernel (24 < M <= 256) | VLLM_GFX908_MOE_MR | c=16..64 tiers |
| exact radix sampler top-k/top-p (<=64), merged passes | VLLM_GFX908_SAMPLER_FASTK | 7 -> 6 launches, no selection replay |
| QSA decode glue (norms, MRoPE, cache writes, indexer, split-K reduce) | VLLM_GFX908_QSA_GLUE | 45 -> 25 launches/layer (M=1) |
| push all-reduce over xGMI (sentinel slots) | VLLM_GFX908_PUSH_AR | 5.3 vs 7.8 us per AR |
| HC-AR consumer fused into the HC combine (split kernels, counters at model build) | VLLM_GFX908_HC_AR_FUSED | -0.25 ms/step |
| W4A8 bf16 epilogue on the dense QSA GEMVs | VLLM_GFX908_W4A8_BF16_EPILOGUE | -36 cast launches |
| PLE decode glue (23 -> 1 launches; splitting op; compiled fallback body) | VLLM_GFX908_PLE_GLUE | -22 launches (within boot noise) |
| fused push-AR producer (GDN out_proj, QSA o_proj push from the epilogue) | VLLM_GFX908_PUSH_AR_FUSED_PRODUCER | -0.05 ms/step (rc9) |
| strict extension loaders, hashed build dirs, off-GPU prebuild gate, boot banner | VLLM_GFX908_STRICT_EXT | no silent fallbacks |

## Levers tried and rejected (with the number that killed them)

megakernel (barrier 1.3 us but in-kernel waits idle CU slots); fork/join
(+15 us/section); HC norm folded into the mix_down prologue (+0.2 ms: 120-WG
GEMV replicates the norm); reduced-expert self-draft; small-M GEMV
amortization; larger MoE K tiles; max-num-seqs 96 (KV halves); logits
captured into the decode graph (wash); GDN in_proj merge (boot crash, not
root-caused); calibrated 4-bit HC/GDN (declined on accuracy risk); MTP n=2
(parity only: verify step 2.2-2.5x a plain step, acceptance 0.53-0.59).

## Lessons (each is a memory in the assistant's notes)

- Throughput probes cannot see wrong output; every boot prints a greedy-parity
  line before its numbers count.
- A custom op that replaces a vLLM splitting op must be registered as one, or
  the piecewise graph captures per-step temporaries and faults on replay.
- Persistent kernel state must never be allocated inside a graph capture; a
  "coherent" config can be winning a memory-layout lottery.
- torch's extension loader can hand back a stale .so after a failed rebuild:
  content-hashed build dirs + an off-GPU compile gate before any boot.
- Code-object size is a per-dispatch cost; boot-to-boot noise is 0.3-0.4
  ms/step (three boots or in-process A/B); the profiler inflates tiny kernels.
- Debug multi-rank / graph-capture kernels in a standalone 4-GPU harness first;
  server boots are for gates.

## Where things live

Fork `btbtyler09/vllm-gfx908` branch `qwen38-flash-next` (65 gfx908 files,
~18k lines, 41 kernels); reports, start scripts and the kernel-by-kernel step
map in `btbtyler09/mi100-llm-testing`; this campaign's per-round log in
`qwen38_flash_next_gfx908.md`. Next, non-GPU: reusable kernels into the aiter
fork as a gfx908 op library; a thin MI100 serving project on upstream vLLM via
platform/model plugins.
