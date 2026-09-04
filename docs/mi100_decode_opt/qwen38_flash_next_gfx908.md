# Qwen3.8-Flash-Next (qwen4_exp, 180B) on 4×MI100 — bring-up + perf campaign (2026-09-01/02)

Branch `qwen38-flash-next`. Artifact: `Qwen3.8-Flash-Next-GPTQ-4bit` (W4 GS32 sym body,
bf16 PLE table as 128 shard tensors, bf16 GDN projections).

## Canonical serve

```
docker run -d --name vllm-q38fn --network=host --cpuset-cpus="0-11" --group-add=video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
  --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 \
  --env HSA_OVERRIDE_GFX_VERSION=9.0.8 --env HF_HOME=/huggingface \
  --env VLLM_ROCM_USE_AITER=1 --env VLLM_ROCM_USE_AITER_CUSTOM_AR=0 --env VLLM_MI100_TORCH_COMPILE=1 \
  --env VLLM_PLE_MMAP=1 --env VLLM_PLE_MMAP_PINNED=1 --env VLLM_PLE_MMAP_PREWARM=1 \
  --env VLLM_DISABLED_KERNELS=ConchLinearKernel \
  -v /home/tyler/.cache/huggingface:/huggingface -v /mnt/slow-storage:/mnt/slow-storage:ro \
  btbtyler09/vllm-rocm-gfx908:v0.28.0rc1.dev-q38fn \
  vllm serve /mnt/slow-storage/quant/Qwen3.8-Flash-Next-GPTQ-4bit --served-model-name qwen38-flash-next \
  --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 --max-num-seqs 48 --hf-overrides '{"language_model_only": true}'
```

- `VLLM_PLE_MMAP=1 + PINNED`: the 102 GB n-gram table stays in host RAM (page cache + pinned
  staging rows prepared in `prepare_inputs`), which is what makes FULL cudagraphs legal.
- `--max-num-seqs 48`: FULL graphs need one Mamba cache block per sequence (53 available at
  0.92 util). Side effect: KV pool 313k tokens.
- `--dtype bfloat16` is required (QSA). First bf16 model served on gfx908; Triton `tl.dot`
  bf16 is exact on gfx908 (checked).
- MTP is off by decision.

## Accuracy

| stack | wikitext-2 PPL (64 win, seq 2048, stride 512) |
|---|---|
| HF BF16 | 3.1206 |
| HF W4 (same artifact) | 3.1386 |
| served, before fix | 7.2883 |
| served, after `004b1ca779` | **3.1362** |

Root cause: the AMD QSA sparse paged attention Triton kernel tiles `BLOCK_M = next_pow2(Q heads
per KV head)`; 24 Q / 2 KV heads at TP4 → 6 per rank → `BLOCK_M=8`, which miscompiles on gfx908
(rel err 0.8–0.98 for every group ≤ 8; 12/24 exact). TP1/TP2 run group 12 and were exact. Fix:
clamp `BLOCK_M ≥ 16` on ROCm. Found by per-layer logprob parity against transformers on a 4-layer
rehearsal artifact (TP1 0.015 nats, TP4 0.21 diverging at the first QSA layer, fixed 0.017).

GSM8K (500Q, thinking mode, temp 0.6, c=16) on the final stack: **490/500 = 98.0%**.

Lesson (now a standing rule): every new kernel path gets unit tests + logprob parity vs HF at
production TP before any perf work; "coherent output" is not evidence.

## Performance levers (c=1 decode, TP4, FULL graphs)

| commit | change | c=1 tok/s |
|---|---|---|
| baseline | FULL graphs + pinned table | 17.5 |
| `1b2ae92f3d` | `shared_expert_gate` Linear(2560→1): rocBLAS picks a 508 µs Tensile kernel on gfx908 for any M → einsum (39 µs). Was 45% of the step. | 31.6 |
| `0b24060152` | Triton W4A16 GEMV for M≤16 (MFMA kernel launched 5–56 programs, latency-bound; gate_up 75→8 µs, QSA qkv 84→16, o_proj 45→10); wvSplitK before LLMM1 (wins at n=1 on every shape); tuned MoE config `E=512,N=160` for MI100 (neutral at M=1, 2.2× at M=24–32, 1.6× at 48–512) | 37.7 |
| `164f71d485` | Triton split-K bf16 GEMM for 5≤M≤64 where rocBLAS is 2–7× off (HC mix_down 117 µs → 16–34, router 69 → 7–15, QSA indexer 69 → 8–17, GDN in_proj 70 → 33–37) | (batched) |

Batched decode, aggregate tok/s (200-token completions, distinct prompts), before → after
`164f71d485`: c=8 107 → 136, c=16 158 → 263, c=48 232 → 486.

Profile of the 32 ms step after the first fix (rank 0, 64 decode tokens, ~2,500 launches/token,
GPU ~100% busy): W4 dense GEMMs 26%, W4 MoE 13%, custom all-reduce 11% (97/token × 37 µs),
bf16 GDN projections 11% (1.04 GB/token/rank at ~30% of HBM BW), ~1,700 tiny glue kernels 28%.

## Rounds 2–3 (2026-09-02 afternoon/evening): 37.7 → 59.6 tok/s at c=1

| commit | change | c=1 tok/s |
|---|---|---|
| `71d3402a4d` | W4A16 GEMM dispatch as an opaque custom op. The `M ≤ 16` GEMV branch ran under dynamo tracing with a symbolic M and was specialized at trace time (large M), so the 12 QSA layers' qkv/o_proj ran the MFMA kernel with a large-M tile config at M=1 (162/98 µs). | 46.3 |
| `5d4d45abd0` | Fused small-M shared-expert MLP: two GEMV partial kernels + reduce·silu·mul + reduce·sigmoid(x·w_gate) (10 launches → 4). | 49.2 |
| `701f1f41e4` | QSA indexer GemmaRMSNorms as single Triton launches (were ~7 eager launches each inside the custom op). | 51.2 |
| `cfac8d0d97` | HIP W4A16 GEMV MoE path for M ≤ 8 (`fused_moe/csrc/gfx908_w4gemv.hip`, JIT via `torch.utils.cpp_extension`, prebuilt in the image) + fused Triton reduces: 57 vs 96 µs per layer at M=1. Kernel zero-fills for out-of-range expert ids (cudagraph-capture routing on dummy inputs). | 57.4 |
| `518e7d4394` | Cached HIP row-index tensors; Triton small-M top-k (neutral in-graph, default off). | 59.6 |

Lessons: (1) any Python shape dispatch inside the traced region is frozen at trace time — wrap it in a custom op whenever a kernel choice "ignores" M; (2) the dense projections (0.4–4.6 MB) sit at the ~8 µs launch/latency floor — a faster kernel can't help them, only fewer launches can; the MoE experts (6 MB/layer) were the one GEMM where a better kernel paid; (3) `VLLM_PLE_MMAP_SERIAL=4096` cuts the PLE host gather from 3–10 ms to 0.4 ms per step at c=16 (thread-pool dispatch overhead).

Validation: greedy per-token logprob fingerprints (16 GSM8K prompts × 160 tokens, temp 0, c=1 and c=16) differ from the round-1 image by 0.004–0.007 nats mean — the same-stack c=1-vs-c=16 noise floor. GSM8K at temp 0.6 varies ±2% run to run on this model (98.0 / 96.4 on identical-parity stacks), so it is a coarse gate here.

Round-3c per-token profile (c=1, 18.3 ms GPU, 2,135 launches): bf16 GEMMs 4.2 ms (23%), all-reduce 1.5–3.4 (jitter), MoE HIP 2.2, dense W4 GEMV 1.8, HC ops 1.4, eager glue 1.2, QSA 1.0, norms 0.9, copies 0.8, GDN 0.7.

Image: `btbtyler09/vllm-rocm-gfx908:v0.28.0rc2.dev-q38fn` (HIP ext at `/opt/vllm-gfx908-ext`; add `VLLM_PLE_MMAP_SERIAL=4096` to the serve env).

### Tried and rejected
- EP-within-TP (`--enable-expert-parallel`): c=1 −8%, c=48 −18% (untuned E=128,N=640). DP
  layouts don't fit (≥35 GB/GPU).
- Triton MoE GEMV for small M: slower than the stock kernel (Triton nibble-unpack GEMV codegen
  tops out ~80 GB/s on gfx908). A hand-written wave-level HIP kernel (exllama-class) is the
  remaining ~10–13% c=1 lever; vLLM's CUDA `moe_wna16.cu` uses PTX `lop3/prmt` + packed bf16 so
  it is a rewrite, not a hipify.
- MoE tuner gotchas on gfx908: `BLOCK_SIZE_K > group_size` faults the gptq_awq kernel (search
  restricted to 32); never set both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`; the ray
  tuner dies on the first GPU fault — use the crash-isolated driver in the job dir.

### Round 3 findings that did not ship
- **All-reduce is at its floor.** `ar_bench.py` (4 processes, graph-captured): vLLM custom AR 7.8 µs per 5 KB call, RCCL 19 µs. The 25–37 µs seen in the model profile is profiler inflation plus barrier accounting; all four ranks carry equal non-AR work and the profiled busy time (18.3 ms) exceeds the real step (16.8 ms). True AR cost ≈ 1 ms/token. Sanity-check profiler sums against wall time before attributing.
- **Fused split-K finalize (v3 HIP GEMV)**: last-arriving block sums partials and applies the epilogue (cast / silu·mul / routed accumulate). 25% faster in the microbench (43 vs 57 µs per MoE layer), **4% slower in the server** (57.0–57.7 vs 59.6–59.9 tok/s over 3 probes each) — the serialized reduction on one block delays the dependent kernel more than a separate parallel reduce. Dense 1-launch variant only ties the Triton 2-launch GEMV. Rule: fusion wins need ≥3 in-server c=1 probes before commit.
- Triton small-M top-k: neutral in-graph (21 vs 24 µs), default off.
- GPU clocks are pinned at max (sclk 1502 / mclk 1200 MHz); bf16 GEMMs run at 65–98% of HBM peak via wvSplitK, so only fewer bytes helps there.

### Open levers
- Quantize the bf16 GDN projections (Quantizer-side): ~3.5 ms/token at c=1.
- Hand-written W4 GEMV (dense + MoE).
- Fuse the ~35 glue kernels per layer (HC combine/mix/silu, MoE align/sum, fills/copies).
- All-reduce count (2 per layer, latency-bound at 37 µs).

## Round 4 (2026-09-03/04): the step was never GPU-bound — PLE zero-copy

**The profiler lied.** The torch profiler adds ~2.4 us to every tiny kernel on gfx908 (1.33 us
real inside a graph vs 3.75 us in the trace; a 2,000-node graph of trivial kernels replays in
2.66 ms). Worse, kernels made of many tiny programs inflate more: the GDN decode kernel reads
14.2 us in the trace and 1.8 us graph-timed with HBM-cold state. Every "launch floor" number in
round 3 was mostly artifact, which is why fusing clusters gave +1% each. Rule: microbench
(graph-timed, cold) before believing any per-kernel number under ~30 us; size host idle with
py-spy + step period, not with the kernel table.

**The real hole: ~2.5 ms/token of GPU idle at the step boundary.** py-spy on the worker showed
68% of the main thread blocked in the PLE `ids.to("cpu")` (i.e. waiting for the whole previous
step), then a serial chain: host memmap gather (0.2 ms) -> pinned stage -> H2D -> attention
metadata -> graph launch. With the GPU drained every step, nothing overlapped.

**Fix (`03869b6e62`, `VLLM_PLE_ZEROCOPY=1`, default in the serve script):** each rank
`hipHostRegister`s the PLE shard files it owns (`shard % 4 == rank`, 23.8 GiB each; the kernel
driver caps pinned system memory at ~307 GB per node so no rank can pin all 95 GiB), a HIP
kernel gathers the 16 x 320 B rows per token straight from the host page cache inside the
captured graph (zeros for shards it does not own), and one 5 KB TP all-reduce reassembles them
exactly. The n-gram ids are computed on the GPU in the same opaque op. The worker never waits on
the GPU, so steps queue back to back.

Two bugs this exposed, both fixed in the same commit:
- The V2 runner's host-mapped (UVA) input pools are a 2-deep round-robin ring, safe only while
  the GPU drains each step; queued steps let the host overwrite a slot before the GPU read it
  (non-deterministic outputs, 3/16 prompts identical between two runs). Zero-copy implies an
  8-deep ring (`VLLM_UVA_MAX_CONCURRENCY`).
- The padded tail of the PLE `query_start_loc` buffer held stale entries from earlier batches
  (`[0, 13, 2, 3, 4, 5]`), and `searchsorted` over a non-monotonic array mis-bucketed real
  prefill tokens (48/128 wrong ids on a 13-token prefill). The op takes a running max.

Same-tree A/B (200-token completions, distinct prompts):

| arm | c=1 | c=16 | c=48 |
|---|---|---|---|
| zero-copy off | 59.1-59.6 | 248.9 | 486.0 |
| zero-copy on | 64.1-64.5 | 252.8 | 558.4 |

Greedy per-token logprob parity vs the round-3 build: 0.0055 nats at c=1 (6/16 prompts
bit-identical), 0.0063 at c=16 — the same-stack noise floor. Debug aids stay env-gated:
`VLLM_PLE_ZEROCOPY_CHECK=1` (eager steps compare with the host gather),
`VLLM_PLE_ZEROCOPY_DEBUG_IDS=1` (every step compares captured ids with host-computed ids).

Overlay gotcha: rsync of the repo into the container must exclude `*.so` — the host tree carries
a stale `_C.abi3.so` that silently removes `wvSplitK` and every custom op.

Research reports (five agents, read-only): `research/*.md` — W4 GEMV redesign (current MoE
kernel at ~12% of HBM), int8 weights for the 2.89 GB/token of bf16 projections, push-based
all-reduce over XGMI, launch fusion / megakernel verdict, GDN/QSA Triton codegen (root cause of
the BLOCK_M=8 miscompile: a 4x64 broadcast-MFMA branch new in Triton 3.7). Ground-truth checks
so far: rocBLAS N=24 at 5<=M<=64 is 5.7 us (not 10-20), GDN BV=8 only helps B>=4 (bit-exact,
`VLLM_GDN_DECODE_BV`).

## Round 5 (2026-09-04): three kernels from the research reports

Kernels were designed from the research reports and built + microbenched by subagents (one GPU
each, graph-timed, HBM-cold), then integrated env-gated and gated in the server.

| lever | env | kernel result (per rank, M=1) | server |
|---|---|---|---|
| W4A8 int8-dot GEMV for routed experts, shared expert, dense qkv/o_proj (`6d6bf0ff2a`) | `VLLM_GFX908_W4A8=1` | gate_up 21.7 -> 8.9 us, down 11.7 -> 5.8 (0.8-3.5 us above a pure stream of the bytes); M=8: 201 -> 42, 51 -> 24 | c=1 65.9 -> 72.7-73.1; parity 0.0052 (7/16 identical); PPL 3.130; GSM8K c=8 97.0% |
| fused router GEMV + fp32 softmax + top-10 (one launch, MFMA 16x16x8 per wave, last-arriving WGs finalize, branch-free DPP top-k) | `VLLM_GFX908_ROUTER_FUSED=1` (+`_BF16=1` routes on bf16 logits exactly like stock) | 25.2 -> 9.2 us (stock = GEMV 5.9 + cast 1.8 + topkGating 17.6); ids bit-identical to `topk_softmax` on 180/180 tie/NaN/pad cases | see combined row |
| W8A16 group-128 int8 GEMV for the bf16 GDN in_proj_qkvz / out_proj and lm_head at M<=4 (int8 side copy built at first eager call, ~1.4 GB/rank duplicated for now) | `VLLM_GFX908_W8A16=1`, `_GS=128` | lm_head 356 -> 180 us, qkvz 23.7 -> 13.5, out_proj 10.7 -> 7.0; bit-exact vs a fp64-rounded reference; per-channel scales fail on outlier columns (5-17%), gs128 ~2% | see combined row |
| combined stack (all three) | | | c=1 77.4-77.5; parity vs the W4A8 build 0.0061 (6/16 c=1, 7/16 c=16); PPL 3.130 |

Why A8 and not fp8: gfx908 has no fp8, but it has `v_dot4_i32_i8` (4 int8 MACs per VALU op) and
`v_dot2_f32_f16`. The activation is quantized per token in blocks of 32 with an fp32 absmax scale
(llama.cpp Q8_1 layout, zero point folded into the block sums), so the int8 dot is exact and the only
added error is the block-32 rounding of the activation. An exact fp16-dot2 variant (bf16 -> fp16 cast
is bit-exact, 0x6400 magic dequant, `v_and_or_b32` via inline asm, LDS-staged activation) was built as
the fallback: 1.8-4.2x over stock with stock-identical accuracy, 6-41% slower than int8 (VALU-bound,
not bytes). It is wired as `VLLM_GFX908_W4A8_MODE=f16`.

Integration bug worth remembering: the fused router replaces the gate GEMM that lives in
`MoERunner._forward_impl`, while the top-k lives in `FusedTopKRouter._compute_routing`. Two
independent predicates (runner skips the GEMM; router fuses) drifted apart on the first boot — the
router could not find the gate and fell back to stock top-k on the placeholder zero logits — and the
model routed on zeros (0.51 nats, coherent-looking text). Now the runner hands the gate to the router
and the router recomputes the logits itself whenever it cannot fuse. Coherent output is not evidence.

Gate caveats: wikitext PPL is prefill-only (M >= 2048) and GSM8K at c=8 runs M=8, so neither exercises
the M <= 4 W8A16 path; the greedy c=1 parity does, and sits at the noise floor. GSM8K at c <= 4 is the
remaining check for W8A16. In-server throughput is only quotable with no microbench containers on
GPUs 0-2 (c=48 drops from 558 to ~350 under a single agent's graph-replay bench).

## Round 6 (2026-09-04, later): 77 -> 88 tok/s at c=1

| lever | env | result |
|---|---|---|
| exact fp16-dot2 W4A16 mode (`7d380678f7`) | `VLLM_GFX908_W4A8_MODE=f16` | fallback/A-B arm: stock-identical accuracy, 1.8-4.2x stock, 6-41% below int8 |
| W8A16 int8-only storage for the GDN projections | `VLLM_GFX908_W8A16_FREE=gdn` (default) | 36 layers 997 MB bf16 -> 552 MB int8 per rank; M>4 dequant +54 us/layer (+4.6% prefill); prefill now int8 -> PPL 3.130 -> 3.141 (HF W4 3.139) |
| fused GDN decode glue (5 launches -> 1) | `VLLM_GFX908_GDN_FUSED=1` | 13.4 -> 7.8 us/layer at M=1, -144 launches/token; conv state bit-exact |
| shared expert as expert #E (11 -> 7 launches per MoE layer) | `VLLM_GFX908_SHARED_AS_EXPERT=1` | 36 -> 24 us per MoE layer at M=1 |

Full stack, quiet GPUs: **c=1 87.7-87.8 tok/s** (17.5 at the start of the bring-up, 59.6 at the end
of round 3), c=48 545. Greedy parity vs the round-5 build 0.0083 (c=1) / 0.0070 (c=16); PPL 3.141;
GSM8K 500q c=8 97.0%. The c=16 probe is bimodal across boots (250 vs 380) and not yet understood.

Working method that got here: research agents (read-only, one bottleneck each) -> kernel agents
(one GPU each, graph-timed cold microbench + correctness vs an fp32 reference) -> integration
agents (env-gated, unit test in the container) -> the orchestrator's server gate (3 probes,
greedy parity at c=1/c=16, PPL, GSM8K at a concurrency that actually exercises the new path).
Two integration bugs were caught only by the parity gate (router on zero logits; UVA ring).

### Release rc3 (2026-09-04)

`btbtyler09/vllm-rocm-gfx908:v0.28.0rc3.dev-q38fn` = tree `e79c3b2f49` + every gfx908 HIP extension
prebuilt in `/opt/vllm-gfx908-ext` (`tools/prebuild_gfx908_exts.py`, `docker/Dockerfile.q38fn`) + the
validated env defaults baked in (`docker run ... env | grep GFX908`), so the canonical serve line
needs no flags beyond the base command in this doc's first section. Validation: boots in ~9 min
(no JIT), c=1 88.7-89.0 tok/s, greedy fingerprints bit-identical to the overlay build (16/16).

## Round 7 (2026-09-04, later): prefill

Profile of a 7840-token pass (torch profiler is fine for kernels this large): MoE W4 GEMM 19.5%,
RCCL all-reduce 18.9% (575 ms — the decode-era `NCCL_ALGO=Tree,PROTO=LL` pin applied to 40 MB
messages), QSA sparse + indexer 24%, rocBLAS bf16 GEMMs ~15% (hyper-connection mixes, replicated
on all ranks), GDN chunk ~4%. At 2K the custom 2-stage all-reduce takes 15% (10 MB messages at
~12 GB/s) and QSA 17%. Research: `research/prefill_gfx908.md`.

Shipped (commit `gfx908 prefill round 1`), same-tree A/B with `ttft_probe.sh`:

| prompt | before | after |
|---|---|---|
| 1,597 tokens | 535-551 ms | 439-451 ms (-18%) |
| 12,780 tokens (single request) | 3.50 s | 2.75 s (-21%) |

Levers: RCCL unpin + custom-AR cap at 2 MB; indexer scorer bounded to the batch's context
(4485 -> 360 us/layer at 2K); dense causal attention when every context fits the indexer budget
(the top-2048 selection is then the identity — exact; 12 layers 105 -> 19 ms at 2K); W4 large-M
dequant-to-rocBLAS escape (-8.8 ms/pass at 2K). Greedy parity vs rc3 16/16 bit-identical; PPL 3.145.

Still open for prefill: the MoE config for 8192-token chunks (borrows the 4096 entry, ~54 ms per
chunk), hyper-connection mixes sharded over TP with one all-gather (M-split, staged), the QSA
prefill kernel itself for contexts above the budget (per-token programs, 5.5x issued/useful), and
the 784-token "align" block that turns a 16K prompt into 3 passes (9 at c=4).

MTP (n=2) on this stack accepts 58% of draft tokens (2.16 tokens/step) but nets only +4% at c=1:
the spec step adds ~5 ms of uncaptured proposer glue, sampler cumsum/sorts over (n+1) rows and
copies. Parked; needs a captured proposer, a sampler fix and a W4 drafter. Loading required a fix
(`9fc17890e4`: the checkpoint's mtp.* weights are unquantized) and memory headroom
(`--max-num-batched-tokens 4096`, the spec profile run's logits peak).

### Release rc4 (2026-09-04, later)

`btbtyler09/vllm-rocm-gfx908:v0.28.0rc4.dev-q38fn` = tree `636b8047cd` (rc3 + W8 hyper-connection mixes
+ prefill round 1), extensions prebuilt, validated env baked (note `VLLM_GFX908_HC_W8_FREE=0`: freeing the
HC bf16 masters costs ~90 ms per 2K prefill through the M>3 dequant fallback). 12-tier vs rc3: single-user
TTFT 850 -> 663 ms (TPOT 11.2), 16K c=4 TTFT 12.1 -> 9.4 s, c=32/64/128 337/301/337 -> 360/338/379 tok/s.
Report + start script in mi100-llm-testing (`scripts/serve_qwen38_flash_next_rc4.sh`).

## Round 8 (2026-09-04/05): batched-decode range, determinism, and the step-timer rule

Ground truth for every entry here is the in-server step timer
(`VLLM_GFX908_STEP_TIMING=1`: CUDA events around execute_model, /200 steps)
on the same working-tree overlay, not microbenches — three of the four
levers below had microbench wins that did not survive the real op.

| arm (c=1, TP4, FULL graphs) | ms/step | c=1 tok/s | c=4 ms/step (tok/s) |
|---|---|---|---|
| first tree: control (new flags off, but NB4..8 kernels linked) | 11.10 | 89.4 | 20.7 (185.7) |
| first tree: `HC_FUSED_MAX_M=4` | 11.16 | 88.8 | 18.65 (198) |
| first tree: + `MOE_PREP_FOLD=1` | 11.25 | 88.3 | — |
| first tree: + `QSA_STABLE_TOPK=1` (v1) | 11.43 | 87.5 | — |
| **fixed tree** (HC M<=3 kernel byte-identical, NB>3 in own ext): `HC_FUSED_MAX_M=4` | **10.87** | **91.3** | 18.8 (207.5) |
| fixed tree: + `QSA_STABLE_TOPK=1` (v2) | 11.37 | 87.3 | 19.0 (203) |
| fixed tree: + `MOE_PREP_FOLD=1` | 11.24 | 88.4 | 18.7 (205.9) |

Caveat: a no-op-flag control pair on the fixed tree read 10.87 vs 11.24
ms/step, so the boot-to-boot floor is ~0.3-0.4 ms (3%), not the 0.05 ms the
200-step windows suggest. Deltas below 0.4 ms need repeated boots or an
in-process A/B before they count; the c=4 numbers (>1.5 ms deltas) and the HC
code-object fix are above that floor, the stable top-k cost is not yet.

* **HC fused range M<=4** (`VLLM_GFX908_HC_FUSED_MAX_M`, default 4): the
  fused bf16/W8 mix-chain kernels now cover M<=8 (K-chunked LDS staging);
  it only *pays* to M=4 because the kernels are GEMVs whose cost grows with
  M while rocBLAS goes flat on MFMA at M>=5. W8 chain at M=4: 53.7 -> 36.9
  us/module; c=4 +7-10% (185.7 -> 198 tok/s). Adopted.
  Two hidden M<=3 regressions found on the way and fixed: (a) the chunked
  loop kept 16 waves alive across per-chunk barriers (+2.6..4.2 us/module,
  ~0.3 ms/step over 96 modules) -> HEAD's kernel is compiled in byte-identical
  for M<=3; (b) merely *linking* the NB=4..8 kernels into the same code
  object cost +1.1..2.0 us per M<=3 dispatch -> they live in a separate
  extension (`gfx908_wv_fused_w8_big_ext`, `-DHW8_NB_BIG`) that is only
  JIT-built when the range is > 3. Lesson: code-object size is a
  per-dispatch cost on gfx908; keep hot GEMV kernels in small modules.
* **MoE prep fold** (`VLLM_GFX908_MOE_PREP_FOLD`, default off): folds the
  activation quantize/pack into the gate_up GEMV's LDS staging (5 -> 4
  launches per MoE layer). Real-op timing at M=1, warm and cold-L2 (weight
  rotation and 64 MB eviction): MoE layer -7..-13%, dense W4A8 GEMVs
  -19..-25%, bit-identical. But in the shipping config it is a no-op: with
  `SHARED_AS_EXPERT=1` the routed MoE takes the fused shared+routed path,
  which the fold does not cover, and the GDN projections are W8A16. The
  "+0.37 ms" first attributed to it was therefore boot-to-boot variance
  (see the caveat under the table). Being extended to the shared-as-expert
  path; expected value ~0.08 ms/step (48 layers x ~1.7 us).
* **Stable top-k** (`VLLM_GFX908_QSA_STABLE_TOPK`, **default on** since rc5):
  `topKPerRowDecode` resolves ties and slot order via atomicAdd, so long-context
  greedy output differed run-to-run at token 0 (3/4 prompts at 6K). A repair
  kernel re-sorts each row's k indices deterministically (short rows with
  visible<=k get the identity without reading logits); with a per-run
  `cache_salt` this gives 4/4 bit-identical ~6K outputs. Cost measured the
  right way (same process, whole select op captured in a graph, alternating
  arms): +1.14 us per QSA layer, 12 layers -> ~0.014 ms/step. Under eager
  dispatch the repair is instead fused into the expand kernel (+11.5 us vs
  +31 us of launcher time), chosen on `is_current_stream_capturing()`.
  Prefill: +0.2..1.4% of the select at 2K..8K rows. The "+0.2..0.5 ms"
  seen in single-boot arms was boot variance plus a stale overlay (the
  served copy was still v1) -- see the caveat above.
* **MFMA W8 GEMM 5<=M<=64** (`VLLM_GFX908_W8A16_MFMA`): swizzled int8 MFMA
  16x16x16 path for batched decode with a deterministic LDS reduce (the
  first version used `ds_add_f32` atomics and failed cudagraph replay
  parity). Numerics within bf16 half-ulp (a 3e-3 bound was the wrong test).
  c>=4 win, c=1 -0.5% and a small greedy-parity shift; default pending a
  same-overlay ablation.
* **W4 at load time for GDN/HC** (`VLLM_GFX908_W4_LOADTIME`): +0.5% at
  10x the RTN error of the GPTQ W4 experts. Kept in tree, off; the right
  version is Quantizer-side GPTQ W4 for those weights.
* **MoE Triton W4 config, M=8192 key** (`E=512,N=160 ... int4_w4a16.json`):
  BLOCK_M 256 / BLOCK_N 64 / BLOCK_K 32 / 8 warps, from a 112-config sweep.
  Real 7840-row prefill chunk 10.94 -> 9.22 ms per MoE layer per rank
  (-15.7%), i.e. -82 ms per 16K prefill pass; M<=4096 keys were already
  optimal and are unchanged. The dequant-to-bf16 MoE alternative was
  measured and loses (dequant round-trip 3.55 ms/layer alone exceeds the
  whole W4 call at M=2048; bf16 pair only 15% faster than the in-kernel
  nibble path). Same-FLOP dense rocBLAS is ~2x faster than the W4 MoE kernel
  at M=8192, so the remaining prefill headroom is the MoE GEMM structure
  (gather/padding/tiling), not the weight format.

### Release rc5 (2026-09-04, later): `btbtyler09/vllm-rocm-gfx908:v0.28.0rc5.dev-q38fn`

Tree 59255ea5cd, every extension prebuilt (incl. the new `w8big` and `w4lt`
modules). Validated on the pure image (no overlay), TP4, util 0.90:

| gate | rc5 | rc4 |
|---|---|---|
| c=1 step timer (1200-token decode windows) | 11.04 ms/step | 11.01 |
| c=1 / c=4 / c=16 / c=48 probes (tok/s) | 89.9 / 210 / 264 / 568 | 89 / 186 / — / — |
| greedy parity c=1 vs rc4 | 16/16 identical, 0.0000 | — |
| greedy parity c=16 vs rc4 / vs itself | 5/16, 0.0076 / 7/16, 0.0068 | (floor) |
| long-context determinism (4 x ~6K, two salts) | 4/4 bit-identical | 1/4 |
| TTFT 3.2K / 12.8K (probe, warm) | 558 ms / 1.81 s | 655 ms / 2.29 s |

c=16 parity has the same score against itself as against rc4: that is the
batched path's own run-to-run floor, not a change. c=1 is bit-identical to
rc4, and stable top-k now makes ~6K-context greedy output reproducible by
default. The c=1 step time is within the boot floor of rc4; the c=4 gain is
the HC range extension, the TTFT gain is the MoE M=8192 key plus the prefill
round already in rc4 measured warm.

12-tier BenchAndReport on the pure rc5 image (mixed real-text corpus,
`mi100-llm-testing/Model_Reports/benchmark_Qwen3.8-Flash-Next-GPTQ-4bit_rc5.md`):

| tier | rc4 | rc5 |
|---|---|---|
| Single user 2K/512 TTFT / TPOT | 663 ms / 11.24 ms | 541 ms / 11.43 ms |
| Decode stress 128/2048 c=1 | 89.6 tok/s, 11.04 ms | 89.2 tok/s, 11.08 ms |
| Short context c=16 | 251.6 tok/s, TTFT 2.35 s | 276.0 tok/s, TTFT 1.87 s |
| Long context 16K c=4 | 107.6 tok/s, TTFT 9.39 s, TPOT 23.8 | 115.8 tok/s, TTFT 9.02 s, TPOT 21.6 |
| Mixed c=8 | 260.6 tok/s | 264.9 tok/s |
| Concurrency c=4 / c=32 / c=64 / c=128 | 164 / 360 / 338 / 379 | 175 / 357 / 341 / 387 |

c=1 decode is unchanged within the boot floor; the gains are TTFT (-18%
single user), c=4..16 decode (+7..10%, HC range 4) and the 16K tier (+8%,
MoE M=8192 key + tiled QSA measured warm).

## Round 9 scoping (2026-09-04, night): megakernel = no; the GEMVs are latency-bound

Feasibility study on gfx908 (agents/megakernel/REPORT.md), all graph-captured:

| quantity | measured |
|---|---|
| graph node cost (empty kernel) | 1.52 us @ grid 1, 1.6 @ 120-220 WGs, 2.0 @ 480, 2.8 @ 960 |
| co-residency, 256-thread WGs | 3 blocks/CU at 69..100 VGPR (360 resident); 8 at <=25 VGPR |
| grid barrier (sense-reversing atomic + s_sleep) | 1.31 us @ 120 WGs, 1.36 @ 240, 2.31 @ 480, 5.65 @ 960 |
| per-row dataflow counter waits (440) | 0.3-0.9 us aggregate |
| bandwidth stolen by 60..360 spinning WGs | <= 1.3% (deadlock above residency) |
| MoE layer M=1 shared-as-expert int8, cold | 23.5 us, 5 nodes, 7.25 MiB -> byte floor 7.95 us (2.95x) |
| prototype persistent MoE layer (slab -> silu/quant -> rowlane, counters) | 21.3 us vs 16.2 us for the 3-node chain (+31%); 15.9 us with the waits removed (-2%) |

So synchronization is cheap on this GPU; what loses is any fusion that makes
part of the grid wait while the rest works, because a workgroup's CU slot is
bound at dispatch and a kernel boundary is a free machine-wide re-pack. That
is why the shipped fusions (shared-as-expert, GDN glue, fused router, HC
chain) work: none adds a cross-workgroup dependency. Ceiling of a perfect
non-blocking scheduler here: ~1.8% of the MoE layer. Dropped.

The number that matters from the study is the baseline: the M=1 MoE layer
streams 7.25 MiB in 23.5 us, a third of HBM bandwidth. The per-expert slices
are small (~22 KB per workgroup), so the GEMV is latency-bound, not
bandwidth-bound. Closing half of that gap is ~-0.35 ms/step across 48 layers,
the largest single lever left at c=1. Next agent: bytes-in-flight per lane in
the W4A8 slab kernel (deeper unroll / more outstanding buffer loads, wider
per-lane vectors, expert-slice-aware tiling), cold-L2, same bit-exactness gate.

## Round 9 (2026-09-04/05): three-boot protocol results

Each arm = pure rc5 image, three separate boots, step timer on 1200-token c=1
decode windows, three c=1 probes + one c=4 probe per boot (`ablate.sh`).

| arm | c=1 ms/step (3 boots) | c=1 tok/s | c=4 tok/s | verdict |
|---|---|---|---|---|
| base (rc5 defaults) | 11.05 / 11.11 / 11.17 | 88.7-89.9 | 208 | control |
| `VLLM_GFX908_W8A16_MFMA=1` | 11.18 / 11.26 / 11.29 | 88.0-88.9 | 215-219 | c=4 +4%, c=1 +0.15 ms: M<=4 also runs through the padded MFMA kernel because the swizzled copy is the only resident int8; swizzled-read GEMV for M<=4 in progress, then default on |
| `VLLM_GFX908_MOE_PREP_FOLD=1` (shared-as-expert fold) | 11.02 / 10.94 / 10.95 | 89.8-90.7 | 205-207 | every boot faster than every base boot (-0.16 ms median, the predicted -3 us x 48 layers); c=4 neutral. **Default on.** |

Also from the gemv_flight study (agents/gemv_flight/REPORT.md), applied as
plain parameter changes, `torch.equal` to the shipping kernels: MoE reduce
`BLOCK` 1024 -> 256 (3 -> 10 workgroups, -1.1 / -1.0 / -0.8 us at M=1/4/8),
slab `wpb` 4 -> 1 gated on tile count (P*N1/4 <= 4096; unconditional wpb=1
regresses +1.2 us at M=8), gate_row's reduction 8 -> 2 barriers (-0.8 us,
bitwise identical). The silu*mul+Q8 fold into the down GEMV prologue is in
too (`VLLM_GFX908_MOE_SILU_FOLD=1`, off): bit-exact with the Triton kernel it
replaces, but only -0.3 / -0.9 / -1.3 us per layer at M=1/4/8 because every
down workgroup redoes the group quant. The study also
refuted the "latency-bound GEMV" reading of the megakernel baseline: load
latency is ~210 ns flat, the kernels already keep 10-20x the Little's-law
bytes in flight, and against a size-matched cold floor the slab / rowlane
GEMVs sit at 78% / 84% (the GDN/HC "44-55%" figures were an L2-warm floor
artifact; they are at 77-97%). What remains per launch is a fixed
1.8-2.3 us, so the only lever left in the MoE layer is one fewer launch
without a cross-workgroup wait (silu fold into the down GEMV prologue,
in progress).
