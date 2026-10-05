# Qwen3.8-Flash-Next MTP decode release (measured 2026-09-30)

Branch `qwen38-flash-next-mtp` (this branch). Image `btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn-mtp` = the rc10
image with this branch's changed Python/HIP files baked in. Serve script: `scripts/gfx908/serve_qwen38_flash_next_mtp.sh`.

**Summary.** Speculative decoding with the model's own MTP head (K=3) now pays at low concurrency:
c=1 decode **166.9 tok/s (thinking off) / 153.8 (thinking on)** against **104.9** without MTP, at **16.4 ms per step**.
GSM8K 500Q 489/500 (reference 485-491). Above ~6 concurrent requests MTP-off is faster (c=16: 601 vs ~400 tok/s aggregate),
so the release ships two modes: `MODE=interactive` (MTP K=3) and `MODE=batch` (MTP off, max-num-seqs 40).
All numbers: 4x MI100 TP=4, 200 W cap, 32k context, measured 2026-09-30 on rc10 + these changes.

rc10 image `btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn`, 4x MI100 TP=4, 32k context, 200 W cap.

**Screen protocol.** c=1, T=0, 8 real-text prompts, thinking off and on (`screen.py`).
- step ms = tokens per step ÷ tok/s.
- Spec counters are read as-is: they are not TP-summed on this stack.

**Concurrency probe** (`conc.py`). 16 clients × 2 requests × 512 tokens, thinking off; the result is aggregate tok/s.

**Keep rule** (`compare.py` in the A/B harness, fixed before each run). A change is kept only if it passes all of these:
- the step gets faster in **both** modes, with no overlap between the two arms' boots;
- acceptance drops by at most 0.02;
- c=16 aggregate stays within 3% of the base.

**Boots.** Paired and alternated (ABBA), with a gate on a log line proving the change is loaded (an arm without it is void).

## Kept items (cumulative, K3, max-num-seqs 16)

| step | change | ms/step | c=1 tok/s (thinking off / on) | c=16 aggregate | KV @32k | branch |
|---|---|---|---|---|---|---|
| ref | MTP off (leak fix) | 9.5 | 104.9 / 104.9 | 601 | 228,733 | |
| 0 | W8-prep leak fix | – | – | – | +4% off; K1 2.6x; K3 now boots | `gfx908-w8-prep-leak` 6be647229d |
| 1 | #4 W4 draft experts (`VLLM_GFX908_MTP_QUANT=w4`) | 24.3 → 23.5 | 115.2 → 118.6 / 105.9 → 109.1 | – | +50% at mns8 | `gfx908-mtp-quant-w4` 6add0ecc7e |
| 2 | #1 QSA eager gate (qsa.py) | 23.5 → 19.5 | 118.5 → 140.7 / 109.5 → 131.7 | 362 → 372 | 102k | `gfx908-mtp-verify` |
| 3 | #3 upstream #58114 + #55054 + #55404 | 19.3 → 18.2 | 141.7 → 150.3 / 131.1 → 139.6 | 377 → 379 | 102k | `gfx908-mtp-upstream-picks` |
| 4 | #2 small-M HC W8 kernels (`VLLM_GFX908_SMALLM_HC_W8=1`) | 18.3 → 16.4 | 148.9 → 166.2 / 138.9 → 155.3 | 394 → 400 | 102k | `gfx908-push-hc` 8d4ba646fa |

**Not kept:**
- HC strided injection: +0.08 ms, reverted.
- HC final-mixer W8: not run. It puts int8 weights on 2 more modules, so it needs its own GSM8K gate.
- T=0 local argmax: not run. It only applies at T=0, and its estimated effect (≤0.2%) is under boot noise.

**Consolidated branch:** this branch (local name `gfx908-q38fn-mtp-best`). Its files are byte-identical to the tested overlays.
- Env: `VLLM_GFX908_MTP_QUANT=w4 VLLM_GFX908_SMALLM_HC_W8=1 VLLM_GFX908_GDN_FUSED_SPEC=1`.
- Spec config: K=3, probabilistic draft.

**Gate on the best config:**
- GSM8K 500Q (thinking, temp 0.6, c=16) = **489/500**, against a reference of 485–491. 0 faults.
- Greedy agreement vs MTP-off: 0/8 exact, token-sim 0.44 / 0.50. The off-vs-off floor is 2/8 at 0.68 / 0.47. Spec-vs-off agreement sits at the level seen in earlier spec runs, which gives no evidence of a defect.

## Regimes and the batch-size-gated spec

**Measured aggregate tok/s by concurrency** (K3 best config vs MTP off, max-num-seqs 16, 512-token outputs):

| c | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| K3 | 157 | 198 | 316 | 307 | 365 | 414 | 422 |
| off | 101 | 126 | 226 | 306 | 399 | 481 | 622 |

- The crossover is at **c≈6**, which gives N=6. KV pool: K3 102,532 vs off 280,776 tokens.

**Gated spec** (`num_speculative_tokens_per_batch_size=[[1,6,3],[7,16,0]]`) first needed a fix.
- The draft graph manager captured a negative query length for the K=0 bucket.
- Fixed in `gfx908-q38fn-mtp-best` 4d19048c05 (`v1/worker/gpu/cudagraph_utils.py`).

**Gated spec result: REVERTED.**
- c=1 matches K3 (1.007 / 1.010).
- c=16 is only 249 aggregate, against K3 383 and off 554. The K=0 bucket runs slower than either pure config.
- KV stays at spec level (102k).
- Root cause not investigated. Suspects: the K=0 decode query length not being graph-captured on the verify runner, or the drafter still running.

**Recommendation until that is fixed:** per-regime configs.
- Interactive (c ≤ 4): K3 best config.
- Throughput (c ≥ 8): MTP off.

## Decode-only profile (prof/best2 vs prof/off2)

Setup: short prompt, 256 tokens, c=1; K3 best config vs off with the same overlays. Numbers are profiler-inflated.
- Off: 12.2 ms/step profiled.
- Best: 18.5 ms/step profiled, 2.25 tokens/step in this window.

Top per-step deltas:

| kernel | ms/step delta |
|---|---|
| `gdn_fused_spec_kernel<64,4>` | +1.42 |
| `hcsm_up<4>` | +1.18 |
| `hcsm_down<4,3,1>` | +1.14 |
| `w8mfma<16,1,1,1>` | +0.90 |
| wvSplitK in_proj_ba M=4 | +0.67 (+0.22 QSA) |
| eager at::native | +1.33 (393 vs 160 launches) |
| QSA attention/indexer | +0.93 (231 vs 64 launches) |
| NCCL | **only +0.34** |

The NCCL figure corrects the prefill-contaminated first profile, where it showed +1.9 ms.

## Open items (not in this release)

Ranked from the decode-only profile; each is a multi-hour kernel job:
HC small-M + W8 MFMA at M=4 (~0.5-1.0 ms/step), the GDN fused spec kernel at M=4 (~0.4-0.8 ms), proposer/QSA spec glue capture
(~0.5-1.0 ms), folding the bf16 in_proj_ba into the qkvz W8 GEMV (needs a GSM8K gate), and the gated-spec K=0 slowdown
(would let one config serve both regimes).
