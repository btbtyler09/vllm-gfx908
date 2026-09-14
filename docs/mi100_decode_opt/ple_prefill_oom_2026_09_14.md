# PLE short-conv prefill OOM (Qwen3.8-Flash-Next, gfx908, 2026-09-14)

## Incident

Vision-enabled serve of `Qwen3.8-Flash-Next-GPTQ-4bit` (image
`v0.28.0rc9.dev-q38fn`, TP4, `--gpu-memory-utilization 0.90`,
`--max-num-batched-tokens 8192 --max-num-seqs 48`, `--limit-mm-per-prompt
image:4`) died at 11:09:56 UTC after 17 minutes / 576 completions of the shrew
page-review loop at 12 concurrent agents:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 704.00 MiB.
GPU 1 ... 0 bytes is free. Of the allocated memory 28.97 GiB is allocated
  ple_layer.py:942 _short_conv_dilated_prefill_batched
```

on all four ranks in the same step. KV pool usage was 64-76 %, the encoder
cache (16k tokens) resident. Log:
`~/work/flashnext_vision_smoke/vision_serve_loop_crash.log`.

## Root cause

The dumped scheduler output of the fatal step had 12 requests:
7 decodes (1 token) and 5 prefills of **16, 16, 21, 384 and 7200** tokens
(7,644 scheduled tokens, one request also carried 4 encoder inputs).

`_short_conv_dilated_prefill_batched` packed the prefills into a rectangle
`[num_prefills, C, max_len + L]` with `C = hc_hidden_size = hidden_size x
hc_count = 2560 x 4 = 10240`, `L = conv_state_len = 9`, and ran `F.conv1d`
on it. At the fatal shape one such tensor is

    5 x 10240 x 7209 x 2 B = 738,201,600 B = 704.0 MiB

which is exactly the failing allocation. The function materialized several of
them in sequence (`packed_tokens`, its transposed contiguous copy, `history`,
`conv_output`, the SiLU output and its transposed copy), so the live transient
of one PLE layer call was 2-3 GiB on top of the conv workspace, for 7,637 real
tokens. The rectangle wastes `num_prefills x max_len / total_tokens` = 4.7x
here, and the worst case allowed by the config (one 8,000-token chunk beside
47 short prefills, still under 8192 batched tokens) is 48 x 10240 x 8201 x 2 B
= **7.5 GiB per copy**.

Why the memory profiler never reserves for it:

1. `GPUModelRunner._dummy_run(is_profile=True)` splits
   `max_num_batched_tokens` evenly over `max_num_seqs`: 48 requests x 170
   tokens. The rectangle for that batch is 48 x 171, i.e. no bigger than the
   real tokens. The transient depends on the *shape* of the batch (longest
   request x number of prefills), not on the token budget the profiler
   exercises.
2. During profiling the PLE layer receives no Mamba-family metadata
   (`_short_conv` -> `_short_conv_fallback`, which convolves the whole batch as
   one padded sequence), so the batched path is not even executed.

So `--gpu-memory-utilization` was honest for decode and for uniform prefill
batches and dishonest for mixed chunked-prefill batches, which is exactly what
a multi-agent loop produces (one large page prefill arriving while others are
mid-way).

## Fix (this branch, `ple-prefill-flat-conv`)

`vllm/models/qwen4_exp/amd/ple_short_conv_flat.py`: compute the dilated
causal depthwise conv directly on the flat `[T, C]` token stream. Each of the
`K = 4` taps is a masked gather (`x[t - d]` inside the request, the initial
conv state before it), accumulated in fp32 in chunks of
`VLLM_GFX908_PLE_CONV_CHUNK` (default 2048) rows and cast back; the
`next_state` write-back gathers the last `L` history entries per request the
same way. `_short_conv_dilated_prefill_batched` now calls it and keeps its
state gather / masked write-back unchanged.

Transient bound: `chunk x C x (2 B tap + 4 B accumulator) + T x C x 2 B`
output, independent of `num_prefills` and of the longest request:
about 120 MiB at chunk 2048 plus the output, versus 704 MiB *per copy* at the
crash shape and 7.5 GiB at the config worst case. No MIOpen conv, so no
im2col workspace either.

Numerics: fp32 accumulate of 4 products in both formulations; only summation
order can differ. `tests/models/qwen4_exp/test_ple_short_conv_flat.py`
(17 cases, CPU, run in the rc9 image) shows fp32 within 1e-5, bf16 within
2e-2 (before SiLU rounding), and the conv-state write-back bit-identical,
including requests shorter than the 9-entry state window, mixed initial
states and null-block rows. GPU parity (greedy logprob parity vs the rc9
image on the standard prompt set) is part of the verification window.

The spec-decode path `_short_conv_dilated_spec_batched` has the same
rectangle (`[num_reqs + 1, max_len, C]`) but `max_len = num_spec + 1`, so it
is bounded and left as is (MTP is off for this model anyway). The decode
path is per-token and unaffected.

## Measured (one MI100, 2026-09-14 11:44 UTC, `tools/ple_conv_membench_gfx908.py`)

Peak transient of one PLE prefill short-conv call, bf16, C = 10240:

| batch shape | padded (rc9) | flat (rc10) |
|---|---|---|
| crash shape: 5 prefills, longest 7200 | 3,520 MiB, 41.5 ms | 470 MiB, 13.8 ms |
| one 8192-token chunk | 800 MiB, 10.8 ms | 480 MiB, 14.2 ms |
| 12 x 600 (even agents) | 705 MiB, 9.2 ms | 463 MiB, 13.2 ms |
| worst: 8000 + 47 x 4 (48 seqs) | RuntimeError (rectangle > 2^31 elements, `canUse32BitIndexMath`) | 489 MiB, 15.1 ms |

Output max abs diff 0.0 and conv state bit-identical on every shape the
padded path can run. The flat transient is ~470-490 MiB independent of the
batch shape (fp32 accumulator chunk + output); the padded one scaled with
`num_prefills x longest request` and at the config worst case could not run
at all.

## Verification plan (GPU window)

1. `~/work/flashnext_vision_smoke/ple_conv_membench.py` on one MI100: peak
   transient and time, padded vs flat, at the crash shape, one 8192 chunk,
   12 x 600, and the 8000 + 47 x 4 worst case; max |diff| and state equality.
2. Bake `v0.28.0rc10.dev-q38fn` from this branch (do not touch rc9).
3. Boot with the vision serve config, greedy parity vs rc9 on the standard
   prompts, then the synthetic worst case: 12 B3 pages (6.8k image tokens
   each) with crops in flight at `--gpu-memory-utilization 0.90`, and the
   20-page smoke for agreement.

## Proposed serve config for 12-16 agents (after the fix)

Same as `serve_vision.sh` (TP4, bf16, 8192 batched tokens, 48 seqs,
image:4, qwen3_xml tool parser) at `--gpu-memory-utilization 0.90`.
`--max-num-seqs 48` stays (KV pool 220k tokens = 6.7 x 32k contexts, enough
for 16 agents with page + crops). If the window shows the encoder cache
plus 12 B3 pages still crowding the pool, lower `--max-model-len` to 16384
(pages are < 9k tokens) rather than the utilization. Until rc10 is verified,
the loop runs on rc9 at 0.85 utilization and 8 agents.
