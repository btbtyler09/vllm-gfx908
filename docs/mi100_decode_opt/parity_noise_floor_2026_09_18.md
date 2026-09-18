# READ BEFORE TRUSTING A GREEDY-PARITY RESULT (2026-09-18)

`parity.py` compares greedy (temperature 0) generations between two serves and
reports "identical N/20". **That gate has a noise floor of roughly one
divergence in twenty within a single boot.** A 17/20 or 18/20 between two
builds is therefore not evidence of a behavioural difference, and several past
rc-versus-rc parity comparisons at that granularity were partly reading noise.

## Measured, Qwen3.8-Flash-Next rc10, TP4, the standard `serve_vision.sh` config

| comparison | identical |
|---|---|
| **same live server, back-to-back, same build** | **19 / 20** |
| two different boots, different builds (an all-reduce barrier patch) | 17 / 20 |
| `parity_rc9.json` vs `parity_rc10.json` (two stock builds) | 15 / 20 |

At scale on one live server, 8 prompts x 8 greedy repeats x 256 tokens
(`car4/repeat_det.py`): **7 of 8 prompts produced more than one distinct
output**. Greedy decoding on this stack is not reproducible run to run.

Mean |dlogprob| on the shared prefix was ~0.001-0.002 even for the
same-server comparison, so these are genuine numerical differences, not
tie-breaking in the sampler.

## Attribution: this is the stock stack, not any one patch

The figures above were first measured on a server carrying an experimental
all-reduce barrier patch, so they were repeated on a **stock, unpatched** rc10
boot (verified unpatched: zero injector lines in the boot log):

| build | prompts non-deterministic (of 8) |
|---|---|
| patched (AR barrier scope) | 7 |
| **stock rc10, run 1** | **7** |
| **stock rc10, run 2** | **5** |

Stock is non-deterministic at the same order as patched, so the behaviour
belongs to the serving stack. Note also that the *metric itself* moved 7 -> 5
on identical code: those two stock runs overlapped in time against one server,
so their requests interleaved and changed each other's batch composition. That
is the mechanism below, visible in the measurement.

## Likely mechanism

Continuous batching. Each parity request is issued standalone, but the server
batches whatever happens to be in flight, so batch composition differs between
runs. Batch shape selects different kernel paths (M-dependent GEMM dispatch,
MoE expert grouping, attention split-K) and therefore different floating-point
reduction orders. Nothing here is a bug; it is the normal consequence of
shape-dependent kernels plus a scheduler that does not guarantee identical
batching.

## How to get a discriminating parity test

- Serve the arms with **`--max-num-seqs 1`** so every request is its own batch,
  or
- drive an **offline `LLM()`** with a fixed, explicit batch, or
- issue the parity prompts strictly serially with the server otherwise idle and
  verify the intra-boot floor is 20/20 *before* comparing arms.

Whichever you choose, **measure the intra-boot floor first on the same build**
and report it next to the cross-arm number. A cross-arm result that is not
better than the floor says nothing.

## What this does not invalidate

Coarser gates are unaffected: GSM8K totals (1281/1319 class), PPL, and
throughput tiers all aggregate over enough samples that a ~5% per-prompt
divergence rate does not move them. It is specifically the token-exact 20-case
parity comparison that lacks the resolution it appears to have.

Raw data: `/home/tyler/work/car4/` (`det_patched.json`, `parity_car_patched.json`)
and `parity_rc9.json` / `parity_rc10.json` here.
