# Custom all-reduce barrier scope on gfx908 — audit + patch (2026-09-16)

**Status:** patch written, NOT built, NOT merged, NOT validated. Branch
`gfx908-car-barrier-scope` off `ple-prefill-flat-conv` (rc10 lineage).

## What was audited

`csrc/custom_collective_common.cuh` and `csrc/custom_all_reduce.cuh` on our
serving branches. The blob is **identical on `mi100-main-sync-2026-08-27`,
`qwen38-flash-next`, `ple-prefill-flat-conv`, `gfx908-dense-gptq-fallback`
and on upstream `vllm-project/vllm` main** (`353dcd07e8`) — we have never
modified it. The last upstream change was `7c6729b769`.

## Finding 1 — cross-rank acquires are DEVICE-scope (ROCm path)

The file has two implementations behind `#if !defined(USE_ROCM)`. gfx908 takes
the `#else` branch, which uses `__scoped_atomic_*`:

| helper | flag store | flag load (the acquire) |
|---|---|---|
| `barrier_at_start` | `__ATOMIC_RELAXED`, `__MEMORY_SCOPE_SYSTEM` | `__ATOMIC_RELAXED`, **`__MEMORY_SCOPE_DEVICE`** |
| `barrier_at_start_release` | `__ATOMIC_RELEASE`, `__MEMORY_SCOPE_SYSTEM` | `__ATOMIC_ACQUIRE`, **`__MEMORY_SCOPE_DEVICE`** |
| `barrier_at_end` | `__ATOMIC_RELEASE` (`RELAXED` if `final_sync`), `__MEMORY_SCOPE_SYSTEM` | `__ATOMIC_ACQUIRE` (`RELAXED` if `final_sync`), **`__MEMORY_SCOPE_DEVICE`** |

The stores are SYSTEM-scope; every acquire is DEVICE-scope. The payload these
flags guard is written by **remote agents** — peer GPUs staging into
IPC-shared buffers across XGMI. A device-scope acquire does not order a remote
agent's writes, so a reduce kernel can pass its barrier and read a peer buffer
whose writes are still in flight. A torn fp16/bf16 element is a NaN or
huge-finite seed.

`barrier_at_start` additionally has no acquire at all — it is fully
`__ATOMIC_RELAXED`.

## Finding 2 — reduce kernels and accumulation order

- `cross_device_reduce_1stage` calls the fully relaxed `barrier_at_start`,
  then `barrier_at_end<ngpus, /*final_sync=*/true>` (also relaxed). It reads
  every peer's input buffer in fixed rank order 0..ngpus-1 — upstream's own
  comment says this is deliberate, "ensuring bitwise identical results".
- `cross_device_reduce_2stage` also calls the relaxed `barrier_at_start`, and
  **does** rotate: `int target = (rank + i) % ngpus`, so each partition is
  accumulated in the order of the rank that owns it. Deterministic for a fixed
  shape, but a different summation order per partition — and a different
  result from what 1stage would produce for the same input.
- Algorithm selection is by **byte size** (`custom_all_reduce.cuh`, macro
  `REDUCE_CASE`): `world_size == 2` → 1stage; otherwise, when fully connected,
  `world_size <= 4 && bytes < 512 KiB` → 1stage, else 2stage.
  `VLLM_CUSTOM_ALLREDUCE_ALGO=1stage|oneshot|2stage|twoshot` forces one.

  For our TP4 serves with hidden 5120 in fp16 (10,240 B/token), the crossover
  is **51.2 tokens**: decode batches take 1stage, prefill chunks take 2stage.
  The same layer's all-reduce therefore reduces in a different order depending
  on batch shape. That is a reproducibility defect, not a wrong answer.

  Side note, unrelated but in the same macro: if `fully_connected_` is false
  and `world_size != 2`, **no kernel is launched at all** and the output
  buffer is left untouched. Our four cards are one XGMI hive, so we always take
  the `fully_connected_` path.

## Blast radius

- **Exposure is ours, not reduced.** All four MI100s sit in a single XGMI hive
  (all-to-all, 1 hop). Peer writes are remote-agent writes exactly as on the
  box where this was found. We are not a PCIe-P2P fallback case.
- **Failure mode differs by finding.** Finding 1 → torn read → NaN/garbage
  activations → junk-token walls; loud, episodic, load-dependent. Finding 2 →
  no wrong answer, only order-dependent nondeterminism across batch shapes.
- **Conditions.** Needs a peer's staging write still in flight when the reader
  clears the barrier: back-to-back ARs of differing sizes, deep pipelining,
  high concurrency, speculative decode alternating prefill chunks with decode
  steps. Larger `ngpus` and more blocks widen the window.
- **Our published numbers.** Throughput figures are unaffected — this is a
  correctness question and the proposed fix is claimed free. Quality figures
  could in principle have lost isolated samples to episodic corruption, which
  would make them **pessimistic, never optimistic**. No observed instance: the
  rc9/rc10 gates (GSM8K 1281 and 1280 of 1319), the 9,834-page vision pass and
  the 1,300-request soaks all ran with CUSTOM AR on and coherent output.
- **Is GSM8K 1280/1319 evidence against corruption?** It rules out *frequent*
  corruption; it does not rule out rare episodic corruption, because a torn
  read in one sample is indistinguishable from an ordinary wrong answer at
  that granularity. The real evidence would be a bitwise cross-rank comparison,
  which is what the validation below asks for.

## Upstream

Reported by curvedinf as `vllm-project/vllm#57059` (2026-09-15, labels
`bug`, `rocm`). Two bot comments only; **no maintainer response** as of
2026-09-16. Upstream main still carries the DEVICE-scope acquires — blob
`353dcd07e8`, byte-identical to ours.

## The patch

1. All three cross-rank acquires promoted `__MEMORY_SCOPE_DEVICE` →
   `__MEMORY_SCOPE_SYSTEM` (ROCm path only; the CUDA PTX path is untouched).
   Ordering (`RELAXED` / `ACQUIRE`) is left exactly as upstream — this is a
   pure scope change.
2. Both reduce kernels now call `barrier_at_start_for_reduce`, which on ROCm
   resolves to `barrier_at_start_release` (RELEASE store + ACQUIRE load)
   instead of the fully relaxed `barrier_at_start`.

Two build-time escape hatches so the A/B does not need a revert:

- `-DVLLM_CAR_PEER_ACQUIRE_SCOPE_DEVICE=1` — restores upstream DEVICE scope.
- `-DVLLM_CAR_UPSTREAM_START_BARRIER=1` — restores the relaxed start barrier.

## Validation plan (needs MI100s; nothing here has been run)

Correctness, in order:

1. **4-rank standalone harness first** (per the standing rule), not a server
   boot. Drive `cross_device_reduce_1stage` and `_2stage` directly at TP4 with
   back-to-back ARs of alternating sizes straddling the 512 KiB crossover, with
   each rank's producer kernel writing its staging buffer immediately before
   the AR. Compare every output element against a single-rank gathered fp64
   reference. Patched build must show zero mismatches; the `DEVICE`-scope build
   is the control (a mismatch there is the proof, but absence of one is not a
   refutation — the race is timing-dependent).
2. **Bitwise cross-rank check in-engine**: dump logits for a fixed greedy
   prompt on all four ranks, assert identical. Repeat 3 boots.
3. **Greedy parity gate**: rc10 config, the standard parity corpus, patched vs
   unpatched — outputs must be identical, since neither change alters
   arithmetic.
4. **Quality gate**: full GSM8K 1319 with `tools/gsm8k_eval_openai.py`,
   `GSM_CONC=8`, against the rc10 reference 1280.

Cost, measured not assumed (they claim zero on gfx908; a SYSTEM-scope acquire
on CDNA1 can emit an L2 invalidate, and we issue on the order of 96 ARs per
decode step, so this must be measured):

5. **Step timer, 3 boots per arm** (boot-to-boot noise is 0.3–0.4 ms/step, so
   a single boot cannot resolve this): patched vs
   `-DVLLM_CAR_PEER_ACQUIRE_SCOPE_DEVICE=1 -DVLLM_CAR_UPSTREAM_START_BARRIER=1`.
   Report c=1 ms/step and tok/s, plus c=16 and c=64.
6. **AR microbench**: per-call latency at 10 KiB, 100 KiB, 400 KiB, 1 MiB,
   4 MiB across both algorithms, patched vs control.

Ship only if correctness passes and the step-time delta is inside the
0.15 ms/step flip threshold.

---

## CORRECTED 2026-09-16 (second pass): vLLM's CAR is the live path, not AITER's

An earlier revision of this section claimed the opposite. It was wrong. It was
derived from a branch default (`_GFX908_DEFAULTS` sets
`VLLM_ROCM_USE_AITER_CUSTOM_AR: "1"`) without checking a single boot log.
Tyler pushed back; the logs settle it.

### Evidence: every boot selected CUSTOM

Scanned every log under `~/work`, `~/bench_results_v027`, `~/mi100-llm-testing`
and the session scratch that contains the backend-dispatch line — 60+ boots
spanning `v0.21.0` through `v0.27.2rc1.dev682+g45eacfa4c`, 2026-08 to 2026-09.
**Every one selected `CUSTOM`. None selected `AITER_CUSTOM`**, although
`AITER_CUSTOM` is listed as a potential backend in all of them:

```
cuda_communicator.py:269  Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order)
for group 'tp:0' out of potential backends: ['FLASHINFER', 'NCCL_SYMM_MEM', 'QUICK_REDUCE',
'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
```

Three independent confirmations in the same boots:

- `allreduce_rms_fusion.py:1584` — "AITER allreduce fusions are disabled
  because AITER Custom All Reduce is not enabled."
- `rocm.py:1104` fires the *opposite* branch ("enabling AITER
  allreduce+rmsnorm fusion"), which is gated on
  `os.environ.get("VLLM_ROCM_USE_AITER_CUSTOM_AR", "0") != "1"`.
- `gfx908_push_ar.py:393` prints on all four ranks. The push AR is constructed
  inside vLLM's `CustomAllreduce` (`maybe_create_push_ar`) and is referenced
  nowhere in `aiter_custom_all_reduce.py`. If AITER CAR were on, `ca_comm`
  would never be constructed (`cuda_communicator.py:120` guards on
  `self.aiter_ar_comm is None`) and the push AR could not exist.

### Mechanism

`docker/Dockerfile.q38fn` bakes `ENV VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_CUSTOM_AR=0`. `_GFX908_DEFAULTS` only fills vars that are
**not already in `os.environ`**, so the image ENV wins and the branch default
never applies. It is the only Dockerfile in the tree that sets either var.

Source read alone would have been misleading twice over: the branch default is
`"1"` while the comment directly above it said "Default off", and the image
overrides both. **Per-image runtime logs, not branch source, decide which
all-reduce ships.** That is the lesson from this correction.

## Corrected blast radius — vLLM `csrc/custom_collective_common.cuh`

This is the file curvedinf patched and reported as
`vllm-project/vllm#57059`, and it is the one we ship.

### Does anything equivalent to the uncached pool protect this path?

**Yes** — and it is upstream code, not something I should have attributed to
AITER. `csrc/libtorch_stable/custom_all_reduce.cu:157`
`allocate_shared_buffer_and_handle` allocates the IPC staging buffers with
`hipExtMallocWithFlags(..., hipDeviceMallocUncached)` under `#if
defined(USE_ROCM)`, with the comment "data buffers need to be 'uncached' for
signal on MI200". Those buffers back the `registered=False` path.

Our fork forces `registered=False` on gfx908 in **both** directions —
eagerly (`custom_all_reduce.py`, the `else` arm) and under capture (the
`on_gfx908()` branch, which exists precisely because HIP IPC views of cached
`cudaMalloc`'d memory drift under graph replay). So every gfx908 CAR message
stages through uncached memory, and the stale-peer-L2 mechanism is absent here
too.

**What does NOT carry over:** AITER's `start_sync` already uses a
RELEASE/ACQUIRE pair. vLLM's `barrier_at_start` is fully `__ATOMIC_RELAXED`
with no acquire at all, and both reduce kernels enter through it. The path we
ship is therefore the **weaker** of the two — mitigated on the caching axis,
unmitigated on the ordering axis.

### The exposed slice, per model

Two filters sit in front of the suspect barriers. The push AR
(`gfx908_push_ar.py:424` `eligible`) takes a message only if it is
**bfloat16**, contiguous, `dim >= 2`, `n % 8 == 0`, `n <= 8192`, and
`numel <= slot_elems = 122,880`. Whatever it declines falls to
`CustomAllreduce`, which itself only handles messages up to `max_size` —
capped by the image ENV `VLLM_GFX908_CUSTOM_AR_MAX_SIZE_MB=2` to **2 MiB**
(`custom_all_reduce.py:234`). Anything larger goes to PYNCCL/RCCL and never
touches these barriers.

**Qwen3.8-Flash-Next (bf16, AR width 2560, 2 B/elem):**

| tokens T in the AR | bytes | path |
|---|---|---|
| 1 – 48 | ≤ 245 KiB | push AR (the boot log's own "T <= 48 at width 2560") |
| 49 – 102 | 245 KiB – 512 KiB | **vLLM CAR, 1stage** |
| 103 – 409 | 512 KiB – 2 MiB | **vLLM CAR, 2stage** |
| ≥ 410 | > 2 MiB | PYNCCL |

Decode at our serving cap (`--max-num-seqs 48`) is entirely push AR. Prefill
chunks (`--max-num-batched-tokens 8192` → 41.9 MB) are entirely PYNCCL. The
exposed band is the middle: mixed batches, chunked-prefill tails and
spec-verify rows with 49 ≤ T ≤ 409. Narrow, and it excludes the two shapes
that dominate the step count.

**Dense Qwen3.8-27B-GPTQ-8bit (fp16, hidden 5120, 2 B/elem):** served with
`--dtype half` (`serve_dense27b.sh:20`) on the same image. The push AR is
**bf16-only**, so it declines every message and the filter in front of the
barriers disappears:

| tokens T | bytes | path |
|---|---|---|
| 1 – 51 | ≤ 512 KiB | **vLLM CAR, 1stage** |
| 52 – 204 | 512 KiB – 2 MiB | **vLLM CAR, 2stage** |
| ≥ 205 | > 2 MiB | PYNCCL |

**Every decode step of every fp16 dense serve we run goes through the relaxed
`barrier_at_start` and the 1stage kernel.** That is the real exposed surface,
and it is the one behind the published dense-27B numbers and the long-context
sweep. I have no boot log for the `v0.27.4rc2.dev` campaign image, so I state
that rather than infer it; the logs I do have span v0.21 to v0.27.2 and are
unanimous.

### Does the 512 KiB crossover still bite?

Yes, inside the exposed band. It is not absorbed by the push AR in either
model: for Flash-Next the push AR cuts off at T=48, below the T=102 crossover,
so the band straddles it; for fp16 dense the push AR is absent entirely and
the crossover sits at T=51, inside the CAR range. So the same layer's
all-reduce still sums in a different order either side of that boundary. It
remains a reproducibility defect, not a wrong answer.

### Patch ranking — inverted from the first pass

1. **`btbtyler09/vllm-gfx908` branch `gfx908-car-barrier-scope` — the one that
   matters.** It is the live path on every image we ship, it carries the fully
   relaxed start barrier, and for fp16 dense serves it handles all decode
   traffic. The patch does both halves: scope promotion on the three acquires
   and `barrier_at_start` → `barrier_at_start_release` in both reduce kernels.
2. **`btbtyler09/aiter-gfx908` branch `gfx908-car-barrier-scope` — near
   irrelevant today.** AITER's CAR is not selected on any image we ship, and
   it already has the release/acquire pair. Keep it (it costs nothing and
   protects anyone who sets `VLLM_ROCM_USE_AITER_CUSTOM_AR=1`), but it is not
   the fix to validate first.

### Effect on the validation plan

The correctness harness must exercise **vLLM's** kernels, at the sizes that
actually reach them: 49–409 tokens at width 2560 bf16, and 1–204 tokens at
width 5120 fp16, straddling the 512 KiB crossover in both. Add an fp16 dense
arm — it is the configuration with no push AR in front of it. The cost
measurement is correspondingly less interesting for c=1 Flash-Next decode
(push AR handles it) and more interesting for fp16 dense decode and for
mid-size mixed batches.

## Superseded first-pass claims

For the record, so nobody re-derives from the wrong version: the first pass of
this note asserted that AITER's CAR was live on our serves, that the
uncached-pool mitigation was an AITER-only property, and that the AITER-side
patch was the one on the hot path. All three are wrong. The graph-capture
finding from that pass (captured ARs route through a pre-registered uncached
buffer rather than binding the input pointer) is correct and holds on the vLLM
path as well, by the same `registered=False` mechanism.


---

## VALIDATION RESULT 2026-09-18 (correctness half only; cost arms deferred)

Cards were free 09:23-14:00; the hub re-scoped to correctness only.

**Arm 1 — 4-rank standalone harness: PASS.** 1.2M all-reduce calls per build
across three builds (stock image `_C`, standalone control with
`-DVLLM_CAR_PEER_ACQUIRE_SCOPE_DEVICE=1 -DVLLM_CAR_UPSTREAM_START_BARRIER=1`,
standalone patched) = 3.6M total. **Zero mismatches, zero NaN.** Per rank per
arm: 150,000 1stage and 150,000 2stage calls, alternating across the 512 KiB
crossover every iteration, 96 queued with no host sync, 8 MB producer write
before each. Bit-exact (integer payloads < 61, 4-rank sum < 244 exact in bf16).
Harness: `/home/tyler/work/car4/car4.py`.

The control passed too, so **the defect is not reproduced**. The patch closes a
formal ordering gap; there is no empirical demonstration it was ever firing.

**Arm 2 — greedy parity: INCONCLUSIVE by construction.** Patched vs the stock
rc10 reference scored 17/20 identical, but the same live server scores only
19/20 against itself back-to-back, and 5-7 of 8 prompts are non-deterministic
across greedy repeats on *stock*. The gate's noise floor swamps the signal.
Full write-up: `parity_noise_floor_2026_09_18.md` (and next to the gate itself
at `~/work/flashnext_vision_smoke/PARITY_NOISE_FLOOR.md`).

**Cost arms (step timer, AR microbench) NOT RUN.** Ship condition is
correctness AND cost inside the 0.15 ms/step flip threshold, so the patch
**remains unshippable**: correctness partially established, cost unmeasured.

**Method notes.** Avoided a 45 MB `_C_stable_libtorch` rebuild by compiling
`libtorch_stable/custom_all_reduce.cu` standalone (shallow include graph) and
re-registering the ops under `_car_test`, then redirecting `vllm._custom_ops`
via a lazy `sitecustomize` injector — two-minute builds, exact single-TU A/B.
Two near-misses: the `.so` was silently truncated to 0 bytes twice on a
writable bind mount, and one boot came up **healthy with `ops_loaded=0`**,
which would have produced a clean "parity passed" while running entirely stock
kernels. Gate on the artifact actually loading, never on `/health`.

## COST ARMS 2026-09-18 (both run; patch is CHEAP, and the in-server test cannot resolve it)

Probe choice: **fp16 dense Qwen3.8-27B-GPTQ-8bit**, because nothing sits in
front of vLLM's CAR there. Flash-Next decode is owned by the gfx908 push AR
(bf16, numel <= 122880), so it barely exercises this patch at all.

### AR microbench — the instrument that can actually resolve it

Same translation unit both arms, only the barrier macros differ, ops redirected
to a standalone `.so` so injection overhead is common-mode.

| message | control us/call | patched us/call | delta | pct |
|---|---|---|---|---|
| 1 tok / 10 KiB | 13.07 | 13.31 | +0.24 | +1.8% |
| 8 tok / 80 KiB | 14.31 | 14.60 | +0.29 | +2.0% |
| 16 tok / 160 KiB | 19.57 | 20.22 | +0.65 | +3.3% |
| 48 tok / 480 KiB | 37.51 | 38.04 | +0.53 | +1.4% |
| 96 tok / 960 KiB | 43.72 | 44.21 | +0.49 | +1.1% |
| 192 tok / 1.9 MiB | 73.55 | 74.18 | +0.64 | +0.9% |
| 384 tok / 3.8 MiB | 134.22 | 134.64 | +0.42 | +0.3% |

Decode sizes median-of-3, positive on 7/7; larger sizes 2 replicates, positive
on 10/10. **Mean +0.37 us per all-reduce call.** At 96-128 ARs per decode step
that is **+0.036 to +0.048 ms/step** — roughly a quarter to a third of the
0.15 ms/step flip threshold. Small, consistently signed, **not zero**.

### In-server step timer — 3 alternating pairs (P,C,P,C,P,C), c=1 TPOT

| pair | patched | control | delta |
|---|---|---|---|
| 1 | 18.514 | 18.365 | +0.149 |
| 2 | 18.515 | 18.401 | +0.114 |
| 3 | 18.502 | 18.521 | **-0.019** |

patched spread 0.013 ms; **control spread 0.156 ms**. The delta is NOT
consistently signed and the control arm's own boot-to-boot drift exceeds the
effect. **The in-server test cannot resolve this patch** — 0.04 ms is 0.2% of
an 18.5 ms step. c=16 and c=64 are worse: the c=64 mean delta is +1.29 ms,
**17x larger than the physical ceiling** the microbench allows (+0.59 us/call
x 128 ARs = +0.075 ms), so that tier is pure batching/scheduling variance.

An earlier two-pair reading of this data suggested a real +0.13 ms/step cost
and reached for an L2-working-set explanation for why it beat the microbench.
The third pair falsified it: there was no effect, only two low control boots.
Recorded because the failure mode is instructive — an alternating design plus
the discipline of finishing all three pairs is what caught it.

### Verdict

**Cost is not a barrier to shipping.** By the only instrument with the
resolution to measure it, the patch costs ~0.04-0.05 ms/step, well inside the
0.15 ms flip threshold. The ship condition remains unmet only on the
**correctness** side, where the defect was never reproduced (harness passes on
control too) and greedy parity is uninformative on this stack.

All six boots gated on `ops_loaded >= 4` (the four TP ranks) before any
measurement was taken. Raw: `/home/tyler/work/car4/cost_*.json`, `arb*.json`.
