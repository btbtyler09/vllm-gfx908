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

## Which all-reduce actually runs on our serves — and the AITER copy

Added 2026-09-16 after the audit above, because it changes where the fix has
to land.

`vllm/platforms/rocm.py` `_GFX908_DEFAULTS` sets
`VLLM_ROCM_USE_AITER_CUSTOM_AR: "1"`. In
`vllm/distributed/device_communicators/cuda_communicator.py`, when
`use_aiter_allreduce` is true the communicator builds `AiterCustomAllreduce`
and the vLLM `CustomAllreduce` (`ca_comm`) is **not constructed at all**
(`if use_custom_allreduce and self.aiter_ar_comm is None`). So on our stock
gfx908 serves the live reduce kernels are **AITER's**, in the aiter fork at
`csrc/include/custom_all_reduce.cuh` — not the vLLM `csrc/` file patched
above. The vLLM patch only matters for `VLLM_ROCM_USE_AITER_CUSTOM_AR=0`
configs.

*(Note: the comment block above that default in `rocm.py` still says "Default
off until the CDNA1 numerics are debugged" while the value is `"1"`. The value
was flipped after the 2026-08-27 sync verified CAR coherent; the comment is
stale and should be corrected.)*

AITER's `start_sync` / `end_sync` have the same shape:

| | flag store | flag load |
|---|---|---|
| `start_sync` (ROCm) | `__ATOMIC_RELEASE`, `__MEMORY_SCOPE_SYSTEM` | `__ATOMIC_ACQUIRE`, **`__MEMORY_SCOPE_DEVICE`** (both the seq and legacy waits) |
| `end_sync` (ROCm) | `__ATOMIC_RELEASE` (`RELAXED` if `final_sync`), `__MEMORY_SCOPE_SYSTEM` | `__ATOMIC_ACQUIRE` (`RELAXED` if `final_sync`), **`__MEMORY_SCOPE_DEVICE`** |

So the **ordering** half of the fix is already present here — `start_sync` is
release/acquire, not relaxed, from the 2026-08 signal-hardening work — and
only the acquire **scope** is still DEVICE. Four load sites.

Two mitigations already in this path make the practical window much narrower
than upstream vLLM's CAR:

1. `start_sync` publishes with a SYSTEM-scope RELEASE, so the producing side
   already orders its pool writes before the flag.
2. The IPC input pool is allocated **uncached** on gfx908
   (`AITER_CAR_UNCACHED_POOL` defaults to `1`, `hipExtMallocWithFlags` with
   `hipDeviceMallocUncached`), so peer reads go to memory and the stale-L2
   mechanism that caused the 2026-08 serving corruption is gone. With no cache
   line to invalidate, the reader's acquire scope has little left to do.

That is a reasoning argument, not a measurement. The scope promotion costs
nothing to carry, so the patch lands on both sides:

- `btbtyler09/aiter-gfx908` branch `gfx908-car-barrier-scope` (commit
  `e2b5092a7`): four acquires promoted to `__MEMORY_SCOPE_SYSTEM`, reversible
  with `-DAITER_CAR_PEER_ACQUIRE_SCOPE_DEVICE=1`. **This is the one on our hot
  path.**
- `btbtyler09/vllm-gfx908` branch `gfx908-car-barrier-scope` (this commit):
  the vLLM-side fix, for `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` configs.

**Graph-capture pool — answered, it is the same uncached pool.**
`aiter/dist/device_communicators/custom_all_reduce.py:1250` computes
`reg = self.enable_register_for_capturing and not _on_gfx908()`, so on gfx908
a captured all-reduce always takes `registered_input=False` — the copy-in path
that stages into the pre-registered `input` pool, with the copy captured inside
the graph (the "registered" path would bake a cached-memory IPC view of the
input tensor's own pointer into the graph, which is the 2026-08 replay
corruption: first decode token correct, every replayed token after it wrong).
That `input` pool is created once at line 1084 with `uncached=uncached_pool`,
where line 1080 sets `uncached_default = "1" if _on_gfx908() else "0"`. So the
graph path and the eager path share one uncached allocation; graphs do **not**
reintroduce the stale-L2 mechanism, and the blast radius above stands as
written. (Overridable with `AITER_CAR_UNCACHED_POOL=0`, A/B only.)

**Fabric, measured not inferred.** `rocm-smi --showtopotype --showtopohops
--showtopoweight` on this node: every off-diagonal pair reads `XGMI`, 1 hop,
weight 15 — a genuine all-to-all 4-card hive, no bridged pairs and no PCIe
cross-pair link.
