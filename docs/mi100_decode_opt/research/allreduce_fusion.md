# Fusing the TP all-reduce into producer/consumer kernels on 4x MI100 (gfx908)

Research note, 2026-09-03. Read-only study; nothing here has been run. Numbers marked
(measured) come from Tyler's `ar_bench.py` / profiles; everything else is derived or cited.

## 0. Where the microseconds go today

Per decode token at c=1 (`_profile_c1_launch_map.txt`): 96 x `vllm::cross_device_reduce_1stage`
(2 per layer x 48 layers: after the GDN/QSA `o_proj` RowParallel GEMV, and after the MoE
routed+shared sum) plus 187 x `__amd_rocclr_copyBuffer`, 96 of which are the custom-AR staging
copies. Measured in a captured graph: **7.8 us per 5 KB all-reduce** (RCCL: 19 us), i.e. ~0.75 ms
of AR kernels + ~0.13 ms of staging copies ~= 0.9-1.0 ms/token out of a 16.7 ms step.

What one AR call does on the gfx908 path (all file paths below are repo-absolute):

1. Python `CustomAllreduce.custom_all_reduce()` in
   `/home/tyler/vllm-gfx908/vllm/distributed/device_communicators/custom_all_reduce.py` — under
   capture on gfx908 it forces `registered=False` (the cached-IPC-view `registered=True` path
   drifted to NaN under replay; see the soak in
   `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/test_e_persistent_car/`).
2. C++ `all_reduce()` in `/home/tyler/vllm-gfx908/csrc/libtorch_stable/custom_all_reduce.cu`:
   `cudaMemcpyAsync(reg_buffer, inp)` — this is the copyBuffer blit kernel — then
   `fa->allreduce<T>()`.
3. Kernel `cross_device_reduce_1stage<T,4>` in `/home/tyler/vllm-gfx908/csrc/custom_all_reduce.cuh`:
   `barrier_at_start` (relaxed system-scope store of `flag` into each peer's
   `start[block][rank]`, spin on own `start[block][peer]`), pull-reduce 4 pointers in fixed rank
   order (bitwise identical on all ranks), `barrier_at_end<..., final_sync=true>` (relaxed).
   For 5 KB: `size/8 = 320` packed elements -> **one 512-thread block** (`defaultBlockLimit=16` on
   ROCm is irrelevant at this size).
4. Both the signal/tmp region (`meta_ptrs`) and the staging pool (`buffer_ptrs`) are allocated
   with `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` (same file, line ~171: "data buffers
   need to be uncached for signal on MI200").

So one AR = 2 launches (~1.3 us each real, per the profiler-inflation note) + a start barrier
(1 XGMI write + poll) + 3 remote 16 B pulls per thread + an end barrier. The payload
(T x 2560 bf16, T <= 48 -> 5-245 KB) is irrelevant to latency: 5 KB over one XGMI link at
~46 GB/s/direction is ~0.1 us. Everything is launches and round trips.

The consumer on both AR sites is the same kernel: Triton `_hc_combine_norm_kernel`
(`/home/tyler/vllm-gfx908/vllm/models/qwen4_exp/amd/ops/hc.py:267`, grid `(N, hc_count)`), which
reads `block_output[row, 0:2560]`, the residual stream slice, the injection logits, and writes the
combined residual and the RMSNorm'd `xn`. The producer on the attention site at small M is the
wvSplitK copy `/home/tyler/vllm-gfx908/vllm/models/qwen4_exp/amd/csrc/gfx908_wv_fused.hip`
(lane 63 of each wave stores `YTILE` bf16 values per column tile; persistent loop over `m`).
The MoE site's producer is a Triton reduce kernel (routed sum + shared expert), not the HIP GEMV.

## 1. Memory-ordering facts for gfx908 (what is architecturally guaranteed)

Source: LLVM `AMDGPUUsage.rst`, "Memory Model GFX6-GFX9" (gfx908 uses this model; the GFX90A
section adds `buffer_wbl2`/`buffer_invl2`, which **do not exist on gfx908**). Code sequences:

| LLVM op (global addr space)            | gfx908 machine code                                                |
|----------------------------------------|--------------------------------------------------------------------|
| store atomic release, agent or system  | `s_waitcnt vmcnt(0) lgkmcnt(0)` ; `global_store`                    |
| load atomic acquire, agent or system   | `global_load glc=1` ; `s_waitcnt vmcnt(0)` ; `buffer_wbinvl1_vol`   |
| fence release, agent or system         | `s_waitcnt vmcnt(0) lgkmcnt(0)`                                     |
| fence acquire, agent or system         | `s_waitcnt vmcnt(0) lgkmcnt(0)` ; `buffer_wbinvl1_vol`              |
| volatile / nontemporal load            | `global_load glc=1` (+`slc=1` for nontemporal)                      |
| fence release, **workgroup** scope     | `s_waitcnt lgkmcnt(0)` only (no `vmcnt(0)`!)                        |

Consequences that matter for a push design:

- **No L2 writeback/invalidate exists on gfx908.** System-scope release is just "drain this
  wave's outstanding vector memory ops"; acquire is "drain + invalidate the per-CU vector L1".
  Both are cheap (no whole-L2 flush as on gfx90a's `buffer_wbl2`). The cost of a cross-rank
  handshake on MI100 is therefore purely XGMI round trips, not fences.
- Because there is no L2 maintenance, **cross-agent coherence must come from the page MTYPE**:
  the GFX6-9 text says the L2 "can be kept coherent with other agents on some targets, or ranges
  of virtual addresses can be set up to bypass it". The GFX90A text spells out the mapping the
  driver uses on CDNA: `MTYPE UC (used for remote fine grain memory) bypasses the L2`, while
  remote *coarse-grained* memory is `MTYPE NC` and is cached in the reader's L2 and only kept
  coherent by `buffer_invl2` — which gfx908 does not have. This is the architectural explanation
  of Tyler's empirical result: IPC views of cached (`hipMalloc`/torch-allocator) memory drift
  across graph replays on gfx908 (stale L2 lines survive kernel boundaries; only L1/K$ are
  invalidated at dispatch), while `hipDeviceMallocUncached` views are coherent.
  **Rule 1: every byte that crosses ranks — payload slots and flags — lives in
  `hipDeviceMallocUncached` memory on the receiving GPU**, exactly like today's `meta_ptrs` /
  `buffer_ptrs`. (HIP docs: fine-grained memory "typically bypasses the L2", recommended for
  "atomic flags, signals, small synchronization variables".)
- `s_waitcnt vmcnt(0)` is **per wave**. A workgroup barrier (`__syncthreads`) on gfx908 emits a
  workgroup-scope fence, which does *not* wait for vector stores. To publish stores made by all
  waves of a block with one flag, every wave must execute a system/agent-scope release fence
  (`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "")` or `__threadfence_system()`) *before*
  `__syncthreads()`, and only then may one thread write the flag. vLLM's `barrier_at_end` gets
  this right by having the flag store itself be `__ATOMIC_RELEASE` in each of the `ngpus`
  signalling threads after a `__syncthreads`, but note that relies on the release being executed
  by the same waves that did the stores (one block, 512 threads) — in a multi-wave GEMV epilogue
  it is not automatic.
- Does a store ack (`vmcnt` decrement) for a UC store to peer VRAM imply visibility at the peer?
  On CDNA the write completes at the destination memory side before the ack returns (this is what
  vLLM's and AITER's release/acquire 2-stage kernels rely on across XGMI; vLLM's own comment:
  "I did not manage to make [a visibility violation] happen through a lot of testing"). Treat it
  as reliable but keep the sentinel variant (section 3.2) as the option that does not depend on
  it at all.
- P2P atomics over XGMI are supported on gfx908 ("peer-device memory within an Infinity Fabric
  hive is treated as device memory for atomic operations", ROCm atomics doc), 32- and 64-bit
  integer. `global_atomic_add_f32` exists but is irrelevant here.
- Reads of a UC slot must bypass L1: use `glc=1` loads (`__builtin_nontemporal_load`, volatile,
  or any `__scoped_atomic_load_n`) for polling, and issue `buffer_wbinvl1_vol` (part of an
  acquire load/fence) before reading payload with ordinary loads. Reading payload with
  nontemporal loads makes the invalidate unnecessary.

Existing in-tree evidence of these rules: AITER's `start_sync`/`end_sync`
(`/home/tyler/aiter/csrc/include/custom_all_reduce.cuh:161-262`) use
`__scoped_atomic_{store,load}_n` with `RELEASE/ACQUIRE` at `SYSTEM`/`DEVICE` scope; the comment at
line 170 documents a real gfx908 race ("mixed-size race fix": grid extents differ across launch
sizes, blocks skipped by a smaller launch leave stale counters that satisfy a later wait) and
line 3680 documents why its fix (a host-bumped cookie) is disabled: **kernel-argument cookies are
baked at graph capture**. AITER also forces the naive (vLLM-derived) kernels on gfx908 because its
LDS double-buffered kernels "corrupt peer reads when captured into CUDA graphs" (line 3808). The
2-stage `_write_mode` kernel (line 590) is the in-tree example of pushing results into peers'
registered output buffers with `__builtin_nontemporal_store`.

## 2. Prior art and what transfers

| System | Mechanism | Numbers | Transfers to gfx908? |
|---|---|---|---|
| vLLM 1-stage CAR | pull; start+end flag barriers per block | 7.8 us / 5 KB on 4x MI100 (measured) | baseline |
| TRT-LLM / FlashInfer `trtllm_allreduce_fusion` "Lamport" one-shot | push into every peer's slot; **no flags**: buffers pre-filled with `-0.0`, consumer spins until no element of any rank's slot is `-0.0`; producer sanitizes `-0.0 -> +0.0`; 3 rotating buffers per rank indexed by `flag%3`, the consumer clears the buffer from 2 calls ago; fused residual+RMSNorm after the wait | (NVLink) | Yes — the data path needs no fences, only UC slots; rotation must be graph-safe (section 3.4) |
| MSCCL++ MemoryChannel LL protocol | push; 8 B stores carrying data+flag, receiver polls the flag word (relies on single-copy atomicity of one store) | H100 1 KB AR 9.5 -> 5.0 us vs NCCL; MI300X up to 3.8x vs RCCL small msgs; NVLink P2P latency 0.82 us | Only if a 16 B store is single-copy atomic over XGMI — not documented for gfx908; per-element sentinel (2 B bf16) avoids the question |
| "Every us matters" (GB200) | push vs pull: push costs half an RTT; LL flags-in-data; sentinel `-NaN`; double buffering as credit flow control | SoL 1.4 us, one-shot 1.5 us (2 GPU), NCCL 11 -> 2.4 us | Design principles yes; hardware numbers no |
| SiFAR (H200, TP4/8, 8 KB) | dual buffering removes bottom barrier; speculative validation flag reduced with payload (needs NVSwitch `ld_reduce`) | TP4 8 KB: 4.38 -> 2.39 us; TP8: 5.11 -> 2.44 us | Dual buffering yes; in-switch reduction no |
| AITER fused AR+RMSNorm (`local_device_load_rmsnorm*`) | AR kernel then norm kernel reading the reduced tmp buffer | — | Shape (consumer fusion) yes; gfx908 forced onto naive kernels |
| MI300A IF paper | direct remote load 690 ns vs local 346 ns; RCCL p2p floor 20 us; MPI 1.9-4.8 us | | Order-of-magnitude for XGMI RTT (~1 us class) |
| MI250X IF paper | `hipMemcpyPeer` 16 B: 8.7-18 us; xGMI 50 GB/s/direction/link | | Confirms SDMA/memcpy is the wrong tool at this size |

No published one-shot latency for 4x MI100 was found; Tyler's 7.8 us is the only datum. XGMI
one-way store latency on MI100 is not published; assume 1-2 us per hop for planning and measure.

## 3. Recommended design

Three steps of increasing intrusiveness; each is independently shippable and testable, and each
later step keeps the earlier plumbing.

### 3.1 Step A — kill the staging copy (no kernel changes)

Give the producer an output tensor that *is* a view of a UC, IPC-registered slot, and call the AR
with that pointer as the registered input. Today `buffer_ptrs[rank]` is exactly such a buffer
(uncached, registered via `register_buffer`), so the change is: at each AR site, allocate `C`
as `torch.frombuffer`-style view of `buffer_ptrs[rank] + site_offset`, run the GEMV/reduce into
it, then `ops.all_reduce(fa, C_view, out, 0, 0)` (the `_reg_buffer == 0` branch skips the
memcpy). Under capture the `registered=True` path was the one that drifted — but only because it
registered *cached* torch allocations; here the input already lives in the uncached pool, so it
is the same memory the `registered=False` path copies into. Per-site offsets are needed because
the start barrier only orders "my copy vs peers' reads" when the same slot is reused; with the
end barrier still present a single shared slot is also safe, but per-site slots are required by
Step B anyway. Saves 96 blit launches (~0.13 ms/token) and removes the SDMA-vs-compute-queue
ordering hazard AITER's `car_pool_copy_kernel` comment describes.

Caveat: the producer then stores into UC memory (L2-bypassing). For a 5-245 KB output that is
free; do not do this for large-M GEMMs.

### 3.2 Step B — fused pull consumer: wait + reduce + HC-combine + RMSNorm in one kernel

Replace `cross_device_reduce_1stage` + `_hc_combine_norm_kernel` with one HIP kernel that
(a) does the start barrier, (b) pulls the 4 slots in fixed rank order, (c) computes
`out = res + block*inj`, `y = rmsnorm(out)`, exactly as the Triton kernel does, and (d) has
**no end barrier**: each site owns its own slot, and slot reuse is ordered by the next site's
exchange (invariant in section 3.4). Removes one launch and one barrier per AR
(~2.5-3.5 us). Works for every producer (HIP GEMV, Triton MoE reduce, any M).

### 3.3 Step C — push from the producer epilogue, sentinel-validated (the recommended endpoint)

Push mode, Lamport-style, adapted to gfx908:

- Layout (per rank, allocated once with `hipDeviceMallocUncached`, IPC-exchanged at init like
  `meta_ptrs`, pointers passed as *fixed* kernel args — no post-capture registration):
  `slot[site][src_rank][T_max][2560] bf16` on the **receiving** GPU. 96 sites x 4 x 48 x 2560 x
  2 B = 94 MB per GPU, or size `T_max` to the largest captured batch. Pre-filled with the
  sentinel at init.
- Sentinel: `-0.0` (bf16 `0x8000`). The producer sanitizes: `if (v == 0.f) v = +0.f` before the
  bf16 convert (TRT-LLM does the same). Arithmetically harmless: `-0.0` and `+0.0` sum
  identically into the residual. Do not use a NaN pattern; a real NaN must propagate.
- Producer epilogue (wvSplitK, lane 63 of each wave, per output element):
  ```
  // gfx908_wv_fused.hip, EPI 0/1 store site, replaces  C[m+y+n*M] = __float2s(v);
  bf16 b = sanitize_neg_zero(__float2bfloat16(v));
  #pragma unroll
  for (int r = 0; r < 4; ++r)                     // peers incl. self, fixed order
      __builtin_nontemporal_store(b, slot_base[r] + ((site*4 + my_rank)*T_max + n)*2560 + m + y);
  // no fence, no flag, no atomics: each element self-validates at the consumer
  ```
  `slot_base[r]` are 4 pointers in a `__constant__`/kernarg struct. Remote stores are
  fire-and-forget; the kernel's normal end-of-dispatch drain covers them. For the Triton MoE
  producer the same epilogue is four `tl.store`s to four base pointers — no fences needed, so the
  sentinel scheme is the only one that is Triton-friendly on the producer side.
- Consumer (HIP; one workgroup per `(row, stream)` like the Triton grid, 640 threads x 4 elems or
  256 threads x 10 elems over the 2560 columns; every program of a row polls the same 4 x 2560
  slot rows — reads are local UC, cheap):
  ```
  const bf16* s[4] = {slot(site,0,row), slot(site,1,row), slot(site,2,row), slot(site,3,row)};
  bf16x8 v[4]; bool ready;
  do {
      ready = true;
      for r in 0..3: v[r] = nontemporal_load16(s[r] + col);        // glc=1 slc=1, bypasses L1/L2
      for r in 0..3: for e in 0..7: ready &= (bits(v[r][e]) != 0x8000);
      // optional: bounded spin -> write diagnostic, __builtin_trap()
  } while (!ready);
  float acc[8] = 0; for r in 0..3: acc += float(v[r]);             // fixed rank order: bitwise-identical on all ranks
  block = bf16(acc)                                                 // matches today's AR output rounding
  ... HC combine + RMSNorm exactly as _hc_combine_norm_kernel (fp32 math, same rounding points) ...
  __syncthreads();                                                  // all of this program's polls are done
  if (stream == 0) for r in 0..3: nontemporal_store16(s[r] + col, SENTINEL16);   // clear for next use
  ```
  Only one of the 4 stream-programs of a row clears; the others merely read. A program that reads
  slot data before the clearing program has cleared it is fine (data is still there); a program
  that reads *after* the clear would spin forever — so the clear must be ordered after all 4
  programs' reads. Simplest: make the row's 4 streams one workgroup (4 x 640 columns = 2560 =
  one 512-thread block with 5 elements/thread, or 1024 threads), so a single `__syncthreads`
  orders reads before the clear. That also halves the polling traffic.
- No epoch, no counters, no wrap-around, nothing baked at capture time. Graph replay is safe
  because the slot contents themselves are the state, and every replay leaves them cleared.

Why sentinel over flags on gfx908: the flag variant needs (1) a per-wave system-scope release
before a block barrier (every wave drains its remote stores: one XGMI ack RTT on the critical
path), (2) a per-block remote atomic or flag to each peer (another one-way hop), (3) a
device-resident epoch (the AITER cookie lesson), and (4) a wrap-around story. The sentinel
variant's critical path is one one-way XGMI store latency plus one local poll. Estimated 1.5-2 us
better per AR and far less state to get wrong. Keep the flag variant (section 3.5) as the
fallback if the sentinel scan turns out to cost more than expected at T=48 (4 x 245 KB per poll
round per row-group — still local UC reads, ~0.5 us per round).

### 3.4 Correctness argument for single-slot-per-site reuse (no end barrier, no rotation)

Claim: rank B's producer for site k at token t+1 cannot write `A.slot[k][B]` before rank A's
consumer for site k at token t has finished reading and clearing it.

- On A: consumer(k,t) precedes producer(k+1,t) in stream order (kernel boundary = full drain of
  A's stores, including the local sentinel clear).
- Producer(k+1,t) on A pushes A's partial into B.slot[k+1][A]; B's consumer(k+1,t) cannot
  complete before that push lands.
- On B: consumer(k+1,t) precedes producer(k,t+1) in stream order (95 sites and a sampler in
  between; for k = 96 the guard is site 95 of token t+1: B.producer(96,t+1) follows
  B.consumer(95,t+1) which needs A.producer(95,t+1) which follows A.consumer(96,t)).
- Therefore A's clear of slot k (a completed local write) happens-before B's next write to it.
  Two writes to the same HBM address by the same completion point cannot reorder once the first
  has been acked; there is no cache in between (UC).

Requirements this imposes (all already true for today's custom AR): every rank executes the same
sequence of sites; there are >= 2 sites per cycle; no rank runs an AR site outside the shared
sequence (a dummy/profiling eager call on one rank only would deadlock the others — same as
today). Mixed batch sizes across steps are fine: the slot is indexed by `(site, src_rank, row)`
and only rows `< T` are written, read and cleared, with the same `T` on all ranks.

Graph capture: producer and consumer kernels take only static pointers (`slot_base[4]`, site
index, `T_max`) plus the usual tensors. Capture warm-up runs execute the real exchange (they must
— all ranks warm up together, as now). `register_graph_buffers` is not needed for these buffers;
it stays for any remaining legacy CAR calls (e.g. the PLE `tensor_model_parallel_all_reduce` in
`gfx908_ple_zc.py`).

### 3.5 Flag-based variant (fallback), kept graph-safe

If a producer cannot be modified per-element (e.g. a library GEMM), use: producer writes its
output into `slot[site][me]` on every peer (or, in pull mode, only locally), then per block:
`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "")` in every thread; `__syncthreads()`; thread 0:
`__scoped_atomic_fetch_add(&peer_cnt[r][site][me], 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)`
for r in 0..3 (the release fence already ordered the data). Consumer: spin until
`cnt[site][r] == nblocks_of_producer` for all r with `__scoped_atomic_load_n(ACQUIRE, DEVICE)`,
then read, then reset its own 4 counters to 0 (same reuse argument as 3.4 — resetting instead of
epochs removes wrap-around and capture-baked cookies). `nblocks_of_producer` must be computed by
the same function on all ranks and passed to both kernels. Note the wvSplitK early return
(`if (threadIdx.y >= _WvPrGrp) return;`) must become a fall-through to the final barrier.

## 4. Expected savings per token (c=1, 96 ARs)

| Step | Removes | Adds | Est. per AR | Est. per token |
|---|---|---|---|---|
| A: direct write to UC slot | 1 blit launch (~1.3 us real) | producer stores to UC (~0) | -1.3 us | -0.13 ms |
| B: fused pull consumer | AR launch (~1.3 us), end barrier (~1-1.5 us), `hc_combine_norm` launch already merged (-1.3 us), one intermediate write/read | remote pulls inside the consumer (same as today) | -3…-4 us | -0.3…-0.4 ms |
| C: sentinel push | start barrier (1 remote write + poll, ~1.5-2 us) and remote pull RTT (~1 us) | 3 extra remote stores per element in the epilogue (fire-and-forget), local sentinel poll, local clear (~0.5 us) | -2…-3 us | -0.2…-0.3 ms |
| **A+B+C** | | | 7.8+1.3 -> ~2.5-3.5 us | **-0.6…-0.75 ms (~4% of the 16.7 ms step; ~+2.5 tok/s at 60)** |

The AR floor cannot go below ~one XGMI one-way latency plus one poll (~1.5-2 us) because the
consumer must observe the slowest peer; the remaining cost is cross-rank skew, which no AR design
removes. These estimates must be validated with an extended `ar_bench.py` (4 processes,
graph-captured, 50 back-to-back sites) before any server work — per the "probes are not results"
and ">= 3 in-server c=1 probes before commit" rules. The split-K finalize lesson applies: a
microbench win can lose in-server if the fused kernel delays the dependent kernel.

At c=16-48 the producers are different kernels (MFMA/Triton paths), so only Steps A and B apply
unless their epilogues are also given the 4-pointer store; the consumer kernel handles both modes
via a `push` flag, so the model code stays uniform.

## 5. Fewer all-reduces per layer? (direction 4)

Per layer: `o_proj -> AR -> hc_combine_norm (RMSNorm) -> mix -> MoE -> AR -> next layer's
hc_combine_norm`. The HC combine is linear in `block_output`, but RMSNorm and the block itself are
not, so neither AR can be deferred into the other, and the shared expert is already summed with
the routed experts before a single AR (`reduce_results=False` on the shared expert in
`qwen3_next.py:208`). The base model's `use_sequence_parallel_moe` path
(`qwen3_next.py:561-600`: all-gather before attention, reduce-scatter after) replaces one AR with
RS+AG — strictly worse at T <= 4 (pads T to the world size). Conclusion: **2 per layer is the
floor for TP4**; the lever is the per-call cost, not the count. The only count-reducing option is
a different parallelism for the 36 GDN layers (replicated GDN with sharded MoE), which trades the
o_proj AR for replicated bf16 GDN weights (1.04 GB/token/rank already ~11% of the step) — not
attractive on 32 GB.

## 6. Failure modes and how to test each

| Failure | Symptom | Cause | Test |
|---|---|---|---|
| Hang | one rank spins forever | site-sequence mismatch across ranks; producer grid/`T` mismatch; a wave returned before its block's release (flag variant); consumer cleared before all programs read (sentinel) | bounded spin (e.g. 2^26 iters) -> write `(site, rank, row, col, observed bits)` to a UC debug slot and `__builtin_trap()`; 4-proc harness with per-rank timeouts; run mixed-size sequences T=1,48,1,3 (the AITER gfx908 race) |
| Stale slot data | wrong sums, no NaN, drifts over time | reading a cached view of the slot (Rule 1 violated); missing L1 bypass on the poll/payload loads; clear executed before a sibling program's read | E3-style 1000-replay soak with input perturbation each replay, bitwise compare vs `torch.distributed.all_reduce` in the same fixed rank order; check `hipPointerGetAttributes`/allocation flags of every cross-rank pointer at init |
| Rank divergence | ranks' residual streams differ -> model desyncs silently, then garbage | different summation order per rank, or one rank reading a torn/old element | per-token hash of `hidden_states` on every rank (cheap, TP-summed via a CPU all-gather every N tokens in a debug mode); the design sums in rank order 0..3 on every rank |
| Sentinel collision | consumer hangs on a legitimately-produced `-0.0` | missing sanitize in a producer epilogue | unit test with inputs that round to `-0.0` in bf16 (e.g. `-1e-45`), and a fuzz over random bf16 including all-zero rows |
| Stale flags across replays / eager interleave | early pass -> stale data, or hang | epoch baked as a kernel argument at capture (AITER cookie bug) | sentinel: nothing to test; flag variant: alternate eager calls and graph replays of two different `T`, verify counters are 0 after each consumer |
| Wrap-around | after ~2^32 exchanges | monotone counters | flag variant resets counters per call; if epochs are ever used, unit-test with the epoch pre-set to `0xFFFFFFF0` |
| Slot reuse race at the cycle seam | corruption once per token at site 96 or 1 | fewer than 2 sites, or a rank skipping the last site (e.g. spec-decode draft path with different AR count) | run with MTP/DFlash on and off; assert equal AR-site counts across ranks at capture |
| Throughput regression at large T | c=48 slower | UC stores from a large-M GEMM; 4 x 245 KB polls | keep Step C for the small-M HIP GEMV only; Step B elsewhere; 12-tier bench per the no-corner-cutting rule |
| Early-return waves vs barrier | hang or UB in the flag variant | wvSplitK `threadIdx.y >= _WvPrGrp` early return | restructure to fall through; sentinel variant has no block barrier and is immune |

Instrumentation worth adding from day one: a per-site UC counter of spin iterations (max and
sum) read back in the profile table, so cross-rank skew becomes a first-class number instead of
"AR jitter" (the profile currently attributes 1.5-3.4 ms/token of jitter to AR at c=1).

## 7. Implementation notes / plumbing

- Allocation: extend `CustomAllreduce.__init__` (Python) with one more `create_shared_buffer(...,
  uncached=True)` of `96 x 4 x T_max x hidden x 2` bytes + a small debug region; exchange via the
  existing `allocate_shared_buffer_and_handle`/`open_mem_handle` pair. Fill with the sentinel
  with a tiny kernel (not `hipMemset`: 16-bit pattern).
- Site ids: assign at model construction (layer index x 2 + {attn, mlp}); pass as ints into the
  custom ops so they are constants under dynamo (the "Python shape dispatch freezes at trace
  time" lesson makes this a feature).
- Kernels: `gfx908_wv_fused.hip` gains `EPI 3` (push) with a `PushArgs{void* base[4]; int site,
  T_max, rank;}` kernarg; new `gfx908_ar_consume.hip` implementing 3.3 with a `PULL` template
  mode implementing 3.2. Both built through the same `torch.utils.cpp_extension` path as the
  existing extensions in `/opt/vllm-gfx908-ext`.
- Do not use `hipMemcpyAsync` anywhere on this path (SDMA queue is not ordered with the compute
  queue on gfx908 — see the `car_pool_copy_kernel` rationale in
  `/home/tyler/aiter/csrc/kernels/custom_all_reduce.cu:441`).
- Keep `VLLM_CUSTOM_ALLREDUCE_ALGO`-style env gating: `VLLM_GFX908_AR_FUSION={0,A,B,C}`, default
  off until the soak and GSM8K/PPL gates pass at production TP (kernel-parity rule).

## 8. References

In-tree:
- `/home/tyler/vllm-gfx908/csrc/custom_all_reduce.cuh` — 1-stage/2-stage kernels, graph buffer registration (`register_graph_buffers`, `get_graph_buffer_ipc_meta`).
- `/home/tyler/vllm-gfx908/csrc/custom_collective_common.cuh` — `Signal`, ROCm `barrier_at_start/_end` (`__scoped_atomic_*`, SYSTEM/DEVICE scopes).
- `/home/tyler/vllm-gfx908/csrc/libtorch_stable/custom_all_reduce.cu` — host `all_reduce` (staging memcpy), `hipDeviceMallocUncached` allocation.
- `/home/tyler/vllm-gfx908/vllm/distributed/device_communicators/custom_all_reduce.py` — gfx908 `registered=False` routing under capture; `capture()`/`register_graph_buffers`.
- `/home/tyler/vllm-gfx908/vllm/distributed/device_communicators/cuda_communicator.py:278` — AR backend selection order.
- `/home/tyler/vllm-gfx908/csrc/quickreduce/base.h:26,361` — gfx908 `glc` acquire bit, `set_sync_flag`/`wait_sync_flag`; `quick_reduce.h:139` device-side flag color (graph-safe epoch pattern).
- `/home/tyler/aiter/csrc/include/custom_all_reduce.cuh:161-262` (`start_sync`/`end_sync`, gfx908 mixed-size race), `:590` (`cross_device_reduce_2stage_write_mode`, push with nontemporal stores into registered output buffers), `:1395-1560` (fused AR+RMSNorm consumers), `:3680-3820` (cookie-under-capture problem, gfx908 naive-kernel forcing).
- `/home/tyler/aiter/csrc/kernels/custom_all_reduce.cu:441-460` — `car_pool_copy_kernel` (why no SDMA copies).
- `/home/tyler/vllm-gfx908/vllm/models/qwen4_exp/amd/csrc/gfx908_wv_fused.hip` — producer epilogue site (`C[m + y + n * M] = ...`).
- `/home/tyler/vllm-gfx908/vllm/models/qwen4_exp/amd/ops/hc.py:267-388` — consumer math to replicate.
- `/home/tyler/vllm-gfx908/vllm/models/qwen4_exp/amd/model.py:298-352`, `amd/hyperconnection.py:150-230` — per-layer AR/combine placement.
- `/home/tyler/vllm-gfx908/vllm/model_executor/models/qwen3_next.py:84,208,484-600` — sequence-parallel MoE path; shared-expert `reduce_results=False`.
- `/home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/test_e_persistent_car/` — E1-E4 soak harness to extend.
- `/home/tyler/.claude/jobs/bf4d1877/tmp/q4exp/ar_bench.py` — the 7.8 us / 19 us measurement harness.

External:
- LLVM AMDGPUUsage, "Memory Model GFX6-GFX9" and "Memory Model GFX90A" (code-sequence tables; MTYPE UC/NC/CC/RW and `buffer_wbl2`/`buffer_invl2` text): https://llvm.org/docs/AMDGPUUsage.html
- HIP coherence control (fine- vs coarse-grained, `hipDeviceMallocFinegrained`/`Uncached`): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html
- ROCm hardware atomics (peer memory in an IF hive treated as device memory): https://rocm.docs.amd.com/en/docs-6.4.2/reference/gpu-atomics-operation.html
- FlashInfer `trtllm_allreduce_fusion.cuh` (Lamport one-shot: `-0.0` sentinel, `flag%3` triple buffer, clear of `(flag+2)%3`): https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/comm/trtllm_allreduce_fusion.cuh
- TensorRT-LLM `allReduceFusionKernels.cu` (neg-zero sanitize, fused residual+RMSNorm after the spin): https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
- MSCCL++ (ASPLOS'26): LL protocol, MI300X results, "copy to all peers simultaneously on IF": https://arxiv.org/abs/2504.09014
- SiFAR (H200, dual buffering, speculative validation): https://arxiv.org/abs/2607.08973
- "Every us Matters" (GB200; push vs pull, sentinel/LL, SoL 1.4 us): https://arxiv.org/abs/2607.16100
- MI300A Infinity Fabric deep dive (remote load 690 ns, RCCL floor 20 us): https://arxiv.org/abs/2508.11298
- MI250X Infinity Fabric data movement (hipMemcpyPeer 8.7-18 us at 16 B): https://arxiv.org/abs/2410.00801
- AMD lab notes, MI200 memory space overview: https://github.com/amd/amd-lab-notes/blob/release/mi200-memory-space/Overview.md
