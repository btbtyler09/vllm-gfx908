# Round-4 Final Summary (2026-04-27)

## Outcome

| Metric (Qwen3.6-35B-A3B-GPTQ-8bit, 4×MI100 TP=4) | Round-3 baseline | Round-4 final | Δ |
|---|---:|---:|---:|
| TPOT @ c=1 (Single Latency) | 11.04 ms | 9.59 ms | **−13.1%** |
| tok/s @ c=1 (Single Latency) | 87.6 | ≥99.97 | +14.1% |
| tok/s @ c=1 (Decode Stress) | — | 105.91 | crossed 100 mark |
| Wall-clock TPOT (smoke direct) | — | 8.75 ms / 114.2 tok/s | matches profile |

**Cumulative round-1 → round-4: +115% c=1 throughput.**

All 12 BenchAndReport tiers improved or held vs round-3. Coherence 4/4 PASS at every gate (pre, mid, post-bench, smoke).

## What shipped

### Phase 2 E — Mori-pattern persistent-handle CAR (commit `134809438`)

Python-only fix in `vllm/distributed/device_communicators/custom_all_reduce.py`. Routes gfx908 captured all-reduce through `self.all_reduce(input, registered=False)` (uncached `buffer_ptrs[rank]`) instead of `registered=True` (cached IPC view). Resolves the round-2 NaN-under-replay bug.

**TPOT 11.04 → 9.97 ms (−9.7%).**

### Phase 3 B1'-Tune — fused_moe_wna16 W8 autotune JSON (commit `f7a614bd2`)

One JSON file, no code changes:
`vllm/model_executor/layers/fused_moe/configs/E=256,N=128,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int8_w8a16.json`

W4 MI100 reference config ported verbatim with one knob change: M=16 entry's `waves_per_eu` 1 → 2 (W4's value broke W8's c=16 by 4.8% in cand_b round 1; v2 fix recovered to +0.5%).

**TPOT 9.97 → 9.59 ms (−3.8%).**

## Phase 4 audit — closed without further ship

Live torch-profile capture against the shipped image (2026-04-27), per-bucket breakdown via `parse_profile.py`, comparison vs round-3 baseline trace. Full audit: `round4_phase4_profile_audit.md`.

**Headline finding:** the original "elephant" — `MT128x128x32` rocBLAS GEMM at 43.8% of round-3 baseline — **is already 70% smaller** in the shipped round-4 image. Round-3 stage 5h's custom-op inductor escape hatch silently switched dispatch from rocBLAS Tensile (wasted M=1 tile) to vLLM's `LLGemm1_kernel` (custom skinny GEMM). We never measured this — round-4 has been benefiting silently.

**No "single config file ship" lever remains** comparable to B1'. The biggest remaining single contributor (`LLGemm1_kernel` at 16.7%) is a custom skinny-GEMM kernel where per-call cost (8.3 µs) is dominated by **HIP launch overhead**, not math (math is ~50 ns at MFMA peak). Reducing call count via Python-side fusion is **not possible** — `QKVParallelLinear` and `MergedColumnParallelLinear` already fuse Q+K+V and gate+up at the model architecture level.

Round-4 closed. No further phases shipped.

## Image

| Tag | Digest |
|---|---|
| `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` | `sha256:143f37f906ff5444ad2cc29bfb767e0176b6328fddce25b10977e08fefabe711` |
| `btbtyler09/vllm-rocm-gfx908:latest` | `sha256:143f37f906ff5444ad2cc29bfb767e0176b6328fddce25b10977e08fefabe711` |

Built from `mi100-optimized` HEAD `f7a614bd2` on top of `aiter-mi100:latest`. Both round-4 commits (`134809438` Phase 2 E + `f7a614bd2` Phase 3 B1'-Tune) baked into the wheel — no overlay mounts needed.

## Recommended `docker run`

```bash
docker run -d --name vllm \
  --network=host --cpuset-cpus="0-11" --group-add=video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 \
  --env HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  --env HF_HOME=/huggingface \
  --env VLLM_ROCM_USE_AITER=1 \
  --env VLLM_MI100_TORCH_COMPILE=1 \
  --env VLLM_ROCM_USE_AITER_TRITON_GEMM=1 \
  --env NCCL_ALGO=Tree --env NCCL_PROTO=LL \
  -v $HOME/.cache/huggingface:/huggingface \
  -v $HOME/quantize/quant:/models:ro \
  btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev \
  vllm serve "/models/Qwen3.6-35B-A3B-GPTQ-8bit" \
    --served-model-name qwen3.6-35b-8bit \
    --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --attention-backend TRITON_ATTN \
    --mamba-cache-mode align \
    --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

## Round-5 entry points

The audit identified the dispatch-level wins are exhausted on the **35B-A3B-GPTQ-8bit** target. Remaining levers require kernel engineering (multi-day risk). The most plausible round-5 targets:

1. **Dense GPTQ-8 models (27B-8bit etc.) — original B1**: `TritonW8A16LinearKernel`. The original B1 (extend W4 LinearKernel to W8) was falsified for the 35B-A3B target (attention is fp16 on that model). On dense GPTQ-8 models where attention IS quantized, every QKV/output projection × ~64 layers falls to `ExllamaLinearKernel` (scalar dequant, no MFMA). This is an unmeasured-but-documented lever for round-5.
2. **Lever F — custom gfx908 MFMA fmoe kernel**: 4-5% TPOT, 2-3 days. Modest but tractable on the 35B-A3B target.
3. **Lever #2 — custom HIP MFMA M=1 kernel** (replacing LLGemm1): theoretical 6-10%, 1-2 weeks of HIP work, **gated on a 1-day microbench** to confirm hand-rolled M=1 MFMA can beat the ~5-8 µs HIP launch floor on gfx908.

## Critical files

- `vllm/distributed/device_communicators/custom_all_reduce.py` — Phase 2 E (CAR fix)
- `vllm/model_executor/layers/fused_moe/configs/E=256,N=128,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int8_w8a16.json` — Phase 3 B1' ship
- `docs/mi100_decode_opt/round4_phase3_audit.md` — Phase 3 audit
- `docs/mi100_decode_opt/round4_phase4_profile_audit.md` — Phase 4 profile audit
- `docs/mi100_decode_opt/scripts/test_e_persistent_car/` — E isolated tests (smoke gate for any future CAR change on gfx908)
- `~/mi100-llm-testing/Model_Reports/benchmark_Qwen3.6-35B-A3B-GPTQ-8bit_v0.20_round4_phase3_b1prime_cand_b_v2.md` — round-4 final bench

## Pending closeout (optional)

- Re-bench other 3 production models (35B-4bit, 27B-8bit, 27B-4bit) against `:v0.20.0rc1.dev` to confirm no regression. The W8 wna16 JSON only fires on lookup key `E=256,N=128,...int8_w8a16` so other models won't load it (different E/N or dtype) — round-3 results should hold modulo Phase 2 E gain on all-reduce.
