# Release: `vllm-rocm-gfx908:v0.19.2rc1.dev` (round-3 baked-in, 2026-04-26)

## Image

| Tag | Digest |
|---|---|
| `btbtyler09/vllm-rocm-gfx908:v0.19.2rc1.dev` | `sha256:c3c584ec338008240ddc557a898f0b745264efe521616d1aa4cc7f8e726c0178` |
| `btbtyler09/vllm-rocm-gfx908:latest` | `sha256:c3c584ec338008240ddc557a898f0b745264efe521616d1aa4cc7f8e726c0178` |

Built from `mi100-optimized` HEAD = `2ae323c98a219b9ae837b3fe56e68b121dc65f54` on top of `aiter-mi100:latest`. The 4 round-3 commits (`23bbb8696..2ae323c98`) are baked into the wheel — no runtime overlay mounts needed.

## What's new vs `:v0.19.2rc1` (pre-round-3)

Round-3 decode optimizations on Qwen3.6-35B-A3B-GPTQ-8bit benchmarked at **+25.7% throughput** vs round-2 (5h custom-op + 5j NCCL Tree+LL). See `docs/mi100_decode_opt/project_decode_opt_round3_2026_04_25.md` for the stage-by-stage methodology.

The 4 commits:

```
2ae323c98 [MI100] round-3 decode opt: inductor escape hatch + NCCL Tree+LL
fb72c78b8 [MI100] Stage 5 round-2 summary: Stage 5g LLMM1+wvSplitK = +40% throughput
7db67cd92 [MI100] Stage 5g: dispatch wvSplitK + LLMM1 in gfx908 unquantized GEMM path
23bbb8696 [MI100] Surgical AITER lm_head dispatch + custom-op bypass on gfx908
```

## Validated models (4×MI100 TP=4, `--mamba-cache-mode align`, `--dtype half`)

Benchmarked 2026-04-26 with `~/mi100-llm-testing/BenchAndReport.py` (12 scenarios each, full reports under `~/mi100-llm-testing/Model_Reports/benchmark_*_v0.19_round3.md`):

| Model | TPOT c=1 (ms) | tok/s c=1 | tok/s c=128 | Coherence pre/post |
|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B-GPTQ-8bit | 11.04 | 87.6 | 1371.0 | PASS / PASS |
| Qwen3.6-35B-A3B-GPTQ-4bit | 11.48 | 84.3 | 1297.9 | PASS / PASS |
| Qwen3.6-27B-GPTQ-8bit | 20.07 | 43.8 | 238.3 | PASS / PASS |
| Qwen3.6-27B-GPTQ-4bit | 17.55 | 49.3 | 243.0 | PASS / PASS |

Dense 27B is much slower than 35B-MoE at concurrency (expected: A3B routes only ~3B active params per token; dense 27B reads the full model). 4bit/8bit deltas are small for 27B and 35B alike.

## Recommended `docker run`

No overlay mounts needed. Verified equivalent to overlay-mounted boot (no-overlay 35B-8bit smoke: TPOT 11.00ms / 90.9 tok/s, coherence 4/4 PASS).

```
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
  -v /path/to/your/models:/models:ro \
  btbtyler09/vllm-rocm-gfx908:v0.19.2rc1.dev \
  vllm serve /models/<MODEL_DIR> \
    --served-model-name <served-name> \
    --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --attention-backend TRITON_ATTN \
    --mamba-cache-mode align \
    --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

`--mamba-cache-mode align` is required for Qwen3.5/3.6 hybrid attention. `--dtype half` is required for any GPTQ variant (default bf16 is incompatible with the GPTQ kernels). `NCCL_ALGO=Tree NCCL_PROTO=LL` is the round-3j tuning.

## Known unpushed source

The 4 round-3 commits are baked into the image but **not yet pushed to `origin/mi100-optimized`** as of release. Push-back-pending: `git push origin mi100-optimized` from `/home/tyler/vllm-gfx908`. The image is reproducible from the source as-is on this host.
