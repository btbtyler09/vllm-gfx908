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

### Open levers
- Quantize the bf16 GDN projections (Quantizer-side): ~3.5 ms/token at c=1.
- Hand-written W4 GEMV (dense + MoE).
- Fuse the ~35 glue kernels per layer (HC combine/mix/silu, MoE align/sum, fills/copies).
- All-reduce count (2 per layer, latency-bound at 37 µs).
