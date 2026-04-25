#!/usr/bin/env bash
# Stage F — combined runner: Stage 5h custom-op overlay + Stage 5f TunableOp replay.
# Same shape as test_stage5_baseline.sh but always sets TUNABLEOP=replay (read-only).
# Usage: bash test_stage_combined.sh [image]
set -eu

IMG="${1:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-35b-8bit"

if [[ ! -f /home/tyler/vllm-gfx908/docs/mi100_decode_opt/tunableop_results/tunableop_results.csv ]]; then
  echo "ERROR: /home/tyler/vllm-gfx908/docs/mi100_decode_opt/tunableop_results/tunableop_results.csv not found." >&2
  echo "Run 'TUNABLEOP=tune bash test_stage5_baseline.sh' first." >&2
  exit 1
fi

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > /tmp/decode_opt/serve_combined.log

echo "=== launch $IMG (Stage 5h overlay + TunableOp replay) ==="
docker run -d --name decode_opt \
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
  --env PYTORCH_TUNABLEOP_ENABLED=1 \
  --env PYTORCH_TUNABLEOP_TUNING=0 \
  --env PYTORCH_TUNABLEOP_FILENAME=/host_tunableop/tunableop_results.csv \
  -v /home/tyler/vllm-gfx908/docs/mi100_decode_opt/tunableop_results:/host_tunableop:ro \
  -v /home/tyler/.cache/huggingface:/huggingface \
  -v /home/tyler/quantize/quant:/models:ro \
  -v /home/tyler/vllm-gfx908/vllm/platforms/rocm.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/distributed/device_communicators/custom_all_reduce.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py:ro \
  "$IMG" \
  vllm serve /models/Qwen3.6-35B-A3B-GPTQ-8bit \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --attention-backend TRITON_ATTN \
    --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'

nohup docker logs -f decode_opt > /tmp/decode_opt/serve_combined.log 2>&1 &
TAILER_PID=$!
disown $TAILER_PID

echo "=== waiting for startup (up to 15min) ==="
for i in $(seq 1 180); do
  if grep -qE 'Application startup complete|Uvicorn running' /tmp/decode_opt/serve_combined.log 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if ! docker ps --filter name=decode_opt -q | grep -q .; then
    echo "FAIL: container exited during startup"
    tail -50 /tmp/decode_opt/serve_combined.log
    kill $TAILER_PID 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

echo ""
echo "=== pre-bench coherence ==="
if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" combined_pre; then
  echo "FAIL: pre-bench coherence"
  kill $TAILER_PID 2>/dev/null || true
  exit 2
fi

echo ""
echo "=== TPOT 3-run set (256-tok decode, c=1) ==="
for i in 1 2 3; do
  t=$(date +%s.%N)
  ct=$(curl -s -m 60 -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$SERVED" '{
      model: $m,
      messages: [{role:"user", content:"Write a 200-word essay about the industrial revolution."}],
      max_tokens: 256, temperature: 0.7,
      chat_template_kwargs: {enable_thinking: false}
    }')" \
    http://localhost:8000/v1/chat/completions | jq -r '.usage.completion_tokens // 0')
  e=$(date +%s.%N)
  if [[ "$ct" -gt 50 ]]; then
    dur=$(python3 -c "print(f'{$e - $t:.3f}')")
    tps=$(python3 -c "print(f'{$ct / ($e - $t):.2f}')")
    tpot=$(python3 -c "print(f'{($e - $t) / $ct * 1000:.2f}')")
    echo "run $i: ${ct} tok in ${dur}s = ${tps} tok/s, TPOT=${tpot}ms"
  fi
done

echo ""
echo "=== post-bench coherence ==="
if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" combined_post; then
  echo "FAIL: post-bench coherence (drift)"
  kill $TAILER_PID 2>/dev/null || true
  exit 3
fi

echo ""
echo "=== DONE — leaving container running for full BenchAndReport ==="
kill $TAILER_PID 2>/dev/null || true
