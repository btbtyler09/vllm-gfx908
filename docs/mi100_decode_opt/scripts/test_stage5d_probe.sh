#!/usr/bin/env bash
# Stage 5d: shape probe — record per-shape GEMM call counts during decode.
# Mounts utils.py (with _probe_record) + sets VLLM_GFX908_PROBE_SHAPES=N.
# Sends a single 96-tok decode request to capture ~96 layers × per-step calls.
set -eu

IMG="${1:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-35b-8bit"
PROBE_TARGET="${2:-3000}"  # capture enough calls to span several decode steps

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > /tmp/decode_opt/serve_stage5d.log

echo "=== launch $IMG with utils.py overlay + VLLM_GFX908_PROBE_SHAPES=$PROBE_TARGET ==="
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
  --env VLLM_GFX908_PROBE_SHAPES="$PROBE_TARGET" \
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

nohup docker logs -f decode_opt > /tmp/decode_opt/serve_stage5d.log 2>&1 &
TAILER_PID=$!
disown $TAILER_PID

echo "=== waiting for startup (up to 15min) ==="
for i in $(seq 1 180); do
  if grep -qE 'Application startup complete|Uvicorn running' /tmp/decode_opt/serve_stage5d.log 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if ! docker ps --filter name=decode_opt -q | grep -q .; then
    echo "FAIL: container exited during startup"
    tail -50 /tmp/decode_opt/serve_stage5d.log
    exit 1
  fi
  sleep 5
done

echo ""
echo "=== send 96-token decode request to populate probe ==="
curl -s -m 60 -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$SERVED" '{
    model: $m,
    messages: [{role:"user", content:"Count to 30."}],
    max_tokens: 96, temperature: 0.7,
    chat_template_kwargs: {enable_thinking: false}
  }')" \
  http://localhost:8000/v1/chat/completions > /tmp/decode_opt/probe_response.json
echo "completion_tokens: $(jq -r '.usage.completion_tokens' /tmp/decode_opt/probe_response.json)"

sleep 3

echo ""
echo "=== probe dump ==="
grep -E "PROBE_SHAPES|AITER_DISPATCH" /tmp/decode_opt/serve_stage5d.log | tail -100

echo ""
echo "=== shutting down container ==="
kill $TAILER_PID 2>/dev/null || true
docker rm -f decode_opt
