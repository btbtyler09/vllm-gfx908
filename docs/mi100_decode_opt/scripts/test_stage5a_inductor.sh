#!/usr/bin/env bash
# Stage 5a: capture inductor compile output to verify GEMM fusion
# Sets TORCH_COMPILE_DEBUG=1 + TORCHINDUCTOR_TRACE so inductor dumps its
# generated kernels to a host-mounted dir for inspection.
set -eu

IMG="${1:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-35b-8bit"

OUTDIR=/tmp/decode_opt/inductor_trace
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > /tmp/decode_opt/serve_stage5a.log

echo "=== launch $IMG with inductor debug enabled, trace -> $OUTDIR ==="
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
  --env TORCH_COMPILE_DEBUG=1 \
  --env TORCHINDUCTOR_TRACE_RANK=0 \
  --env TORCHINDUCTOR_TRACE=/inductor_trace \
  -v "$OUTDIR":/inductor_trace \
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

nohup docker logs -f decode_opt > /tmp/decode_opt/serve_stage5a.log 2>&1 &
TAILER_PID=$!
disown $TAILER_PID

echo "=== waiting for startup (up to 20min — inductor debug is slower) ==="
for i in $(seq 1 240); do
  if grep -qE 'Application startup complete|Uvicorn running' /tmp/decode_opt/serve_stage5a.log 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if ! docker ps --filter name=decode_opt -q | grep -q .; then
    echo "FAIL: container exited during startup"
    tail -50 /tmp/decode_opt/serve_stage5a.log
    exit 1
  fi
  sleep 5
done

echo ""
echo "=== send 1 decode request ==="
curl -s -m 60 -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$SERVED" '{
    model: $m,
    messages: [{role:"user", content:"Hi"}],
    max_tokens: 16, temperature: 0.0,
    chat_template_kwargs: {enable_thinking: false}
  }')" \
  http://localhost:8000/v1/chat/completions > /dev/null

echo ""
echo "=== inductor trace contents ==="
find "$OUTDIR" -type f -name 'output_code.py' | head -10
echo ""
echo "=== count of fused/unfused kernels in output_code.py files ==="
TOTAL_FUSED=$(grep -hcE 'def triton_(red_)?fused' "$OUTDIR"/**/output_code.py 2>/dev/null | awk '{s+=$1} END {print s+0}')
TOTAL_EXTERN=$(grep -hcE 'extern_kernels\.(addmm|mm|bmm|baddbmm)' "$OUTDIR"/**/output_code.py 2>/dev/null | awk '{s+=$1} END {print s+0}')
echo "fused-kernel definitions: $TOTAL_FUSED"
echo "extern_kernels mm/addmm/bmm calls: $TOTAL_EXTERN"

echo ""
echo "=== shutting down ==="
kill $TAILER_PID 2>/dev/null || true
docker rm -f decode_opt
