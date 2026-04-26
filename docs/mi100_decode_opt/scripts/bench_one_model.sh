#!/usr/bin/env bash
# Per-model BenchAndReport runner for the round-3 multi-model validation.
# Boots the container with round-3 overlays, runs the 12-scenario suite via
# BenchAndReport.py inside the container, saves the report to the host
# mi100-llm-testing/Model_Reports dir, then tears down.
#
# Usage: bench_one_model.sh <model-dir-name> <served-name> <report-suffix>
#   e.g. bench_one_model.sh Qwen3.6-35B-A3B-GPTQ-8bit qwen3.6-35b-8bit Qwen3.6-35B-A3B-GPTQ-8bit
set -eu

MODEL_DIR="${1:?usage: $0 <model-dir-name> <served-name> <report-suffix>}"
SERVED="${2:?usage: $0 <model-dir-name> <served-name> <report-suffix>}"
REPORT_SUFFIX="${3:?usage: $0 <model-dir-name> <served-name> <report-suffix>}"

IMG="${IMG:-vllm-rocm-gfx908:latest}"
LOG="/tmp/decode_opt/serve_${SERVED}.log"
REPORTS_DIR="/home/tyler/mi100-llm-testing/Model_Reports"
REPORT_PATH_HOST="${REPORTS_DIR}/benchmark_${REPORT_SUFFIX}_v0.19_round3.md"
REPORT_PATH_CONT="/bench/Model_Reports/benchmark_${REPORT_SUFFIX}_v0.19_round3.md"

mkdir -p /tmp/decode_opt "$REPORTS_DIR"

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > "$LOG"

echo "=== launch $IMG for $SERVED ==="
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
  -v /home/tyler/.cache/huggingface:/huggingface \
  -v /home/tyler/quantize/quant:/models:ro \
  -v /home/tyler/mi100-llm-testing:/bench \
  -v /home/tyler/vllm-gfx908/vllm/platforms/rocm.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/distributed/device_communicators/custom_all_reduce.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/v1/attention/ops:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops:ro \
  "$IMG" \
  vllm serve "/models/${MODEL_DIR}" \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --attention-backend TRITON_ATTN \
    --mamba-cache-mode align \
    --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'

nohup docker logs -f decode_opt > "$LOG" 2>&1 &
TAILER_PID=$!
disown $TAILER_PID

echo "=== waiting for startup (up to 15min) ==="
for i in $(seq 1 180); do
  if grep -qE 'Application startup complete|Uvicorn running' "$LOG" 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if ! docker ps --filter name=decode_opt -q | grep -q .; then
    echo "FAIL: container exited during startup"
    tail -120 "$LOG"
    kill $TAILER_PID 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

if ! grep -qE 'Application startup complete|Uvicorn running' "$LOG" 2>/dev/null; then
  echo "FAIL: 15min timeout waiting for startup"
  tail -120 "$LOG"
  kill $TAILER_PID 2>/dev/null || true
  exit 1
fi

echo ""
echo "=== priming request (warmup) ==="
curl -s -m 60 -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$SERVED" '{
    model: $m,
    messages: [{role:"user", content:"Hi"}],
    max_tokens: 8, temperature: 0.0,
    chat_template_kwargs: {enable_thinking: false}
  }')" \
  http://localhost:8000/v1/chat/completions > /dev/null
echo "priming done"

echo ""
echo "=== pre-bench coherence ==="
if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" "${SERVED}_pre"; then
  echo "FAIL: pre-bench coherence"
  kill $TAILER_PID 2>/dev/null || true
  exit 2
fi

echo ""
echo "=== running BenchAndReport.py inside container ==="
docker exec decode_opt python3 /bench/BenchAndReport.py \
  --model "$SERVED" \
  --tokenizer "/models/${MODEL_DIR}" \
  --base-url http://localhost:8000 \
  --hardware "${HARDWARE:-4x AMD MI100 (gfx908) - Round-3 ship: 5h custom-op + 5j NCCL Tree+LL}" \
  --output "$REPORT_PATH_CONT" \
  --scaffolded-report \
  --save-results 2>&1 | tee "/tmp/decode_opt/bench_${SERVED}.log"

echo ""
echo "=== post-bench coherence ==="
if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" "${SERVED}_post"; then
  echo "FAIL: post-bench coherence"
  kill $TAILER_PID 2>/dev/null || true
  exit 3
fi

echo ""
echo "=== teardown ==="
docker rm -f decode_opt 2>/dev/null || true
kill $TAILER_PID 2>/dev/null || true

if [[ -f "$REPORT_PATH_HOST" ]]; then
  echo "=== DONE — report saved: $REPORT_PATH_HOST ==="
else
  echo "WARN: report not found at $REPORT_PATH_HOST"
  exit 4
fi
