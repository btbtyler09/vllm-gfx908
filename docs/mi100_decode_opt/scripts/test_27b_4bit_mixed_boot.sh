#!/usr/bin/env bash
# Light-touch boot validation for Qwen3.6-27B-GPTQ-4bit_mixed (mixed-precision GPTQ:
# bulk INT4, INT8 on quant-sensitive projections via dynamic overrides).
#
# Confirms the model loads cleanly after the fused-name dynamic-field fix
# (in_proj_qkvz added to dynamic{} so vLLM allocates the fused MergedColumnParallelLinear
# buffer at 8-bit pack_factor instead of defaulting to 4-bit).
#
# Light-touch: boot, coherence pre, 3 chats (factual/code/reasoning), coherence post.
# No 12-scenario BenchAndReport — that's a follow-up if this boot succeeds.
set -eu

IMG="${1:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-27b-4bit-mixed"
LOG="/tmp/decode_opt/serve_27b_mixed.log"
CHAT_DIR="/tmp/decode_opt"

mkdir -p "$CHAT_DIR"

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > "$LOG"

echo "=== launch $IMG ==="
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
  -v /home/tyler/vllm-gfx908/vllm/platforms/rocm.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/distributed/device_communicators/custom_all_reduce.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py:ro \
  "$IMG" \
  vllm serve /models/Qwen3.6-27B-GPTQ-4bit_mixed \
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
if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" 27b_mixed_pre; then
  echo "FAIL: pre-bench coherence"
  kill $TAILER_PID 2>/dev/null || true
  exit 2
fi

# Three diverse chat completions to verify generation quality across task types.
do_chat() {
  local label="$1"; shift
  local prompt="$1"; shift
  local out="$CHAT_DIR/27b_mixed_chat_${label}.txt"
  echo ""
  echo "=== chat: $label ==="
  echo "PROMPT: $prompt" > "$out"
  echo "---" >> "$out"
  resp=$(curl -s -m 90 -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$SERVED" --arg p "$prompt" '{
      model: $m,
      messages: [{role:"user", content:$p}],
      max_tokens: 256, temperature: 0.7,
      chat_template_kwargs: {enable_thinking: false}
    }')" \
    http://localhost:8000/v1/chat/completions)
  content=$(echo "$resp" | jq -r '.choices[0].message.content // "(empty)"')
  ctok=$(echo "$resp" | jq -r '.usage.completion_tokens // 0')
  echo "$content" >> "$out"
  echo "(tokens: $ctok)" >> "$out"
  echo "$content" | head -8
  echo "  [... ${ctok} tok total, full response in $out]"
}

do_chat factual "What is the capital of France? Answer in one short sentence."
do_chat code "Write a Python function fib(n) that returns the first n Fibonacci numbers as a list."
do_chat reasoning "If I have 17 marbles and give 5 to each of 3 friends, how many marbles do I have left? Show your reasoning step by step."

echo ""
echo "=== post-bench coherence ==="
if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" 27b_mixed_post; then
  echo "FAIL: post-bench coherence"
  kill $TAILER_PID 2>/dev/null || true
  exit 3
fi

echo ""
echo "=== DONE — model loads + generates coherent output. Container left running for inspection. ==="
echo "(use 'docker rm -f decode_opt' to tear down)"
kill $TAILER_PID 2>/dev/null || true
