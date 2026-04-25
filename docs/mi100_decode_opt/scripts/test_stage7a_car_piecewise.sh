#!/usr/bin/env bash
# Stage 7a-piecewise: test if CAR works under PIECEWISE-only cudagraph (no FULL).
# Hypothesis: FULL_AND_PIECEWISE captures whole-decode-step graphs that include
# the all-reduce ops. If we disable FULL capture and use PIECEWISE only, the
# all-reduce happens BETWEEN piecewise sub-graphs (in eager Python), and CAR
# can run without the graph-replay path. Tests whether the graph capture itself
# is the trigger.
set -eu

IMG="${1:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-35b-8bit"

LOG=/tmp/decode_opt/serve_stage7a_piecewise.log

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > "$LOG"

echo "=== launch with no-bypass CAR overlay + PIECEWISE-only cudagraph ==="
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
  -v /home/tyler/.cache/huggingface:/huggingface \
  -v /home/tyler/quantize/quant:/models:ro \
  -v /home/tyler/vllm-gfx908/vllm/platforms/rocm.py:/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py:ro \
  -v /home/tyler/vllm-gfx908/vllm/model_executor/layers/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py:ro \
  -v /tmp/decode_opt/custom_all_reduce_no_bypass.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py:ro \
  "$IMG" \
  vllm serve /models/Qwen3.6-35B-A3B-GPTQ-8bit \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 4 --dtype half --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --attention-backend TRITON_ATTN \
    --compilation-config '{"mode": 3, "cudagraph_mode": "PIECEWISE"}'

nohup docker logs -f decode_opt > "$LOG" 2>&1 &
TAILER_PID=$!
disown $TAILER_PID

echo "=== waiting for startup ==="
for i in $(seq 1 180); do
  if grep -qE 'Application startup complete|Uvicorn running' "$LOG" 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if ! docker ps --filter name=decode_opt -q | grep -q .; then
    echo "FAIL: container exited during startup"
    tail -50 "$LOG"
    kill $TAILER_PID 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

echo ""
echo "=== priming + immediate coherence ==="
curl -s -m 30 -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$SERVED" '{
    model: $m,
    messages: [{role:"user", content:"Hi"}],
    max_tokens: 8, temperature: 0.0,
    chat_template_kwargs: {enable_thinking: false}
  }')" \
  http://localhost:8000/v1/chat/completions > /dev/null

if /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" stage7a_piecewise_pre; then
  COHERENCE_PRE=PASS
else
  COHERENCE_PRE=FAIL
fi
echo "COHERENCE PRE: $COHERENCE_PRE"

if [[ "$COHERENCE_PRE" == "PASS" ]]; then
  echo ""
  echo "=== TPOT 3-run set ==="
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
    else
      echo "run $i: completion_tokens=${ct}"
    fi
  done
  echo ""
  echo "=== post-bench coherence ==="
  if /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh "$SERVED" stage7a_piecewise_post; then
    echo "COHERENCE POST: PASS"
  else
    echo "COHERENCE POST: FAIL"
  fi
fi

kill $TAILER_PID 2>/dev/null || true
