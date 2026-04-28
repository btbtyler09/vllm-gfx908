#!/usr/bin/env bash
# Stage 2: AITER gemm_a16w16 dispatch for lm_head + linear_attn.in_proj_z
# Mounts patched utils.py (whitelist extended) + rocm.py (env var flipped)
# over the baseline vllm-rocm-gfx908:latest image. No rebuild needed.
set -eu

IMG="${1:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-35b-8bit"

echo "=== teardown any prior container ==="
docker rm -f decode_opt 2>/dev/null || true
: > /tmp/decode_opt/serve_stage2.log

echo "=== launch $IMG with utils.py + rocm.py overlays + AITER_TRITON_LOG_LEVEL=INFO ==="
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
  --env AITER_TRITON_LOG_LEVEL=INFO \
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

nohup docker logs -f decode_opt > /tmp/decode_opt/serve_stage2.log 2>&1 &
disown

echo "=== waiting for startup (up to 15min) ==="
for i in $(seq 1 180); do
  if grep -qE 'Application startup complete|Uvicorn running' /tmp/decode_opt/serve_stage2.log 2>/dev/null; then
    echo "READY after ${i}x5s"
    break
  fi
  if ! docker ps --filter name=decode_opt -q | grep -q .; then
    echo "FAIL: container exited during startup"
    tail -50 /tmp/decode_opt/serve_stage2.log
    exit 1
  fi
  sleep 5
done

echo ""
echo "=== verify AITER gemm_a16w16 dispatch fired for lm_head ==="
# Send 1 token request to force a decode step that hits lm_head
curl -s -m 30 -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "$SERVED" '{
    model: $m,
    messages: [{role:"user", content:"Hi"}],
    max_tokens: 8, temperature: 0.0,
    chat_template_kwargs: {enable_thinking: false}
  }')" \
  http://localhost:8000/v1/chat/completions > /dev/null
sleep 2
LM_HEAD_HITS=$(grep -c 'GEMM_A16W16: x=(1, 2048) w=(62080, 2048)' /tmp/decode_opt/serve_stage2.log || true)
echo "lm_head shape (62080, 2048): $LM_HEAD_HITS dispatches"
if [[ "$LM_HEAD_HITS" -lt 1 ]]; then
  echo "FAIL: AITER dispatch not engaging for lm_head"
  exit 4
fi
echo "PASS: AITER dispatch confirmed active for lm_head"

echo ""
echo "=== pre-bench coherence (expect PASS) ==="
if ! /tmp/decode_opt/coherence.sh "$SERVED" stage2_pre; then
  echo "FAIL: pre-bench coherence failed — accuracy regression from AITER kernel"
  exit 2
fi

echo ""
echo "=== TPOT measurement (c=1, 256 tok decode), 3 runs ==="
for i in $(seq 1 3); do
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
    echo "run $i: completion_tokens=${ct} (too short, skipping)"
  fi
done

echo ""
echo "=== post-bench coherence (expect PASS — no drift) ==="
if ! /tmp/decode_opt/coherence.sh "$SERVED" stage2_post; then
  echo "FAIL: post-bench coherence failed — drift under sustained load"
  exit 3
fi

echo ""
echo "=== DONE — Stage 2 complete ==="
