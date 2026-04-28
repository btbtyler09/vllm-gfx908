#!/usr/bin/env bash
# Round-4 Phase 1 lever A — residual NCCL/RCCL sweep.
# For each variant: boot container, coherence pre, 3-run c=1 TPOT,
# coherence post, teardown. Append result line to SUMMARY.
#
# Variants (one-at-a-time, on top of round-3 defaults Tree+LL):
#   baseline      — round-3 defaults only (sanity reference)
#   BUFFSIZE_8M   — NCCL_BUFFSIZE=8388608
#   BUFFSIZE_16M  — NCCL_BUFFSIZE=16777216
#   P2P_NVL       — NCCL_P2P_LEVEL=NVL  (force P2P, equivalent of XGMI peer)
#   P2P_SYS       — NCCL_P2P_LEVEL=SYS  (allow any path, may regress)
#   MINCHANS_4    — NCCL_MIN_NCHANNELS=4
#   NET_GDR_0     — NCCL_NET_GDR_LEVEL=0  (no GDR — irrelevant single-node, sanity)
#
# Per "no regression across test map" rule, anything showing ≥0.5% c=1 win
# here is ONLY a candidate — it must pass the full BenchAndReport before
# being baked into _GFX908_DEFAULTS.
set -u

IMG="${IMG:-vllm-rocm-gfx908:latest}"
SERVED="qwen3.6-35b-8bit"
MODEL_DIR="Qwen3.6-35B-A3B-GPTQ-8bit"
SUMMARY="/tmp/decode_opt/round4_phase1_a_summary.txt"

mkdir -p /tmp/decode_opt
{
  echo "# Round-4 Phase 1 lever A sweep"
  echo "# started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# image:   $IMG"
  echo "# model:   $SERVED"
  echo "# Round-3 baseline reference: TPOT 11.04 ms / 87.6 tok/s c=1"
  echo
  printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-7s | %s\n" \
    "VARIANT" "TPOT_run1" "TPOT_run2" "TPOT_run3" "TPOT_avg" "COH" "NOTES"
  echo "----------------+------------+------------+------------+------------+---------+----------"
} > "$SUMMARY"

run_variant() {
  local LABEL="$1" ; shift
  local EXTRA_ENV="$*"

  local LOG=/tmp/decode_opt/sweep_${LABEL}.log
  : > "$LOG"

  echo
  echo "================================================================"
  echo "=== variant: $LABEL"
  echo "=== extra env: $EXTRA_ENV"
  echo "================================================================"

  docker rm -f decode_opt 2>/dev/null || true

  # shellcheck disable=SC2086
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
    $EXTRA_ENV \
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
      --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    > /dev/null

  nohup docker logs -f decode_opt > "$LOG" 2>&1 &
  local TAILER_PID=$!
  disown $TAILER_PID

  echo "waiting for startup (up to 15 min)…"
  local READY=0
  for i in $(seq 1 180); do
    if grep -qE 'Application startup complete|Uvicorn running' "$LOG" 2>/dev/null; then
      echo "READY after ${i}x5s"
      READY=1
      break
    fi
    if ! docker ps --filter name=decode_opt -q | grep -q .; then
      echo "FAIL: container exited during startup"
      tail -120 "$LOG"
      kill $TAILER_PID 2>/dev/null || true
      printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-7s | %s\n" \
        "$LABEL" "—" "—" "—" "—" "BOOT" "container exited" >> "$SUMMARY"
      return 1
    fi
    sleep 5
  done
  if [[ $READY -ne 1 ]]; then
    echo "FAIL: 15min timeout"
    tail -120 "$LOG"
    docker rm -f decode_opt 2>/dev/null || true
    kill $TAILER_PID 2>/dev/null || true
    printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-7s | %s\n" \
      "$LABEL" "—" "—" "—" "—" "TMOUT" "no startup in 15min" >> "$SUMMARY"
    return 1
  fi

  echo "priming…"
  curl -s -m 60 -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$SERVED" '{
      model: $m,
      messages: [{role:"user", content:"Hi"}],
      max_tokens: 8, temperature: 0.0,
      chat_template_kwargs: {enable_thinking: false}
    }')" \
    http://localhost:8000/v1/chat/completions > /dev/null

  echo "coherence pre…"
  local COH_PRE=PASS
  if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh \
      "$SERVED" "sweep_${LABEL}_pre" > /tmp/decode_opt/sweep_${LABEL}_coh_pre.log 2>&1; then
    COH_PRE=FAIL
  fi

  echo "TPOT 3 runs at c=1…"
  local TPOTS=()
  for i in 1 2 3; do
    local t=$(date +%s.%N)
    local ct=$(curl -s -m 90 -H 'Content-Type: application/json' \
      -d "$(jq -nc --arg m "$SERVED" '{
        model: $m,
        messages: [{role:"user", content:"Write a 200-word essay about the industrial revolution."}],
        max_tokens: 256, temperature: 0.7,
        chat_template_kwargs: {enable_thinking: false}
      }')" \
      http://localhost:8000/v1/chat/completions | jq -r '.usage.completion_tokens // 0')
    local e=$(date +%s.%N)
    local tpot="—"
    if [[ "$ct" -gt 50 ]]; then
      tpot=$(python3 -c "print(f'{($e - $t) / $ct * 1000:.2f}')")
      local tps=$(python3 -c "print(f'{$ct / ($e - $t):.2f}')")
      echo "  run $i: ${ct} tok, TPOT=${tpot}ms, ${tps} tok/s"
    else
      echo "  run $i: completion_tokens=${ct} (likely degenerate)"
    fi
    TPOTS+=("$tpot")
  done

  local AVG="—"
  if [[ "${TPOTS[0]}" != "—" && "${TPOTS[1]}" != "—" && "${TPOTS[2]}" != "—" ]]; then
    AVG=$(python3 -c "print(f'{(${TPOTS[0]} + ${TPOTS[1]} + ${TPOTS[2]}) / 3:.2f}')")
  fi

  echo "coherence post…"
  local COH_POST=PASS
  if ! /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/coherence.sh \
      "$SERVED" "sweep_${LABEL}_post" > /tmp/decode_opt/sweep_${LABEL}_coh_post.log 2>&1; then
    COH_POST=FAIL
  fi

  printf "%-15s | %-10s | %-10s | %-10s | %-10s | %-3s/%-3s | %s\n" \
    "$LABEL" "${TPOTS[0]}" "${TPOTS[1]}" "${TPOTS[2]}" "$AVG" \
    "$COH_PRE" "$COH_POST" "$EXTRA_ENV" >> "$SUMMARY"

  echo "teardown…"
  docker rm -f decode_opt 2>/dev/null || true
  kill $TAILER_PID 2>/dev/null || true
  # Brief idle to let GPUs settle
  sleep 5
}

# baseline first (round-3 defaults only, no extra NCCL vars)
run_variant baseline ""
run_variant BUFFSIZE_8M  "--env NCCL_BUFFSIZE=8388608"
run_variant BUFFSIZE_16M "--env NCCL_BUFFSIZE=16777216"
run_variant P2P_NVL      "--env NCCL_P2P_LEVEL=NVL"
run_variant P2P_SYS      "--env NCCL_P2P_LEVEL=SYS"
run_variant MINCHANS_4   "--env NCCL_MIN_NCHANNELS=4"
run_variant NET_GDR_0    "--env NCCL_NET_GDR_LEVEL=0"

echo
echo "============================================================"
echo "SWEEP DONE — see $SUMMARY"
echo "============================================================"
cat "$SUMMARY"
