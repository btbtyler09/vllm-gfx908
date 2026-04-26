#!/usr/bin/env bash
# Driver for Phase 2 E isolated tests. Boots a vllm-rocm-gfx908 container
# (4 GPUs, network=host), mounts THIS directory + any overlay vllm files,
# and runs the three tests via 4-rank torchrun.
#
# Usage:
#   ./run_isolated_tests.sh [--image vllm-rocm-gfx908:latest] [--mode registered_false]
#   ./run_isolated_tests.sh --reproduce-bug   # alias for --mode=registered_true
#
# Exit codes:
#   0 — all three tests PASS
#   non-zero — see per-test stderr in /tmp/decode_opt/e_isolated_*.log
set -u

IMG="${IMG:-vllm-rocm-gfx908:latest}"
MODE="registered_false"
REPRODUCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)         IMG="$2"; shift 2 ;;
    --mode)          MODE="$2"; shift 2 ;;
    --reproduce-bug) MODE="registered_true"; REPRODUCE=1; shift ;;
    *) echo "unknown flag: $1"; exit 64 ;;
  esac
done

mkdir -p /tmp/decode_opt
LOG_DIR=/tmp/decode_opt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR=/home/tyler/vllm-gfx908

echo "============================================================"
echo "Phase 2 E isolated tests"
echo "  image:     $IMG"
echo "  mode:      $MODE  (reproduce-bug=$REPRODUCE)"
echo "  test dir:  $SCRIPT_DIR"
echo "  log dir:   $LOG_DIR"
echo "============================================================"

docker rm -f decode_opt_e_test 2>/dev/null || true

run_in_container() {
  local TEST="$1" ; shift
  local OUT="$LOG_DIR/e_isolated_${TEST}.log"
  : > "$OUT"

  echo
  echo "--- $TEST ---"
  docker run --rm --name decode_opt_e_test \
    --network=host --cpuset-cpus="0-11" --group-add=video --ipc=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 \
    --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 \
    --env HSA_OVERRIDE_GFX_VERSION=9.0.8 \
    --env HF_HOME=/huggingface \
    --env MASTER_ADDR=127.0.0.1 --env MASTER_PORT=29500 \
    -v /home/tyler/.cache/huggingface:/huggingface \
    -v "$SCRIPT_DIR":/test:ro \
    -v "$VLLM_DIR/vllm/platforms/rocm.py":/usr/local/lib/python3.12/dist-packages/vllm/platforms/rocm.py:ro \
    -v "$VLLM_DIR/vllm/distributed/device_communicators/custom_all_reduce.py":/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py:ro \
    "$IMG" \
    bash -c "cd /test && torchrun --nproc_per_node=4 ${TEST} $*" \
    2>&1 | tee "$OUT"
  local RC=${PIPESTATUS[0]}
  echo "--- $TEST exit=$RC ---"
  return $RC
}

OVERALL=0

run_in_container test_e1_eager.py
RC1=$?
[[ $RC1 -ne 0 ]] && OVERALL=1

run_in_container test_e2_graph_capture.py --mode="$MODE" --replays=100
RC2=$?
[[ $RC2 -ne 0 ]] && OVERALL=1

run_in_container test_e3_graph_replay_soak.py --mode="$MODE" --replays=1000
RC3=$?
[[ $RC3 -ne 0 ]] && OVERALL=1

echo
echo "============================================================"
echo "Phase 2 E isolated tests SUMMARY (mode=$MODE)"
echo "  test_e1_eager:        $([[ $RC1 -eq 0 ]] && echo PASS || echo FAIL)"
echo "  test_e2_graph:        $([[ $RC2 -eq 0 ]] && echo PASS || echo FAIL)"
echo "  test_e3_soak:         $([[ $RC3 -eq 0 ]] && echo PASS || echo FAIL)"
echo "  overall:              $([[ $OVERALL -eq 0 ]] && echo PASS || echo FAIL)"
echo "============================================================"

exit $OVERALL
