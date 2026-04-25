#!/usr/bin/env bash
# Stage 7a runner: cycle through baseline / hipblock / amdserial modes
# Each takes ~10-15min so total ~45min wall.
set -u

OUT=/tmp/decode_opt/stage7a_results.md
echo "# Stage 7a — CAR fix attempts (eager-only isolation)" > "$OUT"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
echo >> "$OUT"

for mode in baseline hipblock amdserial; do
  echo "" >> "$OUT"
  echo "## MODE: $mode" >> "$OUT"
  echo "==== running $mode ===="
  bash /home/tyler/vllm-gfx908/docs/mi100_decode_opt/scripts/test_stage7a_car_eager.sh \
    vllm-rocm-gfx908:latest "$mode" 2>&1 | tee /tmp/decode_opt/stage7a_${mode}.log
  echo "" >> "$OUT"
  echo "### key outcomes" >> "$OUT"
  grep -E "READY|COHERENCE PRE|TPOT|COHERENCE POST|FAIL" /tmp/decode_opt/stage7a_${mode}.log >> "$OUT"
done

echo "all 3 modes done. summary in $OUT"
