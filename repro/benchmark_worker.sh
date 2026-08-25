#!/usr/bin/env bash
set -u

ROOT="/mnt/shared-storage-user/yicheng-data/Qwen-3D"
GPU="${1:?GPU index required}"
shift
RUN_ROOT="$ROOT/output/benchmark_runs"
EVAL_SCRIPT="${EVAL_SCRIPT:-repro/eval.sh}"
RUN_TAG="${RUN_TAG:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
mkdir -p "$RUN_ROOT"

run_one() {
  local size="$1" task="$2" label
  local attempt log run rc start end metric_source
  label="${size}_${task}${RUN_TAG}"
  local status="$RUN_ROOT/${label}.status.tsv"
  if [[ ! -f "$status" ]]; then
    printf 'gpu\tsize\ttask\tattempt\tstate\tstart\tend\trc\tlog\n' > "$status"
  fi
  if awk -F $'\t' '$5 == "SUCCESS" { ok=1 } END { exit !ok }' "$status" 2>/dev/null; then
    return 0
  fi
  local last_attempt=0
  last_attempt="$(awk -F $'\t' '
    $4 ~ /^[0-9]+$/ && $4 > max { max=$4 }
    END { print max + 0 }
  ' "$status" 2>/dev/null)"
  local first_attempt=$((last_attempt + 1))
  if (( first_attempt > MAX_ATTEMPTS )); then
    return 1
  fi
  for ((attempt=first_attempt; attempt<=MAX_ATTEMPTS; attempt++)); do
    run="${label}_r${attempt}"
    log="$RUN_ROOT/${run}.log"
    start="$(date -Is)"
    printf '%s\t%s\t%s\t%s\tRUNNING\t%s\t-\t-\t%s\n' "$GPU" "$size" "$task" "$attempt" "$start" "$log" >> "$status"
    set +e
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" RUN_NAME="$run" NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=2 \
      bash "$EVAL_SCRIPT" "$size" "$task" > "$log" 2>&1
    rc=$?
    set -e
    end="$(date -Is)"
    metric_source="$log $ROOT/output/$run/log.txt"
    if [[ "$rc" -eq 0 ]] && grep -q 'Total inference time' $metric_source 2>/dev/null; then
      printf '%s\t%s\t%s\t%s\tSUCCESS\t%s\t%s\t%s\t%s\n' "$GPU" "$size" "$task" "$attempt" "$start" "$end" "$rc" "$log" >> "$status"
      return 0
    fi
    printf '%s\t%s\t%s\t%s\tFAILED\t%s\t%s\t%s\t%s\n' "$GPU" "$size" "$task" "$attempt" "$start" "$end" "$rc" "$log" >> "$status"
    if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then sleep 10; fi
  done
  return 1
}

worker_status="$RUN_ROOT/gpu${GPU}.worker.log"
{
  echo "worker gpu=$GPU pid=$$ start=$(date -Is)"
  failed=0
  while [[ "$#" -gt 0 ]]; do
    spec="$1"; shift
    size="${spec%%:*}"; task="${spec#*:}"
    echo "START $size $task $(date -Is)"
    if run_one "$size" "$task"; then
      echo "DONE $size $task $(date -Is)"
    else
      echo "STOP_AFTER_3_ATTEMPTS $size $task $(date -Is)"
      failed=1
    fi
  done
  echo "worker gpu=$GPU end=$(date -Is) failed=$failed"
  exit "$failed"
} >> "$worker_status" 2>&1
