#!/usr/bin/env bash
set -euo pipefail

# Start one serialized full dense-evaluation queue after all of its baseline
# and smoke gates have succeeded.  This script is intentionally offline: all
# model/data paths are local to the shared Qwen-3D workspace.
ROOT="/mnt/shared-storage-user/yicheng-data/Qwen-3D"
GPU="${1:?GPU index required}"
RUN_ROOT="$ROOT/output/benchmark_runs"
mkdir -p "$RUN_ROOT"

exec 9>"$RUN_ROOT/dense_full_gpu${GPU}.lock"
if ! flock -n 9; then
  echo "dense full worker for GPU${GPU} is already running" >&2
  exit 0
fi

wait_success() {
  local file="$1"
  local limit=3
  [[ "$file" == *_dense_smoke_fix1.status.tsv ]] && limit=1
  while true; do
    if awk -F $'\t' '$5 == "SUCCESS" { ok=1 } END { exit !ok }' \
        "$RUN_ROOT/$file" 2>/dev/null; then
      return 0
    fi
    local attempts running
    attempts="$(awk -F $'\t' '$4 ~ /^[0-9]+$/ && $4 > max { max=$4 } END { print max + 0 }' \
      "$RUN_ROOT/$file" 2>/dev/null)"
    running="$(awk -F $'\t' '
      NR == 1 { next }
      { state[$4] = $5 }
      END { for (attempt in state) if (state[attempt] == "RUNNING") n++; print n + 0 }
    ' \
      "$RUN_ROOT/$file" 2>/dev/null)"
    if (( attempts >= limit && running == 0 )); then
      echo "gate failed after $attempts attempts: $file" >&2
      return 1
    fi
    sleep 30
  done
}

case "$GPU" in
  0)
    gates=(
      3b_sr3d.status.tsv 7b_sr3d.status.tsv
      3b_sr3d_dense_smoke_fix1.status.tsv 7b_sr3d_dense_smoke_fix1.status.tsv
    )
    tasks=(3b:sr3d 7b:sr3d)
    ;;
  1)
    gates=(
      3b_nr3d.status.tsv 7b_nr3d.status.tsv
      3b_nr3d_dense_smoke.status.tsv 7b_nr3d_dense_smoke_fix4.status.tsv
    )
    tasks=(3b:nr3d 7b:nr3d)
    ;;
  2)
    gates=(
      3b_scanrefer.status.tsv 7b_scanrefer.status.tsv 7b_sqa3d.status.tsv
      3b_scanrefer_dense_smoke_fix1.status.tsv 7b_scanrefer_dense_smoke_fix1.status.tsv
      3b_sqa3d_dense_smoke_fix1.status.tsv 7b_sqa3d_dense_smoke_fix1.status.tsv
    )
    tasks=(3b:scanrefer 7b:scanrefer 3b:sqa3d 7b:sqa3d)
    ;;
  3)
    gates=(
      3b_scanqa.status.tsv 7b_scanqa.status.tsv 3b_sqa3d.status.tsv
      3b_scanqa_dense_smoke.status.tsv 7b_scanqa_dense_smoke_fix1.status.tsv
      3b_sqa3d_dense_smoke_fix1.status.tsv 7b_sqa3d_dense_smoke_fix1.status.tsv
    )
    tasks=(3b:scanqa 7b:scanqa)
    ;;
  *)
    echo "GPU must be one of 0, 1, 2, or 3" >&2
    exit 2
    ;;
esac

for gate in "${gates[@]}"; do
  echo "WAIT GPU${GPU} $gate $(date -Is)"
  wait_success "$gate"
done

echo "START GPU${GPU} dense full $(date -Is) tasks=${tasks[*]}"
RUN_TAG=_dense_full EVAL_SCRIPT=repro/eval_dense.sh MAX_ATTEMPTS=3 \
  bash "$ROOT/repro/benchmark_worker.sh" "$GPU" "${tasks[@]}"
rc=$?
echo "END GPU${GPU} dense full rc=$rc $(date -Is)"
exit "$rc"
