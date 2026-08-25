#!/usr/bin/env bash
set -u

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
  while ! awk -F $'\t' '$5 == "SUCCESS" { ok=1 } END { exit !ok }' \
      "$RUN_ROOT/$file" 2>/dev/null; do
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
      3b_nr3d_dense_smoke.status.tsv 7b_nr3d_dense_smoke_fix1.status.tsv
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
