#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/shared-storage-user/yicheng-data/Qwen-3D"
SPLIT_FILE="${ROOT_DIR}/splits/scannet_splits/scannetv2_trainval.txt"
SCAN_ROOT="${ROOT_DIR}/data/raw/scannet/scans"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/scannet_download_monitor.log"
STATE_FILE="${LOG_DIR}/.scannet_last_complete"
LOCK_FILE="${LOG_DIR}/.scannet_monitor.lock"
INTERVAL_SECONDS="${1:-600}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[WARN] monitor already running. lock: ${LOCK_FILE}"
  exit 1
fi

required_suffixes=(
  ".aggregation.json"
  ".txt"
  "_vh_clean_2.0.010000.segs.json"
  "_vh_clean_2.ply"
  "_vh_clean_2.labels.ply"
)

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"

  total=0
  complete=0
  while IFS= read -r scan_id || [ -n "${scan_id}" ]; do
    [[ -z "${scan_id}" ]] && continue
    total=$((total + 1))

    ok=1
    for suf in "${required_suffixes[@]}"; do
      if [[ ! -f "${SCAN_ROOT}/${scan_id}/${scan_id}${suf}" ]]; then
        ok=0
        break
      fi
    done
    [[ ${ok} -eq 1 ]] && complete=$((complete + 1))
  done < "${SPLIT_FILE}"

  missing=$((total - complete))
  if (( total > 0 )); then
    percent="$(awk -v c="${complete}" -v t="${total}" 'BEGIN{printf "%.2f", (c/t)*100}')"
  else
    percent="0.00"
  fi

  last_complete=0
  if [[ -f "${STATE_FILE}" ]]; then
    last_complete="$(cat "${STATE_FILE}")"
  fi
  delta=$((complete - last_complete))
  echo "${complete}" > "${STATE_FILE}"

  if [[ "${delta}" -eq 0 && -f "${STATE_FILE}" ]]; then
    note="stalled_since_last_check"
    run_state="NO_PROGRESS"
  elif [[ "${delta}" -gt 0 ]]; then
    note="ok"
    run_state="DOWNLOADING"
  else
    note="ok"
    run_state="INITIALIZED"
  fi

  echo "[${ts}] state=${run_state} complete=${complete} total=${total} missing=${missing} pct=${percent}% delta=${delta} note=${note}" >> "${LOG_FILE}"

  sleep "${INTERVAL_SECONDS}"
done
