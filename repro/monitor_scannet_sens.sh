#!/usr/bin/env bash
set -u

ROOT="/mnt/shared-storage-user/yicheng-data/Qwen-3D"
PID_FILE="$ROOT/output/scannet_sens_download.pid"
LOG="$ROOT/output/scannet_sens_progress.log"
SCANS="$ROOT/data/raw/scannet/scans"

mkdir -p "$(dirname "$LOG")"
while true; do
    pid=""
    [ -f "$PID_FILE" ] && pid="$(cat "$PID_FILE")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        state=RUNNING
    else
        state=STOPPED
    fi
    sens_files="$(find "$SCANS" -type f -name '*.sens' 2>/dev/null | wc -l)"
    partial_files="$(find "$SCANS" -type f -name '*.sens.part' 2>/dev/null | wc -l)"
    total_bytes="$(du -sb "$SCANS" 2>/dev/null | awk '{print $1}')"
    partial_bytes="$(find "$SCANS" -type f -name '*.sens.part' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    printf '%s state=%s pid=%s sens_files=%s partial_files=%s total_bytes=%s partial_bytes=%s\n' \
        "$(date '+%F %T%z')" "$state" "$pid" "$sens_files" "$partial_files" "$total_bytes" "$partial_bytes" >> "$LOG"
    [ "$state" = STOPPED ] && exit 0
    sleep 600
done
