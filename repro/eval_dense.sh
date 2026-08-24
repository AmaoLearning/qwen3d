#!/usr/bin/env bash
set -euo pipefail

# Keep the original sparse ScanNet package intact and point Detectron2 at the
# dense evaluation tree produced by the .sens extraction pipeline.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$ROOT/data/posed_rgbd_eval_dense}"
exec "$ROOT/repro/eval.sh" "$@"
