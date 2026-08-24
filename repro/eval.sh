#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/repro/activate.sh"
cd "$ROOT"

SIZE="${1:-3b}"
TASK="${2:-sr3d}"
USE_WANDB="${USE_WANDB:-False}"
RUN_NAME="${RUN_NAME:-${SIZE}_${TASK}_$(date +%Y%m%d_%H%M%S)}"
case "$SIZE" in
  3b) CKPT="$ROOT/models/qwen3d/qwen3d_3b.pth"; BACKBONE="$ROOT/models/backbones/Qwen2.5-VL-3B-Instruct" ;;
  7b) CKPT="$ROOT/models/qwen3d/qwen3d_7b.pth"; BACKBONE="$ROOT/models/backbones/Qwen2.5-VL-7B-Instruct" ;;
  *) echo "size must be 3b or 7b" >&2; exit 2 ;;
esac

case "$TASK" in
  sr3d)
    TRAIN="('sr3d_ref_scannet_train_single',)"
    TEST="('sr3d_ref_scannet_val_single_batched',)"
    ;;
  scanrefer_nr3d)
    TRAIN="('scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single',)"
    TEST="('scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched',)"
    ;;
  *) echo "task must be sr3d or scanrefer_nr3d" >&2; exit 2 ;;
esac

python repro/verify_setup.py --runtime --size "$SIZE"
source scripts/setup.sh
configure_local
export NAME="$RUN_NAME"
BS=1 EVAL_ONLY=1 RESUME=0 NUM_VAL_DATALOADERS="${NUM_VAL_DATALOADERS:-2}" \
  NUM_DATALOADERS="${NUM_DATALOADERS:-2}" "$PREFIX" "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
  GENERATION False USE_AUTO_NOUN_DETECTION False INPUT.INPAINT_DEPTH False \
  USE_WANDB "$USE_WANDB" \
  MODEL.WEIGHTS "$CKPT" QWEN_MODEL "$BACKBONE" \
  DATASETS.TRAIN "$TRAIN" DATASETS.TEST "$TEST" "${@:3}"
