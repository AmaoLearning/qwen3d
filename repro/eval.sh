#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/repro/activate.sh"
cd "$ROOT"

SIZE="${1:-3b}"
TASK="${2:-sr3d}"
USE_WANDB="${USE_WANDB:-False}"
RUN_NAME="${RUN_NAME:-${SIZE}_${TASK}_$(date +%Y%m%d_%H%M%S)}"
GENERATION_MODE=False
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
  scanrefer)
    TRAIN="('scanrefer_scannet_anchor_train_single',)"
    TEST="('scanrefer_scannet_anchor_val_single_batched',)"
    ;;
  nr3d)
    TRAIN="('nr3d_ref_scannet_anchor_train_single',)"
    TEST="('nr3d_ref_scannet_anchor_val_single_batched',)"
    ;;
  scanqa)
    TRAIN="('scanqa_ref_scannet_train_single',)"
    TEST="('scanqa_ref_scannet_val_single_batched',)"
    GENERATION_MODE=True
    ;;
  sqa3d)
    TRAIN="('sqa3d_ref_scannet_train_single',)"
    TEST="('sqa3d_ref_scannet_test_single_batched',)"
    GENERATION_MODE=True
    ;;
  scannet200)
    TRAIN="('scannet200_context_instance_train_200cls_single_highres_100k',)"
    TEST="('scannet200_context_instance_val_200cls_single_highres_100k',)"
    ;;
  *) echo "task must be sr3d, nr3d, scanrefer, scanqa, sqa3d, or scannet200" >&2; exit 2 ;;
esac

if [[ "${QWEN3D_SMOKE:-0}" == "1" ]]; then
  case "$TASK" in
    sr3d) TEST="('sr3d_ref_scannet_val_50_single_batched',)" ;;
    scanrefer) TEST="('scanrefer_scannet_anchor_val_50_single_batched',)" ;;
    nr3d) TEST="('nr3d_ref_scannet_anchor_val_50_single_batched',)" ;;
    scanrefer_nr3d) TEST="('scanrefer_scannet_anchor_val_50_single_batched','nr3d_ref_scannet_anchor_val_50_single_batched',)" ;;
    scanqa) TEST="('scanqa_ref_scannet_val_50_single_batched',)" ;;
    sqa3d) TEST="('sqa3d_ref_scannet_test_2_single_batched',)" ;;
    scannet200) TEST="('scannet200_context_instance_debug_200cls_single_highres_100k',)" ;;
  esac
fi

python repro/verify_setup.py --runtime --size "$SIZE"
source scripts/setup.sh
configure_local
export NAME="$RUN_NAME"
BS=1 EVAL_ONLY=1 RESUME=0 NUM_VAL_DATALOADERS="${NUM_VAL_DATALOADERS:-2}" \
  NUM_DATALOADERS="${NUM_DATALOADERS:-2}" "$PREFIX" "${PREFIX_ARGS[@]}" scripts/main_qwen.sh \
  GENERATION "$GENERATION_MODE" USE_AUTO_NOUN_DETECTION False INPUT.INPAINT_DEPTH False \
  USE_WANDB "$USE_WANDB" \
  MODEL.WEIGHTS "$CKPT" QWEN_MODEL "$BACKBONE" \
  DATASETS.TRAIN "$TRAIN" DATASETS.TEST "$TEST" "${@:3}"
