#!/usr/bin/env bash
# Reproduce Qwen-3D experiments (eval-only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-3b,7b}"
TASKS="${TASKS:-sr3d,scanrefer_nr3d,scanqa}"
LOG_DIR="${LOG_DIR:-$ROOT/output/repro_logs}"
OUTPUT_DIR_PREFIX="${OUTPUT_DIR_PREFIX:-$ROOT/output/repro}"
USE_WANDB="${USE_WANDB:-False}"

mkdir -p "$LOG_DIR"

source "$ROOT/repro/activate.sh"

ckpt_path_for() {
  local size="$1"
  case "$size" in
    3b) echo "$ROOT/models/qwen3d/qwen3d_3b.pth" ;;
    7b) echo "$ROOT/models/qwen3d/qwen3d_7b.pth" ;;
    *) echo "unknown size: $size" >&2; return 1 ;;
  esac
}

qwen_backbone_for() {
  local size="$1"
  case "$size" in
    3b) echo "$ROOT/models/backbones/Qwen2.5-VL-3B-Instruct" ;;
    7b) echo "$ROOT/models/backbones/Qwen2.5-VL-7B-Instruct" ;;
    *) echo "unknown size: $size" >&2; return 1 ;;
  esac
}

run_scanqa_or_sqa3d() {
  local size="$1"
  local task="$2"
  local ckpt qwen_model

  ckpt="$(ckpt_path_for "$size")"
  qwen_model="$(qwen_backbone_for "$size")"

  local dataset_train dataset_test
  case "$task" in
    scanqa)
      dataset_train="('scanqa_ref_scannet_train_single',)"
      dataset_test="('scanqa_ref_scannet_val_single_batched',)"
      ;;
    sqa3d)
      dataset_train="('sqa3d_ref_scannet_train_single',)"
      dataset_test="('sqa3d_ref_scannet_val_single_batched',)"
      ;;
    *)
      echo "unsupported task for direct launcher: $task" >&2
      return 1
      ;;
  esac

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local run_name="${size}_${task}_${stamp}"

  source scripts/setup.sh
  configure_local

  export OUTPUT_DIR_PREFIX="$OUTPUT_DIR_PREFIX"
  export NAME="$run_name"

  {
    echo "=== [$run_name] START $(date) ==="
    BS=1 EVAL_ONLY=1 RESUME=0 NUM_VAL_DATALOADERS="${NUM_VAL_DATALOADERS:-2}" \
      NUM_DATALOADERS="${NUM_DATALOADERS:-2}" "$PREFIX" "${PREFIX_ARGS[@]}" \
      scripts/main_qwen.sh \
      USE_WANDB "$USE_WANDB" \
      GENERATION False USE_AUTO_NOUN_DETECTION False INPUT.INPAINT_DEPTH False \
      MODEL.WEIGHTS "$ckpt" QWEN_MODEL "$qwen_model" \
      DATASETS.TRAIN "$dataset_train" DATASETS.TEST "$dataset_test" \
      "${@:3}" 2>&1 | tee "${LOG_DIR}/${run_name}.log"
    echo "=== [$run_name] DONE $(date) ==="
  } 
}

main() {
  IFS=',' read -r -a model_list <<< "$MODELS"
  IFS=',' read -r -a task_list <<< "$TASKS"

  for size in "${model_list[@]}"; do
    echo "Running preflight for $size"
    python repro/verify_setup.py --runtime --size "$size"

    for task in "${task_list[@]}"; do
      case "$task" in
        sr3d|scanrefer_nr3d)
          bash repro/eval.sh "$size" "$task" 2>&1 | tee "${LOG_DIR}/${size}_${task}.log"
          ;;
        scanqa|sqa3d)
          run_scanqa_or_sqa3d "$size" "$task"
          ;;
        *)
          echo "unsupported task: $task" >&2
          return 1
          ;;
      esac
    done
  done
}

main "$@"
