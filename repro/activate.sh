#!/usr/bin/env bash

_QWEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN3D_ROOT="$_QWEN3D_ROOT"
export VIRTUAL_ENV="$QWEN3D_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
if [[ -d "$QWEN3D_ROOT/.cuda/targets/x86_64-linux" ]]; then
  export CUDA_HOME="$QWEN3D_ROOT/.cuda/targets/x86_64-linux"
  export PATH="$CUDA_HOME/bin:$QWEN3D_ROOT/.cuda/bin:$QWEN3D_ROOT/.cuda/nvvm/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "/usr/local/nvidia/lib64" ]]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "/usr/local/nvidia/bin" ]]; then
  export PATH="/usr/local/nvidia/bin:$PATH"
fi
export PYTHONPATH="$QWEN3D_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$QWEN3D_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
# Reproduction runs are self-contained; do not silently fall back to the Hub
# when a local model file is missing or a path is accidentally hard-coded.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NLTK_DATA="${NLTK_DATA:-$QWEN3D_ROOT/data/nltk}"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$QWEN3D_ROOT/data/posed_rgbd}"
export DETECTRON2_DATASETS_2D="${DETECTRON2_DATASETS_2D:-$QWEN3D_ROOT/data/datasets_2d}"
export REF_DATASET="${REF_DATASET:-$QWEN3D_ROOT/data/refer_it_3d}"
export PRECOMPUTED_SCANNET_PATH="${PRECOMPUTED_SCANNET_PATH:-$QWEN3D_ROOT/data/scannet_precomputed}"
export COCO_REF_DATASET="${COCO_REF_DATASET:-$DETECTRON2_DATASETS_2D}"
export CLIP_SAMPLING_PATH="${CLIP_SAMPLING_PATH:-$REF_DATASET}"
export LLAVA_INSTRUCT_PATH="${LLAVA_INSTRUCT_PATH:-$DETECTRON2_DATASETS_2D/llava_instruct_150k.json}"
export SCANNET_DATA_DIR="${SCANNET_DATA_DIR:-$QWEN3D_ROOT/data/mask3d_processed/scannet/train_validation_database.yaml}"
export SCANNET200_DATA_DIR="${SCANNET200_DATA_DIR:-$QWEN3D_ROOT/data/mask3d_processed/scannet200/train_validation_database.yaml}"
export MATTERPORT_DATA_DIR="${MATTERPORT_DATA_DIR:-$QWEN3D_ROOT/data/mask3d_processed/matterport/train_validation_database.yaml}"
export SCANNETPP_DATA_DIR="${SCANNETPP_DATA_DIR:-$QWEN3D_ROOT/data/mask3d_processed/scannetpp/validation_database.yaml}"
export S3DIS_DATA_DIR="${S3DIS_DATA_DIR:-$QWEN3D_ROOT/data/mask3d_processed/s3dis/train_validation_database.yaml}"
export FEATURE_DIR="${FEATURE_DIR:-$QWEN3D_ROOT/data/scannet_image_qwen_features}"
export OUTPUT_DIR_PREFIX="${OUTPUT_DIR_PREFIX:-$QWEN3D_ROOT/output}"
unset _QWEN3D_ROOT
