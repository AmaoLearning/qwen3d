#!/usr/bin/env python3
"""Evaluate a Qwen-3D or RFT-LoRA checkpoint on 3D QA splits."""

import argparse
import os
import subprocess
from pathlib import Path

from train_rft_lora import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-size", choices=("3b", "7b"), default="3b")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="all")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sampling-frame-num", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    backbone = root / f"models/backbones/Qwen2.5-VL-{args.model_size.upper()}-Instruct"
    output_root = (args.output_root or root / "output/rft_eval").resolve()
    _, val_datasets, test_datasets, _ = DATASETS[args.dataset]
    eval_datasets = val_datasets if args.eval_split == "val" else test_datasets

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not (backbone / "config.json").is_file():
        raise FileNotFoundError(f"Missing Qwen backbone: {backbone}")
    if args.num_gpus < 1 or args.num_workers < 0:
        raise ValueError("num-gpus must be positive and num-workers non-negative")
    if args.sampling_frame_num < 1 or args.sampling_frame_num % 2 == 0:
        raise ValueError("sampling-frame-num must be a positive odd number")

    environment = os.environ.copy()
    environment.update(
        {
            "QWEN3D_ROOT": str(root),
            "QWEN_MODEL": str(backbone),
            "OUTPUT_DIR_PREFIX": str(output_root),
            "NAME": args.name,
            "NUM_GPUS": str(args.num_gpus),
            "NUM_MACHINES": "1",
            "USE_SLURM": "0",
            "EVAL_ONLY": "1",
            "RESUME": "0",
            "BS": "1",
            "NUM_DATALOADERS": "0",
            "NUM_VAL_DATALOADERS": str(args.num_workers),
            "SAMPLING_FRAME_NUM": str(args.sampling_frame_num),
            "WANDB_MODE": "disabled",
        }
    )
    environment.pop("QWEN3D_SMOKE_SCENES", None)

    command = [
        "bash",
        "scripts/main_qwen.sh",
        "USE_WANDB",
        "False",
        "GENERATION",
        "True",
        "MODEL.DECODER_3D",
        "True",
        "USE_LORA",
        "True",
        "LORA_RANK",
        str(args.lora_rank),
        "LORA_ALPHA",
        str(args.lora_alpha),
        "LORA_DROPOUT",
        str(args.lora_dropout),
        "MODEL.WEIGHTS",
        str(checkpoint),
        "QWEN_MODEL",
        str(backbone),
        "DATASETS.TEST",
        repr(eval_datasets),
        "QA_GROUND_LOSS",
        "False",
        "USE_AUTO_NOUN_DETECTION",
        "False",
        "INPUT.INPAINT_DEPTH",
        "False",
        "SAMPLING_STRATEGY_REF",
        "False",
        "RFT_LOSS.ENABLED",
        "False",
    ]
    print(f"checkpoint={checkpoint}")
    print(f"eval_split={args.eval_split} datasets={eval_datasets}")
    print("command=" + " ".join(command))
    subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
