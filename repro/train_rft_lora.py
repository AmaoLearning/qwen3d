#!/usr/bin/env python3
"""Launch 3D-only Qwen-3D LoRA post-training with Real-3DQA losses.

Run after ``source repro/activate.sh``.  The default is deliberately a single
smoke iteration with no evaluation or checkpoint write.  Pass ``--full-run``
explicitly before requesting a longer training job.
"""

import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


LOSS_TYPES = (
    "original",
    "paper_ratio",
    "neg_log_phi",
    "focal",
    "one_minus_log1p",
)

DATASETS = {
    "sqa3d": (
        ("sqa3d_ref_scannet_train_single",),
        ("sqa3d_ref_scannet_val_single_batched",),
        ("SQA_train.json",),
    ),
    "scanqa": (
        ("scanqa_ref_scannet_train_single",),
        ("scanqa_ref_scannet_val_single_batched",),
        ("ScanQA_v1.0_train.json",),
    ),
    "all": (
        (
            "sqa3d_ref_scannet_train_single",
            "scanqa_ref_scannet_train_single",
        ),
        (
            "sqa3d_ref_scannet_val_single_batched",
            "scanqa_ref_scannet_val_single_batched",
        ),
        ("SQA_train.json", "ScanQA_v1.0_train.json"),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("3b", "7b"), default="3b")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="sqa3d")
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="focal")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help=(
            "Micro-batches accumulated per optimizer step. With 8 GPUs and "
            "the fixed per-rank micro-batch of one, 2 gives effective batch 16."
        ),
    )
    parser.add_argument("--original-loss-coef", type=float, default=1.0)
    parser.add_argument("--rft-loss-coef", type=float, default=1.0)
    parser.add_argument("--generation-weight", type=float, default=5.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Training dataloader workers per GPU process (8 GPUs => 16 by default).",
    )
    parser.add_argument(
        "--num-val-workers",
        type=int,
        default=2,
        help="Validation dataloader workers per GPU process.",
    )
    parser.add_argument("--smoke-scenes", type=int, default=1)
    parser.add_argument("--sampling-frame-num", type=int, default=3)
    parser.add_argument(
        "--use-relevant-frame-map",
        action="store_true",
        help=(
            "Use Qwen-3D's precomputed object-relevant frame map. The prepared "
            "yicloud data does not include this optional cache, so consecutive "
            "3D frame sampling is the default."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    return parser.parse_args()


def require_path(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1")
    if args.gamma <= 0:
        raise ValueError("--gamma must be positive")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    if args.num_workers < 0 or args.num_val_workers < 0:
        raise ValueError("dataloader worker counts must be non-negative")
    if args.smoke_scenes < 1:
        raise ValueError("--smoke-scenes must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if min(
        args.original_loss_coef,
        args.rft_loss_coef,
        args.generation_weight,
    ) < 0:
        raise ValueError("loss coefficients must be non-negative")
    if args.original_loss_coef == 0 and (
        args.loss_type == "original" or args.rft_loss_coef == 0
    ):
        raise ValueError("at least one active loss coefficient must be positive")
    if args.sampling_frame_num < 1 or args.sampling_frame_num % 2 == 0:
        raise ValueError("--sampling-frame-num must be a positive odd number")
    if not args.full_run and args.max_iter > 10:
        raise ValueError(
            "Refusing a non-smoke job longer than 10 iterations without --full-run"
        )

    backbone = args.qwen_model or (
        root / f"models/backbones/Qwen2.5-VL-{args.model_size.upper()}-Instruct"
    )
    checkpoint = args.model_weights or (
        root / f"models/qwen3d/qwen3d_{args.model_size}.pth"
    )
    feature_dir = args.feature_dir or (
        root / "data/scannet_image_qwen_features" / args.model_size
    )
    output_root = args.output_root or (root / "output/rft_lora")
    train_datasets, test_datasets, annotation_files = DATASETS[args.dataset]

    require_path(backbone / "config.json", "local Qwen-VL backbone")
    require_path(checkpoint, "Qwen-3D checkpoint")
    # FEATURE_DIR is only consumed when CACHE_QWEN_FEATURES is enabled.  The
    # prepared yicloud runs encode RGB frames directly, so an absent optional
    # cache directory must not block post-training.
    for annotation_file in annotation_files:
        require_path(
            root / "data/refer_it_3d" / annotation_file,
            "3D QA training data",
        )
    require_path(
        root / "data/mask3d_processed/scannet200/train_validation_database.yaml",
        "ScanNet 3D database",
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gamma_tag = str(args.gamma).replace(".", "p")
    run_name = (
        f"{args.model_size}_{args.dataset}_{args.loss_type}_g{gamma_tag}_"
        f"{stamp}_{'full' if args.full_run else 'smoke'}"
    )

    environment = os.environ.copy()
    smoke = not args.full_run
    environment.update(
        {
            "QWEN3D_ROOT": str(root),
            "QWEN_MODEL": str(backbone),
            "FEATURE_DIR": str(feature_dir),
            "OUTPUT_DIR_PREFIX": str(output_root),
            "NAME": run_name,
            "NUM_GPUS": str(args.num_gpus),
            "NUM_MACHINES": "1",
            "USE_SLURM": "0",
            "EVAL_ONLY": "0",
            "RESUME": "0",
            "BS": "1",
            "NUM_DATALOADERS": str(args.num_workers),
            "NUM_VAL_DATALOADERS": str(args.num_val_workers),
            "SAMPLING_FRAME_NUM": str(args.sampling_frame_num),
            "CHECKPOINT_PERIOD": str(args.max_iter + 1),
            "EVAL_PERIOD": str(args.max_iter + 1),
            "WANDB_MODE": "disabled",
        }
    )
    if smoke:
        environment["QWEN3D_SMOKE_SCENES"] = str(args.smoke_scenes)
    else:
        # Do not inherit a scene cap exported by a prior smoke run.
        environment.pop("QWEN3D_SMOKE_SCENES", None)

    if args.full_run and not args.dry_run and shutil.which(
        "java", path=environment.get("PATH")
    ) is None:
        raise RuntimeError(
            "Full-run evaluation requires Java for pycocoevalcap METEOR. "
            "Install default-jre-headless on the training node first."
        )

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
        "FEATURE_DIR",
        str(feature_dir),
        "DATASETS.TRAIN",
        repr(train_datasets),
        "DATASETS.TEST",
        repr(test_datasets),
        "QA_GROUND_LOSS",
        "False",
        "USE_AUTO_NOUN_DETECTION",
        "False",
        "INPUT.INPAINT_DEPTH",
        "False",
        "SAMPLING_STRATEGY_REF",
        str(args.use_relevant_frame_map),
        "FIND_UNUSED_PARAMETERS",
        str(args.num_gpus > 1),
        "SOLVER.MAX_ITER",
        str(args.max_iter),
        "SOLVER.BASE_LR",
        str(args.learning_rate),
        "SOLVER.WARMUP_ITERS",
        str(args.warmup_iters),
        "GRAD_ACCUMULATION_STEPS",
        str(args.gradient_accumulation_steps),
        "MODEL.MASK_FORMER.GENERATION_WEIGHT",
        str(args.generation_weight),
        "RFT_LOSS.ENABLED",
        "True",
        "RFT_LOSS.TYPE",
        args.loss_type,
        "RFT_LOSS.GAMMA",
        str(args.gamma),
        "RFT_LOSS.ORIGINAL_COEF",
        str(args.original_loss_coef),
        "RFT_LOSS.RFT_COEF",
        str(args.rft_loss_coef),
        "RFT_LOSS.SMOKE_TEST",
        str(smoke),
    ]

    print(f"root={root}")
    print(f"run_name={run_name}")
    print(
        f"loss_type={args.loss_type} gamma={args.gamma} "
        f"original_coef={args.original_loss_coef} "
        f"rft_coef={args.rft_loss_coef} "
        f"generation_weight={args.generation_weight} smoke={smoke} "
        f"scene_limit={args.smoke_scenes if smoke else 'all'} "
        f"find_unused_parameters={args.num_gpus > 1} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch={args.num_gpus * args.gradient_accumulation_steps} "
        f"workers_per_rank={args.num_workers} "
        f"val_workers_per_rank={args.num_val_workers}"
    )
    print("command=" + " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
