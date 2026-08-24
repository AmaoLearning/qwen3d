#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def retry(label, fn, attempts: int = 3, wait_seconds: int = 120):
    for attempt in range(1, attempts + 1):
        try:
            print(f"[{label}] attempt {attempt}/{attempts}", flush=True)
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"[{label}] failed ({exc!r}); waiting {wait_seconds}s before retry", flush=True)
            time.sleep(wait_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-only", action="store_true")
    parser.add_argument("--backbones-only", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--attempts", type=int, default=3,
                        help="maximum attempts for each requested download (default: 3)")
    parser.add_argument("--checkpoint", choices=("3b", "7b"),
                        help="download only one Qwen-3D checkpoint; implies --checkpoints-only")
    args = parser.parse_args()
    if args.checkpoints_only and args.backbones_only:
        parser.error("--checkpoints-only and --backbones-only are mutually exclusive")
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.checkpoint:
        if args.backbones_only:
            parser.error("--checkpoint cannot be combined with --backbones-only")
        args.checkpoints_only = True
    checkpoint_dir = ROOT / "models/qwen3d"
    backbone_dir = ROOT / "models/backbones"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir.mkdir(parents=True, exist_ok=True)

    if not args.backbones_only:
        checkpoint_files = ("qwen3d_3b.pth", "qwen3d_7b.pth")
        if args.checkpoint:
            checkpoint_files = (f"qwen3d_{args.checkpoint}.pth",)
        for filename in checkpoint_files:
            retry(filename, lambda f=filename: hf_hub_download(
                "katefgroup/Qwen-3D", f, local_dir=checkpoint_dir
            ), attempts=args.attempts, wait_seconds=args.wait_seconds)

    if not args.checkpoints_only:
        for size in ("3B", "7B"):
            repo = f"Qwen/Qwen2.5-VL-{size}-Instruct"
            target = backbone_dir / f"Qwen2.5-VL-{size}-Instruct"
            retry(repo, lambda r=repo, t=target: snapshot_download(
                r, local_dir=t, max_workers=4
            ), attempts=args.attempts, wait_seconds=args.wait_seconds)


if __name__ == "__main__":
    main()
