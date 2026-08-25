#!/usr/bin/env python3
"""Guarded entry point for Real-3DQA 3D-RFT.

The checked-in Real-3DQA repository contains only test/rotation annotations and
RFT pseudocode.  A real RFT run needs a training split plus a frozen blind
reference model.  This script makes that prerequisite explicit and refuses to
silently train on the test set or invent a blind reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded Real-3DQA 3D-RFT launcher")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--blind-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    problems = []
    if not args.train_jsonl.is_file():
        problems.append(f"training JSONL not found: {args.train_jsonl}")
    if args.train_jsonl.name in {"debiased_test.jsonl", "rotation_0.jsonl", "rotation_90.jsonl", "rotation_180.jsonl", "rotation_270.jsonl"}:
        problems.append("Real-3DQA test/rotation annotations must not be used as RFT training data")
    if not args.blind_checkpoint.is_file():
        problems.append(f"blind reference checkpoint not found: {args.blind_checkpoint}")
    if problems:
        print(json.dumps({
            "status": "stopped",
            "reason": "RFT prerequisites are unavailable; no training was started",
            "problems": problems,
        }, indent=2))
        return 2
    raise NotImplementedError(
        "The current Qwen-3D repository has no Real-3DQA train mapper or blind-model loss hook. "
        "Add those only after supplying an independently constructed train split and blind checkpoint."
    )


if __name__ == "__main__":
    raise SystemExit(main())
