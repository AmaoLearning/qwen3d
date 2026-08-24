#!/usr/bin/env python3
"""Audit the dense ScanNet evaluation extraction on the shared filesystem."""
from __future__ import annotations

import argparse
from pathlib import Path

from extract_eval_dense import scene_ids, validate_scene


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/shared-storage-user/yicheng-data/Qwen-3D"))
    p.add_argument("--max-scenes", type=int, default=0)
    args = p.parse_args()
    root = args.root.resolve()
    ids = scene_ids(root / "data/refer_it_3d")
    if args.max_scenes:
        ids = ids[: args.max_scenes]
    out = root / "data/posed_rgbd_eval_dense/frames_square_highres"
    raw = root / "data/posed_rgbd_eval_dense/raw_tmp"
    errors: list[str] = []
    total_frames = 0
    for scene in ids:
        ok, detail = validate_scene(out / scene)
        if ok:
            total_frames += int(detail.split("=", 1)[1])
        else:
            errors.append(f"{scene}: {detail}")
    extras = sorted(
        p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith(".")
        and p.name not in ids
    ) if out.is_dir() else []
    leftovers = sorted(p.name for p in raw.glob("*.sens")) + sorted(p.name for p in raw.glob("*.sens.part"))
    work = sorted(p.name for p in out.glob(".*.work")) if out.is_dir() else []
    if extras:
        errors.append(f"unexpected output scenes: {extras}")
    if leftovers:
        errors.append(f"raw leftovers: {leftovers}")
    if work:
        errors.append(f"work directories: {work}")
    print(f"scenes={len(ids)} valid={len(ids)-len(errors)} frames={total_frames}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print("AUDIT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
