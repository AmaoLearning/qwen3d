#!/usr/bin/env python3
"""Consume completed shared .sens files while the CPU downloader runs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from extract_eval_dense import process_scene, scene_ids, validate_scene


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/shared-storage-user/yicheng-data/Qwen-3D"))
    p.add_argument("--max-scenes", type=int, default=20)
    p.add_argument("--poll-seconds", type=int, default=20)
    args = p.parse_args()
    ids = scene_ids(args.root / "data/refer_it_3d")[: args.max_scenes]
    raw = args.root / "data/posed_rgbd_eval_dense/raw_tmp"
    output = args.root / "data/posed_rgbd_eval_dense/frames_square_highres"
    done: set[str] = set()
    while len(done) < len(ids):
        for scene in ids:
            if scene in done:
                continue
            final = output / scene
            if final.is_dir() and validate_scene(final)[0]:
                sens = raw / f"{scene}.sens"
                if sens.is_file():
                    sens.unlink()
                done.add(scene)
                print(f"SKIP already valid {scene} ({len(done)}/{len(ids)})", flush=True)
                continue
            sens = raw / f"{scene}.sens"
            if not sens.is_file() or sens.stat().st_size == 0:
                continue
            process_scene(scene, raw, output, frame_skip=20, attempts=3)
            done.add(scene)
            print(f"CONSUMED {scene} ({len(done)}/{len(ids)})", flush=True)
        if len(done) < len(ids):
            time.sleep(args.poll_seconds)
    print(f"ALL READY SCENES COMPLETE: {len(done)}", flush=True)


if __name__ == "__main__":
    main()
