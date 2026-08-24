#!/usr/bin/env python3
"""Keep at most N networked CPU download workers for the evaluation scenes."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from download_eval_sens import scenes


def valid_output(path: Path) -> bool:
    try:
        counts = [len(list((path / k).iterdir())) for k in ("color", "depth", "depth_inpainted", "pose")]
        return counts[0] > 0 and len(set(counts)) == 1 and (path / "intrinsic").is_dir()
    except FileNotFoundError:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/shared-storage-user/yicheng-data/Qwen-3D"))
    p.add_argument("--max-scenes", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--max-use", type=float, default=0.80,
                   help="stop before shared filesystem usage reaches this fraction")
    args = p.parse_args()
    ids = scenes(args.root / "data/refer_it_3d")[: args.max_scenes]
    raw = args.root / "data/posed_rgbd_eval_dense/raw_tmp"
    output = args.root / "data/posed_rgbd_eval_dense/frames_square_highres"
    logdir = args.root / "data/posed_rgbd_eval_dense/download_workers"
    logdir.mkdir(parents=True, exist_ok=True)
    active: dict[str, subprocess.Popen[bytes]] = {}
    failed: list[str] = []
    done: set[str] = set()
    while len(done) < len(ids):
        usage = shutil.disk_usage(args.root)
        if usage.used / usage.total >= args.max_use:
            raise RuntimeError(
                f"shared storage usage {usage.used / usage.total:.1%} >= {args.max_use:.1%}"
            )
        for scene in ids:
            if scene in done:
                continue
            # Reap finished download workers before checking the output.  A
            # worker's .sens is normally consumed and deleted by the GPU
            # process before the scheduler sees a valid output directory.  If
            # the validity check ran first, the scene would be marked done
            # while its stale Popen entry remained in ``active`` forever,
            # permanently consuming a worker slot.
            proc = active.get(scene)
            if proc is not None and proc.poll() is not None:
                del active[scene]
                if proc.returncode != 0:
                    failed.append(scene)
                    raise RuntimeError(f"download worker failed after retries: {scene}")
            if valid_output(output / scene):
                done.add(scene)
                continue
            final = raw / f"{scene}.sens"
            if final.is_file() and final.stat().st_size:
                # The GPU consumer owns completed .sens files.  Do not start
                # a second downloader while it is waiting to consume one.
                continue
        if failed:
            raise RuntimeError(failed)
        candidates = [s for s in ids if s not in done and s not in active and not (raw / f"{s}.sens").is_file()]
        # Bound the complete ready+active queue.  The GPU is a single
        # consumer and can be slower than network transfer; without this
        # guard, every finished download would immediately launch another
        # one and raw_tmp would grow far beyond the intended 2--4 scenes.
        ready = sum(1 for path in raw.glob("*.sens") if path.is_file() and path.stat().st_size)
        while candidates and len(active) < args.workers and ready + len(active) < args.workers:
            scene = candidates.pop(0)
            log = (logdir / f"{scene}.log").open("ab")
            active[scene] = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("download_eval_sens.py")),
                 "--root", str(args.root), "--scene", scene, "--attempts", str(args.attempts)],
                stdout=log, stderr=subprocess.STDOUT,
            )
            print(f"START {scene} active={len(active)} done={len(done)}/{len(ids)}", flush=True)
        if len(done) < len(ids):
            time.sleep(15)
    print(f"COMPLETE {len(done)} scenes", flush=True)


if __name__ == "__main__":
    main()
