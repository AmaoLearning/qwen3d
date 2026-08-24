#!/usr/bin/env python3
"""Single-consumer ScanNet downloader using the alternate TUM endpoint.

This recovery helper deliberately performs no retries.  Each 8 MiB Range
request must complete before it is appended to the scene's .part file; any
network error aborts the process and leaves the resumable .part intact.
"""
from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/shared-storage-user/yicheng-data/Qwen-3D")
RAW = ROOT / "data/posed_rgbd_eval_dense/raw_tmp"
OUT = ROOT / "data/posed_rgbd_eval_dense/frames_square_highres"
BASE = "https://kaldir.vc.in.tum.de/scannet/v1/scans"
CHUNK = 8 * 1024 * 1024


def scene_ids() -> list[str]:
    ids: set[str] = set()
    for name in ("sr3d_val.csv", "ScanEnts3D_Nr3D_val.csv", "ScanRefer_filtered_val_ScanEnts3D_val.csv"):
        with (ROOT / "data/refer_it_3d" / name).open(newline="") as handle:
            rows = csv.DictReader(handle)
            column = "scan_id" if "scan_id" in (rows.fieldnames or []) else "scene_id"
            ids.update(str(row[column]) for row in rows)
    return sorted(ids)


def valid(scene: str) -> bool:
    path = OUT / scene
    try:
        counts = [len(list((path / key).iterdir())) for key in ("color", "depth", "depth_inpainted", "pose")]
        return (counts[0] > 0 and len(set(counts)) == 1 and
                (path / "intrinsic/intrinsic_color.txt").is_file() and
                (path / "intrinsic/intrinsic_depth.txt").is_file())
    except FileNotFoundError:
        return False


def size(url: str) -> int:
    text = subprocess.check_output(
        ["curl", "-4", "-sSIL", "--connect-timeout", "20", "--max-time", "60", url],
        text=True,
    )
    for line in reversed(text.splitlines()):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"missing Content-Length: {url}")


def download(scene: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    part = RAW / f"{scene}.sens.part"
    final = RAW / f"{scene}.sens"
    url = f"{BASE}/{scene}/{scene}.sens"
    total = size(url)
    offset = part.stat().st_size if part.exists() else 0
    print(f"ALT_START {scene} {offset}/{total}", flush=True)
    while offset < total:
        end = min(offset + CHUNK, total) - 1
        tmp = part.with_name(part.name + ".chunk")
        tmp.unlink(missing_ok=True)
        subprocess.run(
            ["curl", "-4", "-sS", "-L", "--fail", "--connect-timeout", "20",
             "--max-time", "300", "--range", f"{offset}-{end}", url, "-o", str(tmp)],
            check=True,
        )
        expected = end - offset + 1
        got = tmp.stat().st_size
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"short range: scene={scene} offset={offset} got={got} expected={expected}")
        with part.open("ab") as dst, tmp.open("rb") as src:
            dst.write(src.read())
        tmp.unlink()
        offset = end + 1
        print(f"ALT_PROGRESS {scene} {offset}/{total}", flush=True)
    if part.stat().st_size != total:
        raise RuntimeError(f"size mismatch: {scene}")
    os.replace(part, final)
    print(f"ALT_READY {scene} bytes={total}", flush=True)


def main() -> None:
    for scene in scene_ids():
        if valid(scene):
            continue
        final = RAW / f"{scene}.sens"
        while final.exists() and not valid(scene):
            print(f"WAIT_CONSUME {scene}", flush=True)
            time.sleep(15)
        if valid(scene):
            continue
        download(scene)
        while not valid(scene):
            if (RAW / f"{scene}.sens").exists():
                print(f"WAIT_CONSUME {scene}", flush=True)
            time.sleep(15)
    print("ALT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
