#!/usr/bin/env python3
"""Stream ScanNet .sens files into dense posed RGB-D evaluation data.

The input .sens file is kept only in ``raw_tmp`` while one scene is being
decoded.  A scene is committed only after all expected output directories have
matching frame stems; failed downloads stop the run after three attempts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://kaldir.vc.cit.tum.de/scannet/v1/scans"


def scene_ids(root: Path) -> list[str]:
    files = (
        root / "sr3d_val.csv",
        root / "ScanEnts3D_Nr3D_val.csv",
        root / "ScanRefer_filtered_val_ScanEnts3D_val.csv",
    )
    out: set[str] = set()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        data = pd.read_csv(path)
        column = "scan_id" if "scan_id" in data.columns else "scene_id"
        out.update(data[column].astype(str).tolist())
    # Include official QA validation and test scenes in addition to the grounding union.
    for filename in (
        "ScanQA_v1.0_val.json",
        "ScanQA_v1.0_test_w_obj.json",
        "SQA_val.json",
        "SQA_test.json",
    ):
        path = root / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        records = payload.get("questions", payload.get("data", [])) if isinstance(payload, dict) else payload
        for record in records:
            scene = record.get("scene_id", record.get("scan_id"))
            if scene is not None:
                out.add(str(scene))
    return sorted(out)


def download(url: str, destination: Path, attempts: int) -> None:
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context()
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            request = urllib.request.Request(url)
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            # Keep a failed endpoint from occupying a worker for minutes.  A
            # retry is deliberately counted at the scene level.
            with urllib.request.urlopen(request, timeout=300, context=context) as response:
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    total_size = int(content_range.rsplit("/", 1)[1])
                else:
                    total_size = int(response.headers.get("Content-Length", "0") or 0) + (offset if status == 206 else 0)
                mode = "ab" if offset and status == 206 else "wb"
                if offset and status != 206:
                    offset = 0
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if total_size and part.stat().st_size < total_size:
                raise IOError(f"incomplete download: {part.stat().st_size}/{total_size} bytes")
            os.replace(part, destination)
            return
        except Exception as exc:  # network/proxy errors are expected here
            last = exc
            print(f"download attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(min(10 * attempt, 30))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last


def stems(path: Path) -> set[str]:
    return {p.stem for p in path.iterdir() if p.is_file()}


def validate_scene(scene: Path) -> tuple[bool, str]:
    required = ("color", "depth", "depth_inpainted", "pose")
    counts = {name: len(stems(scene / name)) for name in required}
    if len(set(counts.values())) != 1 or not counts["color"]:
        return False, f"frame counts={counts}"
    intrinsic = scene / "intrinsic"
    if not (intrinsic / "intrinsic_color.txt").is_file():
        return False, "missing intrinsic_color.txt"
    if not (intrinsic / "intrinsic_depth.txt").is_file():
        return False, "missing intrinsic_depth.txt"
    common = stems(scene / "color") & stems(scene / "depth") & stems(scene / "depth_inpainted") & stems(scene / "pose")
    if len(common) != counts["color"]:
        return False, "frame stem mismatch"
    return True, f"frames={counts['color']}"


def process_scene(scene_id: str, raw_tmp: Path, output_root: Path, frame_skip: int, attempts: int) -> None:
    final = output_root / scene_id
    ok, detail = validate_scene(final) if final.is_dir() else (False, "not present")
    if ok:
        print(f"SKIP {scene_id}: {detail}", flush=True)
        return

    work = output_root / f".{scene_id}.work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    sens = raw_tmp / f"{scene_id}.sens"
    url = f"{BASE_URL}/{scene_id}/{scene_id}.sens"
    if sens.is_file() and sens.stat().st_size > 0:
        print(f"PRELOADED {scene_id}: {sens.stat().st_size} bytes", flush=True)
    else:
        print(f"DOWNLOAD {scene_id} {url}", flush=True)
    try:
        if not (sens.is_file() and sens.stat().st_size > 0):
            download(url, sens, attempts)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_preparation" / "scannet"))
        from SensorData import SensorData

        print(f"DECODE {scene_id} frame_skip={frame_skip}", flush=True)
        sensor = SensorData(str(sens))
        sensor.export_depth_images(str(work / "depth"), image_size=[480, 640], frame_skip=frame_skip)
        sensor.export_color_images(str(work / "color"), image_size=[480, 640], frame_skip=frame_skip)
        sensor.export_poses(str(work / "pose"), frame_skip=frame_skip)
        sensor.export_intrinsics(str(work / "intrinsic"))
        good, detail = validate_scene(work)
        if not good:
            raise RuntimeError(f"validation failed: {detail}")
        if final.exists():
            shutil.rmtree(final)
        os.replace(work, final)
        print(f"DONE {scene_id}: {detail}", flush=True)
    finally:
        if sens.exists():
            sens.unlink()
        if work.exists():
            shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--scene", action="append", default=None,
                        help="process only these scene IDs; may be repeated")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data/posed_rgbd_eval_dense/frames_square_highres"
    raw_tmp = root / "data/posed_rgbd_eval_dense/raw_tmp"
    output.mkdir(parents=True, exist_ok=True)
    raw_tmp.mkdir(parents=True, exist_ok=True)
    ids = args.scene if args.scene else scene_ids(root / "data/refer_it_3d")
    if args.max_scenes:
        ids = ids[: args.max_scenes]
    state = args.state or (root / "data/posed_rgbd_eval_dense/scene_state.jsonl")
    print(json.dumps({"scenes": len(ids), "frame_skip": args.frame_skip, "output": str(output)}), flush=True)
    with state.open("a") as journal:
        for index, scene_id in enumerate(ids, 1):
            print(f"[{index}/{len(ids)}] {scene_id}", flush=True)
            try:
                process_scene(scene_id, raw_tmp, output, args.frame_skip, args.attempts)
                journal.write(json.dumps({"scene": scene_id, "status": "done", "time": time.time()}) + "\n")
                journal.flush()
            except Exception as exc:
                journal.write(json.dumps({"scene": scene_id, "status": "failed", "error": repr(exc), "time": time.time()}) + "\n")
                journal.flush()
                print(f"ABORT after scene {scene_id}: {exc}", file=sys.stderr, flush=True)
                raise
    leftovers = list(raw_tmp.glob("*.sens")) + list(raw_tmp.glob("*.sens.part"))
    if leftovers:
        raise RuntimeError(f"raw leftovers remain: {leftovers}")


if __name__ == "__main__":
    main()
