#!/usr/bin/env python3
"""Download only evaluation ScanNet .sens files on the networked host.

Files are written to the shared ``raw_tmp`` directory as ``.part`` and renamed
only after the byte count is complete.  The GPU host consumes and deletes the
completed file with ``extract_eval_dense.py``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import ssl
import time
import urllib.request
from pathlib import Path

import csv

BASE = "https://kaldir.vc.cit.tum.de/scannet/v1/scans"
CHUNK_SIZE = 64 * 1024 * 1024


def scenes(ref: Path) -> list[str]:
    names = ("sr3d_val.csv", "ScanEnts3D_Nr3D_val.csv", "ScanRefer_filtered_val_ScanEnts3D_val.csv")
    ids: set[str] = set()
    for name in names:
        with (ref / name).open(newline="") as handle:
            rows = csv.DictReader(handle)
            column = "scan_id" if "scan_id" in (rows.fieldnames or []) else "scene_id"
            ids.update(str(row[column]) for row in rows)
    return sorted(ids)


def remote_size(url: str, ctx: ssl.SSLContext) -> int:
    """Read the remote size without starting a full transfer."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30, context=ctx) as response:
            length = response.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception:
        pass
    # A one-byte ranged GET is the fallback for servers/proxies that omit
    # Content-Length on HEAD.
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30, context=ctx) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" not in content_range:
            raise RuntimeError(f"remote size unavailable: {content_range!r}")
        return int(content_range.rsplit("/", 1)[1])


def download(scene: str, out: Path, attempts: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{scene}.sens"
    part = out / f"{scene}.sens.part"
    url = f"{BASE}/{scene}/{scene}.sens"
    ctx = ssl._create_unverified_context()
    total = remote_size(url, ctx)
    offset = part.stat().st_size if part.exists() else 0
    if offset > total:
        part.unlink()
        offset = 0
    failures = 0
    while offset < total:
        end = min(offset + CHUNK_SIZE, total) - 1
        chunk_path = part.with_name(part.name + ".chunk")
        if chunk_path.exists():
            chunk_path.unlink()
        try:
            request = urllib.request.Request(url)
            request.add_header("Range", f"bytes={offset}-{end}")
            with urllib.request.urlopen(request, timeout=60, context=ctx) as response:
                status = getattr(response, "status", None)
                if status != 206:
                    raise RuntimeError(f"expected 206 for range, got {status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                expected = end - offset + 1
                received = 0
                with chunk_path.open("wb") as handle:
                    while received < expected:
                        data = response.read(min(8 * 1024 * 1024, expected - received))
                        if not data:
                            break
                        handle.write(data)
                        received += len(data)
                if received != expected:
                    raise RuntimeError(f"chunk incomplete: got={received} expected={expected}")
            with part.open("ab") as handle, chunk_path.open("rb") as chunk:
                shutil.copyfileobj(chunk, handle, length=8 * 1024 * 1024)
            chunk_path.unlink()
            offset = end + 1
            print(f"PROGRESS {scene} {offset}/{total}", flush=True)
        except Exception as exc:
            failures += 1
            if chunk_path.exists():
                chunk_path.unlink()
            print(f"{scene} attempt {failures}/{attempts} failed at {offset}: {exc}", flush=True)
            if failures >= attempts:
                raise
            time.sleep(10 * failures)
    if part.stat().st_size != total:
        raise RuntimeError(f"size incomplete: got={part.stat().st_size} expected={total}")
    os.replace(part, final)
    print(f"DONE {scene} bytes={final.stat().st_size}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/shared-storage-user/yicheng-data/Qwen-3D"))
    p.add_argument("--scene", action="append")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--attempts", type=int, default=3)
    args = p.parse_args()
    ids = args.scene or scenes(args.root / "data/refer_it_3d")
    if args.max_scenes:
        ids = ids[: args.max_scenes]
    out = args.root / "data/posed_rgbd_eval_dense/raw_tmp"
    for i, scene in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {scene}", flush=True)
        final = out / f"{scene}.sens"
        if final.is_file() and final.stat().st_size:
            print(f"READY {scene} bytes={final.stat().st_size}", flush=True)
            continue
        download(scene, out, args.attempts)


if __name__ == "__main__":
    main()
