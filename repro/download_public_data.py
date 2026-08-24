#!/usr/bin/env python3
from __future__ import annotations

import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/refer_it_3d"


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


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    nr3d = REF / "ScanEnts3D_Nr3D.csv"
    scanrefer_zip = REF / "ScanEnts3D_ScanRefer.zip"
    retry("ScanEnts3D Nr3D", lambda: fetch("https://scanents3d.github.io/ScanEnts3D_Nr3D.csv", nr3d))
    retry("ScanEnts3D ScanRefer", lambda: fetch("https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip", scanrefer_zip))
    with zipfile.ZipFile(scanrefer_zip) as archive:
        archive.extractall(REF)

    sr3d_tmp = REF / "Sr3D"
    retry("Sr3D", lambda: gdown.download_folder(
        id="1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV", output=str(sr3d_tmp), quiet=False
    ))
    if sr3d_tmp.is_dir():
        for item in sr3d_tmp.iterdir():
            target = REF / item.name
            if not target.exists():
                shutil.move(str(item), target)
        sr3d_tmp.rmdir()

    precomputed = ROOT / "data/scannet_precomputed"
    retry("UniVLG ScanNet metadata", lambda: snapshot_download(
        "katefgroup/UniVLG", allow_patterns="scannet/*", local_dir=precomputed, max_workers=4
    ))
    nested = precomputed / "scannet"
    if nested.is_dir():
        for item in nested.iterdir():
            target = precomputed / item.name
            if not target.exists():
                shutil.move(str(item), target)
        nested.rmdir()

    print("Public 3D grounding annotations and precomputed metadata downloaded.")


if __name__ == "__main__":
    main()
