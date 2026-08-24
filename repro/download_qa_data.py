#!/usr/bin/env python3
from __future__ import annotations

import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import gdown

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
    urllib.request.urlretrieve(url, target)


def copy_json_tree(source: Path) -> None:
    for item in source.rglob("*.json"):
        target = REF / item.name
        if not target.exists():
            shutil.copy2(item, target)


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    scanqa = REF / "ScanQA_v1.0"
    retry("ScanQA annotations", lambda: gdown.download_folder(
        id="1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2", output=str(scanqa), quiet=False
    ))
    copy_json_tree(scanqa)

    for name in ("sqa_task.zip", "ScanQA_format.zip"):
        archive = REF / name
        url = f"https://zenodo.org/api/records/7792397/files/{name}/content"
        retry(f"SQA3D {name}", lambda u=url, a=archive: fetch(u, a))
        target = REF / name.removesuffix(".zip")
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        copy_json_tree(target)

    required = (
        "ScanQA_v1.0_train.json",
        "ScanQA_v1.0_val.json",
        "ScanQA_v1.0_test_w_obj.json",
        "SQA_train.json",
        "SQA_val.json",
        "SQA_test.json",
    )
    missing = [name for name in required if not (REF / name).is_file()]
    if missing:
        raise RuntimeError("downloaded archives do not contain expected Qwen-3D files: " + ", ".join(missing))
    print("ScanQA and SQA3D annotations downloaded and normalized.")


if __name__ == "__main__":
    main()
