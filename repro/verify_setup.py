#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CHECKPOINTS = {
    "3b": ("qwen3d_3b.pth", 7_894_337_084, "5bb643fe1979fc538af0c7897284a2f57308a589e3118a684c6a73d109891e84"),
    "7b": ("qwen3d_7b.pth", 17_239_044_731, "7c02974186df8d35ccd7334774575e42728ef5f47f5734cf37a7a1daedd949ca"),
}
BACKBONE_SHARDS = {
    "3b": (
        ("model-00001-of-00002.safetensors", 3_982_649_232, "41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8"),
        ("model-00002-of-00002.safetensors", 3_526_688_744, "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4"),
    ),
    "7b": (
        ("model-00001-of-00005.safetensors", 3_900_233_256, "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e"),
        ("model-00002-of-00005.safetensors", 3_864_726_320, "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024"),
        ("model-00003-of-00005.safetensors", 3_864_726_424, "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507"),
        ("model-00004-of-00005.safetensors", 3_864_733_680, "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed"),
        ("model-00005-of-00005.safetensors", 1_089_994_880, "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67"),
    ),
}
SCANNET_PRECOMPUTED = {
    "scannet_object_id_frame_map.pth": 30_819_686,
    "scannet_object_id_frame_map_clip.pth": 383_184_006,
    "scannet_object_id_frame_map_clip_text.pth": 1_255_926_372,
    "scannet_object_id_frame_map_filenames.pth": 2_200_408,
    "span_pred_text.pth": 2_582_268,
}


def file_info(path: Path, expected_bytes: int, expected_sha256: str, do_hash: bool) -> dict[str, object]:
    size = path.stat().st_size if path.is_file() else 0
    result: dict[str, object] = {
        "path": str(path), "exists": path.is_file(), "bytes": size,
        "expected_bytes": expected_bytes, "size_ok": size == expected_bytes,
        "expected_sha256": expected_sha256,
    }
    if do_hash and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
        result["sha256_ok"] = result["sha256"] == expected_sha256
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local Qwen-3D reproduction workspace")
    parser.add_argument("--runtime", action="store_true", help="also require GPU, compiled ops, and licensed ScanNet data")
    parser.add_argument("--size", choices=("3b", "7b"), default="3b")
    parser.add_argument("--hash", action="store_true", help="compute full SHA-256 for both large checkpoints")
    args = parser.parse_args()

    backbones: dict[str, object] = {}
    for size, shards in BACKBONE_SHARDS.items():
        name = f"Qwen2.5-VL-{size.upper()}-Instruct"
        root = ROOT / "models/backbones" / name
        shard_info = [file_info(root / filename, expected_size, sha256, args.hash)
                      for filename, expected_size, sha256 in shards]
        backbones[size] = {
            "config": (root / "config.json").is_file(),
            "shards": shard_info,
            "complete": (root / "config.json").is_file()
            and all(item["size_ok"] and (not args.hash or item.get("sha256_ok", False)) for item in shard_info),
        }

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "checkpoints": [file_info(ROOT / "models/qwen3d" / filename, size, sha256, args.hash)
                        for filename, size, sha256 in CHECKPOINTS.values()],
        "backbones": backbones,
        "annotations": {name: (ROOT / "data/refer_it_3d" / filename).is_file() for name, filename in {
            "nr3d_train": "ScanEnts3D_Nr3D_train.csv",
            "nr3d_val": "ScanEnts3D_Nr3D_val.csv",
            "scanrefer_train": "ScanRefer_filtered_train_ScanEnts3D_train.csv",
            "scanrefer_val": "ScanRefer_filtered_val_ScanEnts3D_val.csv",
            "sr3d_train": "sr3d_train.csv",
            "sr3d_val": "sr3d_val.csv",
            "scanqa_train": "ScanQA_v1.0_train.json",
            "scanqa_val": "ScanQA_v1.0_val.json",
            "sqa3d_train": "SQA_train.json",
            "sqa3d_val": "SQA_val.json",
        }.items()},
        "scannet_precomputed": {
            filename: ((ROOT / "data/scannet_precomputed" / filename).stat().st_size
                       if (ROOT / "data/scannet_precomputed" / filename).is_file() else 0) == expected_size
            for filename, expected_size in SCANNET_PRECOMPUTED.items()
        },
    }

    required_modules = ("torch", "transformers", "detectron2", "pytorch3d", "pointops2")
    modules = {name: importlib.util.find_spec(name) is not None for name in required_modules}
    report["modules"] = modules
    errors: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 is required")
    for item in report["checkpoints"]:
        if not item["exists"] or not item["size_ok"]:
            errors.append(f"missing/incomplete checkpoint: {item['path']}")
        if args.hash and not item.get("sha256_ok", False):
            errors.append(f"checkpoint SHA-256 mismatch: {item['path']}")
    if not all(item["complete"] for item in backbones.values()):
        errors.append("one or more Qwen2.5-VL backbone snapshots are missing")
    if not all(report["annotations"].values()):
        errors.append("one or more public annotation splits are missing")
    if not all(report["scannet_precomputed"].values()):
        errors.append("one or more UniVLG ScanNet precomputed files are missing/incomplete")

    if args.runtime:
        if shutil.which("nvidia-smi") is None:
            errors.append("nvidia-smi is unavailable")
        if not Path(os.environ.get("SCANNET_DATA_DIR", "")).is_file():
            errors.append("SCANNET_DATA_DIR does not point to train_validation_database.yaml")
        missing = [name for name, found in modules.items() if not found]
        if missing:
            errors.append("missing runtime modules: " + ", ".join(missing))

    report["errors"] = errors
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
