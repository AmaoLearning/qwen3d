#!/usr/bin/env python3
"""Build a ScanNet200 COCO index from the dense evaluation extraction."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from extract_eval_dense import scene_ids, validate_scene


def frame_number(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):012d}")
    except ValueError:
        return (1, path.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/shared-storage-user/yicheng-data/Qwen-3D"))
    p.add_argument("--frames", type=Path, default=None)
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    root = args.root.resolve()
    frames = (args.frames or root / "data/posed_rgbd_eval_dense/frames_square_highres").resolve()
    template = (args.template or root / "data/posed_rgbd/scannet200_highres_val.coco.json").resolve()
    output = (args.output or root / "data/posed_rgbd_eval_dense/scannet200_highres_val.coco.json").resolve()
    base = json.loads(template.read_text())
    ids = scene_ids(root / "data/refer_it_3d")
    images: list[dict] = []
    depths: list[dict] = []
    poses: list[dict] = []
    image_id = 1
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    missing: list[str] = []
    invalid_pose = 0
    for scene in ids:
        scene_dir = frames / scene
        ok, detail = validate_scene(scene_dir)
        if not ok:
            missing.append(f"{scene}: {detail}")
            continue
        colors = sorted((scene_dir / "color").glob("*"), key=frame_number)
        for color in colors:
            stem = color.stem
            pose = scene_dir / "pose" / f"{stem}.txt"
            depth = scene_dir / "depth" / f"{stem}.png"
            if not depth.is_file() or not pose.is_file():
                missing.append(f"{scene}/{stem}: missing paired depth/pose")
                continue
            try:
                if not np.isfinite(np.loadtxt(pose)).all():
                    invalid_pose += 1
                    continue
                with Image.open(color) as image:
                    width, height = image.size
            except Exception as exc:
                missing.append(f"{scene}/{stem}: {exc}")
                continue
            rel_color = f"{scene}/color/{color.name}"
            rel_depth = f"{scene}/depth/{depth.name}"
            rel_pose = f"{scene}/pose/{pose.name}"
            common = {"id": image_id, "width": width, "height": height, "license": 1, "date_captured": stamp}
            images.append({"file_name": rel_color, **common})
            depths.append({"file_name": rel_depth, **common})
            poses.append({"file_name": rel_pose, **common})
            image_id += 1
    if missing:
        raise RuntimeError("incomplete dense extraction:\n" + "\n".join(missing[:40]))
    result = {
        "info": base.get("info", {}),
        "licenses": base.get("licenses", []),
        "categories": base.get("categories", []),
        "images": images,
        "depths": depths,
        "poses": poses,
        "valids": [],
        "segments": [],
        "annotations": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"scenes={len(ids)} images={len(images)} invalid_pose_skipped={invalid_pose}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
