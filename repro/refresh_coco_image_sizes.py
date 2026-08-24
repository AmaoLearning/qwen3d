#!/usr/bin/env python3
"""Refresh COCO image dimensions from the extracted RGB frames.

The high-resolution archives contain a small mixture of 640x480 and
1296x968 scenes, while the generated COCO metadata used a single 640x480
default.  Detectron2 correctly rejects that mismatch, so dimensions must be
read from the actual frame files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/posed_rgbd"))
    parser.add_argument(
        "--frames-dir", type=Path, default=Path("data/posed_rgbd/frames_square_highres")
    )
    parser.add_argument(
        "--pattern", default="*highres*.coco.json", help="Annotation filename glob"
    )
    args = parser.parse_args()

    changed = 0
    for annotation_file in sorted(args.data_root.glob(args.pattern)):
        data = json.loads(annotation_file.read_text())
        file_changed = 0
        for image in data.get("images", []):
            frame = args.frames_dir / image["file_name"]
            if not frame.is_file():
                raise FileNotFoundError(f"Missing RGB frame for {annotation_file}: {frame}")
            width, height = Image.open(frame).size
            if image.get("width") != width or image.get("height") != height:
                image["width"] = width
                image["height"] = height
                file_changed += 1
        if file_changed:
            annotation_file.write_text(json.dumps(data, indent=2) + "\n")
        changed += file_changed
        print(f"{annotation_file}: refreshed {file_changed}/{len(data.get('images', []))}")
    print(f"refreshed {changed} image records")


if __name__ == "__main__":
    main()
