#!/usr/bin/env python3
"""Build Qwen-3D's ScanNet axis-alignment lookup from raw ScanNet metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scans",
        type=Path,
        default=Path("data/raw/scannet/scans"),
        help="ScanNet scans directory containing sceneXXXX_YY/sceneXXXX_YY.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qwen3d/data_video/scans_axis_alignment_matrices.json"),
    )
    args = parser.parse_args()

    matrices: dict[str, list[float]] = {}
    for metadata in sorted(args.scans.glob("*/*.txt")):
        scene = metadata.stem
        line = next(
            (
                item
                for item in metadata.read_text(errors="replace").splitlines()
                if item.startswith("axisAlignment = ")
            ),
            None,
        )
        if line is None:
            continue
        values = [float(value) for value in line.split()[2:]]
        if len(values) != 16:
            raise ValueError(f"{metadata}: expected 16 axisAlignment values")
        matrices[scene] = values

    if not matrices:
        raise FileNotFoundError(f"No ScanNet axisAlignment metadata found under {args.scans}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrices, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(matrices)} scene matrices to {args.output}")


if __name__ == "__main__":
    main()
