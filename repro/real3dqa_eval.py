#!/usr/bin/env python3
"""Run Qwen-3D generation on the Real-3DQA point-cloud benchmark.

Real-3DQA ships point clouds (xyz, rgb, labels) but no RGB-D frames.  Qwen-3D's
public forward path consumes RGB frames and an xyz map aligned with the visual
tokens, so this adapter makes a deterministic orthographic RGB/xyz projection
from the supplied point cloud.  It deliberately uses the model's own
generation-only path and never uses the answer while predicting.

The adapter is an explicit feasibility baseline, not a replacement for the
authors' RGB-D preprocessing.  Its output is compatible with the official
Real-3DQA EM and VRS evaluators.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _allow_safe_numpy_pickle() -> None:
    """Allow only numpy containers used by the published .pth point clouds."""
    from numpy.core.multiarray import _reconstruct

    torch.serialization.add_safe_globals(
        [
            _reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.Float32DType,
            np.dtypes.Int64DType,
        ]
    )


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _allow_safe_numpy_pickle()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        raise ValueError(f"Unexpected Real-3DQA point-cloud payload: {path}")
    xyz = np.asarray(payload[0], dtype=np.float32)
    rgb = np.asarray(payload[1], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError(f"Expected xyz/rgb arrays of shape [N,3], got {xyz.shape}/{rgb.shape}")
    if float(rgb.max(initial=0.0)) <= 1.5:
        rgb = rgb * 255.0
    return xyz, np.clip(rgb, 0.0, 255.0)


def _yaw_from_quaternion(q: dict[str, Any]) -> float:
    x, y, z, w = (float(q.get(k, 0.0)) for k in ("_x", "_y", "_z", "_w"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def render_projection(
    xyz: np.ndarray,
    rgb: np.ndarray,
    position: dict[str, Any],
    rotation: dict[str, Any],
    size: int = 448,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create CHW uint8 RGB and HWC float32 xyz images."""
    pos = np.array([float(position.get(k, 0.0)) for k in ("x", "y", "z")], dtype=np.float32)
    rel = xyz - pos[None]
    yaw = _yaw_from_quaternion(rotation)
    c, s = math.cos(yaw), math.sin(yaw)
    # Camera frame: x is horizontal, z is vertical, y is view depth.
    cam_x = c * rel[:, 0] + s * rel[:, 1]
    cam_y = -s * rel[:, 0] + c * rel[:, 1]
    cam_z = rel[:, 2]

    def bounds(values: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(values, [1.0, 99.0]).tolist()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-5:
            lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-5:
            hi = lo + 1.0
        margin = 0.04 * (hi - lo)
        return lo - margin, hi + margin

    x0, x1 = bounds(cam_x)
    z0, z1 = bounds(cam_z)
    u = np.clip(((cam_x - x0) / (x1 - x0) * (size - 1)).round().astype(np.int64), 0, size - 1)
    v = np.clip(((z1 - cam_z) / (z1 - z0) * (size - 1)).round().astype(np.int64), 0, size - 1)
    flat = v * size + u

    image = np.full((size * size, 3), np.clip(rgb.mean(axis=0), 0, 255), dtype=np.float32)
    xyz_image = np.broadcast_to(xyz.mean(axis=0), (size * size, 3)).copy()
    # The final point in each pixel wins; sorting makes the selection stable.
    order = np.lexsort((np.arange(len(flat)), cam_y))
    image[flat[order]] = rgb[order]
    xyz_image[flat[order]] = xyz[order]
    image = image.reshape(size, size, 3).clip(0, 255).astype(np.uint8)
    xyz_image = xyz_image.reshape(size, size, 3).astype(np.float32)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous(), torch.from_numpy(xyz_image)


def _find_split_dir(root: Path) -> Path:
    for candidate in (root / "data" / "test", root / "data" / "data" / "test"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find Real-3DQA test splits below {root}")


class Real3DQADataset:
    def __init__(self, root: Path, split: str, projection_size: int):
        self.root = root
        self.split = split
        split_name = "debiased_test" if split == "debiased" else split
        self.jsonl = _find_split_dir(root) / f"{split_name}.jsonl"
        if not self.jsonl.is_file():
            raise FileNotFoundError(self.jsonl)
        self.pointcloud_dir = root / "data" / "point_clouds"
        self.projection_size = projection_size
        self.rows = [json.loads(line) for line in self.jsonl.read_text().splitlines() if line.strip()]
        self.pointclouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.projections: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.missing = [r for r in self.rows if not (self.pointcloud_dir / f"{r['scene_id']}.pth").is_file()]
        self.rows = [r for r in self.rows if (self.pointcloud_dir / f"{r['scene_id']}.pth").is_file()]

    def _projection(self, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            row["scene_id"],
            json.dumps([row.get("position", {}), row.get("rotation", {})], sort_keys=True),
        )
        if key not in self.projections:
            if row["scene_id"] not in self.pointclouds:
                self.pointclouds[row["scene_id"]] = load_point_cloud(
                    self.pointcloud_dir / f"{row['scene_id']}.pth"
                )
            self.projections[key] = render_projection(
                *self.pointclouds[row["scene_id"]],
                row.get("position", {}),
                row.get("rotation", {}),
                self.projection_size,
            )
        return self.projections[key]

    def sample(self, row: dict[str, Any], processor: Any, xyz_scale: int = 7) -> dict[str, Any]:
        image, xyz = self._projection(row)
        from qwen3d.data_video.data_utils import qwen_preprocess_frames

        pixel_values, grid = qwen_preprocess_frames(processor.image_processor, [image])
        # Qwen2.5-VL uses 14px patches merged 2x2: visual features are 28px cells.
        feature_h = int(grid[0][1].item()) // 2
        feature_w = int(grid[0][2].item()) // 2
        target_h, target_w = feature_h * 4, feature_w * 4
        xyz_chw = xyz.permute(2, 0, 1).unsqueeze(0)
        xyz_fine = torch.nn.functional.interpolate(
            xyz_chw, size=(target_h, target_w), mode="nearest"
        ).squeeze(0).permute(1, 2, 0).contiguous()
        xyz_token = torch.nn.functional.interpolate(
            xyz_chw, size=(feature_h, feature_w), mode="nearest"
        ).squeeze(0).permute(1, 2, 0).contiguous()
        prompt = (row.get("situation", "") + " " + row["question"]).strip()
        return {
            "images": [image],
            "qwen_pixel_values": pixel_values,
            "qwen_grid_thw": grid,
            "multi_scale_xyz": [xyz_fine.unsqueeze(0), xyz_token.unsqueeze(0)],
            "new_h": feature_h,
            "new_w": feature_w,
            "decoder_3d": True,
            "actual_decoder_3d": True,
            "dataset_name": "real3dqa_bench",
            "text_caption": prompt,
            "answer": row["answer"],
            "generate_only": True,
            "do_generate": True,
            "width": image.shape[2],
            "height": image.shape[1],
            "full_scene_dataset": True,
            "use_ghost": False,
            "do_camera_drop": False,
            "camera_drop_prob": 0.0,
            "camera_drop_min_frames_keep": 1,
            "always_keep_first_frame": True,
            "max_frames": 1,
            "pseudo_2d_aug": False,
        }


def build_cfg(args: argparse.Namespace):
    from detectron2.engine import default_argument_parser
    from train import setup

    opts = [
        "MODEL.WEIGHTS", str(args.checkpoint),
        "QWEN_MODEL", str(args.backbone),
        "OUTPUT_DIR", str(args.output_dir),
        "DATASETS.TRAIN", "('real3dqa_bench',)",
        "DATASETS.TEST", "('real3dqa_bench',)",
        "MODEL.DECODER_3D", "True",
        "GENERATION", "True",
        "QA_GROUND_LOSS", "False",
        "USE_GHOST_POINTS", "False",
        "CACHE_QWEN_FEATURES", "False",
        "USE_WANDB", "False",
    ]
    train_args = default_argument_parser().parse_args(
        ["--config-file", str(ROOT / "qwen3d" / "configs" / "qwen_3d.yaml"),
         "--num-gpus", "1", "--num-machines", "1", "--machine-rank", "0",
         "--dist-url", "tcp://127.0.0.1:19877"] + opts
    )
    return setup(train_args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("3b", "7b"), required=True)
    parser.add_argument("--split", choices=("rotation_0", "rotation_90", "rotation_180", "rotation_270", "debiased"), required=True)
    parser.add_argument("--real3dqa-root", type=Path, default=Path("/mnt/shared-storage-user/yicheng-data/Real-3DQA"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--projection-size", type=int, default=448)
    args = parser.parse_args()
    args.checkpoint = ROOT / "models" / "qwen3d" / f"qwen3d_{args.size}.pth"
    args.backbone = ROOT / "models" / "backbones" / f"Qwen2.5-VL-{args.size.upper()}-Instruct"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Real3DQADataset(args.real3dqa_root, args.split, args.projection_size)
    rows = dataset.rows[: args.limit] if args.limit > 0 else dataset.rows
    print(json.dumps({"split": args.split, "input": len(dataset.rows), "missing_pointcloud": len(dataset.missing), "running": len(rows)}, ensure_ascii=False), flush=True)

    from detectron2.checkpoint import DetectionCheckpointer
    from train import Trainer

    cfg = build_cfg(args)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=str(args.output_dir)).resume_or_load(str(args.checkpoint), resume=False)
    model.eval()
    processor = model.qwen_processor
    predictions = []
    with torch.inference_mode():
        for idx, row in enumerate(rows):
            sample = dataset.sample(row, processor)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded, _ = model([sample])
            prediction = decoded[0] if isinstance(decoded, list) else str(decoded)
            predictions.append({
                "question_id": row["question_id"],
                "scene_id": row["scene_id"],
                "response_gt": [row["answer"]],
                "response_pred": prediction.strip(),
                "question_type": row.get("question_type"),
            })
            if (idx + 1) % 25 == 0 or idx + 1 == len(rows):
                print(f"processed {idx + 1}/{len(rows)}", flush=True)

    out_file = args.output_dir / f"{args.split}.json"
    out_file.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "size": args.size,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "predictions": str(out_file),
        "input_rows": len(dataset.rows),
        "missing_pointcloud_rows": len(dataset.missing),
        "predicted_rows": len(predictions),
    }
    (args.output_dir / f"{args.split}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
