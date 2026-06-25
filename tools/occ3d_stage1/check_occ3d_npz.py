#!/usr/bin/env python3
"""Check FlashOCC/Occ3D labels.npz files generated for Stage 1."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate labels.npz files.")
    parser.add_argument(
        "root",
        nargs="?",
        default="data/GT_occupancy_mini/stage1_occ3d_current/gts",
        help="GT root containing scene/token/labels.npz files, or one labels.npz file.",
    )
    parser.add_argument(
        "--occ-size",
        nargs=3,
        type=int,
        default=(200, 200, 16),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def iter_label_files(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.glob("*/*/labels.npz"))


def check_file(path, occ_size):
    data = np.load(path)
    keys = set(data.files)
    required = {"semantics", "mask_lidar", "mask_camera"}
    missing = sorted(required - keys)
    if missing:
        raise AssertionError(f"{path}: missing keys {missing}")

    semantics = data["semantics"]
    mask_lidar = data["mask_lidar"]
    mask_camera = data["mask_camera"]
    expected_shape = tuple(occ_size)

    if semantics.shape != expected_shape:
        raise AssertionError(f"{path}: semantics shape {semantics.shape} != {expected_shape}")
    if mask_lidar.shape != expected_shape:
        raise AssertionError(f"{path}: mask_lidar shape {mask_lidar.shape} != {expected_shape}")
    if mask_camera.shape != expected_shape:
        raise AssertionError(f"{path}: mask_camera shape {mask_camera.shape} != {expected_shape}")
    if semantics.dtype not in (np.uint8, np.int32, np.int64):
        raise AssertionError(f"{path}: unsupported semantics dtype {semantics.dtype}")
    if mask_lidar.dtype not in (np.bool_, np.uint8):
        raise AssertionError(f"{path}: unsupported mask_lidar dtype {mask_lidar.dtype}")
    if mask_camera.dtype not in (np.bool_, np.uint8):
        raise AssertionError(f"{path}: unsupported mask_camera dtype {mask_camera.dtype}")
    if semantics.min() < 0 or semantics.max() > 17:
        raise AssertionError(
            f"{path}: semantics range [{semantics.min()}, {semantics.max()}] outside 0-17"
        )

    occupied = (semantics != 17)
    return {
        "path": str(path),
        "semantics_dtype": str(semantics.dtype),
        "mask_lidar_dtype": str(mask_lidar.dtype),
        "mask_camera_dtype": str(mask_camera.dtype),
        "occupied_count": int(occupied.sum()),
        "free_or_unknown_count": int((semantics == 17).sum()),
        "mask_lidar_count": int(mask_lidar.astype(bool).sum()),
        "mask_camera_count": int(mask_camera.astype(bool).sum()),
        "labels": [int(v) for v in np.unique(semantics)],
    }


def main():
    args = parse_args()
    label_files = iter_label_files(args.root)
    if not label_files:
        raise FileNotFoundError(f"No labels.npz files found under {args.root}")

    file_stats = [check_file(path, args.occ_size) for path in label_files]
    summary = {
        "root": str(args.root),
        "occ_size": list(args.occ_size),
        "num_files": len(file_stats),
        "total_occupied": int(sum(item["occupied_count"] for item in file_stats)),
        "total_mask_lidar": int(sum(item["mask_lidar_count"] for item in file_stats)),
        "total_mask_camera": int(sum(item["mask_camera_count"] for item in file_stats)),
        "files": file_stats,
    }

    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
