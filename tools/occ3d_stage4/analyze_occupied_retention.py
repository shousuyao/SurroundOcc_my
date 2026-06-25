#!/usr/bin/env python3
"""Aggregate global, LiDAR-observed, and camera-visible occupied retention."""

import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "free",
]


def count_label_arrays(semantics, mask_lidar, mask_camera, num_classes=18):
    semantics = np.asarray(semantics)
    mask_lidar = np.asarray(mask_lidar, dtype=bool)
    mask_camera = np.asarray(mask_camera, dtype=bool)
    if semantics.shape != mask_lidar.shape or semantics.shape != mask_camera.shape:
        raise ValueError("semantics, mask_lidar, and mask_camera must have identical shapes")
    if semantics.size and (int(semantics.min()) < 0 or int(semantics.max()) >= num_classes):
        raise ValueError("semantic label is outside [0, num_classes)")
    if np.any(mask_camera & ~mask_lidar):
        raise ValueError("mask_camera must be a subset of mask_lidar")

    return {
        "global": np.bincount(semantics.ravel(), minlength=num_classes).astype(np.int64),
        "lidar": np.bincount(semantics[mask_lidar].ravel(), minlength=num_classes).astype(np.int64),
        "camera": np.bincount(semantics[mask_camera].ravel(), minlength=num_classes).astype(np.int64),
    }


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def aggregate_labels(label_paths, num_classes=18):
    totals = {
        "global": np.zeros(num_classes, dtype=np.int64),
        "lidar": np.zeros(num_classes, dtype=np.int64),
        "camera": np.zeros(num_classes, dtype=np.int64),
    }
    for path in label_paths:
        labels = np.load(path)
        counts = count_label_arrays(
            labels["semantics"], labels["mask_lidar"], labels["mask_camera"], num_classes
        )
        for key in totals:
            totals[key] += counts[key]

    free_label = num_classes - 1
    global_occupied = int(totals["global"][:free_label].sum())
    lidar_occupied = int(totals["lidar"][:free_label].sum())
    camera_occupied = int(totals["camera"][:free_label].sum())
    per_class = []
    for class_id in range(num_classes):
        global_count = int(totals["global"][class_id])
        lidar_count = int(totals["lidar"][class_id])
        camera_count = int(totals["camera"][class_id])
        per_class.append({
            "id": class_id,
            "name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
            "global": global_count,
            "lidar": lidar_count,
            "camera": camera_count,
            "lidar_over_global": _ratio(lidar_count, global_count),
            "camera_over_global": _ratio(camera_count, global_count),
            "camera_over_lidar": _ratio(camera_count, lidar_count),
        })

    return {
        "frames": int(len(label_paths)),
        "totals": {
            "global_occupied": global_occupied,
            "lidar_occupied": lidar_occupied,
            "camera_occupied": camera_occupied,
            "camera_over_global_occupied": _ratio(camera_occupied, global_occupied),
            "camera_over_lidar_occupied": _ratio(camera_occupied, lidar_occupied),
            "camera_mask": int(totals["camera"].sum()),
            "lidar_mask": int(totals["lidar"].sum()),
        },
        "per_class": per_class,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", required=True, help="Root containing gts/<scene>/<token>/labels.npz")
    parser.add_argument("--official-root", default=None, help="Optional official root containing <scene>/<token>/labels.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, default=18)
    return parser.parse_args()


def main():
    args = parse_args()
    labels_root = Path(args.labels_root)
    own_paths = sorted(labels_root.glob("gts/*/*/labels.npz"))
    if not own_paths:
        raise FileNotFoundError("No labels.npz found under {}".format(labels_root))
    report = {"generated": aggregate_labels(own_paths, args.num_classes)}

    if args.official_root:
        official_root = Path(args.official_root)
        official_paths = []
        missing = []
        for path in own_paths:
            scene_name = path.parent.parent.name
            token = path.parent.name
            official_path = official_root / scene_name / token / "labels.npz"
            if official_path.exists():
                official_paths.append(official_path)
            else:
                missing.append(str(official_path))
        report["official"] = aggregate_labels(official_paths, args.num_classes)
        report["official_missing"] = missing

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["generated"]["totals"], indent=2))
    print("wrote {}".format(output))


if __name__ == "__main__":
    main()
