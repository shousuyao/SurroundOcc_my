#!/usr/bin/env python3
"""Analyze occupied voxels inside camera frusta that are not camera first hits."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation",
]


def lidar_points_to_camera(points, sensor2lidar_rotation, sensor2lidar_translation):
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(sensor2lidar_rotation, dtype=np.float64)
    translation = np.asarray(sensor2lidar_translation, dtype=np.float64)
    return (points - translation[None, :]) @ rotation


def project_camera_points(points, intrinsic, width, height, depth_max):
    points = np.asarray(points, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    depth = points[:, 2]
    positive = (depth > 0.0) & (depth <= float(depth_max))
    safe_depth = np.where(positive, depth, 1.0)
    u = intrinsic[0, 0] * points[:, 0] / safe_depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * points[:, 1] / safe_depth + intrinsic[1, 2]
    visible = positive & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    return visible, depth


def frustum_union(points_lidar, cameras, width, height, depth_max):
    union = np.zeros(points_lidar.shape[0], dtype=bool)
    min_depth = np.full(points_lidar.shape[0], np.inf, dtype=np.float64)
    per_camera = {}
    for camera in cameras:
        points_camera = lidar_points_to_camera(
            points_lidar,
            camera["sensor2lidar_rotation"],
            camera["sensor2lidar_translation"],
        )
        visible, depth = project_camera_points(
            points_camera, camera["intrinsic"], width, height, depth_max
        )
        union |= visible
        min_depth[visible] = np.minimum(min_depth[visible], depth[visible])
        per_camera[camera["name"]] = int(visible.sum())
    return union, min_depth, per_camera


def _load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["infos"] if isinstance(data, dict) else data


def _hist(values, bins):
    return [int(value) for value in np.histogram(values, bins=bins)[0]]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ann", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-root", default=None, help="Optional <scene>/<token>/labels.npz root")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=(1600, 900))
    parser.add_argument("--depth-max", type=float, default=45.0)
    parser.add_argument("--pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    parser.add_argument("--depth-bins", nargs="+", type=float, default=(0, 5, 10, 15, 20, 30, 45))
    return parser.parse_args()


def main():
    args = parse_args()
    infos = [
        info for info in _load_infos(args.input_ann)
        if args.scene_name is None or info.get("scene_name") == args.scene_name
    ]
    if args.max_frames is not None:
        infos = infos[:args.max_frames]
    pc_min = np.asarray(args.pc_range[:3], dtype=np.float64)
    voxel_size = np.asarray(args.voxel_size, dtype=np.float64)
    width, height = args.image_size
    depth_bins = np.asarray(args.depth_bins, dtype=np.float64)

    totals = {key: np.zeros(17, dtype=np.int64) for key in (
        "global", "in_frustum", "first_hit", "frustum_non_first"
    )}
    class_depth = np.zeros((17, len(depth_bins) - 1), dtype=np.int64)
    class_z = np.zeros((17, 16), dtype=np.int64)
    per_camera = {}
    first_hit_outside_frustum = 0
    files = []

    for info in infos:
        if args.labels_root:
            labels_path = Path(args.labels_root) / info["scene_name"] / info["token"] / "labels.npz"
        else:
            labels_path = Path(info["occ_path"]) / "labels.npz"
        labels = np.load(labels_path)
        semantics = labels["semantics"]
        mask_camera = labels["mask_camera"].astype(bool)
        coords = np.argwhere(semantics != 17)
        classes = semantics[tuple(coords.T)].astype(np.int64)
        centers = pc_min[None, :] + (coords.astype(np.float64) + 0.5) * voxel_size[None, :]
        cameras = [{
            "name": name,
            "intrinsic": np.asarray(cam["cam_intrinsic"]),
            "sensor2lidar_rotation": np.asarray(cam["sensor2lidar_rotation"]),
            "sensor2lidar_translation": np.asarray(cam["sensor2lidar_translation"]),
        } for name, cam in info["cams"].items()]
        in_frustum, min_depth, frame_per_camera = frustum_union(
            centers, cameras, width, height, args.depth_max
        )
        first_hit = mask_camera[tuple(coords.T)]
        non_first = in_frustum & ~first_hit
        outside = first_hit & ~in_frustum
        first_hit_outside_frustum += int(outside.sum())
        for name, count in frame_per_camera.items():
            per_camera[name] = per_camera.get(name, 0) + int(count)

        masks = {
            "global": np.ones(classes.shape[0], dtype=bool),
            "in_frustum": in_frustum,
            "first_hit": first_hit,
            "frustum_non_first": non_first,
        }
        for key, mask in masks.items():
            totals[key] += np.bincount(classes[mask], minlength=17)
        for class_id in range(17):
            class_mask = non_first & (classes == class_id)
            class_depth[class_id] += np.histogram(min_depth[class_mask], bins=depth_bins)[0]
            class_z[class_id] += np.bincount(coords[class_mask, 2], minlength=16)[:16]
        files.append({
            "scene_name": info["scene_name"],
            "token": info["token"],
            "global_occupied": int(classes.size),
            "in_frustum": int(in_frustum.sum()),
            "first_hit": int(first_hit.sum()),
            "frustum_non_first": int(non_first.sum()),
            "first_hit_outside_center_frustum": int(outside.sum()),
        })

    per_class = []
    for class_id, name in enumerate(CLASS_NAMES):
        per_class.append({
            "id": class_id,
            "name": name,
            **{key: int(totals[key][class_id]) for key in totals},
            "first_hit_over_frustum": (
                float(totals["first_hit"][class_id] / totals["in_frustum"][class_id])
                if totals["in_frustum"][class_id] else None
            ),
            "non_first_depth_hist": class_depth[class_id].tolist(),
            "non_first_z_hist": class_z[class_id].tolist(),
        })
    report = {
        "input_ann": args.input_ann,
        "frames": len(files),
        "depth_bins": depth_bins.tolist(),
        "totals": {key: int(value.sum()) for key, value in totals.items()},
        "first_hit_outside_center_frustum": int(first_hit_outside_frustum),
        "per_camera_frustum_with_overlap": per_camera,
        "per_class": per_class,
        "files": files,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["totals"], indent=2))
    print("wrote {}".format(output))


if __name__ == "__main__":
    main()
