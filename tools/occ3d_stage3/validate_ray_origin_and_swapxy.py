#!/usr/bin/env python3
"""Lightweight diagnostics for ray-source origins and swapxy consistency."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ray origins and dense coordinate consistency.")
    parser.add_argument("--ray-source-points-dir", required=True)
    parser.add_argument("--ray-source-voxels-dir", required=True)
    parser.add_argument("--dense-candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    return parser.parse_args()


def transform_dense_coords(coords, occ_size, transform_name):
    if transform_name == "identity":
        return coords
    if transform_name == "swapxy_flipy":
        transformed = coords.copy()
        transformed[:, 0] = coords[:, 1]
        transformed[:, 1] = occ_size[1] - 1 - coords[:, 0]
        return transformed
    raise ValueError(transform_name)


def transform_metric_points(points, transform_name):
    if transform_name == "identity":
        return points
    if transform_name == "swapxy_flipy":
        transformed = points.copy()
        transformed[:, 0] = points[:, 1]
        transformed[:, 1] = -points[:, 0]
        return transformed
    raise ValueError(transform_name)


def points_to_voxels(points, pc_range, voxel_size, occ_size):
    pc_min = np.asarray(pc_range[:3], dtype=np.float32)
    pc_max = np.asarray(pc_range[3:], dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    coords = np.floor((points - pc_min[None, :]) / voxel_size[None, :]).astype(np.int64)
    valid = (
        (points >= pc_min[None, :]).all(axis=1)
        & (points < pc_max[None, :]).all(axis=1)
        & (coords >= 0).all(axis=1)
        & (coords < occ_size[None, :]).all(axis=1)
    )
    return coords[valid], valid


def unique_coord_set(coords):
    if coords.size == 0:
        return set()
    return {tuple(int(v) for v in row) for row in coords}


def load_voxel_file(path, occ_size, coord_transform):
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected (N, 4), got {data.shape}: {path}")
    coords = data[:, :3].astype(np.int64)
    labels = data[:, 3].astype(np.int64)
    coords = transform_dense_coords(coords, np.asarray(occ_size, dtype=np.int64), coord_transform)
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < occ_size[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < occ_size[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < occ_size[2])
        & (labels >= 0)
        & (labels <= 16)
    )
    return coords[valid], labels[valid]


def overlap_summary(coords_a, coords_b):
    set_a = unique_coord_set(coords_a)
    set_b = unique_coord_set(coords_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return {
        "a_count": len(set_a),
        "b_count": len(set_b),
        "intersection": inter,
        "union": union,
        "iou": float(inter / union) if union else 0.0,
        "a_in_b": float(inter / len(set_a)) if set_a else 0.0,
        "b_in_a": float(inter / len(set_b)) if set_b else 0.0,
    }


def origin_stats(origins):
    norms = np.linalg.norm(origins, axis=1)
    if origins.shape[0] == 0:
        return {}
    return {
        "count": int(origins.shape[0]),
        "near_zero_1cm": int((norms < 0.01).sum()),
        "near_zero_5cm": int((norms < 0.05).sum()),
        "near_zero_10cm": int((norms < 0.10).sum()),
        "near_zero_1m": int((norms < 1.0).sum()),
        "near_zero_10cm_fraction": float((norms < 0.10).mean()),
        "norm_mean": float(norms.mean()),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p90": float(np.percentile(norms, 90)),
        "norm_p99": float(np.percentile(norms, 99)),
        "norm_max": float(norms.max()),
        "xyz_mean": [float(v) for v in origins.mean(axis=0)],
        "xyz_min": [float(v) for v in origins.min(axis=0)],
        "xyz_max": [float(v) for v in origins.max(axis=0)],
        "z_mean": float(origins[:, 2].mean()),
        "z_std": float(origins[:, 2].std()),
        "z_min": float(origins[:, 2].min()),
        "z_max": float(origins[:, 2].max()),
        "abs_z_p99": float(np.percentile(np.abs(origins[:, 2]), 99)),
    }


def main():
    args = parse_args()
    ray_dir = Path(args.ray_source_points_dir)
    voxel_dir = Path(args.ray_source_voxels_dir)
    dense_dir = Path(args.dense_candidate_dir)
    occ_size = np.asarray(args.occ_size, dtype=np.int64)

    files = sorted(ray_dir.glob("*.npy"))
    totals = {
        "files": 0,
        "point_rows": 0,
        "near_zero_10cm": 0,
        "metric_swap_voxels": 0,
        "metric_identity_voxels": 0,
        "ray_voxels_swap": 0,
        "dense_voxels_swap": 0,
        "metric_swap_to_ray_intersection": 0,
        "metric_identity_to_ray_intersection": 0,
        "metric_swap_to_dense_intersection": 0,
        "metric_identity_to_dense_intersection": 0,
    }
    frame_reports = []

    all_origins_sample = []
    rng = np.random.default_rng(0)
    for path in files:
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] != 7:
            raise ValueError(f"Expected ray source points shape (N, 7), got {data.shape}: {path}")
        points = data[:, :3].astype(np.float32)
        origins = data[:, 3:6].astype(np.float32)
        labels = data[:, 6].astype(np.int64)

        points_swap = transform_metric_points(points, "swapxy_flipy")
        coords_swap, valid_swap = points_to_voxels(points_swap, args.pc_range, args.voxel_size, occ_size)
        coords_identity, valid_identity = points_to_voxels(points, args.pc_range, args.voxel_size, occ_size)

        voxel_path = voxel_dir / path.name
        dense_path = dense_dir / path.name
        ray_voxels_swap, _ = load_voxel_file(voxel_path, occ_size, "swapxy_flipy")
        dense_voxels_swap, _ = load_voxel_file(dense_path, occ_size, "swapxy_flipy")

        swap_to_ray = overlap_summary(coords_swap, ray_voxels_swap)
        identity_to_ray = overlap_summary(coords_identity, ray_voxels_swap)
        swap_to_dense = overlap_summary(coords_swap, dense_voxels_swap)
        identity_to_dense = overlap_summary(coords_identity, dense_voxels_swap)

        stats = origin_stats(origins)
        report = {
            "file": path.name,
            "rows": int(data.shape[0]),
            "origin_stats": stats,
            "valid_metric_swap_points": int(valid_swap.sum()),
            "valid_metric_identity_points": int(valid_identity.sum()),
            "point_voxel_swap_vs_ray_voxels_swap": swap_to_ray,
            "point_voxel_identity_vs_ray_voxels_swap": identity_to_ray,
            "point_voxel_swap_vs_dense_voxels_swap": swap_to_dense,
            "point_voxel_identity_vs_dense_voxels_swap": identity_to_dense,
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
        frame_reports.append(report)

        totals["files"] += 1
        totals["point_rows"] += int(data.shape[0])
        totals["near_zero_10cm"] += int(stats["near_zero_10cm"])
        totals["metric_swap_voxels"] += swap_to_ray["a_count"]
        totals["metric_identity_voxels"] += identity_to_ray["a_count"]
        totals["ray_voxels_swap"] += swap_to_ray["b_count"]
        totals["dense_voxels_swap"] += swap_to_dense["b_count"]
        totals["metric_swap_to_ray_intersection"] += swap_to_ray["intersection"]
        totals["metric_identity_to_ray_intersection"] += identity_to_ray["intersection"]
        totals["metric_swap_to_dense_intersection"] += swap_to_dense["intersection"]
        totals["metric_identity_to_dense_intersection"] += identity_to_dense["intersection"]

        if origins.shape[0] > 0:
            take = min(2000, origins.shape[0])
            idx = rng.choice(origins.shape[0], size=take, replace=False)
            all_origins_sample.append(origins[idx])

    if all_origins_sample:
        sampled_origins = np.concatenate(all_origins_sample, axis=0)
        aggregate_origin_stats = origin_stats(sampled_origins)
    else:
        aggregate_origin_stats = {}

    aggregate = {
        "near_zero_10cm_fraction": float(totals["near_zero_10cm"] / totals["point_rows"]) if totals["point_rows"] else 0.0,
        "metric_swap_to_ray_voxel_hit_fraction": float(
            totals["metric_swap_to_ray_intersection"] / totals["metric_swap_voxels"]
        )
        if totals["metric_swap_voxels"]
        else 0.0,
        "metric_identity_to_ray_voxel_hit_fraction": float(
            totals["metric_identity_to_ray_intersection"] / totals["metric_identity_voxels"]
        )
        if totals["metric_identity_voxels"]
        else 0.0,
        "metric_swap_to_dense_voxel_hit_fraction": float(
            totals["metric_swap_to_dense_intersection"] / totals["metric_swap_voxels"]
        )
        if totals["metric_swap_voxels"]
        else 0.0,
        "metric_identity_to_dense_voxel_hit_fraction": float(
            totals["metric_identity_to_dense_intersection"] / totals["metric_identity_voxels"]
        )
        if totals["metric_identity_voxels"]
        else 0.0,
        "sampled_origin_stats": aggregate_origin_stats,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "ray_source_points_dir": str(ray_dir),
        "ray_source_voxels_dir": str(voxel_dir),
        "dense_candidate_dir": str(dense_dir),
        "totals": totals,
        "aggregate": aggregate,
        "frames": frame_reports,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()
