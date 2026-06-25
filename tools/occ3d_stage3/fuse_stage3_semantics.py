#!/usr/bin/env python3
"""Fuse Stage 2 raw LiDAR ray casting with SurroundOcc semantic candidates.

The main output keeps Stage 2 geometry unchanged:

- mask_lidar is copied from Stage 2 and is not expanded.
- raw free voxels remain free.
- only raw-hit voxels may receive occupied semantics.

SurroundOcc sparse voxels can be consumed either from the original SurroundOcc
grid or from an already regenerated FlashOCC/Occ3D grid. When Stage 2 applies a
dense coordinate transform, Stage 3 must apply the same transform before fusion.
"""

import argparse
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


FREE_LABEL = 17


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 semantic fusion.")
    parser.add_argument(
        "--stage2-root",
        default="data/GT_occupancy_mini/stage2_raycast_occ3d",
        help="Stage 2 output root containing gts/, debug/ and stats.",
    )
    parser.add_argument(
        "--stage2-ann",
        default="data/GT_occupancy_mini/stage2_raycast_occ3d/bevdetv2-nuscenes_infos_stage2_train.pkl",
    )
    parser.add_argument(
        "--sparse-dir",
        default="data/GT_occupancy_mini/dense_voxels_with_semantic",
        help="SurroundOcc sparse semantic candidate directory.",
    )
    parser.add_argument(
        "--candidate-grid",
        choices=("surroundocc", "occ3d"),
        default="surroundocc",
        help=(
            "Coordinate convention of --sparse-dir. surroundocc converts from "
            "the original SurroundOcc metric grid; occ3d treats rows as direct "
            "FlashOCC/Occ3D voxel indices."
        ),
    )
    parser.add_argument(
        "--dense-coordinate-transform",
        choices=("identity", "swapxy_flipy"),
        default="identity",
        help="Optional coordinate transform applied to candidate voxel indices before fusion.",
    )
    parser.add_argument(
        "--output-root",
        default="data/GT_occupancy_mini/stage3_fused_occ3d",
    )
    parser.add_argument("--surround-pc-range", nargs=6, type=float, default=(-50, -50, -5, 50, 50, 3))
    parser.add_argument("--surround-voxel-size", type=float, default=0.5)
    parser.add_argument("--occ-pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--occ-voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    return parser.parse_args()


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data, data["infos"]
    if isinstance(data, list):
        return data, data
    raise ValueError(f"Unsupported annotation pkl format: {path}")


def save_ann_with_paths(ann_data, infos, output_path):
    if isinstance(ann_data, dict) and "infos" in ann_data:
        out = dict(ann_data)
        out["infos"] = infos
    else:
        out = infos
    with open(output_path, "wb") as f:
        pickle.dump(out, f)


def lidar_basename_from_info(info):
    return Path(info["lidar_path"]).name


def sparse_path_for_info(sparse_dir, info):
    return Path(sparse_dir) / (lidar_basename_from_info(info) + ".npy")


def majority_from_counts(counts):
    if int(counts.sum()) == 0:
        return 0
    return int(np.argmax(counts))


def transform_dense_coords(coords, occ_size, transform_name):
    if transform_name == "identity":
        return coords
    if transform_name == "swapxy_flipy":
        transformed = coords.copy()
        transformed[:, 0] = coords[:, 1]
        transformed[:, 1] = occ_size[1] - 1 - coords[:, 0]
        return transformed
    raise ValueError(f"Unsupported dense coordinate transform: {transform_name}")


def build_candidate_grid_from_occ_coords(coords, labels, args):
    occ_size = np.asarray(args.occ_size, dtype=np.int64)
    coords = transform_dense_coords(coords.astype(np.int64), occ_size, args.dense_coordinate_transform)
    labels = labels.astype(np.int64)

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
    coords = coords[valid]
    labels = labels[valid]

    candidate_grid = np.zeros(tuple(args.occ_size), dtype=bool)
    label_counts = np.zeros(tuple(args.occ_size) + (17,), dtype=np.uint16)
    conflict_voxels = 0

    if len(labels) == 0:
        return candidate_grid, label_counts, 0, int((~valid).sum())

    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    xyz_sorted = coords[order]
    labels_sorted = labels[order]
    unique_xyz, starts, counts = np.unique(
        xyz_sorted, axis=0, return_index=True, return_counts=True
    )

    for coord, start, count in zip(unique_xyz, starts, counts):
        label_slice = labels_sorted[start : start + count]
        counter = Counter(int(v) for v in label_slice)
        if len(counter) > 1:
            conflict_voxels += 1
        coord_tuple = tuple(int(v) for v in coord)
        candidate_grid[coord_tuple] = True
        for label, label_count in counter.items():
            label_counts[coord_tuple + (label,)] = min(label_count, 65535)

    return candidate_grid, label_counts, int(conflict_voxels), int((~valid).sum())


def convert_surroundocc_to_occ_grid(sparse, args):
    surround_min = np.asarray(args.surround_pc_range[:3], dtype=np.float32)
    occ_min = np.asarray(args.occ_pc_range[:3], dtype=np.float32)
    occ_max = np.asarray(args.occ_pc_range[3:], dtype=np.float32)
    occ_voxel_size = np.asarray(args.occ_voxel_size, dtype=np.float32)
    occ_size = np.asarray(args.occ_size, dtype=np.int64)

    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError(f"Expected sparse shape (N, 4), got {sparse.shape}")

    surround_xyz = sparse[:, :3].astype(np.float32)
    labels = sparse[:, 3].astype(np.int64)
    centers = surround_min[None, :] + (surround_xyz + 0.5) * float(args.surround_voxel_size)
    occ_xyz = np.floor((centers - occ_min[None, :]) / occ_voxel_size[None, :]).astype(np.int64)

    valid = (
        (centers >= occ_min[None, :]).all(axis=1)
        & (centers < occ_max[None, :]).all(axis=1)
        & (occ_xyz >= 0).all(axis=1)
        & (occ_xyz < occ_size[None, :]).all(axis=1)
        & (labels >= 0)
        & (labels <= 16)
    )
    occ_xyz = occ_xyz[valid]
    labels = labels[valid]

    candidate_grid, label_counts, conflict_voxels, transformed_out_of_range = (
        build_candidate_grid_from_occ_coords(occ_xyz, labels, args)
    )
    return (
        candidate_grid,
        label_counts,
        int(conflict_voxels),
        int((~valid).sum()) + int(transformed_out_of_range),
    )


def load_candidate_grid(sparse_path, args):
    sparse = np.load(sparse_path)
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError(f"Expected sparse shape (N, 4), got {sparse.shape}: {sparse_path}")

    if args.candidate_grid == "occ3d":
        return build_candidate_grid_from_occ_coords(sparse[:, :3], sparse[:, 3], args)
    return convert_surroundocc_to_occ_grid(sparse, args)


def build_raw_semantics(stage2_debug, free_grid, raw_hit_grid):
    hit_label_counts = stage2_debug["hit_label_counts"]
    semantics = np.full(raw_hit_grid.shape, FREE_LABEL, dtype=np.uint8)
    hit_coords = np.argwhere(raw_hit_grid)
    for x, y, z in hit_coords:
        semantics[x, y, z] = majority_from_counts(hit_label_counts[x, y, z])
    semantics[free_grid & ~raw_hit_grid] = FREE_LABEL
    return semantics


def fuse_frame(stage2_labels_path, stage2_debug_path, sparse_path, args):
    stage2_labels = np.load(stage2_labels_path)
    stage2_debug = np.load(stage2_debug_path)

    free_grid = stage2_debug["free_grid"].astype(bool)
    raw_hit_grid = stage2_debug["raw_hit_grid"].astype(bool)
    mask_lidar = stage2_labels["mask_lidar"].astype(bool)
    mask_camera = stage2_labels["mask_camera"].astype(bool)

    candidate_grid, candidate_counts, candidate_conflicts, candidate_out_of_range = (
        load_candidate_grid(sparse_path, args)
    )
    raw_semantics = build_raw_semantics(stage2_debug, free_grid, raw_hit_grid)
    fused_semantics = raw_semantics.copy()

    raw_hit_coords = np.argwhere(raw_hit_grid)
    raw_hit_count = int(raw_hit_grid.sum())
    candidate_raw_hit_count = 0
    changed_semantic_count = 0
    raw_hit_without_candidate = 0
    candidate_only_count = int((candidate_grid & ~raw_hit_grid).sum())

    for x, y, z in raw_hit_coords:
        if candidate_grid[x, y, z]:
            candidate_raw_hit_count += 1
            candidate_label = majority_from_counts(candidate_counts[x, y, z])
            if fused_semantics[x, y, z] != candidate_label:
                changed_semantic_count += 1
            fused_semantics[x, y, z] = candidate_label
        else:
            raw_hit_without_candidate += 1

    # Safety assertions for the Stage 3 contract.
    assert np.array_equal(mask_lidar, free_grid | raw_hit_grid)
    assert not np.any((candidate_grid & ~raw_hit_grid) & (fused_semantics != raw_semantics))
    assert not np.any((free_grid & ~raw_hit_grid) & (fused_semantics != FREE_LABEL))

    stats = {
        "raw_hit_count": raw_hit_count,
        "candidate_occ_count": int(candidate_grid.sum()),
        "candidate_on_raw_hit_count": int(candidate_raw_hit_count),
        "candidate_only_count": candidate_only_count,
        "raw_hit_without_candidate": int(raw_hit_without_candidate),
        "changed_semantic_count": int(changed_semantic_count),
        "candidate_conflict_voxels": int(candidate_conflicts),
        "candidate_out_of_occ3d_range_rows": int(candidate_out_of_range),
        "free_count": int((mask_lidar & (fused_semantics == FREE_LABEL) & ~raw_hit_grid).sum()),
        "observed_count": int(mask_lidar.sum()),
        "unknown_count": int((~mask_lidar).sum()),
    }
    return fused_semantics, mask_lidar, mask_camera, candidate_grid, stats


def main():
    args = parse_args()
    stage2_root = Path(args.stage2_root)
    output_root = Path(args.output_root)
    gts_root = output_root / "gts"
    debug_root = output_root / "debug"
    ann_output = output_root / "bevdetv2-nuscenes_infos_stage3_train.pkl"
    stats_output = output_root / "stats_semantic.json"

    gts_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    ann_data, infos = load_infos(args.stage2_ann)
    updated_infos = []
    file_stats = []

    for frame_idx, info in enumerate(infos):
        scene_name = info["scene_name"]
        token = info["token"]
        stage2_labels_path = Path(info["occ_path"]) / "labels.npz"
        stage2_debug_path = stage2_root / "debug" / scene_name / token / "raycast_debug.npz"
        sparse_path = sparse_path_for_info(args.sparse_dir, info)

        if not sparse_path.exists():
            raise FileNotFoundError(f"Missing SurroundOcc sparse candidate: {sparse_path}")
        if not stage2_debug_path.exists():
            raise FileNotFoundError(f"Missing Stage 2 debug file: {stage2_debug_path}")

        fused_semantics, mask_lidar, mask_camera, candidate_grid, stats = fuse_frame(
            stage2_labels_path, stage2_debug_path, sparse_path, args
        )

        occ_dir = gts_root / scene_name / token
        debug_dir = debug_root / scene_name / token
        occ_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            occ_dir / "labels.npz",
            semantics=fused_semantics,
            mask_lidar=mask_lidar,
            mask_camera=mask_camera,
        )
        np.savez_compressed(debug_dir / "stage3_debug.npz", poisson_occ_grid=candidate_grid)

        updated_info = dict(info)
        updated_info["occ_path"] = str(occ_dir.resolve())
        updated_infos.append(updated_info)

        stats.update(
            {
                "frame_idx": frame_idx,
                "scene_name": scene_name,
                "token": token,
                "sparse_path": str(sparse_path),
                "occ_path": str(occ_dir.resolve()),
                "debug_path": str((debug_dir / "stage3_debug.npz").resolve()),
            }
        )
        file_stats.append(stats)
        print(
            f"[{frame_idx + 1}/{len(infos)}] {scene_name} {token}: "
            f"raw_hit={stats['raw_hit_count']} candidate_on_raw={stats['candidate_on_raw_hit_count']} "
            f"changed={stats['changed_semantic_count']} candidate_only={stats['candidate_only_count']}"
        )

    save_ann_with_paths(ann_data, updated_infos, ann_output)

    summary = {
        "stage": "Stage 3: semantic fusion",
        "fusion_policy": (
            "mask_lidar/free/raw_hit geometry is copied from Stage 2. SurroundOcc "
            "semantic candidates can change labels only on raw_hit_grid voxels."
        ),
        "grid": {
            "occ_point_cloud_range": [float(v) for v in args.occ_pc_range],
            "occ_voxel_size": [float(v) for v in args.occ_voxel_size],
            "occ_size": [int(v) for v in args.occ_size],
            "surroundocc_point_cloud_range": [float(v) for v in args.surround_pc_range],
            "surroundocc_voxel_size": float(args.surround_voxel_size),
            "candidate_grid": args.candidate_grid,
            "dense_coordinate_transform": args.dense_coordinate_transform,
        },
        "stage2_root": str(stage2_root),
        "sparse_dir": str(args.sparse_dir),
        "ann_output": str(ann_output),
        "gts_root": str(gts_root),
        "debug_root": str(debug_root),
        "num_frames": len(file_stats),
        "totals": {
            "raw_hit_count": int(sum(item["raw_hit_count"] for item in file_stats)),
            "candidate_occ_count": int(sum(item["candidate_occ_count"] for item in file_stats)),
            "candidate_on_raw_hit_count": int(sum(item["candidate_on_raw_hit_count"] for item in file_stats)),
            "candidate_only_count": int(sum(item["candidate_only_count"] for item in file_stats)),
            "raw_hit_without_candidate": int(sum(item["raw_hit_without_candidate"] for item in file_stats)),
            "changed_semantic_count": int(sum(item["changed_semantic_count"] for item in file_stats)),
            "free_count": int(sum(item["free_count"] for item in file_stats)),
            "observed_count": int(sum(item["observed_count"] for item in file_stats)),
            "unknown_count": int(sum(item["unknown_count"] for item in file_stats)),
        },
        "files": file_stats,
    }
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote Stage 3 GT root: {gts_root}")
    print(f"Wrote Stage 3 ann pkl: {ann_output}")
    print(f"Wrote stats: {stats_output}")


if __name__ == "__main__":
    main()
