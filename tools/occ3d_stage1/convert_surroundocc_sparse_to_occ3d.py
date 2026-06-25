#!/usr/bin/env python3
"""Convert SurroundOcc sparse occupancy npy files to FlashOCC labels.npz.

Stage 1 intentionally does not perform ray casting. It only materializes the
Occ3D/FlashOCC file shape and keys so the data pipeline can be smoke-tested.
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
    parser = argparse.ArgumentParser(
        description="Convert sparse (N, 4) occupancy labels to dense labels.npz."
    )
    parser.add_argument(
        "--sparse-dir",
        default="data/GT_occupancy_mini/dense_voxels_with_semantic",
        help="Directory containing SurroundOcc sparse .npy files.",
    )
    parser.add_argument(
        "--ann-file",
        default="../FlashOCC/data/nuScenes/bevdetv2-nuscenes_infos_train.pkl",
        help="FlashOCC/BEVDet info pkl used to map lidar filename to scene/token.",
    )
    parser.add_argument(
        "--output-root",
        default="data/GT_occupancy_mini/stage1_occ3d_current",
        help="Output root. labels.npz files are written under output-root/gts/.",
    )
    parser.add_argument(
        "--occ-size",
        nargs=3,
        type=int,
        default=(200, 200, 16),
        metavar=("X", "Y", "Z"),
        help="Dense occupancy grid size.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("occupied", "all"),
        default="occupied",
        help=(
            "Temporary Stage 1 mask policy. 'occupied' marks only sparse occupied "
            "voxels as observed; 'all' marks every voxel as observed."
        ),
    )
    parser.add_argument(
        "--ann-output",
        default=None,
        help=(
            "Optional output pkl path. Defaults to "
            "output-root/bevdetv2-nuscenes_infos_stage1_train.pkl."
        ),
    )
    return parser.parse_args()


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data, data["infos"]
    if isinstance(data, list):
        return data, data
    raise ValueError(f"Unsupported annotation pkl format: {path}")


def basename_without_npy(path):
    name = Path(path).name
    if name.endswith(".npy"):
        return name[:-4]
    return name


def build_lidar_lookup(infos):
    lookup = {}
    for info in infos:
        lidar_path = info.get("lidar_path")
        if not lidar_path:
            continue
        lookup[Path(lidar_path).name] = info
    return lookup


def dense_from_sparse(sparse, occ_size):
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError(f"Expected sparse shape (N, 4), got {sparse.shape}")

    xyz = sparse[:, :3].astype(np.int64)
    labels = sparse[:, 3].astype(np.int64)
    occ_size_np = np.asarray(occ_size, dtype=np.int64)

    valid = (
        (xyz >= 0).all(axis=1)
        & (xyz < occ_size_np[None, :]).all(axis=1)
        & (labels >= 0)
        & (labels <= 16)
    )
    if not np.all(valid):
        raise ValueError(
            "Sparse file contains out-of-range voxel indices or labels; "
            f"invalid rows: {int((~valid).sum())}"
        )

    semantics = np.full(tuple(occ_size), FREE_LABEL, dtype=np.uint8)
    occupied_mask = np.zeros(tuple(occ_size), dtype=bool)

    if len(sparse) == 0:
        return semantics, occupied_mask, 0

    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    xyz_sorted = xyz[order]
    labels_sorted = labels[order]

    unique_xyz, starts, counts = np.unique(
        xyz_sorted, axis=0, return_index=True, return_counts=True
    )
    conflict_voxels = 0

    for coord, start, count in zip(unique_xyz, starts, counts):
        label_slice = labels_sorted[start : start + count]
        label_counts = Counter(int(v) for v in label_slice)
        if len(label_counts) > 1:
            conflict_voxels += 1
        # Deterministic tie-break: highest count, then smaller label id.
        label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        coord_tuple = tuple(int(v) for v in coord)
        semantics[coord_tuple] = label
        occupied_mask[coord_tuple] = True

    return semantics, occupied_mask, conflict_voxels


def save_ann_with_stage1_paths(ann_data, matched_infos, ann_output):
    if isinstance(ann_data, dict) and "infos" in ann_data:
        out_data = dict(ann_data)
        out_data["infos"] = matched_infos
    else:
        out_data = matched_infos

    with open(ann_output, "wb") as f:
        pickle.dump(out_data, f)


def main():
    args = parse_args()
    sparse_dir = Path(args.sparse_dir)
    output_root = Path(args.output_root)
    gts_root = output_root / "gts"
    ann_output = (
        Path(args.ann_output)
        if args.ann_output
        else output_root / "bevdetv2-nuscenes_infos_stage1_train.pkl"
    )
    stats_output = output_root / "stats_stage1_format.json"

    ann_data, infos = load_infos(args.ann_file)
    lidar_lookup = build_lidar_lookup(infos)
    sparse_files = sorted(sparse_dir.glob("*.npy"))

    output_root.mkdir(parents=True, exist_ok=True)
    gts_root.mkdir(parents=True, exist_ok=True)
    ann_output.parent.mkdir(parents=True, exist_ok=True)

    matched_infos = []
    missing_from_ann = []
    file_stats = []
    total_conflict_voxels = 0
    total_sparse_rows = 0
    total_occupied = 0

    for sparse_path in sparse_files:
        lidar_basename = basename_without_npy(sparse_path)
        info = lidar_lookup.get(lidar_basename)
        if info is None:
            missing_from_ann.append(sparse_path.name)
            continue

        sparse = np.load(sparse_path)
        semantics, occupied_mask, conflict_voxels = dense_from_sparse(sparse, args.occ_size)
        if args.mask_mode == "occupied":
            mask_lidar = occupied_mask.copy()
            mask_camera = occupied_mask.copy()
        else:
            mask_lidar = np.ones(tuple(args.occ_size), dtype=bool)
            mask_camera = np.ones(tuple(args.occ_size), dtype=bool)

        scene_name = info["scene_name"]
        token = info["token"]
        occ_dir = gts_root / scene_name / token
        occ_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            occ_dir / "labels.npz",
            semantics=semantics,
            mask_lidar=mask_lidar,
            mask_camera=mask_camera,
        )

        updated_info = dict(info)
        updated_info["occ_path"] = str(occ_dir.resolve())
        matched_infos.append(updated_info)

        occupied_count = int(occupied_mask.sum())
        total_sparse_rows += int(sparse.shape[0])
        total_occupied += occupied_count
        total_conflict_voxels += conflict_voxels
        file_stats.append(
            {
                "file": sparse_path.name,
                "scene_name": scene_name,
                "token": token,
                "sparse_rows": int(sparse.shape[0]),
                "occupied_voxels": occupied_count,
                "conflict_voxels": int(conflict_voxels),
                "occ_path": str(occ_dir.resolve()),
            }
        )

    save_ann_with_stage1_paths(ann_data, matched_infos, ann_output)

    stats = {
        "stage": "Stage 1: Occ3D file format export",
        "note": "Temporary masks only; no raw LiDAR ray casting has been applied.",
        "sparse_dir": str(sparse_dir),
        "ann_file": str(args.ann_file),
        "output_root": str(output_root),
        "gts_root": str(gts_root),
        "ann_output": str(ann_output),
        "occ_size": list(args.occ_size),
        "free_label": FREE_LABEL,
        "mask_mode": args.mask_mode,
        "num_sparse_files": len(sparse_files),
        "num_converted": len(matched_infos),
        "num_missing_from_ann": len(missing_from_ann),
        "missing_from_ann": missing_from_ann,
        "total_sparse_rows": total_sparse_rows,
        "total_occupied_voxels": total_occupied,
        "total_conflict_voxels": int(total_conflict_voxels),
        "files": file_stats,
    }
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Converted {len(matched_infos)} / {len(sparse_files)} files")
    print(f"Wrote GT root: {gts_root}")
    print(f"Wrote ann pkl: {ann_output}")
    print(f"Wrote stats: {stats_output}")
    if missing_from_ann:
        print(f"Missing from ann pkl: {len(missing_from_ann)}")


if __name__ == "__main__":
    main()
