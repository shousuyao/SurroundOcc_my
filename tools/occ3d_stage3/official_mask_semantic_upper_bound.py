#!/usr/bin/env python3
"""Evaluate semantic candidates under the official Occ3D lidar mask.

This is an upper-bound style diagnostic: mask_lidar is copied from the official
Occ3D GT, while occupied semantics come from a candidate sparse occupancy file.
Voxels without a candidate are exported as free class 17.
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


FREE_LABEL = 17


def parse_args():
    parser = argparse.ArgumentParser(description="Official-mask semantic upper bound.")
    parser.add_argument("--ann-file", required=True, help="Annotation pkl used for frame order and lidar_path.")
    parser.add_argument(
        "--official-root",
        default="/home/fjm/shousuyao/FlashOCC/data/nuscenes/gts",
        help="Official Occ3D gts root.",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Directory with candidate .npy files named by lidar filename.",
    )
    parser.add_argument(
        "--candidate-grid",
        choices=("occ3d",),
        default="occ3d",
        help="Currently only direct Occ3D-grid candidate indices are supported.",
    )
    parser.add_argument(
        "--dense-coordinate-transform",
        choices=("identity", "swapxy_flipy"),
        default="identity",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    parser.add_argument("--num-classes", type=int, default=18)
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


def lidar_basename(info):
    return Path(info["lidar_path"]).name


def candidate_path_for_info(candidate_dir, info):
    return Path(candidate_dir) / f"{lidar_basename(info)}.npy"


def transform_dense_coords(coords, occ_size, transform_name):
    if transform_name == "identity":
        return coords
    if transform_name == "swapxy_flipy":
        transformed = coords.copy()
        transformed[:, 0] = coords[:, 1]
        transformed[:, 1] = occ_size[1] - 1 - coords[:, 0]
        return transformed
    raise ValueError(f"Unsupported dense coordinate transform: {transform_name}")


def fast_hist(pred, gt, mask, num_classes):
    pred = pred[mask].astype(np.int64)
    gt = gt[mask].astype(np.int64)
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = num_classes * gt[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def miou_from_hist(hist):
    intersection = np.diag(hist).astype(np.float64)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    valid = union > 0
    ious = np.full(hist.shape[0], np.nan, dtype=np.float64)
    ious[valid] = intersection[valid] / union[valid]
    return float(np.nanmean(ious)) if np.any(valid) else float("nan"), ious


def safe_div(numer, denom):
    return float(numer / denom) if denom else 0.0


def build_candidate_semantics(candidate_path, occ_size, transform_name):
    sparse = np.load(candidate_path)
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError(f"Expected candidate shape (N, 4), got {sparse.shape}: {candidate_path}")

    coords = sparse[:, :3].astype(np.int64)
    labels = sparse[:, 3].astype(np.int64)
    occ_size_np = np.asarray(occ_size, dtype=np.int64)
    coords = transform_dense_coords(coords, occ_size_np, transform_name)

    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < occ_size_np[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < occ_size_np[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < occ_size_np[2])
        & (labels >= 0)
        & (labels <= 16)
    )
    coords = coords[valid]
    labels = labels[valid]

    semantics = np.full(tuple(occ_size), FREE_LABEL, dtype=np.uint8)
    candidate_occ = np.zeros(tuple(occ_size), dtype=bool)
    conflicts = 0
    if labels.size == 0:
        return semantics, candidate_occ, 0, int((~valid).sum())

    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    coords_sorted = coords[order]
    labels_sorted = labels[order]
    unique_xyz, starts, counts = np.unique(
        coords_sorted, axis=0, return_index=True, return_counts=True
    )

    for coord, start, count in zip(unique_xyz, starts, counts):
        label_slice = labels_sorted[start : start + count]
        counter = Counter(int(v) for v in label_slice)
        if len(counter) > 1:
            conflicts += 1
        label = counter.most_common(1)[0][0]
        coord_tuple = tuple(int(v) for v in coord)
        semantics[coord_tuple] = np.uint8(label)
        candidate_occ[coord_tuple] = True

    return semantics, candidate_occ, int(conflicts), int((~valid).sum())


def main():
    args = parse_args()
    ann_data, infos = load_infos(args.ann_file)
    official_root = Path(args.official_root)
    candidate_dir = Path(args.candidate_dir)
    output_root = Path(args.output_root)
    gts_root = output_root / "gts"
    ann_output = output_root / "bevdetv2-nuscenes_infos_official_mask_semantic_upper_bound.pkl"
    report_output = output_root / "official_mask_semantic_upper_bound.json"

    gts_root.mkdir(parents=True, exist_ok=True)
    updated_infos = []
    file_stats = []

    hist_lidar = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    totals = {
        "frames": 0,
        "missing_official": 0,
        "missing_candidate": 0,
        "official_lidar_count": 0,
        "official_occupied_count": 0,
        "candidate_occupied_count": 0,
        "candidate_occupied_in_official_lidar": 0,
        "candidate_official_occupied_intersection": 0,
        "candidate_official_occupied_union": 0,
        "candidate_conflict_voxels": 0,
        "candidate_out_of_range_rows": 0,
    }

    for frame_idx, info in enumerate(infos):
        scene_name = info["scene_name"]
        token = info["token"]
        official_path = official_root / scene_name / token / "labels.npz"
        candidate_path = candidate_path_for_info(candidate_dir, info)
        if not official_path.exists():
            totals["missing_official"] += 1
            continue
        if not candidate_path.exists():
            totals["missing_candidate"] += 1
            continue

        official = np.load(official_path)
        official_sem = official["semantics"]
        official_lidar = official["mask_lidar"].astype(bool)
        official_camera = official["mask_camera"].astype(bool)
        official_occ = official_sem != FREE_LABEL

        candidate_sem, candidate_occ, conflicts, out_of_range = build_candidate_semantics(
            candidate_path, tuple(args.occ_size), args.dense_coordinate_transform
        )

        exported_sem = candidate_sem.copy()
        exported_mask_lidar = official_lidar.copy()
        exported_mask_camera = official_camera.copy()

        occ_dir = gts_root / scene_name / token
        occ_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            occ_dir / "labels.npz",
            semantics=exported_sem,
            mask_lidar=exported_mask_lidar,
            mask_camera=exported_mask_camera,
        )

        hist_lidar += fast_hist(exported_sem, official_sem, official_lidar, args.num_classes)

        candidate_in_lidar = candidate_occ & official_lidar
        occ_intersection = int((candidate_occ & official_occ).sum())
        occ_union = int((candidate_occ | official_occ).sum())

        totals["frames"] += 1
        totals["official_lidar_count"] += int(official_lidar.sum())
        totals["official_occupied_count"] += int(official_occ.sum())
        totals["candidate_occupied_count"] += int(candidate_occ.sum())
        totals["candidate_occupied_in_official_lidar"] += int(candidate_in_lidar.sum())
        totals["candidate_official_occupied_intersection"] += occ_intersection
        totals["candidate_official_occupied_union"] += occ_union
        totals["candidate_conflict_voxels"] += int(conflicts)
        totals["candidate_out_of_range_rows"] += int(out_of_range)

        updated_info = dict(info)
        updated_info["occ_path"] = str(occ_dir.resolve())
        updated_infos.append(updated_info)

        frame_hist = fast_hist(exported_sem, official_sem, official_lidar, args.num_classes)
        frame_miou, _ = miou_from_hist(frame_hist)
        file_stats.append(
            {
                "frame_idx": frame_idx,
                "scene_name": scene_name,
                "token": token,
                "candidate_path": str(candidate_path),
                "occ_path": str(occ_dir.resolve()),
                "official_lidar_count": int(official_lidar.sum()),
                "official_occupied_count": int(official_occ.sum()),
                "candidate_occupied_count": int(candidate_occ.sum()),
                "candidate_occupied_in_official_lidar": int(candidate_in_lidar.sum()),
                "candidate_official_occupied_iou": safe_div(occ_intersection, occ_union),
                "semantic_miou_official_lidar_mask": frame_miou,
                "candidate_conflict_voxels": int(conflicts),
                "candidate_out_of_range_rows": int(out_of_range),
            }
        )

    save_ann_with_paths(ann_data, updated_infos, ann_output)
    miou, ious = miou_from_hist(hist_lidar)
    report = {
        "ann_file": args.ann_file,
        "official_root": str(official_root),
        "candidate_dir": str(candidate_dir),
        "candidate_grid": args.candidate_grid,
        "dense_coordinate_transform": args.dense_coordinate_transform,
        "ann_output": str(ann_output),
        "gts_root": str(gts_root),
        "totals": totals,
        "aggregate": {
            "mask_lidar_iou": 1.0,
            "mask_lidar_precision": 1.0,
            "mask_lidar_recall": 1.0,
            "candidate_official_occupied_iou_full_grid": safe_div(
                totals["candidate_official_occupied_intersection"],
                totals["candidate_official_occupied_union"],
            ),
            "candidate_occupied_official_lidar_fraction": safe_div(
                totals["candidate_occupied_in_official_lidar"],
                totals["candidate_occupied_count"],
            ),
            "semantic_miou_official_lidar_mask": miou,
            "class_iou_official_lidar_mask": [None if np.isnan(v) else float(v) for v in ious],
        },
        "files": file_stats,
    }
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Wrote upper-bound GT root: {gts_root}")
    print(f"Wrote upper-bound ann pkl: {ann_output}")
    print(f"Wrote report: {report_output}")


if __name__ == "__main__":
    main()
