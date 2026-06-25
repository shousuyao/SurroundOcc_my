#!/usr/bin/env python3
"""Compare generated GT-like labels.npz files against official Occ3D GT."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compare generated occupancy GT with Occ3D GT.")
    parser.add_argument("--pred-ann", required=True, help="Annotation pkl whose occ_path points to generated GT.")
    parser.add_argument(
        "--official-root",
        default="/home/fjm/shousuyao/FlashOCC/data/nuscenes/gts",
        help="Official Occ3D gts root.",
    )
    parser.add_argument("--output", required=True, help="Output JSON report.")
    parser.add_argument("--num-classes", type=int, default=18)
    return parser.parse_args()


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


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


def summarize_binary(pred_mask, gt_mask):
    inter = int((pred_mask & gt_mask).sum())
    union = int((pred_mask | gt_mask).sum())
    pred_count = int(pred_mask.sum())
    gt_count = int(gt_mask.sum())
    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
        "intersection": inter,
        "union": union,
        "iou": safe_div(inter, union),
        "precision": safe_div(inter, pred_count),
        "recall": safe_div(inter, gt_count),
    }


def main():
    args = parse_args()
    infos = load_infos(args.pred_ann)
    official_root = Path(args.official_root)
    num_classes = args.num_classes

    hist_ref_lidar = np.zeros((num_classes, num_classes), dtype=np.int64)
    hist_common_lidar = np.zeros((num_classes, num_classes), dtype=np.int64)
    hist_ref_camera = np.zeros((num_classes, num_classes), dtype=np.int64)

    totals = {
        "frames": 0,
        "missing_official": 0,
        "mask_lidar_intersection": 0,
        "mask_lidar_union": 0,
        "pred_lidar_count": 0,
        "official_lidar_count": 0,
        "occupied_intersection": 0,
        "occupied_union": 0,
        "pred_occupied_count": 0,
        "official_occupied_count": 0,
    }
    files = []

    for info in infos:
        scene_name = info["scene_name"]
        token = info["token"]
        pred_path = Path(info["occ_path"]) / "labels.npz"
        official_path = official_root / scene_name / token / "labels.npz"
        if not official_path.exists():
            totals["missing_official"] += 1
            continue

        pred = np.load(pred_path)
        gt = np.load(official_path)
        pred_sem = pred["semantics"]
        gt_sem = gt["semantics"]
        pred_lidar = pred["mask_lidar"].astype(bool)
        gt_lidar = gt["mask_lidar"].astype(bool)
        gt_camera = gt["mask_camera"].astype(bool)

        pred_occ = pred_sem != 17
        gt_occ = gt_sem != 17
        common_lidar = pred_lidar & gt_lidar

        hist_ref_lidar += fast_hist(pred_sem, gt_sem, gt_lidar, num_classes)
        hist_common_lidar += fast_hist(pred_sem, gt_sem, common_lidar, num_classes)
        hist_ref_camera += fast_hist(pred_sem, gt_sem, gt_camera, num_classes)

        mask_summary = summarize_binary(pred_lidar, gt_lidar)
        occ_summary = summarize_binary(pred_occ, gt_occ)

        totals["frames"] += 1
        totals["mask_lidar_intersection"] += mask_summary["intersection"]
        totals["mask_lidar_union"] += mask_summary["union"]
        totals["pred_lidar_count"] += mask_summary["pred_count"]
        totals["official_lidar_count"] += mask_summary["gt_count"]
        totals["occupied_intersection"] += occ_summary["intersection"]
        totals["occupied_union"] += occ_summary["union"]
        totals["pred_occupied_count"] += occ_summary["pred_count"]
        totals["official_occupied_count"] += occ_summary["gt_count"]

        frame_hist_common = fast_hist(pred_sem, gt_sem, common_lidar, num_classes)
        frame_miou_common, _ = miou_from_hist(frame_hist_common)
        files.append(
            {
                "scene_name": scene_name,
                "token": token,
                "mask_lidar_iou": mask_summary["iou"],
                "mask_lidar_precision": mask_summary["precision"],
                "mask_lidar_recall": mask_summary["recall"],
                "occupied_iou_full_grid": occ_summary["iou"],
                "semantic_miou_common_lidar_mask": frame_miou_common,
                "pred_lidar_count": mask_summary["pred_count"],
                "official_lidar_count": mask_summary["gt_count"],
                "pred_occupied_count": occ_summary["pred_count"],
                "official_occupied_count": occ_summary["gt_count"],
            }
        )

    miou_ref_lidar, ious_ref_lidar = miou_from_hist(hist_ref_lidar)
    miou_common_lidar, ious_common_lidar = miou_from_hist(hist_common_lidar)
    miou_ref_camera, ious_ref_camera = miou_from_hist(hist_ref_camera)

    summary = {
        "pred_ann": args.pred_ann,
        "official_root": str(official_root),
        "num_classes": num_classes,
        "totals": totals,
        "aggregate": {
            "mask_lidar_iou": safe_div(totals["mask_lidar_intersection"], totals["mask_lidar_union"]),
            "mask_lidar_precision": safe_div(totals["mask_lidar_intersection"], totals["pred_lidar_count"]),
            "mask_lidar_recall": safe_div(totals["mask_lidar_intersection"], totals["official_lidar_count"]),
            "occupied_iou_full_grid": safe_div(totals["occupied_intersection"], totals["occupied_union"]),
            "semantic_miou_ref_lidar_mask": miou_ref_lidar,
            "semantic_miou_common_lidar_mask": miou_common_lidar,
            "semantic_miou_ref_camera_mask": miou_ref_camera,
            "class_iou_ref_lidar_mask": [None if np.isnan(v) else float(v) for v in ious_ref_lidar],
            "class_iou_common_lidar_mask": [None if np.isnan(v) else float(v) for v in ious_common_lidar],
            "class_iou_ref_camera_mask": [None if np.isnan(v) else float(v) for v in ious_ref_camera],
        },
        "files": files,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
