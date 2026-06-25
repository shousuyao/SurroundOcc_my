#!/usr/bin/env python3
"""Evaluate one fixed prediction set under multiple fixed masks.

This is useful for separating model prediction quality from changes in the
evaluation mask. It reads one FlashOcc prediction pkl and computes semantic
mIoU under:

- official Occ3D mask_camera;
- generated Stage 4 mask_camera;
- placeholder mask_lidar from the input annotation.
"""

import argparse
import json
import pickle
from pathlib import Path

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fixed predictions under multiple masks.")
    parser.add_argument(
        "--pred-pkl",
        default="data/GT_occupancy_mini/flashocc_train_smoke_stage4_camera_z1_4/preds_epoch1.pkl",
    )
    parser.add_argument(
        "--stage4-ann",
        default="data/GT_occupancy_mini/stage4_camera_raymask_z1_4/bevdetv2-nuscenes_infos_stage4_train.pkl",
    )
    parser.add_argument(
        "--placeholder-ann",
        default=(
            "data/GT_occupancy_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy/"
            "bevdetv2-nuscenes_infos_stage3_train.pkl"
        ),
        help="Annotation whose mask_lidar is used as the placeholder evaluation mask.",
    )
    parser.add_argument(
        "--official-root",
        default="/home/fjm/shousuyao/FlashOCC/data/nuscenes/gts",
    )
    parser.add_argument(
        "--output",
        default="data/GT_occupancy_mini/flashocc_train_smoke_stage4_camera_z1_4/fixed_mask_eval_epoch1.json",
    )
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument(
        "--exclude-free-from-miou",
        action="store_true",
        default=True,
        help="Match FlashOcc mIoU reporting by averaging classes 0-16 and excluding free label 17.",
    )
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


def per_class_iu(hist):
    intersection = np.diag(hist).astype(np.float64)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    ious = np.full(hist.shape[0], np.nan, dtype=np.float64)
    valid = union > 0
    ious[valid] = intersection[valid] / union[valid]
    return ious


def summarize(hist, valid_count, exclude_free):
    ious = per_class_iu(hist)
    miou_values = ious[:-1] if exclude_free else ious
    return {
        "valid_voxels": int(valid_count),
        "miou": float(np.nanmean(miou_values) * 100.0) if np.any(~np.isnan(miou_values)) else float("nan"),
        "per_class_iou": [None if np.isnan(v) else float(v * 100.0) for v in ious],
    }


def eval_one_mask(preds, infos, mask_loader, num_classes, exclude_free):
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid_total = 0
    files = []
    for idx, (pred, info) in enumerate(zip(preds, infos)):
        gt_sem, mask, mask_path = mask_loader(info)
        frame_hist = fast_hist(pred, gt_sem, mask, num_classes)
        frame_valid = int(mask.sum())
        hist += frame_hist
        valid_total += frame_valid
        frame_summary = summarize(frame_hist, frame_valid, exclude_free)
        files.append(
            {
                "frame_idx": idx,
                "scene_name": info["scene_name"],
                "token": info["token"],
                "mask_path": str(mask_path),
                "valid_voxels": frame_valid,
                "miou": frame_summary["miou"],
            }
        )
    summary = summarize(hist, valid_total, exclude_free)
    summary["files"] = files
    return summary


def main():
    args = parse_args()
    preds = mmcv.load(args.pred_pkl)
    stage4_infos = load_infos(args.stage4_ann)
    placeholder_infos = load_infos(args.placeholder_ann)
    placeholder_by_token = {info["token"]: info for info in placeholder_infos}

    if len(preds) != len(stage4_infos):
        raise ValueError(f"Prediction count {len(preds)} != info count {len(stage4_infos)}")

    official_root = Path(args.official_root)

    def load_official_camera(info):
        path = official_root / info["scene_name"] / info["token"] / "labels.npz"
        labels = np.load(path)
        return labels["semantics"], labels["mask_camera"].astype(bool), path

    def load_stage4_camera(info):
        path = Path(info["occ_path"]) / "labels.npz"
        labels = np.load(path)
        return labels["semantics"], labels["mask_camera"].astype(bool), path

    def load_placeholder_lidar(info):
        placeholder_info = placeholder_by_token[info["token"]]
        path = Path(placeholder_info["occ_path"]) / "labels.npz"
        labels = np.load(path)
        return labels["semantics"], labels["mask_lidar"].astype(bool), path

    results = {
        "pred_pkl": args.pred_pkl,
        "stage4_ann": args.stage4_ann,
        "placeholder_ann": args.placeholder_ann,
        "official_root": args.official_root,
        "num_frames": len(preds),
        "num_classes": args.num_classes,
        "miou_excludes_free_label_17": bool(args.exclude_free_from_miou),
        "masks": {
            "official_mask_camera": eval_one_mask(
                preds, stage4_infos, load_official_camera, args.num_classes, args.exclude_free_from_miou
            ),
            "stage4_mask_camera": eval_one_mask(
                preds, stage4_infos, load_stage4_camera, args.num_classes, args.exclude_free_from_miou
            ),
            "placeholder_mask_lidar": eval_one_mask(
                preds, stage4_infos, load_placeholder_lidar, args.num_classes, args.exclude_free_from_miou
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in results["masks"].items()}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
