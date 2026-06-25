#!/usr/bin/env python3
"""Summarize pred-vs-official mask differences by z layer."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Occ3D mask diff by z layer.")
    parser.add_argument("--pred-ann", required=True)
    parser.add_argument(
        "--official-root",
        default="/home/fjm/shousuyao/FlashOCC/data/nuscenes/gts",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    return parser.parse_args()


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def safe_div(numer, denom):
    return float(numer / denom) if denom else 0.0


def main():
    args = parse_args()
    infos = load_infos(args.pred_ann)
    official_root = Path(args.official_root)
    z_size = int(args.occ_size[2])

    tp_z = np.zeros(z_size, dtype=np.int64)
    fp_z = np.zeros(z_size, dtype=np.int64)
    fn_z = np.zeros(z_size, dtype=np.int64)
    pred_z = np.zeros(z_size, dtype=np.int64)
    official_z = np.zeros(z_size, dtype=np.int64)
    files = []

    missing = 0
    for idx, info in enumerate(infos):
        pred_path = Path(info["occ_path"]) / "labels.npz"
        official_path = official_root / info["scene_name"] / info["token"] / "labels.npz"
        if not pred_path.exists() or not official_path.exists():
            missing += 1
            continue

        pred_mask = np.load(pred_path)["mask_lidar"].astype(bool)
        official_mask = np.load(official_path)["mask_lidar"].astype(bool)
        tp = pred_mask & official_mask
        fp = pred_mask & ~official_mask
        fn = ~pred_mask & official_mask

        frame_tp = int(tp.sum())
        frame_fp = int(fp.sum())
        frame_fn = int(fn.sum())
        frame_pred = int(pred_mask.sum())
        frame_official = int(official_mask.sum())
        frame_union = frame_tp + frame_fp + frame_fn
        files.append(
            {
                "frame_idx": idx,
                "scene_name": info["scene_name"],
                "token": info["token"],
                "tp": frame_tp,
                "fp": frame_fp,
                "fn": frame_fn,
                "pred": frame_pred,
                "official": frame_official,
                "iou": safe_div(frame_tp, frame_union),
                "precision": safe_div(frame_tp, frame_pred),
                "recall": safe_div(frame_tp, frame_official),
            }
        )

        for z in range(z_size):
            tp_z[z] += int(tp[:, :, z].sum())
            fp_z[z] += int(fp[:, :, z].sum())
            fn_z[z] += int(fn[:, :, z].sum())
            pred_z[z] += int(pred_mask[:, :, z].sum())
            official_z[z] += int(official_mask[:, :, z].sum())

    total_tp = int(tp_z.sum())
    total_fp = int(fp_z.sum())
    total_fn = int(fn_z.sum())
    total_pred = int(pred_z.sum())
    total_official = int(official_z.sum())
    total_union = total_tp + total_fp + total_fn

    z_layers = []
    for z in range(z_size):
        union = int(tp_z[z] + fp_z[z] + fn_z[z])
        z_layers.append(
            {
                "z": z,
                "tp": int(tp_z[z]),
                "fp": int(fp_z[z]),
                "fn": int(fn_z[z]),
                "pred": int(pred_z[z]),
                "official": int(official_z[z]),
                "iou": safe_div(int(tp_z[z]), union),
                "precision": safe_div(int(tp_z[z]), int(pred_z[z])),
                "recall": safe_div(int(tp_z[z]), int(official_z[z])),
            }
        )

    report = {
        "pred_ann": args.pred_ann,
        "official_root": str(official_root),
        "missing": int(missing),
        "aggregate": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "pred": total_pred,
            "official": total_official,
            "union": total_union,
            "iou": safe_div(total_tp, total_union),
            "precision": safe_div(total_tp, total_pred),
            "recall": safe_div(total_tp, total_official),
        },
        "z_layers": z_layers,
        "top_fp_z_layers": sorted(z_layers, key=lambda item: item["fp"], reverse=True)[:5],
        "top_fn_z_layers": sorted(z_layers, key=lambda item: item["fn"], reverse=True)[:5],
        "files": files,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["aggregate"], indent=2))
    print("top_fp_z_layers", [(item["z"], item["fp"]) for item in report["top_fp_z_layers"]])
    print("top_fn_z_layers", [(item["z"], item["fn"]) for item in report["top_fn_z_layers"]])


if __name__ == "__main__":
    main()
