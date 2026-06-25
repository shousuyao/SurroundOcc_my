#!/usr/bin/env python3
"""Export pred-vs-official Occ3D mask differences as colored PLY files."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


COLORS = {
    "tp": np.array([120, 120, 120], dtype=np.uint8),
    "fp": np.array([255, 60, 40], dtype=np.uint8),
    "fn": np.array([40, 210, 80], dtype=np.uint8),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export mask TP/FP/FN PLY diagnostics.")
    parser.add_argument("--pred-ann", required=True)
    parser.add_argument(
        "--official-root",
        default="/home/fjm/shousuyao/FlashOCC/data/nuscenes/gts",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--top-k", type=int, default=3, help="Export worst-IoU frames.")
    parser.add_argument("--frame-index", type=int, default=None, help="Also export this frame index.")
    parser.add_argument("--stride", type=int, default=2, help="Export every Nth voxel in each diff set.")
    parser.add_argument("--pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    return parser.parse_args()


def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def safe_div(numer, denom):
    return float(numer / denom) if denom else 0.0


def write_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def coords_to_points(coords, pc_range, voxel_size):
    pc_min = np.asarray(pc_range[:3], dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    return pc_min[None, :] + (coords.astype(np.float32) + 0.5) * voxel_size[None, :]


def summarize_mask(pred_mask, official_mask):
    tp = pred_mask & official_mask
    fp = pred_mask & ~official_mask
    fn = ~pred_mask & official_mask
    inter = int(tp.sum())
    union = int((pred_mask | official_mask).sum())
    pred_count = int(pred_mask.sum())
    official_count = int(official_mask.sum())
    return {
        "tp_count": inter,
        "fp_count": int(fp.sum()),
        "fn_count": int(fn.sum()),
        "pred_count": pred_count,
        "official_count": official_count,
        "union": union,
        "iou": safe_div(inter, union),
        "precision": safe_div(inter, pred_count),
        "recall": safe_div(inter, official_count),
    }


def export_frame(info, frame_idx, official_root, output_root, pc_range, voxel_size, stride):
    scene_name = info["scene_name"]
    token = info["token"]
    pred_path = Path(info["occ_path"]) / "labels.npz"
    official_path = official_root / scene_name / token / "labels.npz"
    pred = np.load(pred_path)
    official = np.load(official_path)
    pred_mask = pred["mask_lidar"].astype(bool)
    official_mask = official["mask_lidar"].astype(bool)

    masks = {
        "tp": pred_mask & official_mask,
        "fp": pred_mask & ~official_mask,
        "fn": ~pred_mask & official_mask,
    }
    frame_dir = output_root / f"{frame_idx:03d}_{scene_name}_{token}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    combined_points = []
    combined_colors = []
    export_stats = {}
    for name, mask in masks.items():
        coords = np.argwhere(mask)
        raw_count = int(coords.shape[0])
        if stride > 1:
            coords = coords[::stride]
        points = coords_to_points(coords, pc_range, voxel_size) if coords.size else np.zeros((0, 3), dtype=np.float32)
        colors = np.repeat(COLORS[name][None, :], points.shape[0], axis=0)
        write_ply(frame_dir / f"{name}.ply", points, colors)
        combined_points.append(points)
        combined_colors.append(colors)
        export_stats[f"{name}_raw_count"] = raw_count
        export_stats[f"{name}_exported_count"] = int(points.shape[0])

    if combined_points:
        points = np.concatenate(combined_points, axis=0)
        colors = np.concatenate(combined_colors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    write_ply(frame_dir / "tp_fp_fn_combined.ply", points, colors)
    export_stats.update(
        {
            "frame_idx": frame_idx,
            "scene_name": scene_name,
            "token": token,
            "pred_path": str(pred_path),
            "official_path": str(official_path),
            "frame_dir": str(frame_dir),
        }
    )
    export_stats.update(summarize_mask(pred_mask, official_mask))
    return export_stats


def main():
    args = parse_args()
    infos = load_infos(args.pred_ann)
    official_root = Path(args.official_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_summaries = []
    for idx, info in enumerate(infos):
        official_path = official_root / info["scene_name"] / info["token"] / "labels.npz"
        pred_path = Path(info["occ_path"]) / "labels.npz"
        if not official_path.exists() or not pred_path.exists():
            continue
        pred = np.load(pred_path)
        official = np.load(official_path)
        summary = summarize_mask(pred["mask_lidar"].astype(bool), official["mask_lidar"].astype(bool))
        summary.update({"frame_idx": idx, "scene_name": info["scene_name"], "token": info["token"]})
        frame_summaries.append(summary)

    selected = []
    for item in sorted(frame_summaries, key=lambda x: x["iou"])[: max(args.top_k, 0)]:
        selected.append(item["frame_idx"])
    if args.frame_index is not None:
        selected.append(args.frame_index)
    selected = sorted(set(i for i in selected if 0 <= i < len(infos)))

    exported = [
        export_frame(infos[idx], idx, official_root, output_root, args.pc_range, args.voxel_size, args.stride)
        for idx in selected
    ]
    totals = {
        "frames": len(frame_summaries),
        "tp_count": int(sum(item["tp_count"] for item in frame_summaries)),
        "fp_count": int(sum(item["fp_count"] for item in frame_summaries)),
        "fn_count": int(sum(item["fn_count"] for item in frame_summaries)),
        "pred_count": int(sum(item["pred_count"] for item in frame_summaries)),
        "official_count": int(sum(item["official_count"] for item in frame_summaries)),
        "union": int(sum(item["union"] for item in frame_summaries)),
    }
    totals["iou"] = safe_div(totals["tp_count"], totals["union"])
    totals["precision"] = safe_div(totals["tp_count"], totals["pred_count"])
    totals["recall"] = safe_div(totals["tp_count"], totals["official_count"])

    report = {
        "pred_ann": args.pred_ann,
        "official_root": str(official_root),
        "output_root": str(output_root),
        "stride": int(args.stride),
        "color_legend": {
            "tp": "gray, predicted and official observed",
            "fp": "red, predicted observed only",
            "fn": "green, official observed only",
        },
        "totals": totals,
        "selected_frame_indices": selected,
        "exported": exported,
        "frames": frame_summaries,
    }
    report_path = output_root / "mask_diff_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"totals": totals, "selected_frame_indices": selected}, indent=2))
    print(f"Wrote mask diff report: {report_path}")


if __name__ == "__main__":
    main()
