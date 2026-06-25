#!/usr/bin/env python3
"""Build Occ3D-style camera visibility masks from image pixel rays.

Stage 4 keeps Stage 3 semantics and mask_lidar unchanged. It replaces the
placeholder mask_camera with a ray-cast visibility mask:

- rays start from camera pixels, not from voxel projection;
- camera rays only inherit LiDAR-observed FREE/OCCUPIED states;
- LiDAR-unknown voxels remain camera NOT_OBSERVED;
- the first occupied voxel stops the current ray.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

try:
    from numba import njit
except Exception as exc:  # pragma: no cover - exercised only outside flashocc env.
    raise RuntimeError(
        "Stage 4 needs numba for pixel-ray traversal. Run with the flashocc "
        "environment, e.g. /home/fjm/miniconda3/envs/flashocc/bin/python."
    ) from exc


FREE_LABEL = 17
EPS = 1.0e-6
INF = 1.0e18


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 camera ray visibility mask.")
    parser.add_argument(
        "--input-ann",
        default=(
            "data/GT_occupancy_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy/"
            "bevdetv2-nuscenes_infos_stage3_train.pkl"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/GT_occupancy_mini/stage4_camera_raymask_z1_4",
    )
    parser.add_argument(
        "--official-root",
        default="/home/fjm/shousuyao/FlashOCC/data/nuscenes/gts",
        help="Official Occ3D GT root. If unavailable, official comparison is skipped.",
    )
    parser.add_argument("--depth-max", type=float, default=45.0)
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=(1600, 900),
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=1,
        help="Use every Nth pixel in u/v. Official Stage 4 runs should use 1.",
    )
    parser.add_argument("--occ-pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--occ-voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=18,
        help="Number of semantic classes including free label 17.",
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


def save_ann_with_paths(ann_data, infos, output_path):
    if isinstance(ann_data, dict) and "infos" in ann_data:
        out = dict(ann_data)
        out["infos"] = infos
    else:
        out = infos
    with open(output_path, "wb") as f:
        pickle.dump(out, f)


@njit(cache=True)
def _clip_segment_to_box(origin, endpoint, box_min, box_max):
    t0 = 0.0
    t1 = 1.0
    dx = endpoint[0] - origin[0]
    dy = endpoint[1] - origin[1]
    dz = endpoint[2] - origin[2]
    for axis in range(3):
        if axis == 0:
            o = origin[0]
            d = dx
        elif axis == 1:
            o = origin[1]
            d = dy
        else:
            o = origin[2]
            d = dz

        if abs(d) < EPS:
            if o < box_min[axis] or o > box_max[axis]:
                return False, 0.0, 0.0
            continue

        inv_d = 1.0 / d
        near_t = (box_min[axis] - o) * inv_d
        far_t = (box_max[axis] - o) * inv_d
        if near_t > far_t:
            tmp = near_t
            near_t = far_t
            far_t = tmp
        if near_t > t0:
            t0 = near_t
        if far_t < t1:
            t1 = far_t
        if t0 > t1:
            return False, 0.0, 0.0
    return True, t0, t1


@njit(cache=True)
def _mark_ray(origin, endpoint, semantics, mask_lidar, mask_camera, pc_min, pc_max, voxel_size, occ_size):
    hit_box, t_enter, t_exit = _clip_segment_to_box(origin, endpoint, pc_min, pc_max)
    if not hit_box:
        return 0, 0, 0

    if t_exit <= 0.0 or t_enter >= 1.0:
        return 0, 0, 0
    if t_enter < 0.0:
        t_enter = 0.0
    if t_exit > 1.0:
        t_exit = 1.0

    dx = endpoint[0] - origin[0]
    dy = endpoint[1] - origin[1]
    dz = endpoint[2] - origin[2]
    if dx * dx + dy * dy + dz * dz < EPS:
        return 0, 0, 0

    sx = origin[0] + t_enter * dx
    sy = origin[1] + t_enter * dy
    sz = origin[2] + t_enter * dz

    ix = int(np.floor((sx - pc_min[0]) / voxel_size[0]))
    iy = int(np.floor((sy - pc_min[1]) / voxel_size[1]))
    iz = int(np.floor((sz - pc_min[2]) / voxel_size[2]))
    if ix < 0:
        ix = 0
    elif ix >= occ_size[0]:
        ix = occ_size[0] - 1
    if iy < 0:
        iy = 0
    elif iy >= occ_size[1]:
        iy = occ_size[1] - 1
    if iz < 0:
        iz = 0
    elif iz >= occ_size[2]:
        iz = occ_size[2] - 1

    if dx > 0.0:
        step_x = 1
        next_x = pc_min[0] + (ix + 1) * voxel_size[0]
        t_max_x = (next_x - origin[0]) / dx
        t_delta_x = voxel_size[0] / dx
    elif dx < 0.0:
        step_x = -1
        next_x = pc_min[0] + ix * voxel_size[0]
        t_max_x = (next_x - origin[0]) / dx
        t_delta_x = -voxel_size[0] / dx
    else:
        step_x = 0
        t_max_x = INF
        t_delta_x = INF

    if dy > 0.0:
        step_y = 1
        next_y = pc_min[1] + (iy + 1) * voxel_size[1]
        t_max_y = (next_y - origin[1]) / dy
        t_delta_y = voxel_size[1] / dy
    elif dy < 0.0:
        step_y = -1
        next_y = pc_min[1] + iy * voxel_size[1]
        t_max_y = (next_y - origin[1]) / dy
        t_delta_y = -voxel_size[1] / dy
    else:
        step_y = 0
        t_max_y = INF
        t_delta_y = INF

    if dz > 0.0:
        step_z = 1
        next_z = pc_min[2] + (iz + 1) * voxel_size[2]
        t_max_z = (next_z - origin[2]) / dz
        t_delta_z = voxel_size[2] / dz
    elif dz < 0.0:
        step_z = -1
        next_z = pc_min[2] + iz * voxel_size[2]
        t_max_z = (next_z - origin[2]) / dz
        t_delta_z = -voxel_size[2] / dz
    else:
        step_z = 0
        t_max_z = INF
        t_delta_z = INF

    traversed = 0
    inherited = 0
    first_occ = 0
    while ix >= 0 and ix < occ_size[0] and iy >= 0 and iy < occ_size[1] and iz >= 0 and iz < occ_size[2]:
        traversed += 1
        if mask_lidar[ix, iy, iz]:
            if not mask_camera[ix, iy, iz]:
                mask_camera[ix, iy, iz] = True
                inherited += 1
            if semantics[ix, iy, iz] != FREE_LABEL:
                first_occ = 1
                break

        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            if t_max_x > t_exit:
                break
            ix += step_x
            t_max_x += t_delta_x
        elif t_max_y <= t_max_z:
            if t_max_y > t_exit:
                break
            iy += step_y
            t_max_y += t_delta_y
        else:
            if t_max_z > t_exit:
                break
            iz += step_z
            t_max_z += t_delta_z

    return traversed, inherited, first_occ


@njit(cache=True)
def _cast_camera_rays(width, height, stride, depth_max, intrinsic, sensor2lidar_rot, sensor2lidar_tran,
                      semantics, mask_lidar, mask_camera, pc_min, pc_max, voxel_size, occ_size):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    origin = np.empty(3, dtype=np.float64)
    endpoint = np.empty(3, dtype=np.float64)
    cam_endpoint = np.empty(3, dtype=np.float64)

    origin[0] = sensor2lidar_tran[0]
    origin[1] = sensor2lidar_tran[1]
    origin[2] = sensor2lidar_tran[2]

    rays = 0
    traversed = 0
    inherited = 0
    first_occ = 0

    for v in range(0, height, stride):
        for u in range(0, width, stride):
            cam_endpoint[0] = ((u + 0.5) - cx) / fx * depth_max
            cam_endpoint[1] = ((v + 0.5) - cy) / fy * depth_max
            cam_endpoint[2] = depth_max

            endpoint[0] = (
                sensor2lidar_rot[0, 0] * cam_endpoint[0]
                + sensor2lidar_rot[0, 1] * cam_endpoint[1]
                + sensor2lidar_rot[0, 2] * cam_endpoint[2]
                + sensor2lidar_tran[0]
            )
            endpoint[1] = (
                sensor2lidar_rot[1, 0] * cam_endpoint[0]
                + sensor2lidar_rot[1, 1] * cam_endpoint[1]
                + sensor2lidar_rot[1, 2] * cam_endpoint[2]
                + sensor2lidar_tran[1]
            )
            endpoint[2] = (
                sensor2lidar_rot[2, 0] * cam_endpoint[0]
                + sensor2lidar_rot[2, 1] * cam_endpoint[1]
                + sensor2lidar_rot[2, 2] * cam_endpoint[2]
                + sensor2lidar_tran[2]
            )

            t, h, o = _mark_ray(
                origin,
                endpoint,
                semantics,
                mask_lidar,
                mask_camera,
                pc_min,
                pc_max,
                voxel_size,
                occ_size,
            )
            rays += 1
            traversed += t
            inherited += h
            first_occ += o

    return rays, traversed, inherited, first_occ


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
        "iou": float(inter / union) if union else 0.0,
        "precision": float(inter / pred_count) if pred_count else 0.0,
        "recall": float(inter / gt_count) if gt_count else 0.0,
    }


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


def build_camera_mask_for_frame(info, labels_path, args, pc_min, pc_max, voxel_size, occ_size):
    labels = np.load(labels_path)
    semantics = labels["semantics"].astype(np.uint8, copy=False)
    mask_lidar = labels["mask_lidar"].astype(np.bool_, copy=False)
    mask_camera = np.zeros(tuple(args.occ_size), dtype=np.bool_)

    per_camera = {}
    width, height = args.image_size
    for cam_name, cam_info in info["cams"].items():
        before = int(mask_camera.sum())
        rays, traversed, inherited, first_occ = _cast_camera_rays(
            int(width),
            int(height),
            int(args.pixel_stride),
            float(args.depth_max),
            np.asarray(cam_info["cam_intrinsic"], dtype=np.float64),
            np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64),
            np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64),
            semantics,
            mask_lidar,
            mask_camera,
            pc_min,
            pc_max,
            voxel_size,
            occ_size,
        )
        after = int(mask_camera.sum())
        per_camera[cam_name] = {
            "pixel_rays": int(rays),
            "traversed_voxels_with_repetition": int(traversed),
            "new_camera_voxels": int(after - before),
            "new_inherited_events": int(inherited),
            "first_occupied_hits": int(first_occ),
        }

    if np.any(mask_camera & ~mask_lidar):
        raise AssertionError(f"{labels_path}: mask_camera exceeds mask_lidar")

    stats = {
        "occupied_count": int((semantics != FREE_LABEL).sum()),
        "mask_lidar_count": int(mask_lidar.sum()),
        "mask_camera_count": int(mask_camera.sum()),
        "camera_free_count": int((mask_camera & (semantics == FREE_LABEL)).sum()),
        "camera_occ_count": int((mask_camera & (semantics != FREE_LABEL)).sum()),
        "camera_lidar_ratio": float(mask_camera.sum() / mask_lidar.sum()) if int(mask_lidar.sum()) else 0.0,
        "per_camera": per_camera,
    }
    return semantics, mask_lidar, mask_camera, stats


def check_stage4_file(path, occ_size):
    data = np.load(path)
    required = {"semantics", "mask_lidar", "mask_camera"}
    missing = sorted(required - set(data.files))
    if missing:
        raise AssertionError(f"{path}: missing keys {missing}")

    semantics = data["semantics"]
    mask_lidar = data["mask_lidar"]
    mask_camera = data["mask_camera"]
    expected_shape = tuple(occ_size)
    if semantics.shape != expected_shape:
        raise AssertionError(f"{path}: semantics shape {semantics.shape} != {expected_shape}")
    if mask_lidar.shape != expected_shape:
        raise AssertionError(f"{path}: mask_lidar shape {mask_lidar.shape} != {expected_shape}")
    if mask_camera.shape != expected_shape:
        raise AssertionError(f"{path}: mask_camera shape {mask_camera.shape} != {expected_shape}")
    if semantics.dtype not in (np.uint8, np.int32, np.int64):
        raise AssertionError(f"{path}: unsupported semantics dtype {semantics.dtype}")
    if mask_lidar.dtype not in (np.bool_, np.uint8):
        raise AssertionError(f"{path}: unsupported mask_lidar dtype {mask_lidar.dtype}")
    if mask_camera.dtype not in (np.bool_, np.uint8):
        raise AssertionError(f"{path}: unsupported mask_camera dtype {mask_camera.dtype}")
    if semantics.min() < 0 or semantics.max() > 17:
        raise AssertionError(f"{path}: semantics range outside 0-17")
    if np.any(mask_camera.astype(bool) & ~mask_lidar.astype(bool)):
        raise AssertionError(f"{path}: mask_camera exceeds mask_lidar")

    return {
        "path": str(path),
        "semantics_dtype": str(semantics.dtype),
        "mask_lidar_dtype": str(mask_lidar.dtype),
        "mask_camera_dtype": str(mask_camera.dtype),
        "occupied_count": int((semantics != FREE_LABEL).sum()),
        "mask_lidar_count": int(mask_lidar.astype(bool).sum()),
        "mask_camera_count": int(mask_camera.astype(bool).sum()),
        "mask_camera_subset_mask_lidar": True,
        "labels": [int(v) for v in np.unique(semantics)],
    }


def compare_official(updated_infos, official_root, output_path, num_classes):
    official_root = Path(official_root)
    totals = {
        "frames": 0,
        "missing_official": 0,
        "mask_camera_intersection": 0,
        "mask_camera_union": 0,
        "pred_camera_count": 0,
        "official_camera_count": 0,
    }
    hist_ref_camera = np.zeros((num_classes, num_classes), dtype=np.int64)
    files = []

    for info in updated_infos:
        pred_path = Path(info["occ_path"]) / "labels.npz"
        official_path = official_root / info["scene_name"] / info["token"] / "labels.npz"
        if not official_path.exists():
            totals["missing_official"] += 1
            continue

        pred = np.load(pred_path)
        gt = np.load(official_path)
        pred_sem = pred["semantics"]
        gt_sem = gt["semantics"]
        pred_camera = pred["mask_camera"].astype(bool)
        gt_camera = gt["mask_camera"].astype(bool)

        mask_summary = summarize_binary(pred_camera, gt_camera)
        hist_ref_camera += fast_hist(pred_sem, gt_sem, gt_camera, num_classes)

        totals["frames"] += 1
        totals["mask_camera_intersection"] += mask_summary["intersection"]
        totals["mask_camera_union"] += mask_summary["union"]
        totals["pred_camera_count"] += mask_summary["pred_count"]
        totals["official_camera_count"] += mask_summary["gt_count"]
        files.append(
            {
                "scene_name": info["scene_name"],
                "token": info["token"],
                "mask_camera_iou": mask_summary["iou"],
                "mask_camera_precision": mask_summary["precision"],
                "mask_camera_recall": mask_summary["recall"],
                "pred_camera_count": mask_summary["pred_count"],
                "official_camera_count": mask_summary["gt_count"],
            }
        )

    miou_ref_camera, ious_ref_camera = miou_from_hist(hist_ref_camera)
    aggregate = {
        "mask_camera_iou": float(totals["mask_camera_intersection"] / totals["mask_camera_union"])
        if totals["mask_camera_union"]
        else 0.0,
        "mask_camera_precision": float(totals["mask_camera_intersection"] / totals["pred_camera_count"])
        if totals["pred_camera_count"]
        else 0.0,
        "mask_camera_recall": float(totals["mask_camera_intersection"] / totals["official_camera_count"])
        if totals["official_camera_count"]
        else 0.0,
        "semantic_miou_ref_camera_mask": miou_ref_camera,
        "class_iou_ref_camera_mask": [None if np.isnan(v) else float(v) for v in ious_ref_camera],
    }
    report = {
        "official_root": str(official_root),
        "totals": totals,
        "aggregate": aggregate,
        "files": files,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    args = parse_args()
    if args.pixel_stride < 1:
        raise ValueError("--pixel-stride must be >= 1")

    output_root = Path(args.output_root)
    gts_root = output_root / "gts"
    ann_output = output_root / "bevdetv2-nuscenes_infos_stage4_train.pkl"
    stats_output = output_root / "stats_camera_mask.json"
    check_output = output_root / "check_stage4_format.json"
    compare_output = output_root / "compare_official_camera.json"
    gts_root.mkdir(parents=True, exist_ok=True)

    ann_data, infos = load_infos(args.input_ann)
    selected_infos = []
    for info in infos:
        if args.scene_name is not None and info.get("scene_name") != args.scene_name:
            continue
        selected_infos.append(info)
        if args.max_frames is not None and len(selected_infos) >= args.max_frames:
            break

    pc_range = np.asarray(args.occ_pc_range, dtype=np.float64)
    pc_min = pc_range[:3].copy()
    pc_max = pc_range[3:].copy()
    voxel_size = np.asarray(args.occ_voxel_size, dtype=np.float64)
    occ_size = np.asarray(args.occ_size, dtype=np.int64)

    # Warm up numba before timing frame 1. This compiles the exact signatures used below.
    dummy_sem = np.full((1, 1, 1), FREE_LABEL, dtype=np.uint8)
    dummy_lidar = np.ones((1, 1, 1), dtype=np.bool_)
    dummy_camera = np.zeros((1, 1, 1), dtype=np.bool_)
    dummy_origin = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    dummy_endpoint = np.array([0.9, 0.5, 0.5], dtype=np.float64)
    _mark_ray(
        dummy_origin,
        dummy_endpoint,
        dummy_sem,
        dummy_lidar,
        dummy_camera,
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([1, 1, 1], dtype=np.int64),
    )

    updated_infos = []
    file_stats = []
    check_files = []

    for frame_idx, info in enumerate(selected_infos):
        scene_name = info["scene_name"]
        token = info["token"]
        labels_path = Path(info["occ_path"]) / "labels.npz"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing input labels: {labels_path}")

        semantics, mask_lidar, mask_camera, stats = build_camera_mask_for_frame(
            info, labels_path, args, pc_min, pc_max, voxel_size, occ_size
        )

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
        updated_infos.append(updated_info)

        stats.update(
            {
                "frame_idx": frame_idx,
                "scene_name": scene_name,
                "token": token,
                "input_occ_path": str(labels_path.parent),
                "occ_path": str(occ_dir.resolve()),
            }
        )
        file_stats.append(stats)
        check_files.append(check_stage4_file(occ_dir / "labels.npz", args.occ_size))
        print(
            f"[{frame_idx + 1}/{len(selected_infos)}] {scene_name} {token}: "
            f"camera={stats['mask_camera_count']} lidar={stats['mask_lidar_count']} "
            f"ratio={stats['camera_lidar_ratio']:.4f} occ={stats['camera_occ_count']} "
            f"free={stats['camera_free_count']}"
        )

    save_ann_with_paths(ann_data, updated_infos, ann_output)

    summary = {
        "stage": "Stage 4: camera pixel-ray visibility mask",
        "input_ann": args.input_ann,
        "ann_output": str(ann_output),
        "gts_root": str(gts_root),
        "depth_max": float(args.depth_max),
        "image_size": [int(v) for v in args.image_size],
        "pixel_stride": int(args.pixel_stride),
        "grid": {
            "occ_point_cloud_range": [float(v) for v in args.occ_pc_range],
            "occ_voxel_size": [float(v) for v in args.occ_voxel_size],
            "occ_size": [int(v) for v in args.occ_size],
        },
        "known_issue": (
            "No camera/LiDAR timestamp motion compensation is applied; dynamic objects may keep "
            "minor ghost occlusion around visibility boundaries."
        ),
        "num_frames": len(file_stats),
        "totals": {
            "occupied_count": int(sum(item["occupied_count"] for item in file_stats)),
            "mask_lidar_count": int(sum(item["mask_lidar_count"] for item in file_stats)),
            "mask_camera_count": int(sum(item["mask_camera_count"] for item in file_stats)),
            "camera_free_count": int(sum(item["camera_free_count"] for item in file_stats)),
            "camera_occ_count": int(sum(item["camera_occ_count"] for item in file_stats)),
        },
        "files": file_stats,
    }
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    check_summary = {
        "root": str(gts_root),
        "occ_size": [int(v) for v in args.occ_size],
        "num_files": len(check_files),
        "total_occupied": int(sum(item["occupied_count"] for item in check_files)),
        "total_mask_lidar": int(sum(item["mask_lidar_count"] for item in check_files)),
        "total_mask_camera": int(sum(item["mask_camera_count"] for item in check_files)),
        "all_mask_camera_subset_mask_lidar": all(item["mask_camera_subset_mask_lidar"] for item in check_files),
        "files": check_files,
    }
    with open(check_output, "w", encoding="utf-8") as f:
        json.dump(check_summary, f, indent=2)

    official_report = compare_official(updated_infos, args.official_root, compare_output, args.num_classes)

    print(f"Wrote Stage 4 GT root: {gts_root}")
    print(f"Wrote Stage 4 ann pkl: {ann_output}")
    print(f"Wrote stats: {stats_output}")
    print(f"Wrote check: {check_output}")
    print(f"Wrote official camera comparison: {compare_output}")
    print(json.dumps(official_report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
