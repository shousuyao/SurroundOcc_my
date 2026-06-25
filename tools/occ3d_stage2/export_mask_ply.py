#!/usr/bin/env python3
"""Export Occ3D labels.npz masks to simple ASCII PLY point clouds."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export an Occ3D mask from labels.npz to PLY.")
    parser.add_argument("labels_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask", choices=("mask_lidar", "mask_camera", "occupied", "free"), default="mask_lidar")
    parser.add_argument("--pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth point after masking.")
    return parser.parse_args()


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


def main():
    args = parse_args()
    data = np.load(args.labels_npz)
    semantics = data["semantics"]

    if args.mask == "mask_lidar":
        mask = data["mask_lidar"].astype(bool)
        color = np.array([40, 180, 255], dtype=np.uint8)
    elif args.mask == "mask_camera":
        mask = data["mask_camera"].astype(bool)
        color = np.array([255, 180, 40], dtype=np.uint8)
    elif args.mask == "occupied":
        mask = data["mask_lidar"].astype(bool) & (semantics != 17)
        color = np.array([255, 80, 80], dtype=np.uint8)
    else:
        mask = data["mask_lidar"].astype(bool) & (semantics == 17)
        color = np.array([120, 220, 120], dtype=np.uint8)

    coords = np.argwhere(mask)
    if args.stride > 1:
        coords = coords[:: args.stride]

    pc_range = np.asarray(args.pc_range[:3], dtype=np.float32)
    voxel_size = np.asarray(args.voxel_size, dtype=np.float32)
    points = pc_range[None, :] + (coords.astype(np.float32) + 0.5) * voxel_size[None, :]
    colors = np.repeat(color[None, :], points.shape[0], axis=0)
    write_ply(args.output, points, colors)

    if points.shape[0] == 0:
        summary = {"count": 0}
    else:
        summary = {
            "count": int(points.shape[0]),
            "mean": [float(v) for v in points.mean(axis=0)],
            "min": [float(v) for v in points.min(axis=0)],
            "max": [float(v) for v in points.max(axis=0)],
        }
    print(summary)


if __name__ == "__main__":
    main()
