#!/usr/bin/env python3
"""Generate Occ3D-style LiDAR ray-casting masks for FlashOCC.

This Stage 2 tool can raycast either raw nuScenes LIDAR_TOP sweeps or dense
SurroundOcc occupancy candidates directly in the FlashOCC/Occ3D grid:

    point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
    voxel_size = [0.4, 0.4, 0.4]

Use ``--ray-source dense-candidate`` with candidates generated in the same
Occ3D grid when matching Occ3D-style aggregated-point raycasting.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml
from numba import njit
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


FREE_LABEL = 17
UNKNOWN_LABEL = 255


@njit
def _raycast_points(
    origins,
    points,
    labels,
    protect_rays,
    protected_grid,
    pc_range,
    voxel_size,
    occ_size,
    lidar_max_range,
    skip_free_z_min,
    skip_free_z_max,
    skip_free_z_grazing_angle_rad,
    skip_free_z_min_ray_length,
):
    free_grid = np.zeros((occ_size[0], occ_size[1], occ_size[2]), np.bool_)
    raw_hit_grid = np.zeros((occ_size[0], occ_size[1], occ_size[2]), np.bool_)
    hit_label_counts = np.zeros((occ_size[0], occ_size[1], occ_size[2], 17), np.uint16)

    grid_min = pc_range[:3]
    grid_max = pc_range[3:]
    eps = 1e-6
    max_steps = occ_size[0] + occ_size[1] + occ_size[2] + 8

    rays_total = points.shape[0]
    rays_intersect = 0
    rays_endpoint_inside = 0
    rays_endpoint_outside = 0
    rays_no_intersection = 0
    free_writes = 0
    hit_writes = 0
    protected_truncations = 0
    max_range_clipped = 0

    for n in range(points.shape[0]):
        origin = origins[n]
        endpoint = points[n]
        direction = endpoint - origin
        has_hit = True
        stopped_by_protected = False

        ray_length = np.sqrt(
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]
        )
        if lidar_max_range > 0.0 and ray_length > lidar_max_range:
            scale = np.float32(lidar_max_range / ray_length)
            endpoint = (origin + direction * scale).astype(np.float32)
            direction = (endpoint - origin).astype(np.float32)
            has_hit = False
            max_range_clipped += 1

        apply_z_skip = skip_free_z_min >= 0
        if apply_z_skip and skip_free_z_min_ray_length >= 0.0 and ray_length < skip_free_z_min_ray_length:
            apply_z_skip = False
        if apply_z_skip and skip_free_z_grazing_angle_rad >= 0.0:
            horizontal = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
            vertical_angle = abs(np.arctan2(direction[2], horizontal))
            if vertical_angle > skip_free_z_grazing_angle_rad:
                apply_z_skip = False

        t_entry = 0.0
        t_exit = 1.0
        intersects = True

        for axis in range(3):
            if abs(direction[axis]) < eps:
                if origin[axis] < grid_min[axis] or origin[axis] >= grid_max[axis]:
                    intersects = False
                    break
            else:
                inv_d = 1.0 / direction[axis]
                t0 = (grid_min[axis] - origin[axis]) * inv_d
                t1 = (grid_max[axis] - origin[axis]) * inv_d
                if t0 > t1:
                    tmp = t0
                    t0 = t1
                    t1 = tmp
                if t0 > t_entry:
                    t_entry = t0
                if t1 < t_exit:
                    t_exit = t1
                if t_entry > t_exit:
                    intersects = False
                    break

        if not intersects or t_exit <= 0.0 or t_entry >= 1.0:
            rays_no_intersection += 1
            continue

        rays_intersect += 1

        endpoint_inside = True
        hit_ix = -1
        hit_iy = -1
        hit_iz = -1
        for axis in range(3):
            if endpoint[axis] < grid_min[axis] or endpoint[axis] >= grid_max[axis]:
                endpoint_inside = False
                break

        if endpoint_inside:
            hit_ix = int(np.floor((endpoint[0] - grid_min[0]) / voxel_size[0]))
            hit_iy = int(np.floor((endpoint[1] - grid_min[1]) / voxel_size[1]))
            hit_iz = int(np.floor((endpoint[2] - grid_min[2]) / voxel_size[2]))
            if (
                hit_ix < 0
                or hit_ix >= occ_size[0]
                or hit_iy < 0
                or hit_iy >= occ_size[1]
                or hit_iz < 0
                or hit_iz >= occ_size[2]
            ):
                endpoint_inside = False

        if endpoint_inside and has_hit:
            rays_endpoint_inside += 1
            traverse_end = 1.0
        else:
            endpoint_inside = False
            rays_endpoint_outside += 1
            traverse_end = t_exit

        start_t = t_entry
        if start_t < 0.0:
            start_t = 0.0
        if t_entry > 0.0:
            start_t = start_t + eps

        if start_t >= traverse_end:
            if endpoint_inside and has_hit:
                raw_hit_grid[hit_ix, hit_iy, hit_iz] = True
                label = labels[n]
                if label >= 0 and label <= 16:
                    if hit_label_counts[hit_ix, hit_iy, hit_iz, label] < 65535:
                        hit_label_counts[hit_ix, hit_iy, hit_iz, label] += 1
                hit_writes += 1
            continue

        start = origin + direction * start_t
        ix = int(np.floor((start[0] - grid_min[0]) / voxel_size[0]))
        iy = int(np.floor((start[1] - grid_min[1]) / voxel_size[1]))
        iz = int(np.floor((start[2] - grid_min[2]) / voxel_size[2]))

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

        step_x = 0
        step_y = 0
        step_z = 0
        t_max_x = 1e30
        t_max_y = 1e30
        t_max_z = 1e30
        t_delta_x = 1e30
        t_delta_y = 1e30
        t_delta_z = 1e30

        if direction[0] > eps:
            step_x = 1
            boundary = grid_min[0] + (ix + 1) * voxel_size[0]
            t_max_x = (boundary - origin[0]) / direction[0]
            t_delta_x = voxel_size[0] / direction[0]
        elif direction[0] < -eps:
            step_x = -1
            boundary = grid_min[0] + ix * voxel_size[0]
            t_max_x = (boundary - origin[0]) / direction[0]
            t_delta_x = -voxel_size[0] / direction[0]

        if direction[1] > eps:
            step_y = 1
            boundary = grid_min[1] + (iy + 1) * voxel_size[1]
            t_max_y = (boundary - origin[1]) / direction[1]
            t_delta_y = voxel_size[1] / direction[1]
        elif direction[1] < -eps:
            step_y = -1
            boundary = grid_min[1] + iy * voxel_size[1]
            t_max_y = (boundary - origin[1]) / direction[1]
            t_delta_y = -voxel_size[1] / direction[1]

        if direction[2] > eps:
            step_z = 1
            boundary = grid_min[2] + (iz + 1) * voxel_size[2]
            t_max_z = (boundary - origin[2]) / direction[2]
            t_delta_z = voxel_size[2] / direction[2]
        elif direction[2] < -eps:
            step_z = -1
            boundary = grid_min[2] + iz * voxel_size[2]
            t_max_z = (boundary - origin[2]) / direction[2]
            t_delta_z = -voxel_size[2] / direction[2]

        for _ in range(max_steps):
            if ix < 0 or ix >= occ_size[0] or iy < 0 or iy >= occ_size[1] or iz < 0 or iz >= occ_size[2]:
                break

            if endpoint_inside and ix == hit_ix and iy == hit_iy and iz == hit_iz:
                break

            if protect_rays[n] and protected_grid[ix, iy, iz]:
                protected_truncations += 1
                stopped_by_protected = True
                break

            skip_free = apply_z_skip and iz >= skip_free_z_min and iz <= skip_free_z_max
            if not skip_free:
                if not free_grid[ix, iy, iz]:
                    free_writes += 1
                free_grid[ix, iy, iz] = True

            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                if t_max_x >= traverse_end:
                    break
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_x and t_max_y <= t_max_z:
                if t_max_y >= traverse_end:
                    break
                iy += step_y
                t_max_y += t_delta_y
            else:
                if t_max_z >= traverse_end:
                    break
                iz += step_z
                t_max_z += t_delta_z

        if endpoint_inside and has_hit and not stopped_by_protected:
            raw_hit_grid[hit_ix, hit_iy, hit_iz] = True
            label = labels[n]
            if label >= 0 and label <= 16:
                if hit_label_counts[hit_ix, hit_iy, hit_iz, label] < 65535:
                    hit_label_counts[hit_ix, hit_iy, hit_iz, label] += 1
            hit_writes += 1

    stats = np.array(
        [
            rays_total,
            rays_intersect,
            rays_endpoint_inside,
            rays_endpoint_outside,
            rays_no_intersection,
            free_writes,
            hit_writes,
            protected_truncations,
            max_range_clipped,
        ],
        dtype=np.int64,
    )
    return free_grid, raw_hit_grid, hit_label_counts, stats


@njit
def _raycast_points_occ3d_point_to_origin(
    origins,
    points,
    labels,
    protect_rays,
    protected_grid,
    pc_range,
    voxel_size,
    occ_size,
    lidar_max_range,
    skip_free_z_min,
    skip_free_z_max,
    skip_free_z_grazing_angle_rad,
    skip_free_z_min_ray_length,
):
    free_grid = np.zeros((occ_size[0], occ_size[1], occ_size[2]), np.bool_)
    raw_hit_grid = np.zeros((occ_size[0], occ_size[1], occ_size[2]), np.bool_)
    hit_label_counts = np.zeros((occ_size[0], occ_size[1], occ_size[2], 17), np.uint16)

    grid_min = pc_range[:3]
    grid_max = pc_range[3:]
    eps = 1e-6
    max_steps = occ_size[0] + occ_size[1] + occ_size[2] + 8

    rays_total = points.shape[0]
    rays_intersect = 0
    rays_endpoint_inside = 0
    rays_endpoint_outside = 0
    rays_no_intersection = 0
    free_writes = 0
    hit_writes = 0
    protected_truncations = 0
    max_range_clipped = 0

    for n in range(points.shape[0]):
        point = points[n]
        origin = origins[n]

        point_inside = True
        for axis in range(3):
            if point[axis] < grid_min[axis] or point[axis] >= grid_max[axis]:
                point_inside = False
                break
        if not point_inside:
            rays_no_intersection += 1
            continue

        hit_ix = int(np.floor((point[0] - grid_min[0]) / voxel_size[0]))
        hit_iy = int(np.floor((point[1] - grid_min[1]) / voxel_size[1]))
        hit_iz = int(np.floor((point[2] - grid_min[2]) / voxel_size[2]))
        if (
            hit_ix < 0
            or hit_ix >= occ_size[0]
            or hit_iy < 0
            or hit_iy >= occ_size[1]
            or hit_iz < 0
            or hit_iz >= occ_size[2]
        ):
            rays_no_intersection += 1
            continue

        rays_intersect += 1
        rays_endpoint_inside += 1
        raw_hit_grid[hit_ix, hit_iy, hit_iz] = True
        label = labels[n]
        if label >= 0 and label <= 16:
            if hit_label_counts[hit_ix, hit_iy, hit_iz, label] < 65535:
                hit_label_counts[hit_ix, hit_iy, hit_iz, label] += 1
        hit_writes += 1

        direction = origin - point
        ray_length = np.sqrt(
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]
        )
        if ray_length <= eps:
            continue
        if lidar_max_range > 0.0 and ray_length > lidar_max_range:
            scale = np.float32(lidar_max_range / ray_length)
            origin = (point + direction * scale).astype(np.float32)
            direction = (origin - point).astype(np.float32)
            ray_length = lidar_max_range
            max_range_clipped += 1

        apply_z_skip = skip_free_z_min >= 0
        if apply_z_skip and skip_free_z_min_ray_length >= 0.0 and ray_length < skip_free_z_min_ray_length:
            apply_z_skip = False
        if apply_z_skip and skip_free_z_grazing_angle_rad >= 0.0:
            horizontal = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
            vertical_angle = abs(np.arctan2(direction[2], horizontal))
            if vertical_angle > skip_free_z_grazing_angle_rad:
                apply_z_skip = False

        origin_inside = True
        last_ix = -1
        last_iy = -1
        last_iz = -1
        for axis in range(3):
            if origin[axis] < grid_min[axis] or origin[axis] >= grid_max[axis]:
                origin_inside = False
                break
        if origin_inside:
            last_ix = int(np.floor((origin[0] - grid_min[0]) / voxel_size[0]))
            last_iy = int(np.floor((origin[1] - grid_min[1]) / voxel_size[1]))
            last_iz = int(np.floor((origin[2] - grid_min[2]) / voxel_size[2]))
            if (
                last_ix < 0
                or last_ix >= occ_size[0]
                or last_iy < 0
                or last_iy >= occ_size[1]
                or last_iz < 0
                or last_iz >= occ_size[2]
            ):
                origin_inside = False

        ix = hit_ix
        iy = hit_iy
        iz = hit_iz

        step_x = 0
        step_y = 0
        step_z = 0
        t_max_x = 1e30
        t_max_y = 1e30
        t_max_z = 1e30
        t_delta_x = 1e30
        t_delta_y = 1e30
        t_delta_z = 1e30

        if direction[0] > eps:
            step_x = 1
            boundary = grid_min[0] + (ix + 1) * voxel_size[0]
            t_max_x = (boundary - point[0]) / direction[0]
            t_delta_x = voxel_size[0] / direction[0]
        elif direction[0] < -eps:
            step_x = -1
            boundary = grid_min[0] + ix * voxel_size[0]
            t_max_x = (boundary - point[0]) / direction[0]
            t_delta_x = -voxel_size[0] / direction[0]

        if direction[1] > eps:
            step_y = 1
            boundary = grid_min[1] + (iy + 1) * voxel_size[1]
            t_max_y = (boundary - point[1]) / direction[1]
            t_delta_y = voxel_size[1] / direction[1]
        elif direction[1] < -eps:
            step_y = -1
            boundary = grid_min[1] + iy * voxel_size[1]
            t_max_y = (boundary - point[1]) / direction[1]
            t_delta_y = -voxel_size[1] / direction[1]

        if direction[2] > eps:
            step_z = 1
            boundary = grid_min[2] + (iz + 1) * voxel_size[2]
            t_max_z = (boundary - point[2]) / direction[2]
            t_delta_z = voxel_size[2] / direction[2]
        elif direction[2] < -eps:
            step_z = -1
            boundary = grid_min[2] + iz * voxel_size[2]
            t_max_z = (boundary - point[2]) / direction[2]
            t_delta_z = -voxel_size[2] / direction[2]

        stopped_by_protected = False
        for _ in range(max_steps):
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                if t_max_x >= 1.0:
                    break
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_x and t_max_y <= t_max_z:
                if t_max_y >= 1.0:
                    break
                iy += step_y
                t_max_y += t_delta_y
            else:
                if t_max_z >= 1.0:
                    break
                iz += step_z
                t_max_z += t_delta_z

            if ix < 0 or ix >= occ_size[0] or iy < 0 or iy >= occ_size[1] or iz < 0 or iz >= occ_size[2]:
                break

            if origin_inside and ix == last_ix and iy == last_iy and iz == last_iz:
                break

            if protect_rays[n] and protected_grid[ix, iy, iz]:
                protected_truncations += 1
                stopped_by_protected = True
                break

            skip_free = apply_z_skip and iz >= skip_free_z_min and iz <= skip_free_z_max
            if not skip_free and not (ix == hit_ix and iy == hit_iy and iz == hit_iz):
                if not free_grid[ix, iy, iz]:
                    free_writes += 1
                free_grid[ix, iy, iz] = True

        if stopped_by_protected:
            continue

    stats = np.array(
        [
            rays_total,
            rays_intersect,
            rays_endpoint_inside,
            rays_endpoint_outside,
            rays_no_intersection,
            free_writes,
            hit_writes,
            protected_truncations,
            max_range_clipped,
        ],
        dtype=np.int64,
    )
    return free_grid, raw_hit_grid, hit_label_counts, stats


@njit
def _raycast_surface_points(
    origins,
    points,
    labels,
    pc_range,
    voxel_size,
    occ_size,
    surface_free_distance,
    skip_free_z_min,
    skip_free_z_max,
    skip_free_z_grazing_angle_rad,
    skip_free_z_min_ray_length,
):
    free_grid = np.zeros((occ_size[0], occ_size[1], occ_size[2]), np.bool_)
    raw_hit_grid = np.zeros((occ_size[0], occ_size[1], occ_size[2]), np.bool_)
    hit_label_counts = np.zeros((occ_size[0], occ_size[1], occ_size[2], 17), np.uint16)

    grid_min = pc_range[:3]
    grid_max = pc_range[3:]
    eps = 1e-6
    max_steps = occ_size[0] + occ_size[1] + occ_size[2] + 8

    rays_total = points.shape[0]
    rays_intersect = 0
    rays_endpoint_inside = 0
    rays_endpoint_outside = 0
    rays_no_intersection = 0
    free_writes = 0
    hit_writes = 0
    protected_truncations = 0
    max_range_clipped = 0

    for n in range(points.shape[0]):
        point = points[n]
        origin = origins[n]

        inside = True
        for axis in range(3):
            if point[axis] < grid_min[axis] or point[axis] >= grid_max[axis]:
                inside = False
                break
        if not inside:
            rays_no_intersection += 1
            continue

        hit_ix = int(np.floor((point[0] - grid_min[0]) / voxel_size[0]))
        hit_iy = int(np.floor((point[1] - grid_min[1]) / voxel_size[1]))
        hit_iz = int(np.floor((point[2] - grid_min[2]) / voxel_size[2]))
        if (
            hit_ix < 0
            or hit_ix >= occ_size[0]
            or hit_iy < 0
            or hit_iy >= occ_size[1]
            or hit_iz < 0
            or hit_iz >= occ_size[2]
        ):
            rays_no_intersection += 1
            continue

        rays_intersect += 1
        rays_endpoint_inside += 1
        raw_hit_grid[hit_ix, hit_iy, hit_iz] = True
        label = labels[n]
        if label >= 0 and label <= 16:
            if hit_label_counts[hit_ix, hit_iy, hit_iz, label] < 65535:
                hit_label_counts[hit_ix, hit_iy, hit_iz, label] += 1
        hit_writes += 1

        direction_full = origin - point
        ray_length = np.sqrt(
            direction_full[0] * direction_full[0]
            + direction_full[1] * direction_full[1]
            + direction_full[2] * direction_full[2]
        )
        if ray_length <= eps or surface_free_distance <= 0.0:
            continue

        apply_z_skip = skip_free_z_min >= 0
        if apply_z_skip and skip_free_z_min_ray_length >= 0.0 and ray_length < skip_free_z_min_ray_length:
            apply_z_skip = False
        if apply_z_skip and skip_free_z_grazing_angle_rad >= 0.0:
            horizontal = np.sqrt(
                direction_full[0] * direction_full[0] + direction_full[1] * direction_full[1]
            )
            vertical_angle = abs(np.arctan2(direction_full[2], horizontal))
            if vertical_angle > skip_free_z_grazing_angle_rad:
                apply_z_skip = False

        free_length = surface_free_distance
        if free_length > ray_length:
            free_length = ray_length
        else:
            max_range_clipped += 1
        direction = direction_full * np.float32(free_length / ray_length)
        traverse_end = 1.0

        start = point + direction * eps
        ix = int(np.floor((start[0] - grid_min[0]) / voxel_size[0]))
        iy = int(np.floor((start[1] - grid_min[1]) / voxel_size[1]))
        iz = int(np.floor((start[2] - grid_min[2]) / voxel_size[2]))

        step_x = 0
        step_y = 0
        step_z = 0
        t_max_x = 1e30
        t_max_y = 1e30
        t_max_z = 1e30
        t_delta_x = 1e30
        t_delta_y = 1e30
        t_delta_z = 1e30

        if direction[0] > eps:
            step_x = 1
            boundary = grid_min[0] + (ix + 1) * voxel_size[0]
            t_max_x = (boundary - point[0]) / direction[0]
            t_delta_x = voxel_size[0] / direction[0]
        elif direction[0] < -eps:
            step_x = -1
            boundary = grid_min[0] + ix * voxel_size[0]
            t_max_x = (boundary - point[0]) / direction[0]
            t_delta_x = -voxel_size[0] / direction[0]

        if direction[1] > eps:
            step_y = 1
            boundary = grid_min[1] + (iy + 1) * voxel_size[1]
            t_max_y = (boundary - point[1]) / direction[1]
            t_delta_y = voxel_size[1] / direction[1]
        elif direction[1] < -eps:
            step_y = -1
            boundary = grid_min[1] + iy * voxel_size[1]
            t_max_y = (boundary - point[1]) / direction[1]
            t_delta_y = -voxel_size[1] / direction[1]

        if direction[2] > eps:
            step_z = 1
            boundary = grid_min[2] + (iz + 1) * voxel_size[2]
            t_max_z = (boundary - point[2]) / direction[2]
            t_delta_z = voxel_size[2] / direction[2]
        elif direction[2] < -eps:
            step_z = -1
            boundary = grid_min[2] + iz * voxel_size[2]
            t_max_z = (boundary - point[2]) / direction[2]
            t_delta_z = -voxel_size[2] / direction[2]

        for _ in range(max_steps):
            if ix < 0 or ix >= occ_size[0] or iy < 0 or iy >= occ_size[1] or iz < 0 or iz >= occ_size[2]:
                break

            skip_free = apply_z_skip and iz >= skip_free_z_min and iz <= skip_free_z_max
            if not skip_free and not (ix == hit_ix and iy == hit_iy and iz == hit_iz):
                if not free_grid[ix, iy, iz]:
                    free_writes += 1
                free_grid[ix, iy, iz] = True

            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                if t_max_x >= traverse_end:
                    break
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_x and t_max_y <= t_max_z:
                if t_max_y >= traverse_end:
                    break
                iy += step_y
                t_max_y += t_delta_y
            else:
                if t_max_z >= traverse_end:
                    break
                iz += step_z
                t_max_z += t_delta_z

    stats = np.array(
        [
            rays_total,
            rays_intersect,
            rays_endpoint_inside,
            rays_endpoint_outside,
            rays_no_intersection,
            free_writes,
            hit_writes,
            protected_truncations,
            max_range_clipped,
        ],
        dtype=np.int64,
    )
    return free_grid, raw_hit_grid, hit_label_counts, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 LiDAR ray casting for Occ3D grid.")
    parser.add_argument(
        "--dataroot",
        default="data/nuscenes",
        help="nuScenes dataroot containing v1.0-mini, samples and lidarseg.",
    )
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--ann-file",
        default="../FlashOCC/data/nuScenes/bevdetv2-nuscenes_infos_train.pkl",
        help="FlashOCC info pkl. Used for scene/token order and output occ_path mapping.",
    )
    parser.add_argument(
        "--label-mapping",
        default="tools/generate_occupancy_nuscenes/nuscenes.yaml",
        help="nuScenes lidarseg learning_map yaml.",
    )
    parser.add_argument(
        "--output-root",
        default="data/GT_occupancy_mini/stage2_raycast_occ3d",
        help="Output root for gts, debug npz and updated ann pkl.",
    )
    parser.add_argument(
        "--ray-source",
        choices=("raw-sweeps", "dense-candidate"),
        default="raw-sweeps",
        help="Ray endpoints source: raw nuScenes sweeps or dense SurroundOcc occupancy candidates.",
    )
    parser.add_argument(
        "--dense-candidate-dir",
        default="data/GT_occupancy_mini_occ3d_grid/dense_voxels_with_semantic",
        help="Directory with Occ3D-grid dense candidate .npy files when --ray-source dense-candidate.",
    )
    parser.add_argument(
        "--dense-ray-source-dir",
        default=None,
        help=(
            "Optional directory with LiDAR-supported dense ray source .npy files. "
            "When set, occupied voxels come from --dense-candidate-dir but free rays use this directory."
        ),
    )
    parser.add_argument(
        "--dense-ray-source-points-dir",
        default=None,
        help=(
            "Optional directory with per-point ray source .npy files of shape (N, 7): "
            "point_xyz, origin_xyz, semantic. Overrides --dense-ray-source-dir for free rays."
        ),
    )
    parser.add_argument(
        "--dense-coordinate-transform",
        choices=("identity", "swapxy_flipy"),
        default="identity",
        help="Optional voxel-index transform for dense candidate coordinates before raycasting.",
    )
    parser.add_argument("--pc-range", nargs=6, type=float, default=(-40, -40, -1, 40, 40, 5.4))
    parser.add_argument("--voxel-size", nargs=3, type=float, default=(0.4, 0.4, 0.4))
    parser.add_argument("--occ-size", nargs=3, type=int, default=(200, 200, 16))
    parser.add_argument("--self-range", nargs=3, type=float, default=(3.0, 3.0, 3.0))
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=1,
        help="Number of LIDAR_TOP sample_data sweeps per target frame, including the keyframe.",
    )
    parser.add_argument(
        "--sweep-direction",
        choices=("previous", "next", "both", "scene"),
        default="previous",
        help="Where to collect additional sweeps from the sample_data chain.",
    )
    parser.add_argument(
        "--scene-sweep-stride",
        type=int,
        default=1,
        help="Use every Nth LIDAR_TOP sample_data in --sweep-direction scene mode.",
    )
    parser.add_argument(
        "--truncate-protected-free",
        action="store_true",
        help="Stop non-key sweep rays when they enter target-frame protected dynamic boxes.",
    )
    parser.add_argument(
        "--dynamic-classes",
        default="car,truck,bus,trailer,construction_vehicle,bicycle,motorcycle,pedestrian",
        help="Comma-separated gt_names used to build target-frame dynamic protection boxes.",
    )
    parser.add_argument(
        "--protected-box-margin",
        type=float,
        default=0.0,
        help="Metric margin added to each protected dynamic box dimension.",
    )
    parser.add_argument(
        "--lidar-max-range",
        type=float,
        default=0.0,
        help="If >0, rays beyond this distance are clipped and do not contribute hit voxels.",
    )
    parser.add_argument(
        "--surface-free-distance",
        type=float,
        default=0.0,
        help=(
            "Dense-candidate mode only. If >0, mark occupied candidates first, then cast free "
            "space from each surface point back toward the LiDAR origin for this metric distance."
        ),
    )
    parser.add_argument(
        "--ray-traversal",
        choices=("origin-to-point", "occ3d-point-to-origin"),
        default="origin-to-point",
        help=(
            "Voxel traversal convention for full-path raycasting. "
            "origin-to-point keeps the legacy clipped line traversal; "
            "occ3d-point-to-origin starts at each hit point and walks toward points_origin, "
            "excluding the hit voxel and the origin voxel."
        ),
    )
    parser.add_argument(
        "--skip-free-z-min",
        type=int,
        default=-1,
        help="If >=0, do not write free voxels whose z index is in [min, max].",
    )
    parser.add_argument(
        "--skip-free-z-max",
        type=int,
        default=-1,
        help="Upper z index for --skip-free-z-min.",
    )
    parser.add_argument(
        "--skip-free-z-grazing-angle-deg",
        type=float,
        default=-1.0,
        help=(
            "If >=0, apply the z free-write skip only to rays whose absolute vertical angle "
            "is at most this many degrees. This refines low-Z suppression to near-horizontal rays."
        ),
    )
    parser.add_argument(
        "--skip-free-z-min-ray-length",
        type=float,
        default=-1.0,
        help=(
            "If >=0, apply the z free-write skip only to rays whose point-origin length is at "
            "least this many meters."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for debugging. By default all matching frames are processed.",
    )
    parser.add_argument(
        "--scene-name",
        default=None,
        help="Optional scene filter, e.g. scene-0061.",
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


def save_ann_with_stage2_paths(ann_data, infos, ann_output):
    if isinstance(ann_data, dict) and "infos" in ann_data:
        out_data = dict(ann_data)
        out_data["infos"] = infos
    else:
        out_data = infos
    with open(ann_output, "wb") as f:
        pickle.dump(out_data, f)


def load_learning_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    learning_map = data["learning_map"]
    max_key = max(int(k) for k in learning_map.keys())
    lut = np.zeros(max_key + 1, dtype=np.int16)
    for key, value in learning_map.items():
        lut[int(key)] = int(value)
    return lut


def apply_self_range_filter(points, labels, self_range):
    self_range = np.asarray(self_range, dtype=np.float32)
    keep = (
        (np.abs(points[:, 0]) > self_range[0])
        | (np.abs(points[:, 1]) > self_range[1])
        | (np.abs(points[:, 2]) > self_range[2])
    )
    return points[keep], labels[keep], int((~keep).sum())


def transform_matrix(rotation, translation):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def lidar_to_global(nusc, lidar_sd):
    calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    ego_from_lidar = transform_matrix(calib["rotation"], calib["translation"])
    global_from_ego = transform_matrix(ego_pose["rotation"], ego_pose["translation"])
    return global_from_ego @ ego_from_lidar


def transform_points(points, transform):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (points_h @ transform.T)[:, :3].astype(np.float32)


def transform_origin(transform):
    origin_h = transform @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return origin_h[:3].astype(np.float32)


def voxel_centers_for_ranges(x0, x1, y0, y1, z0, z1, pc_range, voxel_size):
    xs = pc_range[0] + (np.arange(x0, x1, dtype=np.float32) + 0.5) * voxel_size[0]
    ys = pc_range[1] + (np.arange(y0, y1, dtype=np.float32) + 0.5) * voxel_size[1]
    zs = pc_range[2] + (np.arange(z0, z1, dtype=np.float32) + 0.5) * voxel_size[2]
    return np.meshgrid(xs, ys, zs, indexing="ij")


def build_protected_dynamic_grid(info, pc_range, voxel_size, occ_size, dynamic_classes, margin):
    protected = np.zeros(tuple(occ_size), dtype=bool)
    gt_boxes = info.get("gt_boxes")
    gt_names = info.get("gt_names")
    valid_flag = info.get("valid_flag")
    if gt_boxes is None or gt_names is None:
        return protected, 0

    dynamic_set = {name.strip() for name in dynamic_classes.split(",") if name.strip()}
    pc_min = np.asarray(pc_range[:3], dtype=np.float32)
    pc_max = np.asarray(pc_range[3:], dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    protected_boxes = 0

    for idx, box in enumerate(gt_boxes):
        if valid_flag is not None and not bool(valid_flag[idx]):
            continue
        if str(gt_names[idx]) not in dynamic_set:
            continue

        center = box[:3].astype(np.float32)
        dims = box[3:6].astype(np.float32) + float(margin) * 2.0
        yaw = float(box[6])
        half_xy = np.abs(
            np.array(
                [
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)],
                ],
                dtype=np.float32,
            )
        ) @ (dims[:2] * 0.5)
        min_corner = np.array(
            [center[0] - half_xy[0], center[1] - half_xy[1], center[2] - dims[2] * 0.5],
            dtype=np.float32,
        )
        max_corner = np.array(
            [center[0] + half_xy[0], center[1] + half_xy[1], center[2] + dims[2] * 0.5],
            dtype=np.float32,
        )
        if np.any(max_corner <= pc_min) or np.any(min_corner >= pc_max):
            continue

        lo = np.floor((np.maximum(min_corner, pc_min) - pc_min) / voxel_size).astype(np.int64)
        hi = np.ceil((np.minimum(max_corner, pc_max) - pc_min) / voxel_size).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, occ_size)
        if np.any(hi <= lo):
            continue

        X, Y, Z = voxel_centers_for_ranges(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], pc_range, voxel_size)
        dx = X - center[0]
        dy = Y - center[1]
        local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
        local_z = Z - center[2]
        inside = (
            (np.abs(local_x) <= dims[0] * 0.5)
            & (np.abs(local_y) <= dims[1] * 0.5)
            & (np.abs(local_z) <= dims[2] * 0.5)
        )
        protected[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
        protected_boxes += 1

    return protected, protected_boxes


def get_lidarseg_record(nusc, sample_data_token):
    try:
        return nusc.get("lidarseg", sample_data_token)
    except KeyError:
        return None


def collect_scene_lidar_sweeps(nusc, key_lidar_sd, stride):
    sample = nusc.get("sample", key_lidar_sd["sample_token"])
    scene = nusc.get("scene", sample["scene_token"])

    sample_tokens = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample_tokens.append(sample_token)
        sample_token = nusc.get("sample", sample_token)["next"]

    sample_token_set = set(sample_tokens)
    lidar_sds = [
        sd
        for sd in nusc.sample_data
        if sd["sample_token"] in sample_token_set
        and "LIDAR_TOP" in sd["filename"]
        and sd["filename"].endswith(".pcd.bin")
    ]
    lidar_sds = sorted(lidar_sds, key=lambda item: item["timestamp"])

    stride = max(1, int(stride))
    if stride > 1:
        selected = lidar_sds[::stride]
        if key_lidar_sd["token"] not in {sd["token"] for sd in selected}:
            selected.append(key_lidar_sd)
            selected = sorted(selected, key=lambda item: item["timestamp"])
        return selected
    return lidar_sds


def collect_lidar_sweeps(nusc, key_lidar_sd, num_sweeps, direction, scene_sweep_stride=1):
    if direction == "scene":
        return collect_scene_lidar_sweeps(nusc, key_lidar_sd, scene_sweep_stride)

    if num_sweeps <= 1:
        return [key_lidar_sd]

    sweeps = [key_lidar_sd]
    if direction in ("previous", "both"):
        prev_token = key_lidar_sd["prev"]
        while prev_token and len(sweeps) < num_sweeps:
            sd = nusc.get("sample_data", prev_token)
            if sd["filename"].endswith(".pcd.bin") and "LIDAR_TOP" in sd["filename"]:
                sweeps.append(sd)
            prev_token = sd["prev"]

    if direction == "next":
        sweeps = [key_lidar_sd]

    if direction in ("next", "both"):
        next_token = key_lidar_sd["next"]
        while next_token and len(sweeps) < num_sweeps:
            sd = nusc.get("sample_data", next_token)
            if sd["filename"].endswith(".pcd.bin") and "LIDAR_TOP" in sd["filename"]:
                sweeps.append(sd)
            next_token = sd["next"]

    return sweeps


def load_sweep_points_and_labels(nusc, dataroot, lidar_sd, learning_lut, self_range):
    lidar_path = resolve_lidar_path(dataroot, lidar_sd["filename"])
    raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points = raw[:, :3].astype(np.float32)

    lidarseg = get_lidarseg_record(nusc, lidar_sd["token"])
    labels_path = None
    if lidarseg is not None:
        labels_path = os.path.join(dataroot, lidarseg["filename"])
        raw_labels = np.fromfile(labels_path, dtype=np.uint8)
        if raw_labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"Point/label count mismatch for {lidar_sd['filename']}: "
                f"{points.shape[0]} points vs {raw_labels.shape[0]} labels"
            )
        labels = learning_lut[raw_labels].astype(np.int16)
    else:
        labels = np.full(points.shape[0], -1, dtype=np.int16)

    points, labels, self_filtered = apply_self_range_filter(points, labels, self_range)
    return points, labels, self_filtered, lidarseg, labels_path, int(raw.shape[0])


def transform_dense_coords(coords, occ_size, transform_name):
    if transform_name == "identity":
        return coords
    if transform_name == "swapxy_flipy":
        transformed = coords.copy()
        transformed[:, 0] = coords[:, 1]
        transformed[:, 1] = occ_size[1] - 1 - coords[:, 0]
        return transformed
    raise ValueError(f"Unsupported dense coordinate transform: {transform_name}")


def transform_dense_metric_points(points, transform_name):
    if transform_name == "identity":
        return points
    if transform_name == "swapxy_flipy":
        transformed = points.copy()
        transformed[:, 0] = points[:, 1]
        transformed[:, 1] = -points[:, 0]
        return transformed
    raise ValueError(f"Unsupported dense coordinate transform: {transform_name}")


def voxel_centers_from_coords(coords, pc_range, voxel_size):
    points = np.empty((coords.shape[0], 3), dtype=np.float32)
    points[:, 0] = pc_range[0] + (coords[:, 0].astype(np.float32) + 0.5) * voxel_size[0]
    points[:, 1] = pc_range[1] + (coords[:, 1].astype(np.float32) + 0.5) * voxel_size[1]
    points[:, 2] = pc_range[2] + (coords[:, 2].astype(np.float32) + 0.5) * voxel_size[2]
    return points


def load_dense_candidate_voxels(dense_candidate_dir, lidar_filename, occ_size, coord_transform):
    candidate_path = Path(dense_candidate_dir) / f"{Path(lidar_filename).name}.npy"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing dense candidate file: {candidate_path}")

    dense = np.load(candidate_path)
    if dense.ndim != 2 or dense.shape[1] != 4:
        raise ValueError(f"Expected dense candidate shape (N, 4), got {dense.shape}: {candidate_path}")

    coords = dense[:, :3].astype(np.int64)
    labels = dense[:, 3].astype(np.int16)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    coords = transform_dense_coords(coords, occ_size, coord_transform)
    in_range = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < occ_size[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < occ_size[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < occ_size[2])
        & (labels >= 0)
        & (labels <= 16)
    )
    coords = coords[in_range]
    labels = labels[in_range]
    return coords, labels, str(candidate_path), int(dense.shape[0]), int((~in_range).sum())


def load_dense_candidate_points(dense_candidate_dir, lidar_filename, pc_range, voxel_size, occ_size, coord_transform):
    coords, labels, candidate_path, raw_count, filtered = load_dense_candidate_voxels(
        dense_candidate_dir, lidar_filename, occ_size, coord_transform)
    points = voxel_centers_from_coords(coords, pc_range, voxel_size)
    return points, labels, candidate_path, raw_count, filtered


def load_dense_ray_source_points(points_dir, lidar_filename, coord_transform):
    point_path = Path(points_dir) / f"{Path(lidar_filename).name}.npy"
    if not point_path.exists():
        raise FileNotFoundError(f"Missing dense ray source points file: {point_path}")
    data = np.load(point_path)
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"Expected ray source points shape (N, 7), got {data.shape}: {point_path}")
    points = data[:, :3].astype(np.float32)
    origins = data[:, 3:6].astype(np.float32)
    labels = data[:, 6].astype(np.int16)
    points = transform_dense_metric_points(points, coord_transform)
    origins = transform_dense_metric_points(origins, coord_transform)
    valid = (labels >= 0) & (labels <= 16)
    return points[valid], origins[valid], labels[valid], str(point_path), int(data.shape[0]), int((~valid).sum())


def build_hit_grid_from_voxels(coords, labels, occ_size):
    raw_hit_grid = np.zeros(tuple(occ_size), dtype=bool)
    hit_label_counts = np.zeros((int(occ_size[0]), int(occ_size[1]), int(occ_size[2]), 17), dtype=np.uint16)
    for coord, label in zip(coords, labels):
        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
        raw_hit_grid[x, y, z] = True
        if 0 <= int(label) <= 16 and hit_label_counts[x, y, z, int(label)] < 65535:
            hit_label_counts[x, y, z, int(label)] += 1
    return raw_hit_grid, hit_label_counts


def majority_semantics(hit_label_counts, raw_hit_grid, free_grid):
    semantics = np.full(raw_hit_grid.shape, UNKNOWN_LABEL, dtype=np.uint8)
    semantics[free_grid] = FREE_LABEL

    hit_coords = np.argwhere(raw_hit_grid)
    conflict_voxels = 0
    unlabeled_hit_voxels = 0
    for x, y, z in hit_coords:
        counts = hit_label_counts[x, y, z]
        total = int(counts.sum())
        if total == 0:
            unlabeled_hit_voxels += 1
            semantics[x, y, z] = 0
            continue
        if int((counts > 0).sum()) > 1:
            conflict_voxels += 1
        semantics[x, y, z] = int(np.argmax(counts))

    export_semantics = semantics.copy()
    export_semantics[export_semantics == UNKNOWN_LABEL] = FREE_LABEL
    return export_semantics, conflict_voxels, unlabeled_hit_voxels


def resolve_lidar_path(dataroot, lidar_path):
    if os.path.isabs(lidar_path):
        return lidar_path
    normalized = lidar_path
    if normalized.startswith("data/nuscenes/"):
        normalized = normalized[len("data/nuscenes/") :]
    return os.path.join(dataroot, normalized)


def main():
    args = parse_args()
    dataroot = str(Path(args.dataroot).resolve())
    output_root = Path(args.output_root)
    gts_root = output_root / "gts"
    debug_root = output_root / "debug"
    ann_output = output_root / "bevdetv2-nuscenes_infos_stage2_train.pkl"
    stats_output = output_root / "stats_raycast.json"

    output_root.mkdir(parents=True, exist_ok=True)
    gts_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    ann_data, all_infos = load_infos(args.ann_file)
    infos = [info for info in all_infos if args.scene_name is None or info.get("scene_name") == args.scene_name]
    if args.max_frames is not None:
        infos = infos[: args.max_frames]

    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=False)
    learning_lut = load_learning_map(args.label_mapping)

    pc_range = np.asarray(args.pc_range, dtype=np.float32)
    voxel_size = np.asarray(args.voxel_size, dtype=np.float32)
    occ_size = np.asarray(args.occ_size, dtype=np.int64)
    skip_free_z_grazing_angle_rad = -1.0
    if args.skip_free_z_grazing_angle_deg >= 0.0:
        skip_free_z_grazing_angle_rad = float(args.skip_free_z_grazing_angle_deg) * float(np.pi) / 180.0

    converted_infos = []
    file_stats = []

    for frame_idx, info in enumerate(infos):
        sample = nusc.get("sample", info["token"])
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

        protected_grid, protected_box_count = build_protected_dynamic_grid(
            info,
            pc_range,
            voxel_size,
            occ_size,
            args.dynamic_classes,
            args.protected_box_margin,
        )

        points_list = []
        origins_list = []
        labels_list = []
        protect_rays_list = []
        sweep_stats = []
        self_filtered = 0
        raw_points_total = 0
        sweeps_with_lidarseg = 0
        dense_candidate_path = None
        dense_candidate_filtered = 0
        dense_ray_source_path = None
        dense_ray_source_filtered = 0
        dense_occupied_count = 0
        dense_ray_source_count = 0
        occupied_coords = None
        occupied_labels = None

        if args.ray_source == "dense-candidate":
            occupied_coords, occupied_labels, dense_candidate_path, dense_raw_count, dense_candidate_filtered = (
                load_dense_candidate_voxels(
                    args.dense_candidate_dir,
                    lidar_sd["filename"],
                    occ_size,
                    args.dense_coordinate_transform,
                )
            )
            dense_occupied_count = int(occupied_coords.shape[0])

            if args.dense_ray_source_points_dir:
                points, origins, labels, dense_ray_source_path, raw_points_total, dense_ray_source_filtered = (
                    load_dense_ray_source_points(
                        args.dense_ray_source_points_dir,
                        lidar_sd["filename"],
                        args.dense_coordinate_transform,
                    )
                )
                protect_rays = np.zeros(points.shape[0], dtype=bool)
                dense_ray_source_count = int(points.shape[0])
            else:
                ray_source_dir = args.dense_ray_source_dir or args.dense_candidate_dir
                ray_coords, labels, dense_ray_source_path, raw_points_total, dense_ray_source_filtered = (
                    load_dense_candidate_voxels(
                        ray_source_dir,
                        lidar_sd["filename"],
                        occ_size,
                        args.dense_coordinate_transform,
                    )
                )
                dense_ray_source_count = int(ray_coords.shape[0])
                points = voxel_centers_from_coords(ray_coords, pc_range, voxel_size)
                origins = np.zeros((points.shape[0], 3), dtype=np.float32)
                protect_rays = np.zeros(points.shape[0], dtype=bool)
            sweep_sds = [lidar_sd]
            sweep_stats.append(
                {
                    "sweep_idx": 0,
                    "sample_data_token": lidar_sd["token"],
                    "filename": lidar_sd["filename"],
                    "is_key_frame": bool(lidar_sd["is_key_frame"]),
                    "has_lidarseg": False,
                    "lidarseg_path": None,
                    "points_raw": int(dense_raw_count),
                    "self_filtered_points": 0,
                    "points_after_filter": int(points.shape[0]),
                    "origin_in_key_lidar": [0.0, 0.0, 0.0],
                    "dense_candidate_path": dense_candidate_path,
                    "dense_candidate_filtered": int(dense_candidate_filtered),
                    "dense_ray_source_path": dense_ray_source_path,
                    "dense_ray_source_filtered": int(dense_ray_source_filtered),
                }
            )
        else:
            key_lidar_from_global = np.linalg.inv(lidar_to_global(nusc, lidar_sd)).astype(np.float32)
            sweep_sds = collect_lidar_sweeps(
                nusc, lidar_sd, args.num_sweeps, args.sweep_direction, args.scene_sweep_stride
            )

            for sweep_idx, sweep_sd in enumerate(sweep_sds):
                points_local, labels, sweep_self_filtered, lidarseg, labels_path, raw_count = (
                    load_sweep_points_and_labels(
                        nusc, dataroot, sweep_sd, learning_lut, args.self_range
                    )
                )
                self_filtered += sweep_self_filtered
                raw_points_total += raw_count
                if lidarseg is not None:
                    sweeps_with_lidarseg += 1

                key_lidar_from_sweep_lidar = (
                    key_lidar_from_global @ lidar_to_global(nusc, sweep_sd)
                ).astype(np.float32)
                points_in_key = transform_points(points_local, key_lidar_from_sweep_lidar)
                origin_in_key = transform_origin(key_lidar_from_sweep_lidar)
                origins = np.repeat(origin_in_key[None, :], points_in_key.shape[0], axis=0)
                protect_rays = np.full(
                    points_in_key.shape[0],
                    bool(args.truncate_protected_free and sweep_sd["token"] != lidar_sd["token"]),
                    dtype=bool,
                )

                points_list.append(points_in_key)
                origins_list.append(origins)
                labels_list.append(labels)
                protect_rays_list.append(protect_rays)
                sweep_stats.append(
                    {
                        "sweep_idx": sweep_idx,
                        "sample_data_token": sweep_sd["token"],
                        "filename": sweep_sd["filename"],
                        "is_key_frame": bool(sweep_sd["is_key_frame"]),
                        "has_lidarseg": lidarseg is not None,
                        "lidarseg_path": None if labels_path is None else os.path.relpath(labels_path, dataroot),
                        "points_raw": int(raw_count),
                        "self_filtered_points": int(sweep_self_filtered),
                        "points_after_filter": int(points_in_key.shape[0]),
                        "origin_in_key_lidar": [float(v) for v in origin_in_key],
                    }
                )

            points = np.concatenate(points_list, axis=0)
            origins = np.concatenate(origins_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            protect_rays = np.concatenate(protect_rays_list, axis=0)

        if args.ray_source == "dense-candidate" and args.surface_free_distance > 0.0:
            free_grid, raw_hit_grid, hit_label_counts, ray_stats = _raycast_surface_points(
                origins,
                points,
                labels,
                pc_range,
                voxel_size,
                occ_size,
                float(args.surface_free_distance),
                int(args.skip_free_z_min),
                int(args.skip_free_z_max),
                float(skip_free_z_grazing_angle_rad),
                float(args.skip_free_z_min_ray_length),
            )
        else:
            if args.ray_traversal == "occ3d-point-to-origin":
                free_grid, raw_hit_grid, hit_label_counts, ray_stats = _raycast_points_occ3d_point_to_origin(
                    origins,
                    points,
                    labels,
                    protect_rays,
                    protected_grid,
                    pc_range,
                    voxel_size,
                    occ_size,
                    float(args.lidar_max_range),
                    int(args.skip_free_z_min),
                    int(args.skip_free_z_max),
                    float(skip_free_z_grazing_angle_rad),
                    float(args.skip_free_z_min_ray_length),
                )
            else:
                free_grid, raw_hit_grid, hit_label_counts, ray_stats = _raycast_points(
                    origins,
                    points,
                    labels,
                    protect_rays,
                    protected_grid,
                    pc_range,
                    voxel_size,
                    occ_size,
                    float(args.lidar_max_range),
                    int(args.skip_free_z_min),
                    int(args.skip_free_z_max),
                    float(skip_free_z_grazing_angle_rad),
                    float(args.skip_free_z_min_ray_length),
                )
        if args.ray_source == "dense-candidate" and (args.dense_ray_source_dir or args.dense_ray_source_points_dir):
            raw_hit_grid, hit_label_counts = build_hit_grid_from_voxels(
                occupied_coords, occupied_labels, occ_size)
        mask_lidar = free_grid | raw_hit_grid
        semantics, conflict_voxels, unlabeled_hit_voxels = majority_semantics(
            hit_label_counts, raw_hit_grid, free_grid
        )

        # Stage 2 has no image visibility reasoning yet. Use a conservative
        # placeholder equal to mask_lidar for loader smoke tests only.
        mask_camera = mask_lidar.copy()

        scene_name = info["scene_name"]
        token = info["token"]
        occ_dir = gts_root / scene_name / token
        debug_dir = debug_root / scene_name / token
        occ_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            occ_dir / "labels.npz",
            semantics=semantics,
            mask_lidar=mask_lidar,
            mask_camera=mask_camera,
        )
        np.savez_compressed(
            debug_dir / "raycast_debug.npz",
            free_grid=free_grid,
            raw_hit_grid=raw_hit_grid,
            mask_lidar=mask_lidar,
            hit_label_counts=hit_label_counts,
        )

        updated_info = dict(info)
        updated_info["occ_path"] = str(occ_dir.resolve())
        converted_infos.append(updated_info)

        stat_names = [
            "rays_total",
            "rays_intersect",
            "rays_endpoint_inside",
            "rays_endpoint_outside",
            "rays_no_intersection",
            "free_voxels_written",
            "hit_writes",
            "protected_truncations",
            "max_range_clipped",
        ]
        ray_stat_dict = {name: int(value) for name, value in zip(stat_names, ray_stats)}
        frame_stat = {
            "frame_idx": frame_idx,
            "scene_name": scene_name,
            "token": token,
            "lidar_sample_data_token": lidar_sd["token"],
            "lidar_path": lidar_sd["filename"],
            "lidarseg_path": sweep_stats[0]["lidarseg_path"],
            "num_sweeps_requested": int(args.num_sweeps),
            "num_sweeps_used": int(len(sweep_sds)),
            "sweep_direction": args.sweep_direction,
            "scene_sweep_stride": int(args.scene_sweep_stride),
            "sweeps_with_lidarseg": int(sweeps_with_lidarseg),
            "ray_source": args.ray_source,
            "dense_candidate_path": dense_candidate_path,
            "dense_candidate_filtered": int(dense_candidate_filtered),
            "dense_ray_source_dir": args.dense_ray_source_dir,
            "dense_ray_source_points_dir": args.dense_ray_source_points_dir,
            "dense_ray_source_path": dense_ray_source_path,
            "dense_ray_source_filtered": int(dense_ray_source_filtered),
            "dense_occupied_count": int(dense_occupied_count),
            "dense_ray_source_count": int(dense_ray_source_count),
            "dense_coordinate_transform": args.dense_coordinate_transform,
            "ray_traversal": args.ray_traversal,
            "truncate_protected_free": bool(args.truncate_protected_free),
            "protected_box_count": int(protected_box_count),
            "protected_voxel_count": int(protected_grid.sum()),
            "protected_box_margin": float(args.protected_box_margin),
            "dynamic_classes": args.dynamic_classes,
            "lidar_max_range": float(args.lidar_max_range),
            "surface_free_distance": float(args.surface_free_distance),
            "skip_free_z_min": int(args.skip_free_z_min),
            "skip_free_z_max": int(args.skip_free_z_max),
            "skip_free_z_grazing_angle_deg": float(args.skip_free_z_grazing_angle_deg),
            "skip_free_z_min_ray_length": float(args.skip_free_z_min_ray_length),
            "points_raw": int(raw_points_total),
            "self_filtered_points": int(self_filtered),
            "points_after_filter": int(points.shape[0]),
            "free_count": int((mask_lidar & (semantics == FREE_LABEL) & ~raw_hit_grid).sum()),
            "occupied_count": int(raw_hit_grid.sum()),
            "observed_count": int(mask_lidar.sum()),
            "unknown_count": int((~mask_lidar).sum()),
            "semantic_conflict_voxels": int(conflict_voxels),
            "unlabeled_hit_voxels": int(unlabeled_hit_voxels),
            "ray_stats": ray_stat_dict,
            "sweeps": sweep_stats,
            "occ_path": str(occ_dir.resolve()),
            "debug_path": str((debug_dir / "raycast_debug.npz").resolve()),
        }
        file_stats.append(frame_stat)
        print(
            f"[{frame_idx + 1}/{len(infos)}] {scene_name} {token}: "
            f"occupied={frame_stat['occupied_count']} free={frame_stat['free_count']} "
            f"unknown={frame_stat['unknown_count']}"
        )

    save_ann_with_stage2_paths(ann_data, converted_infos, ann_output)

    summary = {
        "stage": "Stage 2: LiDAR ray casting",
        "ray_source": args.ray_source,
        "grid": {
            "point_cloud_range": [float(v) for v in pc_range],
            "voxel_size": [float(v) for v in voxel_size],
            "occ_size": [int(v) for v in occ_size],
        },
        "source_grid_note": (
            "This output uses FlashOCC/Occ3D grid directly. It does not reuse "
            "SurroundOcc sparse voxel indices from pc_range [-50,-50,-5,50,50,3] "
            "and voxel_size 0.5."
        ),
        "dataroot": dataroot,
        "ann_file": str(args.ann_file),
        "dense_candidate_dir": str(args.dense_candidate_dir),
        "dense_ray_source_dir": args.dense_ray_source_dir,
        "dense_ray_source_points_dir": args.dense_ray_source_points_dir,
        "dense_coordinate_transform": args.dense_coordinate_transform,
        "ray_traversal": args.ray_traversal,
        "ann_output": str(ann_output),
        "gts_root": str(gts_root),
        "debug_root": str(debug_root),
        "num_frames": len(file_stats),
        "sweep_config": {
            "num_sweeps": int(args.num_sweeps),
            "sweep_direction": args.sweep_direction,
            "scene_sweep_stride": int(args.scene_sweep_stride),
            "note": "Each sweep keeps its own LiDAR origin transformed into the target keyframe LiDAR frame.",
        },
        "free_path_constraints": {
            "truncate_protected_free": bool(args.truncate_protected_free),
            "dynamic_classes": args.dynamic_classes,
            "protected_box_margin": float(args.protected_box_margin),
            "lidar_max_range": float(args.lidar_max_range),
            "surface_free_distance": float(args.surface_free_distance),
            "skip_free_z_min": int(args.skip_free_z_min),
            "skip_free_z_max": int(args.skip_free_z_max),
            "skip_free_z_grazing_angle_deg": float(args.skip_free_z_grazing_angle_deg),
            "skip_free_z_min_ray_length": float(args.skip_free_z_min_ray_length),
        },
        "mask_camera_note": "Temporary placeholder equal to mask_lidar; real camera mask belongs to Stage 4.",
        "totals": {
            "occupied_count": int(sum(item["occupied_count"] for item in file_stats)),
            "free_count": int(sum(item["free_count"] for item in file_stats)),
            "observed_count": int(sum(item["observed_count"] for item in file_stats)),
            "unknown_count": int(sum(item["unknown_count"] for item in file_stats)),
            "self_filtered_points": int(sum(item["self_filtered_points"] for item in file_stats)),
            "protected_truncations": int(sum(item["ray_stats"]["protected_truncations"] for item in file_stats)),
            "max_range_clipped": int(sum(item["ray_stats"]["max_range_clipped"] for item in file_stats)),
        },
        "files": file_stats,
    }
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote Stage 2 GT root: {gts_root}")
    print(f"Wrote Stage 2 ann pkl: {ann_output}")
    print(f"Wrote stats: {stats_output}")


if __name__ == "__main__":
    main()
