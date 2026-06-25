"""Class-aware occupancy surface reconstruction for controlled ablations."""

import numpy as np

from surface_completion import (
    fill_flat_height_holes,
    filter_mesh_components,
    mesh_topology_stats,
    orient_normals_toward_origins,
    points_to_unique_voxels,
    surface_voxels_from_mesh,
)


GROUP_PRIORITY = {
    "other": 0,
    "vegetation": 1,
    "manmade": 2,
    "flat": 3,
    "dynamic": 4,
}


def semantic_group(label):
    label = int(label)
    if 1 <= label <= 10:
        return "dynamic"
    if 11 <= label <= 14:
        return "flat"
    if label == 15:
        return "manmade"
    if label == 16:
        return "vegetation"
    return "other"


def merge_group_voxels(group_outputs):
    merged = {}
    conflict_coords = set()
    ordered = sorted(group_outputs, key=lambda name: GROUP_PRIORITY[name])
    for group_name in ordered:
        rows = np.asarray(group_outputs[group_name], dtype=np.int64)
        for row in rows:
            coord = tuple(int(value) for value in row[:3])
            label = int(row[3])
            if coord in merged and merged[coord] != label:
                conflict_coords.add(coord)
            merged[coord] = label
    output = np.asarray(
        [[*coord, label] for coord, label in sorted(merged.items())],
        dtype=np.int64,
    )
    if output.size == 0:
        output = np.zeros((0, 4), dtype=np.int64)
    return output, int(len(conflict_coords))


def _voxelize_points_by_label(points_with_semantic, pc_range, voxel_size, occ_size):
    outputs = []
    labels = points_with_semantic[:, 3].astype(np.int64)
    for label in np.unique(labels):
        if label < 0 or label > 16:
            continue
        coords = points_to_unique_voxels(
            points_with_semantic[labels == label, :3], pc_range, voxel_size, occ_size
        )
        if coords.shape[0]:
            outputs.append(np.concatenate([coords, np.full((coords.shape[0], 1), label)], axis=1))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 4), dtype=np.int64)


def reconstruct_semantic_groups(
    points_with_semantic,
    origins,
    pc_range,
    voxel_size,
    occ_size,
    poisson_depth=10,
    min_density=0.1,
    max_nn=20,
    min_static_points=100,
    surface_mode="triangle",
    sample_spacing=0.2,
    min_component_triangles=0,
):
    import open3d as o3d

    points_with_semantic = np.asarray(points_with_semantic, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    labels = points_with_semantic[:, 3].astype(np.int64)
    group_outputs = {}
    stats = {"groups": {}}

    for group_name in ("other", "dynamic", "flat"):
        group_mask = np.array([semantic_group(label) == group_name for label in labels])
        voxels = _voxelize_points_by_label(
            points_with_semantic[group_mask], pc_range, voxel_size, occ_size
        )
        fill_stats = None
        if group_name == "flat":
            voxels, fill_stats = fill_flat_height_holes(
                voxels,
                occ_size=occ_size,
                radius=1,
                min_neighbors=5,
                max_z_spread=1,
                flat_labels=(11, 12, 13, 14),
            )
        group_outputs[group_name] = voxels
        stats["groups"][group_name] = {
            "source_points": int(group_mask.sum()),
            "output_voxels": int(voxels.shape[0]),
            "flat_fill": fill_stats,
            "method": "source-voxel-height-fill" if group_name == "flat" else "source-voxel",
        }

    for label, group_name in ((15, "manmade"), (16, "vegetation")):
        mask = labels == label
        source_points = points_with_semantic[mask, :3]
        source_origins = origins[mask]
        group_stat = {"source_points": int(mask.sum()), "method": "poisson"}
        if source_points.shape[0] < min_static_points:
            coords = points_to_unique_voxels(source_points, pc_range, voxel_size, occ_size)
            group_stat["method"] = "source-voxel-fallback"
            group_stat["fallback_reason"] = "insufficient_points"
        else:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(source_points)
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(int(max_nn)))
            normals, flips = orient_normals_toward_origins(
                source_points, np.asarray(cloud.normals), source_origins)
            cloud.normals = o3d.utility.Vector3dVector(normals)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=int(poisson_depth), n_threads=8
            )
            densities = np.asarray(densities)
            if min_density:
                mesh.remove_vertices_by_mask(densities < np.quantile(densities, min_density))
            topology_before = mesh_topology_stats(mesh)
            mesh, filter_stats = filter_mesh_components(mesh, min_component_triangles)
            coords, surface_stats = surface_voxels_from_mesh(
                mesh,
                mode=surface_mode,
                pc_range=pc_range,
                voxel_size=voxel_size,
                occ_size=occ_size,
                sample_spacing=sample_spacing,
            )
            group_stat.update({
                "normal_flips": int(flips),
                "topology_before": topology_before,
                "component_filter": filter_stats,
                "surface": surface_stats,
            })
        group_outputs[group_name] = np.concatenate(
            [coords.astype(np.int64), np.full((coords.shape[0], 1), label, dtype=np.int64)],
            axis=1,
        )
        group_stat["output_voxels"] = int(coords.shape[0])
        stats["groups"][group_name] = group_stat

    merged, conflicts = merge_group_voxels(group_outputs)
    stats["conflicts"] = int(conflicts)
    stats["output_voxels"] = int(merged.shape[0])
    return merged, stats
