"""Surface voxelization and conservative flat-class hole filling helpers."""

import numpy as np


def orient_normals_toward_origins(points, normals, origins):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64).copy()
    origins = np.asarray(origins, dtype=np.float64)
    if points.shape != normals.shape or points.shape != origins.shape:
        raise ValueError("points, normals, and origins must have identical Nx3 shapes")
    view_directions = origins - points
    flip_mask = np.einsum("ij,ij->i", normals, view_directions) < 0.0
    normals[flip_mask] *= -1.0
    return normals, int(flip_mask.sum())


def mesh_topology_stats(mesh):
    triangle_count = len(mesh.triangles)
    if triangle_count:
        _, component_counts, _ = mesh.cluster_connected_triangles()
        counts = sorted((int(value) for value in component_counts), reverse=True)
    else:
        counts = []
    return {
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(triangle_count),
        "component_count": int(len(counts)),
        "component_triangle_counts": counts,
        "is_watertight": bool(mesh.is_watertight()) if triangle_count else False,
        "is_edge_manifold": bool(mesh.is_edge_manifold()) if triangle_count else False,
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()) if triangle_count else False,
    }


def filter_mesh_components(mesh, min_triangles):
    from copy import deepcopy

    filtered = deepcopy(mesh)
    if min_triangles <= 1 or len(filtered.triangles) == 0:
        return filtered, {
            "removed_components": 0,
            "removed_triangles": 0,
            "component_count_before": None,
            "component_count_after": None,
            "largest_component_triangles": None,
        }
    labels, counts, _ = filtered.cluster_connected_triangles()
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    remove_mask = counts[labels] < int(min_triangles)
    removed_components = int(np.unique(labels[remove_mask]).size) if np.any(remove_mask) else 0
    removed_triangles = int(remove_mask.sum())
    if removed_triangles:
        filtered.remove_triangles_by_mask(remove_mask.tolist())
        filtered.remove_unreferenced_vertices()
    return filtered, {
        "removed_components": removed_components,
        "removed_triangles": removed_triangles,
        "component_count_before": int(counts.size),
        "component_count_after": int(counts.size - removed_components),
        "largest_component_triangles": int(counts.max()) if counts.size else 0,
    }


def points_to_unique_voxels(points, pc_range, voxel_size, occ_size):
    points = np.asarray(points, dtype=np.float64)
    pc_range = np.asarray(pc_range, dtype=np.float64)
    voxel_size = np.asarray(voxel_size, dtype=np.float64)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    coords = np.floor((points[:, :3] - pc_range[:3]) / voxel_size).astype(np.int64)
    valid = np.all((coords >= 0) & (coords < occ_size), axis=1)
    return np.unique(coords[valid], axis=0)


def sample_mesh_surface_points(vertices, triangles, spacing):
    """Deterministically sample triangle interiors with a maximum edge step."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if triangles.size == 0:
        return vertices.copy()

    sampled = []
    for tri in triangles:
        a, b, c = vertices[tri]
        max_edge = max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
        subdivisions = max(1, int(np.ceil(max_edge / float(spacing))))
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1 - i):
                wb = i / subdivisions
                wc = j / subdivisions
                sampled.append(a + wb * (b - a) + wc * (c - a))
    return np.asarray(sampled, dtype=np.float64)


def voxelize_triangle_mesh(mesh, pc_range, voxel_size, occ_size):
    """Voxelize triangle surfaces with Open3D and map centers to the target grid."""
    import open3d as o3d

    voxel_size = np.asarray(voxel_size, dtype=np.float64)
    if not np.allclose(voxel_size, voxel_size[0]):
        raise ValueError("Open3D triangle voxelization requires isotropic voxel_size")
    pc_range = np.asarray(pc_range, dtype=np.float64)
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        float(voxel_size[0]),
        pc_range[:3],
        pc_range[3:],
    )
    voxels = grid.get_voxels()
    if not voxels:
        return np.zeros((0, 3), dtype=np.int64)
    centers = np.asarray(
        [grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels],
        dtype=np.float64,
    )
    return points_to_unique_voxels(centers, pc_range, voxel_size, occ_size)


def surface_voxels_from_mesh(
    mesh,
    mode,
    pc_range,
    voxel_size,
    occ_size,
    sample_spacing=0.2,
):
    """Convert a triangle mesh surface to target-grid voxel coordinates."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    stats = {
        "mode": str(mode),
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "sample_count": 0,
    }
    if mode == "vertices":
        points = vertices
    elif mode == "uniform":
        if sample_spacing <= 0:
            raise ValueError("sample_spacing must be positive")
        area = float(mesh.get_surface_area())
        sample_count = max(int(vertices.shape[0]), int(np.ceil(2.0 * area / (sample_spacing ** 2))))
        if sample_count == 0:
            points = vertices
        else:
            import open3d as o3d

            o3d.utility.random.seed(0)
            sampled = mesh.sample_points_uniformly(number_of_points=sample_count)
            points = np.asarray(sampled.points, dtype=np.float64)
        stats["sample_count"] = int(points.shape[0])
        stats["surface_area"] = area
    elif mode == "triangle":
        coords = voxelize_triangle_mesh(mesh, pc_range, voxel_size, occ_size)
        stats["voxel_count"] = int(coords.shape[0])
        return coords, stats
    else:
        raise ValueError("Unsupported surface mode: {}".format(mode))

    coords = points_to_unique_voxels(points, pc_range, voxel_size, occ_size)
    stats["voxel_count"] = int(coords.shape[0])
    return coords, stats


def _deduplicate_semantic_voxels(voxels):
    voxels = np.asarray(voxels, dtype=np.int64)
    if voxels.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    coords, inverse = np.unique(voxels[:, :3], axis=0, return_inverse=True)
    labels = np.zeros(coords.shape[0], dtype=np.int64)
    for idx in range(coords.shape[0]):
        values = voxels[inverse == idx, 3]
        labels[idx] = np.bincount(values, minlength=17).argmax()
    return np.concatenate([coords, labels[:, None]], axis=1)


def fill_flat_height_holes(
    voxels_with_semantic,
    occ_size,
    radius=1,
    min_neighbors=5,
    max_z_spread=1,
    flat_labels=(11, 12, 13, 14),
):
    """Fill supported one-cell XY holes in class-specific flat height maps."""
    voxels = _deduplicate_semantic_voxels(voxels_with_semantic)
    occ_size = np.asarray(occ_size, dtype=np.int64)
    occupied = {tuple(row[:3]) for row in voxels}
    additions = []
    added_by_class = {str(int(label)): 0 for label in flat_labels}

    for label in flat_labels:
        class_voxels = voxels[voxels[:, 3] == int(label)]
        if class_voxels.shape[0] < min_neighbors:
            continue
        heights = {}
        for x, y, z, _ in class_voxels:
            heights.setdefault((int(x), int(y)), []).append(int(z))
        height_map = {xy: int(np.rint(np.median(zs))) for xy, zs in heights.items()}

        candidates = set()
        for x, y in height_map:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    tx, ty = x + dx, y + dy
                    if 0 <= tx < occ_size[0] and 0 <= ty < occ_size[1] and (tx, ty) not in height_map:
                        candidates.add((tx, ty))

        for x, y in sorted(candidates):
            neighbor_z = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    z = height_map.get((x + dx, y + dy))
                    if z is not None:
                        neighbor_z.append(z)
            if len(neighbor_z) < min_neighbors:
                continue
            if max(neighbor_z) - min(neighbor_z) > max_z_spread:
                continue
            z = int(np.rint(np.median(neighbor_z)))
            coord = (x, y, z)
            if z < 0 or z >= occ_size[2] or coord in occupied:
                continue
            additions.append([x, y, z, int(label)])
            occupied.add(coord)
            added_by_class[str(int(label))] += 1

    if additions:
        voxels = _deduplicate_semantic_voxels(
            np.concatenate([voxels, np.asarray(additions, dtype=np.int64)], axis=0)
        )
    return voxels, {
        "added_total": int(len(additions)),
        "added_by_class": added_by_class,
    }
