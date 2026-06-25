import sys
from pathlib import Path

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "generate_occupancy_nuscenes"))

from surface_completion import (  # noqa: E402
    filter_mesh_components,
    fill_flat_height_holes,
    mesh_topology_stats,
    orient_normals_toward_origins,
    points_to_unique_voxels,
    sample_mesh_surface_points,
    surface_voxels_from_mesh,
    voxelize_triangle_mesh,
)


def test_normals_are_oriented_toward_each_points_origin():
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    origins = np.zeros((2, 3), dtype=np.float64)

    oriented, flipped = orient_normals_toward_origins(points, normals, origins)

    np.testing.assert_array_equal(oriented, np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert flipped == 2


def test_mesh_topology_and_component_filter_remove_only_small_fragment():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [5, 5, 0], [6, 5, 0], [5, 6, 0],
            ],
            dtype=np.float64,
        )
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int32)
    )

    before = mesh_topology_stats(mesh)
    filtered, filter_stats = filter_mesh_components(mesh, min_triangles=2)
    after = mesh_topology_stats(filtered)

    assert before["component_count"] == 2
    assert before["component_triangle_counts"] == [2, 1]
    assert after["component_count"] == 1
    assert after["triangle_count"] == 2
    assert filter_stats["removed_components"] == 1
    assert filter_stats["removed_triangles"] == 1


def test_sample_mesh_surface_points_covers_triangle_interior():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    sampled = sample_mesh_surface_points(vertices, triangles, spacing=0.4)

    assert sampled.shape[0] > vertices.shape[0]
    assert np.any(np.linalg.norm(sampled - np.array([0.8, 0.8, 0.0]), axis=1) < 0.3)


def test_triangle_voxelization_marks_cells_without_mesh_vertices():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array(
            [[0.1, 0.1, 0.2], [1.9, 0.1, 0.2], [1.9, 1.9, 0.2], [0.1, 1.9, 0.2]],
            dtype=np.float64,
        )
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    )

    coords = voxelize_triangle_mesh(
        mesh,
        pc_range=np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.0]),
        voxel_size=np.array([0.5, 0.5, 0.5]),
        occ_size=np.array([4, 4, 2]),
    )

    assert any(np.array_equal(coord, np.array([1, 1, 0])) for coord in coords)
    assert coords.shape[0] > 4


def test_surface_modes_share_target_grid_contract():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0.1, 0.1, 0.2], [1.9, 0.1, 0.2], [0.1, 1.9, 0.2]])
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32))
    kwargs = dict(
        pc_range=np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.0]),
        voxel_size=np.array([0.5, 0.5, 0.5]),
        occ_size=np.array([4, 4, 2]),
        sample_spacing=0.25,
    )

    vertices, vertex_stats = surface_voxels_from_mesh(mesh, mode="vertices", **kwargs)
    uniform, uniform_stats = surface_voxels_from_mesh(mesh, mode="uniform", **kwargs)
    triangle, triangle_stats = surface_voxels_from_mesh(mesh, mode="triangle", **kwargs)

    assert vertices.shape[1] == uniform.shape[1] == triangle.shape[1] == 3
    assert uniform.shape[0] >= vertices.shape[0]
    assert triangle.shape[0] >= vertices.shape[0]
    assert vertex_stats["mode"] == "vertices"
    assert uniform_stats["sample_count"] > 0
    assert triangle_stats["triangle_count"] == 1


def test_points_to_unique_voxels_deduplicates_and_clips():
    points = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 0.1], [2.1, 0.0, 0.0]],
        dtype=np.float64,
    )

    coords = points_to_unique_voxels(
        points,
        pc_range=np.array([0.0, 0.0, 0.0, 2.0, 2.0, 1.0]),
        voxel_size=np.array([1.0, 1.0, 1.0]),
        occ_size=np.array([2, 2, 1]),
    )

    np.testing.assert_array_equal(coords, np.array([[0, 0, 0], [1, 1, 0]]))


def test_flat_height_fill_closes_supported_center_hole_only():
    flat_ring = np.array(
        [
            [0, 0, 1, 11], [1, 0, 1, 11], [2, 0, 1, 11],
            [0, 1, 1, 11],                 [2, 1, 1, 11],
            [0, 2, 1, 11], [1, 2, 1, 11], [2, 2, 1, 11],
        ],
        dtype=np.int64,
    )
    non_flat = np.array([[5, 5, 1, 15]], dtype=np.int64)

    filled, stats = fill_flat_height_holes(
        np.concatenate([flat_ring, non_flat], axis=0),
        occ_size=np.array([8, 8, 4]),
        radius=1,
        min_neighbors=5,
        max_z_spread=1,
        flat_labels=(11, 12, 13, 14),
    )

    assert any(np.array_equal(row, np.array([1, 1, 1, 11])) for row in filled)
    assert not any(np.array_equal(row[:3], np.array([4, 5, 1])) for row in filled)
    assert stats["added_total"] == 1
    assert stats["added_by_class"]["11"] == 1
