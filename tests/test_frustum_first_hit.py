import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "occ3d_stage4"))

from analyze_frustum_first_hit import (  # noqa: E402
    frustum_union,
    lidar_points_to_camera,
    project_camera_points,
)


def test_lidar_to_camera_inverts_sensor2lidar_transform():
    rotation = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    lidar_points = np.array([[1.0, 2.0, 5.0], [2.0, 2.0, 5.0]])

    camera_points = lidar_points_to_camera(lidar_points, rotation, translation)

    np.testing.assert_allclose(camera_points, np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]))


def test_projection_and_frustum_union_handle_depth_and_image_bounds():
    points = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    intrinsic = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    visible, depth = project_camera_points(points, intrinsic, width=2, height=2, depth_max=5.0)
    union, min_depth, per_camera = frustum_union(
        points,
        [{
            "name": "CAM_TEST",
            "intrinsic": intrinsic,
            "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.zeros(3),
        }],
        width=2,
        height=2,
        depth_max=5.0,
    )

    np.testing.assert_array_equal(visible, np.array([True, False, False]))
    np.testing.assert_array_equal(union, visible)
    assert min_depth[0] == 2.0
    assert np.isinf(min_depth[1])
    assert per_camera["CAM_TEST"] == 1
    np.testing.assert_array_equal(depth, np.array([2.0, 2.0, -1.0]))
