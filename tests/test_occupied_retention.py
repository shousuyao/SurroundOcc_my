import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "occ3d_stage4"))

from analyze_occupied_retention import aggregate_labels, count_label_arrays  # noqa: E402


def test_count_label_arrays_tracks_global_lidar_and_camera_by_class():
    semantics = np.array([[[4, 11], [17, 15]], [[4, 17], [16, 17]]], dtype=np.uint8)
    mask_lidar = np.array([[[1, 1], [1, 1]], [[1, 1], [0, 0]]], dtype=bool)
    mask_camera = np.array([[[1, 0], [1, 1]], [[0, 1], [0, 0]]], dtype=bool)

    result = count_label_arrays(semantics, mask_lidar, mask_camera, num_classes=18)

    assert result["global"][4] == 2
    assert result["lidar"][4] == 2
    assert result["camera"][4] == 1
    assert result["global"][16] == 1
    assert result["lidar"][16] == 0
    assert result["camera"][15] == 1


def test_aggregate_labels_reports_nonfree_retention(tmp_path):
    frame_dir = tmp_path / "gts" / "scene-test" / "token"
    frame_dir.mkdir(parents=True)
    semantics = np.array([[[4, 4, 11, 17]]], dtype=np.uint8)
    mask_lidar = np.array([[[1, 1, 1, 1]]], dtype=np.uint8)
    mask_camera = np.array([[[1, 0, 1, 1]]], dtype=np.uint8)
    np.savez_compressed(
        frame_dir / "labels.npz",
        semantics=semantics,
        mask_lidar=mask_lidar,
        mask_camera=mask_camera,
    )

    report = aggregate_labels([frame_dir / "labels.npz"], num_classes=18)

    assert report["frames"] == 1
    assert report["totals"]["global_occupied"] == 3
    assert report["totals"]["camera_occupied"] == 2
    assert report["totals"]["camera_over_global_occupied"] == 2 / 3
    car = report["per_class"][4]
    assert car["name"] == "car"
    assert car["camera_over_global"] == 0.5
