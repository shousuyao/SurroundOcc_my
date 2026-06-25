import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "generate_occupancy_nuscenes"))

from semantic_reconstruction import merge_group_voxels, semantic_group  # noqa: E402


def test_semantic_group_routes_geometry_types():
    assert semantic_group(0) == "other"
    assert semantic_group(4) == "dynamic"
    assert semantic_group(11) == "flat"
    assert semantic_group(14) == "flat"
    assert semantic_group(15) == "manmade"
    assert semantic_group(16) == "vegetation"


def test_merge_group_voxels_uses_deterministic_geometry_priority():
    outputs = {
        "vegetation": np.array([[1, 1, 1, 16], [3, 3, 1, 16]]),
        "manmade": np.array([[2, 2, 1, 15]]),
        "flat": np.array([[1, 1, 1, 11], [2, 2, 1, 13]]),
        "dynamic": np.array([[2, 2, 1, 4]]),
    }

    merged, conflicts = merge_group_voxels(outputs)
    lookup = {tuple(row[:3]): int(row[3]) for row in merged}

    assert lookup[(1, 1, 1)] == 11
    assert lookup[(2, 2, 1)] == 4
    assert lookup[(3, 3, 1)] == 16
    assert conflicts == 2
