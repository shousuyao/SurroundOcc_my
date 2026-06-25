# Occupied Surface Quality Short-Term Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve generated occupancy surface continuity with mesh-surface voxelization and flat-class height filling, then quantify where occupied voxels survive into Stage 4 camera visibility.

**Architecture:** Add pure NumPy geometry helpers in a focused module, invoke them from the existing nuScenes generator behind explicit CLI/config options, and add a standalone statistics tool that compares dense candidates, Stage 3/4 labels, and official Occ3D when available. Keep the current `vertices` behavior as the default and write every ablation to a separate output root.

**Tech Stack:** Python 3.8, NumPy, Open3D, pytest/unittest, existing Stage 2/4 scripts.

---

### Task 1: Surface Geometry Helpers

**Files:**
- Create: `tools/generate_occupancy_nuscenes/surface_completion.py`
- Create: `tests/test_surface_completion.py`

- [ ] Write failing tests for deterministic triangle sampling, triangle-mesh voxel extraction, voxel deduplication, and flat height-field hole filling constrained to flat semantic classes.
- [ ] Run `python -m pytest tests/test_surface_completion.py -q` and verify failures are caused by missing helpers.
- [ ] Implement `sample_mesh_surface_points`, `voxelize_triangle_mesh`, `points_to_unique_voxels`, and `fill_flat_height_holes` as pure NumPy/Open3D-compatible helpers.
- [ ] Run the focused tests and verify they pass.

### Task 2: Generator Integration

**Files:**
- Modify: `tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py`
- Modify: `tools/generate_occupancy_nuscenes/config.yaml`
- Modify: `tests/test_surface_completion.py`

- [ ] Add failing parser/config tests for `--surface-mode {vertices,uniform,triangle}` and flat-fill parameters.
- [ ] Add `uniform` mesh surface sampling and Open3D triangle-mesh voxelization while retaining `vertices` as the default.
- [ ] Add optional flat height-field filling after semantic assignment, restricted to labels `11,12,13,14` and a configurable XY radius.
- [ ] Save per-frame generation statistics next to candidates, including source counts and class histograms.
- [ ] Run focused tests and `py_compile` for modified modules.

### Task 3: Stage Retention Diagnostics

**Files:**
- Create: `tools/occ3d_stage4/analyze_occupied_retention.py`
- Create: `tests/test_occupied_retention.py`

- [ ] Write failing tests for class counts and global→LiDAR→camera retention ratios.
- [ ] Implement aggregation over Stage4 `labels.npz`, optional dense candidate files, and optional official GT.
- [ ] Emit JSON with per-class global occupied, LiDAR occupied, camera occupied, and retention ratios.
- [ ] Run focused tests and validate the tool on the existing 81-frame Stage4 val output.

### Task 4: Single-Scene Ablation

**Files:**
- Generate: `data/GT_occupancy_surface_ablation/`
- Modify: `docs/occ3d_flashocc_gt_plan.md`

- [ ] Run one representative v1.0-mini val scene with `vertices`, `uniform`, `triangle`, and `triangle+flat-fill` into separate directories.
- [ ] Feed each dense candidate variant through frozen coarse `skip free z=1-4` Stage 2 and Stage 4 where runtime permits.
- [ ] Record generation counts, class distributions, Stage4 camera-visible occupied counts, runtime, and failures.
- [ ] Append exact commands, results, and conclusions to `docs/occ3d_flashocc_gt_plan.md`.

### Task 5: Verification

**Files:**
- Verify all modified and created files.

- [ ] Run all new unit tests.
- [ ] Run `py_compile` on all changed Python modules.
- [ ] Confirm existing Stage4 format invariants remain true: shape `(200,200,16)`, label range `0-17`, and `mask_camera <= mask_lidar`.
- [ ] Review `git diff` and ensure unrelated user changes were not modified.
