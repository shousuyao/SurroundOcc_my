# Poisson Geometry and Visibility Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test point-origin normal orientation, mesh connectivity filtering, class-aware reconstruction, and quantify occupied voxels that lie in camera frusta but never become first hits.

**Architecture:** Extend the existing surface helper module with testable normal/topology operations, keep all new generator behavior behind explicit CLI options, and implement frustum diagnostics as a standalone read-only tool. Run controlled one-frame generation ablations before any scene-scale regeneration.

**Tech Stack:** Python 3.8, NumPy, Open3D, nuScenes annotations, pytest, existing Stage 2/4 tools.

---

### Task 1: Point-Origin Normals and Mesh Topology

**Files:**
- Modify: `tools/generate_occupancy_nuscenes/surface_completion.py`
- Modify: `tests/test_surface_completion.py`

- [ ] Add failing tests for per-point normal flipping, connected-component counts, and minimum-triangle component filtering.
- [ ] Implement pure normal orientation and Open3D topology helpers.
- [ ] Run focused tests and verify all pass.

### Task 2: Generator Normal/Topology Integration

**Files:**
- Modify: `tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py`
- Modify: `tools/generate_occupancy_nuscenes/config.yaml`

- [ ] Preserve an origin for every static and dynamic geometry point.
- [ ] Add `--normal-orientation {camera,point-origin}` and `--min-component-triangles`.
- [ ] Save normal flip count, topology before/after filtering, and component removal counts in `surface_stats`.
- [ ] Add `--max-keyframes` for controlled one-frame ablations.
- [ ] Run tests and syntax checks.

### Task 3: Semantic-Group Reconstruction

**Files:**
- Create: `tools/generate_occupancy_nuscenes/semantic_reconstruction.py`
- Create: `tests/test_semantic_reconstruction.py`
- Modify: `tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py`

- [ ] Add failing tests for class-group routing and deterministic voxel conflict priority.
- [ ] Implement flat classes from source voxels plus conservative height filling, static `manmade/vegetation` as separate Poisson meshes, and dynamic/small classes as LiDAR-supported voxels.
- [ ] Add `--reconstruction-mode {global,semantic-groups}` with `global` as default.
- [ ] Save per-group source/output counts and fallback reasons.

### Task 4: Frustum/First-Hit Diagnostics

**Files:**
- Create: `tools/occ3d_stage4/analyze_frustum_first_hit.py`
- Create: `tests/test_frustum_first_hit.py`

- [ ] Add failing tests for lidar-to-camera projection, frustum union, and depth/z histograms.
- [ ] Implement per-class counts for global occupied, in-frustum occupied, camera first-hit occupied, and in-frustum non-first-hit occupied.
- [ ] Emit depth bins, z-layer histograms, and per-camera frustum counts.
- [ ] Run on vertices/triangle/triangle+flat scene-0103 outputs.

### Task 5: Controlled Ablation and Documentation

**Files:**
- Generate: `data/GT_occupancy_surface_ablation/poisson_diagnostics_scene0103_frame1/`
- Modify: `docs/occ3d_flashocc_gt_plan.md`

- [ ] Generate one frame for camera-normal baseline, point-origin normals, point-origin+component filter, and semantic groups.
- [ ] Compare candidate class counts, topology, Stage 2 occupied, and Stage 4 first-hit metrics.
- [ ] Record exact commands, outputs, failures, and conclusions in the main plan document.
- [ ] Run all new tests, syntax checks, and Stage4 invariants.
