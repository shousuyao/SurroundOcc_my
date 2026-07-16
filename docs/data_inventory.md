# Data Directory Guide

本文档说明本项目 `data/` 目录中应该放哪些数据、这些数据分别是什么、从哪里获取，以及本项目的真值生成流水线会产生哪些中间目录。目标是：后续用户即使不下载当前工作区的 `data/`，也能快速知道应该准备什么数据，并能按阶段复现结果。

## 总览

本项目当前主线是真值生成，`data/` 主要包含四类内容：

1. **外部原始数据集**：nuScenes 图像、LiDAR、sweeps、pose、calibration、lidarseg 标签。
2. **外部标注或基准数据**：SurroundOcc 官方 occupancy 标签、Occ3D/FlashOcc 风格 `labels.npz`、BEVDet/FlashOcc ann pkl。
3. **本项目生成的 GT 中间产物**：dense occupied 候选、ray source points、Stage 1-4 的 `labels.npz`。
4. **实验诊断与训练输出**：统计 JSON、对比 PLY、FlashOcc smoke test 日志、checkpoint。

最小理解可以记成：

```text
data/nuscenes/
    原始传感器数据和 nuScenes 元数据

data/GT_occupancy_*/
    本项目生成或对比使用的 occupancy GT、中间结果、实验输出
```

## 推荐目录结构

### 1. 原始 nuScenes 数据

用于 SurroundOcc 原始 GT 生成、Occ3D-style ray casting、相机可见性 mask 计算。

推荐结构：

```text
data/
  nuscenes/
    maps/
    samples/
    sweeps/
    v1.0-mini/        # mini split metadata, optional for quick tests
    v1.0-trainval/    # full train/val metadata
    lidarseg/
      v1.0-mini/
      v1.0-trainval/
```

来源：

- nuScenes official download: https://www.nuscenes.org/download
- 需要下载的核心内容：
  - nuScenes sensor data: `samples/`, `sweeps/`
  - nuScenes metadata: `v1.0-mini` 或 `v1.0-trainval`
  - nuScenes lidarseg labels: `lidarseg/v1.0-mini` 或 `lidarseg/v1.0-trainval`

用途：

- `samples/`：keyframe 图像和 LiDAR。
- `sweeps/`：多帧 LiDAR 聚合和 ray casting。
- `v1.0-*/*.json`：scene/sample/sample_data/ego_pose/calibration 等元数据。
- `lidarseg/`：为 LiDAR 点提供语义类别，生成 semantic occupancy。

当前工作区示例：

```text
data/nuscenes/v1.0-mini/
data/nuscenes/lidarseg/v1.0-mini/
data/nuscenes/samples
data/nuscenes/sweeps
```

## 2. SurroundOcc 官方数据

SurroundOcc 原项目提供两类重要数据：

1. nuScenes info pkl：训练/验证时把图像路径、标定、`occ_path` 等信息组织起来。
2. SurroundOcc dense occupancy labels：格式通常是 sparse `(N, 4)` 的 `.npy`，前三列是 voxel index，第四列是 semantic label。

原始文档入口：

- [docs/data.md](data.md)
- SurroundOcc README 中的数据链接。

推荐结构：

```text
data/
  nuscenes_infos_train.pkl
  nuscenes_infos_val.pkl
  nuscenes_occ/
    ...
```

SurroundOcc 官方标签特征：

```text
shape: (N, 4)
columns: x, y, z, semantic_label
coordinate: LiDAR voxel index
default range: [-50, -50, -5, 50, 50, 3]
default occ_size: [200, 200, 16]
voxel_size: 0.5m
```

用途：

- 训练原始 SurroundOcc。
- 作为本项目 Stage 1/Stage 3 的 dense occupied semantic candidate。
- 作为 Poisson/surface completion 相关实验的输入。

注意：

- SurroundOcc sparse occupancy 主要表达 occupied voxel。
- 它不直接表达 Occ3D/FlashOcc 所需的 `free / unknown` 区分。
- 因此不能简单把空 voxel 全部当成 free。

## 3. Occ3D / FlashOcc 风格 GT

FlashOcc/Occ3D 风格 GT 通常组织为：

```text
gts/
  <scene_name>/
    <sample_token>/
      labels.npz
```

`labels.npz` 至少包含：

```text
semantics    # dense semantic grid, shape [X, Y, Z]
mask_lidar   # LiDAR observed mask
mask_camera  # camera visible mask
```

推荐网格规格：

```text
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size        = 0.4
occ_size          = [200, 200, 16]
free label        = 17
```

来源：

- Occ3D project page: https://tsinghua-mars-lab.github.io/Occ3D/
- Occ3D paper: https://arxiv.org/abs/2304.14365
- FlashOcc 通常使用 Occ3D-nuScenes 风格的 `labels.npz` 作为训练/评估 GT。

用途：

- 训练 FlashOcc。
- 作为本项目生成结果的官方格式目标。
- 与官方 Occ3D GT 做 mask/semantic 对齐诊断。

本项目不假设用户必须直接下载官方 Occ3D GT。当前主线是：用 nuScenes + lidarseg + SurroundOcc/Poisson candidate，复现并改造出可迁移的 Occ3D-style GT。

## 4. 本项目生成的中间数据

### `data/GT_occupancy_mini/`

用途：scene-0061 / mini split 上的快速实验目录。

典型子目录：

```text
dense_voxels_with_semantic/
stage1_occ3d_current/
stage2_raycast_*/
stage3_fused_*/
stage4_camera_*/
flashocc_train_*/
official_mask_semantic_upper_bound_*/
```

含义：

- `dense_voxels_with_semantic/`：由 SurroundOcc 或改造后的生成脚本得到的 sparse occupied semantic candidate，通常为 `.npy`。
- `stage1_occ3d_current/`：只做格式转换，把 sparse `(N,4)` 写成 dense `labels.npz`。
- `stage2_raycast_*`：LiDAR ray casting 结果，负责区分 occupied/free/unknown，并生成 `mask_lidar`。
- `stage3_fused_*`：在 Stage 2 几何可见性基础上融合语义候选。
- `stage4_camera_*`：基于 camera ray/projection 生成 `mask_camera`。
- `flashocc_train_*`：FlashOcc smoke test 或小规模训练输出。
- `official_mask_semantic_upper_bound_*`：与官方 Occ3D mask 或语义上界对比的诊断目录。

如何生成：

- Stage 1: `tools/occ3d_stage1/convert_surroundocc_sparse_to_occ3d.py`
- Stage 2: `tools/occ3d_stage2/raycast_occ3d_from_nuscenes.py`
- Stage 3: `tools/occ3d_stage3/fuse_stage3_semantics.py`
- Stage 4: `tools/occ3d_stage4/build_camera_mask.py`

### `data/GT_occupancy_mini_occ3d_grid/`

用途：把 mini 数据转换到 Occ3D/FlashOcc 推荐网格规格后的中间输入目录。

典型结构：

```text
dense_voxels_with_semantic/
ray_source_points_with_origin/
ray_source_voxels_with_semantic/
stats_baseline_occ3d_grid.json
```

含义：

- `dense_voxels_with_semantic/*.npy`：Occ3D 网格下的 occupied semantic candidate。
- `ray_source_points_with_origin/*.npy`：ray casting 输入，每个点保留 endpoint、真实 LiDAR origin 和 label。
- `ray_source_voxels_with_semantic/*.npy`：ray source 对应的 voxel 级语义候选。
- `stats_baseline_occ3d_grid.json`：该数据规格下的统计摘要。

这些文件用于 Stage 2 中的 dense-candidate / point-origin ray casting 分支。

### `data/GT_occupancy_v1_0_mini/`

用途：更接近完整 nuScenes mini train/val split 的生成结果。

典型结构：

```text
stage2_raycast_*_train/
stage2_raycast_*_val/
stage3_fused_*_train/
stage3_fused_*_val/
stage4_camera_*_train/
stage4_camera_*_val/
flashocc_train_*/
```

含义：

- `*_train` / `*_val`：分别对应 train/val split。
- `stage2_*`：LiDAR visibility 和 `mask_lidar`。
- `stage3_*`：语义融合。
- `stage4_*`：camera visibility 和 `mask_camera`。
- `flashocc_train_*`：用生成 GT 跑 FlashOcc 小规模训练后的输出。

### `data/GT_occupancy_v1_0_mini_occ3d_grid/`

用途：v1.0-mini 下的 Occ3D 网格中间数据。

典型结构：

```text
dense_voxels_with_semantic/
ray_source_points_with_origin/
ray_source_voxels_with_semantic/
```

这些文件是生成 `data/GT_occupancy_v1_0_mini/stage2_*` 的上游输入。

### `data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/`

用途：并行生成版本的 v1.0-mini Occ3D 网格中间数据。

它和 `GT_occupancy_v1_0_mini_occ3d_grid/` 含义类似，但通常由并行脚本或批量任务生成。当前工作区中该目录体积最大，主要作为加速后的缓存。

### `data/GT_occupancy_surface_ablation/`

用途：surface completion、Poisson reconstruction、triangle mesh、point-origin normal、filter100 等质量诊断和消融实验。

典型子目录：

```text
triangle_scene0103/
triangle_flat_scene0103/
point_origin_filter100_scene0103/
poisson_diagnostics_scene0103_frame1/
stage2_*_scene0103/
stage4_*_scene0103/
```

含义：

- `triangle_*`：三角面片或 surface reconstruction 相关 candidate。
- `point_origin_*`：基于点到真实 origin 的 ray/normal 诊断。
- `filter100_*`：过滤参数消融。
- `poisson_diagnostics_*`：Poisson reconstruction 质量检查。
- `stage2_*` / `stage4_*`：把上述 candidate 接入 ray casting 或 camera mask 后的结果。

该目录主要服务研究判断，不是训练必须目录。

## 5. 常见文件类型

| 文件类型 | 典型位置 | 含义 |
|---|---|---|
| `.npy` | `dense_voxels_with_semantic/`, `ray_source_points_with_origin/` | sparse voxel、ray endpoint/origin/label、中间 candidate |
| `labels.npz` | `stage*/gts/<scene>/<token>/` | FlashOcc/Occ3D 风格 GT，包含 `semantics`, `mask_lidar`, `mask_camera` |
| `raycast_debug.npz` | `stage2_*/debug/` | Stage 2 调试网格，如 `free_grid`, `raw_hit_grid`, `hit_label_counts` |
| `stage3_debug.npz` | `stage3_*/debug/` | Stage 3 语义融合调试数据 |
| `.pkl` | `stage*/bevdetv2-nuscenes_infos_*.pkl` | 指向生成 GT 的 ann pkl，供 FlashOcc/BEVDet dataloader 使用 |
| `.json` | 各 stage 根目录 | 统计、格式检查、与官方 GT 对比、mask 指标 |
| `.ply` | compare/export 目录 | 可视化 occupied/free/mask/diff 点云 |
| `.pth` | `flashocc_train_*` | FlashOcc 训练 checkpoint |
| `.log`, TensorBoard events | `flashocc_train_*` | 训练日志 |

## 6. 最小数据准备路径

### 路径 A：只复现原始 SurroundOcc 训练/评估

需要：

```text
data/nuscenes/
data/nuscenes_infos_train.pkl
data/nuscenes_infos_val.pkl
data/nuscenes_occ/
```

来源：

- nuScenes official download。
- SurroundOcc 官方提供的 train/val info pkl。
- SurroundOcc 官方提供的 dense occupancy labels。

参考：

- [docs/data.md](data.md)
- [docs/run.md](run.md)

### 路径 B：复现本项目 Occ3D/FlashOcc GT 生成

需要：

```text
data/nuscenes/
  samples/
  sweeps/
  v1.0-mini/ or v1.0-trainval/
  lidarseg/

SurroundOcc sparse occupancy candidate
FlashOcc/BEVDet-style ann pkl
```

生成顺序：

```text
1. 生成或准备 sparse occupied semantic candidate
2. Stage 1: sparse -> labels.npz format smoke
3. Stage 2: LiDAR ray casting -> mask_lidar
4. Stage 3: semantic fusion
5. Stage 4: camera ray mask -> mask_camera
6. FlashOcc loader / training smoke test
```

对应脚本：

```text
tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py
tools/occ3d_stage1/convert_surroundocc_sparse_to_occ3d.py
tools/occ3d_stage2/raycast_occ3d_from_nuscenes.py
tools/occ3d_stage3/fuse_stage3_semantics.py
tools/occ3d_stage4/build_camera_mask.py
```

### 路径 C：只使用官方 Occ3D/FlashOcc GT

需要：

```text
data/nuscenes/
official Occ3D gts/
FlashOcc/BEVDet ann pkl pointing to official gts
```

来源：

- Occ3D project page: https://tsinghua-mars-lab.github.io/Occ3D/
- FlashOcc 项目文档或其数据准备说明。

该路径适合训练模型，不适合研究“如何生成可迁移 GT”。本项目的主要价值在路径 B。

## 7. 面向自采数据的映射关系

迁移到自采数据时，不需要完全复刻 nuScenes 文件名，但需要提供同等信息：

| nuScenes 中的数据 | 自采数据中需要的对应物 | 用途 |
|---|---|---|
| `samples/CAM_*` | 多相机图像 | camera mask、模型输入 |
| `samples/LIDAR_TOP` / `sweeps/LIDAR_TOP` | keyframe 和历史帧 LiDAR | occupied endpoint、free-space ray |
| `ego_pose.json` | 车辆位姿或 SLAM/INS pose | 多帧聚合、sweep 对齐 |
| `calibrated_sensor.json` | LiDAR-camera 外参、相机内参 | 投影、ray origin 变换 |
| `lidarseg` | 点级语义标签或替代语义来源 | semantic occupancy |
| `sample_data.json` | 时间戳和传感器帧索引 | 多传感器同步和 sweep 检索 |
| `sample_annotation.json` | 动态物体框，可选但重要 | 动态物体保护、拖影处理 |

如果自采数据暂时没有点级语义，可以先生成几何 occupancy：

```text
occupied / free / unknown
```

之后再用语义分割模型、人工标注、检测框类别传播或多帧融合补充 semantic label。

## 8. 当前工作区中的关键数据角色

当前 `data/` 约 28G，各大目录角色如下：

| 目录 | 当前大小 | 角色 |
|---|---:|---|
| `data/nuscenes/` | 48M | nuScenes mini 原始/元数据与 lidarseg。 |
| `data/GT_occupancy_mini/` | 2.5G | mini 快速闭环、Stage 1-4、FlashOcc smoke test。 |
| `data/GT_occupancy_mini_occ3d_grid/` | 4.0G | mini 的 Occ3D 网格中间数据。 |
| `data/GT_occupancy_v1_0_mini/` | 3.4G | v1.0-mini train/val 生成结果和训练实验。 |
| `data/GT_occupancy_v1_0_mini_occ3d_grid/` | 673M | v1.0-mini Occ3D 网格中间数据。 |
| `data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/` | 15G | 并行生成的 v1.0-mini 中间缓存。 |
| `data/GT_occupancy_surface_ablation/` | 2.4G | scene-0103 surface/Poisson/visibility 消融。 |

这些目录不是下载后必须同时具备。最常用的是：

- 做原始数据处理：`data/nuscenes/`
- 做快速实验：`data/GT_occupancy_mini*`
- 做 mini train/val：`data/GT_occupancy_v1_0_mini*`
- 做 surface 质量诊断：`data/GT_occupancy_surface_ablation/`

## 9. 一句话定位

`data/nuscenes/` 是原始传感器数据；`data/GT_occupancy_*_occ3d_grid/` 是生成 GT 的中间 candidate；`data/GT_occupancy_*/stage1-4*/` 是本项目按 Occ3D/FlashOcc 格式生成的真值；`flashocc_train_*` 和各种 `.json/.ply` 是诊断与训练实验输出。
