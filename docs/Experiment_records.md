------

# SurroundOcc到Occ3D Occupancy GT重构及FlashOcc训练适配研究报告

[TOC]

# 1. 研究背景与目标

随着自动驾驶三维场景理解任务的发展，Occupancy Prediction逐渐成为BEV感知的重要方向。相比传统3D检测仅预测目标类别和边界框，Occupancy Network通过预测空间中每个体素的占据状态，实现更加完整的环境建模。目前主流方法如FlashOcc依赖Occ3D-nuScenes提供的occupancy监督数据，而已有SurroundOcc流程生成的occupancy GT主要面向三维场景重建，二者在监督定义上存在明显差异。

SurroundOcc生成结果主要表示空间中的occupied voxel及对应语义：

```
(N,4)

[x,y,z,semantic]
```

其中包含：

- voxel空间位置；
- occupied语义类别。

但是该结果缺少Occ3D训练所需要的：

- LiDAR可观测区域(mask_lidar)；
- free space；
- unknown区域；
- camera可见区域(mask_camera)。

因此，直接将SurroundOcc结果转换为Occ3D格式无法得到有效监督。本项目目标是基于已有SurroundOcc occupancy结果和nuScenes原始传感器数据，重新构建符合Occ3D定义的occupancy GT，并完成FlashOcc训练验证闭环。

整体流程如下：

```
SurroundOcc occupancy
        |
        |
        v
Sparse-to-Dense格式转换
        |
        |
        v
LiDAR ray casting visibility reconstruction
        |
        |
        v
Semantic fusion
        |
        |
        v
Camera visibility generation
        |
        |
        v
FlashOcc training validation
```

最终生成：

```
labels.npz

├── semantics
├── mask_lidar
└── mask_camera
```

其中：

```
shape=(200,200,16)
```

与Occ3D-nuScenes和FlashOcc输入格式保持一致。

------

# 2. SurroundOcc GT格式转换与FlashOcc读取验证

首先完成SurroundOcc sparse occupancy到Occ3D dense occupancy格式转换，主要目标是验证数据格式、路径组织以及FlashOcc数据读取流程是否正确。

实现代码：

```
tools/occ3d_stage1/
├── convert_surroundocc_sparse_to_occ3d.py
└── check_occ3d_npz.py
```

输入：

```
data/GT_occupancy_mini/
dense_voxels_with_semantic/
```

输出：

```
data/GT_occupancy_mini/stage1_occ3d_current/

gts/
 └── scene-0061/
      └── sample_token/
             labels.npz
```

同时生成FlashOcc读取所需：

```
bevdetv2-nuscenes_infos_stage1_train.pkl
```

实验结果：

- scene-0061共39帧全部转换成功；
- `semantics`、`mask_lidar`、`mask_camera`尺寸均为：

```
(200,200,16)
```

- FlashOcc原始`LoadOccGTFromFile`成功读取。

验证日志：

```
data/GT_occupancy_mini/stage1_occ3d_current/
smoke_test_stage1_loader.json
```

该阶段证明：

> SurroundOcc结果可以无损转换为FlashOcc可读取格式，但由于缺少真实LiDAR visibility信息，不能直接作为最终occupancy监督。

因此后续重点转向Occ3D-style visibility reconstruction。

------

# 3. 基于LiDAR Ray Casting的Occupancy Visibility重建

## 3.1 基础Ray Casting方案

Occ3D occupancy监督的核心区别在于其利用LiDAR射线判断空间状态：

- LiDAR命中的voxel → occupied；
- LiDAR经过但未命中的voxel → free；
- 未被射线覆盖 → unknown。

因此第二阶段重新读取nuScenes原始LiDAR数据，通过3D DDA算法生成：

```
free_grid
raw_hit_grid
mask_lidar
```

实现代码：

```
tools/occ3d_stage2/
raycast_occ3d_from_nuscenes.py
```

输出：

```
data/GT_occupancy_mini/
stage2_raycast_occ3d/
```

debug文件：

```
debug/
 └── scene-0061/
       raycast_debug.npz
```

统计文件：

```
stats_raycast.json
```

------

初始keyframe LiDAR实验结果：

| 状态           | 数量     |
| -------------- | -------- |
| occupied voxel | 133125   |
| free voxel     | 4898783  |
| observed voxel | 5031908  |
| unknown voxel  | 19928092 |

与官方Occ3D mask_lidar比较：

```
mask_lidar IoU=0.1543
```

结果说明：

基础ray casting能够恢复free/occupied/unknown三态关系，但空间覆盖范围与官方Occ3D仍存在较大差距。

------

# 4. Per-point LiDAR Origin优化

## 4.1 问题分析

进一步分析发现，主要误差来自多sweep LiDAR数据中的ray origin处理。

初始实现：

所有LiDAR点统一使用：

```
origin=[0,0,0]
```

即默认当前keyframe LiDAR位置。

但是nuScenes多帧LiDAR数据中：

不同sweep对应不同采样时间，因此：

```
point_i
origin_i
```

应该一一对应。

错误方式：

```
sweep t0 point
        |
        |
   keyframe origin
```

会导致射线方向偏移。

------

## 4.2 Per-point Origin方案

重新生成：

```
ray_source_points_with_origin/
```

每条记录：

```
(point_x,
 point_y,
 point_z,

 origin_x,
 origin_y,
 origin_z,

 semantic)
```

共生成：

```
31,870,362
```

条point-origin ray。

随后采用Occ3D Appendix D方式：

```
ray_start = point

ray_end = point_origin
```

进行DDA traversal。

------

实验结果：

| 方法               | mask_lidar IoU |
| ------------------ | -------------- |
| keyframe origin    | 0.2428         |
| point-level origin | 0.2853         |

最佳输出：

```
data/GT_occupancy_mini/

stage2_raycast_occ3d_dense_point_origin_occ3d_dda_swapxy_flipy/
```

结果：

```
mask_lidar_iou:

0.285312
```

该实验说明：

> 对于多帧LiDAR occupancy生成，正确恢复每个点对应sensor origin的重要性明显高于单纯增加occupancy candidate数量。

------

# 5. Semantic Fusion与Occupied Candidate分析

在LiDAR visibility确定后，引入SurroundOcc dense occupancy作为semantic candidate。

融合原则：

- mask_lidar由raw LiDAR ray决定；
- SurroundOcc只提供semantic信息；
- 不允许Poisson补全区域扩展observed区域。

实现：

```
tools/occ3d_stage3/
fuse_stage3_semantics.py
```

输出：

```
data/GT_occupancy_mini/

stage3_fused_occ3d_point_origin_occ3d_dda_swapxy_flipy/
```

包含：

```
labels.npz

stats_semantic.json

bevdetv2-nuscenes_infos_stage3_train.pkl
```

------

实验结果：

semantic mIoU：

```
Stage2:

0.03446


Stage3:

0.03448
```

提升非常有限。

说明：

当前主要瓶颈并非语义传播，而是：

- occupied geometry；
- LiDAR visibility；
- free-space范围。

因此后续优化重点转向visibility建模。

------

# 6. Free-space误差分析与Low-Z约束优化

通过mask FP/FN可视化分析发现，大量false positive集中在低高度区域。

输出：

```
mask_diff_ply/
```

统计：

baseline:

```
FP:
7280696
```

主要分布：

```
low Z layer 1-5
```

原因：

低角度LiDAR ray在地面附近传播距离较长：

```
LiDAR
 \
  \
   \
    ground
```

导致大量区域被错误认为free。

------

因此设计：

```
skip free z=1-4
```

限制低Z voxel free写入。

实验：

| 方法            | mask IoU |
| --------------- | -------- |
| baseline        | 0.285312 |
| skip free z=1-4 | 0.315250 |

详细记录：

```
data/GT_occupancy_mini/

low_z_free_write_ablation_summary.md
```

结果：

- IoU提升约10%；
- precision明显提升；
- recall仍保持较高水平。

当前推荐LiDAR visibility版本：

```
point-level origin

+

Occ3D point-to-origin DDA

+

skip free z=1-4
```

------

# 7. Camera Visibility生成

在LiDAR mask稳定后，实现camera-visible mask生成。

实现：

```
tools/occ3d_stage4/

build_camera_mask.py
```

流程：

```
camera pixel

↓

3D ray

↓

voxel traversal

↓

first occupied stop
```

输出：

```
data/GT_occupancy_mini/

stage4_camera_raymask_z1_4/
```

检查：

```
check_stage4_format.json

check_occ3d_npz_stage4.json
```

结果：

```
mask_lidar_count:

7595390


mask_camera_count:

2454520
```

满足：

```
mask_camera <= mask_lidar
```

与官方Occ3D比较：

```
mask_camera IoU:

0.247
```

说明camera visibility基本逻辑正确，但仍受occupied geometry和动态时间同步影响。

------

# 8. FlashOcc训练验证

最终生成GT接入FlashOcc训练流程。

训练输出：

```
data/GT_occupancy_mini/

flashocc_train_smoke_baseline_full/

epoch_1.pth
```

训练结果：

```
loss_occ:

2.8823

↓

1.3225
```

同时：

- dataloader正常；
- forward/backward正常；
- checkpoint正常保存；
- prediction非全零。

说明：

> 当前生成GT已经具备驱动FlashOcc训练的能力，后续优化主要针对GT质量，而非工程链路。

------

# 9. 当前最佳版本

当前推荐版本：

```
Raw LiDAR point

↓

Per-point origin recovery

↓

Occ3D point-to-origin DDA

↓

Low-Z free suppression

↓

Semantic fusion

↓

Camera ray mask
```

核心输出路径：

```
data/GT_occupancy_mini/

stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy/
```

当前主要指标：

| 指标           | 结果  |
| -------------- | ----- |
| mask_lidar IoU | 0.315 |
| precision      | 0.36  |
| recall         | 0.71  |
| FlashOcc训练   | 成功  |

------

# 10. 后续优化方向

目前距离官方Occ3D仍存在差距，主要问题集中于visibility和geometry。

下一阶段重点：

## （1）Source-aware Ray Casting

当前所有ray统一处理，后续区分：

- raw LiDAR；
- dynamic object；
- static reconstruction；
- mesh completion。

针对不同来源采用不同free传播策略。

------

## （2）Occupied Geometry优化

当前Poisson相关实验：

- triangle surface；
- point-origin normal；
- component filtering。

结果：

geometry IoU有所提升：

```
0.383 → 0.391
```

但semantic下降。

后续方向：

- geometry reconstruction与semantic assignment解耦；
- static/dynamic分别重建；
- voxel-level component filtering。

------

## （3）Camera Visibility增强

后续：

- camera timestamp compensation；
- depth consistency；
- 更精确遮挡建模。

------

# 总结

当前工作已经完成：

1. SurroundOcc → Occ3D格式迁移；
2. LiDAR ray-based occupancy reconstruction；
3. per-point origin优化；
4. semantic fusion；
5. camera mask生成；
6. FlashOcc训练闭环验证。

实验结果表明：

- 最大影响因素不是semantic，而是visibility；
- per-point LiDAR origin是提升occupancy质量的关键；
- 当前主要误差来自free-space over-estimation和occupied geometry不足。

下一阶段研究重点应从“格式转换”转向“更精确的occupancy visibility modeling”。

------

