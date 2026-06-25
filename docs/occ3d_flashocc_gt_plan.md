# SurroundOcc to Occ3D/FlashOcc Occupancy GT Plan

本文档记录从 SurroundOcc 已复现的 nuScenes occupancy GT 生成流程，迁移到 Occ3D 数据格式，并服务于 FlashOcc 训练的计划。重点不是简单转换文件格式，而是复现 Occ3D 风格的 occlusion reasoning，使 `occupied / free / unknown` 三态定义正确。

## 1. 目标

输入：

- SurroundOcc 生成的稀疏 occupancy 标签，格式为 `(N, 4)`。
- 前三列是 voxel index: `x, y, z`。
- 第四列是 semantic label，由 nuScenes lidarseg 标签经映射和传播得到。
- 原始 nuScenes LiDAR / lidarseg sweep，用于恢复每条 ray 的真实 origin、endpoint 和 free-space path。

输出：

- Occ3D/FlashOcc 风格的 dense GT 文件。
- 推荐结构：

```text
gts/
  <scene_name>/
    <sample_token>/
      labels.npz
```

`labels.npz` 至少包含：

```text
semantics    # dense semantic grid, shape [X, Y, Z]
mask_lidar   # LiDAR-observed mask, same shape
mask_camera  # camera-visible mask, same shape
```

目标栅格配置建议优先与 Occ3D-nuScenes / FlashOcc 对齐：

```text
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
voxel_size        = 0.4
occ_size          = [200, 200, 16]
semantic classes  = 0-16
free class        = 17
```

## 2. 核心三态定义

对每个 voxel，LiDAR supervision 应该分为三种状态：

```text
if LiDAR ray 命中该 voxel:
    occupied，写入语义类，mask_lidar = 1
elif LiDAR ray 穿过该 voxel，但没有在该 voxel 命中:
    free，semantics = 17，mask_lidar = 1
else:
    unknown，mask_lidar = 0
```

关键原则：

- 没有 ray 经过的 voxel 不等于 free，而是 unknown。
- unknown 不参与 LiDAR-supervised loss/evaluation。
- `semantics == 17` 只有在 `mask_lidar == 1` 时才表示确认的 free。
- 如果最终文件中 unknown 区域也保存为 `17`，必须依赖 `mask_lidar == 0` 区分。

调试期建议使用 `255` 作为 unknown 占位符，导出前再统一替换为 `17`：

```python
semantics = np.full(occ_size, 255, dtype=np.uint8)
mask_lidar = np.zeros(occ_size, dtype=bool)

# export 前
semantics_export = semantics.copy()
semantics_export[semantics_export == 255] = 17
```

如果直接初始化为 `17`，需要加断言避免把 unknown 当成 free 使用：

```python
assert not np.any(semantics[mask_lidar == 0] != 17), \
    "unknown 区域 semantics 应保持初始值 17，但不代表 free"
```

## 3. Ray Casting 预处理

### 3.1 点云过滤

当前 `tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py` 已经包含自车附近点过滤逻辑：

```python
range = config["self_range"]
oneself_mask = (
    (np.abs(pc0[:, 0]) > range[0]) |
    (np.abs(pc0[:, 1]) > range[1]) |
    (np.abs(pc0[:, 2]) > range[2])
)
points_mask = points_mask & oneself_mask
```

该逻辑等价于去掉自车附近 cuboid 内的点：

```text
abs(x) <= self_range[0]
abs(y) <= self_range[1]
abs(z) <= self_range[2]
```

因此 ray casting 版本不要再无条件叠加一个新的近距离球形过滤，例如 `dist > 0.5m`。否则会和已有 `self_range` 过滤产生冗余，甚至改变 SurroundOcc 原始 GT 的点云分布。

推荐实现策略：

- 将现有 `self_range` 过滤抽成公共函数，原始 occupancy 生成和 Occ3D ray casting 共用。
- ray casting 的 raw sweep points 也必须先经过同样的 `points_mask & oneself_mask`，再用于 free/occupied 判断。
- `self_range` 过滤必须在每个 sweep 自己的 LiDAR 局部坐标系下执行，再将过滤后的点变换到 keyframe LiDAR 坐标系。
- 如果还需要过滤异常远点，将其作为独立的可选 `lidar_max_range` 过滤，并在统计中单独记录过滤数量。
- 不要同时启用多个含义相近的 near-range filter，除非有对比实验支持。

不要先把非 keyframe sweep 点变换到 keyframe LiDAR 坐标系后，再套用 keyframe 坐标系下的 `self_range`。车辆在不同 sweep 时刻的位置不同，先变换再过滤会让自车附近过滤区域错位，导致其他 sweep 的自车反射点漏入 ray casting。

可选远距离过滤：

```python
dist = np.linalg.norm(points[:, :3] - ray_origin[None, :], axis=1)
valid = dist < lidar_max_range
points = points[valid]
labels = labels[valid]
```

建议初始参数：

```text
self_range      = 复用 config.yaml 中的 self_range
lidar_max_range = 可选；按 nuScenes LiDAR 有效范围设置，可先用 80m 左右并在验证中调整
```

### 3.2 多帧 sweep 坐标转换

每个 sweep 的点云需要统一变换到当前 keyframe 的 `LIDAR_TOP` 坐标系。

容易出错的地方是 ray origin：每个 sweep 的 LiDAR origin 不同，不能统一使用 keyframe origin。

正确做法：

```python
for sweep in sweeps:
    sweep_points_local, sweep_labels = apply_self_range_filter(
        sweep.points, sweep.labels, config["self_range"]
    )
    sweep_points_in_keyframe = transform_points_to_keyframe_lidar(sweep_points_local)
    sweep_origin_in_keyframe = transform_point_to_keyframe_lidar(np.array([0, 0, 0]), sweep.pose)

    for point, label in zip(sweep_points_in_keyframe, sweep_labels):
        record(point, sweep_origin_in_keyframe, label)
```

原则：

- keyframe 点云的 origin 是 `[0, 0, 0]`。
- 非 keyframe sweep 的 origin 必须通过该 sweep 的 `lidar -> ego -> global -> keyframe ego -> keyframe lidar` 链路变换得到。
- 点和 origin 必须使用同一条坐标变换链，否则 free-space ray path 会系统性偏移。
- `self_range` 过滤发生在 sweep local frame，坐标变换发生在过滤之后。

注意：Occ3D Appendix D Algorithm 2 的 ray casting 不是以 `origin -> point` 的伪代码形式写入，而是对每个聚合点保存等长的 `points_origin`，并使用：

```text
ray_start = points[i]
ray_end   = points_origin[i]
```

也就是说，官方主线需要逐点保存真实 LiDAR origin，不能把同一 keyframe 的所有点统一近似为 `[0, 0, 0]`。`origin -> point` 与 `point -> origin` 的几何线段相同，但 DDA 起点、终止条件和是否包含起点/终点体素会影响 free/occupied 写入，文档后续以论文的 `point -> origin` 为主线。

## 4. 3D DDA Traversal

使用 3D DDA 或等价的 voxel traversal 算法，得到 ray 穿过的 voxel 序列。

### 4.1 论文精确版：point -> origin

Occ3D Appendix D 的 LiDAR visibility 使用两个计数网格：

```text
voxel_occ_count
voxel_free_count
```

对每个聚合 LiDAR 点：

```text
ray_start = point
ray_end   = point_origin
target_voxel = floor((ray_start - pc_range[:3]) / voxel_size)

if target_voxel inside spatial_shape:
    voxel_occ_count[target_voxel] += 1
    voxel_label[target_voxel] = point_label

    for voxel_index in ray_casting(ray_start, ray_end):
        voxel_free_count[voxel_index] += 1
```

最终状态写入顺序是论文中非常关键的优先级：

```text
voxel_state[voxel_free_count > 0] = FREE
voxel_state[voxel_occ_count > 0] = OCCUPIED
```

因此，`OCCUPIED` 的优先级高于 `FREE`。如果同一个 voxel 既被某条 ray 穿过又被某个点命中，最终必须是 occupied，而不是 free。

注意：论文伪代码没有对同一 voxel 的多个 semantic label 做 majority voting；它是在命中时直接写 `voxel_label[target_voxel] = point_label`。如果工程实现为了稳定性引入多数投票，必须标注为非论文精确分支，不能作为复现 Occ3D 的主线验收标准。

推荐接口：

```python
def occ3d_ray_casting(point, point_origin, voxel_size, pc_range, spatial_shape):
    """
    Returns:
        free_voxels: voxel indices traversed by Algorithm 1 from point to point_origin.
        target_voxel: voxel containing point, written through voxel_occ_count.
    """
```

### 4.2 非论文精确扩展：clipping / grid 外处理

早期工程实验中曾考虑以下鲁棒处理：

```text
endpoint 在 grid 外但 ray 穿过 grid 时，对 grid 内路径标 free。
origin 在 grid 外但 ray 穿过 grid 时，从 entry point 开始。
```

这类处理有工程意义，但不是 Appendix D Algorithm 2 的精确伪代码。论文 Algorithm 2 在进入主循环前会过滤 `points / points_origin / points_label` 到 `pc_range` 内；复现主线应优先遵守官方过滤和 Algorithm 1/2 的终止/边界逻辑。

### 4.3 ray 与 grid 无交集

如果 ray 完全不经过 grid，则跳过，不更新任何 voxel。

## 5. Endpoint 语义融合

SurroundOcc 的 `(N, 4)` 中第 4 列已经是 semantic label，因此最终 occupied voxel 的语义来源可以直接使用该 label。

但实现时需要区分两个来源：

- `mask_lidar` 和 free/unknown 判断应来自原始 LiDAR ray。
- dense occupied semantics 可以来自 SurroundOcc 生成的 `(N, 4)`。

不要把 Poisson/KNN 后的 dense occupied voxel 当成原始 LiDAR endpoint 去做 free-space ray casting。dense voxel 可能是表面重建或语义传播结果，不一定对应一条真实 LiDAR 光束；如果用 dense voxel 反推 ray path，会把补全出来的结构错误地标成 observed。

推荐实现为两路融合：

```text
raw LiDAR rays:
    决定 free voxels 和真实 hit voxels
    生成 mask_lidar 的 observed 区域

SurroundOcc dense occupancy:
    只在 raw LiDAR observed 边界内提供 occupied semantic 候选
    不能扩展 mask_lidar 的覆盖范围
```

两路融合的边界：

- `mask_lidar` 的 observed 区域完全由 raw LiDAR ray casting 决定。
- SurroundOcc dense occupancy 不能把 `mask_lidar == 0` 的 unknown voxel 改成 observed。
- SurroundOcc dense occupancy 不能把 raw ray 明确标成 free 的 voxel 直接覆盖成 occupied，除非该 voxel 同时属于 raw LiDAR hit voxel 集合。
- Poisson 补全出来、没有对应 raw LiDAR hit 支撑的 voxel 保持 unknown，不写入 `mask_lidar = 1`。
- SurroundOcc dense occupancy 更适合作为 raw hit voxel 内的语义候选，用于降低单点 lidarseg 语义噪声。

推荐额外保留两个中间量，避免实现时混淆：

```text
occ_count_grid         # Algorithm 2 中由真实 LiDAR point 命中的 voxel 计数
free_count_grid        # Algorithm 2 中由 point->origin ray 穿过的 voxel 计数
poisson_occ_grid       # SurroundOcc/Poisson/KNN 得到的 dense occupied candidate，仅用于非论文分支或补充实验
```

论文精确主线中，最终 state 由 `occ_count_grid/free_count_grid` 决定：

```text
voxel_state[free_count_grid > 0] = FREE
voxel_state[occ_count_grid > 0] = OCCUPIED
```

如果希望利用 Poisson 补全监督更稠密的 occupied 结构，建议作为单独实验分支记录，并且必须区分“用于 occupied densification”与“用于 LiDAR ray visibility”的点源；Poisson-only 点不应默认发 free ray。

但同一个 voxel 可能被多条 ray 命中，并且语义不一致。Occ3D Appendix D Algorithm 2 的伪代码没有 majority voting，而是在每次命中时执行：

```text
voxel_label[target_voxel] = points_label[i]
```

因此，若目标是严格复现 Occ3D，不能把 majority voting 写成主线规则。多数投票可以作为工程稳健性实验分支，但需要在实验名和统计中明确标注为 `non-paper majority-vote variant`。

冲突处理优先级：

- `voxel_occ_count > 0` 的优先级高于 `voxel_free_count > 0`。
- 如果某个 voxel 同时被 ray 标为 free 且被 LiDAR point 命中，最终 state 必须是 occupied。
- state 的最终写入顺序应为：先 `free_count > 0` 写 FREE，再 `occ_count > 0` 写 OCCUPIED。
- 语义 label 按论文伪代码为逐点赋值；如果实现为了确定性而改成多数投票，需要标注为非论文精确实现。
- label 为 ignore/noise 的点可按实验策略处理：
  - 初版：过滤掉 label `0` 的 hit，不作为 occupied 语义写入。
  - 对齐 SurroundOcc 时：保留 label `0`，但训练/eval 需明确 ignore 逻辑。

## 6. 完整算法草案

```text
初始化：
    semantics[:] = 255     # debug unknown placeholder
    mask_lidar[:] = 0
    voxel_free_count[:] = 0
    voxel_occ_count[:] = 0
    voxel_label[:] = FREE_LABEL

预处理：
    读取聚合后的 LiDAR points、points_origin、points_label
    在每个 sweep local frame 执行 self_range 过滤
    可选执行 lidar_max_range 过滤并记录统计
    将每个 sweep points 变换到目标帧坐标系
    将每个 sweep 的 ray origin 变换到目标帧坐标系
    保持 points 与 points_origin 一一对应

对每个 point i：
    ray_start = points[i]
    ray_end = points_origin[i]
    target_voxel = floor((ray_start - pc_range[:3]) / voxel_size)

    if target_voxel 不在 spatial_shape:
        continue

    voxel_occ_count[target_voxel] += 1
    voxel_label[target_voxel] = points_label[i]

    for voxel_index in occ3d_ray_casting(ray_start, ray_end):
        voxel_free_count[voxel_index] += 1

融合：
    # 写入顺序不能颠倒：先写 FREE，再写 OCCUPIED。
    free_grid = voxel_free_count > 0
    occ_grid = voxel_occ_count > 0

    semantics[free_grid] = 17
    mask_lidar[free_grid] = 1

    semantics[occ_grid] = voxel_label[occ_grid]
    mask_lidar[occ_grid] = 1

    # OCCUPIED 覆盖同一 voxel 上已有的 FREE 标记，这是论文 Algorithm 2 的状态优先级。

导出：
    semantics[semantics == 255] = 17
    save labels.npz with semantics, mask_lidar, mask_camera
```

## 7. Mask Camera

第一阶段可以先生成可用但保守的 `mask_camera`，保证 FlashOcc dataloader 和训练流程跑通。

推荐基础实现：

```text
for each voxel center:
    transform LiDAR -> camera
    if depth > 0 and projected pixel inside image:
        mask_camera = 1
```

更严格版本可加入：

- image bounds。
- depth range。
- z-buffer / occlusion test。
- 与 camera timestamp 对齐。
- 多相机 OR 融合。

注意：`mask_camera` 主要决定 camera-visible 区域的训练/评估范围。基础投影版可能过宽，建议在验证闭环通过后再强化遮挡判断。

nuScenes 的 6 个相机与 LiDAR keyframe 不是严格同时刻采集。基础版 `mask_camera` 可以先不做时间戳补偿；这对静态场景影响较小，但动态物体附近的 camera-visible mask 边界可能有轻微偏差。该问题建议在 Layer 4 的 Image-guided Refinement 或 motion compensation 阶段统一处理。

## 8. 验证闭环

### 8.1 文件和数值检查

每帧导出后检查：

```python
data = np.load(labels_npz)
assert data["semantics"].shape == (200, 200, 16)
assert data["mask_lidar"].shape == (200, 200, 16)
assert data["mask_camera"].shape == (200, 200, 16)
assert data["semantics"].dtype in (np.uint8, np.int64, np.int32)
assert data["mask_lidar"].dtype in (np.bool_, np.uint8)
assert data["mask_camera"].dtype in (np.bool_, np.uint8)
assert data["semantics"].min() >= 0
assert data["semantics"].max() <= 17
```

统计指标：

```text
occupied_count = count(mask_lidar == 1 and semantics != 17)
free_count     = count(mask_lidar == 1 and semantics == 17)
unknown_count  = count(mask_lidar == 0)
self_filtered_points = count(points removed by self_range)
max_range_filtered_points = count(points removed by optional lidar_max_range)
```

需要重点观察：

- free 是否异常少：可能 DDA 或 clipping 有问题。
- occupied 是否异常多：可能噪点未过滤或 endpoint 重复膨胀。
- unknown 是否异常少：可能误把未观测区域标成 free。
- 某些类别突然消失：可能 semantic mapping、逐点 label 写入顺序或 ignore 策略有问题。
- self/max range 过滤数量是否异常：可能过滤策略和原 SurroundOcc 管线不一致。

### 8.2 可视化检查

生成 `.ply` 对比：

- occupied voxels。
- free-space slice。
- mask_lidar observed region。
- 与已有 FlashOcc / Occ3D GT 对齐比较。

重点检查：

- 是否整体平移。
- 是否 z 方向错位。
- 是否 x/y 轴互换或翻转。
- 动态物体是否拖影。
- free-space ray 是否从正确 sweep origin 发出。

### 8.3 FlashOcc 训练 smoke test

最小闭环：

```text
1. FlashOcc dataloader 读取一帧 labels.npz
2. 单 batch forward
3. 单 batch backward
4. mini split 训练若干 iteration
5. 可视化预测和 GT
```

通过标准：

- dataloader 不报字段/shape 错误。
- loss 有效且不出现 NaN。
- occupied/free/unknown 比例合理。
- 预测可视化没有明显坐标错位。

## 9. 分阶段执行与验收标准

建议按阶段推进，每一阶段都保留中间产物、统计文件和可视化结果。不要一开始就直接生成全量 GT；先用 mini split 或单 scene 把坐标、mask、语义和 FlashOcc 读取链路打通。

### Stage 0: 基线复核

当前状态：

- mini scene 已复跑并生成 `data/GT_occupancy_mini/dense_voxels_with_semantic/`。
- 该目录下 39 帧 `.npy` 均满足 sparse `(N, 4)` 格式，voxel index 未越界，semantic label 位于 `0-16`。
- baseline 统计已记录在 `data/GT_occupancy_mini/stats_baseline.json`。
- 已完成可视化人工检查，未发现明显整体平移、轴翻转或 z 方向错位。
- 注意：该阶段验证的是当前 SurroundOcc baseline 配置，即 `voxel_size=0.5`、`pc_range=[-50,-50,-5,50,50,3]`、`occ_size=[200,200,16]`；是否切换到 Occ3D/FlashOcc 推荐的 `0.4m / [-40,-40,-1,40,40,5.4]` 规格，应在 Stage 1 开始前明确。

目标：

- 确认当前 SurroundOcc GT 生成流程稳定可复现。
- 确认输入数据、语义映射、坐标范围和已有过滤逻辑没有漂移。

执行内容：

- 复跑一个 mini scene 的原始 SurroundOcc GT 生成。
- 检查 `config.yaml` 中 `self_range`、`pc_range`、`voxel_size`、`occ_size`。
- 统计原始 `(N, 4)` 标签的 voxel 范围和 semantic label 分布。
- 可视化已有 `.npy` 到 `.ply`，与当前保存的对齐结果比较。

验收成果：

- 一份 baseline 统计文件，例如 `stats_baseline.json`。
- 至少一帧原始 SurroundOcc occupied `.ply`。
- 明确记录当前使用的配置和 nuScenes split。

验收标准：

- 稀疏 GT shape 为 `(N, 4)`。
- voxel index 不越界。
- semantic label 在预期范围内。
- 可视化没有新增的整体平移、轴翻转或 z 方向错位。

### Stage 1: Occ3D 文件格式导出

当前状态：

- 已按当前 SurroundOcc baseline 规格完成 Stage 1 格式导出，暂不切换 Occ3D 推荐坐标范围。
- 转换脚本：`tools/occ3d_stage1/convert_surroundocc_sparse_to_occ3d.py`。
- 检查脚本：`tools/occ3d_stage1/check_occ3d_npz.py`。
- 输出目录：`data/GT_occupancy_mini/stage1_occ3d_current/gts/scene-0061/<sample_token>/labels.npz`。
- 供 FlashOCC mini smoke test 使用的 ann pkl：`data/GT_occupancy_mini/stage1_occ3d_current/bevdetv2-nuscenes_infos_stage1_train.pkl`。
- 39/39 帧已成功从 sparse `(N, 4)` 转为 dense `labels.npz`，字段为 `semantics`、`mask_lidar`、`mask_camera`。
- `semantics.shape == mask_lidar.shape == mask_camera.shape == (200, 200, 16)`，`semantics` 值域为 `0-17`。
- 当前 `mask_lidar` 和 `mask_camera` 均为 occupied-only 临时占位 mask，仅用于格式 smoke test；它们不是最终 Occ3D-style observed/free/unknown mask。
- 已在 `flashocc` 环境下用 FlashOCC 原生 `LoadOccGTFromFile` 读取单帧通过，结果记录在 `data/GT_occupancy_mini/stage1_occ3d_current/smoke_test_stage1_loader.json`。
- 完整 FlashOCC package/dataset import 会触发 `dvr` CUDA extension JIT；在当前沙箱下默认 cache 目录不可写，因此 Stage 1 只验证到 loader-level 文件读取。完整 dataloader/forward/backward 放到 Stage 5，届时需要确保 `TORCH_EXTENSIONS_DIR` 等 cache 路径可写。

目标：

- 先不引入复杂 ray casting，只打通 `labels.npz` 的字段、目录和 FlashOcc dataloader 读取格式。

执行内容：

- 新增转换脚本，将单帧 sparse `(N, 4)` 写成 dense `semantics`。
- 生成 `mask_lidar` 和 `mask_camera` 的临时占位版本，仅用于格式 smoke test。
- 输出目录按 FlashOcc/Occ3D 期望组织。

验收成果：

- 单 scene 或 mini split 的 `labels.npz`。
- `check_occ3d_npz.py` 或等价检查脚本。
- 格式检查日志。

验收标准：

- `semantics.shape == mask_lidar.shape == mask_camera.shape == (200, 200, 16)`。
- `semantics` label 范围为 `0-17`。
- `mask_lidar`、`mask_camera` dtype 为 `bool` 或 `uint8`。
- FlashOcc dataloader 可以读取单帧，不报 key、shape、dtype 错误。

### Stage 2: Raw LiDAR Ray Casting

当前状态：

- 已实现 Stage 2 脚本：`tools/occ3d_stage2/raycast_occ3d_from_nuscenes.py`。
- Stage 2 已切换到 FlashOCC/Occ3D 网格：`point_cloud_range=[-40,-40,-1,40,40,5.4]`、`voxel_size=[0.4,0.4,0.4]`、`occ_size=[200,200,16]`。
- 该阶段不再复用 SurroundOcc sparse voxel index，因为 Stage 0/1 的 SurroundOcc GT 使用 `pc_range=[-50,-50,-5,50,50,3]`、`voxel_size=0.5`，两套 index 不能直接混用。
- 输出目录：`data/GT_occupancy_mini/stage2_raycast_occ3d/gts/scene-0061/<sample_token>/labels.npz`。
- debug 输出：`data/GT_occupancy_mini/stage2_raycast_occ3d/debug/scene-0061/<sample_token>/raycast_debug.npz`，包含 `free_grid`、`raw_hit_grid`、`mask_lidar`、`hit_label_counts`。
- 供 FlashOCC smoke test 使用的 ann pkl：`data/GT_occupancy_mini/stage2_raycast_occ3d/bevdetv2-nuscenes_infos_stage2_train.pkl`。
- 39/39 帧已完成 keyframe raw `LIDAR_TOP` ray casting，统计见 `data/GT_occupancy_mini/stage2_raycast_occ3d/stats_raycast.json`。
- 汇总统计：occupied/raw-hit voxel `133125`，free voxel `4898783`，observed voxel `5031908`，unknown voxel `19928092`。
- 已用 FlashOCC 原生 `LoadOccGTFromFile` 在 `flashocc` 环境下读取单帧通过，结果记录在 `data/GT_occupancy_mini/stage2_raycast_occ3d/smoke_test_stage2_loader.json`。
- 当前 FlashOCC mini info pkl 的 `sweeps` 字段为空，因此本版 Stage 2 使用 keyframe LiDAR ray；多 sweep ray origin 版本需要后续从 nuScenes sample_data 链读取相邻 sweep 并扩展。
- 当前 `mask_camera` 仍为临时占位，设置为 `mask_lidar.copy()`，只保证 loader smoke test 可读；真实 camera-visible mask 放在 Stage 4。
- 已扩展 Stage 2 脚本支持 `--num-sweeps` 与 `--sweep-direction`，从 nuScenes `sample_data` 链读取额外 LIDAR_TOP sweep。每个 sweep 在自己的 LiDAR 局部坐标执行 `self_range`，再变换到目标 keyframe LiDAR 坐标系；ray origin 也逐 sweep 变换，不再统一假设为 keyframe `[0,0,0]`。
- 已完成 `num_sweeps=3, sweep_direction=next` 实验，输出为 `data/GT_occupancy_mini/stage2_raycast_occ3d_next3/`，摘要为 `data/GT_occupancy_mini/stage2_raycast_occ3d_next3/summary_stage2_next3.md`。与 keyframe Stage 2 相比，官方 `mask_lidar` IoU 从 `0.15434566295855257` 到 `0.1614620860735415`，recall 从 `0.30747680833995805` 到 `0.35900648421200526`，但 precision 从 `0.2365919647179559` 降到 `0.22686337162522321`，说明多 sweep 提升覆盖但也扩大了 observed mask。
- 将 `next3` Stage 2 接入 Occ3D-grid SurroundOcc candidate 后，`semantic_mIoU_common_lidar_mask` 从 keyframe Stage 3 的 `0.03446503745865521` 到 `0.03489001319157986`，有小幅提升；下一步应继续比较 sweep window 和过滤策略，而不是盲目增加 sweep 数。
- 根据官方 `mask_lidar` 是 sequence-level accumulative 的判断，已新增 `--sweep-direction scene`，对每个目标 keyframe 使用同一 scene 内全部 LIDAR_TOP sample_data。`scene-0061` 中每帧使用 `382` 个 LIDAR_TOP sweep，其中 `39` 个 keyframe 有 lidarseg。
- scene-level 输出为 `data/GT_occupancy_mini/stage2_raycast_occ3d_scene/`，摘要为 `data/GT_occupancy_mini/stage2_raycast_occ3d_scene/summary_stage2_scene.md`。与官方 Occ3D 对比：`mask_lidar` recall 提升到 `0.8092614775408034`，occupied IoU 提升到 `0.0458034907228422`，但 precision 降到 `0.16836881147551994`。这说明 sequence-level 是正确方向，但简单把全序列 free path 全量 union 会过宽，官方 Occ3D 还包含额外约束。
- scene-level Stage 2 接入 Occ3D-grid candidate 后，`semantic_mIoU_common_lidar_mask` 到 `0.037546636732510914`，是目前最好的一组；下一步应以 scene-level accumulation 为基础，收紧 over-coverage。

目标：

- 实现 Occ3D-style `occupied / free / unknown` 的核心几何逻辑。
- `mask_lidar` 完全由 raw LiDAR ray casting 决定。

执行内容：

- 从 raw LiDAR sweep 读取 points 和 lidarseg label。
- 在每个 sweep local frame 执行 `self_range` 过滤。
- 将过滤后的 sweep points 和 sweep origin 变换到 keyframe LiDAR 坐标系。
- 实现 ray-grid clipping 和 3D DDA。
- 生成 `free_grid`、`raw_hit_grid`、`mask_lidar`。
- 按 Occ3D Appendix D 使用 `ray_start=point`、`ray_end=points_origin[i]` 的 per-point ray casting。
- 处理 `point / point_origin` 过滤、DDA 边界检查、ray 与 grid 无交集等边界情况。

验收成果：

- `raycast_debug.npz`，包含 `free_grid`、`raw_hit_grid`、`mask_lidar`。
- ray casting 统计文件，例如 `stats_raycast.json`。
- `mask_lidar` 和 free-space slice 的 `.ply` 或图片。

验收标准：

- `mask_lidar == 1` 的区域只来自 ray traversed voxels 或 raw hit voxels。
- `voxel_occ_count > 0` 覆盖同 voxel 上的 free，这是论文 Algorithm 2 的预期状态优先级。
- point 所在 `target_voxel` 通过 `voxel_occ_count` 标记 occupied，不应被同一条或其他 ray 的 free 覆盖。
- 如果引入 grid clipping 或 grid 外 endpoint 处理，需要标注为非论文精确扩展。
- unknown 比例合理，不应接近 0。
- 每个 sweep 的 origin 在 keyframe 坐标下可视化后位置合理。

### Stage 3: 语义融合

当前状态：

- 已实现 Stage 3 脚本：`tools/occ3d_stage3/fuse_stage3_semantics.py`。
- 已实现官方 Occ3D 对比脚本：`tools/occ3d_stage3/compare_occ3d_gt.py`。
- Stage 3 输出目录：`data/GT_occupancy_mini/stage3_fused_occ3d/gts/scene-0061/<sample_token>/labels.npz`。
- Stage 3 ann pkl：`data/GT_occupancy_mini/stage3_fused_occ3d/bevdetv2-nuscenes_infos_stage3_train.pkl`。
- Stage 3 统计：`data/GT_occupancy_mini/stage3_fused_occ3d/stats_semantic.json`。
- 与官方 Occ3D GT 的对比报告：`data/GT_occupancy_mini/stage3_fused_occ3d/compare_to_official_occ3d.json` 和 `data/GT_occupancy_mini/stage3_fused_occ3d/summary_stage3_compare.md`。
- FlashOCC loader smoke test：`data/GT_occupancy_mini/stage3_fused_occ3d/smoke_test_stage3_loader.json`。
- 融合策略为保守主线：`mask_lidar`、`free_grid`、`raw_hit_grid` 完全继承 Stage 2；SurroundOcc/Poisson dense candidate 先从 `[-50,-50,-5,50,50,3] / 0.5m` 转回 metric center，再投到 Occ3D `[-40,-40,-1,40,40,5.4] / 0.4m` 网格；只有落在 `raw_hit_grid` 上的 candidate 才能改写 occupied semantic。
- 39 帧统计：raw-hit voxel `133125`，投影到 Occ3D 网格内的 SurroundOcc candidate voxel `857979`，其中与 raw-hit 重叠 `51168`，candidate-only `806811` 被忽略，最终仅 `875` 个 raw-hit voxel 的语义被融合改写。
- Stage 2 与 Stage 3 的 `mask_lidar` 在 39/39 帧完全一致，说明 Stage 3 没有扩展 observed 区域。
- 与官方 Occ3D GT 对比后，Stage 3 相比 Stage 2 只有极小幅语义变化：`semantic_mIoU_common_lidar_mask` 从 `0.03446198972058798` 到 `0.03448132094433069`。主瓶颈仍是几何 observed mask 与官方 Occ3D 差异，而不是 Stage 3 的语义候选。
- 重新用 Occ3D 网格配置运行 SurroundOcc GT 生成后，已基于 `data/GT_occupancy_mini_occ3d_grid/dense_voxels_with_semantic` 重新执行 Stage 3，输出为 `data/GT_occupancy_mini/stage3_fused_occ3d_regenerated_candidate/`。candidate 与 raw-hit 的重叠从旧 candidate 的 `51168` 提升到 `128480`，但与官方 Occ3D 的语义指标仍几乎不变，说明当前主要瓶颈继续集中在 Stage 2 observed mask 几何，而不是 Stage 3 candidate 网格量化。
- 针对官方 Occ3D `mask_lidar` 可能使用 sequence-level accumulative visibility 的问题，已实现 full-scene sweep 累积、动态物体 free-path 截断、以及 per-ray `lidar_max_range` 限制。`scene-0061` 上 `scene + truncate_protected_free + lidar_max_range=50` 的 Stage 2 输出为 `data/GT_occupancy_mini/stage2_raycast_occ3d_scene_protect_range50/`，summary 为 `summary_stage2_scene_protect_range50.md`。与官方 Occ3D 对比：`mask_lidar_iou=0.1627172658765752`，`precision=0.17012567686941837`，`recall=0.7888787990080235`。相比 naive full-scene 累积，precision 只从约 `0.1684` 提升到 `0.1701`，recall 从约 `0.8093` 降到 `0.7889`，说明这两个约束在语义上合理，但不是当前 observed mask 过宽的主因。
- 基于上述 Stage 2 结果重新执行 Stage 3，输出为 `data/GT_occupancy_mini/stage3_fused_occ3d_scene_protect_range50_regenerated_candidate/`。Stage 3 后 `semantic_mIoU_common_lidar_mask=0.03760286130561435`，略高于未加约束的 scene-level Stage 3，但几何指标不变。后续更应聚焦复现官方 Occ3D 的 exact accumulative visibility 策略，而不是继续只调动态框截断或固定最大射程。
- 根据论文 Appendix D Algorithm 2，Stage 2 已新增 dense candidate surface raycasting：`--ray-source dense-candidate` 使用 `data/GT_occupancy_mini_occ3d_grid/dense_voxels_with_semantic` 的 Occ3D-grid `(x,y,z,semantic)` 作为 surface endpoint；`--surface-free-distance` 先将 surface voxel 标为 occupied，再从 surface point 朝 keyframe LiDAR origin 方向短距离标 free。实验输出和对比表见 `data/GT_occupancy_mini/stage2_raycast_occ3d_dense_candidate_surface15/summary_stage2_dense_candidate_surface.md`。该实验直接解释了 observed mask 量级问题：raw scene + dynamic protect + 50m 为 `17953980` observed；dense candidate + full free path 降为 `8595316`；dense candidate + surface free `1.5m` 降为 `3772497`，已接近官方 Occ3D 的 `3871866`。当前剩余问题从“observed 数量过宽”转为“空间 overlap 仍低”，`surface_free_distance=1.5m` 时 `mask_lidar_iou=0.10368963364990473`。后续应继续复现官方 dense point source、`DISTANCE=0.5` 的真实单位/终止条件，以及每个 dense point 对应的原始或最近 sweep origin。
- 进一步验证后，`surface_free_distance=0.2m` 因不足以跨出 `0.4m` voxel，free voxel 为 `0`，`0.4m` 指标也接近 `0.5m`，因此论文 `DISTANCE=0.5` 不宜简单解释为 metric `0.2m`，需要按官方 DDA 终止条件复现。PLY 和离散坐标搜索发现 dense candidate 坐标存在主要 XY 约定错位：使用 `--dense-coordinate-transform swapxy_flipy` (`new_x=old_y`, `new_y=199-old_x`) 后，`surface_free_distance=1.5m` 的全局指标显著改善：`mask_lidar_iou=0.18098803581638717`，`precision=0.3105370233170888`，`recall=0.30257193818174494`，`occupied_iou_full_grid=0.1627559361619307`，`semantic_mIoU_common_lidar_mask=0.25371749680353517`。第一帧 PLY 统计也显示 Y/Z 基本对齐，剩余主要是 X 方向差异，后续应聚焦官方 dense point source、per-point origin 和 exact DDA。
- `swapxy_flipy` 后的 full surface-to-origin free-path 对照显示：`observed=8595189`，`mask_lidar_iou=0.24282502741689624`，`precision=0.283394815401965`，`recall=0.6291106148818166`。这说明官方 mask 很可能包含长 free-path 结构；但当前 full-path 版本 false positive 过多，仍缺官方 DDA 停止/裁剪逻辑或 per-point origin 约束。
- 已执行“方案一”：在 `tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py` 中新增 `ray_source_voxels_with_semantic/` 输出，并提供 `--ray_source_only` 以便在无 CUDA 环境下只生成 Poisson 前的 LiDAR-supported ray source；在 Stage 2 中新增 `--dense-ray-source-dir`，使 occupied 继续来自 `dense_voxels_with_semantic/`，free ray 只来自 `ray_source_voxels_with_semantic/`。mini scene-0061 上 ray-source voxel 总数为 `1185081`，dense occupied candidate 为 `1905956`。方案一 full path：`observed=7570121`，`mask_lidar_iou=0.24250548551266496`，`precision=0.29499990819169203`，`recall=0.5767722849912678`；方案一 surface `1.5m`：`observed=3188524`，`mask_lidar_iou=0.17233910742101138`，`precision=0.32551393685605`，`recall=0.2680642873487874`，`semantic_mIoU_common_lidar_mask=0.2789142691247571`。结论：mesh 补全点不应全部发 ray，这能提高 precision；但 mask IoU 没有显著超过 full dense-ray 版本，剩余瓶颈仍是 official DDA/free-path stopping 与 per-point origin。
- 已升级到 point-level ray source：`generate_occupancy_nuscenes.py --ray_source_only` 新增 `ray_source_points_with_origin/`，每行保存 `(point_x, point_y, point_z, origin_x, origin_y, origin_z, semantic)`，用于对齐 Occ3D Algorithm 2 的 `points[i] / points_origin[i]` 等长输入。mini scene-0061 共生成 39 个文件、`31870362` 条 point-origin ray 记录，origin 非零比例约 `98.7%`。Stage 2 新增 `--dense-ray-source-points-dir`，occupied 仍来自 `dense_voxels_with_semantic/`，free ray 来自逐点 origin。`swapxy_flipy + full path` 输出为 `data/GT_occupancy_mini/stage2_raycast_occ3d_dense_point_origin_fullpath_swapxy_flipy/`：`observed=10464854`，`mask_lidar_iou=0.2853029279785081`，`precision=0.304100659216077`，`recall=0.8219212648371612`，`occupied_iou_full_grid=0.1627559361619307`，`semantic_mIoU_common_lidar_mask=0.1720339043194299`。相比 keyframe-origin full path 的 `mask_lidar_iou=0.24282502741689624` 和 voxel-level ray source full path 的 `0.24250548551266496`，逐点 origin 明显改善 observed mask 空间对齐。
- Stage 2 已新增 `--ray-traversal occ3d-point-to-origin`，用于按论文方向从 hit point 向 `points_origin` 做 DDA，并排除 hit voxel 和 origin voxel。mini scene-0061 输出为 `data/GT_occupancy_mini/stage2_raycast_occ3d_dense_point_origin_occ3d_dda_swapxy_flipy/`：`observed=10462659`，`mask_lidar_iou=0.28531228967837163`，`precision=0.30412565295303995`，`recall=0.821816405836359`，`semantic_mIoU_common_lidar_mask=0.17205603130546765`。相比旧 full-path traversal 只提升约 `9.36e-06` IoU，说明 endpoint/origin voxel 边界规则不是当前主要瓶颈；后续应优先处理 point-origin 来源的精确性、动态物体点 origin 近似、以及 Stage 3 对 `swapxy_flipy` 的一致支持。
- Stage 3 已新增 `--candidate-grid {surroundocc,occ3d}` 和 `--dense-coordinate-transform {identity,swapxy_flipy}`，确保语义候选和 Stage 2 使用相同坐标约定。基于当前最强 Stage 2 运行 `candidate-grid=occ3d + swapxy_flipy`，输出为 `data/GT_occupancy_mini/stage3_fused_occ3d_point_origin_occ3d_dda_swapxy_flipy/`。统计显示 `raw_hit_count=1905956`、`candidate_occ_count=1905956`、`candidate_on_raw_hit_count=1905956`、`candidate_only_count=0`、`changed_semantic_count=0`，因此在这条主线上 Stage 3 是 no-op：Stage 2 已经把 transformed dense candidate 作为 occupied hit/semantic 写入。官方对比与 Stage 2 完全一致：`mask_lidar_iou=0.28531228967837163`，`semantic_mIoU_common_lidar_mask=0.17205603130546765`。结论：坐标一致性问题已修复，但 conservative Stage 3 当前不会继续提升；若要提升语义，需要设计新的 densification/official-mask upper-bound 实验，而不是重复跑同一 conservative fusion。
- 已完成 official-mask semantic upper bound，输出为 `data/GT_occupancy_mini/official_mask_semantic_upper_bound_swapxy/`。该实验直接复制官方 `mask_lidar/mask_camera`，语义只使用 `data/GT_occupancy_mini_occ3d_grid/dense_voxels_with_semantic` 经 `swapxy_flipy` 后的 candidate。结果：`mask_lidar_iou=1.0`，但 `semantic_miou_official_lidar_mask=0.14853824699315687`，`candidate_official_occupied_iou_full_grid=0.1627559361619307`，`candidate_occupied_official_lidar_fraction=0.35004638092379886`。结论：即使 mask 完全正确，当前 dense candidate 的 occupied 空间分布/语义也只能达到约 `0.1485` mIoU；后续不能只优化 free-space mask，还需要追查 candidate occupied 与官方 occupied 的空间差异。
- 已完成 mask FP/FN PLY 诊断，输出为 `data/GT_occupancy_mini/stage3_fused_occ3d_point_origin_occ3d_dda_swapxy_flipy/mask_diff_ply/`。汇总：`tp=3181963`、`fp=7280696`、`fn=689903`，说明主误差是 over-coverage；worst IoU 的前 3 帧已导出 `tp.ply/fp.ply/fn.ply/tp_fp_fn_combined.ply`。空间统计显示 FP 主要集中在低 Z 层 `1-5`，FN 更偏高层 `8-12`，且 FN 均值约为 `[-9.60, 6.99, 2.78]`。下一步应优先可视化这些 PLY，判断 FP 是否为 ground/low free-path 过度扩张，FN 是否来自上方结构或剩余坐标/高度偏移。
- 已完成 path length sweep + z-layer 统计，摘要为 `data/GT_occupancy_mini/path_length_sweep_point_origin_swapxy_summary.md`。实验使用 per-point origin + `swapxy_flipy`，将 surface-to-origin free path 截断为 `2m/4m/6m/8m/12m`，并与 full path 对比。结果显示 IoU 随路径长度单调上升：`2m=0.209948`、`4m=0.243066`、`6m=0.255266`、`8m=0.262783`、`12m=0.273880`、`full=0.285312`；full path 仍最好。短路径 precision 略高但 recall 大幅下降，说明单一 metric 截断不是主解。所有距离下 FP 都稳定集中在低 Z 层 `1-3`，FN 则集中在中高 Z 层，下一步应做 z-aware 或 source-aware 的 free-write 约束，而不是继续只扫全局 path length。
- 已完成 LiDAR-supported occupied ablation，摘要为 `data/GT_occupancy_mini/stage2_raycast_occ3d_lidar_supported_occ_point_origin_fullpath_swapxy_flipy/summary_lidar_supported_occupied_ablation.md`。实验将 occupied source 从 `dense_voxels_with_semantic` 改为 `ray_source_voxels_with_semantic`，free ray 仍使用 `ray_source_points_with_origin`。结果：occupied 从 `1905956` 降到 `1185081`，observed 从 `10462659` 降到 `10290693`，`mask_lidar_iou` 仅从 `0.285312` 微升到 `0.285759`，precision 从 `0.304126` 到 `0.305870`，但 recall 从 `0.821816` 降到 `0.812946`，occupied IoU 从 `0.162756` 降到 `0.133412`，semantic common mIoU 从 `0.172056` 降到约 `0.153`。结论：Mesh/dense-only occupied 不是大面积 mask FP 的主因，简单把 Mesh completion 当 unknown 不合适；主要 FP 仍来自 free-space ray traversal，Mesh/dense completion 对 occupied geometry/semantic 有贡献，但不能随意生成 free ray。
- 已完成 source-aware free-ray label ablation，摘要为 `data/GT_occupancy_mini/source_aware_free_ray_ablation_summary.md`。代码检查确认当前最强主线的 `ray_source_points_with_origin/` 是在 Poisson/mesh completion 之前保存的，因此 Mesh completion 已经没有作为 free-ray endpoint。进一步实验：排除动态类 free ray (`2,3,4,5,6,7,9,10`) 后 `mask_lidar_iou=0.283950`，precision 略升但 recall 下降；排除 flat 类 free ray (`11,12,13,14`) 后 `mask_lidar_iou=0.285309`，几乎等于 baseline。结论：低层 FP 不是单一语义组造成，也不是 Mesh 点发 free ray 造成；下一步需要在生成阶段记录 point provenance（static raw、static KNN、dynamic original-origin、dynamic approximate-origin、mesh sampled），按来源而不是只按 semantic label 做 free-ray ablation。
- 已完成 ray origin 与 `swapxy_flipy` 轻量验证，摘要为 `data/GT_occupancy_mini/ray_origin_swapxy_validation_summary.md`，详细报告为 `data/GT_occupancy_mini/ray_origin_swapxy_validation.json`。metric `swapxy_flipy` 后 ray point voxel 与 transformed `ray_source_voxels_with_semantic` 的重合率为 `0.9999949370592077`；不做 metric transform 时只有 `0.052947390982227394`。与 dense candidate 的重合率也从 identity 的 `0.08284490018412229` 提升到 `0.91185926374715`。结论：当前代码中 occupied candidate 和 free-ray metric point/origin 的坐标变换是一致的，`swapxy_flipy` 不一致基本排除。origin 分布没有明显全局变换错误，但由于缺少 provenance，不能单独验证 keyframe raw / non-keyframe / dynamic transformed 各自的 origin 质量。

目标：

- 在 raw ray observation 约束内写入 occupied 语义。
- 避免 Poisson-only dense occupancy 扩展 `mask_lidar` 或覆盖 raw free。

执行内容：

- 按论文 Algorithm 2 对 LiDAR hit voxel 写入 `voxel_label[target_voxel] = point_label`；majority voting 只作为非论文实验分支。
- 构建 `poisson_occ_grid`，仅作为语义候选或对比项。
- 主线实现中优先使用 `voxel_occ_count > 0` 决定 occupied state；Poisson/mesh dense candidate 是否参与 occupied densification 需与 ray source 分离记录。
- 明确记录 label `0` 的处理策略。
- 写入顺序固定为：先 free，后 raw occupied hit。

验收成果：

- 带最终 `semantics`、`mask_lidar` 的 `labels.npz`。
- `stats_semantic.json`，包含每类 occupied voxel 数、free 数、unknown 数、label 覆盖/冲突统计。
- occupied semantic `.ply` 可视化。

验收标准：

- `mask_lidar == 0` 的 voxel 不被 Poisson dense occupancy 改成 observed。
- Poisson-only occupied candidate 不覆盖 raw free。
- 论文精确主线不使用 majority voting；同 voxel 多语义命中按 Algorithm 2 的逐点写入语义处理。若使用 majority voting，必须作为非论文实验分支单独记录。
- `semantics == 17 and mask_lidar == 1` 表示 free。
- `mask_lidar == 0` 区域不参与训练/eval，即使导出时 `semantics` 被置为 17。

### Stage 4: Camera Mask

目标：

- 生成基础可用的 `mask_camera`，支持 FlashOcc camera-visible 监督和评估。

执行内容：

- 依赖载入： 读取 Stage 3 生成的 3D 体素状态 voxel_state_lidar (包含 OCCUPIED, FREE, NOT_OBSERVED)。
- 像素射线生成 (Pixel Ray Generation)： 为 6 个 Camera 的图像范围生成二维网格 (uv)，结合内参和外参，在自车坐标系下生成从相机光心到远处的 3D 射线 (Ray)。
- 可见性遍历与遮挡截断 (Raycasting & Occlusion)：沿射线遍历 3D 网格，继承 LiDAR 状态。当遇到第一个 OCCUPIED 体素时，标记为可见并 break 停止当前射线。
- 状态聚合： 多相机产生的可见性结果取 OR 聚合。
- Known Issue 记录： 暂不做动态物体的跨帧时间戳运动补偿，在日志中记录由此带来的 Ghost Occlusion 局限性。

验收成果：

- 完整 `labels.npz`，包含 `semantics`、`mask_lidar`、`mask_camera`。（不再出现被建筑遮挡区域被标记为可见的低级错误）。
- 每个 camera 的 visibility voxel 投影到 2D 图像的可视化（需验证近处物体正确遮挡了远处物体）
- `stats_camera_mask.json`（记录可见与不可见体素的比例变化）。

验收标准：

- mask_camera 的占用状态不应超出 mask_lidar 的物理范围。
- (新增核心标准) 投影结果中，前景物体（如车辆、墙壁）必须在 mask_camera 中形成正确的后方阴影（盲区），证明遮挡逻辑生效。
- 没有明显相机顺序错误、内参错误或外参反向。
- 动态物体附近允许轻微边界误差，但需要在日志中记录基础版未做时间戳补偿。

### Stage 5: FlashOcc 集成与训练 Smoke Test

目标：

- 验证生成的 GT 能被 FlashOcc 训练链路实际消费。

执行内容：

- 修改或配置 FlashOcc 数据路径指向新生成的 `gts`。
- 单帧 dataloader 检查。
- 单 batch forward/backward。
- mini split 训练若干 iteration。
- 保存 GT 和预测的可视化结果。

验收成果：

- FlashOcc dataloader 日志。
- forward/backward smoke test 日志。
- mini split 训练日志和 loss 曲线。
- GT/prediction 对比可视化。

验收标准：

- dataloader 无 shape、key、路径错误。
- forward/backward 正常，无 NaN/Inf。
- loss 数值在合理范围内，短训过程中没有立即发散。
- 预测和 GT 在坐标上对齐，没有整体偏移或轴翻转。

### Stage 6: 小规模质量评估

目标：

- 在多个 scene 上确认 GT 分布稳定，避免单帧调通但全量生成崩掉。

执行内容：

- 选择 mini split 或若干 train/val scene 批量生成。
- 汇总每帧 voxel 数、free/occupied/unknown 比例、类别分布。
- 挑选正常帧和异常帧分别可视化。
- 对比已有 FlashOcc/Occ3D GT 或人工检查结果。

验收成果：

- `summary_quality_report.md`。
- `stats_all_frames.csv`。
- 异常帧列表和对应可视化。

验收标准：

- 不同 scene 的 free/occupied/unknown 比例没有大面积异常跳变。
- 类别分布不出现系统性缺类。
- 过滤点数量、ray 数量、hit 数量与帧点云规模基本匹配。
- 抽样可视化通过人工检查。

### Stage 7: Image-guided Refinement

目标：

- 在 LiDAR occlusion reasoning 主线稳定后，再尝试图像引导增强。

执行内容：

- 改进 `mask_camera` 的遮挡判断。
- 可选加入 timestamp/motion compensation。
- 可选利用图像语义或深度修正 camera-visible 边界。
- 保留 refinement 前后的 GT，不覆盖主线结果。

验收成果：

- refinement 前后统计对比。
- refinement 前后可视化对比。
- 对 FlashOcc mini training 的影响对比。

验收标准：

- refinement 不覆盖 raw LiDAR hit 的 occupied 语义。
- refinement 不无声扩展 `mask_lidar`。
- camera-visible 区域边界更合理。
- 训练或验证指标不退化，至少可解释其变化来源。

## 10. Image-guided Refinement

Image-guided Refinement 建议放在验证闭环之后，作为后续增强层，而不是第一版核心实现的一部分。

原因：

- 它会引入图像语义、深度估计、遮挡判断等额外误差源。
- 如果和 ray casting 同时实现，难以定位错误来自 GT 格式、LiDAR occlusion reasoning，还是 image refinement。

建议后续作为 Layer 4：

```text
Layer 1: 格式转换
Layer 2: LiDAR occlusion reasoning
Layer 3: FlashOcc 验证闭环
Layer 4: Image-guided refinement
```

Image-guided refinement 的原则：

- 不轻易覆盖 LiDAR hit 的 occupied voxel。
- 优先用于 camera-visible 区域的边界修正。
- 可以对 unknown/free 的局部区域做补充，但需要记录 refinement 来源。
- refinement 前后分别保存统计和可视化，避免无声改变监督分布。

## 11. 当前结论

修订方案是合理的，并且比初版计划更接近 Occ3D 的真实监督定义。最重要的三处修正是：

1. 每个聚合 LiDAR point 必须保存并使用自己的真实 `points_origin`。
2. Occ3D LiDAR visibility 按 `ray_start=point`、`ray_end=points_origin[i]` 做 DDA，而不是把所有点统一从 keyframe origin 发 ray。
3. 最终状态写入必须遵守 `FREE` 先写、`OCCUPIED` 后写的优先级，即 `voxel_occ_count > 0` 覆盖 `voxel_free_count > 0`。

这三点决定了 `mask_lidar` 的几何正确性。只要 `mask_lidar` 正确，`semantics == 17` 才能可靠地区分被确认的 free space；否则训练时很容易把 unknown 区域错误当成 free，破坏 FlashOcc 的监督信号。

## 12. 当前实验记录

### Low-Z Free-Write Ablation

在 point-level `points_origin`、`swapxy_flipy`、Occ3D-style point-to-origin DDA 的当前最强 Stage 2 基线上，直接限制低 Z 层 free voxel 写入。

结果汇总见：

`data/GT_occupancy_mini/low_z_free_write_ablation_summary.md`

关键结果：

| variant | observed | mask IoU | precision | recall | common mIoU |
|---|---:|---:|---:|---:|---:|
| baseline | 10,462,659 | 0.285312 | 0.304126 | 0.821816 | 0.172056 |
| skip free z=1-3 | 8,339,146 | 0.311727 | 0.347986 | 0.749485 | 0.210623 |
| skip free z=1-4 | 7,595,390 | 0.315250 | 0.361873 | 0.709883 | 0.227520 |
| skip free z=1-5 | 6,861,077 | 0.301435 | 0.362325 | 0.642052 | 0.244194 |
| skip free z=0-4 | 7,093,721 | 0.302788 | 0.359272 | 0.658229 | 0.291646 |

当前最优 coarse rule 是 `skip free z=1-4`。它将 `mask_lidar_iou` 从 0.285312 提升到 0.315250，FP 从 7,280,696 降到 4,846,820，同时保留 0.709883 recall。`z=0-*` 系列没有超过 `z=1-4`，说明 z=0 仍包含一部分官方可见 free space，不宜全局禁写。

这说明低 Z free ray 过扩散是当前 Stage 2 的主要误差来源之一。后续应优先把这个 coarse rule 细化成 grazing-angle 或 per-distance/per-z gating，而不是马上进入 provenance 记录。

### FlashOcc Mini Training Smoke

使用当前 baseline Stage 3 GT 运行 FlashOcc mini 训练 smoke：

- 39/39 iterations 完成。
- `loss_occ` 从 2.8823 降到 1.3225。
- checkpoint 写入：`data/GT_occupancy_mini/flashocc_train_smoke_baseline_full/epoch_1.pth`。
- 使用该 checkpoint 在 39 个 mini samples 上 eval 完成，`mIoU=0.65`，不是全零。

这说明当前 GT 格式已能被 FlashOcc 正常加载、训练和评估，并产生有效梯度。下一步可用 `skip free z=1-4` 版本重新生成 Stage 3，并跑同样的 small train/eval 对比训练指标。

### Z-Gating Refinement

基于 `skip free z=1-4` 继续细化了低 Z free-write 规则，结果见：

`data/GT_occupancy_mini/z_gating_refinement_summary.md`

新增 Stage 2 参数：

- `--skip-free-z-grazing-angle-deg`：只对近水平 ray 应用低 Z 禁写。
- `--skip-free-z-min-ray-length`：只对超过指定长度的 ray 应用低 Z 禁写。

核心结果：

| variant | observed | mask IoU | precision | recall | common mIoU |
|---|---:|---:|---:|---:|---:|
| baseline | 10,462,659 | 0.285312 | 0.304126 | 0.821816 | 0.172056 |
| coarse z=1-4 | 7,595,390 | 0.315250 | 0.361873 | 0.709883 | 0.227520 |
| grazing 15deg | 7,700,478 | 0.316029 | 0.360882 | 0.717732 | 0.222912 |
| min ray length 12m | 8,288,250 | 0.315591 | 0.351948 | 0.753392 | 0.196200 |

当前 best mask IoU 是 `grazing 15deg`，输出为：

`data/GT_occupancy_mini/stage3_fused_occ3d_point_origin_z1_4_grazing15_swapxy_flipy`

随后在 `12/14/15/16/18/20deg` 做了细扫：

| variant | observed | mask IoU | precision | recall | common mIoU |
|---|---:|---:|---:|---:|---:|
| coarse z=1-4 | 7,595,390 | 0.315250 | 0.361873 | 0.709883 | 0.227520 |
| grazing 12deg | 7,773,583 | 0.315332 | 0.359143 | 0.721055 | 0.220658 |
| grazing 14deg | 7,713,649 | 0.316013 | 0.360662 | 0.718522 | 0.222404 |
| grazing 15deg | 7,700,478 | 0.316029 | 0.360882 | 0.717732 | 0.222912 |
| grazing 16deg | 7,691,602 | 0.315981 | 0.360980 | 0.717099 | 0.223274 |
| grazing 18deg | 7,677,014 | 0.315929 | 0.361164 | 0.716105 | 0.223891 |
| grazing 20deg | 7,665,140 | 0.315840 | 0.361274 | 0.715215 | 0.224458 |

`grazing 15deg` 仍是几何 mask IoU 最优，但相比 coarse `z=1-4` 只提升 `0.000779`。FlashOcc 1-epoch smoke eval 中，coarse `z=1-4` 为 `mIoU=2.27`，`grazing15` 为 `mIoU=1.83`。因此当前不建议直接用 `grazing15` 替代 coarse `z=1-4` 作为默认 Stage 2；更稳妥的默认仍是 coarse `z=1-4`，`grazing15` 保留为几何 ablation 候选。

### Stage 4 Camera Ray Mask

已实现 Stage 4 camera visibility，不再使用 `mask_camera = mask_lidar` 占位。实现脚本：

`tools/occ3d_stage4/build_camera_mask.py`

当前默认输入为 coarse `skip free z=1-4` 的 Stage 3：

`data/GT_occupancy_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy/bevdetv2-nuscenes_infos_stage3_train.pkl`

输出为：

`data/GT_occupancy_mini/stage4_camera_raymask_z1_4/`

实现要点：

- 从 6 个 camera 的原始像素 `(uv)` 出发发射射线，默认 `image_size=1600x900`、`DEPTH_MAX=45.0m`。
- 使用 ann pkl 中的 `cam_intrinsic`、`sensor2lidar_rotation`、`sensor2lidar_translation`。
- 沿 voxel grid 做 DDA；`mask_lidar == 0` 的体素即使被 camera ray 穿过也保持 camera NOT_OBSERVED。
- ray 只继承 LiDAR 已判定的 `FREE/OCCUPIED` 状态；遇到第一个 `semantics != 17` 且 `mask_lidar == 1` 的 occupied voxel 后立即停止该 ray。
- 多相机 `mask_camera` 取 OR。
- 导出时断言 `mask_camera <= mask_lidar`。

生成与检查结果：

- 39/39 帧生成完成。
- `mask_lidar_count = 7,595,390`
- `mask_camera_count = 2,454,520`
- `camera_occ_count = 162,760`
- `camera_free_count = 2,291,760`
- `mask_camera <= mask_lidar` 检查通过。
- 输出检查文件：
  - `data/GT_occupancy_mini/stage4_camera_raymask_z1_4/check_stage4_format.json`
  - `data/GT_occupancy_mini/stage4_camera_raymask_z1_4/check_occ3d_npz_stage4.json`

与官方 Occ3D `mask_camera` 对比：

`data/GT_occupancy_mini/stage4_camera_raymask_z1_4/compare_official_camera.json`

| metric | value |
|---|---:|
| mask_camera_iou | 0.247086 |
| mask_camera_precision | 0.455436 |
| mask_camera_recall | 0.350696 |
| semantic_miou_ref_camera_mask | 0.138393 |

这说明 Stage 4 已经把 camera mask 从占位的 LiDAR-wide mask 收紧到 camera-visible 子集，但与 official Occ3D camera visibility 仍有明显空间差异。后续如果继续对齐 official，应优先分析 Stage 4 的 FN/FP 来源，而不是仅看 FlashOcc 1-epoch smoke mIoU。

### Fixed-Prediction Mask Evaluation

为了避免把“预测不同”和“mask 不同”混在一起，已固定同一组 FlashOcc 预测，在三个固定 mask 下离线评估：

- 预测来源：`data/GT_occupancy_mini/flashocc_train_smoke_stage4_camera_z1_4/epoch_1.pth`
- 预测文件：`data/GT_occupancy_mini/flashocc_train_smoke_stage4_camera_z1_4/preds_epoch1.pkl`
- 评估脚本：`tools/occ3d_stage4/evaluate_fixed_predictions_masks.py`
- 输出报告：`data/GT_occupancy_mini/flashocc_train_smoke_stage4_camera_z1_4/fixed_mask_eval_epoch1.json`

同一组预测下，三种 mask 的 mIoU 为：

| eval mask | valid voxels | mIoU |
|---|---:|---:|
| official mask_camera | 3,187,595 | 0.395460 |
| Stage 4 mask_camera | 2,454,520 | 1.072979 |
| placeholder mask_lidar | 7,595,390 | 1.061848 |

结论：

- Stage 4 训练链路是通的：1-epoch smoke 中 `loss_occ` 从约 `2.8364` 降到约 `0.8442`。
- Stage 4 自身 eval log 中 `mIoU=1.07`，但它只是 train/eval 同一 mini scene 上的 smoke 指标。
- 之前记录的 coarse `z=1-4` 占位版 `mIoU=2.27` 与 Stage 4 `mIoU=1.07` 不是严格公平对比，因为它们来自不同训练 run / 不同 checkpoint / 不同预测。
- 固定同一组 Stage 4 预测后，`Stage 4 mask_camera` 与 `placeholder mask_lidar` 的 mIoU 非常接近（`1.072979` vs `1.061848`），说明单纯换 eval mask 并不能解释之前 `2.27 -> 1.07` 的差距。
- official `mask_camera` 下 mIoU 更低（`0.395460`），说明当前预测和 official camera-visible GT 分布仍有明显不一致；这更像是 GT geometry/semantic 与 official 的系统差异，而不是 FlashOcc loader 或 loss 链路问题。

因此，后续不能用单次 1-epoch mini smoke 的 mIoU 作为 Stage 4 好坏的唯一依据。更合理的协议是：固定预测比较不同 mask，或固定 mask 比较不同 GT/训练策略；如果要看训练收益，至少应跑多 epoch 并保持 train/eval split、初始化和评价 mask 一致。

### Surface Quality Short-Term Ablation (scene-0103)

为验证“Poisson mesh 只体素化 vertices 导致表面孔洞”以及“flat 类一格孔洞影响 camera first-hit”的假设，已实现并执行单 scene 对照。实验只替换 occupied candidate，Stage 2 固定使用 coarse `skip free z=1-4`、逐点 origin、`swapxy_flipy` 和 Occ3D point-to-origin DDA；Stage 4 固定使用 6-camera pixel ray、`DEPTH_MAX=45m`。

实现内容：

- 新增 `tools/generate_occupancy_nuscenes/surface_completion.py`：
  - `vertices`：保留原 mesh vertex voxelization；
  - `uniform`：按 mesh area/spacing 做 surface sampling；
  - `triangle`：调用 Open3D triangle-mesh voxelization，覆盖三角面内部 voxel；
  - `fill_flat_height_holes`：只对 `11/12/13/14` flat 类填补同类邻居数足够、z spread 受限的一格 XY 孔洞。
- `generate_occupancy_nuscenes.py` 新增 `--surface_mode {vertices,uniform,triangle}`、`--surface_sample_spacing` 和 `--flat_height_fill`，默认仍为旧 `vertices` 行为。
- 新增 `tools/generate_occupancy_nuscenes/apply_flat_fill_candidates.py`，允许对已有 candidate 离线执行 flat fill，避免重复 Poisson。
- 新增 `tools/occ3d_stage4/analyze_occupied_retention.py`，统一统计逐类 `global -> lidar -> camera` occupied 保留率，并检查 `mask_camera <= mask_lidar`。
- 新增单元测试：`tests/test_surface_completion.py`、`tests/test_occupied_retention.py`。

triangle candidate 生成命令：

```bash
/home/fjm/miniconda3/envs/flashocc/bin/python \
  tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py \
  --dataset nuscenes --config_path tools/generate_occupancy_nuscenes/config.yaml \
  --split all --save_path data/GT_occupancy_surface_ablation/triangle_scene0103 \
  --start 1 --end 2 --dataroot data/nuscenes --version v1.0-mini \
  --nusc_val_list tools/generate_occupancy_nuscenes/nuscenes_val_list.txt \
  --label_mapping tools/generate_occupancy_nuscenes/nuscenes.yaml \
  --surface_mode triangle
```

flat-fill 参数固定为：`radius=1`、`min_neighbors=5`、`max_z_spread=1`。Stage 2/4 输出分别位于：

```text
data/GT_occupancy_surface_ablation/stage2_triangle_scene0103/
data/GT_occupancy_surface_ablation/stage2_triangle_flat_scene0103/
data/GT_occupancy_surface_ablation/stage4_vertices_scene0103/
data/GT_occupancy_surface_ablation/stage4_triangle_scene0103/
data/GT_occupancy_surface_ablation/stage4_triangle_flat_scene0103/
```

40 帧 candidate 与 Stage 4 结果：

| variant | global occupied | camera occupied | camera/global | mask_camera | official mask IoU | semantic mIoU on official camera mask |
|---|---:|---:|---:|---:|---:|---:|
| vertices | 1,563,919 | 203,893 | 0.130373 | 2,619,851 | **0.383079** | 0.231831 |
| triangle | 1,640,764 | 203,502 | 0.124029 | 2,591,752 | 0.381140 | 0.235411 |
| triangle + flat-fill | 1,642,004 | 203,585 | 0.123986 | 2,591,753 | 0.381141 | **0.235477** |
| official Occ3D | 1,528,210 | 904,041 | 0.591568 | 3,091,606 | 1.0 | 1.0 |

补充统计：

- triangle 相比 vertices 增加 `76,845` occupied candidate，即 `+4.91%`；flat 四类从 `247,606` 增至 `254,093`，即 `+2.62%`。
- 保守 flat-fill 只新增 `1,240` voxel，占 triangle candidate 约 `0.076%`，说明 triangle 后满足该严格规则的一格 flat 孔洞很少。
- 所有 camera pixel rays 的 first-occupied hit events：vertices `153,153,024`，triangle `153,493,536`，triangle+flat `153,530,238`。triangle 确实让更多像素射线提前撞到表面。
- 但 unique camera occupied 没有增加：vertices `203,893`，triangle `203,502`，triangle+flat `203,585`。新增表面主要集中在已有前表面附近或其后方，并让部分 ray 更早停止，没有覆盖更多独立可见 occupied voxel。
- triangle 的 camera-visible `car` 增加 `230`、`pedestrian` 增加 `52`，但 `manmade` 减少 `529`、`driveable_surface` 减少 `124`、`sidewalk` 减少 `44`。
- triangle 的 official-camera semantic mIoU 从 `0.231831` 小幅升到 `0.235411`，但 official mask IoU 从 `0.383079` 降到 `0.381140`。这是“语义轻微改善、visibility 几何轻微退化”的 trade-off，不能据此直接替换当前 vertices 主线。

当前结论：

1. “只体素化 vertices”确实漏掉约 5% mesh surface voxel，但不是 camera-visible occupied 从 official `59.16%` 降到当前约 `13%` 的主因。
2. 单纯增加 triangle surface density 会增加 per-pixel first-hit events，却不会增加 unique camera-visible occupied，说明当前 Poisson 表面更可能存在空间分布、闭合/遮挡形态或语义传播问题，而不是仅有稀疏采样问题。
3. 当前保守 flat-fill 影响过小，不建议直接扩到全部 10 scene 或进行 24-epoch FlashOcc 训练。
4. 下一优先级应调整为：审计 Poisson 法向定向和连通/闭合面；按 flat/static/dynamic 分开重建；统计 occupied 在相机视锥内但未成为 first-hit 的数量与深度/z 分布。triangle 模式保留为 ablation，不替换冻结的 vertices + coarse `z=1-4` 默认主线。

验证结果：

```text
7 passed
```

测试命令：

```bash
/home/fjm/miniconda3/envs/flashocc/bin/python -m pytest \
  tests/test_surface_completion.py tests/test_occupied_retention.py -q
```

### Poisson Normals, Connectivity, Semantic Groups, and Frustum Diagnostics

在 triangle/flat-fill 只带来有限收益后，继续验证以下假设：

1. 场景聚合点来自不同 LiDAR origin，但旧实现把全部法向朝统一 `[0,0,0]` 定向，可能产生错误闭合面。
2. 全局 Poisson 产生大量小 connected components，可能在 camera ray 前方形成碎片遮挡。
3. flat/static/dynamic 混合重建导致类别传播和几何互相污染。
4. 大量 occupied 虽位于相机视锥内，但没有成为 first-hit，需要按 depth/z 定量分析。

实现内容：

- `surface_completion.py` 新增：
  - `orient_normals_toward_origins`：逐点使用 `point_origin - point` 翻转法向；
  - `mesh_topology_stats`：统计 connected components、watertight/manifold；
  - `filter_mesh_components`：按 triangle 数过滤小分量。
- `generate_occupancy_nuscenes.py`：
  - 为所有静态和动态几何点保留 origin；
  - 新增 `--normal_orientation {camera,point-origin}`；
  - 新增 `--min_component_triangles`；
  - 新增 `--max_keyframes`，支持单帧受控实验；
  - 新增 `--full_topology_stats`，只在诊断时计算昂贵的 watertight/manifold；
  - 新增 `--reconstruction_mode {global,semantic-groups}`。
- 新增 `semantic_reconstruction.py`：
  - flat `11-14`：LiDAR-supported voxel + conservative height fill；
  - manmade/vegetation `15/16`：分别执行 point-origin Poisson；
  - dynamic/small `1-10`：保留 LiDAR-supported voxel；
  - 冲突优先级：dynamic > flat > manmade > vegetation > other。
- 新增 `analyze_frustum_first_hit.py`：统计 `global / in_frustum / first_hit / frustum_non_first`，并输出 class、depth bin 和 z-layer histogram。

#### 单帧受控实验

使用 scene-0103 第一帧、triangle voxelization、冻结 Stage2/4 参数：

| variant | occupied | components | camera occupied | first-hit / frustum | official mask IoU | semantic mIoU on official camera mask |
|---|---:|---:|---:|---:|---:|---:|
| global camera-origin | 33,786 | 1,268 | 1,620 | 0.0486 | 0.203682 | **0.324751** |
| global point-origin | 30,483 | 1,368 | 1,499 | 0.0498 | 0.214083 | 0.319353 |
| point-origin + filter `<100` | 30,002 | 138 | 1,523 | 0.0513 | **0.222605** | 0.318539 |
| semantic-groups | 28,344 | group-specific | **1,898** | **0.0682** | 0.219614 | 0.246880 |
| official Occ3D | 21,619 | n/a | 14,392 | 0.6834 | 1.0 | 1.0 |

单帧 topology 发现：

- camera-origin mesh 有 `1,159,898` vertices、`2,308,445` triangles、`1,268` components，并非 watertight。
- point-origin mesh 翻转 `785,076` 个 normals，生成 `1,014,364` vertices、`2,012,651` triangles、`1,368` components，也并非 watertight。
- threshold 100 删除 `1,230` 个小 components、`20,781` triangles，但 occupied 只从 `30,483` 降到 `30,002`；小分量数量很多，但对最终 voxel 数影响有限。
- semantic-groups 明显增加 first-hit occupied，但 official-mask semantic mIoU 大幅下降，说明当前“static 分 class Poisson + dynamic/raw voxel”会丢失重要语义表面，不宜直接扩到全 scene。

#### 完整 scene-0103 point-origin + filter100

由于该分支在单帧同时改善 geometry IoU 且语义退化较小，已生成完整 40 帧：

```text
data/GT_occupancy_surface_ablation/point_origin_filter100_scene0103/
data/GT_occupancy_surface_ablation/stage2_point_origin_filter100_scene0103/
data/GT_occupancy_surface_ablation/stage4_point_origin_filter100_scene0103/
```

结果：

| variant | global occupied | camera occupied | camera/global | mask_camera | official mask IoU | semantic mIoU on official camera mask |
|---|---:|---:|---:|---:|---:|---:|
| vertices baseline | 1,563,919 | 203,893 | 0.130373 | 2,619,851 | 0.383079 | 0.231831 |
| triangle | 1,640,764 | 203,502 | 0.124029 | 2,591,752 | 0.381140 | **0.235411** |
| point-origin + filter100 | 1,455,434 | **210,261** | **0.144466** | **2,787,281** | **0.391482** | 0.220818 |
| official Occ3D | 1,528,210 | 904,041 | 0.591568 | 3,091,606 | 1.0 | 1.0 |

40 帧累计：

- point-origin 共翻转 `52,691,271` 个输入 normals。
- component filter 共删除 `24,325` 个小 components、`499,732` triangles。
- 相比 vertices，global occupied 减少 `108,485`，但 camera occupied 增加 `6,368`，mask IoU 增加 `0.008403`，recall 从 `0.511687` 增至 `0.534990`。
- semantic mIoU 从 `0.231831` 降至 `0.220818`。几何收益与语义退化同时存在，尚不能替换默认主线。
- connected-component 聚类对百万 triangle mesh 开销明显；全 10 scene 前需要并行化，或转成更便宜的 voxel-component filtering。

#### Frustum Non-First-Hit Depth/Z

scene-0103 汇总：

| variant | in-frustum occupied | first-hit occupied | frustum non-first | first-hit/frustum |
|---|---:|---:|---:|---:|
| vertices | 1,530,159 | 203,893 | 1,327,251 | 0.133250 |
| point-origin + filter100 | 1,423,271 | 210,261 | 1,213,982 | 0.147731 |
| official Occ3D | 1,494,301 | 904,041 | 608,286 | 0.604993 |

vertices non-first depth bins `[0-5,5-10,10-15,15-20,20-30,30-45]m`：

```text
[1,965, 40,913, 131,008, 193,687, 468,822, 490,856]
```

point-origin + filter100：

```text
[2,355, 39,806, 123,239, 177,632, 425,730, 445,220]
```

official：

```text
[5,504, 26,677, 70,369, 91,754, 196,059, 217,923]
```

结论：

1. point-origin normals + component filtering 是有效的 visibility geometry 改进方向，减少了约 `113k` frustum non-first occupied，并提升 mask IoU/recall。
2. 但 first-hit/frustum 仍只有 `14.77%`，远低于 official `60.50%`；主要差距集中在 `15-45m`，且低 z 层仍有大量非 first-hit occupied。全局 Poisson 的闭合/前表面过遮挡仍未解决。
3. semantic-groups 当前版本虽然减少遮挡、增加 first-hit，但语义表面严重不足；后续应保留全局几何或 non-keyframe geometry，再做 class-aware semantic assignment，而不是完全按 keyframe semantic points 分组重建。
4. 下一步推荐：以 point-origin normals 为基础，改为 voxel connected-component filtering；针对 15-45m 的 low-z/flat 表面做 source-aware reconstruction；将 geometry reconstruction 与 semantic assignment 解耦，避免几何改善带来语义 mIoU 下降。
