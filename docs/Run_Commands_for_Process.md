## 流程中各阶段命令

下面这套是当前最建议保留的“完整 nuscenes-mini GT 生成主线命令”：`coarse skip free z=1-4 + Stage4 camera mask`，也就是你已经证明能接 FlashOcc 训练的版本。

**0. 公共变量**

```bash
cd /home/fjm/shousuyao/SurroundOcc_my

PY=/home/fjm/miniconda3/envs/flashocc/bin/python
DATAROOT=/home/fjm/shousuyao/SurroundOcc_my/data/nuscenes
VERSION=v1.0-mini
ANN_TRAIN=/home/fjm/shousuyao/FlashOCC/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
ANN_VAL=/home/fjm/shousuyao/FlashOCC/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
```

如果你的 FlashOcc ann pkl 实际不在这个路径，把 `ANN_TRAIN/ANN_VAL` 改成当前可跑通的那两个 pkl。

**1. 生成 SurroundOcc/Poisson 候选 + point-origin ray source**

这一步是“真值生成”的第一大步，会输出 dense occupied candidate、LiDAR-supported voxel、逐点 origin ray source。

```bash
MPLCONFIGDIR=/tmp/matplotlib_flashocc \
$PY tools/generate_occupancy_nuscenes/generate_occupancy_nuscenes.py \
  --dataset nuscenes \
  --config_path tools/generate_occupancy_nuscenes/config.yaml \
  --split train \
  --save_path data/GT_occupancy_v1_0_mini_occ3d_grid_parallel \
  --dataroot $DATAROOT \
  --version $VERSION \
  --nusc_val_list tools/generate_occupancy_nuscenes/nuscenes_val_list.txt \
  --label_mapping tools/generate_occupancy_nuscenes/nuscenes.yaml
```

完成后重点检查这三个目录：

```bash
ls data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/dense_voxels_with_semantic | wc -l
ls data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/ray_source_voxels_with_semantic | wc -l
ls data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/ray_source_points_with_origin | wc -l
```

v1.0-mini 全部 10 个 scene 正常应接近 `404` 帧。

**2. Stage2 Train：生成 coarse z=1-4 LiDAR mask**

```bash
$PY tools/occ3d_stage2/raycast_occ3d_from_nuscenes.py \
  --dataroot $DATAROOT \
  --version $VERSION \
  --ann-file $ANN_TRAIN \
  --label-mapping tools/generate_occupancy_nuscenes/nuscenes.yaml \
  --output-root data/GT_occupancy_v1_0_mini/stage2_raycast_occ3d_dense_point_origin_skip_free_z1_4_swapxy_flipy_train \
  --ray-source dense-candidate \
  --dense-candidate-dir data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/dense_voxels_with_semantic \
  --dense-ray-source-points-dir data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/ray_source_points_with_origin \
  --dense-coordinate-transform swapxy_flipy \
  --ray-traversal occ3d-point-to-origin \
  --skip-free-z-min 1 \
  --skip-free-z-max 4
```

**3. Stage2 Val**

```bash
$PY tools/occ3d_stage2/raycast_occ3d_from_nuscenes.py \
  --dataroot $DATAROOT \
  --version $VERSION \
  --ann-file $ANN_VAL \
  --label-mapping tools/generate_occupancy_nuscenes/nuscenes.yaml \
  --output-root data/GT_occupancy_v1_0_mini/stage2_raycast_occ3d_dense_point_origin_skip_free_z1_4_swapxy_flipy_val \
  --ray-source dense-candidate \
  --dense-candidate-dir data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/dense_voxels_with_semantic \
  --dense-ray-source-points-dir data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/ray_source_points_with_origin \
  --dense-coordinate-transform swapxy_flipy \
  --ray-traversal occ3d-point-to-origin \
  --skip-free-z-min 1 \
  --skip-free-z-max 4
```

**4. Stage3 Train：融合语义并生成 Occ3D/FlashOcc labels**

```bash
$PY tools/occ3d_stage3/fuse_stage3_semantics.py \
  --stage2-root data/GT_occupancy_v1_0_mini/stage2_raycast_occ3d_dense_point_origin_skip_free_z1_4_swapxy_flipy_train \
  --stage2-ann data/GT_occupancy_v1_0_mini/stage2_raycast_occ3d_dense_point_origin_skip_free_z1_4_swapxy_flipy_train/bevdetv2-nuscenes_infos_stage2_train.pkl \
  --sparse-dir data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/dense_voxels_with_semantic \
  --candidate-grid occ3d \
  --dense-coordinate-transform swapxy_flipy \
  --output-root data/GT_occupancy_v1_0_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy_train
```

**5. Stage3 Val**

```bash
$PY tools/occ3d_stage3/fuse_stage3_semantics.py \
  --stage2-root data/GT_occupancy_v1_0_mini/stage2_raycast_occ3d_dense_point_origin_skip_free_z1_4_swapxy_flipy_val \
  --stage2-ann data/GT_occupancy_v1_0_mini/stage2_raycast_occ3d_dense_point_origin_skip_free_z1_4_swapxy_flipy_val/bevdetv2-nuscenes_infos_stage2_train.pkl \
  --sparse-dir data/GT_occupancy_v1_0_mini_occ3d_grid_parallel/dense_voxels_with_semantic \
  --candidate-grid occ3d \
  --dense-coordinate-transform swapxy_flipy \
  --output-root data/GT_occupancy_v1_0_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy_val
```

**6. Stage4 Train：生成真实 camera ray mask**

```bash
$PY tools/occ3d_stage4/build_camera_mask.py \
  --input-ann data/GT_occupancy_v1_0_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy_train/bevdetv2-nuscenes_infos_stage3_train.pkl \
  --output-root data/GT_occupancy_v1_0_mini/stage4_camera_raymask_z1_4_train \
  --depth-max 45.0 \
  --image-size 1600 900 \
  --pixel-stride 1
```

**7. Stage4 Val**

```bash
$PY tools/occ3d_stage4/build_camera_mask.py \
  --input-ann data/GT_occupancy_v1_0_mini/stage3_fused_occ3d_point_origin_skip_free_z1_4_swapxy_flipy_val/bevdetv2-nuscenes_infos_stage3_train.pkl \
  --output-root data/GT_occupancy_v1_0_mini/stage4_camera_raymask_z1_4_val \
  --depth-max 45.0 \
  --image-size 1600 900 \
  --pixel-stride 1
```

**8. 格式检查**

```bash
$PY tools/occ3d_stage1/check_occ3d_npz.py \
  --ann-file data/GT_occupancy_v1_0_mini/stage4_camera_raymask_z1_4_train/bevdetv2-nuscenes_infos_stage4_train.pkl

$PY tools/occ3d_stage1/check_occ3d_npz.py \
  --ann-file data/GT_occupancy_v1_0_mini/stage4_camera_raymask_z1_4_val/bevdetv2-nuscenes_infos_stage4_train.pkl
```

**9. 接 FlashOcc 训练**

你现在已有可用 config：

```bash
cd /home/fjm/shousuyao/FlashOCC

PYTHONPATH=/home/fjm/shousuyao/FlashOCC \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_flashocc \
MPLCONFIGDIR=/tmp/matplotlib_flashocc \
TORCH_CUDA_ARCH_LIST='8.6' \
/home/fjm/miniconda3/envs/flashocc/bin/python tools/train.py \
  /home/fjm/shousuyao/SurroundOcc_my/data/GT_occupancy_v1_0_mini/flashocc_train_stage4_camera_z1_4_mini10/flashocc-r50-mini10-stage4-24ep.py \
  --work-dir /home/fjm/shousuyao/SurroundOcc_my/data/GT_occupancy_v1_0_mini/flashocc_train_stage4_camera_z1_4_mini10_24ep
```

核心产物就是：

```bash
data/GT_occupancy_v1_0_mini/stage4_camera_raymask_z1_4_train/
data/GT_occupancy_v1_0_mini/stage4_camera_raymask_z1_4_val/
```

这两个目录里的 `gts/<scene>/<token>/labels.npz` 和 `bevdetv2-nuscenes_infos_stage4_train.pkl` 就是后续 FlashOcc 训练/验证要接的 GT。

## 可视化命令

```bash
ssh -Y fjm@10.10.10.12

conda activate flashocc

cd /home/fjm/shousuyao/SurroundOcc_my
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('data/GT_occupancy_mini/compare_scene0061_ca9a282c9e77460f8360f564131a8af5/compare_shifted_corrected_ours_left_flashocc_right.ply'); o3d.visualization.draw_geometries([p])"
```

**生成的** **GT** **和 FlashOcc GT 的空间范围不同**

- ours 范围大约是 `x/y: [-49.75, 49.75]`，`z: [-4.25, 2.75]`
- FlashOcc GT 范围大约是 `x/y: [-40, 40]`，`z: [-0.8, 5.2]`

**实际surrondocc的配置：**

```
'voxel_size': 0.5
'pc_range':  [-50, -50, -5, 50, 50, 3]
'occ_size':  [200, 200, 16]
```

含义是：

```
x: -50m ~ 50m, 200 格，每格 0.5m
y: -50m ~ 50m, 200 格，每格 0.5m
z: -5m ~ 3m,    16 格，每格 0.5m
```

它覆盖的是一个 `100m x 100m x 8m` 的空间。

**flashocc-occ3D的配置：**

```
'voxel_size': 0.5
'pc_range':  [-50, -50, -5, 50, 50, 3]
'occ_size':  [200, 200, 16]
```

含义是：

```
x: -50m ~ 50m, 200 格，每格 0.5m
y: -50m ~ 50m, 200 格，每格 0.5m
z: -5m ~ 3m,    16 格，每格 0.5m
```

它覆盖的是一个 `100m x 100m x 8m` 的空间。


