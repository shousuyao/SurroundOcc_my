import os
import sys
import pdb
import time
import json
import yaml
import torch
import chamfer
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from mmdet3d.core.bbox import box_np_ops
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation

import open3d
import open3d as o3d
from copy import deepcopy

from surface_completion import (
    filter_mesh_components,
    fill_flat_height_holes,
    mesh_topology_stats,
    orient_normals_toward_origins,
    surface_voxels_from_mesh,
)
from semantic_reconstruction import reconstruct_semantic_groups


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
    origins=None,
    normal_orientation='camera',
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        if normal_orientation == 'camera':
            cloud.orient_normals_towards_camera_location()
            normal_flips = 0
        elif normal_orientation == 'point-origin':
            oriented, normal_flips = orient_normals_toward_origins(
                np.asarray(cloud.points), np.asarray(cloud.normals), origins)
            cloud.normals = o3d.utility.Vector3dVector(oriented)
        else:
            raise ValueError('Unsupported normal orientation: {}'.format(normal_orientation))
    else:
        normal_flips = 0

    return cloud, normal_flips


def preprocess(pcd, config, origins=None, normal_orientation='camera'):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True,
        origins=origins,
        normal_orientation=normal_orientation,
    )

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances




def mask_points_in_range(points, pc_range):
    return (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )


def points_to_voxel_indices(points, pc_range, voxel_size, occ_size):
    voxel_indices = points.copy()
    voxel_indices[:, 0] = (voxel_indices[:, 0] - pc_range[0]) / voxel_size
    voxel_indices[:, 1] = (voxel_indices[:, 1] - pc_range[1]) / voxel_size
    voxel_indices[:, 2] = (voxel_indices[:, 2] - pc_range[2]) / voxel_size
    voxel_indices = np.floor(voxel_indices).astype(np.int64)
    valid = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < occ_size[0]) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < occ_size[1]) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < occ_size[2])
    )
    return voxel_indices[valid], valid


def points_to_semantic_voxels(points_with_semantic, pc_range, voxel_size, occ_size):
    voxel_indices, valid = points_to_voxel_indices(points_with_semantic[:, :3], pc_range, voxel_size, occ_size)
    labels = points_with_semantic[valid, 3].astype(np.int64)
    if voxel_indices.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.int64)

    keys, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    semantic = np.zeros(keys.shape[0], dtype=np.int64)
    for idx in range(keys.shape[0]):
        idx_labels = labels[inverse == idx]
        valid_labels = idx_labels[(idx_labels >= 0) & (idx_labels <= 16)]
        if valid_labels.shape[0] == 0:
            semantic[idx] = 0
        else:
            semantic[idx] = np.bincount(valid_labels, minlength=17).argmax()
    return np.concatenate([keys.astype(np.int64), semantic[:, np.newaxis]], axis=1)


def transform_xyz_lidar_to_lidar(points, lidar_calibrated_sensor, lidar_ego_pose,
                                 target_calibrated_sensor, target_ego_pose):
    points = points.copy()
    points = points @ Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix.T
    points = points + np.array(lidar_calibrated_sensor['translation'])

    points = points @ Quaternion(lidar_ego_pose['rotation']).rotation_matrix.T
    points = points + np.array(lidar_ego_pose['translation'])

    points = points - np.array(target_ego_pose['translation'])
    points = points @ Quaternion(target_ego_pose['rotation']).rotation_matrix

    points = points - np.array(target_calibrated_sensor['translation'])
    points = points @ Quaternion(target_calibrated_sensor['rotation']).rotation_matrix
    return points


def lidar_to_world_to_lidar(pc,lidar_calibrated_sensor,lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose):

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor['translation']))

    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    return pc


def main(nusc, val_list, indice, nuscenesyaml, args, config):

    save_path = args.save_path
    data_root = args.dataroot
    learning_map = nuscenesyaml['learning_map']
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']
    surface_mode = args.surface_mode or config.get('surface_mode', 'vertices')
    surface_sample_spacing = (
        args.surface_sample_spacing
        if args.surface_sample_spacing is not None
        else float(config.get('surface_sample_spacing', 0.2))
    )
    flat_height_fill = bool(args.flat_height_fill or config.get('flat_height_fill', False))
    flat_fill_radius = int(config.get('flat_fill_radius', 1))
    flat_fill_min_neighbors = int(config.get('flat_fill_min_neighbors', 5))
    flat_fill_max_z_spread = int(config.get('flat_fill_max_z_spread', 1))
    normal_orientation = args.normal_orientation or config.get('normal_orientation', 'camera')
    min_component_triangles = (
        args.min_component_triangles
        if args.min_component_triangles is not None
        else int(config.get('min_component_triangles', 0))
    )
    reconstruction_mode = args.reconstruction_mode or config.get('reconstruction_mode', 'global')

    my_scene = nusc.scene[indice]
    sensor = 'LIDAR_TOP'

    if args.split == 'train':
        if my_scene['token'] in val_list:
            return
    elif args.split == 'val':
        if my_scene['token'] not in val_list:
            return
    elif args.split == 'all':
        pass
    else:
        raise NotImplementedError


    # load the first sample to start
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
    lidar_ego_pose0 = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_calibrated_sensor0 = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    # collect LiDAR sequence
    dict_list = []

    while True:
        ############################# get boxes ##########################
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_data['token'])
        boxes_token = [box.token for box in boxes]
        object_tokens = [nusc.get('sample_annotation', box_token)['instance_token'] for box_token in boxes_token]
        object_category = [nusc.get('sample_annotation', box_token)['category_name'] for box_token in boxes_token]

        ############################# get object categories ##########################
        converted_object_category = []
        for category in object_category:
            for (j, label) in enumerate(nuscenesyaml['labels']):
                if category == nuscenesyaml['labels'][label]:
                    converted_object_category.append(np.vectorize(learning_map.__getitem__)(label).item())

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1  # Move the bbox slightly down in the z direction
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1 # Slightly expand the bbox to wrap all object points
        ############################# get LiDAR points with semantics ##########################
        pc_file_name = lidar_data['filename'] # load LiDAR names
        pc0 = np.fromfile(os.path.join(data_root, pc_file_name),
                          dtype=np.float32,
                          count=-1).reshape(-1, 5)[..., :4]
        if lidar_data['is_key_frame']: # only key frame has semantic annotations
            lidar_sd_token = lidar_data['token']
            lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                    nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc_with_semantic = np.concatenate([pc0[:, :3], points_label], axis=1)

        ############################# cut out movable object points and masks ##########################
        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(gt_bbox_3d[np.newaxis, :]))
        object_points_list = []
        object_origins_list = []
        j = 0
        while j < points_in_boxes.shape[-1]:
            object_points_mask = points_in_boxes[0][:,j].bool()
            object_points = pc0[object_points_mask]
            object_points_list.append(object_points)
            object_origins_list.append(np.zeros((object_points.shape[0], 3), dtype=np.float32))
            j = j + 1

        moving_mask = torch.ones_like(points_in_boxes)
        points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
        points_mask = ~(points_in_boxes[0])

        ############################# get point mask of the vehicle itself ##########################
        range = config['self_range']
        oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > range[0]) |
                                        (np.abs(pc0[:, 1]) > range[1]) |
                                        (np.abs(pc0[:, 2]) > range[2]))

        ############################# get static scene segment ##########################
        points_mask = points_mask & oneself_mask
        pc = pc0[points_mask]

        ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
        lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_pc = lidar_to_world_to_lidar(pc.copy(), lidar_calibrated_sensor.copy(), lidar_ego_pose.copy(),
                                           lidar_calibrated_sensor0,
                                           lidar_ego_pose0)
        origin_in_first_lidar = transform_xyz_lidar_to_lidar(
            np.zeros((1, 3), dtype=np.float32),
            lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(),
            lidar_calibrated_sensor0,
            lidar_ego_pose0,
        )[0]
        lidar_pc_origins = np.repeat(origin_in_first_lidar[np.newaxis, :], pc.shape[0], axis=0)
        ################## record Non-key frame information into a dict  ########################
        dict = {"object_tokens": object_tokens,
                "object_points_list": object_points_list,
                "object_origins_list": object_origins_list,
                "lidar_pc": lidar_pc.points,
                "lidar_pc_origins": lidar_pc_origins,
                "lidar_ego_pose": lidar_ego_pose,
                "lidar_calibrated_sensor": lidar_calibrated_sensor,
                "lidar_token": lidar_data['token'],
                "is_key_frame": lidar_data['is_key_frame'],
                "gt_bbox_3d": gt_bbox_3d,
                "converted_object_category": converted_object_category,
                "pc_file_name": pc_file_name.split('/')[-1]}
        ################## record semantic information into the dict if it's a key frame  ########################
        if lidar_data['is_key_frame']:
            pc_with_semantic = pc_with_semantic[points_mask]
            lidar_pc_with_semantic = lidar_to_world_to_lidar(pc_with_semantic.copy(),
                                                             lidar_calibrated_sensor.copy(),
                                                             lidar_ego_pose.copy(),
                                                             lidar_calibrated_sensor0,
                                                             lidar_ego_pose0)
            dict["lidar_pc_with_semantic"] = lidar_pc_with_semantic.points
            dict["lidar_pc_with_semantic_origins"] = np.repeat(
                origin_in_first_lidar[np.newaxis, :], pc_with_semantic.shape[0], axis=0)

        dict_list.append(dict)
        ################## go to next frame of the sequence  ########################
        next_token = lidar_data['next']
        if next_token != '':
            lidar_data = nusc.get('sample_data', next_token)
        else:
            break

    ################## concatenate all static scene segments (including non-key frames)  ########################
    lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
    lidar_pc = np.concatenate(lidar_pc_list, axis=1).T
    lidar_pc_origins = np.concatenate([dict['lidar_pc_origins'] for dict in dict_list], axis=0)

    ################## concatenate all semantic scene segments (only key frames)  ########################
    lidar_pc_with_semantic_list = []
    lidar_pc_with_semantic_origins_list = []
    for dict in dict_list:
        if dict['is_key_frame']:
            lidar_pc_with_semantic_list.append(dict['lidar_pc_with_semantic'])
            lidar_pc_with_semantic_origins_list.append(dict['lidar_pc_with_semantic_origins'])
    lidar_pc_with_semantic = np.concatenate(lidar_pc_with_semantic_list, axis=1).T
    lidar_pc_with_semantic_origins = np.concatenate(lidar_pc_with_semantic_origins_list, axis=0)

    ################## concatenate all object segments (including non-key frames)  ########################
    object_token_zoo = []
    object_semantic = []
    for dict in dict_list:
        for i,object_token in enumerate(dict['object_tokens']):
            if object_token not in object_token_zoo:
                if (dict['object_points_list'][i].shape[0] > 0):
                    object_token_zoo.append(object_token)
                    object_semantic.append(dict['converted_object_category'][i])
                else:
                    continue

    object_points_dict = {}
    object_origins_dict = {}

    for query_object_token in object_token_zoo:
        object_points_dict[query_object_token] = []
        object_origins_dict[query_object_token] = []
        for dict in dict_list:
            for i, object_token in enumerate(dict['object_tokens']):
                if query_object_token == object_token:
                    object_points = dict['object_points_list'][i]
                    if object_points.shape[0] > 0:
                        object_origins = dict['object_origins_list'][i]
                        object_points = object_points[:,:3] - dict['gt_bbox_3d'][i][:3]
                        object_origins = object_origins - dict['gt_bbox_3d'][i][:3]
                        rots = dict['gt_bbox_3d'][i][6]
                        Rot = Rotation.from_euler('z', -rots, degrees=False)
                        rotated_object_points = Rot.apply(object_points)
                        rotated_object_origins = Rot.apply(object_origins)
                        object_points_dict[query_object_token].append(rotated_object_points)
                        object_origins_dict[query_object_token].append(rotated_object_origins)
                else:
                    continue
        object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token],
                                                                axis=0)
        object_origins_dict[query_object_token] = np.concatenate(object_origins_dict[query_object_token],
                                                                 axis=0)


    object_points_vertice = []
    object_origins_vertice = []
    for key in object_points_dict.keys():
        point_cloud = object_points_dict[key]
        object_points_vertice.append(point_cloud[:,:3])
        object_origins_vertice.append(object_origins_dict[key][:,:3])
    # print('object finish')


    i = 0
    processed_keyframes = 0
    while int(i) < 10000:  # Assuming the sequence does not have more than 10000 frames
        if i >= len(dict_list):
            print('finish scene!')
            return
        dict = dict_list[i]
        is_key_frame = dict['is_key_frame']
        if not is_key_frame: # only use key frame as GT
            i = i + 1
            continue

        ################## convert the static scene to the target coordinate system ##############
        lidar_calibrated_sensor = dict['lidar_calibrated_sensor']
        lidar_ego_pose = dict['lidar_ego_pose']
        lidar_pc_i = lidar_to_world_to_lidar(lidar_pc.copy(),
                                             lidar_calibrated_sensor0.copy(),
                                             lidar_ego_pose0.copy(),
                                             lidar_calibrated_sensor,
                                             lidar_ego_pose)
        lidar_pc_i_origins = transform_xyz_lidar_to_lidar(
            lidar_pc_origins.copy(),
            lidar_calibrated_sensor0.copy(),
            lidar_ego_pose0.copy(),
            lidar_calibrated_sensor,
            lidar_ego_pose,
        )
        lidar_pc_i_semantic = lidar_to_world_to_lidar(lidar_pc_with_semantic.copy(),
                                                      lidar_calibrated_sensor0.copy(),
                                                      lidar_ego_pose0.copy(),
                                                      lidar_calibrated_sensor,
                                                      lidar_ego_pose)
        lidar_pc_i_semantic_origins = transform_xyz_lidar_to_lidar(
            lidar_pc_with_semantic_origins.copy(),
            lidar_calibrated_sensor0.copy(),
            lidar_ego_pose0.copy(),
            lidar_calibrated_sensor,
            lidar_ego_pose,
        )
        point_cloud = lidar_pc_i.points.T[:,:3]
        point_cloud_with_semantic = lidar_pc_i_semantic.points.T

        ################## load bbox of target frame ##############
        lidar_path, boxes, _ = nusc.get_sample_data(dict['lidar_token'])
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        gt_bbox_3d = np.concatenate([locs, dims, rots], axis=1).astype(np.float32)
        gt_bbox_3d[:, 6] += np.pi / 2.
        gt_bbox_3d[:, 2] -= dims[:, 2] / 2.
        gt_bbox_3d[:, 2] = gt_bbox_3d[:, 2] - 0.1
        gt_bbox_3d[:, 3:6] = gt_bbox_3d[:, 3:6] * 1.1
        rots = gt_bbox_3d[:,6:7]
        locs = gt_bbox_3d[:,0:3]

        ################## bbox placement ##############
        object_points_list = []
        object_semantic_list = []
        object_origin_list = []
        for j, object_token in enumerate(dict['object_tokens']):
            for k, object_token_in_zoo in enumerate(object_token_zoo):
                if object_token==object_token_in_zoo:
                    points = object_points_vertice[k]
                    origins = object_origins_vertice[k]
                    Rot = Rotation.from_euler('z', rots[j], degrees=False)
                    rotated_object_points = Rot.apply(points)
                    rotated_object_origins = Rot.apply(origins)
                    points = rotated_object_points + locs[j]
                    origins = rotated_object_origins + locs[j]
                    if points.shape[0] >= 5:
                        points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                              torch.from_numpy(gt_bbox_3d[j:j+1][np.newaxis, :]))
                        object_valid = points_in_boxes[0,:,0].bool().numpy()
                        points = points[object_valid]
                        origins = origins[object_valid]

                    object_points_list.append(points)
                    object_origin_list.append(origins)
                    semantics = np.ones_like(points[:,0:1]) * object_semantic[k]
                    object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

        try: # avoid concatenate an empty array
            temp = np.concatenate(object_points_list)
            temp_origins_geometry = np.concatenate(object_origin_list)
            scene_points = np.concatenate([point_cloud, temp])
            scene_origins = np.concatenate([lidar_pc_i_origins, temp_origins_geometry])
        except:
            scene_points = point_cloud
            scene_origins = lidar_pc_i_origins
        try:
            temp = np.concatenate(object_semantic_list)
            temp_origins = np.concatenate(object_origin_list)
            scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
            scene_semantic_origins = np.concatenate([lidar_pc_i_semantic_origins, temp_origins])
        except:
            scene_semantic_points = point_cloud_with_semantic
            scene_semantic_origins = lidar_pc_i_semantic_origins

        ################## remain points with a spatial range ##############
        mask = mask_points_in_range(scene_points, pc_range)
        scene_points = scene_points[mask]
        scene_origins = scene_origins[mask]

        ################## save LiDAR-supported ray source voxels before mesh completion ##############
        ray_source_mask = mask_points_in_range(scene_semantic_points, pc_range)
        ray_source_points = scene_semantic_points[ray_source_mask]
        ray_source_origins = scene_semantic_origins[ray_source_mask]
        ray_source_voxels_with_semantic = points_to_semantic_voxels(
            ray_source_points, pc_range, voxel_size, occ_size)
        ray_source_dirs = os.path.join(save_path, 'ray_source_voxels_with_semantic/')
        if not os.path.exists(ray_source_dirs):
            os.makedirs(ray_source_dirs)
        np.save(os.path.join(ray_source_dirs, dict['pc_file_name'] + '.npy'), ray_source_voxels_with_semantic)

        ray_source_points_with_origin = np.concatenate(
            [
                ray_source_points[:, :3],
                ray_source_origins[:, :3],
                ray_source_points[:, 3:4].astype(np.int64),
            ],
            axis=1,
        )
        ray_source_point_dirs = os.path.join(save_path, 'ray_source_points_with_origin/')
        if not os.path.exists(ray_source_point_dirs):
            os.makedirs(ray_source_point_dirs)
        np.save(os.path.join(ray_source_point_dirs, dict['pc_file_name'] + '.npy'), ray_source_points_with_origin)

        if getattr(args, 'ray_source_only', False):
            i = i + 1
            continue

        if reconstruction_mode == 'semantic-groups':
            dense_voxels_with_semantic, reconstruction_stats = reconstruct_semantic_groups(
                ray_source_points,
                ray_source_origins,
                pc_range=np.asarray(pc_range, dtype=np.float64),
                voxel_size=np.full(3, float(voxel_size), dtype=np.float64),
                occ_size=np.asarray(occ_size, dtype=np.int64),
                poisson_depth=int(config['depth']),
                min_density=float(config['min_density']),
                max_nn=int(config['max_nn']),
                surface_mode=surface_mode,
                sample_spacing=surface_sample_spacing,
                min_component_triangles=min_component_triangles,
            )
            dirs = os.path.join(save_path, 'dense_voxels_with_semantic/')
            os.makedirs(dirs, exist_ok=True)
            np.save(os.path.join(dirs, dict['pc_file_name'] + '.npy'), dense_voxels_with_semantic)
            stats_dirs = os.path.join(save_path, 'surface_stats/')
            os.makedirs(stats_dirs, exist_ok=True)
            reconstruction_stats.update({
                'scene_name': my_scene['name'],
                'lidar_filename': dict['pc_file_name'],
                'reconstruction_mode': reconstruction_mode,
                'surface_mode': surface_mode,
            })
            with open(os.path.join(stats_dirs, dict['pc_file_name'] + '.json'), 'w') as f:
                json.dump(reconstruction_stats, f, indent=2)
            i += 1
            processed_keyframes += 1
            if args.max_keyframes is not None and processed_keyframes >= args.max_keyframes:
                print('finish requested keyframes!')
                return
            continue

        ################## get mesh via Possion Surface Reconstruction ##############
        point_cloud_original = o3d.geometry.PointCloud()
        with_normal2 = o3d.geometry.PointCloud()
        point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
        with_normal, normal_flips = preprocess(
            point_cloud_original,
            config,
            origins=scene_origins,
            normal_orientation=normal_orientation,
        )
        with_normal2.points = with_normal.points
        with_normal2.normals = with_normal.normals
        mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                       config['min_density'], with_normal2)
        if args.full_topology_stats:
            topology_before = mesh_topology_stats(mesh)
        else:
            topology_before = None
        mesh, component_filter_stats = filter_mesh_components(mesh, min_component_triangles)
        if args.full_topology_stats:
            topology_after = mesh_topology_stats(mesh)
        else:
            topology_after = None
        ################## voxelize the complete triangle surface ##############
        surface_coords, surface_stats = surface_voxels_from_mesh(
            mesh,
            mode=surface_mode,
            pc_range=np.asarray(pc_range, dtype=np.float64),
            voxel_size=np.full(3, float(voxel_size), dtype=np.float64),
            occ_size=np.asarray(occ_size, dtype=np.int64),
            sample_spacing=surface_sample_spacing,
        )
        fov_voxels = (
            np.asarray(pc_range[:3], dtype=np.float64)[None, :]
            + (surface_coords.astype(np.float64) + 0.5) * float(voxel_size)
        )

        ################## get semantics of sparse points  ##############
        mask = mask_points_in_range(scene_semantic_points, pc_range)
        scene_semantic_points = scene_semantic_points[mask]

        ################## Nearest Neighbor to assign semantics ##############
        dense_voxels = fov_voxels
        sparse_voxels_semantic = scene_semantic_points

        if dense_voxels.shape[0] == 0:
            raise RuntimeError('Poisson surface produced no voxels inside pc_range')
        if sparse_voxels_semantic.shape[0] == 0:
            raise RuntimeError('No semantic points remain inside pc_range')

        x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
        y = torch.from_numpy(sparse_voxels_semantic[:,:3]).cuda().unsqueeze(0).float()
        d1, d2, idx1, idx2 = chamfer.forward(x,y)
        indices = idx1[0].cpu().numpy()


        dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
        dense_voxels_with_semantic = np.concatenate(
            [surface_coords.astype(np.int64), dense_semantic[:, np.newaxis].astype(np.int64)],
            axis=1,
        )
        flat_fill_stats = {'added_total': 0, 'added_by_class': {str(i): 0 for i in (11, 12, 13, 14)}}
        if flat_height_fill:
            dense_voxels_with_semantic, flat_fill_stats = fill_flat_height_holes(
                dense_voxels_with_semantic,
                occ_size=np.asarray(occ_size, dtype=np.int64),
                radius=flat_fill_radius,
                min_neighbors=flat_fill_min_neighbors,
                max_z_spread=flat_fill_max_z_spread,
                flat_labels=(11, 12, 13, 14),
            )

        dirs = os.path.join(save_path, 'dense_voxels_with_semantic/')
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        np.save(os.path.join(dirs, dict['pc_file_name'] + '.npy'), dense_voxels_with_semantic)

        class_counts = np.bincount(dense_voxels_with_semantic[:, 3], minlength=17)
        stats_dirs = os.path.join(save_path, 'surface_stats/')
        if not os.path.exists(stats_dirs):
            os.makedirs(stats_dirs)
        surface_stats.update({
            'scene_name': my_scene['name'],
            'lidar_filename': dict['pc_file_name'],
            'surface_sample_spacing': float(surface_sample_spacing),
            'flat_height_fill': bool(flat_height_fill),
            'flat_fill': flat_fill_stats,
            'normal_orientation': normal_orientation,
            'normal_flips': int(normal_flips),
            'min_component_triangles': int(min_component_triangles),
            'component_filter': component_filter_stats,
            'topology_before': topology_before,
            'topology_after': topology_after,
            'final_voxel_count': int(dense_voxels_with_semantic.shape[0]),
            'class_counts': [int(value) for value in class_counts],
        })
        with open(os.path.join(stats_dirs, dict['pc_file_name'] + '.json'), 'w') as f:
            json.dump(surface_stats, f, indent=2)

        i = i + 1
        processed_keyframes += 1
        if args.max_keyframes is not None and processed_keyframes >= args.max_keyframes:
            print('finish requested keyframes!')
            return
        continue


def save_ply(points, name):
    point_cloud_original = o3d.geometry.PointCloud()
    point_cloud_original.points = o3d.utility.Vector3dVector(points[:,:3])
    o3d.io.write_point_cloud("{}.ply".format(name), point_cloud_original)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_path(path):
        if os.path.isabs(path) or os.path.exists(path):
            return path
        return os.path.join(script_dir, path)

    parse.add_argument('--dataset', type=str, default='nuscenes')
    parse.add_argument('--config_path', type=str, default='config.yaml')
    parse.add_argument('--split', type=str, default='train')
    parse.add_argument('--save_path', type=str, default='./data/GT_occupancy/')
    parse.add_argument('--start', type=int, default=0)
    parse.add_argument('--end', type=int, default=850)
    parse.add_argument('--dataroot', type=str, default='./data/nuScenes/')
    parse.add_argument('--version', type=str, default='v1.0-trainval')
    parse.add_argument('--nusc_val_list', type=str, default='./nuscenes_val_list.txt')
    parse.add_argument('--label_mapping', type=str, default='nuscenes.yaml')
    parse.add_argument('--ray_source_only', action='store_true',
                       help='Only save LiDAR-supported ray_source_voxels_with_semantic and skip Poisson/chamfer dense GT.')
    parse.add_argument('--surface_mode', choices=('vertices', 'uniform', 'triangle'), default=None,
                       help='Poisson mesh surface voxelization mode. Defaults to config surface_mode.')
    parse.add_argument('--surface_sample_spacing', type=float, default=None,
                       help='Target surface point spacing in meters for --surface_mode uniform.')
    parse.add_argument('--flat_height_fill', action='store_true',
                       help='Conservatively close supported XY holes for flat semantic classes.')
    parse.add_argument('--normal_orientation', choices=('camera', 'point-origin'), default=None,
                       help='Orient Poisson input normals to one camera origin or each point origin.')
    parse.add_argument('--min_component_triangles', type=int, default=None,
                       help='Remove mesh connected components smaller than this triangle count.')
    parse.add_argument('--max_keyframes', type=int, default=None,
                       help='Stop after generating this many keyframes from each selected scene.')
    parse.add_argument('--reconstruction_mode', choices=('global', 'semantic-groups'), default=None,
                       help='Use one global Poisson mesh or class-aware reconstruction branches.')
    parse.add_argument('--full_topology_stats', action='store_true',
                       help='Compute expensive watertight/manifold topology fields for diagnostic runs.')
    args=parse.parse_args()
    args.config_path = resolve_path(args.config_path)
    args.nusc_val_list = resolve_path(args.nusc_val_list)
    args.label_mapping = resolve_path(args.label_mapping)


    if args.dataset=='nuscenes':
        val_list = []
        with open(args.nusc_val_list, 'r') as file:
            for item in file:
                val_list.append(item[:-1])
        file.close()

        nusc = NuScenes(version=args.version,
                        dataroot=args.dataroot,
                        verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
    else:
        raise NotImplementedError

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        nuscenesyaml = yaml.safe_load(stream)


    for i in range(args.start,args.end):
        print('processing sequecne:', i)
        main(nusc, val_list, indice=i,
             nuscenesyaml=nuscenesyaml, args=args, config=config)
