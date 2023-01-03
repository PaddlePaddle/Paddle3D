# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------
# Modify from https://github.com/fundamentalvision/BEVFormer/blob/master/tools/create_data.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import tqdm
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits as nuscenes_split
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from paddle3d.datasets.nuscenes import NuscenesMVDataset
from paddle3d.datasets.nuscenes.nuscenes_det import NuscenesDetDataset
from paddle3d.utils.logger import logger

SENSORS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
    'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create infos for kitti dataset.')
    parser.add_argument(
        '--dataset_root',
        default='data/nuscenes',
        help='Path of the dataset.',
        type=str)
    parser.add_argument(
        '--can_bus_root',
        type=str,
        default='data/nuscenes',
        help='specify the root path of nuScenes canbus')
    parser.add_argument(
        '--save_dir',
        default='data/nuscenes',
        help='Path to save the generated database.',
        type=str)
    parser.add_argument(
        '--mode', default='train', help='mode to generate dataset.', type=str)
    parser.add_argument(
        '--num_sweep',
        default=10,
        help='nummber of sweep frames between two key frame.',
        type=int)
    return parser.parse_args()


def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.
    """
    available_scenes = []
    logger.info('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    logger.info('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))  # absolute path
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': nusc.get('sample_data',
                              sd_rec['token'])['filename'],  # relative path
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def fill_trainval_infos(nusc,
                        nusc_can_bus,
                        train_scenes,
                        val_scenes,
                        test=False,
                        max_sweeps=10):
    """Generate the train/val infos from the raw data.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0

    msg = "Begin to generate a info of nuScenes dataset."

    for sample_idx in logger.range(len(nusc.sample), msg=msg):
        sample = nusc.sample[sample_idx]
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        assert os.path.exists(lidar_path)  # absolute path
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

        info = {
            'lidar_token': lidar_token,
            'lidar_path': nusc.get('sample_data',
                                   lidar_token)['filename'],  # relative path
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token) for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                # NuscenesDetDataset.LABEL_MAP
                if names[i] in NuscenesDetDataset.LABEL_MAP:
                    names[i] = NuscenesDetDataset.LABEL_MAP[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def build_bevformer_nuscenes_data(dataset_root,
                                  is_test,
                                  nusc,
                                  nusc_can_bus,
                                  version,
                                  max_sweeps=10):

    if version == 'v1.0-trainval':
        train_scenes = nuscenes_split.train
        val_scenes = nuscenes_split.val
    elif version == 'v1.0-test':
        train_scenes = nuscenes_split.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = nuscenes_split.mini_train
        val_scenes = nuscenes_split.mini_val
    else:
        raise ValueError('unknown nuscenes dataset version')

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]

    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    if is_test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        nusc,
        nusc_can_bus,
        train_scenes,
        val_scenes,
        is_test,
        max_sweeps=max_sweeps)

    metadata = dict(version=version)

    if is_test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        return [data]
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        train_data = dict(infos=train_nusc_infos, metadata=metadata)
        val_data = dict(infos=val_nusc_infos, metadata=metadata)

        return train_data, val_data


def main(args):
    dataset_root = args.dataset_root
    can_bus_root = args.can_bus_root
    save_dir = args.save_dir
    num_sweep = args.num_sweep

    version = NuscenesDetDataset.VERSION_MAP[args.mode]

    nuscenes = NuScenes(version=version, dataroot=dataset_root, verbose=False)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root)

    is_test = 'test' in args.mode

    if is_test:
        test_ann_cache_file = os.path.join(
            save_dir, 'bevformer_nuscenes_annotation_test.pkl')
        if os.path.exists(test_ann_cache_file):
            raise OSError(
                "{} annotation file is exist!".format(test_ann_cache_file))
    else:
        train_ann_cache_file = os.path.join(
            save_dir, 'bevformer_nuscenes_annotation_train.pkl')
        val_ann_cache_file = os.path.join(
            save_dir, 'bevformer_nuscenes_annotation_val.pkl')
        if os.path.exists(train_ann_cache_file):
            raise OSError(
                "{} annotation file is exist!".format(train_ann_cache_file))
        if os.path.exists(val_ann_cache_file):
            raise OSError(
                "{} annotation file is exist!".format(val_ann_cache_file))

    infos = build_bevformer_nuscenes_data(dataset_root, is_test, nuscenes,
                                          nusc_can_bus, version, num_sweep)

    if is_test:
        infos_dict = {test_ann_cache_file: infos[0]}
    else:
        infos_dict = {
            train_ann_cache_file: infos[0],
            val_ann_cache_file: infos[1]
        }

    msg = "Adding sweep frame annotations"
    for ann_cache_file, key_infos in infos_dict.items():
        pickle.dump(key_infos, open(ann_cache_file, 'wb'))

    logger.info("---------------Data preparation Done---------------")


if __name__ == '__main__':
    args = parse_args()
    main(args)
