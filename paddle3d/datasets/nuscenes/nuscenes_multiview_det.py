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

import copy
import numbers
import os
import os.path as osp
import pickle
import random
from collections.abc import Mapping, Sequence
from functools import reduce
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

import paddle
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.utils import splits as nuscenes_split
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm

import paddle3d.transforms as T
from paddle3d.apis import manager
from paddle3d.datasets.nuscenes.nuscenes_det import NuscenesDetDataset
from paddle3d.datasets.nuscenes.nuscenes_manager import NuScenesManager
from paddle3d.geometries import BBoxes3D, CoordMode
from paddle3d.sample import Sample, SampleMeta
from paddle3d.transforms import TransformABC
from paddle3d.utils.logger import logger
from paddle3d.datasets.nuscenes.nuscenes_metric import NuScenesSegMetric


def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)


@manager.DATASETS.add_component
class NuscenesMVDataset(NuscenesDetDataset):
    """
    Nuscecens dataset for multi-view camera detection task.
    """
    DATASET_NAME = "Nuscenes"

    def __init__(self,
                 dataset_root: str,
                 ann_file: str = None,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 max_sweeps: int = 10,
                 class_balanced_sampling: bool = False,
                 class_names: Union[list, tuple] = None,
                 queue_length=None,
                 use_valid_flag=False,
                 with_velocity=True):

        self.mode = mode
        self.dataset_root = dataset_root
        self.filter_empty_gt = True
        self.box_type_3d = 'LiDAR'
        self.box_mode_3d = None
        self.ann_file = ann_file
        self.version = self.VERSION_MAP[self.mode]

        self.max_sweeps = max_sweeps
        self._build_data()
        self.metadata = self.data_infos['metadata']

        self.data_infos = list(
            sorted(self.data_infos['infos'], key=lambda e: e['timestamp']))

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms

        if 'train' in self.mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)

        self.modality = dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=False,
            use_external=True,
        )
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag
        self.channel = "LIDAR_TOP"
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = list(self.CLASS_MAP.keys())
        self.queue_length = queue_length

    def __len__(self):
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.
        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        Args:
            index (int): Index of the annotation data to get.
        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASS_MAP:
                # gt_labels_3d.append(self.CLASS_MAP[cat])
                gt_labels_3d.append(self.class_names.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        origin = [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0], dtype=gt_bboxes_3d.dtype)
        src = np.array(origin, dtype=gt_bboxes_3d.dtype)
        gt_bboxes_3d[:, :3] += gt_bboxes_3d[:, 3:6] * (dst - src)
        gt_bboxes_3d = BBoxes3D(
            gt_bboxes_3d, coordmode=2, origin=[0.5, 0.5, 0.5])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:
                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        sample = Sample(path=None, modality="multiview")
        sample.sample_idx = info['token']
        sample.meta.id = info['token']
        sample.pts_filename = osp.join(self.dataset_root, info['lidar_path'])
        sample.sweeps = copy.deepcopy(info['sweeps'])
        if self.queue_length is None:
            for i in range(len(sample.sweeps)):
                for cam_type in sample.sweeps[i].keys():
                    data_path = info['sweeps'][i][cam_type]['data_path']
                    sample.sweeps[i][cam_type]['data_path'] = osp.join(
                        self.dataset_root, data_path)

        sample.timestamp = info['timestamp'] / 1e6
        if self.queue_length is not None:
            sample.ego2global_translation = info['ego2global_translation']
            sample.ego2global_rotation = info['ego2global_rotation']
            sample.prev_idx = info['prev']
            sample.next_idx = info['next']
            sample.scene_token = info['scene_token']
            sample.can_bus = info['can_bus']
            sample.frame_idx = info['frame_idx']

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(
                    osp.join(self.dataset_root, cam_info['data_path']))
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                intrinsics.append(viewpad)
                # The extrinsics mean the tranformation from lidar to camera.
                # If anyone want to use the extrinsics as sensor to lidar, please
                # use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage
                # and LoadMultiViewImageFromMultiSweepsFiles.
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            sample.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics))

        if 'train' in self.mode:
            annos = self.get_ann_info(index)
            sample.ann_info = annos

        if self.queue_length is not None:
            rotation = Quaternion(sample['ego2global_rotation'])
            translation = sample['ego2global_translation']
            can_bus = sample['can_bus']
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

        return sample

    def __getitem__(self, index):
        if 'train' not in self.mode:
            sample = self.get_data_info(index)
            sample['img_fields'] = []
            sample['bbox3d_fields'] = []
            sample['pts_mask_fields'] = []
            sample['pts_seg_fields'] = []
            sample['bbox_fields'] = []
            sample['mask_fields'] = []
            sample['seg_fields'] = []
            sample['box_type_3d'] = self.box_type_3d
            sample['box_mode_3d'] = self.box_mode_3d
            sample = self.transforms(sample)
            return sample

        while True:
            if self.queue_length is None:
                sample = self.get_data_info(index)

                if sample is None:
                    index = self._rand_another(index)
                    continue

                sample['img_fields'] = []
                sample['bbox3d_fields'] = []
                sample['pts_mask_fields'] = []
                sample['pts_seg_fields'] = []
                sample['bbox_fields'] = []
                sample['mask_fields'] = []
                sample['seg_fields'] = []
                sample['box_type_3d'] = self.box_type_3d
                sample['box_mode_3d'] = self.box_mode_3d

                sample = self.transforms(sample)

                if self.is_train_mode and self.filter_empty_gt and \
                        (sample is None or len(sample['gt_labels_3d']) == 0 ):
                    index = self._rand_another(index)
                    continue

                return sample
            else:
                queue = []
                index_list = list(range(index - self.queue_length, index))
                random.shuffle(index_list)
                index_list = sorted(index_list[1:])
                index_list.append(index)
                for i in index_list:
                    i = max(0, i)
                    sample = self.get_data_info(i)
                    if sample is None:
                        break

                    sample['img_fields'] = []
                    sample['bbox3d_fields'] = []
                    sample['pts_mask_fields'] = []
                    sample['pts_seg_fields'] = []
                    sample['bbox_fields'] = []
                    sample['mask_fields'] = []
                    sample['seg_fields'] = []
                    sample['box_type_3d'] = self.box_type_3d
                    sample['box_mode_3d'] = self.box_mode_3d

                    sample = self.transforms(sample)
                    if self.filter_empty_gt and \
                            (sample is None or len(sample['gt_labels_3d']) == 0):
                        sample = None
                        break
                    queue.append(sample)
                if sample is None:
                    index = self._rand_another(index)
                    continue
                return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'] for each in queue]
        metas_map = SampleMeta()
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['meta']
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = np.stack(imgs_list)
        queue[-1]['meta'] = metas_map
        queue = queue[-1]
        return queue

    def _build_data(self):
        test = 'test' in self.version

        if self.ann_file is not None:
            self.data_infos = pickle.load(open(self.ann_file, 'rb'))
            return

        if test:
            test_ann_cache_file = os.path.join(
                self.dataset_root,
                '{}_annotation_test.pkl'.format(self.DATASET_NAME))
            if os.path.exists(test_ann_cache_file):
                self.data_infos = pickle.load(open(test_ann_cache_file, 'rb'))
                return
        else:
            train_ann_cache_file = os.path.join(
                self.dataset_root,
                '{}_annotation_train.pkl'.format(self.DATASET_NAME))
            val_ann_cache_file = os.path.join(
                self.dataset_root,
                '{}_annotation_val.pkl'.format(self.DATASET_NAME))
            if os.path.exists(train_ann_cache_file):
                self.data_infos = pickle.load(open(train_ann_cache_file, 'rb'))
                return

        self.nusc = NuScenesManager.get(
            version=self.version, dataroot=self.dataset_root)

        if self.version == 'v1.0-trainval':
            train_scenes = nuscenes_split.train
            val_scenes = nuscenes_split.val
        elif self.version == 'v1.0-test':
            train_scenes = nuscenes_split.test
            val_scenes = []
        elif self.version == 'v1.0-mini':
            train_scenes = nuscenes_split.mini_train
            val_scenes = nuscenes_split.mini_val
        else:
            raise ValueError('unknown nuscenes dataset version')

        available_scenes = get_available_scenes(self.nusc)
        available_scene_names = [s['name'] for s in available_scenes]

        train_scenes = list(
            filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(
            filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in train_scenes
        ])
        val_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in val_scenes
        ])

        if test:
            print('test scene: {}'.format(len(train_scenes)))
        else:
            print('train scene: {}, val scene: {}'.format(
                len(train_scenes), len(val_scenes)))
        train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
            self.nusc,
            train_scenes,
            val_scenes,
            test,
            max_sweeps=self.max_sweeps)

        metadata = dict(version=self.version)

        if test:
            print('test sample: {}'.format(len(train_nusc_infos)))
            data = dict(infos=train_nusc_infos, metadata=metadata)
            pickle.dump(data, open(test_ann_cache_file, 'wb'))
            self.data_infos = data
        else:
            print('train sample: {}, val sample: {}'.format(
                len(train_nusc_infos), len(val_nusc_infos)))
            data = dict(infos=train_nusc_infos, metadata=metadata)

            pickle.dump(data, open(train_ann_cache_file, 'wb'))

            if self.mode == 'train':
                self.data_infos = data

            data['infos'] = val_nusc_infos

            pickle.dump(data, open(val_ann_cache_file, 'wb'))

            if self.mode == 'val':
                self.data_infos = data

    def _filter(self, anno: dict, box: NuScenesBox = None) -> bool:
        # filter out objects that are not being scanned
        mask = (anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 and \
            anno['category_name'] in self.LABEL_MAP and \
            self.LABEL_MAP[anno['category_name']] in self.class_names
        return mask

    def get_sweeps(self, index: int) -> List[str]:
        """
        """
        sweeps = []
        sample = self.data[index]
        token = sample['data'][self.channel]
        sample_data = self.nusc.get('sample_data', token)

        if self.max_sweeps <= 0:
            return sweeps

        # Homogeneous transform of current sample from ego car coordinate to sensor coordinate
        curr_sample_cs = self.nusc.get("calibrated_sensor",
                                       sample_data["calibrated_sensor_token"])
        curr_sensor_from_car = transform_matrix(
            curr_sample_cs["translation"],
            Quaternion(curr_sample_cs["rotation"]),
            inverse=True)
        # Homogeneous transformation matrix of current sample from global coordinate to ego car coordinate
        curr_sample_pose = self.nusc.get("ego_pose",
                                         sample_data["ego_pose_token"])
        curr_car_from_global = transform_matrix(
            curr_sample_pose["translation"],
            Quaternion(curr_sample_pose["rotation"]),
            inverse=True,
        )
        curr_timestamp = 1e-6 * sample_data["timestamp"]

        prev_token = sample_data['prev']
        while len(sweeps) < self.max_sweeps - 1:
            if prev_token == "":
                if len(sweeps) == 0:
                    sweeps.append({
                        "lidar_path":
                        osp.join(self.dataset_root, sample_data['filename']),
                        "time_lag":
                        0,
                        "ref_from_curr":
                        None,
                    })
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_sample_data = self.nusc.get('sample_data', prev_token)
                # Homogeneous transformation matrix of previous sample from ego car coordinate to global coordinate
                prev_sample_pose = self.nusc.get(
                    "ego_pose", prev_sample_data["ego_pose_token"])
                prev_global_from_car = transform_matrix(
                    prev_sample_pose["translation"],
                    Quaternion(prev_sample_pose["rotation"]),
                    inverse=False,
                )
                # Homogeneous transform of previous sample from sensor coordinate to ego car coordinate
                prev_sample_cs = self.nusc.get(
                    "calibrated_sensor",
                    prev_sample_data["calibrated_sensor_token"])
                prev_car_from_sensor = transform_matrix(
                    prev_sample_cs["translation"],
                    Quaternion(prev_sample_cs["rotation"]),
                    inverse=False,
                )

                curr_from_pre = reduce(
                    np.dot,
                    [
                        curr_sensor_from_car, curr_car_from_global,
                        prev_global_from_car, prev_car_from_sensor
                    ],
                )
                prev_timestamp = 1e-6 * prev_sample_data["timestamp"]
                time_lag = curr_timestamp - prev_timestamp

                sweeps.append({
                    "lidar_path":
                    osp.join(self.dataset_root, prev_sample_data['filename']),
                    "time_lag":
                    time_lag,
                    "ref_from_curr":
                    curr_from_pre,
                })
                prev_token = prev_sample_data['prev']
        return sweeps

    @property
    def metric(self):
        if not hasattr(self, 'nusc'):
            self.nusc = NuScenesManager.get(
                version=self.version, dataroot=self.dataset_root)
        return super().metric

    def collate_fn(self, batch: List):
        """
        """
        sample = batch[0]
        if isinstance(sample, np.ndarray):
            try:
                batch = np.stack(batch, axis=0)
                return batch
            except Exception as e:
                return batch
        elif isinstance(sample, SampleMeta):
            return batch
        return super().collate_fn(batch)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.
    Given the raw data, get the information of available scenes for
    further info generation.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
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
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.
    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    msg = "Begin to generate a info of nuScenes dataset."
    for sample_idx in logger.range(len(nusc.sample), msg=msg):
        sample = nusc.sample[sample_idx]
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        assert os.path.exists(lidar_path)

        info = {
            'lidar_token': lidar_token,
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

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


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
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


@manager.DATASETS.add_component
class NuscenesMVSegDataset(NuscenesMVDataset):
    """
    This datset only add camera intrinsics and extrinsics to the results.
    """
    DATASET_NAME = "Nuscenes"

    def __init__(self,
                 dataset_root: str,
                 ann_file: str = None,
                 lane_ann_file: str = None,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 max_sweeps: int = 10,
                 class_names: Union[list, tuple] = None,
                 use_valid_flag: bool = False,
                 load_interval: int = 1):
        self.mode = mode
        self.dataset_root = dataset_root
        self.filter_empty_gt = True
        self.box_type_3d = 'LiDAR'
        self.box_mode_3d = None
        self.ann_file = ann_file
        self.version = self.VERSION_MAP[self.mode]
        self.load_interval = load_interval
        self.queue_length = None

        self.max_sweeps = max_sweeps
        self._build_data()
        self.metadata = self.data_infos['metadata']

        self.data_infos = list(
            sorted(self.data_infos['infos'], key=lambda e: e['timestamp']))

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms

        if not self.is_test_mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)

        self.modality = dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=True,
            use_external=True,
        )
        self.with_velocity = True
        self.use_valid_flag = use_valid_flag
        self.channel = "LIDAR_TOP"
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = list(self.CLASS_MAP.keys())
        self.lane_infos = self.load_annotations(lane_ann_file)

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        lane_info = self.lane_infos[index]

        sample = Sample(path=None, modality="multiview")
        sample.sample_idx = info['token']
        sample.meta.id = info['token']
        sample.pts_filename = osp.join(self.dataset_root, info['lidar_path'])
        sample.sweeps = info['sweeps']
        if self.queue_length is None:
            for i in range(len(sample.sweeps)):
                for cam_type in sample.sweeps[i].keys():
                    data_path = sample.sweeps[i][cam_type]['data_path']
                    sample.sweeps[i][cam_type]['data_path'] = osp.join(
                        self.dataset_root, data_path)
        sample.timestamp = info['timestamp'] / 1e6
        sample.map_filename = lane_info['maps']['map_mask']

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(
                    osp.join(self.dataset_root, cam_info['data_path']))
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            sample.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics))

        if self.mode == 'train':
            annos = self.get_ann_info(index)
            sample.ann_info = annos
        return sample

    def load_annotations(self, lane_ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = pickle.load(open(lane_ann_file, 'rb'))
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    @property
    def metric(self):
        return NuScenesSegMetric(
            class_names=self.class_names,
            data_infos=self.data_infos,
            modality=self.modality,
            version=self.version,
            dataset_root=self.dataset_root)
