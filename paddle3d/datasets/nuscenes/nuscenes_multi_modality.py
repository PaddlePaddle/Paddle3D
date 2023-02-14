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

import pickle

import numpy as np

import paddle3d.transforms as T
from paddle3d.apis import manager
from paddle3d.datasets.nuscenes.nuscenes_det import NuscenesDetDataset
from paddle3d.datasets.nuscenes.nuscenes_manager import NuScenesManager
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample
from paddle3d.utils.logger import logger

__all__ = ['NuscenesMMDataset']


@manager.DATASETS.add_component
class NuscenesMMDataset(NuscenesDetDataset):
    def __init__(
            self,
            ann_file,
            num_views=6,
            data_root=None,
            class_names=None,
            load_interval=1,
            with_velocity=True,
            modality=None,
            box_type_3d='LiDAR',
            filter_empty_gt=True,
            test_mode=False,
            test_gt=False,
            use_valid_flag=False,
            transforms=None,
            mode='train',
            # additional
            extrinsics_noise=False,
            extrinsics_noise_type='single',
            drop_frames=False,
            drop_set=[0, 'discrete'],
            noise_sensor_type='camera'):

        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.data_root = data_root
        self.mode = mode
        self.test_mode = test_mode
        self.test_gt = test_gt
        self.filter_empty_gt = filter_empty_gt
        self.channel = 'LIDAR_TOP'
        self.version = self.VERSION_MAP[self.mode]

        self.num_views = num_views
        assert self.num_views <= 6
        self.with_velocity = with_velocity
        self.modality = modality

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)
        self.transforms = transforms

        if class_names is None:
            self.class_names = list(self.CLASS_MAP.keys())
        else:
            self.class_names = class_names

        self.data_infos = self.load_annotations(ann_file)

        if not self.test_mode:
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

        # for drop foreground points
        self.test_gt = test_gt
        self.extrinsics_noise = extrinsics_noise
        assert extrinsics_noise_type in ['all', 'single']
        self.extrinsics_noise_type = extrinsics_noise_type
        self.drop_frames = drop_frames
        self.drop_ratio = drop_set[0]
        self.drop_type = drop_set[1]
        self.noise_sensor_type = noise_sensor_type

        if self.extrinsics_noise or self.drop_frames:
            raise NotImplementedError
        else:
            self.noise_data = None

        if self.drop_frames:
            logger.info('frame drop setting: drop ratio:', self.drop_ratio,
                        ', sensor type:', self.noise_sensor_type,
                        ', drop type:', self.drop_type)
        if self.extrinsics_noise:
            assert noise_sensor_type == 'camera'
            logger.info(f'add {extrinsics_noise_type} noise to extrinsics')

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.mode != 'train':
            return self.prepare_test_data(idx)
        while True:
            sample = self.prepare_train_data(idx)
            if sample is None:
                idx = self._rand_another(idx)
                continue
            return sample

    def __len__(self):
        return len(self.data_infos)

    def add_new_fields(self, sample):
        sample['img_fields'] = []
        sample['bbox3d_fields'] = []
        sample['pts_mask_fields'] = []
        sample['pts_seg_fields'] = []
        sample['bbox_fields'] = []
        sample['mask_fields'] = []
        sample['seg_fields'] = []
        return sample

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        sample = self.get_data_info(index)
        if sample is None:
            return None

        sample = self.add_new_fields(sample)

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.filter_empty_gt and (sample is None
                                     or ~(sample['gt_labels_3d'] != -1).any()):
            return None
        return sample

    def prepare_test_data(self, index):
        sample = self.get_data_info(index)
        sample = self.add_new_fields(sample)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = pickle.load(open(ann_file, 'rb'))
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        return data_infos

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
        # standard protocal modified from SECOND.Pytorch
        sample = Sample(path=None, modality=self.modality)
        sample.sample_idx = info['token']
        sample.meta.id = info['token']
        sample.pts_filename = info['lidar_path']
        sample.sweeps = info['sweeps']
        sample.timestamp = info['timestamp'] / 1e6

        if self.noise_sensor_type == 'lidar':
            if self.drop_frames:
                pts_filename = sample.pts_filename
                file_name = pts_filename.split('/')[-1]

                if self.noise_data[file_name]['noise']['drop_frames'][
                        self.drop_ratio][self.drop_type]['stuck']:
                    replace_file = self.noise_data[file_name]['noise'][
                        'drop_frames'][self.drop_ratio][
                            self.drop_type]['replace']
                    if replace_file != '':
                        pts_filename = pts_filename.replace(
                            file_name, replace_file)

                        sample.pts_filename = pts_filename
                        sample.sweeps = self.noise_data[replace_file][
                            'mmdet_info']['sweeps']
                        sample.timestamp = self.noise_data[replace_file][
                            'mmdet_info']['timestamp'] / 1e6

        cam_orders = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT'
        ]
        if self.modality == 'multiview' or self.modality == 'multimodal':
            image_paths = []
            lidar2img_rts = []
            caminfos = []
            # for cam_type, cam_info in info['cams'].items():
            for cam_type in cam_orders:
                cam_info = info['cams'][cam_type]

                cam_data_path = cam_info['data_path']
                file_name = cam_data_path.split('/')[-1]
                if self.noise_sensor_type == 'camera':
                    if self.drop_frames:
                        if self.noise_data[file_name]['noise']['drop_frames'][
                                self.drop_ratio][self.drop_type]['stuck']:
                            replace_file = self.noise_data[file_name]['noise'][
                                'drop_frames'][self.drop_ratio][
                                    self.drop_type]['replace']
                            if replace_file != '':
                                cam_data_path = cam_data_path.replace(
                                    file_name, replace_file)

                image_paths.append(cam_data_path)
                # obtain lidar to image transformation matrix
                if self.extrinsics_noise:
                    sensor2lidar_rotation = self.noise_data[file_name]['noise'][
                        'extrinsics_noise'][
                            f'{self.extrinsics_noise_type}_noise_sensor2lidar_rotation']
                    sensor2lidar_translation = self.noise_data[file_name][
                        'noise']['extrinsics_noise'][
                            f'{self.extrinsics_noise_type}_noise_sensor2lidar_translation']
                else:
                    sensor2lidar_rotation = cam_info['sensor2lidar_rotation']
                    sensor2lidar_translation = cam_info[
                        'sensor2lidar_translation']

                lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
                lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                caminfos.append({
                    'sensor2lidar_translation': sensor2lidar_translation,
                    'sensor2lidar_rotation': sensor2lidar_rotation,
                    'cam_intrinsic': cam_info['cam_intrinsic']
                })

            sample.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    caminfo=caminfos))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            sample.ann_info = annos

        return sample

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
            if cat in self.class_names:
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
        dst = np.array([0.5, 0.5, 0])
        src = np.array(origin)
        gt_bboxes_3d[:, :3] += gt_bboxes_3d[:, 3:6] * (dst - src)
        gt_bboxes_3d = BBoxes3D(
            gt_bboxes_3d, coordmode=2, origin=[0.5, 0.5, 0.5])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    @property
    def metric(self):
        if not hasattr(self, 'nusc'):
            self.nusc = NuScenesManager.get(
                version=self.version, dataroot=self.data_root)
        return super().metric
