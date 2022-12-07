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

from typing import Callable, List, Tuple, Union

import numpy as np
import paddle
from nuscenes.utils import splits as nuscenes_split
from pyquaternion import Quaternion

import paddle3d.transforms as T
from paddle3d.datasets import BaseDataset
from paddle3d.datasets.nuscenes.nuscenes_manager import NuScenesManager
from paddle3d.datasets.nuscenes.nuscenes_metric import NuScenesMetric
from paddle3d.geometries import BBoxes2D, BBoxes3D
from paddle3d.sample import Sample
from paddle3d.transforms import TransformABC
from paddle3d.utils.common import generate_tempdir


class NuscenesDetDataset(BaseDataset):
    """
    """

    VERSION_MAP = {
        'train': 'v1.0-trainval',
        'val': 'v1.0-trainval',
        'trainval': 'v1.0-trainval',
        'test': 'v1.0-test',
        'mini_train': 'v1.0-mini',
        'mini_val': 'v1.0-mini'
    }

    LABEL_MAP = {
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone'
    }

    CLASS_MAP = {
        'pedestrian': 0,
        'car': 1,
        'motorcycle': 2,
        'bicycle': 3,
        'bus': 4,
        'truck': 5,
        'construction_vehicle': 6,
        'trailer': 7,
        'barrier': 8,
        'traffic_cone': 9
    }
    CLASS_MAP_REVERSE = {value: key for key, value in CLASS_MAP.items()}

    ATTRIBUTE_MAP = {
        'vehicle.moving': 0,
        'vehicle.stopped': 1,
        'vehicle.parked': 2,
        'cycle.with_rider': 3,
        'cycle.without_rider': 4,
        'pedestrian.sitting_lying_down': 5,
        'pedestrian.standing': 6,
        'pedestrian.moving': 7,
        '': 8
    }
    ATTRIBUTE_MAP_REVERSE = {value: key for key, value in ATTRIBUTE_MAP.items()}

    SUPPORT_CHANNELS = [
        "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT", "LIDAR_TOP", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT"
    ]

    DEFAULT_ATTRIBUTE_MAP = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': ''
    }

    def __init__(self,
                 dataset_root: str,
                 channel: str,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 class_balanced_sampling: bool = False,
                 class_names: Union[list, tuple] = None):
        super().__init__()
        self.dataset_root = dataset_root
        self.mode = mode.lower()
        self.channel = channel
        self.class_balanced_sampling = class_balanced_sampling
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = list(self.CLASS_MAP.keys())

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms

        if self.mode not in [
                'train', 'val', 'trainval', 'test', 'mini_train', 'mini_val'
        ]:
            raise ValueError(
                "mode should be 'train', 'val', 'trainval', 'mini_train', 'mini_val' or 'test', but got {}."
                .format(self.mode))

        if self.channel not in self.SUPPORT_CHANNELS:
            raise ValueError('Only channel {} is supported, but got {}'.format(
                self.SUPPORT_CHANNELS, self.channel))

        self.version = self.VERSION_MAP[self.mode]
        self.nusc = NuScenesManager.get(
            version=self.version, dataroot=self.dataset_root)
        self._build_data(class_balanced_sampling)

    def _build_data(self, class_balanced_sampling):
        scenes = getattr(nuscenes_split, self.mode)
        self.data = []

        for scene in self.nusc.scene:
            if scene['name'] not in scenes:
                continue

            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            cur_token = first_sample_token
            first_sample = self.nusc.get('sample', first_sample_token)

            while True:
                sample = self.nusc.get('sample', cur_token)
                self.data.append(sample)

                if cur_token == last_sample_token:
                    break

                cur_token = sample['next']

        if self.class_balanced_sampling and self.mode.lower(
        ) == 'train' and len(self.class_names) > 1:
            cls_dist = {class_name: [] for class_name in self.class_names}
            for index in range(len(self.data)):
                sample = self.data[index]
                gt_names = []
                for anno in sample['anns']:
                    anno = self.nusc.get('sample_annotation', anno)
                    if not self._filter(anno):
                        continue
                    class_name = self.LABEL_MAP[anno['category_name']]
                    if class_name in self.class_names:
                        gt_names.append(class_name)
                for class_name in set(gt_names):
                    cls_dist[class_name].append(sample)

            num_balanced_samples = sum([len(v) for k, v in cls_dist.items()])
            num_balanced_samples = max(num_balanced_samples, 1)
            balanced_frac = 1.0 / len(self.class_names)
            fracs = [len(v) / num_balanced_samples for k, v in cls_dist.items()]
            sampling_ratios = [balanced_frac / frac for frac in fracs]

            resampling_data = []
            for samples, sampling_ratio in zip(
                    list(cls_dist.values()), sampling_ratios):
                resampling_data.extend(
                    np.random.choice(samples, int(
                        len(samples) * sampling_ratio)).tolist())
            self.data = resampling_data

    def __len__(self):
        return len(self.data)

    def load_annotation(self, index: int, filter: Callable = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        bboxes = []
        labels = []
        velocities = []
        attrs = []

        sample = self.data[index]
        sample_data = self.nusc.get('sample_data', sample['data'][self.channel])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        channel_pose = self.nusc.get('calibrated_sensor',
                                     sample_data['calibrated_sensor_token'])

        for anno in sample['anns']:
            box = self.nusc.get_box(anno)
            box.velocity = self.nusc.box_velocity(box.token)

            # from global-coord to ego-coord
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            # from ego-coord to sensor-coord
            box.translate(-np.array(channel_pose['translation']))
            box.rotate(Quaternion(channel_pose['rotation']).inverse)

            anno = self.nusc.get('sample_annotation', anno)
            if not anno[
                    'category_name'] in self.LABEL_MAP:  # also filter ["DontCare", "ignore", "UNKNOWN"]
                continue

            # filter out objects that do not meet the conditions
            if filter and not filter(anno, box):
                continue

            # add velocity
            # loaded velocity may be nan when using nuscenes_devkit<=1.1.9
            # so we reset nan velocity to zero
            velocity = np.array(box.velocity)
            velocity[np.isnan(velocity)] = 0
            velocities.append(velocity[:2])

            # get attribute
            clsname = self.LABEL_MAP[anno['category_name']]
            label = self.class_names.index(clsname)

            if len(anno['attribute_tokens']) == 0:
                attr_name = self.DEFAULT_ATTRIBUTE_MAP[clsname]
            else:
                attr_token = anno['attribute_tokens'][0]
                attr_name = self.nusc.get('attribute', attr_token)['name']
            attrs.append(self.ATTRIBUTE_MAP[attr_name])

            # TODO: Fix me
            x, y, z = box.center
            w, l, h = box.wlh
            #yaw = box.orientation.yaw_pitch_roll[0] #TODO(luoqianhui): check this yaw
            v = np.dot(box.orientation.rotation_matrix, np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0])

            bbox3d = np.array(
                [x, y, z, w, l, h, -(yaw + np.pi / 2)
                 ],  #TODO(luoqianhui): check this positive sign of yaw
                dtype=np.float32)
            # loaded bounding box may be nan when using nuscenes_devkit<=1.1.9
            # so we reset nan box to zero
            bbox3d[np.isnan(bbox3d)] = 0
            bboxes.append(bbox3d)
            labels.append(label)

        bboxes = BBoxes3D(
            bboxes, origin=(0.5, 0.5, 0.5), velocities=np.array(velocities))
        labels = np.array(labels, dtype=np.int32)
        attrs = np.array(attrs, dtype=np.int32)

        return bboxes, labels, attrs

    def padding_sample(self, samples: List[Sample]):
        # do nothing for sweeps
        if samples[0].labels is None:
            return

        maxlen = max([len(sample.labels) for sample in samples])
        padding_lens = [maxlen - len(sample.labels) for sample in samples]

        for padlen, sample in zip(padding_lens, samples):
            if padlen == 0:
                continue

            _pad_item = np.ones([padlen], np.int32) * -1
            sample.labels = np.append(sample.labels, _pad_item)

            if sample.bboxes_2d is not None:
                _pad_item = np.zeros([padlen, sample.bboxes_2d.shape[1]],
                                     np.float32)
                sample.bboxes_2d = BBoxes2D(
                    np.append(sample.bboxes_2d, _pad_item, axis=0))

            if sample.bboxes_3d is not None:
                _pad_item = np.zeros([padlen, sample.bboxes_3d.shape[1]],
                                     np.float32)
                sample.bboxes_3d = BBoxes3D(
                    np.append(sample.bboxes_3d, _pad_item, axis=0))

            if sample.velocities is not None:
                _pad_item = np.zeros([padlen, 2], np.float32)
                sample.velocities = np.append(
                    sample.velocities, _pad_item, axis=0)

            if sample.attrs is not None:
                _pad_item = np.ones([padlen], np.int32) * -1
                sample.attrs = np.append(sample.attrs, _pad_item)

    @property
    def metric(self):
        return NuScenesMetric(
            nuscense=self.nusc,
            mode=self.mode,
            channel=self.channel,
            class_names=self.class_names,
            attrmap=self.ATTRIBUTE_MAP_REVERSE)

    @property
    def name(self) -> str:
        return "nuScenes"

    @property
    def labels(self) -> List[str]:
        return self.class_names
