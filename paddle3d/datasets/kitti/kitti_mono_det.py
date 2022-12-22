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

import os

import numpy as np

from paddle3d.apis import manager
from paddle3d.datasets.kitti.kitti_det import KittiDetDataset
from paddle3d.datasets.kitti.kitti_utils import camera_record_to_object
from paddle3d.sample import Sample


@manager.DATASETS.add_component
class KittiMonoDataset(KittiDetDataset):
    """
    """

    def __getitem__(self, index: int) -> Sample:
        filename = '{}.png'.format(self.data[index])
        path = os.path.join(self.image_dir, filename)
        calibs = self.load_calibration_info(index)

        sample = Sample(path=path, modality="image")
        # P2
        sample.meta.camera_intrinsic = calibs[2][:3, :3]
        sample.meta.id = self.data[index]
        sample.calibs = calibs

        if not self.is_test_mode:
            kitti_records, ignored_kitti_records = self.load_annotation(index)
            bboxes_2d, bboxes_3d, labels = camera_record_to_object(
                kitti_records)

            sample.bboxes_2d = bboxes_2d
            sample.bboxes_3d = bboxes_3d
            sample.labels = np.array(
                [self.CLASS_MAP[label] for label in labels], dtype=np.int32)

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @property
    def image_dir(self) -> str:
        """
        """
        return os.path.join(self.base_dir, 'image_2')
