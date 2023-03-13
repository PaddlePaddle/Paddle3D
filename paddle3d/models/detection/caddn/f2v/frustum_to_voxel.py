# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler


class FrustumToVoxel(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/backbones_3d/f2v/frustum_to_voxel.py#L8
    """

    def __init__(self, voxel_size, pc_range, sample_cfg, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size [np.array(3)]: Voxel size [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [dict]: Depth discretiziation configuration
        """
        super().__init__()
        point_cloud_range = np.asarray(pc_range)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(
            voxel_size=voxel_size, pc_range=pc_range, disc_cfg=disc_cfg)
        self.sampler = Sampler(**sample_cfg)

    def forward(self, batch_dict):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features [paddle.Tensor(B, C, D, H_image, W_image)]: Image frustum features
                lidar_to_cam [paddle.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
                cam_to_img [paddle.Tensor(B, 3, 4)]: Camera projection matrix
                image_shape [paddle.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features [paddle.Tensor(B, C, Z, Y, X)]: Image voxel features
        """
        # Generate sampling grid for frustum volume
        grid = self.grid_generator(
            lidar_to_cam=batch_dict["trans_lidar_to_cam"],
            cam_to_img=batch_dict["trans_cam_to_img"],
            image_shape=batch_dict["image_shape"])  # (B, X, Y, Z, 3)
        # Sample frustum volume to generate voxel volume
        voxel_features = self.sampler(
            input_features=batch_dict["frustum_features"],
            grid=grid)  # (B, C, X, Y, Z)

        # (B, C, X, Y, Z) -> (B, C, Z, Y, X)
        voxel_features = voxel_features.transpose([0, 1, 4, 3, 2])
        batch_dict["voxel_features"] = voxel_features
        return batch_dict
