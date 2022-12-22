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

import math

import numpy as np
import paddle
import paddle.nn as nn

from paddle3d.utils.depth import bin_depths
from paddle3d.utils.grid import create_meshgrid3d, normalize_coords
from paddle3d.utils.transform import project_to_image, transform_points_3d


class FrustumGridGenerator(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/backbones_3d/f2v/frustum_grid_generator.py#L8
    """

    def __init__(self, voxel_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            voxel_size [np.array(3)]: Voxel size [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [int]: Depth discretiziation configuration
        """
        super().__init__()
        self.dtype = 'float32'
        point_cloud_range = np.asarray(pc_range)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self.grid_size = paddle.to_tensor(grid_size)
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        pc_range = paddle.to_tensor(pc_range).reshape([2, 3])
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = paddle.to_tensor(voxel_size)

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.cast('int32')
        self.voxel_grid = create_meshgrid3d(
            depth=self.depth,
            height=self.height,
            width=self.width,
            normalized_coordinates=False)
        self.voxel_grid = self.voxel_grid.transpose([0, 1, 3, 2,
                                                     4])  # XZY-> XYZ

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(
            pc_min=self.pc_min, voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min [paddle.Tensor(3)]: Minimum of point cloud range [X, Y, Z] (m)
            voxel_size [paddle.Tensor(3)]: Size of each voxel [X, Y, Z] (m)
        Returns:
            unproject [paddle.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size.cpu().numpy()
        x_min, y_min, z_min = pc_min.cpu().numpy()
        unproject = paddle.to_tensor(
            [[x_size, 0, 0, x_min], [0, y_size, 0, y_min],
             [0, 0, z_size, z_min], [0, 0, 0, 1]],
            dtype=self.dtype)  # (4, 4)

        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam,
                       cam_to_img):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid [paddle.Tensor(B, X, Y, Z, 3)]: Voxel sampling grid
            grid_to_lidar [paddle.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
            lidar_to_cam [paddle.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [paddle.Tensor(B, 3, 4)]: Camera projection matrix
        Returns:
            frustum_grid [paddle.Tensor(B, X, Y, Z, 3)]: Frustum sampling grid
        """
        B = paddle.shape(lidar_to_cam)[0]
        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ V_G

        # Reshape to match dimensions
        trans01 = trans.reshape((B, 1, 1, 4, 4))
        voxel_grid = voxel_grid.tile((B, 1, 1, 1, 1))

        # Transform to camera frame
        camera_grid = transform_points_3d(trans_01=trans01, points_1=voxel_grid)

        # Project to image
        I_C = I_C.reshape([B, 1, 1, 3, 4])
        image_grid, image_depths = project_to_image(
            project=I_C, points=camera_grid)
        # Convert depths to depth bins
        image_depths = bin_depths(depth_map=image_depths, **self.disc_cfg)
        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = paddle.concat((image_grid, image_depths), axis=-1)
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam [paddle.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [paddle.Tensor(B, 3, 4)]: Camera projection matrix
            image_shape [paddle.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            frustum_grid [paddle.Tensor(B, X, Y, Z, 3)]: Sampling grids for frustum features
        """

        frustum_grid = self.transform_grid(
            voxel_grid=self.voxel_grid,
            grid_to_lidar=self.grid_to_lidar,
            lidar_to_cam=lidar_to_cam,
            cam_to_img=cam_to_img)
        # Normalize grid
        image_shape = paddle.max(image_shape, axis=0)
        image_depth = paddle.to_tensor([self.disc_cfg["num_bins"]]).cast(
            image_shape.dtype)
        frustum_shape = paddle.concat((image_depth, image_shape))
        frustum_grid = normalize_coords(
            coords=frustum_grid, shape=frustum_shape)

        # Replace any NaNs or infinites with out of bounds
        mask = ~paddle.isfinite(frustum_grid)
        sub_val = paddle.full(
            shape=paddle.shape(mask), fill_value=self.out_of_bounds_val)
        frustum_grid = paddle.where(mask, sub_val, frustum_grid)

        return frustum_grid
