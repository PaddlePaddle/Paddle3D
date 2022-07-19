from typing import Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.ops.voxelize import hard_voxelize


class Voxelization(nn.Layer):
    def __init__(
            self,
            voxel_size: Tuple[float, float, float],
            point_cloud_range: Tuple[float, float, float, float, float, float],
            max_num_points: int,
            max_voxels: Union[int, Tuple[int, int]] = 20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = (max_voxels, max_voxels)

        point_cloud_range = np.asarray(point_cloud_range)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype('int32')
        input_feat_shape = grid_size[:2]

        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, points):
        """
        Args:
            input: N x C points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        batch_voxels, batch_coors, batch_num_points = [], [], []

        for point in points:
            new_point = paddle.to_tensor(point.numpy(), stop_gradient=True)
            voxel_num, voxels, coors, num_points_per_voxel = \
                hard_voxelize(new_point, self.voxel_size,
                    self.point_cloud_range, self.max_num_points,
                    max_voxels, 3)

            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            batch_voxels.append(voxels_out)
            batch_coors.append(coors_out)
            batch_num_points.append(num_points_per_voxel_out)

        voxels_batch = paddle.concat(batch_voxels, axis=0)
        num_points_batch = paddle.concat(batch_num_points, axis=0)
        coors_batch = []

        for i, coor in enumerate(batch_coors):
            bs_idx = paddle.full(
                shape=[coor.shape[0], 1], fill_value=i, dtype=coor.dtype)
            coor_pad = paddle.concat([bs_idx, coor], axis=1)
            coors_batch.append(coor_pad)
        coors_batch = paddle.concat(coors_batch, axis=0)

        return voxels_batch, num_points_batch, coors_batch
