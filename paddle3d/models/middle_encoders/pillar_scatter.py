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
"""
This code is based on https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/pillar_encoder.py
Ths copyright of tianweiy/CenterPoint is as follows:
MIT License [see LICENSE for details].

https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/pillar_encoder.py fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import numpy as np
import paddle
import paddle.nn as nn

from paddle3d.apis import manager

__all__ = ['PointPillarsScatter']


@manager.MIDDLE_ENCODERS.add_component
class PointPillarsScatter(nn.Layer):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
    """

    def __init__(self, in_channels, voxel_size, point_cloud_range):
        super().__init__()
        self.in_channels = in_channels
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self.nx = int(grid_size[0])
        self.ny = int(grid_size[1])

    def forward(self, voxel_features, coords, batch_size):
        """Foraward function to scatter features."""
        return self.forward_batch(voxel_features, coords, batch_size)

    def forward_batch(self, voxel_features, coords, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (paddle.Tensor): Voxel features in shape (N, M, C).
            coords (paddle.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        if not getattr(self, "in_export_mode", False):
            # batch_canvas will be the final output.
            batch_canvas = []
            for batch_itt in range(batch_size):
                # Create the canvas for this sample
                canvas = paddle.zeros([self.nx * self.ny, self.in_channels],
                                      dtype=voxel_features.dtype)

                # Only include non-empty pillars
                batch_mask = coords[:, 0] == batch_itt
                this_coords = coords[batch_mask]
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.astype('int32')
                voxels = voxel_features[batch_mask]
                # Now scatter the blob back to the canvas.
                canvas = paddle.scatter(canvas, indices, voxels, overwrite=True)
                canvas = canvas.transpose([1, 0])

                # Append to a list for later stacking.
                batch_canvas.append(canvas)

            # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
            batch_canvas = paddle.concat(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.reshape(
                [batch_size, self.in_channels, self.ny, self.nx])
            return batch_canvas
        else:
            canvas = paddle.zeros([self.nx * self.ny, self.in_channels],
                                  dtype=voxel_features.dtype)

            # Only include non-empty pillars
            indices = coords[:, 2] * self.nx + coords[:, 3]
            indices = indices.astype('int32')
            canvas = paddle.scatter(
                canvas, indices, voxel_features, overwrite=True)
            canvas = canvas.transpose([1, 0])
            canvas = canvas.reshape([1, self.in_channels, self.ny, self.nx])
            return canvas
