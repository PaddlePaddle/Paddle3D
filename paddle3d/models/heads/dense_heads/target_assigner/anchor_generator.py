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

import paddle


class AnchorGenerator(object):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/dense_heads/target_assigner/anchor_generator.py#L4
    """

    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [
            config['anchor_sizes'] for config in anchor_generator_config
        ]
        self.anchor_rotations = [
            config['anchor_rotations'] for config in anchor_generator_config
        ]
        self.anchor_heights = [
            config['anchor_bottom_heights']
            for config in anchor_generator_config
        ]
        self.align_center = [
            config.get('align_center', False)
            for config in anchor_generator_config
        ]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(
            self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []

        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations,
                self.anchor_heights, self.align_center):

            num_anchors_per_location.append(
                len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (
                    self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (
                    self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (
                    grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (
                    grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = paddle.arange(
                self.anchor_range[0] + x_offset,
                self.anchor_range[3] + 1e-5,
                step=x_stride,
                dtype='float32',
            )
            y_shifts = paddle.arange(
                self.anchor_range[1] + y_offset,
                self.anchor_range[4] + 1e-5,
                step=y_stride,
                dtype='float32',
            )
            z_shifts = paddle.to_tensor(anchor_height, dtype='float32')

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(
            ), anchor_rotation.__len__()
            anchor_rotation = paddle.to_tensor(anchor_rotation, dtype='float32')
            anchor_size = paddle.to_tensor(anchor_size, dtype='float32')
            x_shifts, y_shifts, z_shifts = paddle.meshgrid(
                [x_shifts, y_shifts, z_shifts])  # [x_grid, y_grid, z_grid]
            anchors = paddle.stack((x_shifts, y_shifts, z_shifts),
                                   axis=-1)  # [x, y, z, 3]
            anchors = anchors.unsqueeze([3]).tile(
                [1, 1, 1, paddle.shape(anchor_size)[0], 1])
            anchor_size = anchor_size.reshape([1, 1, 1, -1, 3]).tile(
                [*paddle.shape(anchors)[0:3], 1, 1])
            anchors = paddle.concat((anchors, anchor_size), axis=-1)
            anchors = anchors.unsqueeze([4]).tile(
                [1, 1, 1, 1, num_anchor_rotation, 1])
            anchor_rotation = anchor_rotation.reshape([1, 1, 1, 1, -1, 1]).tile(
                [*paddle.shape(anchors)[0:3], num_anchor_size, 1, 1])
            anchors = paddle.concat((anchors, anchor_rotation),
                                    axis=-1)  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.transpose([2, 1, 0, 3, 4, 5])
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location
