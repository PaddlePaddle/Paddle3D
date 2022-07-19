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

import math

import paddle
import paddle.nn as nn


class AnchorGenerator(nn.Layer):
    """
    Generate SSD style anchors for PointPillars.

    Args:
        output_stride_factor (int): Output stride of the network.
        point_cloud_range (list[float]): [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size (list[float]): [x_size, y_size, z_size].
        anchor_configs (list[Dict[str, Any]]): Anchor configuration for each class. Attributes must include:
            "sizes": (list[float]) Anchor size (in wlh order).
            "strides": (list[float]) Anchor stride.
            "offsets": (list[float]) Anchor offset.
            "rotations": (list[float]): Anchor rotation.
            "matched_threshold": (float) IoU threshold for positive anchors.
            "unmatched_threshold": (float) IoU threshold for negative anchors.
        anchor_area_threshold (float): Threshold for filtering out anchor area. Defaults to 1.
    """

    def __init__(self,
                 output_stride_factor,
                 point_cloud_range,
                 voxel_size,
                 anchor_configs,
                 anchor_area_threshold=1):
        super(AnchorGenerator, self).__init__()
        self.pc_range = paddle.to_tensor(point_cloud_range, dtype="float32")
        self.voxel_size = paddle.to_tensor(voxel_size, dtype="float32")
        self.grid_size = paddle.round((self.pc_range[3:6] - self.pc_range[:3]) /
                                      self.voxel_size).astype("int64")

        anchor_generators = [
            AnchorGeneratorStride(**anchor_cfg) for anchor_cfg in anchor_configs
        ]
        feature_map_size = self.grid_size[:2] // output_stride_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        self._generate_anchors(feature_map_size, anchor_generators)
        self.anchor_area_threshold = float(anchor_area_threshold)

    def _generate_anchors(self, feature_map_size, anchor_generators):
        anchors_list = []
        # match_list = []
        # unmatch_list = []
        for gen in anchor_generators:
            anchors = gen.generate(feature_map_size)
            anchors = anchors.reshape(
                [*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)

        anchors = paddle.concat(anchors_list, axis=-2)
        self.anchors = anchors.reshape([-1, anchors.shape[-1]])

        anchors_bv = rbbox2d_to_circumscribed(
            paddle.index_select(
                self.anchors, paddle.to_tensor([0, 1, 3, 4, 6]), axis=1))
        anchors_bv[:, 0] = paddle.clip(
            paddle.floor(
                (anchors_bv[:, 0] - self.pc_range[0]) / self.voxel_size[0]),
            min=0)
        anchors_bv[:, 1] = paddle.clip(
            paddle.floor(
                (anchors_bv[:, 1] - self.pc_range[1]) / self.voxel_size[1]),
            min=0)
        anchors_bv[:, 2] = paddle.clip(
            paddle.floor(
                (anchors_bv[:, 2] - self.pc_range[0]) / self.voxel_size[0]),
            max=self.grid_size[0] - 1)
        anchors_bv[:, 3] = paddle.clip(
            paddle.floor(
                (anchors_bv[:, 3] - self.pc_range[1]) / self.voxel_size[1]),
            max=self.grid_size[1] - 1)
        self.anchors_bv = anchors_bv.astype("int64")

    def generate_anchors_mask(self, coords):
        # find anchors with area < threshold
        dense_voxel_map = sparse_sum_for_anchors_mask(coords,
                                                      self.grid_size[[1, 0]])
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = fused_get_anchors_area(dense_voxel_map, self.anchors_bv)
        anchors_mask = anchors_area > self.anchor_area_threshold

        return anchors_mask

    @paddle.no_grad()
    def forward(self, coords):
        return self.generate_anchors_mask(coords)


class AnchorGeneratorStride(object):
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, math.pi / 2],
                 matched_threshold=-1,
                 unmatched_threshold=-1):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._match_threshold = matched_threshold
        self._unmatch_threshold = unmatched_threshold

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    def generate(self, feature_map_size):
        """
        Args:
            feature_map_size: list [D, H, W](zyx)

        Returns:
            anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
        """

        x_stride, y_stride, z_stride = self._anchor_strides
        x_offset, y_offset, z_offset = self._anchor_offsets
        z_centers = paddle.arange(feature_map_size[0], dtype="float32")
        y_centers = paddle.arange(feature_map_size[1], dtype="float32")
        x_centers = paddle.arange(feature_map_size[2], dtype="float32")
        z_centers = z_centers * z_stride + z_offset
        y_centers = y_centers * y_stride + y_offset
        x_centers = x_centers * x_stride + x_offset
        sizes = paddle.reshape(
            paddle.to_tensor(self._sizes, dtype="float32"), [-1, 3])
        rotations = paddle.to_tensor(self._rotations, dtype="float32")
        rets = paddle.meshgrid(x_centers, y_centers, z_centers, rotations)
        tile_shape = [1] * 5
        tile_shape[-2] = sizes.shape[0]
        for i in range(len(rets)):
            rets[i] = paddle.tile(rets[i][..., None, :], tile_shape)
            rets[i] = rets[i][..., None]
        sizes = paddle.reshape(sizes, [1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = paddle.tile(sizes, tile_size_shape)
        rets.insert(3, sizes)
        ret = paddle.concat(rets, axis=-1)
        return paddle.transpose(ret, [2, 1, 0, 3, 4, 5])


def rbbox2d_to_circumscribed(rbboxes):
    """convert rotated 2D bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots = paddle.abs(limit_period(rots, 0.5, math.pi))
    cond = (rots > math.pi / 4)[..., None]
    bboxes_center_dim = paddle.where(
        cond,
        paddle.index_select(rbboxes, paddle.to_tensor([0, 1, 3, 2]), axis=1),
        rbboxes[:, :4])
    centers, dims = bboxes_center_dim[:, :2], bboxes_center_dim[:, 2:]
    bboxes = paddle.concat([centers - dims / 2, centers + dims / 2], axis=-1)

    return bboxes


def limit_period(val, offset: float = 0.5, period: float = math.pi):
    return val - paddle.floor(val / period + offset) * period


def sparse_sum_for_anchors_mask(coors, shape):
    ret = paddle.zeros(shape, dtype="float32")
    ret = paddle.scatter_nd_add(ret, coors[:, 1:3],
                                paddle.ones([coors.shape[0]], dtype="float32"))
    return ret


def fused_get_anchors_area(dense_map, anchors_bv):
    D_idx = paddle.index_select(anchors_bv, paddle.to_tensor([3, 2]), axis=1)
    A_idx = paddle.index_select(anchors_bv, paddle.to_tensor([1, 0]), axis=1)
    B_idx = paddle.index_select(anchors_bv, paddle.to_tensor([3, 0]), axis=1)
    C_idx = paddle.index_select(anchors_bv, paddle.to_tensor([1, 2]), axis=1)

    ID = paddle.gather_nd(dense_map, D_idx)
    IA = paddle.gather_nd(dense_map, A_idx)
    IB = paddle.gather_nd(dense_map, B_idx)
    IC = paddle.gather_nd(dense_map, C_idx)

    return ID - IB - IC + IA
