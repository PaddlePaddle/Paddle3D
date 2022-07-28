# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

__all__ = ["GenerateAnchors"]

from typing import Any, Dict, List

import numpy as np

from paddle3d.apis import manager
from paddle3d.geometries.bbox import rbbox2d_to_near_bbox
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC


@manager.TRANSFORMS.add_component
class GenerateAnchors(TransformABC):
    """
    Generate SSD style anchors for PointPillars.

    Args:
        output_stride_factor (int): Output stride of the network.
        point_cloud_range (List[float]): [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size (List[float]): [x_size, y_size, z_size].
        anchor_configs (List[Dict[str, Any]]): Anchor configuration for each class. Attributes must include:
            "sizes": (List[float]) Anchor size (in wlh order).
            "strides": (List[float]) Anchor stride.
            "offsets": (List[float]) Anchor offset.
            "rotations": (List[float]): Anchor rotation.
            "matched_threshold": (float) IoU threshold for positive anchors.
            "unmatched_threshold": (float) IoU threshold for negative anchors.
        anchor_area_threshold (float): Threshold for filtering out anchor area. Defaults to 1.
    """

    def __init__(self,
                 output_stride_factor: int,
                 point_cloud_range: List[float],
                 voxel_size: List[float],
                 anchor_configs: List[Dict[str, Any]],
                 anchor_area_threshold: int = 1):
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[:3]) /
            self.voxel_size).astype(np.int64)

        anchor_generators = [
            AnchorGeneratorStride(**anchor_cfg) for anchor_cfg in anchor_configs
        ]
        feature_map_size = self.grid_size[:2] // output_stride_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        self._generate_anchors(feature_map_size, anchor_generators)
        self.anchor_area_threshold = anchor_area_threshold

    def _generate_anchors(self, feature_map_size, anchor_generators):
        anchors_list = []
        match_list = []
        unmatch_list = []
        for gen in anchor_generators:
            anchors = gen.generate(feature_map_size)
            anchors = anchors.reshape(
                [*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full((num_anchors, ), gen.match_threshold, anchors.dtype))
            unmatch_list.append(
                np.full((num_anchors, ), gen.unmatch_threshold, anchors.dtype))

        anchors = np.concatenate(anchors_list, axis=-2)
        self.matched_thresholds = np.concatenate(match_list, axis=0)
        self.unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        self.anchors = anchors.reshape([-1, anchors.shape[-1]])
        self.anchors_bv = rbbox2d_to_near_bbox(self.anchors[:, [0, 1, 3, 4, 6]])

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GenerateAnchors only supports lidar data!")

        sample.anchors = self.anchors
        sample.matched_thresholds = self.matched_thresholds
        sample.unmatched_thresholds = self.unmatched_thresholds

        if self.anchor_area_threshold >= 0:
            # find anchors with area < threshold
            dense_voxel_map = F.sparse_sum_for_anchors_mask(
                sample.coords, tuple(self.grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = F.fused_get_anchors_area(
                dense_voxel_map, self.anchors_bv, self.voxel_size,
                self.point_cloud_range, self.grid_size)
            anchors_mask = anchors_area > self.anchor_area_threshold
            sample.anchors_mask = anchors_mask

        return sample


class AnchorGeneratorStride(object):
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
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
        return F.create_anchors_3d_stride(feature_map_size, self._sizes,
                                          self._anchor_strides,
                                          self._anchor_offsets, self._rotations)
