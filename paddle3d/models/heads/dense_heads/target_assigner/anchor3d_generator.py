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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/core/anchor/anchor_3d_generator.py

import paddle

from paddle3d.apis import manager

__all__ = ['AlignedAnchor3DRangeGenerator']


class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.
    Due the convention in 3D detection, different anchor sizes are related to
    different ranges for different categories. However we find this setting
    does not effect the performance much in some datasets, e.g., nuScenes.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        scales (list[int]): Scales of anchors in different feature levels.
        rotations (list[float]): Rotations of anchors in a feature grid.
        custom_values (tuple[float]): Customized values of that anchor. For
            example, in nuScenes the anchors have velocities.
        reshape_out (bool): Whether to reshape the output into (N x 4).
        size_per_range: Whether to use separate ranges for different sizes.
            If size_per_range is True, the ranges should have the same length
            as the sizes, if not, it will be duplicated.
    """

    def __init__(self,
                 ranges,
                 sizes=[[1.6, 3.9, 1.56]],
                 scales=[1],
                 rotations=[0, 1.5707963],
                 custom_values=(),
                 reshape_out=True,
                 size_per_range=True):
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes)
        else:
            assert len(ranges) == 1
        assert isinstance(scales, list)

        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.cached_anchors = None
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range

    @property
    def num_base_anchors(self):
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = paddle.to_tensor(self.sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def num_levels(self):
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(self, featmap_sizes):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Returns:
            list[paddle.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature lavel, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(featmap_sizes[i],
                                                     self.scales[i])
            if self.reshape_out:
                anchors = anchors.reshape([-1, anchors.shape[-1]])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self, featmap_size, scale):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            paddle.Tensor: Anchors in the overall feature map.
        """
        if not self.size_per_range:
            return self.anchors_single_range(featmap_size, self.ranges[0],
                                             scale, self.sizes, self.rotations)

        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(featmap_size, anchor_range, scale,
                                          anchor_size, self.rotations))
        mr_anchors = paddle.concat(mr_anchors, axis=-3)
        return mr_anchors

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             scale=1,
                             sizes=[[1.6, 3.9, 1.56]],
                             rotations=[0, 1.5707963]):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (paddle.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int, optional): The scale factor of anchors.
            sizes (list[list] | np.ndarray | paddle.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | paddle.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.

        Returns:
            paddle.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = paddle.to_tensor(anchor_range)
        z_centers = paddle.linspace(anchor_range[2], anchor_range[5],
                                    feature_size[0])
        y_centers = paddle.linspace(anchor_range[1], anchor_range[4],
                                    feature_size[1])
        x_centers = paddle.linspace(anchor_range[0], anchor_range[3],
                                    feature_size[2])
        sizes = paddle.to_tensor(sizes).reshape([-1, 3]) * scale
        rotations = paddle.to_tensor(rotations)

        # paddle.meshgrid default behavior is 'id', np's default is 'xy'
        rets = paddle.meshgrid(x_centers, y_centers, z_centers, rotations)
        # paddle.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).tile(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.tile(tile_size_shape)
        rets.insert(3, sizes)

        ret = paddle.concat(rets, axis=-1).transpose([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            # custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            custom = paddle.zeros([*ret.shape[:-1, custom_ndim]])
            ret = paddle.concat([ret, custom], axis=-1)
            # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret


@manager.HEADS.add_component
class AlignedAnchor3DRangeGenerator(Anchor3DRangeGenerator):
    """Aligned 3D Anchor Generator by range.

    This anchor generator uses a different manner to generate the positions
    of anchors' centers from :class:`Anchor3DRangeGenerator`.

    Note:
        The `align` means that the anchor's center is aligned with the voxel
        grid, which is also the feature grid. The previous implementation of
        :class:`Anchor3DRangeGenerator` does not generate the anchors' center
        according to the voxel grid. Rather, it generates the center by
        uniformly distributing the anchors inside the minimum and maximum
        anchor ranges according to the feature map sizes.
        However, this makes the anchors center does not match the feature grid.
        The :class:`AlignedAnchor3DRangeGenerator` add + 1 when using the
        feature map sizes to obtain the corners of the voxel grid. Then it
        shifts the coordinates to the center of voxel grid and use the left
        up corner to distribute anchors.

    Args:
        anchor_corner (bool): Whether to align with the corner of the voxel
            grid. By default it is False and the anchor's center will be
            the same as the corresponding voxel's center, which is also the
            center of the corresponding greature grid.
    """

    def __init__(self, align_corner=False, **kwargs):
        super(AlignedAnchor3DRangeGenerator, self).__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             scale,
                             sizes=[[1.6, 3.9, 1.56]],
                             rotations=[0, 1.5707963]):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (paddle.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int, optional): The scale factor of anchors.
            sizes (list[list] | np.ndarray | paddle.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | paddle.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.

        Returns:
            paddle.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = paddle.to_tensor(anchor_range)
        z_centers = paddle.linspace(anchor_range[2], anchor_range[5],
                                    feature_size[0] + 1)
        y_centers = paddle.linspace(anchor_range[1], anchor_range[4],
                                    feature_size[1] + 1)
        x_centers = paddle.linspace(anchor_range[0], anchor_range[3],
                                    feature_size[2] + 1)
        sizes = paddle.to_tensor(sizes).reshape([-1, 3]) * scale
        rotations = paddle.to_tensor(rotations)

        # shift the anchor center
        if not self.align_corner:
            z_shift = (z_centers[1] - z_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2
            x_shift = (x_centers[1] - x_centers[0]) / 2
            z_centers += z_shift
            y_centers += y_shift
            x_centers += x_shift

        # paddle.meshgrid default behavior is 'id', np's default is 'xy'
        rets = paddle.meshgrid(x_centers[:feature_size[2]],
                               y_centers[:feature_size[1]],
                               z_centers[:feature_size[0]], rotations)

        # paddle.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).tile(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.tile(tile_size_shape)
        rets.insert(3, sizes)

        ret = paddle.concat(rets, axis=-1).transpose([2, 1, 0, 3, 4, 5])

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = paddle.zeros([*ret.shape[:-1], custom_ndim])
            # TODO: check the support of custom values
            # custom[:] = self.custom_values
            ret = paddle.concat([ret, custom], axis=-1)
        return ret
