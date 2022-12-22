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
import paddle


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    """
    This function is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/common_utils.py#L35
    """
    cosa = paddle.cos(angle)
    sina = paddle.sin(angle)
    zeros = paddle.zeros((points.shape[0], ), dtype='float32')
    ones = paddle.ones((points.shape[0], ), dtype='float32')
    rot_matrix = paddle.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape([-1, 3, 3])
    points_rot = paddle.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = paddle.concat((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    """
    This function is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/box_utils.py#L28
    """
    template = paddle.to_tensor((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].tile([1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.reshape([-1, 8, 3]), boxes3d[:, 6]).reshape([-1, 8, 3])
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def get_voxel_centers(voxel_coords, downsample_strides, voxel_size,
                      point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_strides:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    """
    This function is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/utils/common_utils.py#L66
    """
    assert voxel_coords.shape[1] == 3
    index = paddle.to_tensor([2, 1, 0], dtype='int32')
    voxel_centers = paddle.index_select(
        voxel_coords, index, axis=-1).astype('float32')  # (xyz)
    voxel_size = paddle.to_tensor(voxel_size).astype(
        'float32') * downsample_strides
    pc_range = paddle.to_tensor(point_cloud_range[0:3]).astype('float32')
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def generate_voxel2pinds(sparse_tensor_shape, sparse_tensor_indices):
    batch_size = sparse_tensor_shape[0]
    spatial_shape = sparse_tensor_shape[1:-1]
    point_indices = paddle.arange(sparse_tensor_indices.shape[0], dtype='int32')
    output_shape = [batch_size] + list(spatial_shape)
    return paddle.scatter_nd(
        index=sparse_tensor_indices,
        updates=point_indices + 1,
        shape=output_shape) - 1


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += paddle.to_tensor(
        extra_width, dtype=large_boxes3d.dtype).reshape([1, -1])
    return large_boxes3d
