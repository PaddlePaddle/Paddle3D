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


def convert_points_to_homogeneous(points):
    """Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    """
    if len(list(paddle.shape(points))) == 3:
        data_format = "NCL"
    elif len(list(paddle.shape(points))) == 5:
        data_format = "NCDHW"
    return paddle.nn.functional.pad(
        points, [0, 1], "constant", 1.0, data_format=data_format)


def convert_points_from_homogeneous_3d(points, eps=1e-8):
    """Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = paddle.to_tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    # we check for points at max_val
    z_vec = points[..., 3:]
    mask = paddle.abs(z_vec) > eps
    scale = paddle.where(mask, 1.0 / (z_vec + eps), paddle.ones_like(z_vec))

    return scale * points[..., :3]


def convert_points_from_homogeneous_2d(points, eps=1e-8):
    """Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = paddle.to_tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    # we check for points at max_val
    z_vec = points[..., 2:]
    mask = paddle.abs(z_vec) > eps
    scale = paddle.where(mask, 1.0 / (z_vec + eps), paddle.ones_like(z_vec))

    return scale * points[..., :2]


def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [paddle.tensor(..., 3, 4)]: Projection matrix
        points [paddle.Tensor(..., 3)]: 3D points
    Returns:
        points_img [paddle.Tensor(..., 2)]: Points in image
        points_depth [paddle.Tensor(...)]: Depth of each point
    """
    shape_inp = list(paddle.shape(points))
    shape_inp_len = len(shape_inp)
    points = points.reshape(
        [-1, shape_inp[shape_inp_len - 2], shape_inp[shape_inp_len - 1]])
    shape_inp[shape_inp_len - 1] += 1

    points = convert_points_to_homogeneous(points)
    points = points.reshape([shape_inp[0], -1,
                             shape_inp[-1]]).transpose([0, 2, 1])
    project_shape = project.shape
    project = project.reshape(
        [project_shape[0], project_shape[-2], project_shape[-1]])

    # Transform points to image and get depths
    points_shape = points.shape
    points_t = project @ points
    points_t = points_t.transpose([0, 2, 1])

    points_t_shape = points_t.shape
    points_t = points_t.reshape([
        points_t_shape[0], shape_inp[1], shape_inp[2], shape_inp[3],
        points_t_shape[-1]
    ])

    points_img = convert_points_from_homogeneous_2d(points_t)

    project = project.reshape(project_shape).unsqueeze(axis=1)
    points_depth = points_t[..., -1] - project[..., 2, 3]

    return points_img, points_depth


def transform_points_3d(trans_01, points_1):
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01 (Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    """

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(paddle.shape(points_1))
    dim_num = len(shape_inp)
    points_1 = points_1.reshape([-1, shape_inp[dim_num - 2], 3])
    trans_01 = trans_01.reshape([-1, 4, 4])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = trans_01.tile(
        [paddle.shape(points_1)[0] // paddle.shape(trans_01)[0], 1, 1])
    # to homogeneous

    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1

    # transform coordinates
    points_0_h = paddle.matmul(points_1_h, trans_01.transpose([0, 2, 1]))
    points_0_h = paddle.squeeze(points_0_h, axis=-1)
    # to euclidean

    points_0 = convert_points_from_homogeneous_3d(points_0_h)  # BxNxD
    # reshape to the input shape
    points_0_shape = paddle.shape(points_0)
    points_0_shape_len = len(list(points_0_shape))
    shape_inp[dim_num - 2] = points_0_shape[points_0_shape_len - 2]
    shape_inp[dim_num - 1] = points_0_shape[points_0_shape_len - 1]
    points_0 = points_0.reshape(shape_inp)
    return points_0


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    This function is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/transforms.py#L245

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), axis=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return paddle.concat(bbox_new, axis=-1)


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    This function is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/transforms.py#L259

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), axis=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return paddle.concat(bbox_new, axis=-1)
