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
from paddle import nn

from .smooth_l1_loss import smooth_l1_loss
from paddle3d.utils import transform3d as t3d
from paddle3d.utils.transform import quaternion_to_matrix

BOX3D_CORNER_MAPPING = [[1, 1, 1, 1, -1, -1, -1, -1],
                        [1, -1, -1, 1, 1, -1, -1, 1],
                        [1, 1, -1, -1, 1, 1, -1, -1]]


def homogenize_points(xy):
    """
    Args:
        xy (paddle.Tensor): xy coordinates, shape=(N, ..., 2)
            E.g., (N, 2) or (N, K, 2) or (N, H, W, 2)

    Returns:
        paddle.Tensor: appended to the last dimension. shape=(N, ..., 3)
            E.g, (N, 3) or (N, K, 3) or (N, H, W, 3).
    """
    # NOTE: this seems to work for arbitrary number of dimensions of input
    pad = nn.Pad1D(padding=[0, 1], mode='constant', value=1.0)
    return pad(xy.unsqueeze(0)).squeeze(0)


def unproject_points2d(points2d, inv_K, scale=1.0):
    """
    Args:
        points2d (paddle.Tensor): xy coordinates. shape=(N, ..., 2)
            E.g., (N, 2) or (N, K, 2) or (N, H, W, 2)
        inv_K (paddle.Tensor): Inverted intrinsics; shape=(N, 3, 3)
        scale (float): Scaling factor, default: 1.0s

    Returns:
        paddle.Tensor: Unprojected 3D point. shape=(N, ..., 3)
            E.g., (N, 3) or (N, K, 3) or (N, H, W, 3)
    """
    points2d = homogenize_points(points2d)
    siz = points2d.shape
    points2d = points2d.reshape([-1, 3]).unsqueeze(-1)  # (N, 3, 1)
    unprojected = paddle.matmul(inv_K,
                                points2d)  # (N, 3, 3) x (N, 3, 1) -> (N, 3, 1)
    unprojected = unprojected.reshape(siz)

    return unprojected * scale


class DisentangledBox3DLoss(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/da25b614a29344830c96c2848c02a15b35380c4b/tridet/modeling/dd3d/disentangled_box3d_loss.py#L13
    """

    def __init__(self, smooth_l1_loss_beta=0.05, max_loss_per_group=20.0):
        super(DisentangledBox3DLoss, self).__init__()
        self.smooth_l1_loss_beta = smooth_l1_loss_beta
        self.max_loss_per_group = max_loss_per_group

    def forward(self,
                box3d_pred,
                box3d_targets,
                locations,
                inv_intrinsics,
                weights=None):

        box3d_pred = box3d_pred.cast('float32')
        box3d_targets = box3d_targets.cast('float32')

        target_corners = self.corners(
            box3d_targets[..., :4], box3d_targets[..., 4:6],
            box3d_targets[..., 6:7], box3d_targets[..., 7:], inv_intrinsics)

        disentangled_losses = {}
        index = [0, 4, 6, 7, 10]
        for i, component_key in enumerate(["quat", "proj_ctr", "depth",
                                           "size"]):
            disentangled_boxes = box3d_targets.clone()
            disentangled_boxes[..., index[i]:index[
                i + 1]] = box3d_pred[..., index[i]:index[i + 1]]
            pred_corners = self.corners(
                disentangled_boxes[..., :4], disentangled_boxes[..., 4:6],
                disentangled_boxes[..., 6:7], disentangled_boxes[..., 7:],
                inv_intrinsics)

            loss = smooth_l1_loss(
                pred_corners, target_corners, beta=self.smooth_l1_loss_beta)
            n = paddle.abs(pred_corners - target_corners)

            # Bound the loss
            loss = loss.clip(max=self.max_loss_per_group)

            if weights is not None:
                loss = paddle.sum(loss.reshape([-1, 24]).mean(axis=1) * weights)
            else:
                loss = loss.reshape([-1, 24]).mean()

            disentangled_losses["loss_box3d_" + component_key] = loss

        pred_corners = self.corners(box3d_pred[..., :4], box3d_pred[..., 4:6],
                                    box3d_pred[..., 6:7], box3d_pred[..., 7:],
                                    inv_intrinsics)
        entangled_l1_dist = (
            target_corners - pred_corners).detach().abs().reshape(
                [-1, 24]).mean(axis=1)

        return disentangled_losses, entangled_l1_dist

    def corners(self, quat, proj_ctr, depth, size, inv_intrinsics):
        ray = unproject_points2d(proj_ctr, inv_intrinsics)
        tvec = ray * depth
        translation = t3d.Translate(tvec)

        R = quaternion_to_matrix(quat)
        rotation = t3d.Rotate(R=R.transpose(
            [0, 2, 1]))  # Need to transpose to make it work.

        tfm = rotation.compose(translation)

        _corners = 0.5 * paddle.to_tensor(BOX3D_CORNER_MAPPING).T
        lwh = paddle.stack([size[:, 1], size[:, 0], size[:, 2]],
                           -1)  # wlh -> lwh
        corners_in_obj_frame = lwh.unsqueeze(1) * _corners.unsqueeze(0)

        corners3d = tfm.transform_points(corners_in_obj_frame)
        return corners3d
