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

__all__ = ["PointPillarsCoder"]


class PointPillarsCoder(object):
    @staticmethod
    def encode(boxes, anchors):
        return second_box_encode_paddle(boxes, anchors)

    @staticmethod
    def decode(encodings, anchors):
        return second_box_decode_paddle(encodings, anchors)

    @staticmethod
    def corners_2d(bboxes_3d):
        w, l = bboxes_3d[:, 3:5].t()
        b = bboxes_3d.shape[0]

        x_corners = paddle.to_tensor([[0., 0., 1., 1.]]).repeat_interleave(
            b, axis=0)
        y_corners = paddle.to_tensor([[0., 1., 1., 0.]]).repeat_interleave(
            b, axis=0)

        x_corners = (w[:, None] * (x_corners - .5))[:, :, None]
        y_corners = (l[:, None] * (y_corners - .5))[:, :, None]
        corners_2d = paddle.concat([x_corners, y_corners], axis=-1)

        angle = bboxes_3d[:, -1]
        rot_sin = paddle.sin(angle)
        rot_cos = paddle.cos(angle)
        rotation_matrix = paddle.to_tensor([[rot_cos, -rot_sin],
                                            [rot_sin, rot_cos]])

        corners_2d = paddle.einsum("aij,jka->aik", corners_2d, rotation_matrix)

        centers = bboxes_3d[:, 0:2][:, None, :]
        corners_2d += centers

        return corners_2d

    @staticmethod
    def corner_to_standup(corners):
        ndim = corners.shape[-1]
        standup_boxes = []
        for i in range(ndim):
            standup_boxes.append(paddle.min(corners[:, :, i], axis=1))
        for i in range(ndim):
            standup_boxes.append(paddle.max(corners[:, :, i], axis=1))
        return paddle.stack(standup_boxes, axis=1)

    @staticmethod
    def corners_3d(bboxes_3d, origin=(.5, .5, .5)):
        # corners_3d format: x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0
        h, w, l = bboxes_3d[:, 3:6].t()
        b = h.shape[0]

        x_corners = paddle.to_tensor([[0., 0., 0., 0., 1., 1., 1., 1.]],
                                     bboxes_3d.dtype).repeat_interleave(
                                         b, axis=0)
        y_corners = paddle.to_tensor([[0., 0., 1., 1., 0., 0., 1., 1.]],
                                     bboxes_3d.dtype).repeat_interleave(
                                         b, axis=0)
        z_corners = paddle.to_tensor([[0., 1., 1., 0., 0., 1., 1., 0.]],
                                     bboxes_3d.dtype).repeat_interleave(
                                         b, axis=0)

        x_corners = (w[:, None] * (x_corners - origin[0]))[:, :, None]
        y_corners = (l[:, None] * (y_corners - origin[1]))[:, :, None]
        z_corners = (h[:, None] * (z_corners - origin[2]))[:, :, None]
        corners = paddle.concat([x_corners, y_corners, z_corners], axis=-1)

        angle = bboxes_3d[:, 6:7].squeeze(-1)
        rot_sin = paddle.sin(angle)
        rot_cos = paddle.cos(angle)
        ones = paddle.ones_like(rot_cos)
        zeros = paddle.zeros_like(rot_cos)
        rotation_matrix = paddle.to_tensor(
            [[rot_cos, -rot_sin, zeros], [rot_sin, rot_cos, zeros],
             [zeros, zeros, ones]],
            dtype=bboxes_3d.dtype)

        corners = paddle.einsum("aij,jka->aik", corners, rotation_matrix)
        centers = bboxes_3d[:, 0:3][:, None, :]
        corners += centers

        return corners


def second_box_encode_paddle(boxes, anchors):
    """
    Encode 3D bboxes for VoxelNet/PointPillars.
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = paddle.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = paddle.split(boxes, 7, axis=-1)

    diagonal = paddle.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    lt = paddle.log(lg / la)
    wt = paddle.log(wg / wa)
    ht = paddle.log(hg / ha)

    rt = rg - ra
    return paddle.concat([xt, yt, zt, wt, lt, ht, rt], axis=-1)


def second_box_decode_paddle(encodings, anchors):
    """
    Decode 3D bboxes for VoxelNet/PointPillars.
    Args:
        encodings ([N, 7] Tensor): encoded boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = paddle.split(anchors, 7, axis=-1)
    xt, yt, zt, wt, lt, ht, rt = paddle.split(encodings, 7, axis=-1)

    diagonal = paddle.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = paddle.exp(lt) * la
    wg = paddle.exp(wt) * wa
    hg = paddle.exp(ht) * ha

    rg = rt + ra

    return paddle.concat([xg, yg, zg, wg, lg, hg, rg], axis=-1)
