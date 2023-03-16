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
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.utils.box import denormalize_bbox, normalize_bbox


class ResidualCoder(object):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_coder_utils.py#L5
    """

    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_paddle(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        anchors[:, 3:6] = paddle.clip(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = paddle.clip(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = paddle.split(anchors, 7, axis=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = paddle.split(boxes, 7, axis=-1)

        diagonal = paddle.sqrt(dxa**2 + dya**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = paddle.log(dxg / dxa)
        dyt = paddle.log(dyg / dya)
        dzt = paddle.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = paddle.cos(rg) - paddle.cos(ra)
            rt_sin = paddle.sin(rg) - paddle.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return paddle.concat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], axis=-1)

    def decode_paddle(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = paddle.split(anchors, 7, axis=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = paddle.split(
                box_encodings, 7, axis=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = paddle.split(
                box_encodings, 7, axis=-1)

        diagonal = paddle.sqrt(dxa**2 + dya**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = paddle.exp(dxt) * dxa
        dyg = paddle.exp(dyt) * dya
        dzg = paddle.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + paddle.cos(ra)
            rg_sin = sint + paddle.sin(ra)
            rg = paddle.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return paddle.concat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], axis=-1)


@manager.BBOX_CODERS.add_component
class NMSFreeCoder(object):
    """Bbox coder for NMS-free detector.

    This class is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py

    Args:
        point_cloud_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 point_cloud_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = F.sigmoid(cls_scores)
        scores, indexs = cls_scores.reshape([-1]).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.point_cloud_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = paddle.to_tensor(self.post_center_range)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.shape[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


@manager.BBOX_CODERS.add_component
class DeltaXYZWLHRBBoxCoder(paddle.nn.Layer):
    """Bbox Coder for 3D boxes.
    This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/core/bbox/coders/delta_xyzwhlr_bbox_coder.py#L8

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, code_size=7):
        super(DeltaXYZWLHRBBoxCoder, self).__init__()
        self.code_size = code_size

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, dw, dh, dl,
        dr, dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            src_boxes (paddle.Tensor): source boxes, e.g., object proposals.
            dst_boxes (paddle.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            paddle.Tensor: Box transformation deltas.
        """
        box_ndim = src_boxes.shape[-1]
        cas, cgs, cts = [], [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = paddle.split(
                src_boxes, box_ndim, axis=-1)
            xg, yg, zg, wg, lg, hg, rg, *cgs = paddle.split(
                dst_boxes, box_ndim, axis=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, wa, la, ha, ra = paddle.split(
                src_boxes, box_ndim, axis=-1)
            xg, yg, zg, wg, lg, hg, rg = paddle.split(
                dst_boxes, box_ndim, axis=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = paddle.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = paddle.log(lg / la)
        wt = paddle.log(wg / wa)
        ht = paddle.log(hg / ha)
        rt = rg - ra
        return paddle.concat([xt, yt, zt, wt, lt, ht, rt, *cts], axis=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (paddle.Tensor): Parameters of anchors with shape (N, 7).
            deltas (paddle.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            paddle.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = paddle.split(
                anchors, box_ndim, axis=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = paddle.split(
                deltas, box_ndim, axis=-1)
        else:
            xa, ya, za, wa, la, ha, ra = paddle.split(
                anchors, box_ndim, axis=-1)
            xt, yt, zt, wt, lt, ht, rt = paddle.split(deltas, box_ndim, axis=-1)

        za = za + ha / 2
        diagonal = paddle.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = paddle.exp(lt) * la
        wg = paddle.exp(wt) * wa
        hg = paddle.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return paddle.concat([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)
