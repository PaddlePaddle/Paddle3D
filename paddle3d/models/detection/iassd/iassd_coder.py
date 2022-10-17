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

import numpy as np
import paddle

__all__ = ["PointResidual_BinOri_Coder"]


class PointResidual_BinOri_Coder(paddle.nn.Layer):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.bin_size = kwargs.get('bin_size', 12)
        self.code_size = 6 + 2 * self.bin_size
        self.bin_inter = 2 * np.pi / self.bin_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = paddle.to_tensor(
                kwargs['mean_size'], dtype='float32')
            assert self.mean_size.min() > 0

    def encode_paddle(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = paddle.clip(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg = paddle.split(gt_boxes, 7, axis=-1)
        xa, ya, za = paddle.split(points, 3, axis=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = paddle.split(point_anchor_size, 3, axis=-1)
            diagonal = paddle.sqrt(dxa**2 + dya**2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = paddle.log(dxg / dxa)
            dyt = paddle.log(dyg / dya)
            dzt = paddle.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = paddle.log(dxg)
            dyt = paddle.log(dyg)
            dzt = paddle.log(dzg)

        rg = paddle.clip(rg, max=np.pi - 1e-5, min=-np.pi + 1e-5)
        bin_id = paddle.floor((rg + np.pi) / self.bin_inter)

        bin_res = (
            (rg + np.pi) - (bin_id * self.bin_inter + self.bin_inter / 2)) / (
                self.bin_inter / 2)  # norm to [-1, 1]

        return paddle.concat([xt, yt, zt, dxt, dyt, dzt, bin_id, bin_res],
                             axis=-1)

    def decode_paddle(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, bin_id, bin_res , ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:
            decoded box predictions
        """
        xt, yt, zt, dxt, dyt, dzt = paddle.split(
            box_encodings[..., :6], 6, axis=-1)
        xa, ya, za = paddle.split(points, 3, axis=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = paddle.split(point_anchor_size, 3, axis=-1)
            diagonal = paddle.sqrt(dxa**2 + dya**2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = paddle.exp(dxt) * dxa
            dyg = paddle.exp(dyt) * dya
            dzg = paddle.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = paddle.split(
                paddle.exp(box_encodings[..., 3:6]), 3, axis=-1)

        bin_id = box_encodings[..., 6:6 + self.bin_size]
        bin_res = box_encodings[..., 6 + self.bin_size:]
        bin_id = paddle.argmax(bin_id, axis=-1)
        bin_id_one_hot = paddle.nn.functional.one_hot(
            bin_id.astype('int64'), self.bin_size)
        bin_res = paddle.sum(
            bin_res * bin_id_one_hot.astype('float32'), axis=-1)

        rg = bin_id.astype(
            'float32') * self.bin_inter - np.pi + self.bin_inter / 2
        rg = rg + bin_res * (self.bin_inter / 2)
        rg = rg.unsqueeze(-1)

        return paddle.concat([xg, yg, zg, dxg, dyg, dzg, rg], axis=-1)
