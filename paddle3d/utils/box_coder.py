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


class ResidualCoder(object):
    """
    This function refers to https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_coder_utils.py#L5
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
