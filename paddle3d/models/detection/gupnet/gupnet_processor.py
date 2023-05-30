# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
from paddle3d.apis import manager
from paddle3d.models.detection.gupnet.gupnet_helper import _nms, _topk, _transpose_and_gather_feat

num_heading_bin = 12


@manager.MODELS.add_component
class GUPNETPostProcessor(nn.Layer):
    def __init__(self, cls_mean_size, threshold):
        super().__init__()
        self.cls_mean_size = cls_mean_size
        self.threshold = threshold

    def forward(self, ret, info, calibs_p2):
        # prediction result convert
        predictions = self.extract_dets_from_outputs(ret, K=50)
        predictions = predictions.detach().cpu().numpy()
        calibs_p2 = calibs_p2.detach().cpu().numpy()
        # get corresponding calibs & transform tensor to numpy
        info = {key: val.detach().cpu().numpy() for key, val in info.items()}
        predictions = self.decode_detections(
            dets=predictions,
            info=info,
            calibs_p2=calibs_p2,
            cls_mean_size=self.cls_mean_size,
            threshold=self.threshold)

        return predictions

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)),
            axis=1)
        return pts_rect

    # GUPNET relative function
    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)

        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

        return ry

    def decode_detections(self, dets, info, calibs_p2, cls_mean_size,
                          threshold):
        '''NOTE: THIS IS A NUMPY FUNCTION
        input: dets, numpy array, shape in [batch x max_dets x dim]
        input: img_info, dict, necessary information of input images
        input: calibs, corresponding calibs for the input batch
        output:
        '''
        results = {}
        for i in range(dets.shape[0]):  # batch
            preds = []
            for j in range(dets.shape[1]):  # max_dets
                # encoder calib

                self.cu = calibs_p2[i][0, 2]
                self.cv = calibs_p2[i][1, 2]
                self.fu = calibs_p2[i][0, 0]
                self.fv = calibs_p2[i][1, 1]
                self.tx = calibs_p2[i][0, 3] / (-self.fu)
                self.ty = calibs_p2[i][1, 3] / (-self.fv)

                cls_id = int(dets[i, j, 0])
                score = dets[i, j, 1]
                if score < threshold:
                    continue

                # 2d bboxs decoding
                x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
                y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
                w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
                h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
                bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

                # 3d bboxs decoding
                # depth decoding
                depth = dets[i, j, 6]

                # heading angle decoding
                alpha = get_heading_angle(dets[i, j, 7:31])
                ry = self.alpha2ry(alpha, x)
                # ry = calibs[i].alpha2ry(alpha, x)

                # dimensions decoding
                dimensions = dets[i, j, 31:34]
                dimensions += cls_mean_size[int(cls_id)]
                if True in (dimensions < 0.0):
                    continue

                # positions decoding
                x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
                y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
                locations = self.img_to_rect(x3d, y3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2

                preds.append([cls_id, alpha] + bbox + dimensions.tolist() +
                             locations.tolist() + [ry, score])
            results[info['img_id'][i]] = preds
        return results

    # two stage style
    def extract_dets_from_outputs(self, outputs, K=50):
        # get src outputs
        heatmap = outputs['heatmap']
        size_2d = outputs['size_2d']
        offset_2d = outputs['offset_2d']

        batch, channel, height, width = heatmap.shape  # get shape

        heading = outputs['heading'].reshape((batch, K, -1))
        depth = outputs['depth'].reshape((batch, K, -1))[:, :, 0:1]
        size_3d = outputs['size_3d'].reshape((batch, K, -1))
        offset_3d = outputs['offset_3d'].reshape((batch, K, -1))

        heatmap = paddle.clip(
            paddle.nn.functional.sigmoid(heatmap), min=1e-4, max=1 - 1e-4)

        # perform nms on heatmaps
        heatmap = _nms(heatmap)
        scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

        offset_2d = _transpose_and_gather_feat(offset_2d, inds)
        offset_2d = offset_2d.reshape((batch, K, 2))
        xs2d = xs.reshape((batch, K, 1)) + offset_2d[:, :, 0:1]
        ys2d = ys.reshape((batch, K, 1)) + offset_2d[:, :, 1:2]

        xs3d = xs.reshape((batch, K, 1)) + offset_3d[:, :, 0:1]
        ys3d = ys.reshape((batch, K, 1)) + offset_3d[:, :, 1:2]

        cls_ids = cls_ids.reshape((batch, K, 1)).astype('float32')
        depth_score = (-(0.5 * outputs['depth'].reshape(
            (batch, K, -1))[:, :, 1:2]).exp()).exp()
        scores = scores.reshape((batch, K, 1)) * depth_score

        # check shape
        xs2d = xs2d.reshape((batch, K, 1))
        ys2d = ys2d.reshape((batch, K, 1))
        xs3d = xs3d.reshape((batch, K, 1))
        ys3d = ys3d.reshape((batch, K, 1))

        size_2d = _transpose_and_gather_feat(size_2d, inds)
        size_2d = size_2d.reshape((batch, K, 2))

        detections = paddle.concat([
            cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d,
            ys3d
        ],
                                   axis=2)

        return detections


def class2angle(cls, residual, to_label_format=False):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)
