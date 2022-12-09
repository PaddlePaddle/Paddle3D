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
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers import constant_init, kaiming_normal_init
from paddle3d.ops import assign_score_withk
from paddle3d.utils.logger import logger

from .score_net import ScoreNet


@manager.MODELS.add_component
class PAConv(nn.Layer):
    def __init__(self,
                 k_neighbors=20,
                 calc_scores='softmax',
                 num_matrices=(8, 8, 8, 8),
                 dropout=0.5):
        super(PAConv, self).__init__()
        if calc_scores not in ['softmax', 'sigmoid']:
            raise ValueError(
                "Unsupported calc scores type {}".format(calc_scores))
        self.k = k_neighbors
        self.calc_scores = calc_scores
        self.assign_score_withk = assign_score_withk.assign_score_withk

        self.m1, self.m2, self.m3, self.m4 = num_matrices
        self.scorenet1 = ScoreNet(6, self.m1, hidden_unit=[16])
        self.scorenet2 = ScoreNet(6, self.m2, hidden_unit=[16])
        self.scorenet3 = ScoreNet(6, self.m3, hidden_unit=[16])
        self.scorenet4 = ScoreNet(6, self.m4, hidden_unit=[16])

        i1 = 3  # channel dim of input_1st
        o1 = i2 = 64  # channel dim of output_1st and input_2nd
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 128  # channel dim of output_3rd and input_4th
        o4 = 256  # channel dim of output_4th

        params = paddle.zeros(shape=[self.m1, i1 * 2, o1], dtype='float32')
        kaiming_normal_init(params, nonlinearity='relu')
        params = paddle.transpose(params,
                                  [1, 0, 2]).reshape([i1 * 2, self.m1 * o1])
        matrice1 = paddle.create_parameter(
            shape=[i1 * 2, self.m1 * o1],
            dtype='float32',
            default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice1', matrice1)

        params = paddle.zeros(shape=[self.m2, i2 * 2, o2], dtype='float32')
        kaiming_normal_init(params, nonlinearity='relu')
        params = paddle.transpose(params,
                                  [1, 0, 2]).reshape([i2 * 2, self.m2 * o2])
        matrice2 = paddle.create_parameter(
            shape=[i2 * 2, self.m2 * o2],
            dtype='float32',
            default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice2', matrice2)

        params = paddle.create_parameter(
            shape=[self.m3, i3 * 2, o3], dtype='float32')
        kaiming_normal_init(params, nonlinearity='relu')
        params = paddle.transpose(params,
                                  [1, 0, 2]).reshape([i3 * 2, self.m3 * o3])
        matrice3 = paddle.create_parameter(
            shape=[i3 * 2, self.m3 * o3],
            dtype='float32',
            default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice3', matrice3)

        params = paddle.create_parameter(
            shape=[self.m4, i4 * 2, o4], dtype='float32')
        kaiming_normal_init(params, nonlinearity='relu')
        params = paddle.transpose(params,
                                  [1, 0, 2]).reshape([i4 * 2, self.m4 * o4])
        matrice4 = paddle.create_parameter(
            shape=[i4 * 2, self.m4 * o4],
            dtype='float32',
            default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice4', matrice4)

        self.bn1 = nn.BatchNorm1D(o1)
        self.bn2 = nn.BatchNorm1D(o2)
        self.bn3 = nn.BatchNorm1D(o3)
        self.bn4 = nn.BatchNorm1D(o4)
        self.bn5 = nn.BatchNorm1D(1024)
        self.conv5 = nn.Sequential(
            nn.Conv1D(512, 1024, kernel_size=1, bias_attr=False), self.bn5)

        self.linear1 = nn.Linear(2048, 512, bias_attr=False)
        self.bn11 = nn.BatchNorm1D(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias_attr=False)
        self.bn22 = nn.BatchNorm1D(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, 40)

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, paddle.nn.Linear):
            kaiming_normal_init(m.weight, reverse=True)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, paddle.nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, paddle.nn.Conv1D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, paddle.nn.BatchNorm2D):
            constant_init(m.weight, value=1)
            constant_init(m.bias, value=0)
        elif isinstance(m, paddle.nn.BatchNorm1D):
            constant_init(m.weight, value=1)
            constant_init(m.bias, value=0)

    def knn(self, x, k):
        B, _, N = x.shape
        inner = -2 * paddle.matmul(x.transpose([0, 2, 1]), x)
        xx = paddle.sum(x**2, axis=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose([0, 2, 1])

        _, idx = pairwise_distance.topk(
            k=k, axis=-1)  # (batch_size, num_points, k)

        return idx, pairwise_distance

    def get_scorenet_input(self, x, idx, k):
        """(neighbor, neighbor-center)"""
        batch_size = x.shape[0]
        num_points = x.shape[2]
        x = x.reshape([batch_size, -1, num_points])

        idx_base = paddle.arange(0, batch_size).reshape([-1, 1, 1]) * num_points

        idx = idx + idx_base

        idx = idx.reshape([-1])

        _, num_dims, _ = x.shape

        x = paddle.transpose(x, [0, 2, 1])

        neighbor = x.reshape([batch_size * num_points, -1])
        neighbor = paddle.gather(neighbor, idx, axis=0)
        neighbor = neighbor.reshape([batch_size, num_points, k, num_dims])

        x = x.reshape([batch_size, num_points, 1, num_dims]).tile([1, 1, k, 1])

        xyz = paddle.concat((neighbor - x, neighbor),
                            axis=3).transpose([0, 3, 1, 2])  # b,6,n,k

        return xyz

    def feat_trans_dgcnn(self, point_input, kernel, m):
        """transforming features using weight matrices"""
        B, _, N = point_input.shape  # b, 2cin, n
        point_output = paddle.matmul(
            point_input.transpose([0, 2, 1]).tile([1, 1, 2]),
            kernel).reshape([B, N, m, -1])  # b,n,m,cout
        center_output = paddle.matmul(
            point_input.transpose([0, 2, 1]),
            kernel[:point_input.shape[1]]).reshape([B, N, m, -1])  # b,n,m,cout
        return point_output, center_output

    def get_loss(self, pred, label):

        label = paddle.reshape(
            label, [-1])  # gold is the groundtruth label in the dataloader

        eps = 0.2
        n_class = pred.shape[
            1]  # the number of feature_dim of the output, which is output channels
        one_hot = F.one_hot(label, n_class)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, axis=1)

        loss = -(one_hot * log_prb).sum(axis=1).mean()

        losses = {"loss": loss}
        return losses

    def forward(self, inputs):
        x = inputs['data']
        label = None
        if 'labels' in inputs.keys():
            label = inputs['labels']
        x = paddle.transpose(x, [0, 2, 1])
        B, C, N = x.shape
        idx, _ = self.knn(
            x, k=self.k
        )  # different with DGCNN, the knn search is only in 3D space
        xyz = self.get_scorenet_input(
            x, idx=idx, k=self.k
        )  # ScoreNet input: 3D coord difference concat with coord: b,6,n,k

        # replace all the DGCNN-EdgeConv with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        point1, center1 = self.feat_trans_dgcnn(
            point_input=x, kernel=self.matrice1, m=self.m1)  # b,n,m1,o1
        score1 = self.scorenet1(
            xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        """assemble with scores:"""
        point1 = self.assign_score_withk(
            scores=score1, points=point1, centers=center1,
            knn_idx=idx)  # b,o1,n
        point1 = F.relu(self.bn1(point1))

        point2, center2 = self.feat_trans_dgcnn(
            point_input=point1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(
            xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        point2 = self.assign_score_withk(
            scores=score2, points=point2, centers=center2, knn_idx=idx)
        point2 = F.relu(self.bn2(point2))

        point3, center3 = self.feat_trans_dgcnn(
            point_input=point2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(
            xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        point3 = self.assign_score_withk(
            scores=score3, points=point3, centers=center3, knn_idx=idx)
        point3 = F.relu(self.bn3(point3))

        point4, center4 = self.feat_trans_dgcnn(
            point_input=point3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(
            xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        point4 = self.assign_score_withk(
            scores=score4, points=point4, centers=center4, knn_idx=idx)
        point4 = F.relu(self.bn4(point4))

        point = paddle.concat((point1, point2, point3, point4), axis=1)
        point = F.relu(self.conv5(point))
        point11 = F.adaptive_max_pool1d(point, 1).reshape([B, -1])
        point22 = F.adaptive_avg_pool1d(point, 1).reshape([B, -1])
        point = paddle.concat((point11, point22), 1)

        point = F.relu(self.bn11(self.linear1(point)))
        point = self.dp1(point)
        point = F.relu(self.bn22(self.linear2(point)))
        point = self.dp2(point)
        point = self.linear3(point)

        if self.training:
            loss = self.get_loss(point, label)
            return loss
        else:
            if not getattr(self, "in_export_mode", False):
                return {'preds': point}
            else:
                return F.softmax(point, axis=-1)

    def export(self, save_dir: str, input_shape=(1, 1024, 3), **kwargs):
        self.in_export_mode = True
        save_path = os.path.join(save_dir, 'paconv')

        paddle.jit.to_static(
            self,
            input_spec=[{
                'data':
                paddle.static.InputSpec(shape=input_shape, dtype='float32')
            }])
        paddle.jit.save(self, save_path)

        logger.info("Exported model is saved in {}".format(save_dir))
