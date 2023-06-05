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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.models.detection.gupnet.gupnet_helper import _transpose_and_gather_feat
from paddle3d.apis import manager


class Hierarchical_Task_Learning:
    def __init__(self, epoch0_loss, stat_epoch_nums=5, max_epoch=140):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {
            term: self.index2term.index(term)
            for term in self.index2term
        }  # term2index
        # self.term2index: {
        # 'seg_loss': 0, 'offset2d_loss': 1, 'size2d_loss': 2,
        # 'depth_loss': 3, 'offset3d_loss': 4, 'size3d_loss': 5, 'heading_loss': 6}
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses = []
        self.loss_graph = {
            'seg_loss': [],
            'size2d_loss': [],
            'offset2d_loss': [],
            'offset3d_loss': ['size2d_loss', 'offset2d_loss'],
            'size3d_loss': ['size2d_loss', 'offset2d_loss'],
            'heading_loss': ['size2d_loss', 'offset2d_loss'],
            'depth_loss': ['size2d_loss', 'size3d_loss', 'offset2d_loss']
        }
        self.max_epoch = max_epoch

    def compute_weight(self, current_loss, epoch):
        # compute initial weights
        loss_weights = {}
        eval_loss_input = paddle.concat(
            [_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term]) == 0:
                loss_weights[term] = paddle.to_tensor(1.0)
            else:
                loss_weights[term] = paddle.to_tensor(0.0)

        if len(self.past_losses) == self.stat_epoch_nums:
            past_loss = paddle.concat(self.past_losses)
            mean_diff = (past_loss[:-2] - past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1 - \
                paddle.nn.functional.relu_(
                    mean_diff / self.init_diff).unsqueeze(0)
            time_value = min(((epoch - 5) / (self.max_epoch - 5)), 1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic]) != 0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][
                            self.term2index[pre_topic]]
                    loss_weights[current_topic] = time_value**(
                        1 - control_weight)
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)
        return loss_weights

    def update_e0(self, eval_loss):
        self.epoch0_loss = paddle.concat(
            [_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


@manager.LOSSES.add_component
class GUPNETLoss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.stat = {}

    def forward(self, preds, targets):
        '''
        Args:
            preds: prediction {dict 9}
                'heatmap', 'offset_2d', 'size_2d', 'train_tag',
                'heading', 'depth', 'offset_3d', 'size_3d', 'h3d_log_variance'

            targets: ground truth {dict 11}
            'depth', 'size_2d', 'heatmap', 'offset_2d', 'indices',
            'size_3d', 'offset_3d', 'heading_bin', 'heading_res', 'cls_ids', 'mask_2d'
        '''
        self.stat['seg_loss'] = self.compute_segmentation_loss(preds, targets)
        self.stat['offset2d_loss'], self.stat[
            'size2d_loss'] = self.compute_bbox2d_loss(preds, targets)
        self.stat['depth_loss'], self.stat['offset3d_loss'], self.stat[
            'size3d_loss'], self.stat[
                'heading_loss'] = self.compute_bbox3d_loss(preds, targets)
        return self.stat

    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = paddle.clip(
            paddle.nn.functional.sigmoid(input['heatmap']),
            min=1e-4,
            max=1 - 1e-4)
        loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
        return loss

    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        size2d_input = extract_input_from_tensor(
            input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'],
                                                   target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(
            input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'],
                                                     target['mask_2d'])
        offset2d_loss = F.l1_loss(
            offset2d_input, offset2d_target, reduction='mean')

        return offset2d_loss, size2d_loss

    def compute_bbox3d_loss(self, input, target, mask_type='mask_2d'):
        # compute depth loss
        depth_input = input['depth'][input['train_tag']]
        depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:
                                                                           2]
        depth_target = extract_target_from_tensor(target['depth'],
                                                  target[mask_type])
        depth_loss = laplacian_aleatoric_uncertainty_loss(
            depth_input, depth_target, depth_log_variance)

        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]
        offset3d_target = extract_target_from_tensor(target['offset_3d'],
                                                     target[mask_type])
        offset3d_loss = F.l1_loss(
            offset3d_input, offset3d_target, reduction='mean')

        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']]
        size3d_target = extract_target_from_tensor(target['size_3d'],
                                                   target[mask_type])
        size3d_loss = F.l1_loss(size3d_input[:, 1:], size3d_target[:, 1:], reduction='mean') * 2 / 3 + \
            laplacian_aleatoric_uncertainty_loss(size3d_input[:, 0:1], size3d_target[:, 0:1],
                                                 input['h3d_log_variance'][input['train_tag']]) / 3
        heading_loss = compute_heading_loss(
            input['heading'][input['train_tag']],
            target[mask_type],  # mask_2d
            target['heading_bin'],
            target['heading_res'])

        return depth_loss, offset3d_loss, size3d_loss, heading_loss


# ======================  auxiliary functions  =======================


def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask.astype(paddle.bool)]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask.astype(paddle.bool)]


# compute heading loss two stage style


def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.reshape((1, -1)).squeeze(0)  # B * K  ---> (B*K)
    target_cls = target_cls.reshape((1, -1)).squeeze(0)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.reshape((1, -1)).squeeze(0)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask.astype(paddle.bool)]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask.astype(paddle.bool)]
    cls_onehot = paddle.put_along_axis(
        arr=paddle.zeros([target_cls.shape[0], 12]),
        axis=1,
        indices=target_cls.reshape((-1, 1)),
        values=paddle.to_tensor(1).astype('float32'))
    input_reg = paddle.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')

    return cls_loss + reg_loss


def laplacian_aleatoric_uncertainty_loss(input,
                                         target,
                                         log_variance,
                                         reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * paddle.exp(-0.5 * log_variance) * \
        paddle.abs(input - target) + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def focal_loss_cornernet(input, target, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    '''
    pos_inds = paddle.equal(target, 1).astype('float32')
    neg_inds = paddle.less_than(target,
                                paddle.ones(target.shape)).astype('float32')
    # pos_inds = target.eq(1).float()
    # neg_inds = target.lt(1).float()
    neg_weights = paddle.pow(1 - target, 4)

    loss = 0
    pos_loss = paddle.log(input) * paddle.pow(1 - input, gamma) * pos_inds
    neg_loss = paddle.log(1 - input) * paddle.pow(
        input, gamma) * neg_inds * neg_weights

    num_pos = pos_inds.sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()
