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
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import \
    fused_allreduce_gradients

from paddle3d.sample import Sample
from paddle3d.utils.tensor_fusion_utils import all_reduce_parameters


def parse_losses(losses):
    total_loss = 0
    if isinstance(losses, paddle.Tensor):
        total_loss += losses
    elif isinstance(losses, dict):
        for k, v in losses.items():
            total_loss += v
    return total_loss


def training_step(model: paddle.nn.Layer,
                  optimizer: paddle.optimizer.Optimizer,
                  sample: Sample,
                  cur_iter: int,
                  scaler=None,
                  amp_cfg=dict(),
                  all_fused_tensors=None,
                  group=None) -> dict:

    if optimizer.__class__.__name__ == 'OneCycleAdam':
        optimizer.before_iter(cur_iter - 1)

    model.train()

    if isinstance(model, paddle.DataParallel) and hasattr(model._layers, 'use_recompute') \
        and model._layers.use_recompute:
        with model.no_sync():
            if scaler is not None:
                with paddle.amp.auto_cast(**amp_cfg):
                    outputs = model(sample)
                    loss = parse_losses(outputs['loss'])
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
            else:
                outputs = model(sample)
                loss = parse_losses(outputs['loss'])
                loss.backward()
        if all_fused_tensors is None:
            fused_allreduce_gradients(list(model.parameters()), None)
        else:
            assert group is not None
            all_reduce_parameters(all_fused_tensors, group)
    else:
        if scaler is not None:
            with paddle.amp.auto_cast(**amp_cfg):
                outputs = model(sample)
                loss = parse_losses(outputs['loss'])
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
        else:
            outputs = model(sample)
            loss = parse_losses(outputs['loss'])
            loss.backward()

    if optimizer.__class__.__name__ == 'OneCycleAdam':
        optimizer.after_iter()
    else:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.clear_grad()

        if isinstance(optimizer._learning_rate,
                      paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()

    outputs['total_loss'] = loss
    return outputs


def validation_step(model: paddle.nn.Layer, sample: Sample) -> dict:
    model.eval()
    with paddle.no_grad():
        outputs = model(sample)
    return outputs['preds']
