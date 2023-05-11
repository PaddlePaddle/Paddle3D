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
from paddle.jit import to_static

from paddle3d.sample import Sample
from paddle3d.utils.logger import logger
from paddle3d.utils.tensor_fusion_utils import all_reduce_parameters


def parse_losses(losses):
    """
    Parse the loss tensor in dictionary into a single scalar.
    """
    log_loss = dict()
    if isinstance(losses, paddle.Tensor):
        total_loss = losses
    elif isinstance(losses, dict):
        for loss_name, loss_value in losses.items():
            log_loss[loss_name] = paddle.sum(loss_value)
        total_loss = sum(
            _loss_value for _loss_name, _loss_value in log_loss.items())

    log_loss['total_loss'] = total_loss

    return total_loss, log_loss


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
                    loss, log_loss = parse_losses(outputs['loss'])
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
            else:
                outputs = model(sample)
                loss, log_loss = parse_losses(outputs['loss'])
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
                loss, log_loss = parse_losses(outputs['loss'])
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
        else:
            outputs = model(sample)
            loss, log_loss = parse_losses(outputs['loss'])
            loss.backward()

    # update params
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

    # reduce loss when distributed training
    if paddle.distributed.is_initialized():
        with paddle.no_grad():
            for loss_name, loss_value in log_loss.items():
                loss_clone = loss_value.clone()
                paddle.distributed.all_reduce(
                    loss_clone.scale_(1. / paddle.distributed.get_world_size()))
                log_loss[loss_name] = loss_clone.item()

    return log_loss


def validation_step(model: paddle.nn.Layer, sample: Sample) -> dict:
    model.eval()
    with paddle.no_grad():
        outputs = model(sample)
    return outputs['preds']


def apply_to_static(support_to_static, model, image_shape=None):
    if support_to_static:
        specs = None
        if image_shape is not None:
            specs = image_shape
        model = to_static(model, input_spec=specs)
        logger.info(
            "Successfully to apply @to_static with specs: {}".format(specs))
    return model
