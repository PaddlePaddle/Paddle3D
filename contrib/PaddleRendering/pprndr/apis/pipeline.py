#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from collections import defaultdict
from typing import Dict, Tuple, Union
import paddle
from paddle.distributed.fleet.utils.hybrid_parallel_util import \
    fused_allreduce_gradients

from pprndr.cameras import RayBundle


def parse_losses(losses: Union[Dict[str, paddle.Tensor], paddle.Tensor]
                 ) -> paddle.Tensor:
    if isinstance(losses, paddle.Tensor):
        return losses
    elif isinstance(losses, dict):
        return sum(losses.values())


def training_step(model: paddle.nn.Layer,
                  optimizer: paddle.optimizer.Optimizer,
                  cur_iter: int,
                  sample: Tuple[RayBundle, dict],
                  scaler=None,
                  grad_accum_cfg=None) -> dict:
    model.train()

    if isinstance(model, paddle.DataParallel) and getattr(
            model._layers, 'use_recompute', False):
        with model.no_sync():
            outputs = model(sample, cur_iter=cur_iter)
            loss = parse_losses(outputs['loss'])
            if scaler is not None:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
            else:
                loss.backward()
        fused_allreduce_gradients(list(model.parameters()), None)
    else:
        outputs = model(sample, cur_iter=cur_iter)
        loss = parse_losses(outputs['loss'])
        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()

    if grad_accum_cfg is None or (
            grad_accum_cfg is not None
            and cur_iter % grad_accum_cfg['accum_steps'] == 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.clear_grad()

        if isinstance(optimizer._learning_rate,
                      paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()

    with paddle.no_grad():
        if paddle.distributed.is_initialized():
            loss_clone = loss.clone()
            paddle.distributed.all_reduce(
                loss_clone.scale_(1. / paddle.distributed.get_world_size()))
            outputs['total_loss'] = loss_clone
        else:
            outputs['total_loss'] = loss
    return outputs


@paddle.no_grad()
def inference_step(model: paddle.nn.Layer,
                   ray_bundle: RayBundle,
                   ray_batch_size: int,
                   to_cpu: bool = False) -> dict:
    outputs_all = defaultdict(list)

    model.eval()
    num_rays = len(ray_bundle)
    for b_id in range(0, num_rays, ray_batch_size):
        cur_ray_bundle = ray_bundle[b_id:b_id + ray_batch_size]
        outputs = model(cur_ray_bundle)
        for k, v in outputs.items():
            if isinstance(v, paddle.Tensor) and to_cpu:
                v = v.cpu()
            outputs_all[k].append(v)
        del outputs

    outputs = {}
    for k, v in outputs_all.items():
        if isinstance(v[0], paddle.Tensor):
            outputs[k] = paddle.concat(v, axis=0)
        else:
            outputs[k] = v

    return outputs
