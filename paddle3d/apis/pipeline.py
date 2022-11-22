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

from paddle3d.sample import Sample


def training_step(model: paddle.nn.Layer, optimizer: paddle.optimizer.Optimizer,
                  scaler: paddle.amp.GradScaler, amp_level: str, sample: Sample,
                  cur_iter: int) -> dict:

    if optimizer.__class__.__name__ == 'OneCycleAdam':
        optimizer.before_iter(cur_iter - 1)

    model.train()
    if scaler:
        if not isinstance(optimizer, paddle.optimizer.Optimizer):
            raise TypeError(
                "Optimizer should inherit from paddle.optimizer.Optimizer")
        with paddle.amp.auto_cast(
                custom_black_list=['matmul_v2', 'elementwise_mul'],
                level=amp_level):
            outputs = model(sample)
            loss = outputs['loss']
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        model.clear_gradients()
    else:
        outputs = model(sample)
        loss = outputs['loss']
        loss.backward()

        if optimizer.__class__.__name__ == 'OneCycleAdam':
            optimizer.after_iter()
        else:
            optimizer.step()
            model.clear_gradients()
    if isinstance(optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
        optimizer._learning_rate.step()

    return loss


def validation_step(model: paddle.nn.Layer, sample: Sample) -> dict:
    model.eval()
    with paddle.no_grad():
        outputs = model(sample)
    return outputs['preds']
