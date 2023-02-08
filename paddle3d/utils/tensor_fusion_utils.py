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

from collections import OrderedDict

import numpy as np
import paddle
from paddle.fluid.framework import EagerParamBase
from paddle.framework import core
from paddle import _legacy_C_ops

from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_storage import (
    GradStorage, ParamStorage)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import Type

alignment = {
    "gpu": 256,
}
align = {
    Type.fp16.value: 2,
    Type.fp32.value: 4,
}


def assign_group_by_size(parameters, group_size=256 * 1024 * 1024):
    is_sparse_gradient = [False] * len(parameters)
    group_indices = core.eager_assign_group_by_size(
        parameters, is_sparse_gradient, [group_size, group_size])

    var_groups = OrderedDict()
    for group_idx, indices in enumerate(group_indices):
        for index in indices:
            var_groups.setdefault(group_idx, []).append(parameters[index])
    return var_groups


def flatten_dense_tensors(parameters, lr=1.0):
    is_trainable = parameters[0].trainable
    _buffer_size = 0
    _param2align = {}
    dtype = parameters[0].dtype

    for param in parameters:
        assert param.trainable == is_trainable, "all parameters in the list should have same trainable attribute"
        size = np.prod(param.shape) * align[dtype]
        remaining = size % alignment["gpu"]
        ali = 0 if remaining == 0 else alignment["gpu"] - remaining
        align_ = ali // align[dtype]
        _buffer_size += np.prod(param.shape) + align_
        _param2align[param.name] = align_

    param_storage = ParamStorage(size=_buffer_size, dtype=dtype, device="gpu")

    param_storage.add_rank_params(parameters, _param2align)

    if not is_trainable:
        return param_storage, None

    # process gradient
    grad_storage = GradStorage(
        size=_buffer_size,
        dtype=dtype,
        device="gpu",
        destination="0",
        parm2align=_param2align)

    for param in parameters:
        grad_storage.add_grad(param, _param2align[param.name])

    # for learning rate, create a new EagerParamBase
    eager_param = EagerParamBase(
        shape=[_buffer_size], dtype=dtype, optimize_attr={'learning_rate': lr})
    param_storage.buffer._share_buffer_to(eager_param)

    # param_storage --> grad_storage
    eager_param._copy_gradient_from(grad_storage.buffer)
    eager_param.stop_gradient = False

    return eager_param, grad_storage


def obtain_storage(parameters, lr=1.0):
    if len(parameters) < 1:
        return []

    var_groups = assign_group_by_size(parameters)
    storage = []
    for group_idx, parameters in var_groups.items():
        param_storage, grad_storage = flatten_dense_tensors(parameters, lr)
        if isinstance(param_storage, EagerParamBase):
            storage.append(param_storage)
        else:
            storage.append(param_storage.buffer)
    return storage


def fused_parameters(parameters, use_sharding=False):
    trainable_parameters = {}
    non_trainable_params = []

    for param in parameters:
        if not param.trainable:
            non_trainable_params.append(param)
        else:
            lr = param.optimize_attr['learning_rate']
            if lr is None:
                lr = 1.0
            if lr not in trainable_parameters:
                trainable_parameters[lr] = []
            trainable_parameters[lr].append(param)

    obtain_storage(non_trainable_params)

    all_fused = []
    for lr in trainable_parameters:
        tmp_fused = obtain_storage(trainable_parameters[lr], lr)
        all_fused += tmp_fused

    return all_fused


def all_reduce_parameters(params, group=None):
    if group is None or group.nranks < 2:
        return

    div_factor = 1.0 / group.nranks
    with paddle.framework.no_grad():
        for p in params:
            grad = p.grad.scale_(div_factor)
            paddle.distributed.all_reduce(grad, group=group)


def is_fused_matmul_bias_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        if hasattr(core, "ops"):
            return hasattr(core.ops, 'fused_gemm_epilogue')
        else:
            return hasattr(_legacy_C_ops, "fused_gemm_epilogue")
    else:
        return False
