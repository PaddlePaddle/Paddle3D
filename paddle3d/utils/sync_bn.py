import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.autograd import PyLayer

__all__ = ['NaiveSyncBatchNorm1D', 'NaiveSyncBatchNorm2D']


class AllReduce(PyLayer):
    @staticmethod
    def forward(ctx, input):
        input_list = [
            paddle.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input)
        inputs = paddle.stack(input_list, axis=0)
        return paddle.sum(inputs, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output)
        return grad_output


class NaiveSyncBatchNorm1D(nn.BatchNorm1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        assert input.dtype == paddle.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size(
        ) == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        dim_2 = input.dim() == 2
        if dim_2:
            input.unsqueeze_(2)

        C = input.shape[1]
        mean = paddle.mean(input, axis=[0, 2])
        meansqr = paddle.mean(input * input, axis=[0, 2])

        vec = paddle.concat([mean, meansqr], axis=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = paddle.split(vec, vec.shape[0] // C)
        var = meansqr - mean * mean
        self._mean.set_value(self._mean.value() + (1 - self._momentum) *
                             (mean.detach() - self._mean))
        self._variance.set_value(self._variance.value() + (1 - self._momentum) *
                                 (var.detach() - self._variance))

        invstd = paddle.rsqrt(var + self._epsilon)
        scale = self.weight.value() * invstd
        bias = self.bias.value() - mean * scale
        scale = scale.reshape([1, -1, 1])
        bias = bias.reshape([1, -1, 1])
        input = input * scale + bias
        if dim_2:
            input = input.squeeze(2)
        return input


class NaiveSyncBatchNorm2D(nn.BatchNorm2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        assert input.dtype == paddle.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not dist.is_initialized() or dist.get_world_size(
        ) == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = paddle.mean(input, axis=[0, 2, 3])
        meansqr = paddle.mean(input * input, axis=[0, 2, 3])

        vec = paddle.concat([mean, meansqr], axis=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = paddle.split(vec, vec.shape[0] // C)
        var = meansqr - mean * mean
        # self._mean += (1 - self.momentum) * (
        #     mean.detach() - self._mean)
        # self._variance += (1 - self.momentum) * (var.detach() - self._variance)
        self._mean.set_value(self._mean.value() + (1 - self._momentum) *
                             (mean.detach() - self._mean))
        self._variance.set_value(self._variance.value() + (1 - self._momentum) *
                                 (var.detach() - self._variance))

        invstd = paddle.rsqrt(var + self._epsilon)
        scale = self.weight.value() * invstd
        bias = self.bias.value() - mean * scale
        scale = scale.reshape([1, -1, 1, 1])
        bias = bias.reshape([1, -1, 1, 1])
        return input * scale + bias
