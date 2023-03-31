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

import abc

import paddle

from pprndr.apis import manager
from pprndr.metrics.functionals import ssim as ssim_meter
from pprndr.utils.logger import logger

__all__ = ["MetricABC", "PSNRMeter", "SSIMMeter"]


class MetricABC(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the meter.
        """

    @abc.abstractmethod
    def update(self, predictions: paddle.Tensor,
               ground_truths: paddle.Tensor) -> None:
        """
        Update the meter.
        """

    @abc.abstractmethod
    def accumulate(self, verbose: bool = False) -> paddle.Tensor:
        """
        Accumulate the result.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the metric.
        """


@manager.METRICS.add_component
class PSNRMeter(MetricABC):
    def __init__(self):
        super(PSNRMeter, self).__init__()

        self._name = 'psnr'
        self.reset()

    def reset(self):
        self.psnr = paddle.to_tensor([0.0])
        self.count = 0

    def _compute(self, predictions, ground_truths):
        psnr = -10. * paddle.log10(
            paddle.mean((predictions - ground_truths)**2))

        return psnr

    @paddle.no_grad()
    def update(self, predictions: paddle.Tensor, ground_truths: paddle.Tensor):
        psnr = self._compute(predictions, ground_truths)
        self.psnr += psnr
        self.count += 1

    @paddle.no_grad()
    def accumulate(self, verbose: bool = False) -> paddle.Tensor:
        result = (self.psnr / float(self.count))
        if verbose:
            logger.info("{}: {:.4f}".format(self.name, result.item()))

        return result

    @property
    def name(self) -> str:
        return self._name


@manager.METRICS.add_component
class SSIMMeter(MetricABC):
    def __init__(self):
        super(SSIMMeter, self).__init__()

        self._name = 'ssim'
        self.reset()

    def reset(self):
        self.ssim = paddle.to_tensor([0.0])
        self.count = 0

    def _compute(self, predictions, ground_truths):
        ssim = ssim_meter(predictions, ground_truths, data_range=1.0)
        return ssim

    @paddle.no_grad()
    def update(self, predictions: paddle.Tensor, ground_truths: paddle.Tensor):
        predictions = predictions.transpose([0, 3, 1, 2])
        ground_truths = ground_truths.transpose([0, 3, 1, 2])

        ssim = self._compute(predictions, ground_truths)
        self.ssim += ssim
        self.count += 1

    @paddle.no_grad()
    def accumulate(self, verbose: bool = False) -> paddle.Tensor:
        result = (self.ssim / float(self.count))
        if verbose:
            logger.info("{}: {:.4f}".format(self.name, result.item()))

        return result

    @property
    def name(self) -> str:
        return self._name
