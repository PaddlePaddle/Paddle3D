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

import paddle

from pprndr.data.datasets import BaseDataset
from pprndr.utils.logger import logger

__all__ = ["PrefetchDataLoader"]


class PrefetchDataLoader(object):
    """
    Dataloader that prefetches images to buffer, then rays are sampled from the buffer.

    Args:
        dataset: Dataset object.
        batch_sampler: BatchSampler object.
        batch_size: Batch size. If batch_sampler is not None, batch_size will be ignored.
            If -1, the whole dataset is used.
        resampling_interval: Number of iterations between resampling the buffer.
            If -1, the buffer is not resampled.
    """

    def __init__(self,
                 dataset: BaseDataset,
                 batch_sampler: paddle.io.BatchSampler = None,
                 batch_size: int = -1,
                 resampling_interval: int = -1,
                 **kwargs):
        if batch_sampler is not None:
            self.loader = paddle.io.DataLoader(
                dataset, batch_sampler=batch_sampler, **kwargs)
            self.prefetch_all_images = len(batch_sampler) == 1
        elif batch_size == -1 or batch_size >= len(dataset):
            self.loader = paddle.io.DataLoader(
                dataset, batch_size=len(dataset), **kwargs)
            self.prefetch_all_images = True
        else:
            self.loader = paddle.io.DataLoader(
                dataset, batch_size=batch_size, **kwargs)
            self.prefetch_all_images = False

        self.iter_loader = iter(self.loader)

        self.resampling_interval = resampling_interval
        self.resampling_countdown = 0

        with logger.processing("Pouring images into buffer on GPU"):
            self.buffer = self._get_batch()

    def _get_batch(self):
        try:
            batch = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            batch = next(self.iter_loader)
        return batch

    def __iter__(self):
        while True:
            if self.resampling_interval > 0 and self.resampling_countdown == 0:
                # refresh buffer every 'resampling_interval' iters
                batch = self._get_batch()
                self.buffer = batch
                self.resampling_countdown = self.resampling_interval - 1
            elif self.resampling_interval > 0 and self.resampling_countdown > 0:
                batch = self.buffer
                self.resampling_countdown -= 1
            else:
                batch = self.buffer
            yield batch
