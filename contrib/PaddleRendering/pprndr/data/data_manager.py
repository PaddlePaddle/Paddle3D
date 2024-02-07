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

from typing import Tuple

import paddle
import paddle.nn as nn

import pprndr.utils.env as env
from pprndr.cameras import RayBundle
from pprndr.data.dataloaders import PrefetchDataLoader
from pprndr.data.datasets import BaseDataset
from pprndr.data.pixel_sampler import PixelSampler
from pprndr.models.ray_generators import RayGenerator


class DataManager(nn.Layer):
    def __init__(self,
                 dataset: BaseDataset,
                 image_batch_size: int = -1,
                 image_resampling_interval: int = -1,
                 ray_batch_size: int = 1,
                 use_adaptive_ray_batch_size: bool = False,
                 target_sample_batch_size: int = 1 << 18,
                 num_workers: int = 0,
                 shuffle: bool = False,
                 drop_last: bool = False):
        super(DataManager, self).__init__()

        self.image_batch_size = image_batch_size if image_batch_size > 0 else len(
            dataset) // env.nranks
        self.ray_batch_size = ray_batch_size
        self.use_adaptive_ray_batch_size = use_adaptive_ray_batch_size
        self.target_sample_batch_size = target_sample_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        image_batch_sampler = paddle.io.DistributedBatchSampler(
            dataset,
            batch_size=self.image_batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)
        self.image_data_loader = PrefetchDataLoader(
            dataset=dataset,
            batch_sampler=image_batch_sampler,
            resampling_interval=image_resampling_interval,
            num_workers=num_workers)
        self.iter_image_dataloader = iter(self.image_data_loader)

        self.pixel_sampler = PixelSampler(ray_batch_size)

        self.ray_generator = RayGenerator(dataset.cameras.cuda(),
                                          dataset.image_coords_offset)

    def forward(self):
        """
        DataManager may contain trainable parameters, thus it is a nn.Layer.
        But it is not a model, so its forward method will never be called.
        """
        raise NotImplementedError

    def next_sample(self) -> Tuple[RayBundle, dict]:
        image_batch = next(self.iter_image_dataloader)
        pixel_batch = self.pixel_sampler.sample(image_batch)
        ray_bundle = self.ray_generator(
            camera_ids=pixel_batch["camera_ids"],
            pixel_indices=pixel_batch["pixel_indices"])
        return ray_bundle, pixel_batch

    def next(self):
        return self.__next__()

    def __next__(self) -> Tuple[RayBundle, dict]:
        if hasattr(self, "amp_cfg_"):
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self.next_sample()
        else:
            return self.next_sample()

    def __iter__(self):
        return self

    def update_ray_batch_size(self, num_samples_per_batch: int):
        self.ray_batch_size = int(
            self.ray_batch_size *
            (self.target_sample_batch_size / float(num_samples_per_batch)))
        self.pixel_sampler.ray_batch_size = self.ray_batch_size
