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

import time


class Timer:
    """
    """

    def __init__(self, iters: int = 0, momentum: float = 0.5):
        self.iters = iters
        self.cur_iter = 0
        self.elasped_time = 0
        self.last_time = None
        self._moving_speed = None
        self.momentum = momentum
        self.total_samples = 0

    def step(self, num_samples=None):
        """
        """
        self.cur_iter += 1
        now = time.time()

        if self.last_time is not None:
            iter_speed = now - self.last_time

            if self._moving_speed is None:
                self._moving_speed = iter_speed
            else:
                self._moving_speed = self._moving_speed * self.momentum + (
                    1 - self.momentum) * iter_speed

            self.elasped_time += iter_speed
            if num_samples:
                self.total_samples += num_samples

        self.last_time = now

    @property
    def speed(self):
        """
        """
        if self.cur_iter == 0:
            return 0

        return self.elasped_time / self.cur_iter

    @property
    def ips(self):
        if not self.total_samples or self.cur_iter == 0:
            return 0
        return float(self.total_samples) / self.elasped_time

    @property
    def eta(self):
        """
        """
        if self.iters == 0 or self._moving_speed is None:
            return "--:--:--"

        remaining_iter = max(self.iters - self.cur_iter, 0)
        remaining_time = int(remaining_iter * self._moving_speed)
        result = "{:0>2}:{:0>2}:{:0>2}"
        arr = []

        for i in range(2, -1, -1):
            arr.append(int(remaining_time / 60**i))
            remaining_time %= 60**i

        return result.format(*arr)
