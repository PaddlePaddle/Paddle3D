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
from collections import namedtuple
from typing import Optional

SchedulerStatus = namedtuple('SchedulerStatus',
                             ['do_eval', 'do_log', 'save_checkpoint'])


class SchedulerABC(abc.ABC):
    """
    """

    @abc.abstractmethod
    def step(self, cur_iter: Optional[int] = None) -> SchedulerStatus:
        """
        """


class Scheduler(SchedulerABC):
    """
    """

    def __init__(self,
                 save_interval: int,
                 log_interval: int,
                 do_eval: bool = False):
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.do_eval = do_eval
        self.cur_iter = 0

    def step(self, cur_iter: Optional[int] = None) -> SchedulerStatus:
        """
        """
        if cur_iter is None:
            self.cur_iter += 1
        else:
            self.cur_iter = cur_iter

        save_checkpoint = self.save_interval != 0 and self.cur_iter % self.save_interval == 0

        do_eval = save_checkpoint and self.do_eval
        do_log = self.log_interval != 0 and self.cur_iter % self.log_interval == 0

        return SchedulerStatus(do_eval, do_log, save_checkpoint)
