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

import abc
import contextlib
import copy
import os
import shutil
from typing import Generic, Hashable, Optional, Tuple

import filelock
import paddle
import yaml
from easydict import EasyDict

from paddle3d.utils.logger import logger


class CheckpointABC(abc.ABC):
    """
    """

    @abc.abstractmethod
    def have(self, tag: str):
        """
        """

    @abc.abstractmethod
    def get(self, tag: Optional[str] = None, **kwargs) -> Tuple[dict, dict]:
        """
        """

    @abc.abstractmethod
    def push(self, params_dict: dict, opt_dict: dict = None, **kwargs) -> str:
        """
        """

    def pop(self, **kwargs) -> str:
        """
        """

    @property
    @abc.abstractmethod
    def empty(self) -> bool:
        """
        """

    @abc.abstractmethod
    def record(self, key: Hashable, value: Generic) -> bool:
        """
        """

    @property
    @abc.abstractmethod
    def meta(self) -> dict:
        """
        """

    @property
    @abc.abstractmethod
    def metafile(self) -> str:
        """
        """

    @property
    @abc.abstractmethod
    def rootdir(self) -> str:
        """
        """


class Checkpoint(CheckpointABC):
    """
    """

    def __init__(self,
                 save_dir: str,
                 keep_checkpoint_max: int = 5,
                 overwrite: bool = True):
        self.save_dir = save_dir
        self._meta = EasyDict()

        self._meta.overwrite = overwrite
        self._meta.keep_checkpoint_max = keep_checkpoint_max
        self._meta.counter = 0

        self._meta.queue = []

        os.makedirs(self.save_dir, exist_ok=True)

        if os.path.exists(self.metafile):
            with open(self.metafile) as file, self.rwlock():
                dic = yaml.load(file, Loader=yaml.FullLoader)
                self._meta.update(dic)

        self._sync_to_file()

    def have(self, tag: str):
        """
        """
        return tag in self.meta.queue

    def get(self, tag: Optional[str] = None, ema=None,
            step=0) -> Tuple[dict, dict]:
        """
        """
        if tag is None:
            if len(self.meta.queue) == 0:
                raise RuntimeError('The checkpoint queue is empty!')
            tag = self.meta.queue[-1]

        if not self.have(tag):
            raise ValueError(
                'There is no model parameter corresponding to the specified tag  {{{}}} in checkpoint.'
                .format(tag))

        ema_state_dict = None
        if ema is not None:
            ema_dict_path = os.path.join(self.rootdir, tag,
                                         'model_ema.pdparams')
            if os.path.exists(ema_dict_path):
                ema_state_dict = paddle.load(ema_dict_path)

        params_path = os.path.join(self.rootdir, tag, 'model.pdparams')
        opt_path = os.path.join(self.rootdir, tag, 'model.pdopt')
        params = paddle.load(params_path)
        if os.path.exists(opt_path):
            opt = paddle.load(opt_path)
        else:
            opt = {}

        if ema_state_dict is not None:
            ema.resume(ema_state_dict, step=step)

        return params, opt

    def push(self,
             params_dict: dict,
             opt_dict: dict = None,
             tag: Optional[str] = None,
             enqueue: bool = True,
             verbose: bool = False,
             ema_model=None) -> str:
        """
        """
        tag = str(self._meta.counter) if tag is None else tag
        dirname = os.path.join(self.rootdir, tag)

        params_path = os.path.join(dirname, 'model.pdparams')

        if enqueue:
            if self._meta.keep_checkpoint_max > 0 and len(
                    self._meta.queue) >= self._meta.keep_checkpoint_max:
                self.pop(verbose=verbose)

            self._meta.queue.append(tag)
            self._meta.counter += 1
        else:
            if os.path.exists(params_path) and not self._meta.overwrite:
                raise RuntimeError(
                    'Unable to save parameters to non-empty path {}'.format(
                        params_path))

        os.makedirs(dirname, exist_ok=True)
        paddle.save(params_dict, params_path)

        if ema_model is not None:
            assert isinstance(ema_model,
                              dict), ("ema_model is not a instance of dict, "
                                      "please call model.state_dict() to get.")
            ema_params_path = os.path.join(dirname, 'model_ema.pdparams')
            paddle.save(ema_model, ema_params_path)

        if opt_dict is not None:
            opt_path = os.path.join(dirname, 'model.pdopt')
            paddle.save(opt_dict, opt_path)

        if verbose:
            logger.info('Push model to checkpoint {}'.format(dirname))

        self._sync_to_file()
        return tag

    def pop(self, verbose: bool = False) -> str:
        """
        """
        if len(self._meta.queue) == 0:
            raise RuntimeError('Checkpoint queue is empty!')

        pop_idx = self._meta.queue[0]
        pop_dir = os.path.join(self.rootdir, pop_idx)
        shutil.rmtree(pop_dir)

        if verbose:
            logger.info('Pop model from {}'.format(pop_dir))

        self._meta.queue = self._meta.queue[1:]
        self._sync_to_file()

        return pop_idx

    @property
    def empty(self):
        """
        """
        return len(self._meta.queue) == 0

    def record(self, key: Hashable, value: Generic) -> bool:
        """
        """
        if key in self._meta and not self._meta.overwrite:
            return False

        self._meta[key] = value
        self._sync_to_file()
        return True

    @property
    def meta(self) -> dict:
        """
        """
        return copy.deepcopy(self._meta)

    @property
    def metafile(self) -> str:
        """
        """
        return os.path.join(self.rootdir, 'meta.yaml')

    @property
    def rootdir(self) -> str:
        """
        """
        return self.save_dir

    def _sync_to_file(self):
        with open(self.metafile, 'w') as file, self.rwlock():
            yaml.dump(dict(self.meta), file)

    @contextlib.contextmanager
    def rwlock(self):
        lockfile = os.path.join(self.rootdir, '.lock')
        with filelock.FileLock(lockfile):
            yield
