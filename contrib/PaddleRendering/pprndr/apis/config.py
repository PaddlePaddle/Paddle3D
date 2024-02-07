# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import codecs
import os
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Generic, List, Optional, Union

import paddle
import yaml

from pprndr.data import BaseDataset
from pprndr.metrics import MetricABC

__all__ = ["Config"]


class Config(object):
    """Training configuration parsing. Only yaml/yml files are supported.

    The following hyperparameters are available in the config file:
        image_batch_size: The number of images from which rays are sampled.
        ray_batch_size: The number of rays per GPU.
        image_resampling_interval: How often to resample the image.
        iters: The total training steps.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to pprndr.datasets.
            For specific transforms, please refer to pprndr.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, but currently pprndr only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        learning_rate: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,
             where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to pprndr.models.

    Args:
        path (str) : The path of config file, supports yaml format only.

    Examples:
        from pprndr.apis.config import Config
        # Create a cfg object with yaml file path.
        cfg = Config(yaml_cfg_path)
        # Parsing the argument when its property is used.
        train_dataset = cfg.train_dataset
        # the argument of model should be parsed after dataset,
        # since the model builder uses some properties in dataset.
        model = cfg.model
        ...
    """

    def __init__(self,
                 *,
                 path: str,
                 learning_rate: Optional[float] = None,
                 image_batch_size: Optional[int] = None,
                 ray_batch_size: Optional[int] = None,
                 image_resampling_interval: Optional[int] = None,
                 use_adaptive_ray_batch_size: Optional[bool] = None,
                 iters: Optional[int] = None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        self._model = None
        self._train_dataset = None
        self._val_dataset = None
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate,
            image_batch_size=image_batch_size,
            ray_batch_size=ray_batch_size,
            image_resampling_interval=image_resampling_interval,
            use_adaptive_ray_batch_size=use_adaptive_ray_batch_size,
            iters=iters)

    def _update_dic(self, dic: Dict, base_dic: Dict):
        """Update config from dic based base_dic
        """

        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build config'''

        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(self,
               learning_rate: Optional[float] = None,
               image_batch_size: Optional[int] = None,
               image_resampling_interval: Optional[int] = None,
               ray_batch_size: Optional[int] = None,
               use_adaptive_ray_batch_size: Optional[bool] = None,
               iters: Optional[int] = None):
        """Update config"""

        if learning_rate is not None:
            if 'lr_scheduler' in self.dic:
                self.dic['lr_scheduler']['learning_rate'] = learning_rate
            else:
                self.dic['optimizer']['learning_rate'] = learning_rate

        if image_batch_size is not None:
            self.dic['image_batch_size'] = image_batch_size

        if ray_batch_size is not None:
            self.dic['ray_batch_size'] = ray_batch_size

        if image_resampling_interval is not None:
            self.dic['image_resampling_interval'] = image_resampling_interval

        if use_adaptive_ray_batch_size is not None:
            self.dic[
                'use_adaptive_ray_batch_size'] = use_adaptive_ray_batch_size

        if iters is not None:
            self.dic['iters'] = iters

    @property
    def image_batch_size(self) -> int:
        return self.dic.get('image_batch_size', -1)

    @property
    def ray_batch_size(self) -> int:
        return self.dic.get('ray_batch_size', 1)

    @property
    def eval_pixel_stride(self) -> int:
        return self.dic.get('eval_pixel_stride', 1)

    @property
    def use_adaptive_ray_batch_size(self) -> bool:
        return self.dic.get('use_adaptive_ray_batch_size', False)

    @property
    def image_resampling_interval(self) -> int:
        return self.dic.get('image_resampling_interval', -1)

    @property
    def iters(self) -> int:
        iters = self.dic.get('iters')
        return iters

    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if 'lr_scheduler' not in self.dic:
            return None

        params = self.dic.get('lr_scheduler')
        return self._load_object(params)

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        params = self.dic.get('optimizer', {}).copy()

        if 'lr_scheduler' in self.dic:
            params['learning_rate'] = self.lr_scheduler
        elif 'learning_rate' not in params:
            raise RuntimeError(
                "Neither a `lr_scheduler` nor optimizer's `learning_rate` is specified in the configuration file."
            )

        params['parameters'] = self.model.parameters()

        return self._load_object(params)

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')

        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    @property
    def train_dataset_config(self) -> Dict:
        return self.dic.get('train_dataset', {}).copy()

    @property
    def val_dataset_config(self) -> Dict:
        return self.dic.get('val_dataset', {}).copy()

    @property
    def train_dataset_class(self) -> Generic:
        dataset_type = self.train_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def val_dataset_class(self) -> Generic:
        dataset_type = self.val_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def train_dataset(self) -> Union[BaseDataset, None]:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        if not self._train_dataset:
            self._train_dataset = self._load_object(_train_dataset)
        return self._train_dataset

    @property
    def val_dataset(self) -> Union[BaseDataset, None]:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        if not self._val_dataset:
            self._val_dataset = self._load_object(_val_dataset)
        return self._val_dataset

    @property
    def amp_config(self) -> Union[Dict, None]:
        return self.dic.get('amp_cfg', None)

    @property
    def grad_accum_config(self) -> Union[Dict, None]:
        return self.dic.get('grad_accum_cfg', None)

    @property
    def reinit_optim_config(self) -> Union[Dict, None]:
        return self.dic.get('reinit_optim_cfg', None)

    @property
    def train_metric_meters(self) -> Union[List[MetricABC], None]:
        metrics_cfg = self.dic.get('train_metrics', None)
        if metrics_cfg is None:
            return None
        metric_meters = self._load_object(metrics_cfg.copy())
        if not isinstance(metric_meters, list):
            metric_meters = [metric_meters]
        return metric_meters

    @property
    def val_metric_meters(self) -> Union[List[MetricABC], None]:
        metrics_cfg = self.dic.get('val_metrics', None)
        if metrics_cfg is None:
            return None
        metric_meters = self._load_object(metrics_cfg.copy())
        if not isinstance(metric_meters, list):
            metric_meters = [metric_meters]
        return metric_meters

    def _load_component(self, com_name: str) -> Any:
        # lazy import
        import pprndr.apis.manager as manager

        for com in manager.__all__:
            com = getattr(manager, com)
            if com_name in com.components_dict:
                return com[com_name]
        else:
            if com_name in paddle.optimizer.lr.__all__:
                return getattr(paddle.optimizer.lr, com_name)
            elif com_name in paddle.optimizer.__all__:
                return getattr(paddle.optimizer, com_name)
            elif com_name in paddle.nn.__all__:
                return getattr(paddle.nn, com_name)

            raise RuntimeError(
                'The specified component was not found {}.'.format(com_name))

    def _load_object(self, obj: Generic, recursive: bool = True) -> Any:
        if isinstance(obj, Mapping):
            dic = obj.copy()
            component = self._load_component(
                dic.pop('type')) if 'type' in dic else dict

            if recursive:
                params = {}
                for key, val in dic.items():
                    params[key] = self._load_object(
                        obj=val, recursive=recursive)
            else:
                params = dic

            return component(**params)

        elif isinstance(obj, Iterable) and not isinstance(obj, str):
            return [self._load_object(item) for item in obj]

        return obj

    def _is_meta_type(self, item: Any) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        msg = '---------------Config Information---------------'
        msg += '\n{}'.format(yaml.dump(self.dic))
        msg += '------------------------------------------------'
        return msg

    def to_dict(self) -> Dict:
        dic = {'iters': self.iters}

        dic.update({
            'optimizer': self.optimizer,
            'model': self.model,
            'train_dataset': self.train_dataset,
            'val_dataset': self.val_dataset,
            'train_metric_meters': self.train_metric_meters,
            'val_metric_meters': self.val_metric_meters,
            'image_batch_size': self.image_batch_size,
            'eval_pixel_stride': self.eval_pixel_stride,
            'ray_batch_size': self.ray_batch_size,
            'use_adaptive_ray_batch_size': self.use_adaptive_ray_batch_size,
            'image_resampling_interval': self.image_resampling_interval,
            'amp_cfg': self.amp_config,
            'grad_accum_cfg': self.grad_accum_config,
            'reinit_optim_cfg': self.reinit_optim_config
        })

        return dic
