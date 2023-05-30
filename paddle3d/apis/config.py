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

import codecs
import os
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Generic, Optional

import paddle
import yaml

from paddle3d.utils.logger import logger
from paddle3d.utils.tensor_fusion_utils import (fused_parameters,
                                                is_fused_matmul_bias_supported)


class Config(object):
    '''Training configuration parsing. Only yaml/yml files are supported.

    The following hyper-parameters are available in the config file:
        batch_size: The number of samples per gpu.
        iters: The total training steps.
        epochs: The total training epochs.
        train_dataset: A training data config including type/data_root/transforms/mode.
            For data type, please refer to paddle3d.datasets.
            For specific transforms, please refer to paddle3d.transforms.transforms.
        val_dataset: A validation data config including type/data_root/transforms/mode.
        optimizer: A optimizer config, but currently paddle3d only supports sgd with momentum in config file.
            In addition, weight_decay could be set as a regularization.
        learning_rate: A learning rate config. If decay is configured, learning _rate value is the starting learning rate,
             where only poly decay is supported using the config file. In addition, decay power and end_lr are tuned experimentally.
        model: A model config including type/backbone and model-dependent arguments.
            For model type, please refer to paddle3d.models.
            For backbone, please refer to paddle3d.models.backbones.

    Args:
        path (str) : The path of config file, supports yaml format only.

    Examples:
        from paddle3d.apis.config import Config
        # Create a cfg object with yaml file path.
        cfg = Config(yaml_cfg_path)
        # Parsing the argument when its property is used.
        train_dataset = cfg.train_dataset
        # the argument of model should be parsed after dataset,
        # since the model builder uses some properties in dataset.
        model = cfg.model
        ...
    '''

    def __init__(self,
                 *,
                 path: str,
                 learning_rate: Optional[float] = None,
                 batch_size: Optional[int] = None,
                 iters: Optional[int] = None,
                 epochs: Optional[int] = None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        self._model = None
        self._train_dataset = None
        self._val_dataset = None
        self._all_fused_tensors = None
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate,
            batch_size=batch_size,
            iters=iters,
            epochs=epochs)

    def _update_dic(self, dic: Dict, base_dic: Dict):
        '''Update config from dic based base_dic
        '''

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
               batch_size: Optional[int] = None,
               iters: Optional[int] = None,
               epochs: Optional[int] = None):
        '''Update config'''

        if learning_rate is not None:
            self.dic['lr_scheduler']['learning_rate'] = learning_rate

        if batch_size is not None:
            self.dic['batch_size'] = batch_size

        if iters is not None:
            self.dic['iters'] = iters

        if epochs is not None:
            self.dic['epochs'] = epochs

    @property
    def batch_size(self) -> int:
        return self.dic.get('batch_size', 1)

    @property
    def iters(self) -> int:
        iters = self.dic.get('iters')
        return iters

    @property
    def epochs(self) -> int:
        epochs = self.dic.get('epochs')
        return epochs

    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if 'lr_scheduler' not in self.dic:
            raise RuntimeError(
                'No `lr_scheduler` specified in the configuration file.')

        params = self.dic.get('lr_scheduler')
        return self._load_object(params)

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        params = self.dic.get('optimizer', {}).copy()

        params['learning_rate'] = self.lr_scheduler
        set_params = False
        if 'tensor_fusion' in params:
            tensor_fusion = params.pop('tensor_fusion')
            if tensor_fusion:
                self._all_fused_tensors = fused_parameters(
                    self.model.parameters())
                params['parameters'] = self._all_fused_tensors
                optimizer = self._load_object(params)
                setattr(optimizer, 'all_fused_tensors', self._all_fused_tensors)
                set_params = True

        if not set_params:
            params['parameters'] = filter(lambda p: p.trainable,
                                          self.model.parameters())
            optimizer = self._load_object(params)

        return optimizer

    @property
    def all_fused_tensors(self):
        return self._all_fused_tensors

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')

        if 'fused_linear' in model_cfg:
            fused_linear = model_cfg.pop('fused_linear')
            if fused_linear:
                if not is_fused_matmul_bias_supported():
                    logger.warning(
                        "The flag fused_linear only valid for cuda version higher than 11.6, "
                        "but the paddle is compiled with cuda " +
                        paddle.version.cuda())
                else:
                    paddle.nn.functional.linear = paddle.incubate.nn.functional.fused_linear

        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    @property
    def amp_config(self) -> int:
        return self.dic.get('amp_cfg', None)

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
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        if not self._train_dataset:
            self._train_dataset = self._load_object(_train_dataset)
        return self._train_dataset

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        if not self._val_dataset:
            self._val_dataset = self._load_object(_val_dataset)
        return self._val_dataset

    @property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    # @property
    # def use_ema(self) -> bool:
    #     return self.dic.get('use_ema', False)

    @property
    def temporal_start_epoch(self) -> int:
        return self.dic.get('temporal_start_epoch', -1)

    def _load_component(self, com_name: str) -> Any:
        # lazy import
        import paddle3d.apis.manager as manager

        if com_name.lower().startswith('$paddleseg'):
            return self._load_component_from_paddleseg(com_name[11:])

        if com_name.lower().startswith('$paddledet'):
            return self._load_component_from_paddledet(com_name[11:])

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

    def _load_component_from_paddleseg(self, com_name: str) -> Any:
        from paddleseg.cvlibs import manager
        com_list = [
            manager.BACKBONES, manager.DATASETS, manager.MODELS,
            manager.TRANSFORMS, manager.LOSSES
        ]

        for com in com_list:
            if com_name in com.components_dict:
                return com[com_name]

        raise RuntimeError(
            'The specified component was not found {} in paddleseg.'.format(
                com_name))

    def _load_component_from_paddledet(self, com_name: str) -> Any:
        from ppdet.core.workspace import global_config as ppdet_com_dict

        if com_name in ppdet_com_dict:
            component = ppdet_com_dict[com_name]
            cls = getattr(component.pymodule, component.name)
            return cls

        raise RuntimeError(
            'The specified component was not found {} in paddledet.'.format(
                com_name))

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
            try:
                return component(**params)
            except Exception as e:
                raise type(e)('{} {}'.format(component.__name__, e))

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
        if self.iters is not None:
            dic = {'iters': self.iters}
        else:
            dic = {'epochs': self.epochs}

        dic.update({
            'optimizer': self.optimizer,
            'model': self.model,
            'train_dataset': self.train_dataset,
            'val_dataset': self.val_dataset,
            'batch_size': self.batch_size,
            'amp_cfg': self.amp_config,
            # 'use_ema': self.use_ema,
            'temporal_start_epoch': self.temporal_start_epoch
        })

        return dic
