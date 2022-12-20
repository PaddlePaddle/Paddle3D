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
import yaml


def get_default_qat_config() -> dict:
    return {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'weight_bits': 8,
        'activation_bits': 8,
        'dtype': 'int8',
        'window_size': 10000,
        'moving_rate': 0.9,
        'quantizable_layer_type': ['Conv2D', 'Linear']
    }


def get_qat_config(qat_config_path: str) -> dict:
    with codecs.open(qat_config_path, 'r', 'utf-8') as f:
        slim_dic = yaml.load(f, Loader=yaml.FullLoader)

    slim_type = slim_dic['slim_type']
    if slim_type == "QAT":
        quant_config = slim_dic["slim_config"]['quant_config']
    else:
        raise ValueError("slim method `{}` is not supported yet")

    return quant_config
