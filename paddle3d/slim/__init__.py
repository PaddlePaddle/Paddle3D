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


def get_qat_config(qat_config_path: str) -> dict:
    with codecs.open(qat_config_path, 'r', 'utf-8') as f:
        slim_dic = yaml.load(f, Loader=yaml.FullLoader)

    slim_type = slim_dic['slim_type']
    if slim_type != "QAT":
        raise ValueError("slim method `{}` is not supported yet")

    return slim_dic


def update_dic(dic, another_dic):
    """Recursive update dic by another_dic
    """
    for k, _ in another_dic.items():
        if (k in dic and isinstance(dic[k], dict)) and isinstance(
                another_dic[k], dict):
            update_dic(dic[k], another_dic[k])
        else:
            dic[k] = another_dic[k]
    return dic
