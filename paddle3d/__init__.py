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

__version__ = "0.5.0"

import paddle
from packaging.version import Version

paddle_version = Version(paddle.__version__)
minimum_paddle_version = Version("2.4.0")
develop_version = Version("0.0.0")

if paddle_version < minimum_paddle_version and paddle_version != develop_version:
    raise RuntimeError("Please upgrade PaddlePaddle version to {}".format(
        minimum_paddle_version))

from . import datasets, models, transforms
