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

from paddle3d.utils.logger import logger


class QAT:
    def __init__(self, quant_config, print_model=False):
        self.quant_config = quant_config
        self.print_model = print_model

    def __call__(self, model):
        try:
            import paddleslim
        except:
            raise ImportError("paddleslim module not found")

        if self.print_model:
            logger.info("model before quant")
            logger.info(model)

        self.quanter = paddleslim.QAT(config=self.quant_config)
        self.quanter.quantize(model)

        if self.print_model:
            logger.info("model after quant")
            logger.info(model)

        return model

    def save_quantized_model(self, model, path, input_spec, **kwargs):
        self.quanter.save_quantized_model(
            model=model, path=path, input_spec=input_spec, **kwargs)
