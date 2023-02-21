# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import shutil

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from uapi import PaddleModel, Config

if __name__ == '__main__':
    config = Config('smoke')

    model = PaddleModel(config=config)

    if os.path.exists('uapi/tests/3d_res'):
        shutil.rmtree('uapi/tests/3d_res')

    model.train(
        dataset='uapi/tests/data/KITTI',
        batch_size=1,
        epochs_iters=10,
        device='gpu:0',
        amp='O1',
        save_dir='uapi/tests/3d_res')

    model.evaluate(
        weight_path='uapi/tests/3d_res/iter_10/model.pdparams',
        dataset='uapi/tests/data/KITTI')

    # `model.predict()` not implemented
    try:
        model.predict(
            weight_path='uapi/tests/3d_res/iter_10/model.pdparams',
            device='gpu',
            input_path='uapi/tests/data/KITTI/training/image_2/000004.png',
            save_dir='uapi/tests/3d_res/pred_res')
    except Exception as e:
        print(e)

    model.export(
        weight_path='uapi/tests/3d_res/iter_10/model.pdparams',
        save_dir='uapi/tests/3d_res/infer')

    model.infer(
        model_dir='uapi/tests/3d_res/infer',
        device='gpu',
        input_path='uapi/tests/data/KITTI/training/image_2/000004.png',
        save_dir='uapi/tests/3d_res/infer_res')

    model.compression(
        dataset='uapi/tests/data/KITTI',
        batch_size=2,
        learning_rate=0.1,
        epochs_iters=10,
        device='cpu',
        weight_path='uapi/tests/3d_res/iter_10/model.pdparams',
        save_dir='uapi/tests/3d_res/compress')
