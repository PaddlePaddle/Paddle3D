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

import argparse

import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

from paddle3d import transforms as T
from paddle3d.sample import Sample
from paddle3d.transforms.normalize import NormalizeRangeImage
from paddle3d.transforms.reader import LoadSemanticKITTIRange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        "--params_file",
        type=str,
        help=
        "Parameter filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        '--lidar_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        '--img_mean',
        type=str,
        help='The mean value of range-view image.',
        required=True)
    parser.add_argument(
        '--img_std',
        type=str,
        help='The variance value of range-view image.',
        required=True)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        type=int,
        default=0,
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
    parser.add_argument(
        "--trt_use_static",
        type=int,
        default=0,
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")

    return parser.parse_args()


def preprocess(file_path, img_mean, img_std):
    if isinstance(img_mean, str):
        img_mean = eval(img_mean)
    if isinstance(img_std, str):
        img_std = eval(img_std)

    sample = Sample(path=file_path, modality="lidar")

    transforms = T.Compose([
        LoadSemanticKITTIRange(project_label=False),
        NormalizeRangeImage(mean=img_mean, std=img_std)
    ])

    sample = transforms(sample)

    if "proj_mask" in sample.meta:
        sample.data *= sample.meta.pop("proj_mask")
    return np.expand_dims(sample.data,
                          0), sample.meta.proj_x, sample.meta.proj_y


def init_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, gpu_id)
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)

    predictor = create_predictor(config)
    return predictor


def run(predictor, points):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(points.shape)
    input_tensor.copy_from_cpu(points.copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    pred_label = output_tensor.copy_to_cpu()

    return pred_label[0]


def postprocess(pred_img_label, proj_x, proj_y):
    return pred_img_label[proj_y, proj_x]


def main(args):
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir)
    range_img, proj_x, proj_y = preprocess(args.lidar_file, args.img_mean,
                                           args.img_std)
    pred_img_label = run(predictor, range_img)
    pred_point_label = postprocess(pred_img_label, proj_x, proj_y)
    return pred_point_label


if __name__ == '__main__':
    args = parse_args()

    main(args)
