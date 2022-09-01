# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import time
from os import path as osp

import h5py
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import inference
from paddle.inference import Config, create_predictor

from paddle3d.ops.assign_score_withk import assign_score_withk


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("Paddle3D Inference model script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/example.yaml',
        help='config file path')
    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [file for file in files if (file.endswith(".h5"))]
        files = [osp.join(input_path, file) for file in files]
    return files


def postprocess(input_file, output, print_output=True):
    """
    output: list
    """
    top_k = 1

    output = output[0]  # [B, num_cls]

    N = len(input_file[0])
    if output.shape[0] != N:
        output = output.reshape(
            [N] + [output.shape[0] // N] + list(output.shape[1:]))  # [N, T, C]
        output = output.mean(axis=1)  # [N, C]

    for i in range(N):
        classes = np.argpartition(output[i], -top_k)[-top_k:]
        classes = classes[np.argsort(-output[i, classes])]
        scores = output[i, classes]
        if print_output:
            for j in range(top_k):
                print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                print("\ttop-{0} score: {1}".format(j + 1, scores[j]))


def main():
    args = parse_args()

    model_name = 'PAConv'
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    files = parse_file_paths(args.input_file)

    if args.enable_benchmark:
        num_warmup = 3

        # instantiate auto log
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name="PAConv",
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=0 if args.use_gpu else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)
        test_num = 10
        files = [args.input_file for _ in range(test_num)]

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # auto log start
        if args.enable_benchmark:
            autolog.times.start()
        data_list = []
        for i in range(st_idx, ed_idx):
            f = h5py.File(files[i], mode='r')
            data = f['data'][:].astype('float32')
            labels = f['label'][:].astype('int64')
            f.close()
            data = data[:, :1024, :]
            data_list.append(data)

        data_list = np.concatenate(data_list)

        # Pre process batched input
        batched_inputs = [data_list]

        # get pre process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # run inference
        for i in range(len(input_tensor_list)):
            input_tensor_list[i].copy_from_cpu(batched_inputs[i])
        predictor.run()

        batched_outputs = []
        for j in range(len(output_tensor_list)):
            batched_outputs.append(output_tensor_list[j].copy_to_cpu())

        # get inference process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        postprocess([labels], batched_outputs, not args.enable_benchmark)

        # get post process time cost
        if args.enable_benchmark:
            autolog.times.end(stamp=True)

    # report benchmark log if enabled
    if args.enable_benchmark:
        autolog.report()


if __name__ == "__main__":
    main()
