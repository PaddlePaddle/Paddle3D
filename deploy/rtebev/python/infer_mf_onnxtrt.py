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

import os
import glob
import time
import ctypes
import argparse
import numpy as np
from cuda import cudart
import tensorrt as trt
from typing import Dict, Optional, Sequence, Union

import paddle
from paddle3d.apis.config import Config
from paddle3d.apis.trainer import Trainer
from paddle3d.slim import get_qat_config
from paddle3d.utils.checkpoint import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('--engine', help='checkpoint file')
    parser.add_argument('--plugin', help='plugin file')
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=1)
    parser.add_argument(
        '--model',
        dest='model',
        help='pretrained parameters of the model',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=2)
    args = parser.parse_args()
    return args


class TRTWrapper(paddle.nn.Layer):
    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

    def forward(self, inputs: Dict[str, paddle.Tensor]):
        nIO = self.engine.num_io_tensors  # since TensorRT 8.5, the concept of Binding is replaced by I/O Tensor, all the APIs with "binding" in their name are deprecated
        lTensorName = [
            self.engine.get_tensor_name(i) for i in range(nIO)
        ]  # get a list of I/O tensor names of the engine, because all I/O tensor in Engine and Excution Context are indexed by name, not binding number like TensorRT 8.4 or before
        nInput = [
            self.engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)
        ].count(trt.TensorIOMode.INPUT)  # get the count of input tensor
        bufferH = []
        for i in range(nInput):
            input_data = inputs[i]
            self.context.set_input_shape(lTensorName[i],
                                         tuple(input_data.shape))
            bufferH.append(
                np.ascontiguousarray(input_data)
            )  # set actual size of input tensor if using Dynamic Shape mode

        for i in range(nInput, nIO):
            bufferH.append(
                np.empty(
                    self.context.get_tensor_shape(lTensorName[i]),
                    dtype=trt.nptype(
                        self.engine.get_tensor_dtype(lTensorName[i]))))

        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(
                nInput):  # copy input data from host buffer into device buffer
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data,
                              bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(nIO):
            self.context.set_tensor_address(
                lTensorName[i], int(bufferD[i])
            )  # set address of all input and output data in device buffer

        paddle.device.cuda.synchronize()
        start = time.time()
        self.context.execute_async_v3(0)
        paddle.device.cuda.synchronize()
        time_ = time.time() - start  # do inference computation

        for i in range(
                nInput,
                nIO):  # copy output data from device buffer into host buffer
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i],
                              bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        for b in bufferD:  # free the GPU memory buffer after all work
            cudart.cudaFree(b)
        return bufferH[nInput:nIO], time_


def load_tensorrt_plugin(lib_path) -> bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    success = False
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        print('Successfully loaded tensorrt plugins from', lib_path)
        success = True
    else:
        print(
            'Could not load the library of tensorrt plugins. \
            Because the file does not exist:', lib_path)
    return success


def worker_init_fn(worker_id):
    np.random.seed(1024)


def main():

    args = parse_args()

    if args.cfg is None:
        raise RuntimeError("No configuration file specified!")

    load_tensorrt_plugin(args.plugin)

    if not os.path.exists(args.cfg):
        raise RuntimeError("Config file `{}` does not exist!".format(args.cfg))

    cfg = Config(path=args.cfg, batch_size=args.batch_size)

    if cfg.val_dataset is None:
        raise RuntimeError(
            'The validation dataset is not specified in the configuration file!'
        )
    elif len(cfg.val_dataset) == 0:
        raise ValueError(
            'The length of validation dataset is 0. Please check if your dataset is valid!'
        )

    dic = cfg.to_dict()
    batch_size = dic.pop('batch_size')
    dic.update({
        'dataloader_fn': {
            'batch_size': batch_size,
            'num_workers': args.num_workers,
            'worker_init_fn': worker_init_fn
        }
    })

    if args.model is not None:
        load_pretrained_model(cfg.model, args.model)
        dic['checkpoint'] = None
        dic['resume'] = False
    else:
        dic['resume'] = True

    # build tensorrt model
    trt_model = TRTWrapper(args.engine)

    trainer = Trainer(**dic)
    metric_obj = trainer.val_dataset.metric
    msg = 'evaluate on validate dataset'
    infer_time = 0
    for idx, sample in enumerate(trainer.eval_dataloader):
        if idx % 100 == 0:
            print('predict idx:', idx)

        img_inputs = sample['img_inputs']
        trainer.model.align_after_view_transfromation = True
        feat_prev, input_data = trainer.model.extract_img_feat(
            img_inputs, img_metas=None, pred_prev=True, sequential=False)
        imgs, rots_curr, trans_curr, intrins = input_data[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = input_data[4:]

        if trainer.model.use_ms_depth:
            mlp_input = trainer.model.img_view_transformer.get_mlp_input(*[
                rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
                post_trans, bda[0:1, ...]
            ])

            B, N = rots_curr.shape[:2]
            _, C, H, W = imgs[0].shape
            x_feat = imgs[0].reshape([B, N, C, H, W])
            ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = trainer.model.get_bev_pool_input(
                [
                    x_feat, rots_curr[0:1, ...], trans_curr[0:1, ...], intrins,
                    post_rots, post_trans, bda[0:1, ...], mlp_input
                ])
        else:
            mlp_input = trainer.model.img_view_transformer.get_mlp_input(*[
                rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
                post_trans, bda[0:1, ...]
            ])
            ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = trainer.model.get_bev_pool_input(
                [
                    imgs, rots_curr[0:1, ...], trans_curr[0:1, ...], intrins,
                    post_rots, post_trans, bda[0:1, ...], mlp_input
                ])

        _, C, H, W = feat_prev.shape
        feat_prev = trainer.model.shift_feature(
            feat_prev, [trans_curr, trans_prev], [rots_curr, rots_prev], bda)
        feat_prev = feat_prev.reshape(
            [1, (trainer.model.num_frame - 1) * C, H, W])

        inputs = [
            imgs.numpy(),
            feat_prev.numpy(),
            mlp_input.numpy(),
            interval_starts.numpy(),
            interval_lengths.numpy(),
            ranks_bev.numpy(),
            ranks_feat.numpy(),
            ranks_depth.numpy(),
            mlp_input.numpy()
        ]

        trt_output, time_ = trt_model.forward(inputs)

        if idx >= 10 and idx < 110:
            infer_time += time_
            if idx == 109:
                print('infer time:', infer_time / 100)

        result = {}
        result['boxes_3d'] = trt_output[0]
        result['labels_3d'] = trt_output[1]
        result['scores_3d'] = trt_output[2]
        results = {}
        results['pts_bbox'] = result

        metric_obj.update(predictions=[results], ground_truths=sample)

    metrics = metric_obj.compute(verbose=True)
    print(metrics)


if __name__ == '__main__':
    fps = main()
