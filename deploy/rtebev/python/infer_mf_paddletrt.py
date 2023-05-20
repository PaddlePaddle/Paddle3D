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

import time
import argparse
import os
import random
import numpy as np

import paddle
from paddle import inference

from paddle3d.apis.config import Config
from paddle3d.apis.trainer import Trainer
from paddle3d.slim import get_qat_config
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger
from paddle3d.sample import Sample, SampleMeta
from paddle3d.geometries import BBoxes3D
from paddle3d.ops import bev_pool_v2


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Model evaluation')
    # params of training
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
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        action='store_true',
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
    parser.add_argument(
        "--trt_use_static",
        action='store_true',
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")
    parser.add_argument(
        "--collect_shape_info",
        action='store_true',
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="petr_shape_info.txt",
        help="Path of a dynamic shape file for tensorrt.")
    parser.add_argument(
        '--quant_config',
        dest='quant_config',
        help='Config for quant model.',
        default=None,
        type=str)

    return parser.parse_args()


def load_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    """load_predictor
    initialize the inference engine
    """
    config = inference.Config(model_file, params_file)
    config.enable_use_gpu(1000, gpu_id)

    # enable memory optim
    config.enable_memory_optim()
    # config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=30,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        print('collect_shape_info', collect_shape_info)
        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)
    predictor = inference.create_predictor(config)

    return predictor


def worker_init_fn(worker_id):
    np.random.seed(1024)


def main(args):
    """
    """
    if args.cfg is None:
        raise RuntimeError("No configuration file specified!")

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

    if args.quant_config:
        quant_config = get_qat_config(args.quant_config)
        cfg.model.build_slim_model(quant_config['quant_config'])

    if args.model is not None:
        load_pretrained_model(cfg.model, args.model)
        dic['checkpoint'] = None
        dic['resume'] = False
    else:
        dic['resume'] = True

    predictor = load_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)

    trainer = Trainer(**dic)
    trainer.model.eval()

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

        input_names = predictor.get_input_names()

        input_tensor0 = predictor.get_input_handle(input_names[0])
        input_tensor1 = predictor.get_input_handle(input_names[1])
        input_tensor2 = predictor.get_input_handle(input_names[2])
        input_tensor3 = predictor.get_input_handle(input_names[3])
        input_tensor4 = predictor.get_input_handle(input_names[4])
        input_tensor5 = predictor.get_input_handle(input_names[5])
        input_tensor6 = predictor.get_input_handle(input_names[6])
        input_tensor7 = predictor.get_input_handle(input_names[7])

        input_tensor0.copy_from_cpu(imgs.numpy())
        input_tensor1.copy_from_cpu(feat_prev.numpy())
        input_tensor2.copy_from_cpu(mlp_input.numpy())
        input_tensor3.copy_from_cpu(ranks_depth.numpy())
        input_tensor4.copy_from_cpu(ranks_feat.numpy())
        input_tensor5.copy_from_cpu(ranks_bev.numpy())
        input_tensor6.copy_from_cpu(interval_starts.numpy())
        input_tensor7.copy_from_cpu(interval_lengths.numpy())

        paddle.device.cuda.synchronize()
        start = time.time()
        predictor.run()

        outs = []
        output_names = predictor.get_output_names()
        for name in output_names:
            out = predictor.get_output_handle(name)
            out = out.copy_to_cpu()
            outs.append(out)
        paddle.device.cuda.synchronize()

        if idx >= 10 and idx < 110:
            infer_time += (time.time() - start)
            if idx == 109:
                print('infer time:', infer_time / 100)

        result = {}
        result['boxes_3d'] = outs[0]
        result['scores_3d'] = outs[1]
        result['labels_3d'] = outs[2]
        results = {}
        results['pts_bbox'] = result

        metric_obj.update(predictions=[results], ground_truths=sample)

    metrics = metric_obj.compute(verbose=True)
    print(metrics)


if __name__ == '__main__':
    args = parse_args()
    main(args)
