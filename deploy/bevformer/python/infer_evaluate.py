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

import argparse
import os
import random
import copy

import numpy as np
import paddle
from paddle import inference

from paddle3d.ops import ms_deform_attn
from paddle3d.apis.config import Config
from paddle3d.apis.trainer import Trainer
from paddle3d.slim import get_qat_config
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger
from paddle3d.sample import Sample, SampleMeta
from paddle3d.geometries import BBoxes3D


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
    config.disable_glog_info()

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


def _parse_results_to_sample(results: dict, sample: dict):
    num_samples = len(results)
    new_results = []
    for i in range(num_samples):
        data = Sample(None, sample["modality"][i])
        bboxes_3d = results[i]["boxes_3d"]
        labels = results[i]["labels_3d"]
        confidences = results[i]["scores_3d"]
        bottom_center = bboxes_3d[:, :3]
        gravity_center = np.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + bboxes_3d[:, 5] * 0.5
        bboxes_3d[:, :3] = gravity_center
        data.bboxes_3d = BBoxes3D(bboxes_3d[:, 0:7])
        data.bboxes_3d.coordmode = 'Lidar'
        data.bboxes_3d.origin = [0.5, 0.5, 0.5]
        data.bboxes_3d.rot_axis = 2
        data.bboxes_3d.velocities = bboxes_3d[:, 7:9]
        data['bboxes_3d_numpy'] = bboxes_3d[:, 0:7]
        data['bboxes_3d_coordmode'] = 'Lidar'
        data['bboxes_3d_origin'] = [0.5, 0.5, 0.5]
        data['bboxes_3d_rot_axis'] = 2
        data['bboxes_3d_velocities'] = bboxes_3d[:, 7:9]
        data.labels = labels
        data.confidences = confidences
        data.meta = SampleMeta(id=sample["meta"][i]['id'])
        if "calibs" in sample:
            calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
            data.calibs = calib
        new_results.append(data)
    return new_results


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
    metric_obj = trainer.val_dataset.metric
    msg = 'evaluate on validate dataset'
    prev_bev = None
    prev_pos = np.array([0.0, 0.0, 0.0]).astype('float32')
    prev_angle = np.array([0.0]).astype('float32')
    scene_token = None
    for idx, sample in enumerate(trainer.eval_dataloader):
        input = []
        if idx % 100 == 0:
            print('predict idx:', idx)
        img_metas = sample['meta']
        img = sample['img'][0].numpy().astype(np.float32)
        input.append(img)

        if img_metas[0]['scene_token'] != scene_token:
            # the first sample of each scene is truncated
            prev_bev = None
        # update idx
        scene_token = img_metas[0]['scene_token']

        can_bus = img_metas[0]['can_bus'].numpy().astype(np.float32)
        tmp_pos = copy.deepcopy(can_bus[:3])
        tmp_angle = copy.deepcopy(can_bus[-1])
        if prev_bev is None:
            can_bus[:3] = 0
            can_bus[-1] = 0
        else:
            can_bus[:3] -= prev_pos
            can_bus[-1] -= prev_angle
        if prev_bev is None:
            prev_bev = np.zeros((2500, 1, 256)).astype(np.float32)
        input.append(prev_bev)
        input.append(can_bus)
        prev_pos = tmp_pos
        prev_angle = tmp_angle

        img_shape = np.asarray(img_metas[0]['img_shape']).astype(np.float32)
        input.append(img_shape)

        img2lidars = []
        for img_meta in img_metas:
            img2lidars.append(np.asarray(img_meta['lidar2img']))

        img2lidars = np.asarray(img2lidars).astype('float32')
        input.append(img2lidars)

        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(input[i].shape)
            input_tensor.copy_from_cpu(input[i].copy())
        predictor.run()

        outs = []
        output_names = predictor.get_output_names()
        for name in output_names:
            out = predictor.get_output_handle(name)
            out = out.copy_to_cpu()
            outs.append(out)
        prev_bev = outs[0]
        result = {}
        result['boxes_3d'] = outs[1]
        result['labels_3d'] = outs[2]
        result['scores_3d'] = outs[3]
        pred = _parse_results_to_sample([result], sample)
        metric_obj.update(predictions=pred, ground_truths=sample)
    metrics = metric_obj.compute(verbose=True)
    print(metrics)


if __name__ == '__main__':
    args = parse_args()
    main(args)
