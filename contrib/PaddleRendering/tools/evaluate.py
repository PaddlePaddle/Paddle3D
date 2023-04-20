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
import sys
sys.path.insert(1, "/home/liuxing/workspace/Paddle3D/contrib/PaddleRendering/")
from pprndr.apis.config import Config
from pprndr.apis.trainer import Trainer
from pprndr.utils.checkpoint import load_pretrained_model


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Model evaluation')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--ray_batch_size',
        dest='ray_batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=16384)
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
    return parser.parse_args()


def main(args):
    """
    """
    if args.cfg is None:
        raise RuntimeError("No configuration file specified!")

    if not os.path.exists(args.cfg):
        raise RuntimeError("Config file `{}` does not exist!".format(args.cfg))

    cfg = Config(path=args.cfg)

    if cfg.val_dataset is None:
        raise RuntimeError(
            'The validation dataset is not specified in the configuration file!'
        )
    elif len(cfg.val_dataset) == 0:
        raise ValueError(
            'The length of validation dataset is 0. Please check if your dataset is valid!'
        )

    dic = cfg.to_dict()
    dic.pop('image_batch_size')
    dic.pop('ray_batch_size')
    dic.pop('image_resampling_interval')
    dic.pop('use_adaptive_ray_batch_size')
    dic.update({
        'train_dataset': None,
        'data_manager_fn': {
            'num_workers': args.num_workers
        },
        'checkpoint': False
    })
    if args.model is not None:
        load_pretrained_model(cfg.model, args.model)

    if hasattr(dic["val_dataset"], "max_eval_num"):
        max_eval_num = dic["val_dataset"].max_eval_num
    else:
        max_eval_num = None

    if hasattr(dic["val_dataset"], "validate_mesh"):
        if not dic["val_dataset"].validate_mesh in ["neus_style"]:
            raise NotImplementedError(
                "Only neus_style is supported for extracting mesh.")
        validate_mesh = dic["val_dataset"].validate_mesh
    else:
        validate_mesh = None

    if hasattr(dic["val_dataset"], "mesh_resolution"):
        mesh_resolution = dic["val_dataset"].mesh_resolution
    else:
        mesh_resolution = 64

    if hasattr(dic["val_dataset"], "world_space_for_mesh"):
        world_space_for_mesh = dic["val_dataset"].world_space_for_mesh
    else:
        world_space_for_mesh = False

    if hasattr(dic["val_dataset"], "bound_min"):
        bound_min = dic["val_dataset"].bound_min
    else:
        bound_min = None

    if hasattr(dic["val_dataset"], "bound_max"):
        bound_max = dic["val_dataset"].bound_max
    else:
        bound_max = None

    if hasattr(dic["val_dataset"], "eval_with_grad"):
        eval_with_grad = dic["val_dataset"].eval_with_grad
    else:
        eval_with_grad = False

    trainer = Trainer(**dic)
    trainer.evaluate(
        save_dir=os.path.join(os.path.split(args.model)[0], "renderings"),
        val_ray_batch_size=args.ray_batch_size,
        max_eval_num=max_eval_num,
        validate_mesh=validate_mesh,
        mesh_resolution=mesh_resolution,
        world_space_for_mesh=world_space_for_mesh,
        bound_min=bound_min,  # used for generating mesh
        bound_max=bound_max,  # used for generating mesh
        eval_with_grad=eval_with_grad)


if __name__ == '__main__':
    args = parse_args()
    main(args)
