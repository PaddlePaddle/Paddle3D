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
from pprndr.apis.config import Config
from pprndr.apis.trainer import Trainer
from pprndr.utils.checkpoint import load_pretrained_model


def parse_args():
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
        '--validate_mesh',
        dest='validate_mesh',
        help=
        'Choose a method (from ["neus_style", ]) to generate mesh from the learned field.',
        type=str,
        default="neus_style")
    parser.add_argument(
        '--bound_min',
        dest='bound_min',
        help=
        'The 3D position from which we start sampling points for extracting mesh',
        type=int,
        default=[-1.0, -1.0, -1.0])
    parser.add_argument(
        '--bound_max',
        dest="bound_max",
        help=
        'The 3D position at which we end sampling points for extracting mesh',
        type=int,
        default=[1.0, 1.0, 1.0])
    parser.add_argument(
        '--mesh_resolution',
        dest='mesh_resolution',
        help='Mesh resolution',
        type=int,
        default=256)
    parser.add_argument(
        '--world_space_for_mesh',
        dest='world_space_for_mesh',
        help=
        'Use wolrd_space for generating mesh. If this is set True, a val_dataset must be provided. Usually set it False.',
        type=bool,
        default=False)
    return parser.parse_args()


def main(args):
    """
    Extract mesh from a trained field.
    A dataset is not required for mesh extraction.
    """
    if args.cfg is None:
        raise RuntimeError("No configuration file specified!")

    if not os.path.exists(args.cfg):
        raise RuntimeError("Config file `{}` does not exist!".format(args.cfg))

    cfg = Config(path=args.cfg)
    dic = cfg.to_dict()
    dic.pop('image_batch_size')
    dic.pop('ray_batch_size')
    dic.pop('image_resampling_interval')
    dic.pop('use_adaptive_ray_batch_size')

    dic.update({
        'train_dataset': None,
        'data_manager_fn': {
            'num_workers': 1
        },
        'checkpoint': False
    })

    if args.model is not None:
        load_pretrained_model(cfg.model, args.model)

    validate_mesh = args.validate_mesh
    if validate_mesh is not None:
        if not validate_mesh in ["neus_style"]:
            raise NotImplementedError(
                "Only neus_style is supported for extracting mesh.")

    trainer = Trainer(**dic)
    trainer.extract_mesh(
        save_dir=os.path.join(os.path.split(args.model)[0], "renderings"),
        val_ray_batch_size=args.ray_batch_size,
        validate_mesh=validate_mesh,
        mesh_resolution=args.mesh_resolution,
        world_space_for_mesh=args.world_space_for_mesh,
        bound_min=args.bound_min,  # used for generating mesh
        bound_max=args.bound_max  # used for generating mesh
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
