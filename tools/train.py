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
import os
import random

import numpy as np
import paddle

import paddle3d.env as paddle3d_env
from paddle3d.apis.config import Config
from paddle3d.apis.trainer import Trainer
from paddle3d.utils.logger import logger


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=None)
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=5)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--log_interval',
        dest='log_interval',
        help='Display logging information at every log_interval',
        default=10,
        type=int)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=2)
    parser.add_argument(
        '--resume',
        dest='resume',
        help='Whether to resume training from checkpoint',
        action='store_true')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed of paddle during training.',
        default=None,
        type=int)

    return parser.parse_args()


def main(args):
    """
    """
    place = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
    paddle.set_device(place)

    if args.seed is not None:
        logger.info("use random seed {}".format(args.seed))
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.cfg is None:
        raise RuntimeError("No configuration file specified!")

    if not os.path.exists(args.cfg):
        raise RuntimeError("Config file `{}` does not exist!".format(args.cfg))

    cfg = Config(
        path=args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

    if cfg.train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file!')
    elif len(cfg.train_dataset) == 0:
        raise ValueError(
            'The length of training dataset is 0. Please check if your dataset is valid!'
        )

    logger.info('\n{}'.format(paddle3d_env.get_env_info()))
    logger.info('\n{}'.format(cfg))

    dic = cfg.to_dict()
    batch_size = dic.pop('batch_size')
    dic.update({
        'resume': args.resume,
        'checkpoint': {
            'keep_checkpoint_max': args.keep_checkpoint_max,
            'save_dir': args.save_dir
        },
        'scheduler': {
            'save_interval': args.save_interval,
            'log_interval': args.log_interval,
            'do_eval': args.do_eval
        },
        'dataloader_fn': {
            'batch_size': batch_size,
            'num_workers': args.num_workers,
        }
    })

    trainer = Trainer(**dic)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
