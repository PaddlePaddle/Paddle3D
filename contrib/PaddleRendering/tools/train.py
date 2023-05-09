#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import os
import sys
import random
import numpy as np
import paddle
import pprndr.utils.env as pprndr_env
from pprndr.apis.config import Config
from pprndr.apis.trainer import Trainer
from pprndr.utils.checkpoint import load_pretrained_model
from pprndr.utils.logger import logger


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        '--image_batch_size',
        dest='image_batch_size',
        help='The number of images from which rays are sampled.',
        type=int,
        default=None)
    parser.add_argument(
        '--ray_batch_size',
        dest='ray_batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--image_resampling_interval',
        dest='image_resampling_interval',
        help='How often to resample the image.',
        type=int,
        default=None)
    parser.add_argument(
        '--use_adaptive_ray_batch_size',
        dest='use_adaptive_ray_batch_size',
        help='Whether to use adaptive ray batch size.',
        type=bool,
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
        default=500,
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
        '--model',
        dest='model',
        help='pretrained parameters of the model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training. '
        'Default None means 1000 iters',
        type=int,
        default=None)
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
        image_batch_size=args.image_batch_size,
        ray_batch_size=args.ray_batch_size,
        image_resampling_interval=args.image_resampling_interval,
        use_adaptive_ray_batch_size=args.use_adaptive_ray_batch_size)

    if cfg.train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file!')
    elif len(cfg.train_dataset) == 0:
        raise ValueError(
            'The length of training dataset is 0. Please check if your dataset is valid!'
        )

    logger.info('\n{}'.format(pprndr_env.get_env_info()))
    logger.info('\n{}'.format(cfg))

    dic = cfg.to_dict()
    image_batch_size = dic.pop('image_batch_size')
    ray_batch_size = dic.pop('ray_batch_size')
    image_resampling_interval = dic.pop('image_resampling_interval')
    use_adaptive_ray_batch_size = dic.pop('use_adaptive_ray_batch_size')
    save_interval = args.save_interval
    if save_interval is None:
        if cfg.iters:
            save_interval = 1000

    dic.update({
        'resume': args.resume,
        'checkpoint': {
            'keep_checkpoint_max': args.keep_checkpoint_max,
            'save_dir': args.save_dir
        },
        'scheduler': {
            'save_interval': save_interval,
            'log_interval': args.log_interval,
            'do_eval': args.do_eval
        },
        'data_manager_fn': {
            'image_batch_size': image_batch_size,
            'ray_batch_size': ray_batch_size,
            'use_adaptive_ray_batch_size': use_adaptive_ray_batch_size,
            'image_resampling_interval': image_resampling_interval,
            'num_workers': args.num_workers
        }
    })

    if args.model is not None:
        load_pretrained_model(cfg.model, args.model)

    trainer = Trainer(**dic)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
