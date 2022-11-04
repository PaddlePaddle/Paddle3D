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

from paddle3d.apis.config import Config
from paddle3d.slim import build_slim_model
from paddle3d.utils.checkpoint import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model Export')

    # params of evaluate
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        required=True,
        type=str)
    parser.add_argument(
        '--model',
        dest='model',
        help='pretrained parameters of the model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory saving inference params.',
        type=str,
        default="./exported_model")
    parser.add_argument(
        "--input_shape",
        dest='input_shape',
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)
    parser.add_argument(
        '--slim_config',
        dest='slim_config',
        help='Config for slim model.',
        default=None,
        type=str)

    return parser.parse_args()


def main(args):

    cfg = Config(path=args.cfg)

    model = cfg.model
    model.eval()

    if args.slim_config:
        cfg = build_slim_model(cfg, args.slim_config)

    if args.model is not None:
        load_pretrained_model(model, args.model)

    model.export(
        args.save_dir,
        input_shape=args.input_shape,
        slim=getattr(cfg, 'slim', None))


if __name__ == '__main__':
    args = parse_args()
    main(args)
