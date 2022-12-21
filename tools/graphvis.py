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
import contextlib
import multiprocessing

import paddle
import visualdl
import visualdl.server.app as vdlapp

from paddle3d.apis.config import Config
from paddle3d.slim import get_qat_config
from paddle3d.utils.common import generate_tempdir


def parse_args():
    parser = argparse.ArgumentParser(description='Model Visualization')

    # params of evaluate
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        required=True,
        type=str)
    parser.add_argument(
        "--host",
        dest="host",
        help="VisualDL server host",
        type=str,
        default=None)
    parser.add_argument(
        "--port",
        dest="port",
        help="VisualDL server port",
        type=str,
        default=None)
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help=
        "VisualDL graph save dir. Saves to a temporary directory by default",
        type=str,
        default=None)
    parser.add_argument(
        '--quant_config',
        dest='quant_config',
        help='Config for quant model.',
        default=None,
        type=str)

    return parser.parse_args()


@contextlib.contextmanager
def generate_dir(dir: str = None):
    if dir is not None:
        yield dir
    else:
        with generate_tempdir() as dir:
            yield dir


def main(args):
    cfg = Config(path=args.cfg)

    model = cfg.model
    model.eval()

    if args.quant_config:
        quant_config = get_qat_config(args.quant_config)
        cfg.model.build_slim_model(quant_config['quant_config'])

    with generate_dir(args.save_dir) as _dir:
        with visualdl.LogWriter(logdir=_dir) as writer:
            with model.export_guard():

                writer.add_graph(model, model.input_spec)
                pid = vdlapp.run(
                    logdir=writer._logdir, host=args.host, port=args.port)

                for child in multiprocessing.process._children:
                    if child.pid == pid:
                        child.join()


if __name__ == '__main__':
    args = parse_args()
    main(args)
