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
import datetime
import os

import yaml

from paddle3d.apis.config import Config
from paddle3d.models.base import BaseDetectionModel
from paddle3d.slim import get_qat_config
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger

parser = argparse.ArgumentParser(description='Model Export')


def parse_normal_args():
    normal_args = parser.add_argument_group('model args')

    # params of export
    normal_args.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        required=True,
        type=str)
    normal_args.add_argument(
        "--export_for_apollo",
        dest="export_for_apollo",
        help="Whether to export to the deployment format supported by Apollo.",
        action='store_true')
    normal_args.add_argument(
        '--model',
        dest='model',
        help='pretrained parameters of the model',
        type=str,
        default=None)
    normal_args.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory saving inference params.',
        type=str,
        default="./exported_model")
    normal_args.add_argument(
        '--save_name',
        dest='save_name',
        help='The name of inference params file.',
        type=str,
        default=None)
    parser.add_argument(
        '--quant_config',
        dest='quant_config',
        help='Config for quant model.',
        default=None,
        type=str)

    return parser.parse_known_args()


def parse_model_args(arg_dict: dict):
    model_args = parser.add_argument_group('model args')
    for key, value in arg_dict.items():
        model_args.add_argument(key, **value)

    return parser.parse_args()


def generate_apollo_deploy_file(cfg, save_dir: str):
    yml_file = os.path.join(args.save_dir, 'apollo_deploy.yaml')
    model = cfg.model

    with open(yml_file, 'w') as file:
        # Save the content one by one to ensure the content order of the output file
        file.write('# base information\n')
        yaml.dump({'name': model.apollo_deploy_name}, file)
        yaml.dump({'date': datetime.date.today()}, file)
        yaml.dump({'task_type': '3d_detection'}, file)
        yaml.dump({'sensor_type': model.sensor}, file)
        yaml.dump({'framework': 'PaddlePaddle'}, file)

        file.write('\n# dataset information\n')
        yaml.dump({
            'dataset': {
                'name': cfg.train_dataset.name,
                'labels': cfg.train_dataset.labels
            }
        }, file)

        file.write('\n# model information\n')
        transforms = cfg.export_config.get('transforms', [])
        save_name = args.save_name or cfg.model.save_name
        model_file = '{}.pdmodel'.format(save_name)
        params_file = '{}.pdiparams'.format(save_name)
        data = {
            'model': {
                'inputs':
                model.inputs,
                'outputs':
                model.outputs,
                'preprocess':
                transforms,
                'model_files':
                [{
                    'name':
                    model_file,
                    'type':
                    'model',
                    'size':
                    os.path.getsize(os.path.join(args.save_dir, model_file))
                },
                 {
                     'name':
                     params_file,
                     'type':
                     'params',
                     'size':
                     os.path.getsize(os.path.join(args.save_dir, params_file))
                 }]
            }
        }

        yaml.dump(data, file)


def main(args, rest_args):
    cfg = Config(path=args.cfg)

    model = cfg.model
    model.eval()

    if args.quant_config:
        quant_config = get_qat_config(args.quant_config)
        cfg.model.build_slim_model(quant_config['quant_config'])

    if args.model is not None:
        load_pretrained_model(model, args.model)

    arg_dict = {} if not hasattr(model.export,
                                 'arg_dict') else model.export.arg_dict
    args = parse_model_args(arg_dict)
    kwargs = {key[2:]: getattr(args, key[2:]) for key in arg_dict}

    model.export(args.save_dir, name=args.save_name, **kwargs)

    if args.export_for_apollo:
        if not isinstance(model, BaseDetectionModel):
            logger.error('Model {} does not support Apollo yet!'.format(
                model.__class__.__name__))
        else:
            generate_apollo_deploy_file(cfg, args.save_dir)


if __name__ == '__main__':
    args, rest_args = parse_normal_args()
    main(args, rest_args)
