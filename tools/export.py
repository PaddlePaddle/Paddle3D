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
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model Export')

    # params of export
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        required=True,
        type=str)
    parser.add_argument(
        "--export_for_apollo",
        dest="export_for_apollo",
        help="Whether to export to the deployment format supported by Apollo.",
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
        help='The directory saving inference params.',
        type=str,
        default="./exported_model")
    parser.add_argument(
        '--save_name',
        dest='save_name',
        help='The name of inference params file.',
        type=str,
        default="inference")

    return parser.parse_args()


def generate_apollo_deploy_file(cfg, save_dir: str):
    yml_file = os.path.join(args.save_dir, 'apollo_deploy.yaml')
    model = cfg.model

    with open(yml_file, 'w') as file:
        # Save the content one by one to ensure the content order of the output file
        file.write('# base information\n')
        yaml.dump({'name': model.__class__.__name__}, file)
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
        model_file = '{}.pdmodel'.format(args.save_name)
        params_file = '{}.pdiparams'.format(args.save_name)
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


def main(args):
    cfg = Config(path=args.cfg)

    model = cfg.model
    model.eval()

    if args.model is not None:
        load_pretrained_model(model, args.model)

    model.export(args.save_dir, name=args.save_name)

    if args.export_for_apollo:
        if not isinstance(model, BaseDetectionModel):
            logger.error('Model {} does not support Apollo yet!'.format(
                model.__class__.__name__))
        else:
            generate_apollo_deploy_file(cfg, args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
