import argparse
import os
import numpy as np

from infer import Infer

from paddle3d.apis.config import Config
from paddle3d.slim import get_qat_config
from paddle3d.utils.checkpoint import load_pretrained_model

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
        default=None)
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
        '--quant_config',
        dest='quant_config',
        help='Config for quant model.',
        default=None,
        type=str)

    return parser.parse_args()


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
    print(args.cfg)
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

    infer = Infer(**dic)
    infer.infer('bev')

if __name__ == '__main__':
    args = parse_args()
    main(args)