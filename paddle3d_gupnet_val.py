import argparse
import numpy as np
import os
from paddle3d.models.detection.gupnet.gupnet_trainer import GupTrainer
from paddle3d.apis.config import Config
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import Logger
import paddle3d.env as paddle3d_env


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')
    # params of training
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default='./configs/gupnet/gupnet_dla34_kitti.yml',
        type=str)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=8)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=0.00125)
    parser.add_argument(
        '--model',
        dest='model',
        help='pretrained parameters of the model',
        type=str,
        default='./checkpoint/best_single_gpu_paddle.pdparams')
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=8)
    parser.add_argument(
        '--eval_frequency',
        help='evaluation interval (in epoch)',
        default=1,
        type=int)
    parser.add_argument(
        '--save_start',
        help='the epoch start save checkpoint',
        default=100,
        type=int)
    parser.add_argument(
        '--save_frequency',
        help='checkpoint save interval (in epoch)',
        default=1,
        type=int)
    parser.add_argument(
        '--disp_frequency',
        help='display interval (in batch)',
        default=20,
        type=int)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./runs/test')

    return parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main(args):
    """
    """
    logger = Logger(output=args.save_dir)
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
    logger.info('\n{}'.format(paddle3d_env.get_env_info()))
    logger.info('\n{}'.format(cfg))
    dic = cfg.to_dict()
    batch_size = dic.pop('batch_size')
    dic.update({
        'checkpoint': {
            'save_dir': args.save_dir
        },
        'dataloader_fn': {
            'batch_size': batch_size,
            'num_workers': args.num_workers,
            'worker_init_fn': worker_init_fn,
        },
        'optimizer': {
            'type': 'adam',
            'weight_decay': 0.00001
        },
        'scheduler': {
            'warmup': True,
            'learning_rate': args.learning_rate,
            'decay_rate': 0.1,
            'decay_list': [90, 120]
        },
        'trainer': {
            'eval_frequency': args.eval_frequency,
            'save_start': args.save_start,
            'save_frequency': args.save_frequency,
            'disp_frequency': args.disp_frequency
        }
    })

    if args.model is not None:
        load_pretrained_model(cfg.model, args.model)

    trainer = GupTrainer(**dic)
    trainer.evaluate()


if __name__ == '__main__':
    args = parse_args()
    main(args)
