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

import copy
import numpy as np
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import cv2
import paddle
from visualdl import LogWriter
from paddle.vision.transforms import functional as F
import trimesh
import pprndr.utils.env as env
from pprndr.apis.checkpoint import Checkpoint, CheckpointABC
from pprndr.apis.pipeline import inference_step, training_step
from pprndr.apis.scheduler import Scheduler, SchedulerABC
from pprndr.data import BaseDataset, DataManager
from pprndr.metrics import MetricABC, PSNRMeter
from pprndr.utils.logger import logger
from pprndr.utils.timer import Timer


def default_data_manager_build_fn(**kwargs) -> Callable:
    def _build_data_manager(dataset: BaseDataset) -> DataManager:
        args = kwargs.copy()
        image_batch_size = args.pop('image_batch_size', -1)
        ray_batch_size = args.pop('ray_batch_size', 1)
        image_resampling_interval = args.pop('image_resampling_interval', -1)
        use_adaptive_ray_batch_size = args.pop('use_adaptive_ray_batch_size',
                                               False)
        if dataset.is_train_mode and image_resampling_interval > 0 and 0 < image_batch_size < len(
                dataset):
            shuffle = True
        else:
            shuffle = False
        drop_last = args.pop('drop_last',
                             True if dataset.is_train_mode else False)

        return DataManager(
            dataset=dataset,
            image_batch_size=image_batch_size,
            ray_batch_size=ray_batch_size,
            image_resampling_interval=image_resampling_interval,
            use_adaptive_ray_batch_size=use_adaptive_ray_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **args)

    return _build_data_manager


def default_checkpoint_build_fn(**kwargs) -> Checkpoint:
    """
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('save_dir', 'output')
    kwargs.setdefault('keep_checkpoint_max', 5)
    kwargs.setdefault('overwrite', True)
    return Checkpoint(**kwargs)


def default_scheduler_build_fn(**kwargs) -> Scheduler:
    """
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('save_interval', 1000)
    kwargs.setdefault('log_interval', 10)
    kwargs.setdefault('do_eval', False)

    return Scheduler(**kwargs)


class Trainer(object):
    """
    """

    def __init__(self,
                 model: paddle.nn.Layer,
                 optimizer: paddle.optimizer.Optimizer,
                 iters: Optional[int] = None,
                 eval_pixel_stride: Optional[int] = 1,
                 train_dataset: Optional[BaseDataset] = None,
                 val_dataset: Optional[BaseDataset] = None,
                 train_metric_meters: Optional[List[MetricABC]] = None,
                 val_metric_meters: Optional[List[MetricABC]] = None,
                 resume: bool = False,
                 checkpoint: Union[dict, CheckpointABC] = None,
                 scheduler: Union[dict, SchedulerABC] = None,
                 data_manager_fn: Union[dict, Callable] = None,
                 amp_cfg: Optional[dict] = None,
                 grad_accum_cfg: Optional[dict] = None,
                 reinit_optim_cfg: Optional[dict] = None):

        self.model = model
        self.optimizer = optimizer
        self.eval_pixel_stride = eval_pixel_stride

        if train_dataset is not None:
            data_manager_fn = data_manager_fn or {}
            _data_manager_build_fn = default_data_manager_build_fn(
                **data_manager_fn) if isinstance(data_manager_fn,
                                                 dict) else data_manager_fn

            self.train_data_manager = _data_manager_build_fn(train_dataset)
        else:
            self.train_data_manager = None

        self.val_dataset = val_dataset
        if val_dataset is not None:
            eval_batch_sampler = paddle.io.DistributedBatchSampler(
                val_dataset, batch_size=1, shuffle=False, drop_last=False)
            self.eval_data_loader = paddle.io.DataLoader(
                val_dataset,
                batch_sampler=eval_batch_sampler,
                num_workers=data_manager_fn['num_workers'])
        else:
            self.eval_data_loader = None

        if train_metric_meters is None:
            self.train_metric_meters = [PSNRMeter()]
        else:
            self.train_metric_meters = train_metric_meters

        if val_metric_meters is None:
            self.val_metric_meters = [PSNRMeter()]
        else:
            self.val_metric_meters = val_metric_meters

        if checkpoint == False:
            return

        checkpoint = checkpoint or {}
        self.checkpoint = default_checkpoint_build_fn(
            **checkpoint) if isinstance(checkpoint, dict) else checkpoint

        scheduler = scheduler or {}
        if isinstance(scheduler, dict):
            self.scheduler = default_scheduler_build_fn(**scheduler)
        else:
            self.scheduler = scheduler

        self.resume = resume
        vdl_file_name = None

        self.iters = iters
        self.cur_iter = 0

        if not self.checkpoint.empty:
            if not resume:
                raise RuntimeError(
                    'The checkpoint {} is not emtpy! '
                    'Set `resume=True` to continue training or use another dir as checkpoint'
                    .format(self.checkpoint.rootdir))

            params_dict, opt_dict = self.checkpoint.get()
            self.model.set_dict(params_dict)
            self.optimizer.set_state_dict(opt_dict)
            self.cur_iter = self.checkpoint.meta.get('iters')
            self.scheduler.step(self.cur_iter)

            logger.info(
                'Resume model from Checkpoint {}, current iter set to {}'.
                format(self.checkpoint.rootdir, self.cur_iter))
            vdl_file_name = self.checkpoint.meta['vdl_file_name']
        elif resume:
            logger.warning(
                "Attempt to restore parameters from an empty Checkpoint")

        if env.local_rank == 0:
            self.log_writer = LogWriter(
                logdir=self.checkpoint.rootdir, file_name=vdl_file_name)
            self.checkpoint.record('vdl_file_name',
                                   os.path.basename(self.log_writer.file_name))

        if amp_cfg is not None:
            scaler_cfg_ = dict(init_loss_scaling=2.**15)
            scaler_cfg_.update(**amp_cfg.pop('scaler', dict()))
            self.scaler = paddle.amp.GradScaler(**scaler_cfg_)

            setattr(self.model, 'amp_cfg_', amp_cfg)
            setattr(self.train_data_manager, 'amp_cfg_', amp_cfg)
            logger.info(
                'Use AMP train, AMP config: {}, Scaler config: {}'.format(
                    amp_cfg, scaler_cfg_))
        else:
            self.scaler = None

        self.grad_accum_cfg = grad_accum_cfg
        self.reinit_optim_cfg = reinit_optim_cfg

    def train(self):
        """
        """
        if self.train_data_manager is None:
            raise RuntimeError('No training dataset specified!')

        train_data_loader = iter(self.train_data_manager)

        model = self.model
        if env.nranks > 1:
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()
            model = paddle.DataParallel(self.model)

        losses_sum = defaultdict(float)
        for meter in self.train_metric_meters:
            meter.reset()
        timer = Timer(iters=self.iters - self.cur_iter)
        while self.cur_iter < self.iters:
            for sample in train_data_loader:
                self.cur_iter += 1
                if self.cur_iter > self.iters:
                    break

                lr = self.optimizer.get_lr()
                output = training_step(
                    model,
                    self.optimizer,
                    self.cur_iter,
                    sample,
                    scaler=self.scaler,
                    grad_accum_cfg=self.grad_accum_cfg)

                if isinstance(output['loss'], dict):
                    for k, v in output['loss'].items():
                        losses_sum[k] += float(v)

                losses_sum['total_loss'] += float(output['total_loss'])

                for meter in self.train_metric_meters:
                    meter.update(
                        predictions=output["rgb"],
                        ground_truths=sample[1]["pixels"])

                timer.step(num_samples=self.train_data_manager.ray_batch_size)

                if self.train_data_manager.use_adaptive_ray_batch_size:
                    # update ray batch size according to the number of samples in current iteration
                    num_samples_per_batch = output['num_samples_per_batch']
                    self.train_data_manager.update_ray_batch_size(
                        num_samples_per_batch)

                status = self.scheduler.step()

                if status.do_log and env.local_rank == 0:
                    self.log_writer.add_scalar(
                        tag='Training/learning_rate',
                        value=lr,
                        step=self.cur_iter)

                    loss_log = ''
                    for k, v in losses_sum.items():
                        loss_val = v / self.scheduler.log_interval
                        loss_log += ', {}={:.6f}'.format(k, loss_val)
                        self.log_writer.add_scalar(
                            tag='Training/' + k,
                            value=loss_val,
                            step=self.cur_iter)

                    metric_log = ''
                    for meter in self.train_metric_meters:
                        metric_val = meter.accumulate().item()
                        metric_log += ', {}={:.4f}'.format(
                            meter.name, metric_val)
                        self.log_writer.add_scalar(
                            tag='Training/' + meter.name,
                            value=metric_val,
                            step=self.cur_iter)
                        meter.reset()

                    logger.info(
                        '[TRAIN] iter={}/{}{}{}, lr={:.6f}, throughput={:.2f} rays/sec | ETA {}'
                        .format(self.cur_iter, self.iters, loss_log, metric_log,
                                lr, timer.get_throughput(), timer.eta))

                    losses_sum.clear()
                elif status.do_log:
                    for meter in self.train_metric_meters:
                        meter.reset()
                    losses_sum.clear()

                if status.do_eval:
                    # TODO: whether to save a checkpoint based on the metric
                    eval_ray_batch_size = self.train_data_manager.ray_batch_size

                    eval_to_cpu = getattr(self.val_dataset, "eval_to_cpu",
                                          False)
                    image_coords_offset = getattr(self.val_dataset,
                                                  "image_coords_offset", 0.5)

                    validate_mesh = getattr(self.val_dataset, "validate_mesh",
                                            None)
                    if validate_mesh is not None and validate_mesh not in [
                            "neus_style"
                    ]:
                        raise NotImplementedError(
                            "Only neus_style is supported for extracting mesh.")

                    if validate_mesh is not None:
                        mesh_resolution = getattr(self.val_dataset,
                                                  "mesh_resolution", 64)
                        world_space_for_mesh = getattr(
                            self.val_dataset, "world_space_for_mesh", False)
                        bound_min = getattr(self.val_dataset, "bound_min", None)
                        bound_max = getattr(self.val_dataset, "bound_max", None)
                    else:
                        mesh_resolution = None
                        world_space_for_mesh = None
                        bound_min = None
                        bound_max = None

                    metrics = self.evaluate(
                        save_dir=os.path.join(self.checkpoint.save_dir,
                                              'iter_{}'.format(self.cur_iter),
                                              'renderings'),
                        val_ray_batch_size=eval_ray_batch_size,
                        validate_mesh=validate_mesh,
                        mesh_resolution=mesh_resolution,
                        pixel_stride=self.eval_pixel_stride,
                        world_space_for_mesh=world_space_for_mesh,
                        bound_min=bound_min,
                        bound_max=bound_max,
                        image_coords_offset=image_coords_offset,
                        eval_to_cpu=eval_to_cpu)

                    for k, v in metrics.items():
                        if not isinstance(v, paddle.Tensor) or v.numel() != 1:
                            continue

                        self.log_writer.add_scalar(
                            tag='Evaluation/{}'.format(k),
                            value=float(v),
                            step=self.cur_iter)

                if status.save_checkpoint and env.local_rank == 0:
                    tag = 'iter_{}'.format(self.cur_iter)

                    self.checkpoint.push(
                        tag=tag,
                        params_dict=self.model.state_dict(),
                        opt_dict=self.optimizer.state_dict(),
                        verbose=True)

                    self.checkpoint.record('iters', self.cur_iter)

                if self.reinit_optim_cfg is not None:
                    if self.cur_iter in self.reinit_optim_cfg['upsamp_list']:
                        model.update_to_step(self.cur_iter)
                        self.optimizer = model.reinitialize_optimizer(
                            self.reinit_optim_cfg, model.parameters())

        logger.info('Training is complete.')

        if env.local_rank == 0:
            tag = 'iter_{}'.format(self.iters)

            if not self.checkpoint.have(tag):
                self.checkpoint.push(
                    tag=tag,
                    params_dict=self.model.state_dict(),
                    opt_dict=self.optimizer.state_dict(),
                    verbose=True)

            self.checkpoint.record('iters', self.iters)

    def validate_mesh_neus(self,
                           save_dir: str,
                           cur_iter: int = None,
                           world_space: bool = False,
                           resolution: int = 64,
                           threshold: float = 0.0,
                           bound_min=None,
                           bound_max=None):

        if bound_min is None:
            bound_min = self.val_dataset.object_bbox_min
        elif isinstance(bound_min, list):
            bound_min = np.asarray(bound_min)

        if bound_max is None:
            bound_max = self.val_dataset.object_bbox_max
        elif isinstance(bound_max, list):
            bound_max = np.asarray(bound_max)

        vertices, triangles = self.model.extract_geometry(
            bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(save_dir, 'meshes'), exist_ok=True)
        if world_space:
            vertices = vertices * self.val_dataset.scale_mats_np[0][0, 0] + \
                       self.val_dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if cur_iter is not None:
            mesh_name = '{:0>8d}.ply'.format(cur_iter)
        else:
            mesh_name = "mesh.ply"
        mesh.export(os.path.join(save_dir, 'meshes', mesh_name))

    @paddle.no_grad()
    def extract_mesh(self,
                     save_dir: str,
                     validate_mesh: str = None,
                     mesh_resolution: int = 64,
                     world_space_for_mesh: bool = False,
                     bound_min=None,
                     bound_max=None) -> None:
        """
        """
        os.makedirs(save_dir, exist_ok=True)

        # Only support neus_style for now for generating meshes.
        if validate_mesh == "neus_style":
            if world_space_for_mesh:
                logger.info(
                    "Use world space for generating mesh, Checking if val_dataset is provided..."
                )
                assert self.val_dataset is not None
                logger.info("Done.")

            self.validate_mesh_neus(
                save_dir,
                cur_iter=None,
                world_space=world_space_for_mesh,
                resolution=mesh_resolution,
                bound_min=bound_min,
                bound_max=bound_max)
            logger.info('Mesh generated from sdf.')
        else:
            raise NotImplementedError("Invalid method for validate_mesh.")

    def evaluate(self,
                 save_dir: str,
                 val_ray_batch_size: int = 16384,
                 validate_mesh: str = None,
                 mesh_resolution: int = 64,
                 pixel_stride: int = 1,
                 world_space_for_mesh: bool = False,
                 bound_min=None,
                 bound_max=None,
                 image_coords_offset: float = 0.5,
                 eval_to_cpu: bool = False) -> Dict:
        """
        """
        if self.eval_data_loader is None:
            raise RuntimeError('No evaluation dataset specified!')

        os.makedirs(save_dir, exist_ok=True)
        msg = 'Evaluating on validation dataset...'

        for meter in self.val_metric_meters:
            meter.reset()

        cameras = self.val_dataset.cameras.cuda()
        image_coords = cameras.get_image_coords(
            step=pixel_stride, offset=image_coords_offset).reshape([-1, 2])

        # Only support neus_style for now for generating mesh.
        if validate_mesh is not None:
            if validate_mesh == "neus_style":
                if world_space_for_mesh:
                    logger.info(
                        "Use world space for generating mesh, Checking if val_dataset is provided..."
                    )
                    assert (not self.val_dataset is None)
                    logger.info("Done.")

                self.validate_mesh_neus(
                    save_dir,
                    cur_iter=None,
                    world_space=world_space_for_mesh,
                    resolution=mesh_resolution,
                    bound_min=bound_min,
                    bound_max=bound_max)
                logger.info('Mesh generated from sdf.')
            else:
                raise NotImplementedError("Invalid method for validate_mesh.")

        for idx, image_batch in logger.enumerate(
                self.eval_data_loader, msg=msg):

            # Select GT pixels from rgb image
            if pixel_stride > 1:
                image_batch["image"] = image_batch[
                    "image"][:, ::pixel_stride, ::pixel_stride, :]

            ray_bundle = cameras.generate_rays(
                camera_ids=image_batch["camera_id"], image_coords=image_coords)

            output = inference_step(
                self.model, ray_bundle, val_ray_batch_size, to_cpu=eval_to_cpu)

            output["rgb"] = output["rgb"].reshape(image_batch["image"].shape)

            # Get normals for visualization
            if "normals" in output:
                normal_image = output["normals"]
                rot = cameras.c2w_matrices[idx, :3, :3]
                rot = paddle.linalg.inv(rot)[None, :, :]
                normal_image = normal_image[..., None]
                normal_image = paddle.matmul(rot, normal_image)[:, :, 0]
                H = cameras._image_height[idx]
                W = cameras._image_width[idx]
                if pixel_stride != 1:
                    H = int(H / pixel_stride)
                    W = int(W / pixel_stride)
                normal_image = normal_image.reshape([H, W, 3]) * 128 + 128
                normal_image = normal_image.clip(0, 255)
                normal_image = normal_image.numpy()
                cv2.imwrite("{}/{}_normal.png".format(save_dir, idx),
                            normal_image)

            for meter in self.val_metric_meters:
                meter.update(
                    predictions=output["rgb"],
                    ground_truths=image_batch["image"])

            if env.nranks > 1:
                if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
                ):
                    paddle.distributed.init_parallel_env()
                pred_list = []
                camera_id_list = []
                paddle.distributed.all_gather(pred_list, output["rgb"])
                paddle.distributed.all_gather(camera_id_list,
                                              image_batch["camera_id"])
                pred = paddle.concat(pred_list, axis=0)
                camera_ids = paddle.concat(camera_id_list, axis=0)
            else:
                pred = output["rgb"]
                camera_ids = image_batch["camera_id"]

            if env.local_rank == 0:
                # save predictions
                for camera_id, image in zip(camera_ids.tolist(), pred):
                    cv2.imwrite(
                        os.path.join(save_dir, "{}.png".format(camera_id)),
                        cv2.cvtColor((image * 255.).astype("uint8").numpy(),
                                     cv2.COLOR_RGB2BGR))
            del output

        if env.nranks > 1:
            results = {
                meter.name: paddle.distributed.all_reduce(
                    meter.accumulate(verbose=env.local_rank == 0) / env.nranks)
                for meter in self.val_metric_meters
            }
        else:
            results = {
                meter.name: meter.accumulate(verbose=True)
                for meter in self.val_metric_meters
            }

        return results
