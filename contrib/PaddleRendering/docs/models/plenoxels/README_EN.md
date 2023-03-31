# Plenoxels: Radiance Fields without Neural Networks

## Contents

* [Reference](#1)
* [Introduction](#2)
* [Model Zoo](#3)
* [Training Configuration](#4)
* [Tutorials](#5)
    * [Data Preparation](#51)
    * [Training](#52)
    * [Evaluation](#53)

## <h2 id="1">Reference</h2>

```bibtex
@inproceedings{yu_and_fridovichkeil2021plenoxels,
      title={Plenoxels: Radiance Fields without Neural Networks},
      author={{Sara Fridovich-Keil and Alex Yu} and Matthew Tancik and
      Qinhong Chen and Benjamin Recht and Angjoo Kanazawa},
      year={2022},
      booktitle={CVPR},
}
```

## <h2 id="2">Introduction</h2>

Plenoxels:
A view-dependent sparse voxel model Plenoxel (plenoptic volume element) is proposed to store density and spherical harmonic coefficient information. The authors established a sparse voxel grid field based on Plenoxel, obtained the color and opacity of the sampling point through trilinear interpolation of the grid point information, and finally used differentiable volume rendering to predict the color. The model can be optimized to the same fidelity as Neural Radiation Fields ([NeRF](https://arxiv.org/abs/2003.08934)) without using any neural networks. Compared with NeRF, Plenoxels trains two orders of magnitude faster.

## <h2 id="3">Model Zoo</h2>

- Benchmarks on Blender Dataset.

| Scene |  PSNR   |  SSIM  |                                               Download                                               |                      Configuration                      |
|:-----:|:-------:|:------:|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------:|
| lego  | 31.1187 | 0.9414 | [model](https://paddle3d.bj.bcebos.com/render/models/plenoxels/plenoxels_blender/model.pdparams) | [config](../../../configs/plenoxels/blender_data.yml) |
| lego  | 31.1781 | 0.9420 | [model](https://paddle3d.bj.bcebos.com/render/models/plenoxels/plenoxels_blender_grad_accum/model.pdparams) | [grad_accum_config](../../../configs/plenoxels/blender_data_grad_accum.yml) |

## <h2 id="4">Training Configuration</h2>

For training configuration on open source datasets, refer
to [Plenoxels Training Configuration](../../../configs/plenoxels). Among them, `blender_data_grad_accum.yml` adopts the gradient accumulation update strategy, which is convenient for you to use a larger equivalent `ray_batch_size`.

## <h2 id="5">Tutorials</h2>

### <h3 id="51">Data Preparation</h3>

Soft link the dataset file to `PaddleRender/data/` or specify the dataset path
in [configuration file](../../../configs/plenoxels).

### <h3 id="52">Training</h3>

At `PaddleRendering/`, execute:

```shell
export PYTHONPATH='.'

# Train on single GPU
python tools/train.py \
  --config configs/plenoxels/blender_data.yml \
  --save_dir plenoxels_blender \
  --log_interval 1000 \
  --save_interval 10000

# Train on multiple GPUs (GPU 0, 1 as an example)
python -m paddle.distributed.launch --devices 0,1 \
    tools/train.py \
    --config configs/plenoxels/blender_data.yml \
    --save_dir plenoxels_blender \
    --log_interval 1000 \
    --save_interval 10000
```

The training script accepts the following arguments and options:

| Arguments & Options         | Explanation                                                                                                                                                                                                                                                                                               | Required | Defaults                         |
|:----------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|:---------------------------------|
| config                      | Configuration file.                                                                                                                                                                                                                                                                                       | YES      | -                                |
| image_batch_size            | Batch size of images, from which rays are sampled every iteration.<br>If `-1`, rays are sampled from the entire dataset.                                                                                                                                                                                  | NO       | -1                               |
| ray_batch_size              | Batch size of rays, the number of rays sampled from image mini-batch every iteration.                                                                                                                                                                                                                     | NO       | 1                                |
| image_resampling_interval   | To accelerate training, each GPU maintains a image buffer (image mini-batch is prefetched, rays are sampled from the buffer every iteration).<br>This argument specifies the interval of updating the image buffer. If `-1`, the buffer is never updated (for the case where `image_batch_size` is `-1`). | NO       | -1                               |
| use_adaptive_ray_batch_size | Whether to use an adaptive `ray_batch_size`.<br>If enabled, the number of valid samples fed to the model is stable at `2^18`, which accelerates model convergence.                                                                                                                                        | NO       | FALSE                            |
| iters                       | The number of iterations.                                                                                                                                                                                                                                                                                 | NO       | Specified in configuration file. |
| learning_rate               | Learning rate.                                                                                                                                                                                                                                                                                            | NO       | Specified in configuration file. |
| save_dir                    | Directory where models and VisualDL logs are saved.                                                                                                                                                                                                                                                       | NO       | output                           |
| save_interval               | Interval of saving checkpoints.                                                                                                                                                                                                                                                                           | NO       | 1000                             |
| do_eval                     | Whether to do evaluation after checkpoints are saved.                                                                                                                                                                                                                                                     | NO       | FALSE                            |
| resume                      | Whether to resume interrupted training.                                                                                                                                                                                                                                                                   | NO       | FALSE                            |
| model                       | Path to pretrained model file (`.pdparams`).                                                                                                                                                                                                                                                              | NO       | No pretrained model.             |
| log_interval                | Interval for logging.                                                                                                                                                                                                                                                                                     | NO       | 500                              |
| keep_checkpoint_max         | The maximum number of saved checkpoints (When the number of saved checkpoint exceeds the limit, the oldest checkpoint is automatically deleted).                                                                                                                                                          | NO       | 5                                |

### <h3 id="53">Evaluation</h3>

At `PaddleRendering/`, execute:

```shell
export PYTHONPATH='.'
python tools/evaluate.py \
  --config configs/plenoxels/blender_data.yml \
  --model plenoxels_blender/iter_16000/model.pdparams
```

At the end of the evaluation, the rendering results will be saved in the directory specified by `--model`.

The evaluation script accepts the following arguments and options:

| Arguments & Options | Explanation                                                                                        | Required | Defaults |
|:--------------------|:---------------------------------------------------------------------------------------------------|:---------|:---------|
| config              | Configuration file.                                                                                | YES      | -        |
| model               | Model to be evaluated.                                                                             | YES      | -        |
| ray_batch_size      | Ray batch size.                                                                                    | NO       | 16384    |
| num_workers         | The number of subprocess to load data, `0` for no subprocess used and loading data in main process | NO       | 0        |
