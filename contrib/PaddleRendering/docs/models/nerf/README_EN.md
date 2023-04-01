# NeRF: Scenes as Neural Radiance Fields for View Synthesis

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
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```

## <h2 id="2">Introduction</h2>

NeRF

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

## <h2 id="3">Model Zoo</h2>

- Benchmarks on Blender Dataset.

| Scene |  PSNR   |  SSIM  |                                               Download                                               |                      Configuration                      |
|:-----:|:-------:|:------:|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------:|
| lego | 32.581 | 0.9599 | [model]() | [config](../../../configs/nerf/blender_data.yml) |
| lego | 34.409 | 0.9703 | [model]() | [efficient_config](../../../configs/nerf/blender_data_efficient.yml) |

## <h2 id="4">Training Configuration</h2>

For training configuration on open source datasets, refer
to [NeRF Training Configuration](../../../configs/nerf). Among them, `blender_data_efficient.yml` uses the ray sampling strategy proposed in [Instant-ngp](https://arxiv.org/abs/2201.05989), which speedups the training process by almost 3 times.

## <h2 id="5">Tutorials</h2>

### <h3 id="51">Data Preparation</h3>

Soft link the dataset file to `PaddleRender/data/` or specify the dataset path
in [configuration file](../../../configs/nerf).

### <h3 id="52">Training</h3>

At `PaddleRendering/`, execute:

```shell
export PYTHONPATH='.'

# Train on single GPU
python tools/train.py \
  --config configs/nerf/blender_data.yml \
  --save_dir nerf_blender \
  --log_interval 1000 \
  --save_interval 10000

# Train on multiple GPUs (GPU 0, 1 as an example)
python -m paddle.distributed.launch --devices 0,1 \
    tools/train.py \
    --config configs/nerf/blender_data.yml \
    --save_dir nerf_blender \
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
  --config configs/nerf/blender_data.yml \
  --model nerf_blender/iter_2 00000/model.pdparams
```

At the end of the evaluation, the rendering results will be saved in the directory specified by `--model`.

The evaluation script accepts the following arguments and options:

| Arguments & Options | Explanation                                                                                        | Required | Defaults |
|:--------------------|:---------------------------------------------------------------------------------------------------|:---------|:---------|
| config              | Configuration file.                                                                                | YES      | -        |
| model               | Model to be evaluated.                                                                             | YES      | -        |
| ray_batch_size      | Ray batch size.                                                                                    | NO       | 16384    |
| num_workers         | The number of subprocess to load data, `0` for no subprocess used and loading data in main process | NO       | 0        |
