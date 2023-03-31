# Quick Start

This article takes the instant-ngp model and the Blender dataset as an example to introduce how to perform the whole process of model training, evaluation, and visualization based on PaddleRendering. The process of other models is similar. For detailed tutorials of each model, please refer to [Model Documentation](./models).

## Preparation

Before starting this tutorial, please make sure you have completed the relevant preparations according to [Installation Documentation](./installation_EN.md).

**Dataset**

Please soft link the dataset file to the `PaddleRender/data/` directory, or specify the dataset path in the configuration file `PaddleRender/configs/instant-ngp/blender_data.yml`.

## Model Training

**Start Training**

At the `PaddleRendering/` directory, execute

```shell
export PYTHONPATH='.'
```

**Single GPU Training**

```shell
python tools/train.py \
  --config configs/instant-ngp/blender_data.yml \
  --save_dir instant_ngp_blender \
  --log_interval 500 \
  --save_interval 2000
```

**Multi-GPUs Training**

```shell
python -m paddle.distributed.launch --devices 0,1 \
    tools/train.py \
    --config configs/instant-ngp/blender_data.yml \
    --save_dir instant_ngp_blender \
    --log_interval 500 \
    --save_interval 2000
```

**Automatic Mixed Precision Training**

To train with automatic mixed precision enabled, please refer to the amp parameter item added in [configuration file](../configs/instant-ngp/blender_data.yml#L8-#L13). Available parameters can refer to API **[paddle .amp.auto_cast](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/amp/auto_cast_en.html)**.

**Introduction to Training Script Parameters**

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


## Visualization of the Training Process

PaddleRendering integrates VisualDL to provide the measurements and visualizations needed during the training process. We can use VisualDL to start a server on the command line during the training process and view the corresponding data in the browser.

```shell
# --logdir needs to be consistent with the --save_dir specified in the training script
# Specify actual IP and port
visualdl --logdir output --host ${HOST_IP} --port {$PORT}
```

![img](https://user-images.githubusercontent.com/95727760/225497088-244c8c06-5b2c-4dc8-beea-c6df1e8a7d2f.png)

## Model Evaluation

After the model training is completed, it is necessary to validate the model. You can use the model trained by yourself, or download the [pre-trained model](https://paddle3d.bj.bcebos.com/render/models/instant_ngp/instant_ngp_blender/model.pdparams) provided by us through the wget command. Please use the following command to start the evaluation script.

**Single GPU Evaluation**

```shell
python tools/evaluate.py \
  --config configs/instant-ngp/blender_data.yml \
  --model instant_ngp_blender/iter_20000/model.pdparams
```

**Multi-GPUs Evaluation**

```shell
python -m paddle.distributed.launch --devices 0,1 \
    tools/evaluate.py \
    --config configs/instant-ngp/blender_data.yml \
    --model instant_ngp_blender/iter_20000/model.pdparams
```

After the evaluation is completed, you can view the saved rendering result in the directory corresponding to `--model`.

**Evaluation Script Parameters Introduction**

| Arguments & Options | Explanation                                                                                        | Required | Defaults |
|:--------------------|:---------------------------------------------------------------------------------------------------|:---------|:---------|
| config              | Configuration file.                                                                                | YES      | -        |
| model               | Model to be evaluated.                                                                             | YES      | -        |
| ray_batch_size      | Ray batch size.                                                                                    | NO       | 16384    |
| num_workers         | The number of subprocess to load data, `0` for no subprocess used and loading data in main process | NO       | 0        |
