# 快速开始

本文以 instant-ngp 模型和 Blender 数据集为例，介绍如何基于 PaddleRendering 进行模型训练、评估、可视化的全流程操作。其他模型的全流程操作与此一致，各模型详细的使用教程可参考[模型文档](./models)。

## 准备工作

在开始本教程之前，请确保已经按照[安装文档](./installation.md)完成了相关的准备工作。

**数据集**

请将数据集文件软链至 `PaddleRender/data/` 目录下，或在配置文件 `PaddleRender/configs/instant-ngp/blender_data.yml` 中指定数据集路径。

## 模型训练

**开启训练**

位于 `PaddleRendering/` 目录下，执行

```shell
export PYTHONPATH='.'
```

**单卡训练**

```shell
python tools/train.py \
  --config configs/instant-ngp/blender_data.yml \
  --save_dir instant_ngp_blender \
  --log_interval 500 \
  --save_interval 2000
```

**多卡训练**

```shell
python -m paddle.distributed.launch --devices 0,1 \
    tools/train.py \
    --config configs/instant-ngp/blender_data.yml \
    --save_dir instant_ngp_blender \
    --log_interval 500 \
    --save_interval 2000
```

**混合精度训练**

如果想要启动混合精度训练，请参考[配置文件](../configs/instant-ngp/blender_data.yml#L8-#L13)中添加amp的参数项，可用的参数可以参考 API **[paddle.amp.auto_cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html)**。

**训练脚本参数介绍**

| 参数名                         | 用途                                                                                                                                       | 是否必选项 | 默认值        |
|:----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|:------|:-----------|
| config                      | 配置文件                                                                                                                                     | 是     | -          |
| image_batch_size            | 图片 mini-batch 大小，每次迭代从该参数指定数量的图片中采样光线。如果为`-1`，每次迭代从整个数据集的图片中采样光线                                                                         | 否     | -1         |
| ray_batch_size              | 光线 mini-batch 大小，每次迭代从图片 batch 中采样光线的数量                                                                                                  | 否     | 1          |
| image_resampling_interval   | 为提升训练性能，我们使用了图片预加载策略（将图片 mini-batch 预加载到GPU上缓存，每次迭代从缓存的图片中采样光线），该参数用于指定对GPU上图片缓存进行更新的间隔。如果为`-1`，图片缓存不会被更新（适用于 `image_batch_size` 为`-1`的情况） | 否     | -1         |
| use_adaptive_ray_batch_size | 是否采用自动调整光线 mini-batch 大小的训练策略。如果开启，将确保模型每次迭代处理的有效样本数稳定在`2^18`，有助于提升模型收敛效率，缩短训练时间                                                         | 否     | 否          |
| iters                       | 训练迭代数                                                                                                                                    | 否     | 在训练配置文件中指定 |
| learning_rate               | 学习率                                                                                                                                      | 否     | 在训练配置文件中指定 |
| save_dir                    | 模型和 visualdl 日志文件的保存根路径                                                                                                                    | 否     | output     |
| save_interval               | 模型保存的 iteration 间隔数                                                                                                                        | 否     | 1000       |
| do_eval                     | 是否在保存模型时进行评估                                                                                                                             | 否     | 否          |
| resume                      | 是否恢复中断的训练                                                                                                                                | 否     | 否          |
| model                       | 预训练模型路径（`.pdparams`文件）                                                                                                                   | 否     | 不使用预训练模型   |
| log_interval                | 打印日志的间隔步数                                                                                                                                | 否     | 500        |
| keep_checkpoint_max         | 模型保存个数（当保存模型次数超过该限制时，最旧的模型会被自动删除）                                                                                                        | 否     | 5          |

## 训练过程可视化

PaddleRendering 使用 VisualDL 来记录训练过程中的指标和数据，我们可以在训练过程中，在命令行使用 VisualDL 启动一个 server，并在浏览器查看相应的数据。

```shell
# --logdir 需要和训练脚本中指定的 --save_dir 保持一致
# 指定实际 IP 和端口
visualdl --logdir output --host ${HOST_IP} --port {$PORT}
```

![img](https://user-images.githubusercontent.com/95727760/225490725-cbc5e483-1664-49e9-a4ea-0a3e43cdcc90.png)

## 模型评估

当模型训练完成后，需要对训练完成的模型进行指标评估，确保模型的指标满足诉求。您可以使用自己训练的模型，也可通过 wget 命令下载我们提供的[预训练模型](https://paddle3d.bj.bcebos.com/render/models/instant_ngp/instant_ngp_blender/model.pdparams)。请使用以下命令启动评估脚本。

**单卡评估**

```shell
python tools/evaluate.py \
  --config configs/instant-ngp/blender_data.yml \
  --model instant_ngp_blender/iter_20000/model.pdparams
```

**多卡评估**

```shell
python -m paddle.distributed.launch --devices 0,1 \
    tools/evaluate.py \
    --config configs/instant-ngp/blender_data.yml \
    --model instant_ngp_blender/iter_20000/model.pdparams
```

评估完成后，可在 `--model` 对应的目录下查看保存的渲染结果图片。

**评估脚本参数介绍**

| 参数名            | 用途                             | 是否必选项 | 默认值   |
|:---------------|:-------------------------------|:------|:------|
| config         | 配置文件                           | 是     | -     |
| model          | 待评估模型路径                        | 是     | -     |
| ray_batch_size | 光线 mini-batch 大小               | 否     | 16384 |
| num_workers    | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 | 否     | 0     |
