# 快速开始

本文以SMOKE模型和KITTI数据集为例，介绍如何基于Paddle3D进行模型训练、评估、可视化的全流程操作。其他模型的全流程操作与此一致，各模型详细的使用教程和benchmark可参考[模型文档](./models)。

## 准备工作

在开始本教程之前，请确保已经按照 [安装文档](./installation.md) 完成了相关的准备工作

<br>

## 模型训练

**单卡训练**

使用如下命令启动单卡训练，由于一次完整的训练流程耗时较久，我们只训练100个iter进行快速体验，下面的命令在Telsa V100上大约耗时2分钟

```shell
python tools/train.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --iters 100 --log_interval 10 --save_interval 20
```

**多卡训练**

很多3D感知模型需要使用多卡并行进行训练，Paddle3D同样支持快捷地启动多卡训练，使用如下命令可以启动四卡并行训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun tools/train.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --iters 100 --log_interval 10 --save_interval 20
```

**混合精度训练**

如果想要启动混合精度训练，请参考[配置文件](../configs/smoke/smoke_dla34_no_dcn_kitti_amp.yml#L6-#L11)中添加amp的参数项，可用的参数可以参考 API **[paddle.amp.auto_cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/amp/auto_cast_cn.html#paddle.amp.auto_cast)**

**训练脚本参数介绍**

| 参数名              | 用途                                                         | 是否必选项  | 默认值            |
| :------------------ | :----------------------------------------------------------- | :--------- | :--------------- |
| iters               | 训练迭代步数                                                  | 否         | 配置文件中指定值  |
| epochs               | 训练迭代次数                                                  | 否         | 配置文件中指定值  |
| batch_size          | 单卡batch size                                               | 否         | 配置文件中指定值   |
| learning_rate       | 初始学习率                                                    | 否         | 配置文件中指定值  |
| config              | 配置文件路径                                                  | 是         | -                |
| save_dir            | 检查点（模型和visualdl日志文件）的保存根路径                    | 否         | output           |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据        | 否         | 2                |
| save_interval       | 模型保存的间隔步数                                            | 否          | 1000             |
| do_eval             | 是否在保存模型时启动评估                                       | 否         | 否               |
| log_interval        | 打印日志的间隔步数                                            | 否          | 10               |
| resume              | 是否从检查点中恢复训练状态                | 否          | None             |
| keep_checkpoint_max | 最多保存模型的数量                                              | 否          | 5                |
| quant_config        | 量化配置文件，一般放在[configs/quant](../configs/quant)目录下       | 否         | None              |
| seed                | Paddle/numpy/random的全局随机种子值                                                    | 否         | None              |

*注意：使用一个 batch 数据对模型进行一次参数更新的过程称之为一步，iters 即为训练过程中的训练步数。完整遍历一次数据对模型进行训练的过程称之为一次迭代，epochs 即为训练过程中的训练迭代次数。一个epoch包含多个iter。*

<br>

## 训练过程可视化

Paddle3D使用VisualDL来记录训练过程中的指标和数据，我们可以在训练过程中，在命令行使用VisualDL启动一个server，并在浏览器查看相应的数据

```shell
# logdir需要和训练脚本中指定的save_dir保持一致
# 指定实际IP和端口
visualdl --logdir output --host ${HOST_IP} --port {$PORT}
```

![img](https://user-images.githubusercontent.com/45024560/177712952-91c8d6bd-218f-4722-9f5f-e0b763cfa0ea.JPG)

<br>

## 模型量化（可选）

为了导出量化的模型，我们可以对模型进行量化训练，量化后的模型可以使用TensorRT + int8进行推理，从而提升推理速度，使用如下命令启动量化训练。

以多卡训练为例子，使用如下命令启动多卡量化训练，同样只训练100个iter进行快速体验

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 注意这是一次新的训练，需要指定加载已经训练好的模型参数进行微调
# 并且指定新的模型保存路径
fleetrun tools/train.py \
    --config configs/smoke/smoke_dla34_no_dcn_kitti.yml \
    --iters 100 \
    --log_interval 10 \
    --save_interval 20 \
    --quant_config configs/quant/smoke_kitti.yml \
    --model output/iter_100/model.pdparams \
    --save_dir output_smoke_quant
```

*注意，不同的模型需要探索不同的量化训练配置（如重新训练的次数，学习率衰减等），我们提供了 **SMOKE** 和 **CenterPoint** 的配置文件供参考*

<br>

## 模型评估

**单卡评估**

当模型训练完成后，需要对训练完成的模型进行指标评估，确保模型的指标满足诉求。目前Paddle3D的模型只支持单卡评估，使用以下命令启动评估脚本

```shell
wget https://paddle3d.bj.bcebos.com/models/smoke/smoke_dla34_no_dcn_kitti/model.pdparams
python tools/evaluate.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --model model.pdparams
```

**评估脚本参数介绍**

| 参数名              | 用途                                                         | 是否必选项  | 默认值            |
| :------------------ | :----------------------------------------------------------- | :--------- | :--------------- |
| batch_size          | 评估时的batch size                                            | 否         | 配置文件中指定值   |
| config              | 配置文件路径                                                  | 是         | -                |
| model               | 模型参数路径                                                  | 否         | -                |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据        | 否         | 2                |
| quant_config        | 量化配置文件，一般放在[configs/quant](../configs/quant)目录下，如果模型使用量化训练，则在评估时同样需要指定量化配置文件  | 否         |

<br>

## 模型导出

当完成模型训练后，我们需要将模型导出成推理格式进行部署，我们加载Paddle3D已经训练完成的SMOKE模型参数进行模型导出

```shell
wget https://paddle3d.bj.bcebos.com/models/smoke/smoke_dla34_no_dcn_kitti/model.pdparams
python tools/export.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --model model.pdparams
```

**导出脚本参数介绍**

| 参数名              | 用途                                                                 | 是否必选项  | 默认值                 |
| :------------------ | :------------------------------------------------------------------ | :--------- | :--------------------- |
| config              | 配置文件路径                                                         | 是         | -                      |
| model               | 模型参数路径                                                         | 否         | -                      |
| export_for_apollo   | 是否用于Apollo部署，当打开该开关时，会同步生成用于Apollo部署的meta文件   | 否         | False                  |
| save_dir            | 推理模型文件的保存路径                                                | 否         | exported_model         |
| save_name           | 推理模型文件的保存名字                                                | 否         | None(由各模型自定决定)   |
| quant_config        | 量化配置文件，一般放在[configs/quant](../configs/quant)目录下，如果模型使用量化训练，则在模型导出时同样需要指定量化配置文件  | 否         |

<br>

## 模型部署

请根据实际模型选择对应的[部署文档](./models)进行参照
