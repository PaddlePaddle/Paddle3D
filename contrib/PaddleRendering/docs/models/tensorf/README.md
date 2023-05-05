# TensoRF: Tensorial Radiance Fields

## 目录

* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练配置](#4)
* [使用教程](#5)
    * [数据准备](#51)
    * [训练](#52)
    * [评估](#53)

## <h2 id="1">引用</h2>

```bibtex
@INPROCEEDINGS{Chen2022ECCV,
  author = {Anpei Chen and Zexiang Xu and Andreas Geiger and Jingyi Yu and Hao Su},
  title = {TensoRF: Tensorial Radiance Fields},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```

## <h2 id="2">简介</h2>

[TensoRF (Tensorial Radiance Fields) ](https://arxiv.org/abs/2203.09517)
是一种新的建模和重建辐射场的方法，与纯粹使用MLP的[NeRF](https://arxiv.org/abs/2003.08934)不同，作者将场景的亮度场建模为4D张量，它代表具有每一个体素多通道特征的3D体素网格，中心思想是将4D场景张量分解成多个紧凑的低秩张量分量。为了进一步提高性能，作者引入了一种新的向量矩阵(VM)分解，该分解放松了张量的两种模式的低秩约束，并将张量分解为紧凑的向量和矩阵因子。实验结果表明，具有VM分解的TensoRF提高了渲染质量，优于以前最先进的方法的同时，减少了重建时间并保留了紧凑的模型大小。

## <h2 id="3">模型库</h2>

- TensoRF 在 Blender Dataset 数据集上的表现

|  场景  |  PSNR   |  SSIM  |                                                 模型下载                                                 |                          配置文件                           |
|:----:|:-------:|:------:|:----------------------------------------------------------------------------------------------------:|:-------------------------------------------------------:|
| lego | 35.960 | 0.9794 | [model](https://paddle3d.bj.bcebos.com/render/models/tensorf/tensorf_blender/model.pdparams) | [config](../../../configs/tensorf/blender_data_vm.yml) |
| lego | 34.877 | 0.9735 | [model](https://paddle3d.bj.bcebos.com/render/models/tensorf/tensorf_sh_blender/model.pdparams) | [sh_config](../../../configs/tensorf/blender_data_vm_sh.yml) |

## <h2 id="4">训练配置</h2>

我们提供了在开源数据集上的训练配置与结果，详见 [TensoRF 训练配置](../../../configs/tensorf), 其中 `blender_data_vm_sh.yml` 采用`spherical harmonics (SH)`球谐函数作为特征解码器，提升渲染重建的速度，减少约 **40%** 的时间。

## <h2 id="5">使用教程</h2>

### <h3 id="51">数据准备</h3>

数据文件软链至`PaddleRender/data/`目录下，或在 [配置文件](../../../configs/tensorf) 中指定数据集路径。

### <h3 id="52">训练</h3>

位于`PaddleRendering/`目录下，执行：

```bash
export PYTHONPATH='.'

# 单卡训练
python tools/train.py \
  --config configs/tensorf/blender_data_vm.yml \
  --save_dir tensorf_blender \
  --log_interval 1000 \
  --save_interval 10000

# 多卡训练 (使用0,1号GPU为例)
python -m paddle.distributed.launch --devices 0,1 \
    tools/train.py \
    --config configs/tensorf/blender_data_vm.yml \
    --save_dir tensorf_blender \
    --log_interval 1000 \
    --save_interval 10000
```

训练脚本支持设置如下参数：

| 参数名                         | 用途                                                                                                                                       | 是否必选项 | 默认值        |
|:----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|:------|:-----------|
| config                      | 配置文件                                                                                                                                     | 是     | -          |
| image_batch_size            | 图片 mini-batch 大小，每次迭代从该参数指定数量的图片中采样光线。如果为`-1`，每次迭代从整个数据集的图片中采样光线                                                                         | 否     | -1         |
| ray_batch_size              | 光线 mini-batch 大小，每次迭代从图片 batch 中采样光线的数量                                                                                                  | 否     | 1          |
| image_resampling_interval   | 为提升训练性能，我们使用了图片预加载策略（将图片 mini-batch 预加载到GPU上缓存，每次迭代从缓存的图片中采样光线），该参数用于指定对GPU上图片缓存进行更新的间隔。如果为`-1`，图片缓存不会被更新（适用于`image_batch_size`为`-1`的情况） | 否     | -1         |
| use_adaptive_ray_batch_size | 是否采用自动调整光线 mini-batch 大小的训练策略。如果开启，将确保模型每次迭代处理的有效样本数稳定在`2^18`，有助于提升模型收敛效率，缩短训练时间                                                         | 否     | 否          |
| iters                       | 训练迭代数                                                                                                                                    | 否     | 在训练配置文件中指定 |
| learning_rate               | 学习率                                                                                                                                      | 否     | 在训练配置文件中指定 |
| save_dir                    | 模型和visualdl日志文件的保存根路径                                                                                                                    | 否     | output     |
| save_interval               | 模型保存的iteration间隔数                                                                                                                        | 否     | 1000       |
| do_eval                     | 是否在保存模型时进行评估                                                                                                                             | 否     | 否          |
| resume                      | 是否恢复中断的训练                                                                                                                                | 否     | 否          |
| model                       | 预训练模型路径（`.pdparams`文件）                                                                                                                   | 否     | 不使用预训练模型   |
| log_interval                | 打印日志的间隔步数                                                                                                                                | 否     | 500        |
| keep_checkpoint_max         | 模型保存个数（当保存模型次数超过该限制时，最旧的模型会被自动删除）                                                                                                        | 否     | 5          |

### <h3 id="53">评估</h3>

位于`PaddleRendering/`目录下，执行：

```shell
export PYTHONPATH='.'
python tools/evaluate.py \
  --config configs/tensorf/blender_data_vm.yml \
  --model tensorf_blender/iter_30000/model.pdparams
```

评估完成后会在`--model`对应的目录下保存渲染结果图片。

评估脚本支持设置如下参数：

| 参数名            | 用途                             | 是否必选项 | 默认值   |
|:---------------|:-------------------------------|:------|:------|
| config         | 配置文件                           | 是     | -     |
| model          | 待评估模型路径                        | 是     | -     |
| ray_batch_size | 光线 mini-batch 大小               | 否     | 16384 |
| num_workers    | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 | 否     | 0     |
