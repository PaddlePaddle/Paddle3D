# PETR

## 目录
* [引用](#1)
* [简介](#2)
* [训练配置](#3)
* [使用教程](#4)
* [数据准备](#5)
* [训练](#6)
* [评估](#7)
* [导出 & 部署](#8)


## <h2 id="1">引用</h2>

> Liu, Yingfei and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian. "Petr: Position embedding transformation for multi-view 3d object detection." arXiv preprint arXiv:2203.05625, 2022.

## <h2 id="2">简介</h2>

PETR...

## <h2 id="3">训练配置</h2>

我们提供了在开源数据集上的训练配置与结果，详见[PETR训练配置](../../../configs/petr)


## <h2 id="4">模型库</h2>


## <h2 id="5">使用教程</h2>

## <h2 id="6">数据准备</h2>

请下载Nuscenes测数据集, 下载作者提供的annotion文件。

下载好后的数据集目录结构
```
nuscenes
   ├── maps
   ├── samples
   ├── sweeps
   ├── nuscenes_infos_train.pkl
   ├── nuscenes_infos_val.pkl
   ...
```
将nuscenes数据软链至data/nuscenes，或更改配置文件数据集路径。


## <h2 id="7">训练</h2>

需要预先下载预训练权重：

```
wget https://paddle3d.bj.bcebos.com/pretrained/fcos3d_vovnet_imgbackbone-remapped.pdparams
```

运行以下命令，进行单卡训练

```
python tools/train.py --config configs/petr/petr_vovnet_gridmask_p4_800x320.yml --model fcos3d_vovnet_imgbackbone-remapped.pdparams
```

运行以下命令，进行多卡训练

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch tools/train.py --config configs/petr/petr_vovnet_gridmask_p4_800x320.yml --num_workers 2 --log_interval 50 --save_interval 1 --keep_checkpoint_max 100 --save_dir out_petr --model fcos3d_vovnet_imgbackbone-remapped.pdparams
```

训练中断，可以通过`--resume`进行继续训练。


## <h2 id="8">评估</h2>

运行以下命令，进行评估

```
python tools/evaluate.py --config configs/petr/petr_vovnet_gridmask_p4_800x320.yml --model out_petr/epoch_24/model.pdparams
```

## <h2 id="9">导出 & 部署</h2>

### <h3 id="91">模型导出</h3>模型导出

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
TODO
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`caddn.pdiparams `、`caddn.pdiparams.info`和`caddn.pdmodel` |

提供训练好的导出模型
| 配置文件 | 下载 |
| -- | -- |
| PETR | [下载](https://paddle3d.bj.bcebos.com/models/caddn/caddn_ocrnet_hrnet_w18_kitti/model.zip) |
| PETR | [下载](https://paddle3d.bj.bcebos.com/models/caddn/caddn_deeplabv3p_resnet101_os8_kitti/model.zip) |

### C++部署(TODO)

#### Linux系统

#### 环境依赖

- GCC >= 5.4.0
- Cmake >= 3.5.1
- Ubuntu 16.04/18.04

> 说明：本文档的部署环节在以下环境中进行过测试并通过：

测试环境：
- GCC==8.2.0
- Cmake==3.16.0
- Ubuntu 18.04
- CUDA 11.2
- cuDNN==8.1.1
- Paddle Inferece==2.3.1
- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2

#### 编译步骤
