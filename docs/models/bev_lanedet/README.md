# BEV-LaneDet

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
```
@article{wang2022bev,
  title={BEV Lane Det: Fast Lane Detection on BEV Ground},
  author={Wang, Ruihao and Qin, Jian and Li, Kaiying and Cao, Dong},
  journal={arXiv preprint arXiv:2210.06006},
  year={2022}
}

```

## <h2 id="2">简介</h2>

BEV-LaneDet针对3D车道线检测中复杂的空间变换和不够灵活的车道线表示的问题，提出了三点改进：第一，提出了虚拟相机（Virtual Camera）的概念统一了相机内外参；第二，提出了一个简单但有效的3D车道线表示方法（Key-Points Representation）；第三，提出了一个轻量的空间变换模块（Spatial Transformation Pyramid）。

## <h2 id="3">训练配置</h2>

我们提供了在开源数据集上的训练配置与结果，详见[BEV-LaneDet](../../../configs/bev_lanedet)


## <h2 id="4">模型库</h2>
| 模型 |  骨干网络  | F-Score | X error near | X error far | Z error near | Z error far |模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :--------: | :--------: |:--------: | :--------: | :-------------------: | :------: | :-----: | :--: |
|Bev-LaneDet |  ResNet-34    | 97.7 | 0.027 | 0.244 | 0.02 | 0.221 | [model](https://paddle3d.bj.bcebos.com/models/bev_lanedet/bev_lanedet_apollo_576x1024/model.pdparams) | [config](../../../configs/bev_lanedet/bev_lanedet_apollo_576x1024.yml) | [log](https://paddle3d.bj.bcebos.com/models/bev_lanedet/bev_lanedet_apollo_576x1024/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=fbece8dc8a12ad99e4d3d56fa3729c1a) |


## <h2 id="5">使用教程</h2>

## <h2 id="6">数据准备</h2>

请下载Apollo数据集, 下载作者提供的annotion文件。

下载好后的数据集目录结构
```
apollo
   ├── Apollo_Sim_3D_Lane_Release
      ├──depth
      ├──images
      ├──labels
      ├──segmentation
   ├── data_splits
      ├──standard
      ├──rare_subset
      ├──illus_chg
   ...
```


## <h2 id="7">训练</h2>

需要预先下载预训练权重：

```
wget https://paddle3d.bj.bcebos.com/models/bev_lanedet/resnet34-remapped.pdparams
```

运行以下命令，进行单卡训练

```
python tools/train.py --config configs/bev_lanedet/bev_lanedet_apollo_576x1024.yml --save_interval 1  --log_interval 50
```

运行以下命令，进行多卡训练

```
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch tools/train.py configs/bev_lanedet/bev_lanedet_apollo_576x1024.yml --save_interval 1  --log_interval 50 --num_workers 4
```

训练中断，可以通过`--resume`进行继续训练。


## <h2 id="8">评估</h2>

运行以下命令，进行评估

```
python tools/evaluate.py --config configs/bev_lanedet/bev_lanedet_apollo_256x704.yml --model /path/to/your_trained_model
```

## <h2 id="9">导出 & 部署</h2>

### <h3 id="91">模型导出</h3>模型导出

代码开发中
