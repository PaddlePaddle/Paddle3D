# DD3D: Is Pseudo-Lidar needed for Monocular 3D Object detection?

## 目录
* [引用](#1)
* [简介](#2)
* [训练配置](#3)
* [使用教程](#4)
* [数据准备](#5)
* [训练](#6)
* [评估](#7)

## <h2 id="1">引用</h2>

> Dennis Park and Rares Ambrus and Vitor Guizilini and Jie Li and Adrien Gaidon. "Is Pseudo-Lidar needed for Monocular 3D Object detection?" IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

## <h2 id="2">简介</h2>

DD3D是一个端到端single-stage单目相机目标检测模型，融合了PL方法的优势(scaling with depth pre-training) 和end-to-end方法的优势(simplicity and generalization performance)，训练过程简单，只包含depth pre-training和detection fine-tuning。在模型发布时，DD3D是NuScenes(nuscenes.org/object-det)单目3D检测排名第一的工作(截止2021.9.3)。

## <h2 id="3">训练配置</h2>

我们提供了在开源数据集KITTI上的训练配置与结果，详见[DD3D训练配置](../../../configs/dd3d)

## <h2 id="4">模型库</h2>

| 模型 |  骨干网络  | Car<br>Easy Mod. Hard | 模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :-------------------: | :------: | :-----: | :--: |
|DD3D |  dla_34 |  23.49 17.57 15.21 | [model](https://paddle3d.bj.bcebos.com/models/dd3d/dd3d_dla_34_kitti/model.pdparams) | [config](../../../configs/dd3d/dd3d_dla_34_kitti.yml) | [log](https://paddle3d.bj.bcebos.com/models/dd3d/dd3d_dla_34_kitti/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=9862e660790ca627384dded9e1cd0a50) |
|DD3D |  v2_99 |  29.17 23.42 20.73 | [model](https://paddle3d.bj.bcebos.com/models/dd3d/dd3d_v2_99_kitti/model.pdparams) | [config](../../../configs/dd3d/dd3d_v2_99_kitti.yml) | [log](https://paddle3d.bj.bcebos.com/models/dd3d/dd3d_v2_99_kitti/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=52a9cd89f47b4c91f95bae558323f07c) |


## <h2 id="5">使用教程</h2>

## <h2 id="6">数据准备</h2>

请下载KITTI单目3D检测数据集，数据集信息请参考[KITTI官网](http://www.cvlibs.net/datasets/kitti/)

*注意：KITTI官网只区分了训练集和测试集，我们遵循业界的普遍做法，将7481个训练集样本，进一步划分为3712个训练集样本和3769个验证集样本*

下载好后的数据集目录结构
```
kttti
   ├── ImageSets
   |      ├── test.txt
   |      ├── train.txt
   |      └── val.txt
   ├── testing
   |      ├── calib
   |      └── image_2
   ├── training
          ├── calib
          ├── depth_2
          ├── image_2
          └── label_2

   ...
```
将kitti数据软链至datasets/KITTI，或更改配置文件数据集路径。

## <h2 id="7">训练</h2>

单卡训练，先运行以下命令，进行warmup

```
python -u tools/train.py --config configs/dd3d/dd3d_dla_34_kitti_warmup.yml
```

然后进行训练

```
python -u tools/train.py --config configs/dd3d/dd3d_dla_34_kitti.yml --resume
```

多卡训练，先运行以下命令，进行warmup

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun tools/train.py --config configs/dd3d/dd3d_dla_34_kitti_warmup.yml
```

然后进行训练

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun tools/train.py --config configs/dd3d/dd3d_dla_34_kitti.yml --resume
```

训练中断，可以通过`--resume`进行继续训练。


## <h2 id="8">评估</h2>

运行以下命令，进行评估

```
python tools/evaluate.py --config configs/dd3d/dd3d_dla_34_kitti.yml --model pretrained_model_path
```
