# CAPE

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

> Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, Xiang Bai. CAPE: Camera View Position Embedding for Multi-View 3D Object Detection. In Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, IEEE, 2023.

## <h2 id="2">简介</h2>

CAPE提出了一种相机视角嵌入信息（CAmera viewPosition Embedding）的方法，通过视角归一化的方式，来降低直接使用3D全局位置嵌入信息来学习图像和3D空间之间的对应关系的难度. 该方法在nuScenes数据集的纯视觉配置上取得了SOTA的表现, 并**中稿CVPR2023**。

算法流程图如下所示：
<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/33575107/224946200-b000fe41-cb4a-4582-b8e1-c0b1083e01f2.png" align="middle" width="980"/>
</p>
</div>

## <h2 id="3">训练配置</h2>

目前，我们提供了在开源数据集nuScenes验证集上的三种训练配置与结果，详见[CAPE训练配置](../../../configs/cape)

## <h2 id="4">模型库</h2>
| 模型 |  骨干网络  | 3DmAP | NDS |  模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :--------: | :-------------------: | :------: | :-----: | :--: |
|cape |  r50    | 34.72 | 40.58 | [model](https://paddle3d.bj.bcebos.com/models/cape/cape_r50_1408x512_epoch_24.pdparams) | [config](../../../configs/cape/cape_r50_1408x512_24ep_wocbgs_imagenet_pretrain.yml) | - |
|capet |  r50    | 31.78 | 44.22 | [model](https://paddle3d.bj.bcebos.com/models/cape/capet_r50_704x256_epoch_24.pdparams) | [config](../../../configs/cape/capet_r50_704x256_24ep_wocbgs_imagenet_pretrain.yml) | - |
|capet |  v99    | 44.72 | 54.36 | [model](https://paddle3d.bj.bcebos.com/models/cape/capet_vov99_800x320_epoch_24.pdparams) | [config](../../../configs/cape/capet_vovnet_800x320_24ep_wocbgs_load_dd3d_pretrain.yml) | - |

## <h2 id="5">可视化</h2>

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/33575107/224949381-8b16bf51-3c15-4aac-8bff-70c65ec53e85.png" align="middle" width="980"/>
</p>
</div>

## <h2 id="6">使用教程</h2>

## <h2 id="7">数据准备</h2>

请下载Nuscenes测数据集, 下载作者提供的annotion文件。

下载好后的数据集目录结构
```
nuscenes
   ├── maps
   ├── samples
   ├── sweeps
   ├── v1.0-trainval
   ├── v1.0-test
   ...
```
将nuscenes数据软链至data/nuscenes，或更改配置文件数据集路径。
运行如下命令生成petr模型所需的annotation文件。

```
python tools/create_petr_nus_infos.py
```
生成完后的数据集目录
```
nuscenes
   ├── maps
   ├── samples
   ├── sweeps
   ├── v1.0-trainval
   ├── v1.0-test
   ├── petr_nuscenes_annotation_train.pkl
   ├── petr_nuscenes_annotation_val.pkl
```
为了方便，我们提供了生成好的annotation文件
| 文件名称 | 下载链接 |
| -- | -- |
| petr_nuscenes_annotation_train.pkl | [下载](https://paddle3d.bj.bcebos.com/datasets/nuScenes/petr_nuscenes_annotation_train.pkl) |
| petr_nuscenes_annotation_val.pkl | [下载](https://paddle3d.bj.bcebos.com/datasets/nuScenes/petr_nuscenes_annotation_val.pkl) |

## <h2 id="8">训练</h2>

todo


## <h2 id="9">评估</h2>

运行以下命令，进行评估

```
python tools/evaluate.py --config configs/cape/capet_vovnet_800x320_24ep_wocbgs_load_dd3d_pretrain.yml --model /path/to/your/capet_vov99_800x320_epoch_24.pdparams
```
