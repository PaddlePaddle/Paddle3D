# BEVDet

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
@article{huang2022bevpoolv2,
  title={BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2211.17111},
  year={2022}
}

@article{huang2022bevdet4d,
  title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2203.17054},
  year={2022}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
```

## <h2 id="2">简介</h2>

BEVDet设计了一个采用环视2D图像在BEV视角下进行3D目标检测的感知框架，包含环视图像编码器，BEV视角转换，BEV编码器，检测头等模块。其中bev视角转换采用基于LSS自底向上的方式，通过预测图像像素的深度信息，结合相机内外参来显式的生成图像BEV特征。BEVDet2.0中引入了深度信息监督，进一步提升了模型精度，在nuscenes数据集上NDS可以达到52.3，其中引入的BEVPoolv2相较之前的版本在速度上也有约15.1倍的大幅提升。

## <h2 id="3">训练配置</h2>

我们提供了在开源数据集上的训练配置与结果，详见[BEVDet训练配置](../../../configs/bevdet)


## <h2 id="4">模型库</h2>
| 模型 |  骨干网络  | 3DmAP | NDS |  模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :--------: | :-------------------: | :------: | :-----: | :--: |
|BEVDet2.0 |  ResNet-50    | 37.36 | 47.78 | [model](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/model_ema.pdparams) | [config](../../../configs/bevdet/bevdet4d_r50_depth_nuscenes.yml) | [log](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/training.log) \| [vdl](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/vdlrecords.1681122301.log) |


## <h2 id="5">使用教程</h2>

## <h2 id="6">数据准备</h2>

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

数据集标注打包请参考原版BEVDet实现中的数据打包脚本[tools/create_data_bevdet.py](https://github.com/HuangJunJie2017/BEVDet/blob/dev2.1/tools/create_data_bevdet.py)

生成完后的数据集目录
```
nuscenes
   ├── maps
   ├── samples
   ├── sweeps
   ├── v1.0-trainval
   ├── v1.0-test
   ├── bevdetv2-nuscenes_annotation_train.pkl
   ├── bevdetv2-nuscenes_annotation_val.pkl
```
为了方便，我们提供了生成好的annotation文件
| 文件名称 | 下载链接 |
| -- | -- |
| bevdetv2-nuscenes_annotation_train.pkl | [下载](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/dataset/bevdetv2-nuscenes_infos_train.pkl) |
| bevdetv2-nuscenes_annotation_val.pkl | [下载](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/dataset/bevdetv2-nuscenes_infos_val.pkl) |

## <h2 id="7">训练</h2>

需要预先下载预训练权重：

```
wget https://paddle3d.bj.bcebos.com/pretrained/r50_remapped.pdparams
```

运行以下命令，进行单卡训练

```
python tools/train.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --model r50_remapped.pdparams
```

运行以下命令，进行多卡训练

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch tools/train.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --num_workers 2 --log_interval 50 --save_interval 1 --keep_checkpoint_max 100 --save_dir out_bevdet --model r50_remapped.pdparams
```

训练中断，可以通过`--resume`进行继续训练。


## <h2 id="8">评估</h2>

运行以下命令，进行评估

```
python tools/evaluate.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --model out_bevdet/epoch_20/model_ema.pdparams
```

## <h2 id="9">导出 & 部署</h2>

### <h3 id="91">模型导出</h3>模型导出

代码开发中
