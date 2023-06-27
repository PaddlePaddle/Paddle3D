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

### <h3 id="91">模型导出</h3>


运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
FLAGS_deploy=True python tools/export.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --model /Path/to/model.pdparams --save_dir ./output_bevdet_inference
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`model.pdiparams `、`model.pdiparams.info`和`model.pdmodel` |

### <h3 id="92">模型部署</h3>

### Python部署

部署代码所在路径为deploy/bevdet/python

**注意：目前bevdet仅支持使用GPU进行推理。**

### 使用paddle 推理nuscenes eval
命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| config | 配置文件的所在路径  |
| batch_size | eval推理的batch_size，默认1 |
| model | 预训练模型参数所在路径  |
| num_workers | eval推理的num_workers，默认2  |
| model_file | 导出模型的结构文件`rtebev.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`rtebev.pdiparams`所在路径 |
| use_trt | 是否使用TensorRT进行加速，默认False|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为True时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认False。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |
| quant_config | 量化配置文件的所在路径  |

使用paddle inference，执行预测：

```
python deploy/bevdet/python/infer_evaluate.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --model /path/to/model.pdparams --model_file /path/to/bevdet.pdmodel --params_file /path/to/bevdet.pdiparams
```

使用paddle trt，执行预测：

收集shape
```
python deploy/bevdet/python/infer_evaluate.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --model /path/to/model.pdparams --model_file /path/to/bevdet.pdmodel --params_file /path/to/bevdet.pdiparams --use_trt --collect_shape_info
```
trt-fp16推理
```
python deploy/bevdet/python/infer_evaluate.py --config configs/bevdet/bevdet4d_r50_depth_nuscenes.yml --model /path/to/model.pdparams --model_file /path/to/bevdet.pdmodel --params_file /path/to/bevdet.pdiparams --use_trt --trt_precision 1
```
