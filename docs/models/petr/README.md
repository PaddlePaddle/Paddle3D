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

PETRv1是一个位置嵌入信息感知的多视角3D视觉检测算法。PETRv1将3D坐标信息与图像特征相融合，借助transfomer的结构实现端到端的3D目标检测，PETRv1在比较简洁的架构上达到了精度SOTA（50.4 NDS, 44.1mAP），并且在一段时间内在NuScenes数据集上排名第一。

PETRv2在v1的基础上加入了时序信息，致力于构建一个统一的多视角感知框架。 PETRv2扩展了3D位置信息嵌入模块（3D PE），实现不同时刻帧之间的对齐。并在这个基础上加入了特征指导编码器来提高3D位置信息嵌入模块的数据自适应能力。PETRv2以一个简洁而有效的框架在3D目标检测，BEV分割和3D车道线检测等任务上都取得了SOTA的效果。

## <h2 id="3">训练配置</h2>

我们提供了在开源数据集上的训练配置与结果，详见[PETR训练配置](../../../configs/petr)


## <h2 id="4">模型库</h2>
### 检测模型
| 模型 |  骨干网络  | 3DmAP | NDS |  模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :--------: | :-------------------: | :------: | :-----: | :--: |
|PETR v1 |  v99    | 38.35 | 43.52 | [model](https://paddle3d.bj.bcebos.com/models/petr/petr_vovnet_gridmask_p4_800x320_amp/model.pdparams) | [config](../../../configs/petr/petr_vovnet_gridmask_p4_800x320_amp.yml) | [log](https://paddle3d.bj.bcebos.com/models/petr/petr_vovnet_gridmask_p4_800x320_amp/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=334e6a6ba257c953fe67bac17a1434a6) |
|PETR v2 |  v99    | 41.05 | 49.86 | [model](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_800x320/model.pdparams) | [config](../../../configs/petr/petrv2_vovnet_gridmask_p4_800x320.yml) | [log](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_800x320/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=8888769be49447d6bbabebe78a5fa3ed) |
|PETR v2 + denoise |  v99    | 41.35 | 50.64 | [model](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_800x320_dn_amp/model.pdparams) | [config](../../../configs/petr/petrv2_vovnet_gridmask_p4_800x320_dn_amp.yml) | [log](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_800x320_dn_amp/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=3b3951b08caf6367f469edf7f3863e2b) |
|PETR v2 + denoise + centerview |  v99    | 43.45 | 52.24 | [model](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_800x320_dn_centerview_amp/model.pdparams) | [config](../../../configs/petr/petrv2_vovnet_gridmask_p4_800x320_dn_centerview_amp.yml) | [log](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_800x320_dn_centerview_amp/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=41e8354ab1faef44ab05850e1b4d5383) |
|PETR v2 + denoise + multiview |  v99    | 44.91 | 53.34 | [model](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_1600x640_dn_multiscale_amp/model.pdparams) | [config](../../../configs/petr/petrv2_vovnet_gridmask_p4_1600x640_dn_multiscale_amp.yml) | [log](https://paddle3d.bj.bcebos.com/models/petr/petrv2_vovnet_gridmask_p4_1600x640_dn_multiscale_amp/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=ed5c16888449914ddde4f9554c6edeac) |

### 分割模型
| 模型 | 骨干网络 | Drive | Lane|  Vehicle  | 模型下载 | 配置文件 |  日志 |
|:--------:|:----------:|:---------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| PETR v2 BEVseg   | v99    | 79.0%     | 44.8%   | 49.4%     |  [model](https://paddle3d.bj.bcebos.com/models/petr/petrv2_BEVseg_800x320_amp/model.pdparams) | [config](../../../configs/petr/petrv2_BEVseg_800x320_amp.yml) | [log](https://paddle3d.bj.bcebos.com/models/petr/petrv2_BEVseg_800x320_amp/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=c2d6249c8c791dd79d17c27da7a80621) |
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

如果需要运行分割模型，需要从Nuscenes官网下载`Map expansion`数据集，解压到`maps`文件夹下，并下载[HDmaps-nocovers annotion文件](https://paddle3d.bj.bcebos.com/datasets/nuScenes/HDmaps-nocovers.zip)。

## <h2 id="7">训练</h2>

需要预先下载预训练权重：

```
wget https://paddle3d.bj.bcebos.com/pretrained/fcos3d_vovnet_imgbackbone-remapped.pdparams
```

设置以下环境变量，在backbone阶段使用NHWC的data_format，加快训练速度

```
export FLAGS_opt_layout=True
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
python tools/export.py --config configs/petr/petr_vovnet_gridmask_p4_800x320.yml --model /path/to/model.pdparams --save_dir /path/to/output
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`petr.pdiparams `、`petr.pdiparams.info`和`petr.pdmodel` |

提供训练好的导出模型
| 配置文件 | 下载 |
| -- | -- |
| PETR v1 | [下载](https://paddle3d.bj.bcebos.com/models/petr/petr_exported_model.tar) |
| PETR v2 | [下载](https://paddle3d.bj.bcebos.com/models/petr/petrv2_exported_model.tar) |


### C++部署

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
- cuDNN==8.2
- Paddle Inferece==2.4.0rc0
- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2

#### 编译步骤

**注意：目前CADDN的仅支持使用GPU进行推理。**

- step 1: 进入部署代码所在路径

```
cd deploy/petr/cpp
```

- step 2: 下载Paddle Inference C++预编译库

Paddle Inference针对**是否使用GPU**、**是否支持TensorRT**、以及**不同的CUDA/cuDNN/GCC版本**均提供已经编译好的库文件，请至[Paddle Inference C++预编译库下载列表](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#c)选择符合的版本。

- step 3: 修改`compile.sh`中的编译参数

主要修改编译脚本`compile.sh`中的以下参数：

| 参数 | 说明 |
| -- | -- |
| WITH_GPU | 是否使用gpu。ON或OFF， OFF表示使用CPU，默认ON|
| USE_TENSORRT | 是否使用TensorRT加速。ON或OFF，默认OFF|
| LIB_DIR | Paddle Inference C++预编译包所在路径，该路径下的内容应有：`CMakeCache.txt`、`paddle`、`third_party`和`version.txt` |
| CUDNN_LIB | cuDNN`libcudnn.so`所在路径 |
| CUDA_LIB | CUDA`libcudart.so `所在路径 |
| TENSORRT_ROOT | TensorRT所在路径。**非必须**，如果`USE_TENSORRT`设置为`ON`时，需要填写该路径，该路径下的内容应有`bin`、`lib`和`include`等|

- step 4: 开始编译

```
sh compile.sh
```

### 执行预测

**注意：目前CADDN的仅支持使用GPU进行推理。**

执行命令参数说明

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`petr.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`petr.pdiparams`所在路径 |
| image_files | 待预测的图像文件路径列表，每个文件用逗号分开 |
| with_timestamp | 是否需要时间戳，为True时表示运行petrv2的模型 |

执行命令：

```
# petrv1
./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png

# petrv2
./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png,/path/to/img0_pre.png,/path/to/img1_pre.png,/path/to/img2_pre.png,/path/to/img3_pre.png,/path/to/img4_pre.png,/path/to/img5_pre.png --with_timestamp
```

### 开启TensorRT加速预测【可选】

**注意：请根据编译步骤的step 3，修改`compile.sh`中TensorRT相关的编译参数，并重新编译。**

运行命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`petr.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`petr.pdiparams`所在路径 |
| image_files | 待预测的图像文件路径列表，每个文件用逗号分开  |
| use_trt | 是否使用TensorRT进行加速，默认false|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为true时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认false |
| trt_static_dir | 当trt_use_static设置为true时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认false。**只需首次运行，下次运行时直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 默认:petr_shape_info.txt|
| with_timestamp | 是否需要时间戳，为True时表示运行petrv2的模型 |

* **首次运行TensorRT**，收集模型动态shape信息，并保存至`--dynamic_shape_file`指定的文件中

    ```
    # petrv1
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png --use_trt --collect_shape_info --dynamic_shape_file /path/to/shape_info.txt

    # petrv2
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png,/path/to/img0_pre.png,/path/to/img1_pre.png,/path/to/img2_pre.png,/path/to/img3_pre.png,/path/to/img4_pre.png,/path/to/img5_pre.png --use_trt --collect_shape_info --dynamic_shape_file /path/to/shape_info.txt --with_timestamp
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP32精度进行预测

    ```
    # petrv1
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png  --use_trt --dynamic_shape_file /path/to/shape_info.txt

    # petrv2
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png,/path/to/img0_pre.png,/path/to/img1_pre.png,/path/to/img2_pre.png,/path/to/img3_pre.png,/path/to/img4_pre.png,/path/to/img5_pre.png --use_trt --dynamic_shape_file /path/to/shape_info.txt --with_timestamp
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP16精度进行预测

    ```
    # petrv1
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png  --use_trt --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1

    # petrv2
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png,/path/to/img0_pre.png,/path/to/img1_pre.png,/path/to/img2_pre.png,/path/to/img3_pre.png,/path/to/img4_pre.png,/path/to/img5_pre.png --use_trt --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1 --with_timestamp
    ```

* 如果觉得每次运行时模型加载的时间过长，可以设置`trt_use_static`和`trt_static_dir`，首次运行时将TensorRT的优化信息保存在硬盘中，后续直接反序列化优化信息即可

    ```
    # petrv1
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png  --use_trt --collect_shape_info --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1 --trt_use_static --trt_static_dir /path/to/OptimCacheDir

    # petrv2
    ./build/main --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --image_files /path/to/img0.png,/path/to/img1.png,/path/to/img2.png,/path/to/img3.png,/path/to/img4.png,/path/to/img5.png,/path/to/img0_pre.png,/path/to/img1_pre.png,/path/to/img2_pre.png,/path/to/img3_pre.png,/path/to/img4_pre.png,/path/to/img5_pre.png --use_trt --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1 --with_timestamp --trt_use_static --trt_static_dir /path/to/OptimCacheDir
    ```

### Python部署

进入部署代码所在路径

```
cd deploy/petr/python
```

**注意：目前CADDN仅支持使用GPU进行推理。**

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`petr.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`petr.pdiparams`所在路径 |
| img_paths | 待预测的图像文件路径列表，每个文件用逗号分开  |
| use_trt | 是否使用TensorRT进行加速，默认False|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为True时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认False。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |
| with_timestamp | 是否需要时间戳，为True时表示运行petrv2的模型 |

运行以下命令，执行预测：

```
# petrv1
python infer.py --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --img_paths /path/to/img0.png /path/to/img1.png /path/to/img2.png /path/to/img3.png /path/to/img4.png /path/to/img5.png

# petrv2
python infer.py --model_file /path/to/petr.pdmodel --params_file /path/to/petr.pdiparams --img_paths /path/to/img0.png /path/to/img1.png /path/to/img2.png /path/to/img3.png /path/to/img4.png /path/to/img5.png /path/to/img0_pre.png /path/to/img1_pre.png /path/to/img2_pre.png /path/to/img3_pre.png /path/to/img4_pre.png /path/to/img5_pre.png
```
