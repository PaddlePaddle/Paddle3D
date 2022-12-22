# CADDN：Categorical Depth DistributionNetwork for Monocular 3D Object Detection

## 目录
* [引用](#1)
* [简介](#2)
* [训练配置](#3)
* [使用教程](#4)
* [数据准备](#5)
* [训练](#6)
* [评估](#7)
* [导出 & 部署](#8)
* [自定义数据集](#9)
* [Apollo使用教程](#10)

## <h2 id="1">引用</h2>

> Cody Reading, Ali Harakeh, Julia Chae, Steven L. Waslander. "Categorical Depth DistributionNetwork for Monocular 3D Object Detection." Computer Vision and Pattern Recognition (CVPR), 2021.

## <h2 id="2">简介</h2>

单目3D物体检测是自动驾驶汽车的关键问题，与典型的多传感器系统相比，单目3D检测提供了一种配置简单的解决方案。单目3D检测的主要挑战在于准确预测物体深度，由于缺乏直接的距离测量，必须从物体和场景线索中推断出物体深度。目前一些方法试图通过直接估计深度来辅助3D检测，但由于深度不准确，性能有限。而CaDDN模型提出了解决方案，它使用每个像素的预测分类深度分布，将丰富的上下文特征信息投射到3D空间中适当的深度区间。然后，CaDDN模型使用计算效率高的鸟瞰投影和单级检测器来生成最终的输出包围框。同时CaDDN模型被设计为一种完全可微的端到端联合深度估计和目标检测方法。在模型发布时，CaDDN在Kitti 3D对象检测基准上获得了已发表的单目方法中的第一名，到目前，CaDDN模型的指标仍具有竞争力。

## <h2 id="3">训练配置</h2>

我们提供了在开源数据集上的训练配置与结果，详见[CADDN训练配置](../../../configs/caddn)


## <h2 id="4">模型库</h2>

| 模型 |  骨干网络  | 3DmAP Mod. | Car<br>Easy Mod. Hard | Pedestrian<br>Easy Mod. Hard | Cyclist<br>Easy Mod. Hard | 模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :--------: | :-------------------: | :--------------------------: | :-----------------------: | :------: | :-----: | :--: |
|CADDN |  ocrnet_hrnet_w18    | 7.86 |  22.50 15.78 13.95 | 10.09 7.12 5.57 | 1.27 0.69 0.69 | [model](https://paddle3d.bj.bcebos.com/models/caddn/caddn_ocrnet_hrnet_w18_kitti/model.pdparams) | [config](../../../configs/caddn/caddn_ocrnet_hrnet_w18_kitti.yml) | [log](https://paddle3d.bj.bcebos.com/models/caddn/caddn_ocrnet_hrnet_w18_kitti/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=36ff3161e13f37bb318fc2d78e679983) |
|CADDN |  deeplabv3p_resnet101_os8    | 7.21 |  21.45 14.36 12.57 | 9.15 6.53 5.12 | 1.82 0.74 0.75 | [model](https://paddle3d.bj.bcebos.com/models/caddn/caddn_deeplabv3p_resnet101_os8_kitti/model.pdparams) | [config](../../../configs/caddn/caddn_deeplabv3p_resnet101_os8_kitti.yml) | [log](https://paddle3d.bj.bcebos.com/models/caddn/caddn_deeplabv3p_resnet101_os8_kitti/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=a56f45325b80ce7f7e29f185efaed28c) |

## <h2 id="5">使用教程</h2>

## <h2 id="6">数据准备</h2>

请下载KITTI单目3D检测数据集，数据集信息请参考[KITTI官网](http://www.cvlibs.net/datasets/kitti/)

*注意：KITTI官网只区分了训练集和测试集，我们遵循业界的普遍做法，将7481个训练集样本，进一步划分为3712个训练集样本和3769个验证集样本*

下载好后的数据集目录结构
```
kttti
   ├── gt_database
   ├── ImageSets
   |      ├── test.txt
   |      ├── train.txt
   |      └── val.txt
   ├── testing
   |      ├── calib
   |      └── image_2
   ├── training
   |      ├── calib
   |      ├── depth_2
   |      ├── image_2
   |      └── label_2
   ├── kitti_infos_test.pkl
   ├── kitti_infos_train.pkl
   ├── kitti_infos_val.pkl
   ...
```
将kitti数据软链至data/kitti，或更改配置文件数据集路径。

备注：准备好kitti数据集后，上述的.pkl是通过下列命令生成
```
python tools/creat_caddn_kitti_infos.py
```
| 参数 | 说明 |
| -- | -- |
| dataset_root | **[选填]** kitti数据集路径，默认data/kitti |
| save_dir | **[选填]** 生成的.pkl文件保存路径，默认data/kitti |

## <h2 id="7">训练</h2>

运行以下命令，进行单卡训练

```
python -u tools/train.py --config configs/caddn/caddn_deeplabv3p_resnet101_os8_kitti.yml
```

运行以下命令，进行多卡训练

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun tools/train.py --config configs/caddn/caddn_deeplabv3p_resnet101_os8_kitti.yml
```

训练中断，可以通过`--resume`进行继续训练。


## <h2 id="8">评估</h2>

运行以下命令，进行评估

```
python tools/evaluate.py --config configs/caddn/caddn_deeplabv3p_resnet101_os8_kitti.yml --model pretrained_model_path
```

## <h2 id="9">导出 & 部署</h2>

### <h3 id="91">模型导出</h3>模型导出

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
python tools/export.py --config configs/caddn/caddn_deeplabv3p_resnet101_os8_kitti.yml --model /path/to/model.pdparams --save_dir /path/to/output
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`caddn.pdiparams `、`caddn.pdiparams.info`和`caddn.pdmodel` |

提供训练好的导出模型
| 配置文件 | 下载 |
| -- | -- |
| caddn_ocrnet_hrnet_w18_kitti | [下载](https://paddle3d.bj.bcebos.com/models/caddn/caddn_ocrnet_hrnet_w18_kitti/model.zip) |
| caddn_deeplabv3p_resnet101_os8_kitti | [下载](https://paddle3d.bj.bcebos.com/models/caddn/caddn_deeplabv3p_resnet101_os8_kitti/model.zip) |

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
- cuDNN==8.1.1
- Paddle Inferece==2.3.1
- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2

#### 编译步骤

**注意：目前CADDN的仅支持使用GPU进行推理。**

- step 1: 进入部署代码所在路径

```
cd deploy/caddn/cpp
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
| model_file | 导出模型的结构文件`caddn.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`caddn.pdiparams`所在路径 |
| image_file | 待预测的图像文件所在路径 |

执行命令：

```
./build/main --model_file /path/to/caddn.pdmodel --params_file /path/to/caddn.pdiparams --image_file /path/to/image.png
```

### 开启TensorRT加速预测【可选】

**注意：请根据编译步骤的step 3，修改`compile.sh`中TensorRT相关的编译参数，并重新编译。**

运行命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`caddn.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`caddn.pdiparams`所在路径 |
| image_file | 待预测的图像文件所在路径 |
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认0。**只需首次运行，下次运行时直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |

* **首次运行TensorRT**，收集模型动态shape信息，并保存至`--dynamic_shape_file`指定的文件中

    ```
    ./build/main --model_file /path/to/caddn.pdmodel --params_file /path/to/caddn.pdiparams --image_file /path/to/image.png --use_trt 1 --collect_shape_info 1 --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP32精度进行预测

    ```
    ./build/main --model_file /path/to/caddn.pdmodel --params_file /path/to/caddn.pdiparams --image_file /path/to/image.png  --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP16精度进行预测

    ```
    ./build/main --model_file /path/to/caddn.pdmodel --params_file /path/to/caddn.pdiparams --image_file /path/to/image.png  --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1
    ```

* 如果觉得每次运行时模型加载的时间过长，可以设置`trt_use_static`和`trt_static_dir`，首次运行时将TensorRT的优化信息保存在硬盘中，后续直接反序列化优化信息即可

    ```
    ./build/main --model_file /path/to/caddn.pdmodel --params_file /path/to/caddn.pdiparams --image_file /path/to/image.png  --use_trt 1 --collect_shape_info 1 --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1 --trt_use_static 1 --trt_static_dir /path/to/OptimCacheDir
    ```

### Python部署

进入部署代码所在路径

```
cd deploy/caddn/python
```

**注意：目前CADDN仅支持使用GPU进行推理。**

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`caddn.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`caddn.pdiparams`所在路径 |
| img_path | 待预测的图像文件所在路径 |
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认0。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |

运行以下命令，执行预测：

```
python infer.py --model_file /path/to/caddn.pdmodel --params_file /path/to/caddn.pdiparams --img_path /path/to/image.png
```

## <h2 id="9">自定义数据集</h2>

## <h2 id="10">Apollo使用教程</h2>

基于Paddle3D训练完成的CADDN模型可以直接部署到Apollo架构中使用，请参考[教程](https://github.com/ApolloAuto/apollo/blob/master/modules/perception/README_paddle3D_CN.md)
