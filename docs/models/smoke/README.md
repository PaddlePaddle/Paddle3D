# SMOKE：Single-Stage Monocular 3D Object Detection via Keypoint Estimation

## 目录
* [引用](#引用)
* [简介](#简介)
* [训练配置](#训练配置)
* [使用教程](#使用教程)
* [数据准备](#数据准备)
* [训练](#训练)
* [评估](#评估)
* [导出部署](#导出部署)
* [自定义数据集](#自定义数据集)

<br>

## 引用

> Liu, Zechen, Zizhang Wu, and Roland Tóth. "Smoke: Single-stage monocular 3d object detection via keypoint estimation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 996-997. 2020.

<br>

## 简介

SMOKE是一个单阶段的单目3D检测模型，该论文创新性地提出了预测物体中心点投影来间接预测物体3D检测框的方法。我们参照了Apollo项目对于该模型的[修改](https://github.com/ApolloAuto/apollo/tree/master/modules/perception/camera#architecture)：

* 使用普通卷积替代了原论文中使用的可形变卷积

* 添加了一个头部来预测 2D 中心点和 3D 中心点之间的偏移

* 添加了另一个头部来预测 2D 边界框的宽度和高度。可以通过预测的二维中心、宽度和高度直接获得二维边界框

<br>

## 模型库

| 模型 |  骨干网络  | 3DmAP Mod. | Car<br>Easy Mod. Hard | Pedestrian<br>Easy Mod. Hard | Cyclist<br>Easy Mod. Hard | 模型下载 | 配置文件 |  日志 |
| :--: | :-------: | :--------: | :-------------------: | :--------------------------: | :-----------------------: | :------: | :-----: | :--: |
|SMOKE |  DLA34    | 2.94 |  6.26 5.16 4.54 | 3.04 2.73 2.23 | 1.69 0.95 0.94 | [model](https://bj.bcebos.com/paddle3d/models/smoke/smoke_dla34_no_dcn_kitti/model.pdparams) | [config](../../../configs/smoke/smoke_dla34_no_dcn_kitti.yml) | [log](https://bj.bcebos.com/paddle3d/models/smoke/smoke_dla34_no_dcn_kitti/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=1650ec346b4426486bd079b506fc1f86) |
|SMOKE |  HRNet18  | 4.05 | 8.48 6.44 5.74 | 5.02 4.23 3.06 | 2.59 1.49 1.37 | [model](https://bj.bcebos.com/paddle3d/models/smoke/smoke_hrnet18_no_dcn_kitti/model.pdparams) | [config](../../../configs/smoke/smoke_hrnet18_no_dcn_kitti.yml) | [log](https://bj.bcebos.com/paddle3d/models/smoke/smoke_hrnet18_no_dcn_kitti/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=4e31655b33d0f44b0c19399df8fb7b00) |

**注意：** KITTI benchmark使用4张V100 GPU训练得出。

<br>

## 使用教程

下面的教程将从数据准备开始，说明如何训练SMOKE模型

### 数据准备

目前Paddle3D中提供的SMOKE模型支持在KITTI数据集上训练，因此需要先准备KITTI数据集，请在[官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)进行下载：

1. left color images of object data set (12 GB)

2. training labels of object data set (5 MB)

3. camera calibration matrices of object data set (16 MB)

并下载数据集的划分文件列表：

```shell
wget https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz
```

将数据解压后按照下方的目录结构进行组织

```shell
$ tree KITTI
KITTI
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   └── val.txt
└── training
    ├── calib
    ├── image_2
    └── label_2
```

在Paddle3D的目录下创建软链接 `datasets/KITTI`，指向到上面的数据集目录

### 训练

使用如下命令启动4卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 每隔50步打印一次训练进度
# 每隔5000步保存一次模型，模型参数将被保存在output目录下
fleetrun tools/train.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --num_workers 2 --log_interval 50 --save_interval 5000
```

### 评估

使用如下命令启动评估

```shell
export CUDA_VISIBLE_DEVICES=0

# 使用Paddle3D提供的预训练模型进行评估
python tools/evaluate.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --num_workers 2 --model output/iter_70000/model.pdparams
```

<br>

## 导出部署

使用如下命令导出训练完成的模型

```shell
# 导出Paddle3D提供的预训练模型
python tools/export.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --model output/iter_70000/model.pdparams
```

### 执行预测

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`smoke.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`smoke.pdiparams`所在路径 |
| image | 待预测的图片路径 |
| use_gpu | 是否使用GPU进行预测，默认为False|
| use_trt | 是否使用TensorRT进行加速，默认为False|
| trt_precision | 当use_trt设置为1时，模型精度可设置0/1/2，0表示fp32，1表示int8，2表示fp16。默认0 |
| collect_dynamic_shape_info | 是否收集模型动态shape信息。默认为False。**只需首次运行，下次运行时直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存收集到的模型动态shape信息的文件路径。默认为dynamic_shape_info.txt |

### Python部署

进入代码目录 `deploy/smoke/python`，运行以下命令，执行预测：

* 执行CPU预测

    ```shell
    python infer.py --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image
    ```

* 执行GPU预测

    ```shell
    python infer.py --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image --use_gpu
    ```

* 执行CPU预测并显示3d框

    ```shell
    python vis.py --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image
    ```

* 执行GPU预测并显示3d框

    ```shell
    python vis.py --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image --use_gpu
    ```

* 执行TRT预测

    **注意：需要下载支持TRT版本的paddlepaddle以及nvidia对应版本的TensorRT库**

    * **首次运行TensorRT**，收集模型动态shape信息，并保存至`--dynamic_shape_file`指定的文件中

        ```shell
        python infer.py --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image --collect_shape_info --dynamic_shape_file /path/to/shape_info.txt
        ```

    * 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP32精度进行预测

        ```shell
        python infer.py --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image --use_trt --dynamic_shape_file /path/to/shape_info.txt
        ```


### C++部署

#### 编译步骤

- step 1: 进入部署代码所在路径

```shell
cd deploy/smoke/cpp
```

- step 2: 下载Paddle Inference C++预编译库

Paddle Inference针对**是否使用GPU**、**是否支持TensorRT**、以及**不同的CUDA/cuDNN/GCC版本**均提供已经编译好的库文件，请至[Paddle Inference C++预编译库下载列表](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#c)选择符合的版本。

- step 3: 下载OpenCV

- step 4: 修改`compile.sh`中的编译参数

主要修改编译脚本`compile.sh`中的以下参数：

| 参数 | 说明 |
| -- | -- |
| WITH_GPU | 是否使用gpu。ON或OFF， OFF表示使用CPU，默认ON|
| USE_TENSORRT | 是否使用TensorRT加速。ON或OFF，默认OFF|
| LIB_DIR | Paddle Inference C++预编译包所在路径，该路径下的内容应有：`CMakeCache.txt`、`paddle`、`third_party`和`version.txt` |
| CUDNN_LIB | cuDNN`libcudnn.so`所在路径 |
| CUDA_LIB | CUDA`libcudart.so `所在路径 |
| TENSORRT_ROOT | TensorRT所在路径。**非必须**，如果`USE_TENSORRT`设置为`ON`时，需要填写该路径，该路径下的内容应有`bin`、`lib`和`include`等|

- step 5: 开始编译

```shell
sh compile.sh
```

- step 6: 执行预测

```shell
./build/infer --model_file /path/to/smoke.pdmodel --params_file /path/to/smoke.pdiparams --image /path/to/image
```

**注意：如果要使用TRT预测，请根据编译步骤的step 3，修改`compile.sh`中TensorRT相关的编译参数，并重新编译。**



<br>

## 自定义数据集

如果您想在自定义数据集上进行训练，请参考[自定义数据准备教程](../datasets/custom.md)将数据组织成KITTI数据格式即可
