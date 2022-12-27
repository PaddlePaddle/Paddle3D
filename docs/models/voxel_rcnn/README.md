# Voxel r-cnn: Towards high performance voxel-based 3d object detection

## 目录
* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练 & 评估](#4)
  * [KITTI数据集](#41)
* [导出 & 部署](#5)
* [自定义数据集](#6)

## <h2 id="1">引用</h2>

> Deng, Jiajun, et al. "Voxel r-cnn: Towards high performance voxel-based 3d object detection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 2. 2021.

## <h2 id="2">简介</h2>

Voxel RCNN在仅使用Voxel-Based的情况下，通过调整模型参数达到当时Point-Based和Voxel-Based相结合的SOTA方法的精度。并对RCNN模型结构做了改进，使得模型速度得到了大幅提升。

## <h2 id="3">模型库</h2>

- Voxel-RCNN在KITTI Val set数据集上的表现：

| 模型 | Car Mod@0.7 AP_R11 / AP_R40 | V100 Paddle Inference FP32(FPS) | 模型下载 | 配置文件 |  日志 |
| --- | --------------------------- | -------------------------------- | ------ | --------|--------|
| Voxel-RCNN | 84.64 / 85.49 |  22.39 | [model](https://paddle3d.bj.bcebos.com/models/voxel_rcnn/voxel_rcnn_005voxel_kitti_car/model.pdparams) | [config](../../../configs/voxel_rcnn/voxel_rcnn_005voxel_kitti_car.yml) | [log](https://paddle3d.bj.bcebos.com/models/voxel_rcnn/voxel_rcnn_005voxel_kitti_car/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=15cbecb8132e91dfa4fbd6d8f904c0a7) |

**注意：** KITTI benchmark使用8张V100 GPU训练得出。

## <h2 id="4">训练 & 评估</h2>

### <h3 id="41">KITTI数据集</h3>

- 目前Paddle3D中提供的Voxel-RCNN模型支持在KITTI数据集上训练，因此需要先准备KITTI数据集，请在[官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)进行下载：

1. Download Velodyne point clouds, if you want to use laser information (29 GB)

2. training labels of object data set (5 MB)

3. camera calibration matrices of object data set (16 MB)

并下载数据集的划分文件列表：

```
wget https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz
```

将数据解压后按照下方的目录结构进行组织：

```
kitti_dataset_root
|—— training
|   |—— label_2
|   |   |—— 000001.txt
|   |   |—— ...
|   |—— calib
|   |   |—— 000001.txt
|   |   |—— ...
|   |—— velodyne
|   |   |—— 000001.bin
|   |   |—— ...
|—— ImageSets
│   |—— test.txt
│   |—— train.txt
│   |—— trainval.txt
│   |—— val.txt
```

在Paddle3D的目录下创建软链接 `datasets/KITTI`，指向到上面的数据集目录:

```
mkdir datasets
ln -s /path/to/kitti_dataset_root ./datasets
mv ./datasets/kitti_dataset_root ./datasets/KITTI
```

- 生成训练时数据增强所需的真值库:

```
python tools/create_det_gt_database.py --dataset_name kitti --dataset_root ./datasets/KITTI --save_dir ./datasets/KITTI
```

`--dataset_root`指定KITTI数据集所在路径，`--save_dir`指定用于保存所生成的真值库的路径。该命令执行后，`save_dir`生成的目录如下：

```
kitti_train_gt_database
|—— anno_info_train.pkl
|—— Car
|   |—— 4371_Car_7.bin
|   |—— ...
|—— Cyclist
```
#### 训练

KITTI数据集上的训练使用8张GPU：

```
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/voxel_rcnn/voxel_rcnn_car.yml --save_dir ./output_voxel_rcnn --num_workers 4 --save_interval 1
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。
#### 评估

```
python tools/evaluate.py --config configs/voxel_rcnn/voxel_rcnn_car.yml --model ./output_voxel_rcnn/epoch_80/model.pdparams --batch_size 1 --num_workers 4
```

**注意**：Voxel-RCNN的评估目前只支持batch_size为1。

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。

## <h2 id="5">导出 & 部署</h2>

### 模型导出

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
python tools/export.py --config configs/voxel_rcnn/voxel_rcnn_car.yml --model /path/to/model.pdparams --save_dir /path/to/output
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`voxel_rcnn.pdiparams `、`voxel_rcnn.pdiparams.info`和`voxel_rcnn.pdmodel` |


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

**注意：目前Voxel-RCNN的仅支持使用GPU进行推理。**

- step 1: 进入部署代码所在路径

```
cd deploy/voxel_rcnn/cpp
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

**注意：目前Voxel-RCNN的仅支持使用GPU进行推理。**

执行命令参数说明

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`voxel_rcnn.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`voxel_rcnn.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| num_point_dim | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4 |
| point_cloud_range | 输入模型的点云所处的空间范围，超出此范围内的点将被滤除。格式为`"X_min Y_min Z_min X_max Y_Max Z_max"`|

```
./build/main --model_file /path/to/voxel_rcnn.pdmodel --params_file /path/to/voxel_rcnn.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --point_cloud_range "0 -40 -3 70.4 40 1"
```

**注意：** 请预先确认实际待测试点云文件的维度是否是4，如果不是4，`--num_point_dim`请修改为实际值。


### 开启TensorRT加速预测【可选】

**注意：请根据编译步骤的step 3，修改`compile.sh`中TensorRT相关的编译参数，并重新编译。**

运行命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`voxel_rcnn.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`voxel_rcnn.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| num_point_dim | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4 |
| point_cloud_range | 输入模型的点云所处的空间范围，超出此范围内的点将被滤除。格式为`"X_min Y_min Z_min X_max Y_Max Z_max"`|
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认0。**只需首次运行，下次运行时直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |

* **首次运行TensorRT**，收集模型动态shape信息，并保存至`--dynamic_shape_file`指定的文件中

    ```
    ./build/main --model_file /path/to/voxel_rcnn.pdmodel --params_file /path/to/voxel_rcnn.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --point_cloud_range "0 -40 -3 70.4 40 1" --use_trt 1 --collect_shape_info 1 --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP32精度进行预测

    ```
    ./build/main --model_file /path/to/voxel_rcnn.pdmodel --params_file /path/to/voxel_rcnn.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --point_cloud_range "0 -40 -3 70.4 40 1" --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP16精度进行预测

    ```
    ./build/main --model_file /path/to/voxel_rcnn.pdmodel --params_file /path/to/voxel_rcnn.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --point_cloud_range "0 -40 -3 70.4 40 1" --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1
    ```

* 如果觉得每次运行时模型加载的时间过长，可以设置`trt_use_static`和`trt_static_dir`，首次运行时将TensorRT的优化信息保存在硬盘中，后续直接反序列化优化信息即可

```
./build/main --model_file /path/to/voxel_rcnn.pdmodel --params_file /path/to/voxel_rcnn.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --point_cloud_range "0 -40 -3 70.4 40 1" --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1 --trt_use_static 1 --trt_static_dir /path/to/OptimCacheDir
```

### Python部署

**注意：目前Voxel-RCNN的仅支持使用GPU进行推理。**

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`voxel_rcnn.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`voxel_rcnn.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| num_point_dim | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4 |
| point_cloud_range | 输入模型的点云所处的空间范围，超出此范围内的点将被滤除。格式为`X_min Y_min Z_min X_max Y_Max Z_max`|
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认0。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |


运行以下命令，执行预测：

```
python infer.py --model_file /path/to/voxel_rcnn.pdmodel --params_file /path/to/voxel_rcnn.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --point_cloud_range 0 -40 -3 70.4 40 1
```

## <h2 id="6">自定义数据集</h2>

请参考文档[自定义数据集格式说明](../../../datasets/custom.md)准备自定义数据集。
