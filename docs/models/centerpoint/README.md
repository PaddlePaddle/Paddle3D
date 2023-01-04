# CenterPoint：Center-based 3D Object Detection and Tracking
## 目录
* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练 & 评估](#4)
  * [nuScenes数据集](#41)
  * [KITTI数据集](#42)
* [导出 & 部署](#8)
* [自定义数据集](#9)
* [Apollo使用教程](#10)

## <h2 id="1">引用</h2>

> Yin, Tianwei and Zhou, Xingyi and Krahenbuhl, Philipp. "Center-Based 3D Object Detection and Tracking." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11784-11793. 2021.

## <h2 id="2">简介</h2>

CenterPoint是Anchor-Free的三维物体检测器，以点云作为输入，将三维物体在Bird-View下的中心点作为关键点，基于关键点检测的方式回归物体的尺寸、方向和速度。相比于Anchor-Based的三维物体检测器，CenterPoint不需要人为设定Anchor尺寸，面向物体尺寸多样不一的场景时其精度表现更高，且简易的模型设计使其在性能上也表现更加高效。

Paddle3D实现的CenterPoint做了以下优化：
- 对模型的前后处理做了性能优化。CenterPoint-Pillars在[nuScenes](https://www.nuscenes.org/nuscenes) val set上精度有50.97mAP，速度在Tesla V100上达到了50.28FPS。
- 提供[KITTI数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)上的训练配置和Baseline。CenterPoint-Pillars在KITTI val set上精度达到64.75 mAP，速度在Tesla V100上达到了43.96FPS。

跟原论文相比，Paddle3D实现的CenterPoint有以下差异：
- 未提供第二个阶段的实现。在原论文中，作者还设计了第二个阶段来进一步精炼物体的位置、尺寸和方向，并在[Waymo数据集](https://waymo.com/open/)上做了验证。Paddle3D目前还未适配Waymo数据集，所以第二个阶段暂未实现。
- 未提供在nuScenes数据集上将预测速度用于多目标跟踪的实现。

## <h2 id="3">模型库</h2>

- CenterPoint在nuScenes Val set数据集上的表现

| 模型 | 体素格式 | mAP | NDS | V100 TensorRT FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件 | 日志 |
| ---- | ---------------- | --- | --- | ----------------------- | ----------------------- | -------- | -------- | ---- |
| CenterPoint | 2D-Pillars | 50.97 | 61.30 | 50.28 | 63.43 | [model](https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_02voxel_nuscenes_10sweep/model.pdparams) | [config](../../../configs/centerpoint/centerpoint_pillars_02voxel_nuscenes_10sweep.yml) | [log](https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_02voxel_nuscenes_10sweep/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=f150eb3b4db30c7bd4ff2dfac5ca4166) |
| CenterPoint | 3D-Voxels | 59.25 | 66.74 | 21.90 | 26.93 | [model]( https://bj.bcebos.com/paddle3d/models/centerpoint/centerpoint_voxels_0075voxel_nuscenes_10sweep/model.pdparams) | [config](../../../configs/centerpoint/centerpoint_voxels_0075voxel_nuscenes_10sweep.yml) | [log]( https://bj.bcebos.com/paddle3d/models/centerpoint/centerpoint_voxels_0075voxel_nuscenes_10sweep/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=2cf9f2123ea8393cf873e8f8ae907fdc) |

**注意：nuScenes benchmark使用4张V100 GPU训练得出。3D Sparse Conv功能需要安装Paddle 2.4版。**

- CenterPoint在KITTI Val set数据集上的表现

| 模型 | 体素格式 | 3DmAP Mod. | Car<br>Easy Mod. Hard | Pedestrian<br>Easy Mod. Hard | Cyclist<br>Easy Mod. Hard | V100 TensorRT FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件 |  日志 |
| ---- | ---------------- | ---------- | ------------------ | ------------------------- | -----------------------| ----------------------- | ----------------------- | -------- | -------- | ---- |
| CenterPoint | 2D-Pillars | 64.75 | 85.99 76.69 73.62 | 57.66 54.03 49.75 | 84.30 63.52 59.47 | 43.96 | 74.21 | [model]( https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_016voxel_kitti/model.pdparams) | [config](../../../configs/centerpoint/centerpoint_pillars_016voxel_kitti.yml)| [log]( https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_016voxel_kitti/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=7f2b637cfce7995a55b915216b8b1171) |

| 模型 | 体素格式 | BEVmAP Mod. | Car<br>Easy Mod. Hard | Pedestrian<br>Easy Mod. Hard | Cyclist<br>Easy Mod. Hard | V100 TensorRT FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 | 配置文件 |  日志 |
| ---- | ---------------- | ----------- | ------------------ | ------------------------- | ---------------------- | ----------------------- | ----------------------- | -------- | -------- | ---- |
| CenterPoint | 2D-Pillars | 71.87 | 93.03 87.33 86.21 | 66.46 62.66 58.54 | 86.59 65.62 61.58 | 43.96 | 74.21 | [model]( https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_016voxel_kitti/model.pdparams) | [config](../../../configs/centerpoint/centerpoint_pillars_016voxel_kitti.yml)| [log]( https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_016voxel_kitti/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=7f2b637cfce7995a55b915216b8b1171) |

**注意：** KITTI benchmark使用8张V100 GPU训练得出。

## <h2 id="4">训练 & 评估</h2>

### <h3 id="41">nuScenes数据集</h3>
#### 数据准备

- 目前Paddle3D中提供的CenterPoint模型支持在nuScenes数据集上训练，因此需要先准备nuScenes数据集，请在[官网](https://www.nuscenes.org/nuscenes)进行下载，并将数据集目录准备如下：

```
nuscenes_dataset_root
|—— samples  
|—— sweeps  
|—— maps  
|—— v1.0-trainval  
```

在Paddle3D的目录下创建软链接 `datasets/nuscenes`，指向到上面的数据集目录:

```
mkdir datasets
ln -s /path/to/nuscenes_dataset_root ./datasets
mv ./datasets/nuscenes_dataset_root ./datasets/nuscenes
```

- 生成训练时数据增强所需的真值库:

```
python tools/create_det_gt_database.py --dataset_name nuscenes --dataset_root ./datasets/nuscenes --save_dir ./datasets/nuscenes
```

`--dataset_root`指定nuScenes数据集所在路径，`--save_dir`指定用于保存所生成的真值库的路径。该命令执行后，`save_dir`生成的目录如下：

```
gt_database_train_nsweeps10_withvelo
|—— anno_info_train_nsweeps10_withvelo.pkl
|—— bicycle
|   |—— 20646_bicycle_4.bin
|   |—— ...
|—— car
|—— ...
```
#### 训练

nuScenes数据集上的训练使用4张GPU：

```
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py --config configs/centerpoint/centerpoint_pillars_02voxel_nuscenes_10sweep.yml --save_dir ./output_nuscenes --num_workers 3 --save_interval 5
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。
#### 评估

```
python tools/evaluate.py --config configs/centerpoint/centerpoint_pillars_02voxel_nuscenes_10sweep.yml --model ./output_nuscenes/epoch_20/model.pdparams --batch_size 1 --num_workers 3
```

**注意**：CenterPoint的评估目前只支持batch_size为1。

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。
### <h3 id="42">KITTI数据集</h3>

- 目前Paddle3D中提供的CenterPoint模型支持在KITTI数据集上训练，因此需要先准备KITTI数据集，请在[官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)进行下载：

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
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/centerpoint/centerpoint_pillars_016voxel_kitti.yml --save_dir ./output_kitti --num_workers 4 --save_interval 5
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。
#### 评估


```
python tools/evaluate.py --config configs/centerpoint/centerpoint_pillars_016voxel_kitti.yml --model ./output_kitti/epoch_160/model.pdparams --batch_size 1 --num_workers 4
```

**注意**：CenterPoint的评估目前只支持batch_size为1。

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。

## <h2 id="8">导出 & 部署</h2>

### <h3 id="81">模型导出</h3>


运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
python tools/export.py --config configs/centerpoint/centerpoint_pillars_02voxel_nuscenes_10sweep.yml --model /path/to/model.pdparams --save_dir /path/to/output
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`centerpoint.pdiparams `、`centerpoint.pdiparams.info`和`centerpoint.pdmodel` |

### C++部署

#### Linux系统

#### 环境依赖

- GCC >= 5.4.0
- Cmake >= 3.5.1
- Ubuntu 16.04/18.04

> 说明：本文档的部署环节在以下环境中进行过测试并通过：

测试环境一：
- GCC==8.2.0
- Cmake==3.16.0
- Ubuntu 18.04
- CUDA 11.2
- cuDNN==8.1.1
- Paddle Inferece==2.3.1
- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2

测试环境二：
- GCC==7.5.0
- Cmake==3.19.6
- Ubuntu 18.04
- CUDA==11.1
- cuDNN==8.0.4
- Paddle Inferece==2.3.1
- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2

#### 编译步骤

**注意：目前CenterPoint的仅支持使用GPU进行推理。**

- step 1: 进入部署代码所在路径

```
cd deploy/centerpoint/cpp
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

**注意：目前CenterPoint的仅支持使用GPU进行推理。**

执行命令参数说明

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`centerpoint.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`centerpoint.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| num_point_dim | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4 |
| with_timelag | 该参数仅针对由多帧融合而成的点云文件，融合后的点云文件通常每个点都会包含时间差(timelag)。若点云维度大于等于5且第5维信息是timelag，需设置为1，默认0 |


```
./build/main --model_file /path/to/centerpoint.pdmodel --params_file /path/to/centerpoint.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 5
```

**注意：** 请预先确认实际待测试点云文件的维度是否是5，如果不是5，`--num_point_dim`请修改为实际值。如果待测试的点云文件是由多帧融合而成且点云维度大于等于5且第5维信息是timelag，可将`--with_timelag`设置为1。

### 开启TensorRT加速预测【可选】

**注意：请根据编译步骤的step 3，修改`compile.sh`中TensorRT相关的编译参数，并重新编译。**

运行命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`centerpoint.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`centerpoint.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| num_point_dim | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4 |
| with_timelag | 仅针对`nuscenes`数据集，若使用`nuscenes`数据集训练的模型，需设置为1，默认0 |
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认0。**只需首次运行，下次运行时直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |

* **首次运行TensorRT**，收集模型动态shape信息，并保存至`--dynamic_shape_file`指定的文件中

    ```
    ./build/main --model_file /path/to/centerpoint.pdmodel --params_file /path/to/centerpoint.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 5 --use_trt 1 --collect_shape_info 1 --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP32精度进行预测

    ```
    ./build/main --model_file /path/to/centerpoint.pdmodel --params_file /path/to/centerpoint.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 5 --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP16精度进行预测

    ```
    ./build/main --model_file /path/to/centerpoint.pdmodel --params_file /path/to/centerpoint.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 5 --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1
    ```

* 如果觉得每次运行时模型加载的时间过长，可以设置`trt_use_static`和`trt_static_dir`，首次运行时将TensorRT的优化信息保存在硬盘中，后续直接反序列化优化信息即可

```
./build/main --model_file /path/to/centerpoint.pdmodel --params_file /path/to/centerpoint.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 4 --use_trt 1 --dynamic_shape_file /path/to/shape_info.txt --trt_precision 1 --trt_use_static 1 --trt_static_dir /path/to/OptimCacheDir
```

### Python部署

**注意：目前CenterPoint的仅支持使用GPU进行推理。**

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`centerpoint.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`centerpoint.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| num_point_dim | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4 |
| with_timelag | 该参数仅针对由多帧融合而成的点云文件，融合后的点云文件通常每个点都会包含时间差(timelag)。若点云维度大于等于5且第5维信息是timelag，需设置为1>，默认0 |
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认0。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |

运行以下命令，执行预测：

```
python infer.py --model_file /path/to/centerpoint.pdmodel --params_file /path/to/centerpoint.pdiparams --lidar_file /path/to/lidar.pcd.bin --num_point_dim 5
```

Python开启TensorRT的推理步骤与C++开启TensorRT加速推理一致，请参考文档前面介绍【开启TensorRT加速预测】并将C++命令参数替换成Python的命令参数。推荐使用PaddlePaddle 的官方镜像，镜像内已经预安装TensorRT。官方镜像请至[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)进行下载。

## <h2 id="9">自定义数据集</h2>

请参考文档[自定义数据集格式说明](../../../datasets/custom.md)准备自定义数据集。
## <h2 id="10">Apollo使用教程</h2>

基于Paddle3D训练完成的CenterPoint模型可以直接部署到Apollo架构中使用，请参考[教程](https://github.com/ApolloAuto/apollo/blob/master/modules/perception/README_paddle3D_CN.md)
