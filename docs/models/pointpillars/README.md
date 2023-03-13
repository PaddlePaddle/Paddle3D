# PointPillars: Fast Encoders for Object Detection from Point Clouds

## 目录
* [引用](#h2-id1h2)
* [简介](#h2-id2h2)
* [模型库](#h2-id3h2)
* [训练配置](#h2-id4h2)
* [使用教程](#h2-id5h2)
  * [数据准备](#h3-id51h3)
  * [训练](#h3-id52h3)
  * [评估](#h3-id53h3)
  * [模型导出](#h3-id54h3)
  * [模型部署](#h3-id55h3)

## <h2 id="1">引用</h2>

> Lang, Alex H., Sourabh, Vora, Holger, Caesar, Lubing, Zhou, Jiong, Yang, and Oscar, Beĳbom. "PointPillars: Fast Encoders for Object Detection From Point Clouds." . In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 12689-12697).2019.

## <h2 id="2">简介</h2>
PointPillars是目前工业界应用广泛的点云检测模型，其最主要的特点是检测速度和精度的平衡。PointPillars 在 [VoxelNet](https://arxiv.org/abs/1711.06396) 和 [SECOND](https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf)
 的基础上针对性能进行了优化，将点云转化为柱体（Pillars）表示，从而使得编码后的点云特征可以使用2D卷积神经网络进行检测任务。

## <h2 id="3">模型库</h2>
- PointPillars在KITTI Val set数据集上Car类别的表现

|      模型      | Car<br>Easy Mod. Hard | V100 TensorRT FP32(FPS) | V100 TensorRT FP16(FPS) |                                                   模型下载                                                   |                                    配置文件                                    |
|:------------:|:---------------------:|:-----------------------:|:-----------------------:|:--------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
| PointPillars |   86.90 75.21 71.57   |          37.3           |          40.5           | [model](https://bj.bcebos.com/paddle3d/models/pointpillars/pointpillars_xyres16_kitti_car/model.pdparams) | [config](../../../configs/pointpillars/pointpillars_xyres16_kitti_car.yml) |

- PointPillars在KITTI Val set数据集上Cyclist及Pedestrian类别的表现

|      模型      | Cyclist<br>Easy Mod. Hard | Pedestrian<br>Easy Mod. Hard | V100 TensorRT FP32(FPS) | V100 TensorRT FP16(FPS) |                                                          模型下载                                                           |                                           配置文件                                            |
|:------------:|:-------------------------:|:----------------------------:|:-----------------------:|:-----------------------:|:-----------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|
| PointPillars |     84.36 64.66 60.53     |      66.13 60.36 54.40       |          30.0           |          30.2           | [model](https://bj.bcebos.com/paddle3d/models/pointpillars/pointpillars_xyres16_kitti_cyclist_pedestrian/model.pdparams) | [config](../../../configs/pointpillars/pointpillars_xyres16_kitti_cyclist_pedestrian.yml) |

## <h2 id="4">训练配置</h2>
我们提供了在开源数据集上的训练配置与结果，详见[PointPillars 训练配置](../../../configs/pointpillars)。

## <h2 id="5">使用教程</h2>

### <h3 id="51">数据准备</h3>

- 目前Paddle3D中提供的PointPillars模型支持在KITTI数据集上训练，因此需要先准备KITTI数据集，请在[官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)进行下载：

1. Download Velodyne point clouds, if you want to use laser information (29 GB)

2. training labels of object data set (5 MB)

3. camera calibration matrices of object data set (16 MB)

并下载数据集的划分文件列表：

```shell
wget https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz
```

将数据解压后按照下方的目录结构进行组织：

```
└── kitti_dataset_root
    |—— training
        |—— label_2
            |—— 000001.txt
            |—— ...
        |—— calib
            |—— 000001.txt
            |—— ...
        |—— velodyne
            |—— 000001.bin
            |—— ...
    |—— ImageSets
        |—— test.txt
        |—— train.txt
        |—— trainval.txt
        |—— val.txt
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
└── kitti_train_gt_database
    |—— anno_info_train.pkl
    |—— Car
        |—— 4371_Car_7.bin
        |—— ...
    |—— Cyclist
```

### <h3 id="52">训练</h3>
位于`Paddle3D/`目录下，执行：
```shell
python -m paddle.distributed.launch --gpus 0 \
    tools/train.py \
    --config configs/pointpillars/pointpillars_xyres16_kitti_car.yml \
    --save_interval 1856 \
    --keep_checkpoint_max 100 \
    --save_dir outputs/pointpillars \
    --do_eval \
    --num_workers 8
```

训练脚本支持设置如下参数：

| 参数名                 | 用途                             | 是否必选项  |    默认值    |
|:--------------------|:-------------------------------|:------:|:---------:|
| gpus                | 使用的GPU编号                       |   是    |     -     |
| config              | 配置文件                           |   是    |     -     |
| save_dir            | 模型和visualdl日志文件的保存根路径          |   否    |  output   |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 |   否    |     2     |
| save_interval       | 模型保存的间隔步数                      |   否    |   1000    |
| do_eval             | 是否在保存模型时进行评估                   |   否    |     否     |
| log_interval        | 打印日志的间隔步数                      |   否    |    10     |
| keep_checkpoint_max | 最新模型保存个数                       |   否    |     5     |
| resume              | 是否从断点恢复训练                      |   否    |     否     |
| batch_size          | mini-batch大小（每张GPU）            |   否    | 在配置文件中指定  |
| iters               | 训练轮数                           |   否    | 在配置文件中指定  |
| learning_rate       | 学习率                            |   否    | 在配置文件中指定  |
| seed                | Paddle的全局随机种子值                         |   否    |   None    |

### <h3 id="53">评估</h3>

位于`Paddle3D/`目录下，执行：

```shell
python tools/evaluate.py \
    --config configs/pointpillars/pointpillars_xyres16_kitti_car.yml \
    --model /path/to/model.pdparams \
    --num_workers 8
```

评估脚本支持设置如下参数：

| 参数名                 | 用途                             | 是否必选项  |    默认值    |
|:--------------------|:-------------------------------|:------:|:---------:|
| config              | 配置文件                           |   是    |     -     |
| model               | 待评估模型路径                        |   是    |     -     |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 |   否    |     2     |
| batch_size          | mini-batch大小                   |   否    | 在配置文件中指定  |

### <h3 id="54">模型导出</h3>

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```shell
python tools/export.py \
    --config configs/pointpillars/pointpillars_xyres16_kitti_car.yml \
    --model /path/to/model.pdparams \
    --save_dir /path/to/output
```

模型导出脚本支持设置如下参数：

| 参数名         | 用途                                                                                                           | 是否必选项  |    默认值    |
|:------------|:-------------------------------------------------------------------------------------------------------------|:------:|:---------:|
| config      | 配置文件                                                                                                         |   是    |     -     |
| model       | 待导出模型参数`model.pdparams`路径                                                                                    |   是    |     -     |
| save_dir    | 保存导出模型的路径，`save_dir`下将会生成三个文件：`pointpillars.pdiparams `、`pointpillars.pdiparams.info`和`pointpillars.pdmodel` |   否    | `deploy`  |

### <h3 id="55">模型部署</h3>

#### C++部署（Linux系统）

#### 环境依赖：

- GCC >= 5.4.0
- Cmake >= 3.5.1
- Ubuntu 16.04/18.04

> 说明：本文档的部署环节在以下环境中进行过测试并通过：

- GCC==8.2.0
- Cmake==3.16.0
- Ubuntu 18.04
- CUDA 11.2
- cuDNN==8.1.1
- PaddleInference==2.3.1
- TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2

#### 编译步骤：

**注意：目前PointPillars的仅支持使用GPU进行推理。**

- step 1: 进入部署代码所在路径

```commandline
cd deploy/pointpillars/cpp
```

- step 2: 下载Paddle Inference C++预编译库

Paddle Inference针对**是否使用GPU**、**是否支持TensorRT**、以及**不同的CUDA/cuDNN/GCC版本**
均提供已经编译好的库文件，请至[Paddle Inference C++预编译库下载列表](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#c)
选择符合的版本。

- step 3: 修改`compile.sh`中的编译参数

主要修改编译脚本`compile.sh`中的以下参数：

| 参数名           | 说明                                                                                           | 是否必选项  |                                默认值                                |
|:--------------|:---------------------------------------------------------------------------------------------|:------:|:-----------------------------------------------------------------:|
| WITH_GPU      | 是否使用GPU                                                                                      |   否    |                                ON                                 |
| USE_TENSORRT  | 是否使用TensorRT加速                                                                               |   否    |                                ON                                 |
| LIB_DIR       | Paddle Inference C++预编译包所在路径，该路径下的内容应有：`CMakeCache.txt`、`paddle`、`third_party`和`version.txt` |   是    |                                 -                                 |
| CUDNN_LIB     | cuDNN`libcudnn.so`所在路径                                                                       |   否    |                    `/usr/lib/x86_64-linux-gnu`                    |
| CUDA_LIB      | CUDA`libcuda.so`所在路径                                                                         |   否    |                      `/usr/local/cuda/lib64`                      |
| TENSORRT_ROOT | TensorRT所在路径                                                                                 |   否    | 如果`USE_TENSORRT`设置为`ON`时，需要填写该路径，该路径下的内容应有`bin`、`lib`和`include`等  |

- step 4: 开始编译

```commandline
sh compile.sh
```

#### 执行预测:

**注意：目前Pointpillars仅支持使用GPU进行推理。**

执行命令参数说明

| 参数名                 | 说明                                                                      | 是否必选项  | 默认值  |
|:--------------------|:------------------------------------------------------------------------|:------:|:----:|
| model_file          | 导出模型的结构文件`pointpillars.pdmodel`所在路径                                     |   是    |  -   |
| params_file         | 导出模型的参数文件`pointpillars.pdiparams`所在路径                                   |   是    |  -   |
| lidar_file          | 待预测的点云文件所在路径                                                            |   是    |  -   |
| point_cloud_range   | 模型中将点云划分为柱体（pillars）时选取的点云范围，格式为`"X_min Y_min Z_min X_max Y_Max Z_max"` |   是    |  -   |
| voxel_size          | 模型中将点云划分为柱体（pillars）时每个柱体的尺寸，格式为`"X_size Y_size Z_size"`                |   是    |  -   |
| max_points_in_voxel | 模型中将点云划分为柱体（pillars）时每个柱体包含点数量上限                                        |   是    |  -   |
| max_voxel_num       | 模型中将点云划分为柱体（pillars）时保留的柱体数量上限                                          |   是    |  -   |
| num_point_dim       | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4      |   否    |  4   |

执行命令：

```shell
./build/main \
  --model_file /path/to/pointpillars.pdmodel \
  --params_file /path/to/pointpillars.pdiparams \
  --lidar_file /path/to/lidar.pcd.bin \
  --point_cloud_range "0 -39.68 -3 69.12 39.68 1" \
  --voxel_size ".16 .16 4" \
  --max_points_in_voxel 32 \
  --max_voxel_num 40000
```

#### 开启TensorRT加速预测【可选】:

**注意：请根据编译步骤的step 3，修改`compile.sh`中TensorRT相关的编译参数，并重新编译。**

运行命令参数说明如下：

| 参数名                 | 说明                                                                                      | 是否必选项  | 默认值  |
|:--------------------|:----------------------------------------------------------------------------------------|:------:|:----:|
| model_file          | 导出模型的结构文件`pointpillars.pdmodel`所在路径                                                     |   是    |  -   |
| params_file         | 导出模型的参数文件`pointpillars.pdiparams`所在路径                                                   |   是    |  -   |
| lidar_file          | 待预测的点云文件所在路径                                                                            |   是    |  -   |
| point_cloud_range   | 模型中将点云划分为柱体（pillars）时选取的点云范围，格式为`"X_min Y_min Z_min X_max Y_Max Z_max"`                 |   是    |  -   |
| voxel_size          | 模型中将点云划分为柱体（pillars）时每个柱体的尺寸，格式为`"X_size Y_size Z_size"`                                |   是    |  -   |
| max_points_in_voxel | 模型中将点云划分为柱体（pillars）时每个柱体包含点数量上限                                                        |   是    |  -   |
| max_voxel_num       | 模型中将点云划分为柱体（pillars）时保留的柱体数量上限                                                          |   是    |  -   |
| num_point_dim       | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4                      |   否    |  4   |
| use_trt             | 是否开启TensorRT加速预测                                                                        |   否    |  0   |
| trt_precision       | 当`use_trt`设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16                                             |   否    |  0   |
| trt_use_static      | 当`trt_use_static`设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成** |   否    |  0   |
| trt_static_dir      | 当`trt_use_static`设置为1时，保存优化信息的路径                                                        |   否    |  -   |
| collect_shape_info  | 当`use_trt`设置为1时，是否收集模型动态shape信息。默认0。**只需首次运行，下次运行时直接加载生成的shape信息文件即可进行TensorRT加速推理**    |   否    |  0   |
| dynamic_shape_file  | 当`collect_shape_info`设置为1时，保存模型动态shape信息的文件路径                                           |   否    |  -   |

* **首次运行TensorRT**，收集模型动态shape信息，并保存至`--dynamic_shape_file`指定的文件中

    ```shell
    ./build/main \
      --model_file /path/to/pointpillars.pdmodel \
      --params_file /path/to/pointpillars.pdiparams \
      --lidar_file /path/to/lidar.bin \
      --num_point_dim 4 \
      --point_cloud_range "0 -39.68 -3 69.12 39.68 1" \
      --voxel_size ".16 .16 4" \
      --max_points_in_voxel 32 \
      --max_voxel_num 40000 \
      --use_trt 1 \
      --collect_shape_info 1 \
      --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP32精度进行预测

    ```shell
    ./build/main \
      --model_file /path/to/pointpillars.pdmodel \
      --params_file /path/to/pointpillars.pdiparams \
      --lidar_file /path/to/lidar.bin \
      --num_point_dim 4 \
      --point_cloud_range "0 -39.68 -3 69.12 39.68 1" \
      --voxel_size ".16 .16 4" \
      --max_points_in_voxel 32 \
      --max_voxel_num 40000 \
      --use_trt 1 \
      --dynamic_shape_file /path/to/shape_info.txt
    ```

* 加载`--dynamic_shape_file`指定的模型动态shape信息，使用FP16精度进行预测

    ```shell
    ./build/main \
      --model_file /path/to/pointpillars.pdmodel \
      --params_file /path/to/pointpillars.pdiparams \
      --lidar_file /path/to/lidar.bin \
      --num_point_dim 4 \
      --point_cloud_range "0 -39.68 -3 69.12 39.68 1" \
      --voxel_size ".16 .16 4" \
      --max_points_in_voxel 32 \
      --max_voxel_num 40000 \
      --use_trt 1 \
      --dynamic_shape_file /path/to/shape_info.txt \
      --trt_precision 1
    ```

* 如果觉得每次运行时模型加载的时间过长，可以设置`trt_use_static`和`trt_static_dir`，首次运行时将TensorRT的优化信息保存在硬盘中，后续直接反序列化优化信息即可

  ```shell
  ./build/main \
    --model_file /path/to/pointpillars.pdmodel \
    --params_file /path/to/pointpillars.pdiparams \
    --lidar_file /path/to/lidar.bin \
    --num_point_dim 4 \
    --point_cloud_range "0 -39.68 -3 69.12 39.68 1" \
    --voxel_size ".16 .16 4" \
    --max_points_in_voxel 32 \
    --max_voxel_num 40000 \
    --use_trt 1 \
    --dynamic_shape_file /path/to/shape_info.txt \
    --trt_use_static 1 \
    --trt_static_dir /path/to/OptimCacheDir
    ```

#### Python 部署

**注意：目前PointPillars的仅支持使用GPU进行推理。**

运行命令参数说明如下：

| 参数名                 | 用途                                                                                    | 是否必选项 | 默认值 |
|:--------------------|:--------------------------------------------------------------------------------------|:------|:----|
| mdoel_file          | 导出模型的结构文件`pointpillars.pdmodel`所在路径                                                   | 是     | -   |
| params_file         | 导出模型的参数文件`pointpillars.pdiparams`所在路径                                                 | 是     | -   |
| lidar_file          | 待预测的点云所在路径                                                                            | 是     | -   |
| point_cloud_range   | 模型中将点云划分为柱体（pillars）时选取的点云范围，格式为`X_min Y_min Z_min X_max Y_Max Z_max`                 | 是     | -   |
| voxel_size          | 模型中将点云划分为柱体（pillars）时每个柱体的尺寸，格式为`X_size Y_size Z_size`                                | 是     | -   |
| max_points_in_voxel | 模型中将点云划分为柱体（pillars）时每个柱体包含点数量上限                                                      | 是     | -   |
| max_voxel_num       | 模型中将点云划分为柱体（pillars）时保留的柱体数量上限                                                        | 是     | -   |
| num_point_dim       | 点云文件中每个点的维度大小。例如，若每个点的信息是`x, y, z, intensity`，则`num_point_dim`填写为4                    | 否     | 4   |
| use_trt             | 是否使用TensorRT进行加速                                                                      | 否     | 0   |
| trt_precision       | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16                                             | 否     | 0   |
| trt_use_static      | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成** | 否     | 0   |     |
| trt_static_dir      | 当trt_use_static设置为1时，保存优化信息的路径                                                        | 否     | -   |
| collect_shape_info  | 是否收集模型动态shape信息。默认0。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理**                     | 否     | 0   |     |
| dynamic_shape_file  | 保存模型动态shape信息的文件路径                                                                    | 否     | -   |

运行以下命令，执行预测：

```shell
python infer.py \
  --model_file /path/to/pointpillars.pdmodel \
  --params_file /path/to/pointpillars.pdiparams \
  --lidar_file /path/to/lidar.bin \
  --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
  --voxel_size .16 .16 4 \
  --max_points_in_voxel 32 \
  --max_voxel_num 40000
```
