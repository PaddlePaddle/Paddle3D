# Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds
![IA-SSD](https://user-images.githubusercontent.com/29754889/196080210-ba075afd-8c91-4d76-91af-13fd03a4176a.png)
## 目录
* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练 & 评估](#4)
  * [KITTI数据集](#41)
  * [Waymo数据集](#42)
* [导出 & 部署](#8)

## <h2 id="1">一：引用</h2>
> Zhang, Yifan, et al. "Not all points are equal: Learning highly efficient point-based detectors for 3d lidar point clouds." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR). 2022.

## <h2 id="2">二：简介</h2>

IA-SSD是一个single-stage & point-based的3D点云目标检测器。由于点云数据存在较大的冗余，论文提出了面向检测任务的instance-aware sampling方法来有效的采样出那些具有代表性的点，并引入contextual centroid perception来进一步预测更为精确的物体中心，以此来获得更准确的检测结果。IA-SSD以较小的显存占用和较快的速度在kitti和waymo数据集上取得了具有竞争力的结果。后续通过引入基于注意力机制的学习目标，进一步提升了IA-SSD在小目标和罕见类别上的效果。

## <h2 id="3">三：模型库</h2>

- IA-SSD在KITTI Val set数据集上的表现

| 模型 | Car Mod <br> (IoU=0.7) | Ped. Mod<br>(IoU=0.5) | Cyc. Mod<br>(IoU=0.5) | 模型下载 | 配置文件 | 日志 | VDL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IA-SSD | 79.13 | 58.51 | 71.32 | [model](https://paddle3d.bj.bcebos.com/models/iassd/iassd_kitti/model.pdparams) | [config](../../../configs/iassd/iassd_kitti.yaml) | [log](https://paddle3d.bj.bcebos.com/models/iassd/iassd_kitti/train.log) | [visualdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=7010b963202fe43cdf67cf714678e411) |
| IA-SSD（attention） | 79.31 | 60.36 | 71.84 | [model](https://paddle3d-models.bj.bcebos.com/model.pdparams) | \ | [log](https://paddle3d-models.bj.bcebos.com/train.log) | [visualdl](https://paddle3d-models.bj.bcebos.com/vdlrecords.1679889250.log) |

**注意：** KITTI benchmark使用4张V100 GPU训练得出。


- IA-SSD在Waymo Val set数据集上的表现

| 模型 | Vec_L1<br>AP / APH | Vec_L2<br>AP / APH | Ped_L1<br>AP / APH | Ped_L2<br>AP / APH | Cyc_L1<br>AP / APH | Cyc_L2<br>AP / APH | 模型下载 | 配置文件 | 日志 | VDL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IA-SSD | 73.90 / 73.27| 64.84 / 64.28 | 70.36 / 60.75 | 62.93 / 54.13  | 68.21 / 66.25 | 66.06 / 64.16 | - | [config](../../../configs/iassd/iassd_waymo.yaml) | [log](https://paddle3d.bj.bcebos.com/models/iassd/iassd_waymo/train.log) | [visualdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=12256b19993b8876cf36124e2e0f2bd5) |

**注意：** Waymo benchmark使用4张V100 GPU训练得出。另外，由于Waymo数据集[License](https://waymo.com/open/faq/)许可问题，我们无法提供在Waymo数据上训练出的模型权重，用户可以采用提供的模型配置训练出该精度。

## <h2 id="4">四：训练 & 评估</h2>

### <h3 id="41">KITTI数据集</h3>

#### 1）数据准备
- 目前Paddle3D中提供的IA-SSD模型支持在KITTI数据集上训练，因此需要先准备KITTI数据集，请在[官网](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)进行下载：

1. Download Velodyne point clouds, if you want to use laser information (29 GB)

2. training labels of object data set (5 MB)

3. camera calibration matrices of object data set (16 MB)

4. Download road plane infos from [here](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) (optional)

- 下载数据集的划分文件列表：

```
wget https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz
```

- 将数据解压后按照下方的目录结构进行组织：

```
KITTI
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
|   |—— planes(optional)
|   |   |—— 000001.txt
|   |   |—— ...
|—— ImageSets
│   |—— test.txt
│   |—— train.txt
│   |—— trainval.txt
│   |—— val.txt
```

- 在Paddle3D的目录下创建软链接 `datasets/KITTI`，指向到上面的数据集目录:

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
KITTI
|—— ImageSets
│   |—— test.txt
│   |—— train.txt
│   |—— trainval.txt
│   |—— val.txt
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
|   |—— planes(optional)
|   |   |—— 000001.txt
|   |   |—— ...
|—— kitti_train_gt_database
|   |—— anno_info_train.pkl
|   |—— Car
|   |   |—— 4371_Car_7.bin
|   |   |—— ...
|   |—— Cyclist
```

#### 2）训练

```shell
# 单卡训练
python tools/train.py --config configs/iassd/iassd_kitti.yaml --save_interval 1 --num_workers 4 --save_dir outputs/iassd_kitti

# 多卡训练，每隔1个epoch保存模型至save_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun tools/train.py --config configs/iassd/iassd_kitti.yaml --save_interval 1 --num_workers 4 --save_dir outputs/iassd_kitti
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。
#### 3）评估

```shell
python tools/evaluate.py --config configs/iassd/iassd_kitti.yaml --batch_size 16 --num_workers 4 --model outputs/iass_kitti/epoch_80/model.pdparams
```

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。
### <h3 id="42">Waymo数据集</h3>
#### 1）数据准备
- 目前Paddle3D中提供的IA-SSD模型支持在Waymo数据集上训练，因此需要先准备Waymo数据集，请在[官网](https://waymo.com/open)进行下载，建议下载***v1.3.2***及之后的版本

- 下载数据集的划分文件列表：

```shell
wget https://bj.bcebos.com/paddle3d/datasets/Waymo/1.3.2/ImageSets.tar.gz
```

- 将数据解压后按照下方的目录结构进行组织：

```
waymo
|—— ImageSets
│   |—— train.txt
│   |—— val.txt
|—— raw_data(解压后的所有tfrecord文件)
|   |—— segment-xxxx.tfrecord
|   |—— ...
```

- 在Paddle3D的目录下创建软链接 `datasets/waymo`，指向到上面的数据集目录:

```shell
mkdir datasets
ln -s /path/to/waymo_dataset_root ./datasets
mv ./datasets/waymo_dataset_root ./datasets/waymo
```

- 解析`raw_data`中tfrecord序列中的每个frame数据，并生成训练时数据增强所需的真值库:

```shell
python tools/create_waymo_infos.py
```

该命令执行后，生成的目录信息如下：

```
waymo_dataset_root
|—— ImageSets
│   |—— train.txt
│   |—— val.txt
|—— raw_data(解压后的所有tfrecord文件)
|   |—— segment-xxxx.tfrecord
|   |—— ...
|—— waymo_processed_data_v1_3_2
|   |—— segment-xxxx
|   |   |—— 0000.npy
|   |   |—— ...
|   |   |—— segment-xxx.pkl
|   |—— segment-xxxx
|   |—— ...
|—— waymo_train_gt_database
|   |—— Cyclist
|   |   |—— xxxx.bin
|   |   |—— ...
|   |—— Pedestrian
|   |—— Vehicle
|   |—— waymo_train_gt_database_infos.pkl
```
`train.txt`和`val.txt`存放train和val的tfrecord文件列表，`waymo_processed_data_v1_3_2`是解析后的数据，每一个frame的点云数据以.npy的形式存放，每一个frame的标注信息存放在同级目录的.pkl文件中。`waymo_train_gt_database`是采样出的真值物体点云和box坐标信息。

#### 2）训练

> waymo数据的一个segment是由连续的frames组成，我们遵循业界的普遍做法，对每个segment以间隔5进行采样，取到整个训练数据的***20%约32k个frame***用于训练

```shell
# 单卡训练
python tools/train.py --config configs/iassd/iassd_waymo.yaml --save_interval 1 --num_workers 4 --save_dir outputs/iassd_waymo

# 多卡训练，每隔1个epoch保存模型至save_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3
fleetrun tools/train.py --config configs/iassd/iassd_waymo.yaml --save_interval 1 --num_workers 4 --save_dir outputs/iassd_waymo
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。
#### 3）评估

```
python tools/evaluate.py --config configs/iassd/iassd_waymo.yaml --batch_size 32 --num_workers 4 --model outputs/iassd_waymo/epoch_30/model.pdparams
```

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。

## <h2 id="8">导出 & 部署</h2>

### <h3 id="81">模型导出</h3>

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```shell
python tools/export.py --config configs/iassd/iassd_kitti.yaml --model /path/to/model.pdparams --save_dir /path/to/output
```

| 参数 | <center>说明</center> |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`iassd.pdiparams `、`iassd.pdiparams.info`和`iassd.pdmodel`, `iassd.pdmodel`和`iassd.pdiparams`用于后续模型推理|

### Python部署
目前IA-SSD的模型的Python部署只支持GPU和TensorRT加速。
#### 1）环境依赖
> - Ubuntu 18.04
> - Python==3.7
> - CUDA==11.2
> - cuDNN==8.2.0
> - Paddle Inference==2.3.2
> - TensorRT_8.2.5.1
#### 2）执行预测
进入`deploy/iassd/python`，运行以下命令，执行不同配置的推理（如果需要开启TensorRT加速，请下载带有TRT的PaddlePaddle版本）：
- 执行原生`GPU`预测：
```shell
python3.7 deploy/iassd/python/infer.py --model_file /path/to/iassd.pdmodel --params_file /path/to/iassd.pdiparams --lidar_file /.../000001.bin --gpu_id 0
```
- 执行`TRT_FP32`预测：
```shell
python3.7 deploy/iassd/python/infer.py --model_file /path/to/iassd.pdmodel --params_file /path/to/iassd.pdiparams --lidar_file /.../000001.bin --gpu_id 0 --run_mode trt_fp32
```

- 执行`TRT_FP16`预测：
```shell
python3.7 deploy/iassd/python/infer.py --model_file /path/to/iassd.pdmodel --params_file /path/to/iassd.pdiparams --lidar_file /.../000001.bin --gpu_id 0 --run_mode trt_fp16
```

执行上述命令后可以得到点云3D检测结果：
```
{'boxes': array([[ 5.9013954e+01,  1.6384674e+01, -8.3350408e-01,  4.2825608e+00,
         1.6158756e+00,  1.5127110e+00,  2.9756324e+00],
       [ 1.7047174e+00, -5.7812780e-01, -1.0745335e+00,  3.9012828e+00,
         1.5632588e+00,  1.4671783e+00, -2.4538052e+00],
       [ 4.6258511e+01, -4.5674486e+00, -6.3848123e-03,  1.6965854e+00,
         4.1198540e-01,  1.7212551e+00,  1.1975995e-01],
       [ 4.0068634e+01,  1.9198923e+01, -9.9642396e-01,  4.1801486e+00,
         1.6866810e+00,  1.6624287e+00, -1.9688037e-01]], dtype=float32),
'labels': array([0, 0, 2, 0]),
'scores': array([0.35698408, 0.2946543 , 0.21058026, 0.16246885], dtype=float32)}
```
| 参数 | <center>说明</center> |
|:--| :-- |
| --model_file | 导出模型的结构文件`iassd.pdmodel`路径 |
| --params_file | 导出模型的参数文件`iassd.pdiparams`路径 |
| --lidar_file | 待预测的点云文件`.bin`格式 |
| --gpu_id | 用于预测的GPU ID |
| --run_mode | 推理模式，支持`fp32`,`trt_fp32`,`trt_fp16` |

### C++部署
目前IA-SSD的模型的C++部署只支持GPU和TensorRT加速。
#### 1）环境依赖
> - Ubuntu 18.04
> - Python==3.7
> - CUDA==11.2
> - cuDNN==8.2.0
> - Paddle Inference==2.3.2
> - TensorRT_8.2.5.1
> - GCC==8.2.0
> - CMake==3.16.0
#### 2）编译
- step 1：进入部署代码所在路径
```shell
cd deploy/iassd/cpp
```
- step 2：下载Paddle Inference C++预测库

Paddle Inference针对是否使用GPU、是否支持TensorRT、以及不同的CUDA/cuDNN/GCC版本均提供已经编译好的库文件，请至
[Paddle Inference C++预测库](https://www.paddlepaddle.org.cn/inference/v2.4/guides/introduction/index_intro.html#c)中下载符合的版本。
- step 3：修改compile.sh中的编译参数
主要修改以下参数：

| 参数 | <center>说明</center> |
|:--| :-- |
| WITH_GPU | 是否使用gpu。ON或OFF， OFF表示使用CPU，默认ON |
| USE_TENSORRT | 是否使用TensorRT加速。ON或OFF，默认OFF |
| LIB_DIR | Paddle Inference C++预测库所在路径 |
| CUDNN_LIB | cuDNN `libcudnn.so`所在路径 |
| CUDA_LIB | CUDA `libcudart.so`所在路径 |
| TENSORRT_ROOT | TensorRT安装路径。如果开启`USE_TENSORRT`加速，则需要填写该路径 |
- step 4：编译
```shell
sh compile.sh
```
### 3）执行预测

执行命令参数说明

| 参数 | <center>说明</center> |
| :-- | :-- |
| model_file | 导出模型的结构文件`iassd.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`iassd.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| rum_mode | 预测配置，支持`trt_fp32`, `trt_fp16`，默认采用`gpu_fp32`执行预测 |
| gpu_id | 用于预测的GPU_ID |


- 执行原生`GPU`预测：
```shell
./build/main --model_file /path/to/iassd.pdmodel --params_file /path/to/iassd.pdiparams  --lidar_file /.../000001.bin --gpu_id 0
```
- 执行`trt_fp32`预测：

```shell
./build/main --model_file /path/to/iassd.pdmodel --params_file /path/to/iassd.pdiparams --lidar_file /.../000001.bin --gpu_id 0 --run_mode trt_fp32
```

- 执行`trt_fp16`预测：

```shell
./build/main --model_file /path/to/iassd.pdmodel --params_file /path/to/iassd.pdiparams --lidar_file /.../000001.bin --gpu_id 0 --run_mode trt_fp16
```
