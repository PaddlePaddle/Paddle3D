# BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework

## 目录
* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练 & 评估](#4)
  * [nuScenes数据集](#41)
* [导出 & 部署](#8)

## <h2 id="1">引用</h2>

```
@inproceedings{liang2022bevfusion,
  title={{BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework}},
  author={Tingting Liang, Hongwei Xie, Kaicheng Yu, Zhongyu Xia, Zhiwei Lin, Yongtao Wang, Tao Tang, Bing Wang and Zhi Tang},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## <h2 id="2">简介</h2>

BEVFusion是一种在BEV视角下的多模态融合模型，目前有两个开源版本，分别是[BEVFusion-ADLab](https://github.com/ADLab-AutoDrive/BEVFusion)和[BEVFusion-MIT](https://github.com/mit-han-lab/bevfusion)，Paddle3D提供的是BEVFusion-ADLab的模型实现，BEVFusion采用两个分支处理不同模态的数据，得到lidar和camera在BEV视角下的特征，在统一的坐标系下对两种模态的特征进行融合。camera分支采用LSS这种自底向上的方式来显式的生成图像BEV特征，其通过预测图像像素的深度信息，结合相机内外参将图像特征投影到BEV空间下。lidar分支采用经典的点云检测网络，如pointpillars来提取lidar数据的BEV特征，最后对两种模态的BEV特征进行对齐和融合，应用于检测head或分割head。


## <h2 id="3">模型库</h2>

- BEVFusion在nuScenes Val set数据集上的表现

| 模型 | Head | 3DBackbone | 2DBackbone | mAP | NDS | 模型下载 | 配置文件 | 日志 |
| ---- | ------ | --- | ----| ------- |------- | ---- | ---- | ---- |
| BEVFusion (C) | PointPillars | - | Dual-Swin-T | 22.7 | 29.5 | [model](https://paddle3d.bj.bcebos.com/models/bevfusion/camera/model.pdparams) | [config](../../../configs/bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.yaml) | [log](https://paddle3d.bj.bcebos.com/models/bevfusion/camera/train.log)\|[vdl](https://paddle3d.bj.bcebos.com/models/bevfusion/camera/vdlrecords.1675518011.log) |
| BEVFusion (L) | PointPillars | PointPillars | - | 34.8 | 49.7 | [model](https://paddle3d.bj.bcebos.com/models/bevfusion/lidar/model.pdparams) | [config](../../../configs/bevfusion/lidar_stream/bevf_pp_4x8_2x_nusc_lidar.yaml) | [log](https://paddle3d.bj.bcebos.com/models/bevfusion/lidar/train.log)\|[vdl](https://paddle3d.bj.bcebos.com/models/bevfusion/lidar/vdlrecords.1675690056.log) |
| BEVFusion (L+C) | PointPillars | PointPillars | Dual-Swin-T | 53.9 | 60.9 | [model](https://paddle3d.bj.bcebos.com/models/bevfusion/fusion/model.pdparams) | [config](../../../configs/bevfusion/bevf_pp_2x8_1x_nusc.yaml) | [log](https://paddle3d.bj.bcebos.com/models/bevfusion/fusion/train.log)\|[vdl](https://paddle3d.bj.bcebos.com/models/bevfusion/fusion/vdlrecords.1673110577.log) |

**注意：nuScenes benchmark使用8张V100 GPU训练得出。**


## <h2 id="4">训练 & 评估</h2>

### <h3 id="41">nuScenes数据集</h3>
#### 数据准备

- 目前Paddle3D中提供的BEVFusion模型支持在nuScenes数据集上训练，因此需要先准备nuScenes数据集，请在[官网](https://www.nuscenes.org/nuscenes)进行下载，将数据集目录准备如下：

```
nuscenes_dataset_root
|-- can_bus
|—— samples  
|—— sweeps  
|—— maps  
|—— v1.0-trainval  
```

在Paddle3D的目录下创建软链接 `data/nuscenes`，指向到上面的数据集目录:

```
mkdir data
ln -s /path/to/nuscenes_dataset_root ./data
mv ./data/nuscenes_dataset_root ./data/nuscenes
```

为加速训练过程中Nuscenes数据集的加载和解析，需要事先将Nuscenes数据集里的标注信息存储在`pkl`后缀文件，请下载预先生成好的`pkl`文件[train_infos.pkl](https://paddle3d.bj.bcebos.com/models/bevfusion/nuscenes_infos_train.pkl)，[val_infos.pkl](https://paddle3d.bj.bcebos.com/models/bevfusion/nuscenes_infos_val.pkl)


#### 训练
BEVFusion采用4阶段的训练方式：

- 在nuImage上采用MaskRCNN训练camera分支的backbone和neck
- 在nuScenes上训练camera分支，需要加载上一步的backbone和neck权重
- 在nuScenes上训练lidar分支
- 在nuScenes上训练融合模型，需要加载camera分支和lidar分支的预训练权重

*Paddle3D提供了第一个阶段的预训练权重和后3个阶段的训练步骤*

##### Step1. 训练camera分支
下载在nuImage上的预训练权重：
```
wget https://paddle3d.bj.bcebos.com/models/bevfusion/mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre.pdparams
```
修改配置文件`bevf_pp_4x8_2x_nusc_cam.yaml`中的`load_img_from`为下载的权重路径

执行以下命令训练camera分支
```
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.yaml --save_dir ./outputs/bevf_pp_cam --num_workers 4 --save_interval 1 --log_interval 50
```

执行以下命令评估camera分支
```
python tools/evaluate.py --config configs/bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.yaml --model ./outputs/bevf_pp_cam/epoch_24/model.pdparams --num_workers 2
```

##### Step2. 训练lidar分支
执行以下命令训练lidar分支
```
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevfusion/lidar_stream/bevf_pp_4x8_2x_nusc_lidar.yaml --save_dir ./outputs/bevf_pp_lidar --num_workers 4 --save_interval 1 --log_interval 50
```

执行以下命令评估lidar分支
```
python tools/evaluate.py --config configs/bevfusion/lidar_stream/bevf_pp_4x8_2x_nusc_lidar.yaml --model ./outputs/bevf_pp_lidar/epoch_24/model.pdparams --num_workers 2 --batch_size 1
```

##### Step3. 训练融合模型
修改配置文件`bevf_pp_2x8_1x_nusc.yaml`中的`load_cam_from`为step1训练camera分支的权重路径，修改`load_lidar_from`为step2训练lidar分支的权重路径

执行以下命令训练融合模型
```
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevfusion/bevf_pp_2x8_1x_nusc.yaml --save_dir ./outputs/bevf_pp --num_workers 4 --save_interval 1 --log_interval 50
```

执行以下命令评估融合模型
```
python tools/evaluate.py --config configs/bevfusion/bevf_pp_2x8_1x_nusc.yaml --model ./outputs/bevf_pp/epoch_12/model.pdparams --num_workers 2 --batch_size 1
```


## <h2 id="8">导出 & 部署</h2>

### <h3 id="81">模型导出</h3>

导出代码开发中。

### <h3 id="82">模型部署</h3>

部署代码开发进行中。
