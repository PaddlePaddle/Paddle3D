# Release Notes

## v1.0

2022.12.27

### New Features

* The new version 1.0 of Paddle3D is released, which provides the following features

    * We supports multiple type of 3D perception models, including monocular 3D models SMOKE/CaDDN/DD3D, pointcloud detection models PointPillars/CenterPoint/IA-SSD/PV-RCNN/Voxel R-CNN, BEV visual detection models PETR/PETRv2/BEVFormer, and pointcloud segmentation model SqueezeSegv3

    * We added support for Waymo datasets and now Paddle3D has completed full support for the three open source datasets for autonomous driving

    * Supports automatic mixed-precision training and quantitative deployment capabilities, providing better model acceleration capabilities

    * Supports for sparse convolution, and integrated related SOTA models that are easy to deploy

    * We continue to cooperate with Apollo team to provide one-click deployment of multiple models and integrate them into the perception algorithm part of Apollo to make it easier for developers to debug models

### 新特性

* 全部发布Paddle3D 1.0版本，提供了以下特性：

    * 支持多种3D感知模型，包括单目3D模型SMOKE/CaDDN/DD3D，点云检测模型 PointPillars/CenterPoint/IA-SSD/PV-RCNN/Voxel R-CNN，BEV视觉检测模型 PETR/PETRv2/BEVFormer，点云分割模型SqueezeSegv3

    * 新增Waymo数据集支持，完成了对自动驾驶三大开源数据集的全面支持

    * 支持自动混合精度训练以及量化部署能力，提供更好的模型加速能力

    * 新增了对稀疏卷积能力的支持，并集成了稀疏卷积方向的SOTA模型，模型训推一体，便于部署

    * 持续与Apollo进行合作开发，提供多个模型一键部署集成到Apollo的感知算法部分，便于开发者更好地进行模型调试

## v0.5

2022.08.09

### New Features

* Release the first version of Paddle3D (that is, v0.5 version), the code of this project is still in beta stage, but under rapid development

* Release the monocular 3D model SMOKE/CaDDN, the point cloud detection model PointPillars/CenterPoint, and the point cloud segmentation model SqueezeSegv3, all with deployment tutorials

* Support for KITTI dataset and nuScenes dataset

* Tutorials for quick access to the Apollo platform are provided for all detection models

### 新特性

* 发布Paddle3D 的第一个版本 v0.5，本项目的代码仍在beta阶段，但处于快速迭代中

* 发布单目3D模型SMOKE/CaDDN，点云检测模型 PointPillars/CenterPoint，以及点云分割模型SqueezeSegv3，均带有部署教程

* 支持KITTI数据集和nuScenes数据集

* 为所有的检测模型提供了快速接入Apollo平台的教程
