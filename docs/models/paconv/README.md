# PAConv：Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds

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

> Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan. "PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds." In Proceedings of CVPR 2021.

<br>

## 简介

该论文介绍了位置自适应卷积（PAConv），一种用于三维点云处理的通用卷积运算。PAConv的关键是通过动态组合存储在权重库中的基本权重矩阵来构造卷积矩阵，其中这些权重矩阵的系数通过核心网从点位置自适应学习。通过这种方式，内核构建在数据驱动管理器中，使PAConv比二维卷积具有更大的灵活性，可以更好地处理不规则和无序的点云数据。此外，通过组合权重矩阵而不是从点位置预测核，降低了学习过程的复杂性。

此外，与现有的点云卷积运算不同，他们的网络架构通常是经过精心设计的，我们将我们的PAConv集成到基于经典MLP的点云处理网络中，而不需要改变网络配置。即使建立在简单的网络上，我们的方法仍然接近甚至超过最先进的模型，并显著提高了分类和分割任务的基线性能并且效率相当高。
<br>

## 模型库

| 模型  | Accuracy | Vote Accuracy | 模型下载 | 配置文件 |  日志 |
| :--: | :--------: | :-------------------:| :------: | :-----: | :--: |
|PAConv   | 93.4 |  93.47 | [model]() | [config]() | [log]() \| [vdl]() |

<br>

## 使用教程

下面的教程将从数据准备开始，说明如何训练PAConv模型

### 数据准备

目前Paddle3D中提供的模型支持在ModelNet40数据集上训练，因此需要先准备ModelNet40数据集，请在[官网](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)进行下载。

将数据解压后按照下方的目录结构进行组织

```shell
$ tree modelnet40_ply_hdf5_2048
modelnet40_ply_hdf5_2048
├── ply_data_test0.h5
├── ply_data_test1.h5
├── ply_data_test_0_id2file.json
├── ply_data_test_1_id2file.json
├── ply_data_train0.h5
├── ply_data_train1.h5
├── ply_data_train2.h5
├── ply_data_train3.h5
├── ply_data_train4.h5
├── ply_data_train_0_id2file.json
├── ply_data_train_1_id2file.json
├── ply_data_train_2_id2file.json
├── ply_data_train_3_id2file.json
├── ply_data_train_4_id2file.json
├── shape_names.txt
├── test_files.txt
└── train_files.txt
```

在Paddle3D的目录下创建软链接 `datasets/modelnet40_ply_hdf5_2048`，指向到上面的数据集目录

### 训练

使用如下命令启动训练

```shell

# 每隔10步打印一次训练进度
# 每隔300步保存一次模型，模型参数将被保存在output目录下
python tools/train.py --config configs/paconv/paconv_modelnet40.yml --num_workers 2 --log_interval 10 --save_interval 300 --do_eval
```

### 评估

使用如下命令启动评估

```shell
export CUDA_VISIBLE_DEVICES=0

# 使用Paddle3D提供的预训练模型进行评估
python tools/evaluate.py --config configs/paconv/paconv_modelnet40.yml --num_workers 2 --batch_size 16 --model output/iter_3000/model.pdparams
```

<br>

## 导出部署

使用如下命令导出训练完成的模型

```shell
# 导出Paddle3D提供的预训练模型
python tools/export.py --config configs/paconv/paconv_modelnet40.yml  --model output/iter_70000/model.pdparams
```

### 执行预测

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`paconv.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`paconv.pdiparams`所在路径 |
| input_file | 待预测的点云文件路径 |
| use_gpu | 是否使用GPU进行预测，默认为False|
| precision | 模型精度可设置fp32或fp16。默认fp32 |
| enable_benchmark | 是否开启benchmark |
| batch_size | 批次大小 |
| cpu_threads | cpu线程数 |
| enable_mkldnn | 是否使用mkldnn |


### Python部署

进入代码目录 `deploy/paconv/python`，运行以下命令，执行预测：

* 执行CPU预测

    ```shell
    python infer.py --model_file /path/to/paconv.pdmodel --params_file /path/to/paconv.pdiparams --input_file /path/to/pointcloud --use_gpu=False --batch_size=1
    ```

* 执行GPU预测

    ```shell
    python infer.py --model_file /path/to/paconv.pdmodel --params_file /path/to/paconv.pdiparams --input_file /path/to/pointcloud --use_gpu=True --batch_size=1
    ```


<br>

## 自定义数据集

如果您想在自定义数据集上进行训练，请参考[自定义数据准备教程](../datasets/custom.md)将数据组织成ModelNet40数据格式即可
