# SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation

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

> Xu, Chenfeng, Bichen Wu, Zining Wang, Wei Zhan, Peter Vajda, Kurt Keutzer, και Masayoshi Tomizuka. ‘SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation’. CoRR abs/2004.01803 (2020).

## <h2 id="2">简介</h2>
SqueezeSegV3是一个点云语义分割模型。该论文延续了SqueezeSeg系列将三维空间点云投影至二维空间进行特征提取的思想，并在RangeNet++模型结构的
基础上创新性地提出并应用了空间自适应卷积（Spatially-Adaptive Convolution）。

## <h2 id="3">模型库</h2>
- SqueezeSegV3在SemanticKITTI Val set数据集上的表现

|       模型        | mIoU (Point Cloud / Range View) | mAcc (Point Cloud / Range View) |                                                       模型下载                                                       |                                       配置文件                                        |
|:---------------:|:-------------------------------:|:-------------------------------:|:----------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
| SqueezeSegV3-21 |           46.3 / 51.2           |           87.3 / 90.0           | [model](https://bj.bcebos.com/paddle3d/models/squeezesegv3/squeezesegv3_rangenet21_semantickitti/model.pdparams) | [config](../../../configs/squeezesegv3/squeezesegv3_rangenet21_semantickitti.yml) |
| SqueezeSegV3-53 |           48.8 / 54.2           |           88.4 / 91.2           | [model](https://bj.bcebos.com/paddle3d/models/squeezesegv3/squeezesegv3_rangenet53_semantickitti/model.pdparams) | [config](../../../configs/squeezesegv3/squeezesegv3_rangenet53_semantickitti.yml) |

## <h2 id="4">训练配置</h2>
我们提供了在开源数据集上的训练配置与结果，详见[SqueezeSegV3训练配置](../../../configs/squeezesegv3)。

## <h2 id="5">使用教程</h2>

### <h3 id="51">数据准备</h3>
1. 数据格式  
SqueezeSegV3模型目前仅适配[SemanticKITTI](http://semantic-kitti.org/dataset.html)格式的数据集。需将数据集放置于
`Paddle3D/datasets/SemanticKITTI`目录下，或在[配置文件](../../../configs/_base_/semantickitti.yml)中指定数据集路径。数据集文件结构如下：  
```
└── Paddle3D/datasets/SemanticKITTI
    ├── sqeuences
        ├── 00
            ├── velodyne
                ├── 000000.bin
                ├── ...
            ├── labels
                ├── 000000.label
                ├── ...
            ├── poses.txt
```

2. 数据划分  
SemanticKITTI数据集共包含`00`至`21`共22个序列，其中官方默认的数据集划分为：
   - 训练集：00, 01, 02, 03, 04, 05, 06, 07, 09, 10
   - 验证集：08
   - 测试集：11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21  

    如需使用自己的划分，可在[配置文件](../../../configs/_base_/semantickitti.yml)中指定。

### <h3 id="52">训练</h3>
位于`Paddle3D/`目录下，执行：
```shell
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
    tools/train.py \
    --config configs/squeezesegv3/squeezesegv3_rangenet53_semantickitti.yml \
    --save_interval 1195 \
    --keep_checkpoint_max 50 \
    --save_dir outputs/squeezesegv3/rangenet21_semantickitti \
    --do_eval
```

训练脚本支持设置如下参数：

| 参数名                 | 用途                             | 是否必选项 | 默认值      |
|:--------------------|:-------------------------------|:------|:---------|
| gpus                | 使用的GPU编号                       | 是     | -        |
| config              | 配置文件                           | 是     | -        |
| save_dir            | 模型和visualdl日志文件的保存根路径          | 否     | output   |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 | 否     | 0        |
| save_interval       | 模型保存的间隔步数                      | 否     | 1000     |
| do_eval             | 是否在保存模型时进行评估                   | 否     | 否        |
| log_interval        | 打印日志的间隔步数                      | 否     | 10       |
| keep_checkpoint_max | 最新模型保存个数                       | 否     | 5        |
| resume              | 是否从断点恢复训练                      | 否     | 否        |
| batch_size          | mini-batch大小（每张GPU）            | 否     | 在配置文件中指定 |
| iters               | 训练轮数                           | 否     | 在配置文件中指定 |
| learning_rate       | 学习率                            | 否     | 在配置文件中指定 |
| seed                | 固定随机种子                         | 否     | None     |

### <h3 id="53">评估</h3>
位于`Paddle3D/`目录下，执行：
```shell
python tools/evaluate.py \
    --config configs/squeezesegv3/squeezesegv3_rangenet53_semantickitti.yml \
    --model /path/to/model.pdparams
```

评估脚本支持设置如下参数：

| 参数名                 | 用途                             | 是否必选项 | 默认值      |
|:--------------------|:-------------------------------|:------|:---------|
| config              | 配置文件                           | 是     | -        |
| model               | 待评估模型路径                        | 是     | -        |
| num_workers         | 用于异步读取数据的进程数量， 大于等于1时开启子进程读取数据 | 否     | 0        |
| batch_size          | mini-batch大小                   | 否     | 在配置文件中指定 |


### <h3 id="54">模型导出</h3>

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```shell
python tools/export.py \
    --config configs/squeezesegv3/squeezesegv3_rangenet53_semantickitti.yml \
    --model /path/to/model.pdparams \
    --input_shape 64 1024 \
    --save_dir /path/to/output
```

模型导出脚本支持设置如下参数：

| 参数名         | 用途                                                                                                           | 是否必选项 | 默认值      |
|:------------|:-------------------------------------------------------------------------------------------------------------|:------|:---------|
| config      | 配置文件                                                                                                         | 是     | -        |
| model       | 待导出模型参数`model.pdparams`路径                                                                                    | 是     | -        |
| input_shape | 指定模型的输入尺寸，支持`N, C, H, W`或`H, W`格式                                                                            | 是     | -        |
| save_dir    | 保存导出模型的路径，`save_dir`下将会生成三个文件：`squeezesegv3.pdiparams `、`squeezesegv3.pdiparams.info`和`squeezesegv3.pdmodel` | 否     | `deploy` |



### <h3 id="55">模型部署</h3>

#### C++部署

Coming soon...

#### Python部署

命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| model_file | 导出模型的结构文件`squeezesegv3.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`squeezesegv3.pdiparams`所在路径 |
| lidar_file | 待预测的点云文件所在路径 |
| img_mean | 点云投影到range-view后所成图像的均值，例如为`12.12,10.88,0.23,-1.04,0.21` |
| img_std | 点云投影到range-view后所成图像的方差，例如为`12.32,11.47,6.91,0.86,0.16` |
| use_trt | 是否使用TensorRT进行加速，默认0|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为1时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |


运行以下命令，执行预测：

```
python infer.py --model_file /path/to/squeezesegv3.pdmodel --params_file /path/to/squeezesegv3.pdiparams --lidar_file /path/to/lidar.pcd.bin --img_mean 12.12,10.88,0.23,-1.04,0.21 --img_std 12.32,11.47,6.91,0.86,0.16
```

如果要开启TensorRT的话，请卸载掉原有的`paddlepaddel_gpu`，至[Paddle官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#python)下载与TensorRT连编的预编译Paddle Inferece安装包，选择符合本地环境CUDA/cuDNN/TensorRT版本的安装包完成安装即可。

运行以下命令，开启TensorRT加速模型预测：

```
python infer.py --model_file /path/to/squeezesegv3.pdmodel --params_file /path/to/squeezesegv3.pdiparams --lidar_file /path/to/lidar.pcd.bin --img_mean 12.12,10.88,0.23,-1.04,0.21 --img_std 12.32,11.47,6.91,0.86,0.16 --use_trt 1
```
