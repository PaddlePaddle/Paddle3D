# RTEBEV: Toward Real Time end-to-end Bird's-Eye Perception

## 目录
* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练 & 评估](#4)
  * [nuScenes数据集](#41)
* [导出 & 部署](#8)

## <h2 id="1">引用</h2>


## <h2 id="2">简介</h2>
近年来，基于鸟瞰图(BEV)表示的感知任务越来越受到人们的关注，BEV表示有望成为下一代自动驾驶汽车(AV)感知的基础。而现有的方法计算资源大或依赖于非最大抑制(NMS)，无法达到理想的性能。在本文中，我们将两种方法结合起来，设计了一种实时端到端方法，称为RTE-BEV。RTE-BEV在NuScenes val数据集上达到51.4 NDS，在T4 GPU上达到35.46 FPS。


## <h2 id="3">模型库</h2>

- rtebev在nuScenes Val set数据集上的表现

| 模型 | 相邻帧数 | mAP | NDS | 模型下载 | 配置文件 | 日志 |
| ---- | ------ | --- | ----| ------- |------- | ---- |
| rtebev_r50 | 无 | 34.30 | 42.29 | [model](https://paddle3d.bj.bcebos.com/models/rtebev/rtebev_r50_nuscenes/model.pdparams) | [config](../../../configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_cgbs.yml) | [log](https://paddle3d.bj.bcebos.com/models/rtebev/rtebev_r50_nuscenes/train.log)|
| rtebev_r50_1f | 1 | 36.55 | 48.52 | [model](https://paddle3d.bj.bcebos.com/models/rtebev/rtebev_r50_nuscenes_1f/model.pdparams) | [config](../../../configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_1f_cgbs.yml) | [log](https://paddle3d.bj.bcebos.com/models/rtebev/rtebev_r50_nuscenes_1f/train.log)|
| rtebev_r50_4f | 4 | 39.66 | 50.19 | [model](https://paddle3d.bj.bcebos.com/models/rtebev/rtebev_r50_nuscenes_4f/model.pdparams) | [config](../../../configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_4f_cgbs.yml) | [log](https://paddle3d.bj.bcebos.com/models/rtebev/rtebev_r50_nuscenes_4f/train.log)|

**注意：nuScenes benchmark使用8张V100 GPU训练得出。**


## <h2 id="4">训练 & 评估</h2>

### <h3 id="41">nuScenes数据集</h3>
#### 数据准备

请下载Nuscenes测数据集, 下载作者提供的annotion文件。

下载好后的数据集目录结构
```
nuscenes
   ├── maps
   ├── samples
   ├── sweeps
   ├── v1.0-trainval
   ├── v1.0-test
   ...
```
将nuscenes数据软链至data/nuscenes，或更改配置文件数据集路径。

数据集标注打包请参考原版BEVDet实现中的数据打包脚本[tools/create_data_bevdet.py](https://github.com/HuangJunJie2017/BEVDet/blob/dev2.1/tools/create_data_bevdet.py)

生成完后的数据集目录
```
nuscenes
   ├── maps
   ├── samples
   ├── sweeps
   ├── v1.0-trainval
   ├── v1.0-test
   ├── bevdetv2-nuscenes_annotation_train.pkl
   ├── bevdetv2-nuscenes_annotation_val.pkl
```
为了方便，我们提供了生成好的annotation文件
| 文件名称 | 下载链接 |
| -- | -- |
| bevdetv2-nuscenes_annotation_train.pkl | [下载](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/dataset/bevdetv2-nuscenes_infos_train.pkl) |
| bevdetv2-nuscenes_annotation_val.pkl | [下载](https://paddle3d.bj.bcebos.com/models/bevdet/BEVDet2.0_depth_baseline/dataset/bevdetv2-nuscenes_infos_val.pkl) |

#### 训练

nuScenes数据集上的训练使用8张GPU：

下载骨干网络的预训练模型参数：
```
wget https://paddle3d.bj.bcebos.com/models/rtebev/resnet50-0676ba61.pdparams
```

```
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_1f_cgbs.yml --save_dir ./output_rtebev --num_workers 4 --save_interval 1 --model ./resnet50-0676ba61.pdparams
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。

#### 评估

```
python tools/evaluate.py --config configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_1f_cgbs.yml --model ./output_rtebev/epoch_20/model.pdparams --num_workers 4 --batch_size 1
```

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。

## <h2 id="8">导出 & 部署</h2>

### <h3 id="81">模型导出</h3>


运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
FLAGS_deploy=True python tools/export.py --config configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_1f_cgbs.yml --model ./output_rtebev/epoch_20/model.pdparams --save_dir ./output_rtebev_inference
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`model.pdiparams `、`model.pdiparams.info`和`model.pdmodel` |

### <h3 id="82">模型部署</h3>

### Python部署

部署代码所在路径为deploy/rtebev/python

当num_adj参数大于0时，表示使用相邻帧，使用infer_mf_onnxtrt.py或者infer_mf_paddletrt.py进行推理。对应的配置文件为configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_1f_cgbs.yml、configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_4f_cgbs.yml。

当num_adj参数等于0时，表示不使用相邻帧，使用infer_onnxtrt.py或者infer_paddletrt.py进行推理。对应的配置文件为configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_cgbs.yml。

使用相邻帧信息可以提高准确度。

**注意：目前rtebev仅支持使用GPU进行推理。**

### 使用paddle 推理nuscenes eval
命令参数说明如下：

| 参数 | 说明 |
| -- | -- |
| config | 配置文件的所在路径  |
| batch_size | eval推理的batch_size，默认1 |
| model | 预训练模型参数所在路径  |
| num_workers | eval推理的num_workers，默认2  |
| model_file | 导出模型的结构文件`rtebev.pdmodel`所在路径 |
| params_file | 导出模型的参数文件`rtebev.pdiparams`所在路径 |
| use_trt | 是否使用TensorRT进行加速，默认False|
| trt_precision | 当use_trt设置为1时，模型精度可设置0或1，0表示fp32, 1表示fp16。默认0 |
| trt_use_static | 当trt_use_static设置为True时，**在首次运行程序的时候会将TensorRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成**。默认0 |
| trt_static_dir | 当trt_use_static设置为1时，保存优化信息的路径 |
| collect_shape_info | 是否收集模型动态shape信息。默认False。**只需首次运行，后续直接加载生成的shape信息文件即可进行TensorRT加速推理** |
| dynamic_shape_file | 保存模型动态shape信息的文件路径。 |
| quant_config | 量化配置文件的所在路径  |

使用paddle inference，执行预测：

```
python deploy/rtebev/python/infer_mf_paddletrt.py --config /path/to/config.yml --model /path/to/model.pdparams --model_file /path/to/rtebev.pdmodel --params_file /path/to/rtebev.pdiparams
```

使用paddle trt，执行预测：

收集shape
```
python deploy/rtebev/python/infer_mf_paddletrt.py --config /path/to/config.yml --model /path/to/model.pdparams --model_file /path/to/rtebev.pdmodel --params_file /path/to/rtebev.pdiparams --use_trt --collect_shape_info
```
trt-fp16推理
```
python deploy/rtebev/python/infer_mf_paddletrt.py --config /path/to/config.yml --model /path/to/model.pdparams --model_file /path/to/rtebev.pdmodel --params_file /path/to/rtebev.pdiparams --use_trt --trt_precision 1
```

### 使用onnx-trt 推理nuscenes eval
1. 安装develop paddle2onnx，可参考https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/docs/zh/compile.md

2. onnx导出
```
FLAGS_onnx=True FLAGS_deploy=True python tools/export.py --config configs/rtebev/rtebev_r50_nuscenes_256x704_msdepth_hybird_1f_cgbs.yml --model ./output_rtebev/epoch_20/model.pdparams --save_dir ./output_rtebev_inference

```

3. 编译插入自定义算子
```
cd plugin_ops
# 修改install.sh中TRT安装位置
sh install.sh
cd ..
```

4. onnx转engine
```
/path/to/TensorRT/bin/trtexec --onnx=/path/to/model.onnx  --fp16 --saveEngine=/path/to/model.engine --buildOnly --plugins=/path/to/plugin_ops/build/libtensorrt_ops.so
```

5. trt-fp16推理

```
python deploy/rtebev/python/infer_mf_onnxtrt.py --config /path/to/config.yml --model /path/to/model.pdparams --engine /path/to/model.engine --plugin /path/to/plugin_ops/build/libtensorrt_ops.so
```
