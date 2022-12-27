# BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers

## 目录
* [引用](#1)
* [简介](#2)
* [模型库](#3)
* [训练 & 评估](#4)
  * [nuScenes数据集](#41)
* [导出 & 部署](#8)

## <h2 id="1">引用</h2>

```
@article{li2022bevformer,
  title={BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng}
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```

## <h2 id="2">简介</h2>

BEVFormer以多目图像作为输入，输出三维空间里目标物体的位置、大小、方向角以及类别。整体基于DETR3D的架构设计，分为Encoder和Decoder两个部分。Encoder部分以BEV query map、当前帧的多目图像、历史帧的BEV feature map作为输入，输出当前帧的BEV feature map。其中，设计的spatial-cross-attention使用BEV query map去聚合BEV每个3D位置投影到2D多目图像上的特征，设计的temporal-cross-attention使用BEV query map去聚合BEV每个3D位置在历史帧BEV feature map上的特征，使得当前帧的BEV feature map具备时空融合的特征。在Decoder部分，以object queries作为输入，输出其对应的3D bounding box和label。其中，object queries会聚合self-attention特征以及其在Encoder输出的BEV feature map上的特征。目前BEVFormer在nuScenes数据集上的精度依然处于领先水平。


## <h2 id="3">模型库</h2>

- BEVFormer在nuScenes Val set数据集上的表现

| 模型 | 骨干网络 | mAP | NDS | 模型下载 | 配置文件 | 日志 |
| ---- | ------ | --- | ----| ------- |------- | ---- |
| ResNet50-FPN | BEVFormer-tiny | 26.22 | 36.53 | [model](https://paddle3d.bj.bcebos.com/models/bevformer/bevformer_tiny_r50_fpn_nuscenes/model.pdparams) | [config](../../../configs/bevformer/bevformer_tiny_r50_fpn_nuscenes.yml) | [log](https://paddle3d.bj.bcebos.com/models/bevformer/bevformer_tiny_r50_fpn_nuscenes/train.log)\|[vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app?id=062bfe8678693d3f5a63c43eab7a65aa) |

**注意：nuScenes benchmark使用8张V100 GPU训练得出。**


## <h2 id="4">训练 & 评估</h2>

### <h3 id="41">nuScenes数据集</h3>
#### 数据准备

- 目前Paddle3D中提供的BEVFormer模型支持在nuScenes数据集上训练，因此需要先准备nuScenes数据集，请在[官网](https://www.nuscenes.org/nuscenes)进行下载，并将数据集目录准备如下：

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

为加速训练过程中Nuscenes数据集的加载和解析，需要事先将Nuscenes数据集里的标注信息存储在`pkl`后缀文件中，该文件也能用于PERT模型的训练。执行以下命令会生成`petr_nuscenes_annotation_train.pkl`和`petr_nuscenes_annotation_val.pkl`：

```
python tools/create_petr_nus_infos.py
```
生成完后的数据集目录：

```
nuscenes_dataset_root
├── maps
├── samples
├── sweeps
├── v1.0-trainval
├── petr_nuscenes_annotation_train.pkl
├── petr_nuscenes_annotation_val.pkl

```


#### 训练

nuScenes数据集上的训练使用8张GPU：

下载骨干网络的预训练模型参数：
```
wget https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams
```

```
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py --config configs/bevformer/bevformer_tiny.yml --save_dir ./output_bevformer_tiny --num_workers 4 --save_interval 1 --model ./ResNet50_cos_pretrained.pdparams
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。

#### 评估

```
python tools/evaluate.py --config configs/bevformer/bevformer_tiny.yml --model ./output_bevformer_tiny/epoch_24/model.pdparams --num_workers 4
```

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。

## <h2 id="8">导出 & 部署</h2>

### <h3 id="81">模型导出</h3>


运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
python tools/export.py --config configs/bevformer/bevformer_tiny.yml --model ./output_bevformer_tiny/epoch_24/model.pdparams --save_dir ./output_bevformer_tiny_inference
```

| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| model | **[必填]** 训练时保存的模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`bevformer_inference.pdiparams `、`bevformer_inference.pdiparams.info`和`bevformer_inference.pdmodel` |

### <h3 id="82">模型部署</h3>

部署代码开发进行中。
