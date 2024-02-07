# 动态图量化训练

## 目录
* [量化训练 & 评估](#4)
* [量化模型导出](#8)

## <h2 id="4">量化训练 & 评估</h2>

#### 数据准备
参考具体各个模型的数据集准备

#### 量化训练

以BEVFormer在nuScenes数据集上的量化训练为例：

使用已训练好的模型为预训练模型参数。
```
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py --config configs/bevformer/bevformer_tiny_r50_fpn_nuscenes.yml --quant_config configs/quant/bevformer_PACT.yml --save_dir ./output_bevformer_tiny_quant --num_workers 4 --save_interval 1 --model ./output_bevformer_tiny/epoch_24/model.pdparams
```

训练启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型训练)。

#### 评估

```
python tools/evaluate.py  --quant_config configs/quant/bevformer_PACT.yml --config configs/bevformer/bevformer_tiny_r50_fpn_nuscenes.yml --model ./output_bevformer_tiny_quant/epoch_5/model.pdparams --num_workers 4
```

评估启动参数介绍可参考文档[全流程速览](../../quickstart.md#模型评估)。

## <h2 id="8">量化模型导出</h2>

运行以下命令，将训练时保存的动态图模型文件导出成推理引擎能够加载的静态图模型文件。

```
python tools/export.py --quant_config configs/quant/bevformer.yml --config configs/bevformer/bevformer_tiny.yml --model ./output_bevformer_tiny_quant/epoch_5/model.pdparams --save_dir ./output_bevformer_tiny_quant_inference
```
**注意：模型导出无论是否使用PACT量化训练方法，导出模型的配置文件一律删除PACT配置，参考configs/quant/bevformer.yml和configs/quant/bevformer_PACT.yml可知。**
| 参数 | 说明 |
| -- | -- |
| config | **[必填]** 训练配置文件所在路径 |
| quant_config | **[必填]** 量化配置文件所在路径 |
| model | **[必填]** 训练时保存的量化模型文件`model.pdparams`所在路径 |
| save_dir | **[必填]** 保存导出模型的路径，`save_dir`下将会生成三个文件：`bevformer_inference.pdiparams `、`bevformer_inference.pdiparams.info`和`bevformer_inference.pdmodel` |

- 量化的BEVFormer在nuScenes Val set数据集上的表现

| 模型 |  mAP | NDS |
| ---- |  --- | ----|
| 原模型| 26.22 | 36.53 |
| 量化模型 |  26.35 | 36.54 |
