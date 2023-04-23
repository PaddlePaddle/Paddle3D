# 配置文件详解

Paddle3D支持通过配置文件来描述相关的任务，从而实现配置化驱动的训练、评估、模型导出等流程，Paddle3D的配置化文件具备以下特点：

* 以yaml格式进行编写

* 支持用户配置模型、数据集、训练超参等配置项

* 通过特定的关键字 `type` 指定组件类型，并将其他参数作为实参来初始化组件

* 支持加载PaddleSeg和PaddleDetection中的组件：

  * 在指定类型 `type` 时，加上 `$paddledet.` 前缀即可加载PaddleDetection的组件。

  * 在指定类型 `type` 时，加上 `$paddleseg.` 前缀即可加载PaddleSeg的组件。

## 支持的配置项

| 配置项 | 含义 | 类型 |
| ----- | ---- | :-----: |
|train_dataset |训练数据集 | dict |
|val_dataset |验证数据集 | dict  |
|batch_size|单张卡上，每步迭代训练时的数据量。一般来说，单步训练时的batch_size越大，则样本整体梯度更加稳定，有利于模型的收敛，调大batch_size时往往需要适当调大learning_rate | int |
|iters| 使用一个 batch 数据对模型进行一次参数更新的过程称之为一步，iters 即为训练过程中的训练步数。 | int|
|epochs| 完整遍历一次数据对模型进行训练的过程称之为一次迭代，epochs 即为训练过程中的训练迭代次数。一个epoch包含多个iter | int|
|optimizer|优化器类型，支持飞桨全部的[优化器类型](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#paddle-optimizer) | dict|
|lr_scheduler|调度器类型，支持飞桨全部的[LRScheduler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html) |dict|
|model| 模型类型，所支持值请参考[模型库](./apis/models/)|dict|
|\_\_base\_\_| 基础配置文件，可以不指定，该配置指向另外一个配置文件作为继承的基础配置|str|

## 完整示例

```yaml
# 从另外一个配置文件中继承配置
_base_: '../_base_/kitti_mono.yml'

# 设置batch size为8
batch_size: 8

# 设置训练轮次为70000
iters: 70000

# 指定训练集参数，由于训练集类别在 kitti_mono.yml 中已经指定，此处不需要特殊指定，直接继承
train_dataset:
  # 设置三个Transform对加载的数据进行处理
  transforms:
    - type: LoadImage
      reader: pillow
      to_chw: False
    - type: Gt2SmokeTarget
      mode: train
      num_classes: 3
    - type: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

val_dataset:
  transforms:
    - type: LoadImage
      reader: pillow
      to_chw: False
    - type: Gt2SmokeTarget
      mode: val
      num_classes: 3
    - type: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

# 使用Adam优化器，优化器参数使用默认参数
optimizer:
  type: Adam

# 设置学习率按 指定轮数 进行衰减
lr_scheduler:
  type: MultiStepDecay
  # 衰减轮次分别为 36000 和 55000
  milestones: [36000, 55000]
  # 初始学习率
  learning_rate: 1.25e-4

# 选择SMOKE模型
model:
  type: SMOKE
  backbone:
    # 骨干网络选择DLA34，并从paddle3d的云端存储中下载预训练模型进行加载
    type: DLA34
    pretrained: "https://bj.bcebos.com/paddle3d/pretrained/imagenet/dla34.pdparams"
  head:
    type: SMOKEPredictor
    num_classes: 3
    reg_channels: [1, 2, 3, 2, 2]
    num_channels: 256
    norm_type: "gn"
    in_channels: 64
  depth_ref: [28.01, 16.32]
  dim_ref: [[3.88, 1.63, 1.53], [1.78, 1.70, 0.58], [0.88, 1.73, 0.67]]
  max_detection: 50
  pred_2d: True
```
