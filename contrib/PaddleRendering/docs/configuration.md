# 配置文件详解

PaddleRendering支持通过配置文件来描述相关的任务，从而实现配置化驱动的训练、评估、模型导出等流程，PaddleRendering的配置化文件具备以下特点：

* 以yaml格式进行编写

* 支持用户自定义模型、数据集、训练超参等配置项

* 通过特定的关键字 `type` 指定组件类型，并将其他参数作为实参来初始化组件

## 支持的配置项

| 配置项 | 含义 | 类型 |
| ----- | ---- | :-----: |
| iters | 最大迭代步数 | int |
| image_batch_size | 图片 mini-batch 大小，每次迭代从该参数指定数量的图片中采样光线。如果为`-1`，每次迭代从整个数据集的图片中采样光线 | int |
| ray_batch_size | 光线 mini-batch 大小，每次迭代从图片 batch 中采样光线的数量 | int |
| image_resampling_interval | 为提升训练性能，我们使用了图片预加载策略（将图片 mini-batch 预加载到GPU上缓存，每次迭代从缓存的图片中采样光线），该参数用于指定对GPU上图片缓存进行更新的间隔。如果为`-1`，图片缓存不会被更新（适用于`image_batch_size`为`-1`的情况） | int |
| use_adaptive_ray_batch_size | 是否采用自动调整光线 mini-batch 大小的训练策略。如果开启，将确保模型每次迭代处理的有效样本数稳定在`2^18`，有助于提升模型收敛效率，缩短训练时间。                                                    | bool  |
| amp_cfg | 使用混合精度训练策略，用于训练加速。 | dict |
| grad_accum_cfg | 使用梯度累积更新策略，方便您在显存不足时使用更大的等效光线 mini-batch 大小。建议您使用累积更新时同时减小学习率。 | dict |
| train_metrics | 训练集评估指标，支持PSNRMeter。 | dict |
| val_metrics | 验证集评估指标，支持PSNRMeter和SSIMMeter。 | dict |
| train_dataset | 训练数据集 | dict |
| val_dataset | 验证数据集 | dict  |
|optimizer|优化器类型，支持飞桨全部的[优化器类型](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html#paddle-optimizer)。 | dict |
|lr_scheduler|调度器类型，支持飞桨全部的[LRScheduler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LRScheduler_cn.html) | dict |
|model| 模型类型，所支持值请参考[模型库](../pprndr/models/)。 | dict |
|\_\_base\_\_| 基础配置文件，可以不指定，该配置指向另外一个配置文件作为继承的基础配置|str|


## 完整示例

```yaml
# 设置最大迭代步数为20000
iters: 20000

# 每步迭代在所有图像中采样256条光线。使用自适应光线 mini-batch 大小
image_batch_size: -1  
ray_batch_size: 256
image_resampling_interval: -1
use_adaptive_ray_batch_size: True

# 使用自动混合精度训练
amp_cfg:
  enable: True
  level: O1
  scaler:
    init_loss_scaling: 1024.0
    incr_every_n_steps: 3000

# 设置训练集评估指标
train_metrics:
  - type: PSNRMeter

# 设置验证集评估指标
val_metrics:
  - type: PSNRMeter
  - type: SSIMMeter

# 设置训练集
train_dataset:
  type: BlenderDataset
  dataset_root: data/nerf_synthetic/lego
  camera_scale_factor: 1.0
  background_color: white
  # 设置三个Transform对加载的数据进行处理
  transforms:
    - type: LoadImage
    - type: Normalize
    - type: AlphaBlending
  split: train

# 设置测试集
val_dataset:
  type: BlenderDataset
  dataset_root: data/nerf_synthetic/lego
  camera_scale_factor: 1.0
  background_color: white
  transforms:
    - type: LoadImage
    - type: Normalize
    - type: AlphaBlending
  split: val

# 使用Adam优化器，提供合精度支持
optimizer:
  type: Adam
  beta1: .9
  beta2: .999
  epsilon: 1.0e-15
  weight_decay: 1.0e-6
  multi_precision: True

# 设置学习率线性暖启动，并按指定轮数进行衰减
lr_scheduler:
  type: LinearWarmup
  learning_rate:
    type: MultiStepDecay
    # 衰减起始学习率
    learning_rate: 0.01
    # 衰减轮数
    milestones: [8000, 13000, 16000]
    # 衰减系数
    gamma: .33
  # 暖启动步数
  warmup_steps: 2000
  # 暖启动起始学习率
  start_lr: .001
  # 暖启动最终学习率
  end_lr: .01

# 设置模型为InstantNGP
aabb: &aabb [-1.3, -1.3, -1.3, 1.3, 1.3, 1.3]
model:
  type: InstantNGP
  # 设置光线采样策略
  ray_sampler:
    type: VolumetricSampler
    occupancy_grid:
      type: OccupancyGrid
      resolution: 128
      contraction_type: 0
      aabb: *aabb
    grid_update_interval: 16
    step_size: .005
  # 辐射场相关设置
  field:
    type: InstantNGPField
    dir_encoder:
      type: SHEncoder
      input_dim: 3
      degree: 3
    pos_encoder:
      type: GridEncoder
      input_dim: 3
      num_levels: 16
      level_dim: 2
      per_level_scale: 1.4472692012786865
      base_resolution: 16
      log2_hashmap_size: 19
    density_net:
      type: FFMLP
      input_dim: 32
      output_dim: 16
      hidden_dim: 64
      num_layers: 3
    color_net:
      type: FFMLP
      input_dim: 32
      output_dim: 3
      hidden_dim: 64
      num_layers: 4
      output_activation: sigmoid
    aabb: *aabb
    contraction_type: 0
  rgb_renderer:
    type: RGBRenderer
    background_color: white
  rgb_loss:
    type: SmoothL1Loss

```
