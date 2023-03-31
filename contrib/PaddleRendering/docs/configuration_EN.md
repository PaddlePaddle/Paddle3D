# Configuration File Details

PaddleRendering supports configuration files to describe related tasks, so as to realize configuration-driven training, evaluation, model export and other processes. PaddleRendering configuration files have the following characteristics:

* Write in yaml format

* Support user-defined models, datasets, training hyperparameters, etc.

* Specify the component type through the specific keyword `type`, and initialize the component with other parameters as arguments

## Supported Configurations

| Configuration | Explanation  | Type |
| ----- | ---- | :-----: |
| iters | The number of iterations. | int |
| image_batch_size | Batch size of images, from which rays are sampled every iteration.<br>If `-1`, rays are sampled from the entire dataset. | int |
| ray_batch_size | Batch size of rays, the number of rays sampled from image mini-batch every iteration. | int |
| image_resampling_interval | To accelerate training, each GPU maintains a image buffer (image mini-batch is prefetched, rays are sampled from the buffer every iteration).<br>This argument specifies the interval of updating the image buffer. If `-1`, the buffer is never updated (for the case where `image_batch_size` is `-1`). | int |
| use_adaptive_ray_batch_size | Whether to use an adaptive `ray_batch_size`.<br>If enabled, the number of valid samples fed to the model is stable at `2^18`, which accelerates model convergence.                                                         | bool |
| amp_cfg | Use automatic mixed precision training strategy for training acceleration. | dict |
| grad_accum_cfg | Using a gradient accumulative update strategy allows you to use a larger equivalent ray mini-batch size when memory is insufficient. It is recommended that you reduce the learning rate while using accumulative updates. | dict |
| train_metrics | Training set evaluation metrics, support PSNRMeter. | dict |
| val_metrics | Validation set evaluation metrics, support PSNRMeter and SSIMMeter. | dict |
| train_dataset | Training set. | dict |
| val_dataset | Validation set. | dict |
|optimizer| Optimizer type, supporting all [optimizer types](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/optimizer/Optimizer_en.html) of PaddlePaddle. | dict |
|lr_scheduler| Scheduler type, supporting all [LRScheduler](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/optimizer/lr/LRScheduler_en.html#lrscheduler) of PaddlePaddle. | dict |
|model| Model type, please refer to [models](../pprndr/models/). | dict |
|\_\_base\_\_| The basic configuration file (optional), which points to another configuration file as the inherited basic configuration. | str |

## Example
```yaml
# Set the maximum number of iteration steps to 20000
iters: 20000

# Each iteration samples 256 rays across all images. Using adaptive ray batch size
image_batch_size: -1  
ray_batch_size: 256
image_resampling_interval: -1
use_adaptive_ray_batch_size: True

# Training with Auto-Mixed-Precision
amp_cfg:
  enable: True
  level: O1
  scaler:
    init_loss_scaling: 1024.0
    incr_every_n_steps: 3000

# Set up training set evaluation metrics
train_metrics:
  - type: PSNRMeter

# Set up validation set evaluation metrics
val_metrics:
  - type: PSNRMeter
  - type: SSIMMeter

# Set training set
train_dataset:
  type: BlenderDataset
  dataset_root: data/nerf_synthetic/lego
  camera_scale_factor: 1.0
  background_color: white
  # Set up three Transforms to process the loaded data
  transforms:
    - type: LoadImage
    - type: Normalize
    - type: AlphaBlending
  split: train

# Set validation set
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

# Using the Adam optimizer, providing multi_precision support
optimizer:
  type: Adam
  beta1: .9
  beta2: .999
  epsilon: 1.0e-15
  weight_decay: 1.0e-6
  multi_precision: True

# Set the learning rate to linear warm up start and decay by the specified number of iterastions
lr_scheduler:
  type: LinearWarmup
  learning_rate:
    type: MultiStepDecay
    # Starting learning rate to decay
    learning_rate: 0.01
    # Specified number of decay iterations
    milestones: [8000, 13000, 16000]
    # Decay factor
    gamma: .33
  # Warm up start steps
  warmup_steps: 2000
  # Starting learning rate to warm up start
  start_lr: .001
  # Ending learning rate to warm up start
  end_lr: .01

# Set the model to InstantNGP
aabb: &aabb [-1.3, -1.3, -1.3, 1.3, 1.3, 1.3]
model:
  type: InstantNGP
  # Set the ray sampling strategy
  ray_sampler:
    type: VolumetricSampler
    occupancy_grid:
      type: OccupancyGrid
      resolution: 128
      contraction_type: 0
      aabb: *aabb
    grid_update_interval: 16
    step_size: .005
  # Set radiance field
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
