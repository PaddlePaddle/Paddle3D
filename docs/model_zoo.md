## 模型性能指标

### 测试环境

    * GPU: Tesla V100

    * CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

    * CUDA: 10.1

    * cuDNN: 7.6

    * TensorRT: 6.0.1.5

    * Paddle: 2.3.1

### KITTI数据指标

|模型|类别|mAP@ Easy / Mod / Hard| FP32(FPS) | FP16(FPS) |
|-|-|-|-|-|
|SMOKE-DLA34|Car|6.26 / 5.16 / 4.54|63.5|64.9|
||Cyclist|3.04 / 2.73 / 2.23|||
||Pedestrian|1.69 / 0.95 / 0.94|||
|CaDDN-HRNet18|Car|22.50  / 15.78  / 13.95|18.5|21.3|
||Cyclist|10.09  / 7.12 /  5.57|||
||Pedestrian|1.27  / 0.69  / 0.69|||
|PointPillars|Car|86.90 / 75.21 / 71.57|37.3|40.5|
||Cyclist|84.36 / 64.66 / 60.53|30.0|30.2|  
||Pedestrian|66.13 / 60.36 / 54.40|||  
|CenterPoint|Car|85.99 / 76.69 / 73.62|43.96|74.21|
||Cyclist|84.30 / 63.52 / 59.47|||
||Pedestrian|57.66 / 54.03 / 49.75|||


## 模型使用教程

  * [SMOKE](./models/smoke)

  * [CaDDN](./models/caddn)

  * [PointPillars](./models/pointpillars)

  * [CenterPoint](./docs/models/centerpoint)

  * [SequeezeSeg v3](./models/squeezesegv3)
