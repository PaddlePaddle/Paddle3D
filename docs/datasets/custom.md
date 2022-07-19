# 自定义数据集格式说明

Paddle3D支持按照[KITTTI数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)格式构建自己的数据集，目录结构示意如下：

```
custom_dataset
|—— training
|   |—— image_2
|   |   |—— 000001.png
|   |   |—— ...
|   |—— label_2
|   |   |—— 000001.txt
|   |   |—— ...
|   |—— calib
|   |   |—— 000001.txt
|   |   |—— ...
|   |—— velodyne
|   |   |—— 000001.bin
|   |   |—— ...
|—— ImageSets
|   |—— train.txt
|   |—— val.txt
```

`image_2`、`velodyne`、`label_2`和`calib`存放图像文件、点云文件、真值标注文件、坐标系转换参数文件，4个文件夹下对应同一帧的文件名前缀需相同。`ImageSets`目录存放划分至训练集和验证集的文件名前缀列表。

**注意**：单目三维物体检测任务可以不准备`velodyne`，点云三维物体检测任务可以不准备`image_2`。

- `label_2`

真值标注文件`000001.txt`示意如下：

```
Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Cyclist 0.00 3 -1.65 676.60 163.95 688.98 193.93 1.86 0.60 2.02 4.59 1.32 45.84 -1.55
DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 511.35 174.96 527.81 187.45 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 532.37 176.35 542.68 185.27 -1 -1 -1 -1000 -1000 -1000 -10
DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1000 -1000 -1000 -10
```

标注格式说明如下，具体可见[KITTI官方真值说明工具](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip)。**类别（type）可根据实际情况取名。**

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```
**注意：** 真值标注给的是摄像头坐标系下的，而不是激光坐标系下的。KITTI各坐标系说明可参考[KITTI Coordinate Transformations](https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb)。

- `calib`

坐标系转换参数文件`000001.txt`示例如下：

```
P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03
R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
```

`P2`是摄像头的内参，`R0_rect`是摄像头的外参，该摄像头产生的图片位于`image_2`目录下。如果没有多个摄像头，则`P0`、`P1`、`P3`皆可重复填写成`P2`。`Tr_velo_to_cam`是激光传感器到相机坐标系的转换矩阵。`Tr_imu_to_velo`是IMU到激光传感器的坐标转换矩阵。当前`P0`、`P1`、`P3`和`Tr_imu_to_velo`在Paddle3D中暂未参与计算，但为了适配数据集读取，需提供这四个参数。

- `ImageSets`

训练集列表`train.txt`示意如下：
```
000000
000003
000007
000009
000010
... ...
```

验证集列表`val.txt`示意如下：
```
000001
000002
000004
000005
000006
... ...
```
