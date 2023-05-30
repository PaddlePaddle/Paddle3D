## 3D Visualization Examples of LiDAR Point Cloud/BEV and Camera Images
### Environment Setup
Follow the [official documentation](https://github.com/PaddlePaddle/Paddle3D/blob/develop/docs/installation.md) to install the dependencies for Paddle3D. Then install mayavi for laser point cloud visualization using the following commands:
```
pip install vtk==8.1.2
pip install mayavi==4.7.4
pip install PyQt5
```
### 3D Visualization Example of Camera Images
The 3D visualization files for camera images are located in `demo/visualization_demo/`. The folder provides two examples: `mono_vis_single_frame_demo.py` for visualizing a single frame and `mono_vis_multi_frame_demo.py` for visualizing multiple frames. Both examples use the same visualization interface, which is defined in `paddle3d.apis.infer`.

The implementation methods of `mono_vis_single_frame_demo.py` and `mono_vis_multi_frame_demo.py` are different to provide more visualization options. In `mono_vis_single_frame_demo.py`, visualization is achieved using the inference deployment with Paddle, while `mono_vis_multi_frame_demo.py` performs sequential frame reading and inference by constructing a dataloader for visualization.

To use `mono_vis_single_frame_demo.py`, use the following command:
```
cd demo/visualization_demo
python mono_vis_single_frame_demo.py \
  --model_file model/smoke.pdmodel \
  --params_file model/smoke.pdiparams \
  --image data/image_2/000008.png
```
`--model_file` and `--params_file` are the paths to the model parameter files being used.

`--image` is the path of the input image.

To use `mono_vis_multi_frame_demo.py`, use the following command:

```
python mono_vis_multi_frame_demo.py \
  --config configs/smoke/smoke_dla34_no_dcn_kitti.yml \
  --model demo/smoke.pdparams \
  --batch_size 1
```

`--config` is the path to the model configuration file.

`--model` is the path to the model parameter file being used.

`--batch_size` is the batch size for inference.

The final output of the monocular visualization is shown below:

![](img/mono.jpg)
### 3D Visualization Example of LiDAR Point Cloud
The 3D visualization files for LiDAR point clouds are located in `demo/visualization_demo/`. The folder provides two examples: `pcd_vis_single_frame_demo.py` for visualizing a single frame and `pcd_vis_multi_frame_demo.py` for visualizing multiple frames. Both examples use the same visualization interface, which is defined in `paddle3d.apis.infer`.

The implementation methods of `pcd_vis_single_frame_demo.py` and `pcd_vis_multi_frame_demo.py` are different to provide more visualization options. In `pcd_vis_single_frame_demo.py`, visualization is achieved using the inference deployment with Paddle, while `pcd_vis_multi_frame_demo.py` performs sequential frame reading and inference by constructing a dataloader for visualization.

To use `pcd_vis_single_frame_demo.py`, use the following command:

```
cd demo/visualization_demo
python pcd_vis_single_frame_demo.py \
  --model_file model/pointpillars.pdmodel \
  --params_file model/pointpillars.pdiparams \
  --lidar_file data/velodyne/000008.bin \
  --calib_file data/calib/000008.txt \
  --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
  --voxel_size .16 .16 4 \
  --max_points_in_voxel 32 \
  --max_voxel_num 40000
```
`--model_file` and `--params_file` are the paths to the model parameter files being used.

`--lidar_file` and `--calib_file` are the paths to the LiDAR point cloud and its corresponding calibration file.

`--point_cloud_range` represents the range of the LiDAR point cloud in `(x, y, z)` coordinates.

`--voxel_size` represents the size of the voxels used in processing.

`--max_points_in_voxel` is the maximum number of LiDAR points in each voxel.

`--max_voxel_num` is the maximum number of voxels.


To use `pcd_vis_multi_frame_demo.py`, use the following command:

```
python pcd_vis_multi_frame_demo.py \
  --config configs/pointpillars/pointpillars_xyres16_kitti_car.yml \
  --model demo/pointpillars.pdparams \
  --batch_size 1
```

`--config` is the path to the model configuration file.

`--model` is the path to the model parameter file being used.

`--batch_size` is the batch size for inference.

The final output of the lidar point cloud visualization is shown below:

![](img/pc.png)
### 3D Visualization Example of BEV
The 3D visualization files for LiDAR BEV are located in `demo/visualization_demo/`. The folder provides two examples: `bev_vis_single_frame_demo.py` for visualizing a single frame and `bev_vis_multi_frame_demo.py` for visualizing multiple frames. Both examples use the same visualization interface, which is defined in `paddle3d.apis.infer`.

The implementation methods of `bev_vis_single_frame_demo.py` and `bev_vis_multi_frame_demo.py` are different to provide more visualization options. In `bev_vis_single_frame_demo.py`, visualization is achieved using the inference deployment with Paddle, while `bev_vis_multi_frame_demo.py` performs sequential frame reading and inference by constructing a dataloader for visualization.

To use `bev_vis_single_frame_demo.py`, use the following command:

```
cd demo/visualization_demo
python bev_vis_single_frame_demo.py \
  --model_file model/pointpillars.pdmodel \
  --params_file model/pointpillars.pdiparams \
  --lidar_file data/velodyne/000008.bin \
  --calib_file data/calib/000008.txt \
  --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
  --voxel_size .16 .16 4 \
  --max_points_in_voxel 32 \
  --max_voxel_num 40000
```
`--model_file` and `--params_file` are the paths to the model parameter files being used.

`--lidar_file` and `--calib_file` are the paths to the LiDAR point cloud and its corresponding calibration file.

`--point_cloud_range` represents the range of the LiDAR point cloud in `(x, y, z)` coordinates.

`--voxel_size` represents the size of the voxels used in processing.

`--max_points_in_voxel` is the maximum number of LiDAR points in each voxel.

`--max_voxel_num` is the maximum number of voxels.


To use `bev_vis_multi_frame_demo.py`, use the following command:

```
python bev_vis_multi_frame_demo.py \
  --config configs/pointpillars/pointpillars_xyres16_kitti_car.yml \
  --model demo/pointpillars.pdparams \
  --batch_size 1
```

`--config` is the path to the model configuration file.

`--model` is the path to the model parameter file being used.

`--batch_size` is the batch size for inference.

The final output of the lidar BEV visualization is shown below:

![](img/bev.png)

### Visualization interface for datasets and LOG files
The code of the visualization interface is defined in `paddle3d.apis.infer`, we provide an example as follows:

```
cd demo/visualization_demo
python dataset_vis_demo.py
```

---
If you encounter the following problems, refer to Ref1 and Ref2 solutions:

`qt.qpa.plugin: Could not load the Qt Platform plugin 'xcb' in ..`

[Ref1](https://blog.csdn.net/qq_39938666/article/details/120452028?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-120452028-blog-112303826.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-120452028-blog-112303826.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=3)
& [Ref2](https://blog.csdn.net/weixin_41794514/article/details/128578166?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128578166-blog-119480436.pc_relevant_landingrelevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128578166-blog-119480436.pc_relevant_landingrelevant)
