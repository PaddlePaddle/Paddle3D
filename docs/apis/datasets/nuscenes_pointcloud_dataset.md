# paddle3d.datasets.NuscenesPCDataset

  Nuscenes点云检测数据集，数据集信息请参考[NuScenes官网](https://www.nuscenes.org/)

## \_\_init\_\_

  * **参数**

    * dataset_root: 数据集的根目录

    * mode: 数据集模式，支持 `train` / `val` / `trainval` / `test` / `mini_train` / `mini_val` 等格式

        *注意：当使用NuScenes官方提供的mini数据集时，请指定mode为 mini_train 或者 mini_val*

    * transforms: 数据增强方法

    * max_sweeps: 用于增强每一帧点云的sweeps数量，默认为10

    * class_balanced_sampling: 是否做类别均衡采样，默认为False

    * class_names: 类别名，可以不指定
