# paddle3d.datasets.KittiMonoDataset

  KITTI单目3D检测数据集，数据集信息请参考[KITTI官网](http://www.cvlibs.net/datasets/kitti/)

  *注意：KITTI官网只区分了训练集和测试集，我们遵循业界的普遍做法，将7481个训练集样本，进一步划分为3712个训练集样本和3769个验证集样本*

## \_\_init\_\_

  * **参数**

    * dataset_root: 数据集的根目录

    * mode: 数据集模式，支持 `train` / `val` / `trainval` / `test` 等格式

    * transforms: 数据增强方法
