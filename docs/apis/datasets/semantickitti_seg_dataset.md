# paddle3d.datasets.SemanticKITTIDataset

  SemanticKITTI点云分割数据集，数据集信息请参考[SemanticKITTI官网](http://www.semantic-kitti.org/)

## \_\_init\_\_

  * **参数**

    * dataset_root: 数据集的根目录

    * mode: 数据集模式，支持 `train` / `val` / `trainval` / `test` 等格式

    * sequences: 数据划分序列，可以不指定，默认使用官网推荐的划分方式

    * transforms: 数据增强方法
