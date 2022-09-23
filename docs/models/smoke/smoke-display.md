# SMOKE的输出3d坐标和航向角画在2d照片上


## 简介


* 正常的流程，3d坐标和航向角可以给下游任务用，但是我们需要即使看到模型的效果，所以做了一下demo工具显示模型效果；
* 代码路径：deploy/smoke/vis/smoke_vis.ipynb

<br>


## 使用教程

下面的教程将如何模型推理到模型结果显示


### 模型下载导出
从[这里](./docs/models/smoke/README.md)下载smoke模型

```shell
# 导出Paddle3D提供的预训练模型
python tools/export.py --config configs/smoke/smoke_dla34_no_dcn_kitti.yml --model model.pdparams 
```
<br>

### jupter notebook可视化部分

按照 deploy/smoke/vis/smoke_vis.ipynb  step by step 完成可视化显示
  
<br>