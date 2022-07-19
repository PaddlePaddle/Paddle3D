# paddle3d.models.SMOKE

  单目3D检测模型 《Single-Stage Monocular 3D Object Detection via Keypoint Estimation》

## \_\_init\_\_

  * **参数**

      * backbone: 所用的骨干网络

      * head: 预测头，目前只支持 `SMOKEPredictor`

      * depth_ref: 深度参考值

      * dim_ref: 每个类别的维度参考值

      * max_detection: 最大检测目标数量，默认为50

      * pred_2d: 是否同时预测2D检测框结果，默认为True

<br>

# paddle3d.models.SMOKEPredictor

  SMOKE模型的预测头

## \_\_init\_\_

  * **参数**

      * num_classe：检测类别数

      * norm_type: NormLayer的类型，默认为gn

      * in_channels: 输入channel数量
