# Paddle3D

## 🌈简介

Paddle3D是飞桨官方开源的端到端深度学习3D感知套件，涵盖了许多前沿和经典的3D感知模型，支持多种模态和多种任务，可以助力开发者便捷地完成 **『自动驾驶』** 领域模型 从训练到部署的全流程应用。

![camera](https://user-images.githubusercontent.com/29754889/185546875-b8296cf4-f298-494b-8c15-201a2559d7ea.gif)
![lidar](https://user-images.githubusercontent.com/29754889/185551580-828f08d0-d607-4020-9e05-b96110bce7eb.gif)


## ✨主要特性

### 🧩灵活的框架设计

针对各类3D数据格式，灵活构建数据处理、骨干网络等核心模块，支持基于PaddleDet/PaddleSeg灵活扩展2D视觉感知能力，并提供API与脚本两种训练评估方式，满足开发者灵活定制的差异化需求

### 📱丰富的模型库

聚合主流3D感知算法及精度调优策略，覆盖单目、点云等多种模态及检测、分割等多种任务类型


### 🎗️端到端全流程 

支持KITTI、nuScenes、Waymo等主流3D数据集，提供从数据处理、模型搭建、训练调优到部署落地的全流程能力，极致优化模型性能，适配多种自动驾驶主流芯片，支持计算图优化、TensorRT/OpenVINO等加速库，并提供了开箱即用的部署教程，5分钟即可完成模型部署。

### 🏆无缝衔接Apollo

无缝对接Apollo自动驾驶平台，支持真机与仿真平台实验效果快速验证、多模态模型高性能融合，实现自动驾驶全栈式技术方案的高效搭建

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/61035602/209662380-6f67d4df-12a1-43b0-a79e-424eb4f4dc75.png
" align="middle" width="1280"/>
</p>
</div>

## 📣最新进展

### 💎稳定版本

Paddle3D v1.0正式版本发布!

## 👫开源社区

- **📑项目合作：** 如果您是企业开发者且有明确的目标检测垂类应用需求，请扫描如下二维码入群，并联系`群管理员AI`后可免费与官方团队展开不同层次的合作。
- **🏅️社区贡献：** Paddle3D非常欢迎你加入到飞桨社区的开源建设中，参与贡献方式可以参考[开源项目开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/index_cn.html)。
- **💻直播教程：** Paddle3D会定期在飞桨直播间([B站:飞桨PaddlePaddle](https://space.bilibili.com/476867757)、[微信: 飞桨PaddlePaddle](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ))，针对发新内容、以及产业范例、使用教程等进行直播分享。

<div align="center">
<img src="https://user-images.githubusercontent.com/61035602/209660514-4285abea-a855-44c4-9533-f2e90b9ca608.jpeg"  width = "150" height = "150",caption='' />
<p>Paddle3D官方技术交流群二维码</p>
</div>

### 📱模型库

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>单目3D感知</b>
      </td>
      <td>
        <b>激光雷达3D感知</b>
      </td>
      <td>
        <b>多相机3D感知</b>
      </td>
      <td>
        <b>骨干网络</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
            </ul>
          <li><b>检测</b></li>
        <ul>
      <ul>
          <li><a href="docs/models/caddn">CaDDN</a></li>
          <li><a href="docs/models/smoke">SMOKE</a></li>
          <li><a href="docs/models/dd3d">DD3D</a></li>
      </ul>
      </td>
      <td>
      </ul>
          <li><b>检测</b></li>
        <ul>
        <ul>
           <li><a href="docs/models/pointpillars">PointPillars</a></li>
            <li><a href="docs/models/centerpoint">CenterPoint</a></li>
            <li><a href="docs/models/iassd">IA-SSD</a></li>
            <li><a href="docs/models/pv_rcnn">PV-RCNN</a></li>
            <li><a href="docs/models/voxel_rcnn">Voxel-RCNN</a></li>
            <li><a href="docs/models/paconv">PAConv</a></li>
            </ul>
            </ul>
          <li><b>分割</b></li>
        <ul>
        <ul>
            <li><a href="docs/models/squeezesegv3">SqueezeSegV3</a></li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>BEV-Camera</b></li>
        <ul>
        <ul>
          <li><a href="docs/models/petr">PETR</a></li>
          <li><a href="docs/models/petr">PETRv2</a></li> 
          <li><a href="https://github.com/PaddlePaddle/Paddle3D/pull/152">BEVFormer</a></li>
        </ul>
      </td>
      <td>
        <ul> 
            <li><a href="paddle3d/models/backbones">DLA</a></li>
            <li><a href="paddle3d/models/backbones">HRNet</a></li>
            <li><a href="paddle3d/models/backbones">ResNet</a></li>
            <li><a href="paddle3d/models/backbones">Transformer</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 🔥使用教程

* [安装](./docs/installation.md)

* [全流程速览](./docs/quickstart.md)

* [自定义数据准备](./docs/datasets/custom.md)

* [配置文件详解](./docs/configuration.md)

* [API](./docs/api.md)

* Paddle3D&Apollo集成开发示例
  * [视觉感知算法集成开发示例](https://apollo.baidu.com/community/Apollo-Homepage-Document/Apollo_Doc_CN_8_0/camera)
  * [点云感知算法集成开发示例](https://apollo.baidu.com/community/Apollo-Homepage-Document/Apollo_Doc_CN_8_0/lidar)

* [常见问题](./docs/faq.md)

* [更新日志](./docs/release_note.md)



## 许可证

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。
