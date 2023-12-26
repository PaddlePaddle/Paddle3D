# Paddle3D
## 🌈简介

Paddle3D是飞桨官方开源的端到端深度学习3D感知套件，涵盖了许多前沿和经典的3D感知模型，支持多种模态和多种任务，可以助力开发者便捷地完成 **『自动驾驶』** 领域模型 从训练到部署的全流程应用。

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/29754889/185546875-b8296cf4-f298-494b-8c15-201a2559d7ea.gif" align="middle" width="980"/>
</p>
</div>

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/29754889/185551580-828f08d0-d607-4020-9e05-b96110bce7eb.gif" align="middle" width="980"/>
</p>
</div>

## ✨主要特性

### 🧩灵活的框架设计

针对各类3D数据格式，灵活构建数据处理、骨干网络等核心模块，支持基于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)灵活扩展2D视觉感知能力，并提供API与脚本两种训练评估方式，满足开发者灵活定制的差异化需求。

### 📱丰富的模型库

聚合主流3D感知算法及精度调优策略，覆盖单目、点云等多种模态及检测、分割等多种任务类型。


### 🎗️端到端全流程

支持KITTI、nuScenes、Waymo等主流3D数据集，提供从数据处理、模型搭建、训练调优到部署落地的全流程能力，极致优化模型性能，适配多种自动驾驶主流芯片，支持计算图优化、TensorRT/OpenVINO等加速库，并提供了开箱即用的部署教程，5分钟即可完成模型部署。

### 🏆无缝衔接Apollo

无缝对接Apollo自动驾驶平台，支持真机与仿真平台实验效果快速验证、多模态模型高性能融合，实现自动驾驶全栈式技术方案的高效搭建。

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/33575107/224960808-3bb1a328-cb20-48ce-b6b6-9d6ec1cc8222.png" align="middle" width="980"/>
</p>
</div>

## 📣最新进展

**🏅️飞桨AI套件**

飞桨AI套件 [PaddleX](https://www.paddlepaddle.org.cn/paddle/paddleX) 发布全新版本，围绕飞桨精选模型（包括3D检测）提供了一站式、全流程、高效率的开发平台。


**💎稳定版本**

位于[`主分支`](https://github.com/PaddlePaddle/Paddle3D)，Paddle3D v1.0正式版本发布，详情请参考[release note](https://github.com/PaddlePaddle/Paddle3D/releases/tag/v1.0)。

**🧬预览版本**

位于[`develop`](https://github.com/PaddlePaddle/Paddle3D/tree/develop)分支，体验最新功能请切换到[该分支](https://github.com/PaddlePaddle/Paddle3D/tree/develop)。
## 👫开源社区

- **📑项目合作：** 如果您是企业开发者且有明确的目标检测垂类应用需求，请扫描如下二维码入群，并联系`群管理员AI`后可免费与官方团队展开不同层次的合作。
- **🏅️社区贡献：** Paddle3D非常欢迎你加入到飞桨社区的开源建设中，参与贡献方式可以参考[开源项目开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/index_cn.html)。
- **💻直播教程：** Paddle3D会定期在飞桨直播间([B站:飞桨PaddlePaddle](https://space.bilibili.com/476867757)、[微信: 飞桨PaddlePaddle](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ))，针对发新内容、以及产业范例、使用教程等进行直播分享。

<div align="center">
<img src="https://user-images.githubusercontent.com/61035602/209660514-4285abea-a855-44c4-9533-f2e90b9ca608.jpeg"  width = "150" height = "150",caption='' />
<p>Paddle3D官方技术交流群二维码</p>
</div>

- **🎈社区近期活动**

  - **🎗️Paddle3D v1.0正式版解读**
    - `文章传送门`：[Paddle3D正式版发布！BEV、单目、激光雷达3D感知算法开箱即用，无缝衔接Apollo](https://mp.weixin.qq.com/s/LL0DgKxEVsfhpFO6HedQ7Q)

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/210311019-bdb15ec8-e8b9-471c-aa1d-d2f953a6939a.png"  height = "250" caption='' />
    <p></p>
    </div>

  - **🚦自动驾驶感知系统揭秘**
    - `课程录播&PPT传送门`：[自动驾驶感知系统揭秘](https://aistudio.baidu.com/aistudio/education/group/info/26961)

    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/210315230-83ace5d1-1851-4d9b-b305-4290edf9dde8.png"  height = "300" caption='' />
    <p></p>
    </div>

- **📑 PaddleX**
  * 飞桨低代码开发工具（PaddleX）—— 面向国内外主流AI硬件的飞桨精选模型一站式开发工具。包含如下核心优势：
    * 【产业高精度模型库】：覆盖10个主流AI任务 40+精选模型，丰富齐全。
    * 【特色模型产线】：提供融合大小模型的特色模型产线，精度更高，效果更好。
    * 【低代码开发模式】：图形化界面支持统一开发范式，便捷高效。
    * 【【私有化部署多硬件支持】：适配国内外主流AI硬件，支持本地纯离线使用，满足企业安全保密需要。
    * 【共赢的联创共建】除了便捷地开发AI应用外，PaddleX还为大家提供了获取商业收益的机会，为企业探索更多商业空间。

  * PaddleX官网地址：https://aistudio.baidu.com/intro/paddlex

  * PaddleX官方交流频道：https://aistudio.baidu.com/community/channel/610


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
          <li><a href="docs/models/bevformer">BEVFormer</a></li>
          <li><a href="docs/models/cape">CAPE</a></li>
        </ul>
        </ul>
          <li><b>BEV-Fusion</b></li>
        <ul>
        <ul>
          <li><a href="docs/models/bevfusion">BEVFusion(ADLab)</a></li>
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

## 💡产业实践范例

产业实践范例是Paddle3D针对3D目标检测应用场景，提供的端到端开发示例，帮助开发者打通数据标注-模型训练-模型调优-预测部署全流程。
针对每个范例我们都通过[AI-Studio](https://ai.baidu.com/ai-doc/AISTUDIO/Tk39ty6ho)提供了项目代码以及说明，用户可以同步运行体验。

- [【自动驾驶实战】基于Paddle3D&Apollo的点云3D目标物检测](https://aistudio.baidu.com/aistudio/projectdetail/5268894)
- [【自动驾驶实战】基于Paddle3D&Apollo的单目3D目标物检测](https://aistudio.baidu.com/aistudio/projectdetail/5269115)

## 📝许可证

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。
