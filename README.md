# Paddle3D

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://pypi.org/project/paddle3d/"><img src="https://img.shields.io/pypi/dm/paddle3d?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/Paddle3D/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/Paddle3D?color=ccf"></a>
</p>

![camera](https://user-images.githubusercontent.com/29754889/185546875-b8296cf4-f298-494b-8c15-201a2559d7ea.gif)
![lidar](https://user-images.githubusercontent.com/29754889/185551580-828f08d0-d607-4020-9e05-b96110bce7eb.gif)

Paddle3D是飞桨官方开源的端到端深度学习3D感知套件，涵盖了许多前沿和经典的3D感知模型，支持多种模态和多种任务，可以助力开发者便捷地完成 **『自动驾驶』** 领域模型 从训练到部署的全流程应用。Paddle3D具备以下特性：

* 【丰富的模型库】聚合主流3D感知算法及精度调优策略，覆盖单目、点云等多种模态及检测、分割等多种任务类型

* 【灵活的框架设计】针对各类3D数据格式，灵活构建数据处理、骨干网络等核心模块，支持基于PaddleDet/PaddleSeg灵活扩展2D视觉感知能力，并提供API与脚本两种训练评估方式，满足开发者灵活定制的差异化需求

* 【端到端全流程】支持KITTI、nuScenes等主流3D数据集，提供从数据处理、模型搭建、训练调优到部署落地的全流程能力

* 【工业级部署方案】极致优化模型性能，适配多种自动驾驶主流芯片，支持计算图优化、TensorRT/OpenVINO等加速库，并提供了开箱即用的部署教程，5分钟即可完成模型部署

* 【无缝衔接Apollo】无缝对接Apollo自动驾驶平台，支持真机与仿真平台实验效果快速验证、多模态模型高性能融合，实现自动驾驶全栈式技术方案的高效搭建

目前Paddle3D还在快速发展中，大量的模型将被集成进来，如果您有特定的模型需求，请[反馈](https://github.com/PaddlePaddle/Paddle3D/issues)给我们

## 🔥 热门活动
- 【百度自动驾驶技术大揭秘】Apollo自动驾驶技术架构概览 & 飞桨Paddle3D在自动驾驶中的应用与实战
  - 两日课程回放及PPT材料：https://aistudio.baidu.com/aistudio/course/introduce/26961

## 模型库

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>骨干网络</b>
      </td>
      <td colspan="4">
        <b>3D检测</b>
      </td>
      <td>
        <b>3D分割</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li> HRNet </li>
          <li> DLA </li>
        </ul>
      </td>
      <td>
        <p align="center">📸 单目</p>
        <ul>
          <li> SMOKE </li>
          <li> CaDDN</li>
        </ul>
      </td>
      <td>
        <p align="center">📡 点云</p>
        <ul>
          <li> PointPillars </li>
          <li> CenterPoint </li>
        </ul>
      </td>
      <td>
        <p align="center"> 📸 + 📡 多模态 </p>
        支持中
      </td>
      <td>
        <p align="center"> 📸 + 📸 多视角 </p>
        支持中
      </td>
      <td>
        <p align="center"> 📡点云 </p>
        <ul>
          <li> SqueezeSeg v3 </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 数据集

- [x] KITTI

- [x] NuScenes

- [x] SemanticKITTI

- [ ] waymo - 支持中

## 使用教程

* [安装](./docs/installation.md)

* [全流程速览](./docs/quickstart.md)

* 模型使用教程

  * [SMOKE](./docs/models/smoke)

  * [CaDDN](./docs/models/caddn)

  * [PointPillars](./docs/models/pointpillars)

  * [CenterPoint](./docs/models/centerpoint)

  * [SequeezeSeg v3](./docs/models/squeezesegv3)

* [自定义数据准备](./docs/datasets/custom.md)

* [配置文件详解](./docs/configuration.md)

* [API](./docs/api.md)

* [常见问题](./docs/faq.md)

* [更新日志](./docs/release_note.md)

## 技术交流

- 如果你发现任何Paddle3D存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/Paddle3D/issues)给我们提issues。

- 欢迎加入Paddle3D 微信用户群
  <div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/182345513-bbca647f-1f03-4543-baba-01c09f67addd.jpg"  width = "200" />  
  </div>


## 许可证

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。
