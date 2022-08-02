# Paddle3D

![License](https://img.shields.io/badge/license-Apache%202-blue.svg)

![img](https://user-images.githubusercontent.com/45024560/180968874-b47fd60a-da20-40bb-8b40-2103881d4947.GIF)

Paddle3D是飞桨官方开源的端到端深度学习3D感知套件，涵盖了许多前沿和经典的3D感知模型，支持多种模态和多种任务，可以助力开发者便捷地完成 **『自动驾驶』** 领域模型 从训练到部署的全流程应用。Paddle3D具备以下特性：

* 【丰富的模型库】聚合主流3D感知算法及精度调优策略，覆盖单目、点云等多种模态及检测、分割等多种任务类型

* 【灵活的框架设计】灵活构建数据处理、骨干网络等核心模块，并提供API与脚本两种训练评估方式，满足开发者灵活定制的差异化需求

* 【端到端全流程】提供从数据处理、模型搭建、训练调优到部署落地的全流程能力

* 【工业级部署方案】极致优化模型性能，适配多种自动驾驶主流芯片，支持计算图优化、TRT/OpenVINO等加速库 ，并提供了开箱即用的部署教程，5分钟即可完成模型部署

* 【无缝衔接Apollo】无缝对接Apollo自动驾驶平台，支持真机与仿真平台实验效果快速验证、多模态模型高性能融合，实现自动驾驶全栈式技术方案的高效搭建

目前Paddle3D还在快速发展中，大量的模型将被集成进来，如果您有特定的模型需求，请[反馈](https://github.com/PaddlePaddle/Paddle3D/issues)给我们

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
