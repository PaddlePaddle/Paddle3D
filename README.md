# Paddle3D

![License](https://img.shields.io/badge/license-Apache%202-blue.svg)

Paddle3D是飞桨官方开源的端到端深度学习3D感知套件，涵盖了许多前沿和经典的3D感知模型，支持多种模态和多种任务，可以助力开发者便捷地完成 **『自动驾驶』** 领域模型 从训练到部署的全流程应用。Paddle3D具备以下特性：

* 【丰富的模型库】

* 【灵活的框架设计】

* 【端到端全流程】

* 【工业级部署方案】

* 【无缝衔接Apollo】

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
          <li> PointPillar </li>
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

  * [PointPillar](./docs/models/pointpillars)

  * [CenterPoint](./docs/models/centerpoint)

  * [SequeezeSeg v3](./docs/models/squeezesegv3)

* [自定义数据准备](./docs/datasets/custom.md)

* [配置文件详解](./docs/configuration.md)

* [API](./docs/api.md)

* [常见问题](./docs/faq.md)

* [更新日志](./docs/release_note.md)

## 许可证

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。
