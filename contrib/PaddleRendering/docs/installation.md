# 1. 安装教程

- [1. 安装教程](#1-安装教程)
    * [1.1. 环境要求](#11-环境要求)
    * [1.2. 安装说明](#12-安装说明)
        + [1.2.1. 安装 MiniConda](#121-安装-miniconda)
        + [1.2.2. 安装 PaddlePaddle](#122-安装-paddlepaddle)
            - [1.2.2.1. 创建虚拟环境](#1221-创建虚拟环境)
            - [1.2.2.2. 进入 conda 虚拟环境](#1222-进入-conda-虚拟环境)
            - [1.2.2.3. 添加清华源（可选）](#1223-添加清华源可选)
            - [1.2.2.4. 安装 GPU 版的 PaddlePaddle (首选)](#1224-安装-gpu-版的-paddlepaddle-首选)
            - [1.2.2.5. 安装 CPU 版的 PaddlePaddle (备选)](#1225-安装-cpu-版的-paddlepaddle-备选)
            - [1.2.2.6. 验证安装](#1226-验证安装)
            - [1.2.2.7. 更多 PaddlePaddle 安装方式](#1227-更多-paddlepaddle-安装方式)
        + [1.2.3. 安装 PaddleRendering](#123-安装-paddlerendering)
            - [1.2.3.1. 下载 PaddleRendering 源码](#1231-下载-paddleRendering-源码)
            - [1.2.3.2. 安装 PaddleRendering 依赖](#1232-安装-paddleRendering-依赖)
    * [1.3. 完整安装脚本](#13-完整安装脚本)
    * [1.4. FAQ](#14-faq)

## 1.1. 环境要求

- PaddlePaddle develop (Nightly build)
- Python >= 3.7
- CUDA >= 10.2

`说明：`更多 CUDA 版本可参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile.html#anchor-0)。

## 1.2. 安装说明

### 1.2.1. 安装MiniConda

`说明：`如果已安装 Anaconda 则无需再安装 Miniconda。

Miniconda 是一款小巧的 Python 环境管理工具，其安装程序中包含 conda 软件包管理器和 Python。MiniConda 的包使用软件包管理系统Conda进行管理。Conda 是一个开源包管理系统和环境管理系统，可在 Windows、macOS 和 Linux 上运行。

`传送门：`[MiniConda 安装教程](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

### 1.2.2. 安装 PaddlePaddle

#### 1.2.2.1. 创建虚拟环境

```bash
conda create -n paddle_env python=3.8
```

#### 1.2.2.2. 进入 conda 虚拟环境

```bash
conda activate paddle_env
```

#### 1.2.2.3. 添加清华源（可选）

```bash
# 对于国内用户无法连接到 conda 官方源的可以按照以下命令添加清华源:
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

#### 1.2.2.4. 安装 GPU 版的 PaddlePaddle (首选)

`说明：`如果您的计算机有 NVIDIA® GPU，建议安装 GPU 版的 PaddlePaddle

```bash
# 对于 CUDA 10.2，需要搭配 cuDNN 7.6.5(多卡环境下 NCCL>=2.7)，安装命令为:
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# 对于 CUDA 11.2，需要搭配 cuDNN 8.2.1(多卡环境下 NCCL>=2.7)，安装命令为:
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# 对于 CUDA 11.6，需要搭配 cuDNN 8.4.0(多卡环境下 NCCL>=2.7)，安装命令为:
python -m pip install paddlepaddle-gpu==0.0.0.post116 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# 对于 CUDA 11.7，需要搭配 cuDNN 8.4.1(多卡环境下 NCCL>=2.7)，安装命令为:
python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

您可参考 NVIDIA 官方文档了解 CUDA、CUDNN 的安装流程和配置方法。

`传送门：`

- [CUDA安装说明](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuDNN安装说明](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
- 更多CUDA版本可参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。

#### 1.2.2.5. 安装 CPU 版的 PaddlePaddle(备选)

`说明：`如果您的计算机没有 NVIDIA GPU，请安装 CPU 版的 PaddlePaddle

```bash
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

#### 1.2.2.6. 验证安装

```bash
# 输入 python 进入 python 解释器
python
```

```python
# 在 python 解释器中输入
import paddle
paddle.utils.run_check()
```

如果出现 `PaddlePaddle is installed successfully！`，说明您已成功安装。

```python
# 输入 quit() 退出python解释器
quit()
```

#### 1.2.2.7. 更多 PaddlePaddle 安装方式

`传送门：`[PaddlePaddle 其他安装指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/index_cn.html)

### 1.2.3. 安装 PaddleRendering

#### 1.2.3.1. 下载 PaddleRendering 源码

`说明：`如已下载 PaddleRendering 源码可忽略这一步。

```shell
# 下载源码
git clone https://github.com/PaddlePaddle/Paddle3D
# 下载子模块
cd contrib/PaddleRendering
git submodule update --init --recursive
```

#### 1.2.3.2. 安装 PaddleRendering 依赖

```bash
pip install -r requirements.txt
```

## 1.3. 完整安装脚本

以下是完整的基于 conda 安装 PaddleRendering 的脚本，假设已经成功安装 MiniConda 或 Anaconda，已安装 CUDA 10.2。

```bash
# 创建虚拟环境
conda create -n paddle_env python=3.8
# 进入 conda 虚拟环境
conda activate paddle_env
# 添加清华源（可选）
# 对于国内用户无法连接到 conda 官方源的可以按照以下命令添加清华源:
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
# 安装 GPU 版的 PaddlePaddle
# 对于 CUDA 10.2，需要搭配 cuDNN v7.6+(多卡环境下 NCCL>=2.7)，安装命令为:
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
# 下载 PaddleRendering 代码
# 说明：如已下载 PaddleRendering 源码可忽略这一步。
git clone https://github.com/PaddlePaddle/Paddle3D
# 下载子模块
cd contrib/PaddleRendering
git submodule update --init --recursive
# 安装 PaddleRendering 依赖
pip install -r requirements.txt

```

## 1.4. FAQ

如果在安装过程中遇到什么问题，可以先参考 [FAQ](docs/faq.md) 页面. 如果没有找到对应的解决方案，您也可以在 [issue](https://github.com/PaddlePaddle/Paddle3D/issues) 提出。
