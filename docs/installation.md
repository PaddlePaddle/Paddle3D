# 1. 安装教程

- [1. 安装教程](#1-安装教程)
  - [1.1. 环境要求](#11-环境要求)
  - [1.2. 安装说明](#12-安装说明)
    - [1.2.1. 安装MiniConda](#121-安装miniconda)
    - [1.2.2. 安装PaddlePaddle](#122-安装paddlepaddle)
      - [1.2.2.1. 创建虚拟环境](#1221-创建虚拟环境)
      - [1.2.2.2. 进入 conda 虚拟环境](#1222-进入-conda-虚拟环境)
      - [1.2.2.3. 添加清华源（可选）](#1223-添加清华源可选)
      - [1.2.2.4. 安装GPU版的PaddlePaddle(首选)](#1224-安装gpu版的paddlepaddle首选)
      - [1.2.2.5. 安装CPU版的PaddlePaddle(备选)](#1225-安装cpu版的paddlepaddle备选)
      - [1.2.2.6. 验证安装](#1226-验证安装)
      - [1.2.2.7. 更多PaddlePaddle安装方式](#1227-更多paddlepaddle安装方式)
    - [1.2.3. 安装Paddle3D](#123-安装paddle3d)
      - [1.2.3.1. 下载Paddle3D源码](#1231-下载paddle3d源码)
      - [1.2.3.2. 安装Paddle3D依赖](#1232-安装paddle3d依赖)
      - [1.2.3.3. 安装Paddle3D](#1233-安装paddle3d)
  - [1.3. 完整安装脚本](#13-完整安装脚本)
  - [1.4. FAQ](#14-faq)

## 1.1. 环境要求

- PaddlePaddle >= 2.4.0
- Python >= 3.6
- CUDA 10.2、11.2、11.6、11.7

`说明：`

- 更多CUDA版本可参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。
- Jetson系列芯片可参考预编译好的[Python、C++推理库](https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#:~:text=paddle_inference_c.tgz-,Python%20%E6%8E%A8%E7%90%86%E5%BA%93,-%C2%B6)。


## 1.2. 安装说明

### 1.2.1. 安装MiniConda

```bash
说明：如果已安装Anaconda则无需再安装Miniconda。
```

Miniconda是一款小巧的Python环境管理工具，其安装程序中包含conda软件包管理器和Python。MiniConda的包使用软件包管理系统Conda进行管理。Conda是一个开源包管理系统和环境管理系统，可在Windows、macOS和Linux上运行。

`传送门：`[MiniConda安装教程](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

### 1.2.2. 安装PaddlePaddle

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

#### 1.2.2.4. 安装GPU版的PaddlePaddle(首选)

```bash
说明：如果您的计算机有 NVIDIA® GPU，建议安装 GPU 版的 PaddlePaddle
```

```bash
# 对于 CUDA 10.2，需要搭配 cuDNN 7.6.5(多卡环境下 NCCL>=2.7)，安装命令为:
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

# 对于 CUDA 11.2，需要搭配 cuDNN 8.2.1(多卡环境下 NCCL>=2.7)，安装命令为:
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge

# 对于 CUDA 11.6，需要搭配 cuDNN 8.4.0(多卡环境下 NCCL>=2.7)，安装命令为:
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge

# 对于 CUDA 11.7，需要搭配 cuDNN 8.4.1(多卡环境下 NCCL>=2.7)，安装命令为:
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

您可参考 NVIDIA 官方文档了解 CUDA、CUDNN、TensorRT的安装流程和配置方法。

`传送门：`

- [CUDA安装说明](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuDNN安装说明](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
- [TensorRT安装说明](https://docs.nvidia.com/deeplearning/tensorrt/index.html)
- 更多CUDA版本可参考[从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html)。
- Jetson系列芯片可参考预编译好的[Python、C++推理库](https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#:~:text=paddle_inference_c.tgz-,Python%20%E6%8E%A8%E7%90%86%E5%BA%93,-%C2%B6)。

#### 1.2.2.5. 安装CPU版的PaddlePaddle(备选)

```bash
说明：如果您的计算机没有 NVIDIA® GPU，请安装 CPU 版的 PaddlePaddle
```

```bash
conda install paddlepaddle==2.4.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

#### 1.2.2.6. 验证安装

```bash
# 输入python进入python解释器
python
```

```python
# 在python解释器中输入
import paddle
# 再输入
paddle.utils.run_check()
```

```bash
如果出现PaddlePaddle is installed successfully!，说明您已成功安装。
```

```python
# 输入quit()退出python解释器
quit()
```

#### 1.2.2.7. 更多PaddlePaddle安装方式

`传送门：`(PaddlePaddle其他安装指南)[https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html]

### 1.2.3. 安装Paddle3D

#### 1.2.3.1. 下载Paddle3D源码

```bash
说明：如已下载Paddle3D源码可忽略这一步。
```

```shell
git clone https://github.com/PaddlePaddle/Paddle3D
```

#### 1.2.3.2. 安装Paddle3D依赖

```bash
cd Paddle3D
pip install -r requirements.txt
```

#### 1.2.3.3. 安装Paddle3D

```shell
pip install -e .  # install in edit mode
```

## 1.3. 完整安装脚本

以下是完整的基于conda安装Paddle3D的脚本，假设已经成功安装MiniConda或Anaconda，已安装CUDA 11.6。

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
# 安装GPU版的PaddlePaddle
# 对于 CUDA 11.6，需要搭配 cuDNN 8.4.0(多卡环境下 NCCL>=2.7)，安装命令为:
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
# 下载Paddle3D代码
# 说明：如已下载Paddle3D源码可忽略这一步。
git clone https://github.com/PaddlePaddle/Paddle3D
# 安装Paddle3D依赖
cd Paddle3D
pip install -r requirements.txt
# 安装Paddle3D
pip install -e .  # install in edit mode
```

## 1.4. FAQ

如果在安装过程中遇到什么问题，可以先参考[FAQ](docs/faq.md)页面. 如果没有找到对应的解决方案，你也可以在[issue](https://github.com/PaddlePaddle/Paddle3D/issues)。
