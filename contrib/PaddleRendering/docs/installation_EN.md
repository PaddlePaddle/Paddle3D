# 1. Installation Tutorial

- [1. Installation Tutorial](#1-Installation-Tutorial)
    * [1.1. Environmental Requirements](#11-Environmental-Requirements)
    * [1.2. Installation Instructions](#12-Installation-Instructions)
        + [1.2.1. Install MiniConda](#121-Install-Miniconda)
        + [1.2.2. Install PaddlePaddle](#122-Install-PaddlePaddle)
            - [1.2.2.1. Create Virtual Environment](#1221-Create-Virtual-Environment)
            - [1.2.2.2. Enter Virtual Environment](#1222-Enter-Virtual-Environment)
            - [1.2.2.3. Install PaddlePaddle for GPU (preferred)](#1223-Install-PaddlePaddle-for-GPU-preferred)
            - [1.2.2.4. Install PaddlePaddle for CPU (alternative)](#1224-Install-PaddlePaddle-for-CPU-alternative)
            - [1.2.2.5. Verify Installation](#1225-Verify-Installation)
            - [1.2.2.6. More Ways to Install PaddlePaddle](#1226-More-Ways-to-install-PaddlePaddle)
        + [1.2.3. Install PaddleRendering](#123-Install-paddlerendering)
            - [1.2.3.1. Download PaddleRendering Source Code](#1231-Download-PaddleRendering-Source-Code)
            - [1.2.3.2. Install PaddleRendering Dependencies](#1232-Install-PaddleRendering-Dependencies)
    * [1.3. All-in-One Setup Script](#13-All-in-One-Setup-Script)
    * [1.4. FAQ](#14-faq)

## 1.1. Environmental Requirements

- PaddlePaddle develop (Nightly build)
- Python >= 3.7
- CUDA >= 10.2

`Note:` For more CUDA versions, please refer to [compile from source code](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/compile/linux-compile_en.html)。

## 1.2. Installation Instructions

### 1.2.1. Install MiniConda

`Note:` If Anaconda has been installed, there is no need to install Miniconda.

Miniconda is a free lightweight installer for Conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others.

`Portal:` [MiniConda Installation Tutorial](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

### 1.2.2. Install PaddlePaddle

#### 1.2.2.1. Create Virtual Environment

```bash
conda create -n paddle_env python=3.8
```

#### 1.2.2.2. Enter Virtual Environment

```bash
conda activate paddle_env
```

#### 1.2.2.3. Install PaddlePaddle for GPU (preferred)

`Note:` If your computer has NVIDIA GPUs, it is recommended to install the GPU version of PaddlePaddle

```bash
# CUDA toolkit 10.2 with cuDNN v7.6.5(for multi card support, NCCL2.7 or higher)
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# CUDA toolkit 11.2 with cuDNN v8.2.1(for multi card support, NCCL2.7 or higher)
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# CUDA toolkit 11.6 with cuDNN v8.4.0(for multi card support, NCCL2.7 or higher)
python -m pip install paddlepaddle-gpu==0.0.0.post116 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# CUDA toolkit 11.7 with cuDNN v8.4.1(for multi card support, NCCL2.7 or higher)
python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

You can refer to NVIDIA's official documents for the installation and configuration process of CUDA and cuDNN.

`Portal:`

- [CUDA Installation Instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuDNN Installation Instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
- For more CUDA releases, please refer to [Compile from source code](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/compile/fromsource_en.html)。

#### 1.2.2.4. Install PaddlePaddle for CPU (alternative)

`Note:` If your computer doesn’t have NVIDIA GPU, please install the CPU version of PaddlePaddle.

```bash
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

#### 1.2.2.5. Verify Installation

```bash
# Start the python interpreter
python
```

```python
# Type in the python interpreter
import paddle
paddle.utils.run_check()
```

If `PaddlePaddle is installed successfully` appears, the installation was successful.

```python
# Type quit() to exit the python interpreter
quit()
```

#### 1.2.2.6. More Ways to Install PaddlePaddle

`Portal:` [Other Installation Guides of PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/index_en.html)

### 1.2.3. Install PaddleRendering

#### 1.2.3.1. Download PaddleRendering Source Code

`Note:` If you have downloaded the PaddleRendering source code, you can ignore this step.

```shell
# Download source codes
git clone https://github.com/PaddlePaddle/Paddle3D
# Download submodules
cd contrib/PaddleRendering
git submodule update --init --recursive
```

#### 1.2.3.2. Install PaddleRendering Dependencies

```bash
pip install -r requirements.txt
```

## 1.3. All-in-One Setup Script

The following is a complete script for installing PaddleRendering based on conda, assuming that MiniConda or Anaconda has been successfully installed, and CUDA 10.2 has been installed.

```bash
# Create virtual environment
conda create -n paddle_env python=3.8
# Enter virtual environment
conda activate paddle_env
# Install PaddlePaddle for GPU (preferred)
# CUDA toolkit 10.2 with cuDNN v7.6.5(for multi card support, NCCL2.7 or higher)
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
# Download PaddleRendering source code
# Note: If you have downloaded the PaddleRendering source code, you can ignore this step.
git clone https://github.com/PaddlePaddle/Paddle3D
# Download submodules
cd contrib/PaddleRendering
git submodule update --init --recursive
# Install PaddleRendering dependencies
pip install -r requirements.txt

```

## 1.4. FAQ

If you encounter any problems during the installation process, you can refer to the [FAQ](docs/faq.md) page first. If you do not find the corresponding solution, you can also put up an [issue](https://github.com/PaddlePaddle/Paddle3D/issues).
