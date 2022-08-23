# 安装文档

## 环境要求

- PaddlePaddle >= 2.3
- Python >= 3.6
- CUDA >= 10.1
- cuDNN >= 7.6

## 安装说明

### 1. 安装PaddlePaddle

```shell
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.3.0
```

* 由于3D感知模型对算力要求都比较高，我们建议您下载GPU版本

* 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)

* 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

### 2. 下载Paddle3D代码

```shell
git clone https://github.com/PaddlePaddle/Paddle3D
```

### 3. 安装Paddle3D依赖

```shell
cd Paddle3D
pip install -r requirements.txt
```
### 4. 安装Paddle3D
```shell
pip install .     # regular install
pip install -e .  # develop install
```
