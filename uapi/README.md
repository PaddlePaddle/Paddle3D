# UAPI for Paddle3D: Quick Start

## 1 Install Dependencies

### 1.1 Install PaddlePaddle

Please follow the instruction on [PaddlePaddle official website](https://www.paddlepaddle.org.cn/).

### 1.2 Install Paddle3D from Source

Please follow [the installation docs of Paddle3D](https://github.com/PaddlePaddle/Paddle3D/blob/release/1.0/docs/installation.md).


## 2 Experience UAPI Through Demo

### 2.1 Prepare Dataset for Testing

Create a directory named `uapi_demo` in the root directory of Paddle3D repo. After that, download the demo dataset from [here](https://paddle-model-ecology.bj.bcebos.com/uapi/data/mini_kitti.zip). Unzip the files to `uapi_demo/data/mini_kitti`.

### 2.2 Run Demo Script

Switch to the root directory of Paddle3D repo if you are not there. Then run the following commands:

```shell
python -m uapi.demo
```

Check out the training output files in `uapi_demo/output/`, the exported model in `uapi_demo/output/infer`, the inference (with a static-graph model) results in `uapi_demo/output/infer_res`, and the compression (quantization aware training and export) outputs in `uapi_demo/output/infer_res/compress`.
