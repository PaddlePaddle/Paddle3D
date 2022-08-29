# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称   | 单机单卡 |
|  :----  |:-------|    :----  |
|  CNN  | PAConv | 正常训练 |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下，

| 模型类型 |device | batchsize | tensorrt  | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |:---------:|   :----:   |  :----:  |
| 正常模型 | GPU | 1 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1 |     -     | - | 支持 |


## 2. 测试流程


### 2.1 安装依赖
- 安装PaddlePaddle == 2.3.1
- 安装Paddle3D依赖
    ```
    pip install  -r requirements.txt
    ```
- 安装Paddle3D
    ```
    python setup.py develop
    ```
- 安装autolog（规范化日志输出工具）
    ```
    git clone https://gitee.com/Double_V/AutoLog
    cd AutoLog/
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
    cd ../
    ```


### 2.2 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。


`test_train_inference_python.sh`包含5种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，该项目目前只支持模式1：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/paconv/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/paconv/train_infer_python.txt 'lite_train_lite_infer'
```

运行相应指令后，在`test_tipc/output`文件夹下自动会保存运行日志。如'lite_train_lite_infer'模式下，会运行训练+inference的链条，因此，在`test_tipc/output`文件夹有以下文件：
```
test_tipc/output/
|- results_python.log    # 运行指令状态的日志
|- python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log  # CPU上关闭Mkldnn线程数设置为1，测试batch_size=1条件下的精度fp32预测运行日志
|- python_infer_gpu_usetrt_False_precision_fp32_batchsize_1.log # GPU上关闭TensorRT，测试batch_size=1的精度fp32预测日志
......
```

其中`results_python.log`中包含了每条指令的运行状态，如果运行成功会输出：
```
Run successfully with command - python3.7 infer.py --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_file=output/model.pdmodel --batch_size=1 --input_file=test_tipc/data/predict_example.pkl --enable_benchmark=True --precision=fp32 --params_file=output/model.pdiparams > ./test_tipc/output/PAConv/python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log 2>&1 !
Run successfully with command - python3.7 infer.py --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_file=output/model.pdmodel --batch_size=2 --input_file=test_tipc/data/predict_example.pkl --enable_benchmark=True --precision=fp32 --params_file=output/model.pdiparams > ./test_tipc/output/PAConv/python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_2.log 2>&1 !  
......
```
如果运行失败，会输出：
```
Run failed with command - python3.7 infer.py --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_file=output/model.pdmodel --batch_size=1 --input_file=test_tipc/data/predict_example.pkl --enable_benchmark=True --precision=fp32 --params_file=output/model.pdiparams > ./test_tipc/output/PAConv/python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1.log 2>&1 !
Run failed with command - python3.7 infer.py --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_file=output/model.pdmodel --batch_size=2 --input_file=test_tipc/data/predict_example.pkl --enable_benchmark=True --precision=fp32 --params_file=output/model.pdiparams > ./test_tipc/output/PAConv/python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_2.log 2>&1 !  
......
```
可以很方便的根据`results_python.log`中的内容判定哪一个指令运行错误。
