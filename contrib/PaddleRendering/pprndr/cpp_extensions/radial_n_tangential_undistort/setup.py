from paddle.utils.cpp_extension import CUDAExtension, setup, CppExtension
import paddle

if paddle.fluid.is_compiled_with_cuda() is True:
    setup(name='my_undist',
          ext_modules=CUDAExtension(
              sources=['./src/undist_gpu.cc', './src/undist_gpu.cu']))
else:
    setup(name='my_undist',
          ext_modules=CppExtension(sources=['./src/undist_cpu.cc']))
