import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

if paddle.device.is_compiled_with_cuda():
    setup(
        name='undistort',
        version='1.0.0',
        ext_modules=CUDAExtension(
            sources=['./src/undist_gpu.cc', './src/undist_gpu.cu']),
        extra_compile_args={'cxx': ['-DPADDLE_WITH_CUDA']})
else:
    setup(
        name='undistort',
        version='1.0.0',
        ext_modules=CppExtension(sources=['./src/undist_gpu.cc']))
