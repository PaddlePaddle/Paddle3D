from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='my_undist',
    ext_modules=CppExtension(
        sources=['undistCpu.cc']
    )
)