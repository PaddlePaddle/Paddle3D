import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

from paddle3d.ops import custom_ops

for op_name, op_dict in custom_ops.items():
    sources = op_dict.pop('sources', [])
    flags = None

    if paddle.device.is_compiled_with_cuda():
        extension = CUDAExtension
        flags = {'cxx': ['-DPADDLE_WITH_CUDA']}
    else:
        sources = filter(lambda x: x.endswith('cu'), sources)
        extension = CppExtension

    if len(sources) == 0:
        continue

    extension = extension(sources=sources)
    setup(
        name=op_name,
        ext_modules=extension,
        extra_compile_args=flags,
        **op_dict)
