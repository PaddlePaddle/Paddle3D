import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

custom_ops = [
    # voxelize
    {
        'name': 'voxelize',
        'sources': ['voxel/voxelize_op.cc', 'voxel/voxelize_op.cu']
    },
    # iou3d_nms
    {
        'name':
        'iou3d_nms_cuda',
        'sources': [
            'iou3d_nms/iou3d_cpu.cpp', 'iou3d_nms/iou3d_nms_api.cpp',
            'iou3d_nms/iou3d_nms.cpp', 'iou3d_nms/iou3d_nms_kernel.cu'
        ]
    },
    # centerpoint_postprocess
    {
        'name':
        'centerpoint_postprocess',
        'sources': [
            'centerpoint_postprocess/iou3d_nms_kernel.cu',
            'centerpoint_postprocess/postprocess.cc',
            'centerpoint_postprocess/postprocess.cu'
        ]
    },
    # grid_sample_3d
    {
        'name':
        'grid_sample_3d',
        'sources': [
            'grid_sample_3d/grid_sample_3d.cc',
            'grid_sample_3d/grid_sample_3d.cu'
        ]
    }
]

for op_dict in custom_ops:
    sources = op_dict.get('sources', [])
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
    setup(name=op_dict['name'], ext_modules=extension, extra_compile_args=flags)
