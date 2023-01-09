#include <vector>

#include "paddle/extension.h"

std::vector<paddle::Tensor> hard_voxelize_gpu(
    const paddle::Tensor &points, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels);

std::vector<std::vector<int64_t>> HardInferShape(
    std::vector<int64_t> points_shape, const std::vector<float> &voxel_size,
    const std::vector<float> &point_cloud_range,
    const int &max_num_points_in_voxel, const int &max_voxels) {
  return {{max_voxels, max_num_points_in_voxel, points_shape[1]},
          {max_voxels, 3},
          {max_voxels},
          {1}};
}

std::vector<paddle::DataType> HardInferDtype(paddle::DataType points_dtype) {
  return {points_dtype, paddle::DataType::INT32, paddle::DataType::INT32,
          paddle::DataType::INT32};
}

PD_BUILD_OP(hard_voxelize_v2)
    .Inputs({"POINTS"})
    .Outputs({"VOXELS", "COORS", "NUM_POINTS_PER_VOXEL", "num_voxels"})
    .SetKernelFn(PD_KERNEL(hard_voxelize_gpu))
    .Attrs({"voxel_size: std::vector<float>", "coors_range: std::vector<float>",
            "max_points: int", "max_voxels: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(HardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(HardInferDtype));