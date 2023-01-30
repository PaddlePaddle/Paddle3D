// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "paddle/extension.h"

template <typename T, typename T_int>
void dynamic_voxelize_cpu_kernel(const T *points, T_int *coors,
                                 const std::vector<float> voxel_size,
                                 const std::vector<float> coors_range,
                                 const std::vector<int> grid_size,
                                 const int num_points, const int num_features,
                                 const int NDim) {
  const int ndim_minus_1 = NDim - 1;
  bool failed = false;
  // int coor[NDim];
  int *coor = new int[NDim]();
  int c;

  for (int i = 0; i < num_points; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points[i * num_features + j] - coors_range[j]) /
                voxel_size[j]);

      // necessary to rm points out of range
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }

    for (int k = 0; k < NDim; ++k) {
      if (failed)
        coors[i * NDim + k] = -1;
      else
        coors[i * NDim + k] = coor[k];
    }
  }

  delete[] coor;
  return;
}

std::vector<paddle::Tensor> dynamic_voxelize_cpu(
    const paddle::Tensor &points, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int NDim = 3) {
  // check device
  PD_CHECK(points.is_cpu(), "points must be a CPU tensor");

  std::vector<int> grid_size(NDim);
  const int num_points = points.shape()[0];
  const int num_features = points.shape()[1];

  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  auto coors = paddle::full({num_points, 3}, 0, paddle::DataType::INT32,
                            paddle::CPUPlace());
  // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "dynamic_voxelize_cpu_kernel", ([&] {
        dynamic_voxelize_cpu_kernel<data_t, int>(
            points.data<data_t>(), coors.data<int>(), voxel_size, coors_range,
            grid_size, num_points, num_features, NDim);
      }));

  return {coors};
}

#ifdef PADDLE_WITH_CUDA

std::vector<paddle::Tensor> dynamic_voxelize_cuda(
    const paddle::Tensor &points, const std::vector<float> voxel_size,
    const std::vector<float> point_cloud_range, const int NDim = 3);
#endif

std::vector<paddle::Tensor> dynamic_voxelize(
    const paddle::Tensor &points, const std::vector<float> &voxel_size,
    const std::vector<float> &point_cloud_range) {
  if (points.is_cpu()) {
    return dynamic_voxelize_cpu(points, voxel_size, point_cloud_range);
#ifdef PADDLE_WITH_CUDA
  } else if (points.is_gpu() || points.is_gpu_pinned()) {
    return dynamic_voxelize_cuda(points, voxel_size, point_cloud_range);
#endif
  } else {
    PD_THROW(
        "Unsupported device type for dynamic_voxelize "
        "operator.");
  }
}

std::vector<std::vector<int64_t>> DynamicInferShape(
    std::vector<int64_t> points_shape, const std::vector<float> &voxel_size,
    const std::vector<float> &point_cloud_range) {
  return {{points_shape[0], 3}};
}

std::vector<paddle::DataType> DynamicInferDtype(paddle::DataType points_dtype) {
  return {paddle::DataType::INT32};
}

PD_BUILD_OP(dynamic_voxelize)
    .Inputs({"POINTS"})
    .Outputs({"COORS"})
    .SetKernelFn(PD_KERNEL(dynamic_voxelize))
    .Attrs({"voxel_size: std::vector<float>",
            "point_cloud_range: std::vector<float>"})
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicInferDtype));