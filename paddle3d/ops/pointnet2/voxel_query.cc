// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

void voxel_query_kernel_launcher_stack(int M, int R1, int R2, int R3,
                                       int nsample, float radius, int z_range,
                                       int y_range, int x_range,
                                       const float *new_xyz, const float *xyz,
                                       const int *new_coords,
                                       const int *point_indices, int *idx);

std::vector<paddle::Tensor> voxel_query_wrapper_stack(
    const paddle::Tensor &new_xyz_tensor, const paddle::Tensor &xyz_tensor,
    const paddle::Tensor &new_coords_tensor,
    const paddle::Tensor &point_indices_tensor, const float radius,
    const int nsample, const int z_range, const int y_range,
    const int x_range) {
  CHECK_INPUT(new_coords_tensor);
  CHECK_INPUT(point_indices_tensor);
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor);

  const float *new_xyz = new_xyz_tensor.data<float>();
  const float *xyz = xyz_tensor.data<float>();
  const int *new_coords = new_coords_tensor.data<int>();
  const int *point_indices = point_indices_tensor.data<int>();

  const int M = new_coords_tensor.shape()[0];
  const int B = point_indices_tensor.shape()[0];
  const int Z = point_indices_tensor.shape()[1];
  const int Y = point_indices_tensor.shape()[2];
  const int X = point_indices_tensor.shape()[3];

  auto idx_tensor = paddle::full({M, nsample}, 0, paddle::DataType::INT32,
                                 paddle::GPUPlace());
  int *idx = idx_tensor.data<int>();

  voxel_query_kernel_launcher_stack(M, Z, Y, X, nsample, radius, z_range,
                                    y_range, x_range, new_xyz, xyz, new_coords,
                                    point_indices, idx);
  return {idx_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> VoxelQueryInferShape(
    std::vector<int64_t> new_xyz_shape, std::vector<int64_t> xyz_shape,
    std::vector<int64_t> new_coords_shape,
    std::vector<int64_t> point_indices_shape, const float radius,
    const int nsample, const int z_range, const int y_range,
    const int x_range) {
  return {{new_coords_shape[0], nsample}};
}

// data type infer
std::vector<paddle::DataType> VoxelQueryInferDtype(
    paddle::DataType new_xyz_type, paddle::DataType xyz_type,
    paddle::DataType new_coords_type, paddle::DataType point_indices_type) {
  return {paddle::DataType::INT32};
}

// build forward op
PD_BUILD_OP(voxel_query_wrapper)
    .Inputs({"new_xyz_tensor", "xyz_tensor", "new_coords_tensor",
             "point_indices_tensor"})
    .Outputs({"idx_tensor"})
    .Attrs({"radius: float", "nsample: int", "z_range: int", "y_range: int",
            "x_range: int"})
    .SetKernelFn(PD_KERNEL(voxel_query_wrapper_stack))
    .SetInferShapeFn(PD_INFER_SHAPE(VoxelQueryInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VoxelQueryInferDtype));