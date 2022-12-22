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

#include "paddle/include/experimental/ext_all.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

// cuda launcher declaration
void farthest_point_sampling_kernel_launcher(int b, int n, int m,
                                             const float *dataset, float *temp,
                                             int *idxs);

// op forward wrapper
std::vector<paddle::Tensor> farthest_point_sampling_cuda_forward(
    const paddle::Tensor &points_tensor, const int &npoints) {
  // points_tensor: (B, N, 3)
  // tmp_tensor: (B, N)
  // output:
  //      idx_tensor: (B, npoints)

  const int b = points_tensor.shape()[0];
  const int n = points_tensor.shape()[1];

  auto *points = points_tensor.data<float>();
  auto temp_tensor =
      paddle::full({b, n}, 1e10, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto idx_tensor =
      paddle::empty({b, npoints}, paddle::DataType::INT32, paddle::GPUPlace());
  auto *temp = temp_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();

  farthest_point_sampling_kernel_launcher(b, n, npoints, points, temp, idx);

  return {idx_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> FPSInferShape(
    std::vector<int64_t> points_shape, const int &npoints) {
  return {{points_shape[0], npoints}};
}

// dtype infer
std::vector<paddle::DataType> FPSInferDtype(paddle::DataType points_dtype) {
  return {paddle::DataType::INT32};
}

// build op forward
PD_BUILD_OP(farthest_point_sample)
    .Inputs({"points_tensor"})
    .Outputs({"idx_tensor"})
    .Attrs({"npoints: int"})
    .SetKernelFn(PD_KERNEL(farthest_point_sampling_cuda_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(FPSInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FPSInferDtype));