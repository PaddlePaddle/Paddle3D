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

// cuda launcher declaration
void group_points_cuda_launcher_batch(const int b, const int c, const int n,
                                      const int npoints, const int nsample,
                                      const float *points, const int *idx,
                                      float *out);
void group_points_grad_cuda_launcher_batch(const int b, const int c,
                                           const int n, const int npoints,
                                           const int nsample,
                                           const float *grad_out,
                                           const int *idx, float *grad_points);

// op forward wrapper
std::vector<paddle::Tensor> group_points_cuda_forward_batch(
    const paddle::Tensor &points_tensor, const paddle::Tensor &idx_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT(idx_tensor);
  const int b = points_tensor.shape()[0];
  const int c = points_tensor.shape()[1];
  const int n = points_tensor.shape()[2];
  const int npoints = idx_tensor.shape()[1];
  const int nsample = idx_tensor.shape()[2];

  auto *points = points_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();
  auto out_tensor = paddle::empty(
      {b, c, npoints, nsample}, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto *out = out_tensor.data<float>();

  group_points_cuda_launcher_batch(b, c, n, npoints, nsample, points, idx, out);

  return {out_tensor};
}

// op backward wrapper
std::vector<paddle::Tensor> group_points_cuda_backward_batch(
    const paddle::Tensor &grad_out_tensor, const paddle::Tensor &idx_tensor,
    const paddle::Tensor &points_tensor) {
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(idx_tensor);
  const int b = grad_out_tensor.shape()[0];
  const int c = grad_out_tensor.shape()[1];
  const int npoints = grad_out_tensor.shape()[2];
  const int nsample = grad_out_tensor.shape()[3];
  const int n = points_tensor.shape()[2];

  auto *grad_out = grad_out_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();
  auto grad_points_tensor = paddle::full(
      {b, c, n}, 0.0, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto *grad_points = grad_points_tensor.data<float>();

  group_points_grad_cuda_launcher_batch(b, c, n, npoints, nsample, grad_out,
                                        idx, grad_points);

  return {grad_points_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> GroupInferShapeBatch(
    std::vector<int64_t> points_shape, std::vector<int64_t> idx_shape) {
  const int b = points_shape[0];
  const int c = points_shape[1];
  const int npoints = idx_shape[1];
  const int nsample = idx_shape[2];
  return {{b, c, npoints, nsample}};
}

// data type infer
std::vector<paddle::DataType> GroupInferDtypeBatch(
    paddle::DataType points_dtype, paddle::DataType idx_dtype) {
  return {points_dtype};
}

// build forward op
PD_BUILD_OP(grouping_operation_batch)
    .Inputs({"points_tensor", "idx_tensor"})
    .Outputs({"out_tensor"})
    .SetKernelFn(PD_KERNEL(group_points_cuda_forward_batch))
    .SetInferShapeFn(PD_INFER_SHAPE(GroupInferShapeBatch))
    .SetInferDtypeFn(PD_INFER_DTYPE(GroupInferDtypeBatch));

// build backward op
PD_BUILD_GRAD_OP(grouping_operation_batch)
    .Inputs({paddle::Grad("out_tensor"), "idx_tensor", "points_tensor"})
    .Outputs({paddle::Grad("points_tensor")})
    .SetKernelFn(PD_KERNEL(group_points_cuda_backward_batch));
