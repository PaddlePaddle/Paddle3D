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
void gather_points_cuda_launcher(const int b, const int c, const int n,
                                 const int npoints, const float *points,
                                 const int *idx, float *out);
void gather_points_grad_cuda_launcher(const int b, const int c, const int n,
                                      const int npoints, const float *grad_out,
                                      const int *idx, float *grad_points);

// op forward wrapper
std::vector<paddle::Tensor> gather_points_cuda_forward(
    const paddle::Tensor &points_tensor, const paddle::Tensor &idx_tensor) {
  // points: (B, C, N)
  // idx: (B, npoints)
  // output:
  //      out: (B, C, npoints)

  CHECK_INPUT(points_tensor);
  CHECK_INPUT(idx_tensor);
  const int b = points_tensor.shape()[0];
  const int c = points_tensor.shape()[1];
  const int n = points_tensor.shape()[2];
  const int npoints = idx_tensor.shape()[1];

  auto *points = points_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();
  auto out_tensor = paddle::empty({b, c, npoints}, paddle::DataType::FLOAT32,
                                  paddle::GPUPlace());
  auto *out = out_tensor.data<float>();

  gather_points_cuda_launcher(b, c, n, npoints, points, idx, out);

  return {out_tensor};
}

// op backward wrapper
std::vector<paddle::Tensor> gather_points_cuda_backwarad(
    const paddle::Tensor &grad_out_tensor, const paddle::Tensor &idx_tensor,
    const paddle::Tensor &points_tensor) {
  // grad_out: (B, C, npoints)
  // idx: (B, npoints)
  // output:
  //      grad_points: (B, C, N)

  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(points_tensor);
  const int b = grad_out_tensor.shape()[0];
  const int c = grad_out_tensor.shape()[1];
  const int npoints = grad_out_tensor.shape()[2];
  const int n = points_tensor.shape()[2];

  auto *grad_out = grad_out_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();
  auto grad_points_tensor = paddle::full(
      {b, c, n}, 0.0, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto *grad_points = grad_points_tensor.data<float>();

  gather_points_grad_cuda_launcher(b, c, n, npoints, grad_out, idx,
                                   grad_points);

  return {grad_points_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> GatherInferShape(
    std::vector<int64_t> points_shape, std::vector<int64_t> idx_shape) {
  const int b = points_shape[0];
  const int c = points_shape[1];
  const int npoints = idx_shape[1];
  return {{b, c, npoints}};
}

// data type infer
std::vector<paddle::DataType> GatherInferDtype(paddle::DataType points_dtype,
                                               paddle::DataType idx_dtype) {
  return {points_dtype};
}

// build op forward
PD_BUILD_OP(gather_operation)
    .Inputs({"points", "idx"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(gather_points_cuda_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(GatherInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GatherInferDtype));

// build op backward
PD_BUILD_GRAD_OP(gather_operation)
    .Inputs({paddle::Grad("out"), "idx", "points"})
    .Outputs({paddle::Grad("points")})
    .SetKernelFn(PD_KERNEL(gather_points_cuda_backwarad));