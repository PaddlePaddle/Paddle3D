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
void group_points_kernel_launcher_stack(const int B, const int M, const int C,
                                        const int nsample,
                                        const float *features,
                                        const int *features_batch_cnt,
                                        const int *idx,
                                        const int *idx_batch_cnt, float *out);
void group_points_grad_kernel_launcher_stack(
    const int B, const int M, const int C, const int N, const int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt,
    const int *features_batch_cnt, float *grad_features);

// op forward wrapper
std::vector<paddle::Tensor> group_points_cuda_forward_stack(
    const paddle::Tensor &features_tensor,
    const paddle::Tensor &features_batch_cnt_tensor,
    const paddle::Tensor &idx_tensor,
    const paddle::Tensor &idx_batch_cnt_tensor) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT(features_batch_cnt_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(idx_batch_cnt_tensor);

  const int m = idx_tensor.shape()[0];
  const int nsample = idx_tensor.shape()[1];
  const int n = features_tensor.shape()[0];
  const int c = features_tensor.shape()[1];
  const int b = idx_batch_cnt_tensor.shape()[0];

  const float *features = features_tensor.data<float>();
  const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
  const int *idx = idx_tensor.data<int>();
  const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
  auto out_tensor = paddle::empty({m, c, nsample}, paddle::DataType::FLOAT32,
                                  paddle::GPUPlace());
  float *out = out_tensor.data<float>();

  group_points_kernel_launcher_stack(
      b, m, c, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);

  return {out_tensor};
}

// op backward wrapper
std::vector<paddle::Tensor> group_points_cuda_backward_stack(
    const paddle::Tensor &grad_out_tensor,
    const paddle::Tensor &features_tensor,
    const paddle::Tensor &features_batch_cnt_tensor,
    const paddle::Tensor &idx_tensor,
    const paddle::Tensor &idx_batch_cnt_tensor) {
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(features_tensor);
  CHECK_INPUT(features_batch_cnt_tensor);
  CHECK_INPUT(idx_tensor);
  CHECK_INPUT(idx_batch_cnt_tensor);

  const int m = idx_tensor.shape()[0];
  const int nsample = idx_tensor.shape()[1];
  const int n = features_tensor.shape()[0];
  const int c = features_tensor.shape()[1];
  const int b = idx_batch_cnt_tensor.shape()[0];

  const float *grad_out = grad_out_tensor.data<float>();
  const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
  const int *idx = idx_tensor.data<int>();
  const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
  auto grad_features_tensor =
      paddle::full({n, c}, 0., paddle::DataType::FLOAT32, paddle::GPUPlace());
  float *grad_features = grad_features_tensor.data<float>();

  group_points_grad_kernel_launcher_stack(b, m, c, n, nsample, grad_out, idx,
                                          idx_batch_cnt, features_batch_cnt,
                                          grad_features);

  return {grad_features_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> GroupInferShapeStack(
    std::vector<int64_t> features_shape,
    std::vector<int64_t> features_batch_cnt_shapeshape,
    std::vector<int64_t> idx_shape, std::vector<int64_t> idx_batch_cnt_shape) {
  const int m = idx_shape[0];
  const int nsample = idx_shape[1];
  const int c = features_shape[1];
  return {{m, c, nsample}};
}

// data type infer
std::vector<paddle::DataType> GroupInferDtypeStack(
    paddle::DataType features_dtype, paddle::DataType features_batch_cnt_dtype,
    paddle::DataType idx_dtype, paddle::DataType idx_batch_cnt_dtype) {
  return {features_dtype};
}

// build forward op
PD_BUILD_OP(grouping_operation_stack)
    .Inputs({"features_tensor", "features_batch_cnt_tensor", "idx_tensor",
             "idx_batch_cnt_tensor"})
    .Outputs({"out_tensor"})
    .SetKernelFn(PD_KERNEL(group_points_cuda_forward_stack))
    .SetInferShapeFn(PD_INFER_SHAPE(GroupInferShapeStack))
    .SetInferDtypeFn(PD_INFER_DTYPE(GroupInferDtypeStack));

// build backward op
PD_BUILD_GRAD_OP(grouping_operation_stack)
    .Inputs({paddle::Grad("out_tensor"), "features_tensor",
             "features_batch_cnt_tensor", "idx_tensor", "idx_batch_cnt_tensor"})
    .Outputs({paddle::Grad("features_tensor")})
    .SetKernelFn(PD_KERNEL(group_points_cuda_backward_stack));
