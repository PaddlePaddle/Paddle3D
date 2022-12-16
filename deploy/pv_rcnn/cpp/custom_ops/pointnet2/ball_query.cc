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
void ball_query_kernel_launcher_stack(const int b, const int m,
                                      const float radius, const int nsample,
                                      const float *new_xyz,
                                      const int *new_xyz_batch_cnt,
                                      const float *xyz,
                                      const int *xyz_batch_cnt, int *idx);

// op forward wrapper
std::vector<paddle::Tensor> ball_query_cuda_forward(
    const paddle::Tensor &new_xyz_tensor,
    const paddle::Tensor &new_xyz_batch_cnt_tensor,
    const paddle::Tensor &xyz_tensor,
    const paddle::Tensor &xyz_batch_cnt_tensor, const float radius,
    const int nsample) {
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(new_xyz_batch_cnt_tensor);
  CHECK_INPUT(xyz_tensor);
  CHECK_INPUT(xyz_batch_cnt_tensor);
  const int b = xyz_batch_cnt_tensor.shape()[0];
  const int m = new_xyz_tensor.shape()[0];
  const float *new_xyz = new_xyz_tensor.data<float>();
  const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
  const float *xyz = xyz_tensor.data<float>();
  const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
  auto idx_tensor = paddle::full({m, nsample}, 0, paddle::DataType::INT32,
                                 paddle::GPUPlace());
  int *idx = idx_tensor.data<int>();

  ball_query_kernel_launcher_stack(b, m, radius, nsample, new_xyz,
                                   new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);

  return {idx_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> BallQueryInferShape(
    std::vector<int64_t> new_xyz_shape,
    std::vector<int64_t> new_xyz_batch_cnt_shape,
    std::vector<int64_t> xyz_shape, std::vector<int64_t> xyz_batch_cnt_shape,
    const float radius, const int nsample) {
  return {{new_xyz_shape[0], nsample}};
}

// data type infer
std::vector<paddle::DataType> BallQueryInferDtype(
    paddle::DataType new_xyz_type, paddle::DataType new_xyz_batch_cnt_type,
    paddle::DataType xyz_type, paddle::DataType xyz_batch_cnt_type) {
  return {paddle::DataType::INT32};
}

// build forward op
PD_BUILD_OP(ball_query)
    .Inputs({"new_xyz_tensor", "new_xyz_batch_cnt_tensor", "xyz_tensor",
             "xyz_batch_cnt_tensor"})
    .Outputs({"idx_tensor"})
    .Attrs({"radius: float", "nsample: int"})
    .SetKernelFn(PD_KERNEL(ball_query_cuda_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(BallQueryInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BallQueryInferDtype));
