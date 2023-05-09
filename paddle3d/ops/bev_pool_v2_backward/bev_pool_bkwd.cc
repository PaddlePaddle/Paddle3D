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

#include <paddle/extension.h>

void bev_pool_v2_grad(int c, int n_intervals, const float* out_grad,
                      const float* depth, const float* feat,
                      const int* ranks_depth, const int* ranks_feat,
                      const int* ranks_bev, const int* interval_starts,
                      const int* interval_lengths, float* depth_grad,
                      float* feat_grad);

std::vector<paddle::Tensor> bev_pool_v2_bkwd_forward(
    const paddle::Tensor& _out_grad, const paddle::Tensor& _depth,
    const paddle::Tensor& _feat, const paddle::Tensor& _ranks_depth,
    const paddle::Tensor& _ranks_feat, const paddle::Tensor& _ranks_bev,
    const paddle::Tensor& _interval_lengths,
    const paddle::Tensor& _interval_starts) {
  int c = _out_grad.shape()[4];
  int n_intervals = _interval_lengths.shape()[0];
  const float* out_grad = _out_grad.data<float>();
  const float* depth = _depth.data<float>();
  const float* feat = _feat.data<float>();
  const int* ranks_depth = _ranks_depth.data<int>();
  const int* ranks_feat = _ranks_feat.data<int>();
  const int* ranks_bev = _ranks_bev.data<int>();
  const int* interval_lengths = _interval_lengths.data<int>();
  const int* interval_starts = _interval_starts.data<int>();

  auto _depth_grad = paddle::full(_depth.shape(), 0.0, _depth.type(),
                                  paddle::GPUPlace());  // add to return out

  auto _feat_grad = paddle::full(_feat.shape(), 0.0, _feat.type(),
                                 paddle::GPUPlace());  // add to return out

  float* depth_grad = _depth_grad.data<float>();
  float* feat_grad = _feat_grad.data<float>();
  bev_pool_v2_grad(c, n_intervals, out_grad, depth, feat, ranks_depth,
                   ranks_feat, ranks_bev, interval_starts, interval_lengths,
                   depth_grad, feat_grad);

  return {{_depth_grad, _feat_grad}};
}

std::vector<std::vector<int64_t>> BevPoolV2BkwdInferShape(
    std::vector<int64_t> _out_grad_shape, std::vector<int64_t> _depth_shape,
    std::vector<int64_t> _feat_shape, std::vector<int64_t> _ranks_depth_shape,
    std::vector<int64_t> _ranks_feat_shape,
    std::vector<int64_t> _ranks_bev_shape,
    std::vector<int64_t> _interval_length_shape,
    std::vector<int64_t> _interval_starts_shape) {
  return {_depth_shape, _feat_shape};
}

std::vector<paddle::DataType> BevPoolV2BkwdInferDtype(
    paddle::DataType _out_grad_dtype, paddle::DataType _depth_dtype,
    paddle::DataType _feat_dtype, paddle::DataType _ranks_depth_dtype,
    paddle::DataType _ranks_feat_dtype, paddle::DataType _ranks_bev_dtype,
    paddle::DataType _interval_length_dtype,
    paddle::DataType _interval_starts_dtype) {
  return {_depth_dtype, _feat_dtype};
}

PD_BUILD_OP(bev_pool_v2_bkwd)
    .Inputs({"_out_grad", "_depth", "_feat", "_ranks_depth", "_ranks_feat",
             "_ranks_bev", "_interval_lengths", "_interval_starts"})
    .Outputs({"_depth_grad", "_feat_grad"})
    .SetKernelFn(PD_KERNEL(bev_pool_v2_bkwd_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(BevPoolV2BkwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BevPoolV2BkwdInferDtype));