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

#include "dynamic_scatter_op.h"

#include <vector>

#include "paddle/extension.h"

std::vector<paddle::Tensor> dynamic_point_to_voxel_forward(
    const paddle::Tensor &feats, const paddle::Tensor &coors,
    const std::string &reduce_type) {
  if (feats.is_gpu() || feats.is_gpu_pinned()) {
#ifdef PADDLE_WITH_CUDA
    return dynamic_point_to_voxel_forward_cuda(
        feats, coors, convert_reduce_type(reduce_type));
#endif
  } else {
    PD_THROW(
        "Do not support cpu device for dynamic_scatter "
        "operator.");
  }
}

std::vector<paddle::Tensor> dynamic_point_to_voxel_backward(
    const paddle::Tensor &grad_reduced_feats, const paddle::Tensor &feats,
    const paddle::Tensor &reduced_feats, const paddle::Tensor &coors_map,
    const paddle::Tensor &reduce_count, const std::string &reduce_type) {
  if (grad_reduced_feats.is_gpu() || grad_reduced_feats.is_gpu_pinned()) {
#ifdef PADDLE_WITH_CUDA
    return dynamic_point_to_voxel_backward_cuda(
        grad_reduced_feats, feats, reduced_feats, coors_map, reduce_count,
        convert_reduce_type(reduce_type));
#endif
  } else {
    PD_THROW(
        "Do not support cpu device for dynamic_scatter "
        "operator.");
  }
}

std::vector<std::vector<int64_t>> DynamicScatterInferShape(
    std::vector<int64_t> feats_shape, std::vector<int64_t> coors_shape) {
  return {{-1, feats_shape[1]}, {-1, coors_shape[1]}, {-1}, {-1}};
}

std::vector<paddle::DataType> DynamicScatterInferDtype(
    paddle::DataType feats_dtype, paddle::DataType coors_dtype) {
  return {feats_dtype, coors_dtype, paddle::DataType::INT32,
          paddle::DataType::INT32};
}

PD_BUILD_OP(dynamic_point_to_voxel)
    .Inputs({"feats", "coors"})
    .Outputs({"reduced_feats", "out_coors", "coors_map", "reduce_count"})
    .SetKernelFn(PD_KERNEL(dynamic_point_to_voxel_forward))
    .Attrs({"reduce_type: std::string"})
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicScatterInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicScatterInferDtype));

// build op backward
PD_BUILD_GRAD_OP(dynamic_point_to_voxel)
    .Inputs({paddle::Grad("reduced_feats"), "feats", "reduced_feats",
             "coors_map", "reduce_count"})
    .Outputs({paddle::Grad("feats")})
    .Attrs({"reduce_type: std::string"})
    .SetKernelFn(PD_KERNEL(dynamic_point_to_voxel_backward));