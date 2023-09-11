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

#include "paddle/extension.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> ms_deform_attn_forward_cuda(
    const paddle::Tensor &value, const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights, const int im2col_step);

std::vector<paddle::Tensor> ms_deform_attn_backward_cuda(
    const paddle::Tensor &grad_out, const paddle::Tensor &value,
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights, const int im2col_step);

std::vector<paddle::Tensor> ms_deform_attn_backward(
    const paddle::Tensor &grad_out, const paddle::Tensor &value,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights,
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index, const int im2col_step) {
  if (value.is_gpu()) {
    return ms_deform_attn_backward_cuda(grad_out, value, spatial_shapes,
                                        level_start_index, sampling_locations,
                                        attention_weights, im2col_step);
  } else {
    PD_THROW(
        "Unsupported device type for ms_deform_attn_backward "
        "operator.");
  }
}

std::vector<paddle::Tensor> ms_deform_attn_forward(
    const paddle::Tensor &value, const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights,
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index, const int im2col_step) {
  if (value.is_gpu()) {
    return ms_deform_attn_forward_cuda(value, spatial_shapes, level_start_index,
                                       sampling_locations, attention_weights,
                                       im2col_step);
  } else {
    PD_THROW(
        "Unsupported device type for ms_deform_attn_forward "
        "operator.");
  }
}

// shape infer
std::vector<std::vector<int64_t>> MsDeformAttrnInferShape(
    std::vector<int64_t> value_shape,
    std::vector<int64_t> sampling_locations_shape,
    std::vector<int64_t> attention_weights_shape,
    std::vector<int64_t> spatial_shapes_shape,
    std::vector<int64_t> level_start_index_shape) {
  return {{value_shape[0], sampling_locations_shape[1],
           value_shape[2] * value_shape[3]}};
}

// data type infer
std::vector<paddle::DataType> MsDeformAttrnInferDtype(
    paddle::DataType value_dtype, paddle::DataType sampling_locations_dtype,
    paddle::DataType attention_weights_dtype, paddle::DataType spatial_shapes,
    paddle::DataType level_start_index_dtype) {
  return {value_dtype};
}

// build forward op
PD_BUILD_OP(ms_deform_attn)
    .Inputs({"value", "sampling_locations", "attention_weights",
             "spatial_shapes", "level_start_index"})
    .Attrs({"im2col_step: int"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(ms_deform_attn_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(MsDeformAttrnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MsDeformAttrnInferDtype));

// build backward op
PD_BUILD_GRAD_OP(ms_deform_attn)
    .Inputs({paddle::Grad("out"), "value", "sampling_locations",
             "attention_weights", "spatial_shapes", "level_start_index"})
    .Attrs({"im2col_step: int"})
    .Outputs({paddle::Grad("value"), paddle::Grad("sampling_locations"),
              paddle::Grad("attention_weights")})
    // .Outputs({paddle::Grad("value")})
    .SetKernelFn(PD_KERNEL(ms_deform_attn_backward));