//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "grid_sample_3d.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> GridSample3DCUDAForward(
    const paddle::Tensor& x, const paddle::Tensor& grid,
    const std::string& mode, const std::string& padding_mode,
    bool align_corners);

std::vector<paddle::Tensor> GridSample3DForward(const paddle::Tensor& x,
                                                const paddle::Tensor& grid,
                                                const std::string& mode,
                                                const std::string& padding_mode,
                                                bool align_corners) {
  return GridSample3DCUDAForward(x, grid, mode, padding_mode, align_corners);
}

std::vector<paddle::Tensor> GridSample3DCUDABackward(
    const paddle::Tensor& x, const paddle::Tensor& grid,
    const paddle::Tensor& grad_out, const std::string& mode,
    const std::string& padding_mode, bool align_corners);

std::vector<paddle::Tensor> GridSample3DBackward(
    const paddle::Tensor& x, const paddle::Tensor& grid,
    const paddle::Tensor& grad_out, const std::string& mode,
    const std::string& padding_mode, bool align_corners) {
  return GridSample3DCUDABackward(x, grid, grad_out, mode, padding_mode,
                                  align_corners);
}

std::vector<std::vector<int64_t>> GridSample3DInferShape(
    std::vector<int64_t> x_shape, std::vector<int64_t> grid_shape) {
  return {
      {x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]}};
}

std::vector<std::vector<int64_t>> GridSample3DInferBackShape(
    std::vector<int64_t> x_shape, std::vector<int64_t> grid_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> GridSample3DInferDtype(
    paddle::DataType x_dtype, paddle::DataType grid_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(grid_sample_3d)
    .Inputs({"x", "grid"})
    .Attrs({"mode: std::string", "padding_mode: std::string",
            "align_corners: bool"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(GridSample3DForward))
    .SetInferShapeFn(PD_INFER_SHAPE(GridSample3DInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GridSample3DInferDtype));

PD_BUILD_GRAD_OP(grid_sample_3d)
    .Inputs({"x", "grid", paddle::Grad("out")})
    .Attrs({"mode: std::string", "padding_mode: std::string",
            "align_corners: bool"})
    .Outputs({paddle::Grad("x")})
    .SetKernelFn(PD_KERNEL(GridSample3DBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(GridSample3DInferBackShape));