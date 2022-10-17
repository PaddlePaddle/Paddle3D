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

// cuda kernel declaration
void points_in_boxes_cuda_launcher(const int batch_size, const int boxes_num,
                                   const int pts_num, const float *boxes,
                                   const float *pts, int *box_idx_of_points);

// op forward
std::vector<paddle::Tensor> points_in_boxes_cuda_forward(
    const paddle::Tensor &pts_tensor, const paddle::Tensor &boxes_tensor) {
  // boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
  // pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate
  // output:
  //      boxes_idx_of_points: (B, npoints), default -1

  CHECK_INPUT(boxes_tensor);
  CHECK_INPUT(pts_tensor);

  const int batch_size = boxes_tensor.shape()[0];
  const int boxes_num = boxes_tensor.shape()[1];
  const int pts_num = pts_tensor.shape()[1];
  auto box_idx_of_points_tensor = paddle::full(
      {batch_size, pts_num}, -1, paddle::DataType::INT32, paddle::GPUPlace());
  auto *boxes = boxes_tensor.data<float>();
  auto *pts = pts_tensor.data<float>();
  auto *box_idx_of_points = box_idx_of_points_tensor.data<int>();

  points_in_boxes_cuda_launcher(batch_size, boxes_num, pts_num, boxes, pts,
                                box_idx_of_points);

  return {box_idx_of_points_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> PtInBoxInferShape(
    std::vector<int64_t> pts_shape, std::vector<int64_t> boxes_shape) {
  return {{boxes_shape[0], pts_shape[1]}};
}

// dtype infer
std::vector<paddle::DataType> PtInBoxInferDtype(paddle::DataType pts_dtype,
                                                paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT32};
}

// build op forward
PD_BUILD_OP(points_in_boxes_gpu)
    .Inputs({"pts_tensor", "boxes_tensor"})
    .Outputs({"box_idx_of_points"})
    .SetKernelFn(PD_KERNEL(points_in_boxes_cuda_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(PtInBoxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PtInBoxInferDtype));