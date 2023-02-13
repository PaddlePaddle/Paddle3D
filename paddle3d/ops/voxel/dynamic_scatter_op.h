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

#include <vector>

#include "paddle/extension.h"

typedef enum { SUM = 0, MEAN = 1, MAX = 2 } reduce_t;

inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else
    PD_THROW("do not support reduce type ", reduce_type);
  return reduce_t::SUM;
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> dynamic_point_to_voxel_forward_cuda(
    const paddle::Tensor &feats, const paddle::Tensor &coors,
    const reduce_t reduce_type);

std::vector<paddle::Tensor> dynamic_point_to_voxel_backward_cuda(
    const paddle::Tensor &grad_reduced_feats, const paddle::Tensor &feats,
    const paddle::Tensor &reduced_feats, const paddle::Tensor &coors_map,
    const paddle::Tensor &reduce_count, const reduce_t reduce_type);
#endif
