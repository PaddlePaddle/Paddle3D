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

// Modified from
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/common/cuda/voxelization_cuda_kernel.cuh
// Copyright (c) OpenMMLab. All rights reserved.

#include "paddle/extension.h"

#define CHECK_INPUT_CUDA(x) \
  PD_CHECK(x.is_gpu() || x.is_gpu_pinned(), #x " must be a GPU Tensor.")

int const threadsPerBlock = sizeof(unsigned long long) * 8;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

std::vector<paddle::Tensor> dynamic_voxelize_cuda(
    const paddle::Tensor& points, const std::vector<float> voxel_size,
    const std::vector<float> point_cloud_range, const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT_CUDA(points);

  const int num_points = points.shape()[0];
  const int num_features = points.shape()[1];

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = point_cloud_range[0];
  const float coors_y_min = point_cloud_range[1];
  const float coors_z_min = point_cloud_range[2];
  const float coors_x_max = point_cloud_range[3];
  const float coors_y_max = point_cloud_range[4];
  const float coors_z_max = point_cloud_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  auto coors = paddle::full({num_points, 3}, 0, paddle::DataType::INT32,
                            paddle::GPUPlace());

  const int col_blocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "dynamic_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<data_t, int>
            <<<col_blocks, threadsPerBlock, 0, points.stream()>>>(
                points.data<data_t>(), coors.data<int>(), voxel_x, voxel_y,
                voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
                coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
                num_features, NDim);
      }));

  return {coors};
}