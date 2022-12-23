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

/*
RoI-aware point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <math.h>

#include "paddle/extension.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y,
                                             float rot_angle, float &local_x,
                                             float &local_y) {
  float cosa = cos(-rot_angle), sina = sin(-rot_angle);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d,
                                        float &local_x, float &local_y) {
  // pt: (x, y, z)
  // box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

  const float MARGIN = 1e-5;
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

  if (fabsf(z - cz) > dz / 2.0) return 0;
  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
  float in_flag =
      (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
  return in_flag;
}

__global__ void points_in_boxes_cuda_kernel(
    const int batch_size, const int boxes_num, const int pts_num,
    const float *boxes, const float *pts, int *box_idx_of_points) {
  // boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
  // pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate
  // output:
  //      boxes_idx_of_points: (B, npoints), default -1

  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= batch_size || pt_idx >= pts_num) return;

  boxes += bs_idx * boxes_num * 7;
  pts += bs_idx * pts_num * 3 + pt_idx * 3;
  box_idx_of_points += bs_idx * pts_num + pt_idx;

  float local_x = 0, local_y = 0;
  int cur_in_flag = 0;
  for (int k = 0; k < boxes_num; k++) {
    cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
    if (cur_in_flag) {
      box_idx_of_points[0] = k;
      break;
    }
  }
}

void points_in_boxes_cuda_launcher(const int batch_size, const int boxes_num,
                                   const int pts_num, const float *boxes,
                                   const float *pts, int *box_idx_of_points) {
  // boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
  // pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate
  // output:
  //      boxes_idx_of_points: (B, npoints), default -1
  cudaError_t err;

  dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  points_in_boxes_cuda_kernel<<<blocks, threads>>>(
      batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
