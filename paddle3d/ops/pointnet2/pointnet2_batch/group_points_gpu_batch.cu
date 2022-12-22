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
batch version of point grouping, modified from the original implementation of
official PointNet++ codes. Written by Shaoshuai Shi All Rights Reserved 2018.
*/

#include "paddle/extension.h"

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void group_points_cuda_kernel_batch(const int b, const int c,
                                               const int n, const int npoints,
                                               const int nsample,
                                               const float *__restrict__ points,
                                               const int *__restrict__ idx,
                                               float *__restrict__ out) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_idx = index / nsample;
  if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

  int sample_idx = index % nsample;

  idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
  int in_idx = bs_idx * c * n + c_idx * n + idx[0];
  int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
                pt_idx * nsample + sample_idx;

  out[out_idx] = points[in_idx];
}

void group_points_cuda_launcher_batch(const int b, const int c, const int n,
                                      const int npoints, const int nsample,
                                      const float *points, const int *idx,
                                      float *out) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  cudaError_t err;
  dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_cuda_kernel_batch<<<blocks, threads>>>(b, c, n, npoints, nsample,
                                                      points, idx, out);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void group_points_grad_cuda_kernel_batch(
    const int b, const int c, const int n, const int npoints, const int nsample,
    const float *__restrict__ grad_out, const int *__restrict__ idx,
    float *__restrict__ grad_points) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_idx = index / nsample;
  if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

  int sample_idx = index % nsample;
  grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
              pt_idx * nsample + sample_idx;
  idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;

  atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0], grad_out[0]);
}

void group_points_grad_cuda_launcher_batch(const int b, const int c,
                                           const int n, const int npoints,
                                           const int nsample,
                                           const float *grad_out,
                                           const int *idx, float *grad_points) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  cudaError_t err;
  dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c,
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_grad_cuda_kernel_batch<<<blocks, threads>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
