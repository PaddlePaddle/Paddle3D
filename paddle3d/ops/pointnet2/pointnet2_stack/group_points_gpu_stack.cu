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
Stacked-batch-data version of point grouping, modified from the original
implementation of official PointNet++ codes. Written by Shaoshuai Shi All Rights
Reserved 2019-2020.
*/

#include "paddle/extension.h"

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void group_points_grad_kernel_stack(
    int B, int M, int C, int N, int nsample, const float *grad_out,
    const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt,
    float *grad_features) {
  // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the
  // output from forward :param idx: (M1 + M2 ..., nsample) tensor containing
  // the indices of features to group with :param idx_batch_cnt: (batch_size)
  // [M1 + M2 ...] tensor containing the indices of features to group with
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the
  // indices of features to group with :return:
  //     grad_features: (N1 + N2 ..., C) gradient of the features
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = index % nsample;
  int C_idx = (index / nsample) % C;
  int pt_idx = (index / nsample / C);

  if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

  int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
  for (int k = 1; k < B; k++) {
    if (pt_idx < pt_cnt) break;
    pt_cnt += idx_batch_cnt[k];
    bs_idx = k;
  }

  int features_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++)
    features_batch_start_idx += features_batch_cnt[k];

  grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;
  idx += pt_idx * nsample + sample_idx;
  grad_features += (features_batch_start_idx + idx[0]) * C + C_idx;

  atomicAdd(grad_features, grad_out[0]);
}

void group_points_grad_kernel_launcher_stack(int B, int M, int C, int N,
                                             int nsample, const float *grad_out,
                                             const int *idx,
                                             const int *idx_batch_cnt,
                                             const int *features_batch_cnt,
                                             float *grad_features) {
  // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the
  // output from forward :param idx: (M1 + M2 ..., nsample) tensor containing
  // the indices of features to group with :param idx_batch_cnt: (batch_size)
  // [M1 + M2 ...] tensor containing the indices of features to group with
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the
  // indices of features to group with :return:
  //     grad_features: (N1 + N2 ..., C) gradient of the features

  cudaError_t err;
  // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  //
  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(M * C * nsample,
                    THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_grad_kernel_stack<<<blocks, threads>>>(
      B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt,
      grad_features);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void group_points_kernel_stack(int B, int M, int C, int nsample,
                                          const float *features,
                                          const int *features_batch_cnt,
                                          const int *idx,
                                          const int *idx_batch_cnt,
                                          float *out) {
  // :param features: (N1 + N2 ..., C) tensor of features to group
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the
  // indices of features to group with :param idx: (M1 + M2 ..., nsample)
  // tensor containing the indices of features to group with :param
  // idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indices of
  // features to group with :return:
  //     output: (M1 + M2, C, nsample) tensor
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_idx = index % nsample;
  int C_idx = (index / nsample) % C;
  int pt_idx = (index / nsample / C);

  if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

  int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
  for (int k = 1; k < B; k++) {
    if (pt_idx < pt_cnt) break;
    pt_cnt += idx_batch_cnt[k];
    bs_idx = k;
  }

  int features_batch_start_idx = 0;
  for (int k = 0; k < bs_idx; k++)
    features_batch_start_idx += features_batch_cnt[k];
  features += features_batch_start_idx * C;

  idx += pt_idx * nsample + sample_idx;
  int in_idx = idx[0] * C + C_idx;
  int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

  out[out_idx] = features[in_idx];
}

void group_points_kernel_launcher_stack(const int B, const int M, const int C,
                                        const int nsample,
                                        const float *features,
                                        const int *features_batch_cnt,
                                        const int *idx,
                                        const int *idx_batch_cnt, float *out) {
  // :param features: (N1 + N2 ..., C) tensor of features to group
  // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the
  // indices of features to group with :param idx: (M1 + M2 ..., nsample)
  // tensor containing the indices of features to group with :param
  // idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indices of
  // features to group with :return:
  //     output: (M1 + M2, C, nsample) tensor

  cudaError_t err;
  dim3 blocks(DIVUP(M * C * nsample,
                    THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_kernel_stack<<<blocks, threads>>>(
      B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
