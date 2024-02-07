#include <stdio.h>
#include <stdlib.h>

#include "trt_bev_pool_kernel.hpp"
/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n_points]
    ranks_feat       : input index of feat, IntTensor[n_points]
    ranks_bev        : output index, IntTensor[n_points]
    interval_lengths : starting position for pooled point,
  IntTensor[n_intervals] interval_starts  : how many points in each pooled
  point, IntTensor[n_intervals] out              : output features,
  FloatTensor[b, z, h, w, c]
*/
__global__ void bev_pool_v2_kernel(
    int c, int n_intervals, const float* __restrict__ depth,
    const float* __restrict__ feat, const int* __restrict__ ranks_depth,
    const int* __restrict__ ranks_feat, const int* __restrict__ ranks_bev,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths, float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int* cur_rank = ranks_bev + interval_start;
  float* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

__global__ void bev_pool_v2_set_zero_kernel(int n_points,
                                            float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  float* cur_out = out + idx;
  *cur_out = 0.0;
}

void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
                 const int* ranks_depth, const int* ranks_feat,
                 const int* ranks_bev, const int* interval_starts,
                 const int* interval_lengths, float* out, cudaStream_t stream) {
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256, 0,
                       stream>>>(c, n_intervals, depth, feat, ranks_depth,
                                 ranks_feat, ranks_bev, interval_starts,
                                 interval_lengths, out);
}

void bev_pool_v2_set_zero(int n_points, float* out) {
  bev_pool_v2_set_zero_kernel<<<(int)ceil(((double)n_points / 256)), 256>>>(
      n_points, out);
}