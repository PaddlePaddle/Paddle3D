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
// https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cuda/scatter_points_cuda.cu
// Copyright (c) OpenMMLab. All rights reserved.

#include <vector>

#include "dynamic_scatter_op.h"
#include "paddle/extension.h"

#define CHECK_INPUT_CUDA(x) \
  PD_CHECK(x.is_gpu() || x.is_gpu_pinned(), #x " must be a GPU Tensor.")

int const threadsPerBlock = 512;
int const maxGridDim = 50000;

__device__ __forceinline__ static void reduceMax(float *address, float val) {
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old || __int_as_float(old) < val);
}

__device__ __forceinline__ static void reduceMax(double *address, double val) {
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old || __longlong_as_double(old) < val);
}

// get rid of meaningless warnings when compiling host code
#ifdef __CUDA_ARCH__
__device__ __forceinline__ static void reduceAdd(float *address, float val) {
#if (__CUDA_ARCH__ < 200)
#warning \
    "compute capability lower than 2.x. fall back to use CAS version of atomicAdd for float32"
  int *address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(val + __int_as_float(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}

__device__ __forceinline__ static void reduceAdd(double *address, double val) {
#if (__CUDA_ARCH__ < 600)
#warning \
    "compute capability lower than 6.x. fall back to use CAS version of atomicAdd for float64"
  unsigned long long *address_as_ull =
      reinterpret_cast<unsigned long long *>(address);
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, val);
#endif
}
#endif

template <typename T>
__global__ void feats_reduce_kernel(
    const T *feats, const int32_t *coors_map,
    T *reduced_feats,  // shall be 0 at initialization
    const int num_input, const int num_feats, const reduce_t reduce_type) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) continue;

    const T *feats_offset = feats + x * num_feats;
    T *reduced_feats_offset = reduced_feats + reduce_to * num_feats;
    if (reduce_type == reduce_t::MAX) {
      for (int i = 0; i < num_feats; i++) {
        reduceMax(&reduced_feats_offset[i], feats_offset[i]);
      }
    } else {
      for (int i = 0; i < num_feats; i++) {
        reduceAdd(&reduced_feats_offset[i], feats_offset[i]);
      }
    }
  }
}

__global__ void clean_coors_kernel(const int32_t *coors, int32_t *coors_clean,
                                   const int32_t compare_value,
                                   const int32_t fill_value,
                                   const int num_input, const int NDim) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    bool less_than = false;
    int idx = x * NDim;
    for (int i = 0; i < NDim; ++i) {
      if (coors[idx + i] < compare_value) {
        less_than = true;
        break;
      }
    }
    if (less_than == true) {
      for (int i = 0; i < NDim; ++i) {
        coors_clean[idx + i] = fill_value;
      }
    }
  }
}

template <typename T>
__global__ void cast_kernel(const int32_t *in_data, T *out_data,
                            const int num) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num;
       x += gridDim.x * blockDim.x) {
    out_data[x] = static_cast<T>(in_data[x]);
  }
}

std::vector<paddle::Tensor> dynamic_point_to_voxel_forward_cuda(
    const paddle::Tensor &feats, const paddle::Tensor &coors,
    const reduce_t reduce_type) {
  CHECK_INPUT_CUDA(feats);
  CHECK_INPUT_CUDA(coors);

  const int num_input = feats.shape()[0];
  const int num_feats = feats.shape()[1];
  const int NDim = coors.shape()[1];

  if (num_input == 0)
    return {feats.copy_to(paddle::GPUPlace(), true),
            coors.copy_to(paddle::GPUPlace(), true),
            paddle::full({0}, 0, paddle::DataType::INT32, paddle::GPUPlace()),
            paddle::full({0}, 0, paddle::DataType::INT32, paddle::GPUPlace())};

  paddle::Tensor coors_clean = coors.copy_to(coors.place(), true);
  int col_blocks = (num_input + threadsPerBlock - 1) / threadsPerBlock;
  col_blocks = col_blocks < maxGridDim ? col_blocks : maxGridDim;
  clean_coors_kernel<<<col_blocks, threadsPerBlock, 0, coors.stream()>>>(
      coors.data<int32_t>(), coors_clean.data<int32_t>(), 0, -1, num_input,
      NDim);

  auto unique_outs = paddle::unique(coors_clean, false, true, true, {0},
                                    paddle::DataType::INT32);
  auto out_coors = std::get<0>(unique_outs);
  auto coors_map = std::get<2>(unique_outs);
  auto reduce_count = std::get<3>(unique_outs);

  // the first element of out_coors (-1,-1,-1) and should be removed
  out_coors =
      paddle::strided_slice(out_coors, {0}, {1}, {out_coors.shape()[0]}, {1});
  reduce_count = paddle::strided_slice(reduce_count, {0}, {1},
                                       {reduce_count.shape()[0]}, {1});
  coors_map = paddle::subtract(
      coors_map, paddle::full({1}, 1, coors_map.type(), paddle::GPUPlace()));

  paddle::Tensor reduced_feats;
  PD_DISPATCH_FLOATING_TYPES(
      feats.type(), "feats_reduce_kernel", ([&] {
        if (reduce_type == reduce_t::MAX) {
          reduced_feats = paddle::full({out_coors.shape()[0], num_feats},
                                       -std::numeric_limits<data_t>::infinity(),
                                       feats.type(), paddle::GPUPlace());
        } else {
          reduced_feats = paddle::full({out_coors.shape()[0], num_feats}, 0,
                                       feats.type(), paddle::GPUPlace());
        }
        feats_reduce_kernel<data_t>
            <<<col_blocks, threadsPerBlock, 0, feats.stream()>>>(
                feats.data<data_t>(), coors_map.data<int32_t>(),
                reduced_feats.data<data_t>(), num_input, num_feats,
                reduce_type);
      }));

  if (reduce_type == reduce_t::MEAN) {
    auto uns_reduce_count = paddle::unsqueeze(reduce_count, {1});
    auto float_reduce_count = paddle::empty(
        uns_reduce_count.shape(), reduced_feats.type(), paddle::GPUPlace());
    col_blocks =
        (uns_reduce_count.shape()[0] + threadsPerBlock - 1) / threadsPerBlock;
    col_blocks = col_blocks < maxGridDim ? col_blocks : maxGridDim;
    PD_DISPATCH_FLOATING_TYPES(
        reduced_feats.type(), "cast_kernel", ([&] {
          cast_kernel<data_t>
              <<<col_blocks, threadsPerBlock, 0, reduced_feats.stream()>>>(
                  uns_reduce_count.data<int>(),
                  float_reduce_count.data<data_t>(),
                  uns_reduce_count.shape()[0]);
        }));
    reduced_feats = paddle::divide(reduced_feats, float_reduce_count);
  }
  return {reduced_feats, out_coors, coors_map, reduce_count};
}

template <typename T>
__global__ void add_reduce_traceback_grad_kernel(
    T *grad_feats, const T *grad_reduced_feats, const int32_t *coors_map,
    const int32_t *reduce_count, const int num_input, const int num_feats,
    const reduce_t reduce_type) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];
    if (reduce_to == -1) {
      continue;
    }

    const int input_offset = x * num_feats;
    T *grad_feats_offset = grad_feats + input_offset;
    const int reduced_offset = reduce_to * num_feats;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    if (reduce_type == reduce_t::SUM) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i];
      }
    } else if (reduce_type == reduce_t::MEAN) {
      for (int i = 0; i < num_feats; i++) {
        grad_feats_offset[i] = grad_reduced_feats_offset[i] /
                               static_cast<T>(reduce_count[reduce_to]);
      }
    }
  }
}

template <typename T>
__global__ void max_reduce_traceback_scatter_idx_kernel(
    const T *feats, const T *reduced_feats, int32_t *reduce_from,
    const int32_t *coors_map, const int num_input, const int num_feats) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_input;
       x += gridDim.x * blockDim.x) {
    int32_t reduce_to = coors_map[x];

    const int input_offset = x * num_feats;
    const T *feats_offset = feats + input_offset;

    if (reduce_to == -1) {
      continue;
    }

    const int reduced_offset = reduce_to * num_feats;
    const T *reduced_feats_offset = reduced_feats + reduced_offset;
    int32_t *reduce_from_offset = reduce_from + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      if (feats_offset[i] == reduced_feats_offset[i]) {
        atomicMin(&reduce_from_offset[i], static_cast<int32_t>(x));
      }
    }
  }
}

template <typename T>
__global__ void max_reduce_scatter_grad_kernel(T *grad_feats,
                                               const T *grad_reduced_feats,
                                               const int32_t *reduce_from,
                                               const int num_reduced,
                                               const int num_feats) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < num_reduced;
       x += gridDim.x * blockDim.x) {
    const int reduced_offset = x * num_feats;
    const int32_t *scatter_to_offset = reduce_from + reduced_offset;
    const T *grad_reduced_feats_offset = grad_reduced_feats + reduced_offset;

    for (int i = 0; i < num_feats; i++) {
      grad_feats[scatter_to_offset[i] * num_feats + i] =
          grad_reduced_feats_offset[i];
    }
  }
}

std::vector<paddle::Tensor> dynamic_point_to_voxel_backward_cuda(
    const paddle::Tensor &grad_reduced_feats, const paddle::Tensor &feats,
    const paddle::Tensor &reduced_feats, const paddle::Tensor &coors_map,
    const paddle::Tensor &reduce_count, const reduce_t reduce_type) {
  CHECK_INPUT_CUDA(grad_reduced_feats);
  CHECK_INPUT_CUDA(feats);
  CHECK_INPUT_CUDA(reduced_feats);
  CHECK_INPUT_CUDA(coors_map);
  CHECK_INPUT_CUDA(reduce_count);

  const int num_input = feats.shape()[0];
  const int num_reduced = reduced_feats.shape()[0];
  const int num_feats = feats.shape()[1];

  auto grad_feats = paddle::full(feats.shape(), 0, grad_reduced_feats.type(),
                                 paddle::GPUPlace());
  // copy voxel grad to points

  if (num_input == 0 || num_reduced == 0) return {grad_feats};

  int col_blocks = (num_input + threadsPerBlock - 1) / threadsPerBlock;
  col_blocks = col_blocks < maxGridDim ? col_blocks : maxGridDim;

  if (reduce_type == reduce_t::MEAN || reduce_type == reduce_t::SUM) {
    PD_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.type(), "add_reduce_traceback_grad_kernel", ([&] {
          add_reduce_traceback_grad_kernel<data_t>
              <<<col_blocks, threadsPerBlock, 0, grad_reduced_feats.stream()>>>(
                  grad_feats.data<data_t>(), grad_reduced_feats.data<data_t>(),
                  coors_map.data<int32_t>(), reduce_count.data<int32_t>(),
                  num_input, num_feats, reduce_type);
        }));
  } else {
    auto reduce_from =
        paddle::full({num_reduced, num_feats}, num_input,
                     paddle::DataType::INT32, paddle::GPUPlace());
    PD_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.type(), "max_reduce_traceback_scatter_idx_kernel",
        ([&] {
          max_reduce_traceback_scatter_idx_kernel<data_t>
              <<<col_blocks, threadsPerBlock, 0, grad_reduced_feats.stream()>>>(
                  feats.data<data_t>(), reduced_feats.data<data_t>(),
                  reduce_from.data<int32_t>(), coors_map.data<int32_t>(),
                  num_input, num_feats);
        }));

    col_blocks = (num_reduced + threadsPerBlock - 1) / threadsPerBlock;
    col_blocks = col_blocks < maxGridDim ? col_blocks : maxGridDim;
    PD_DISPATCH_FLOATING_TYPES(
        grad_reduced_feats.type(), "max_reduce_scatter_grad_kernel", ([&] {
          max_reduce_scatter_grad_kernel<data_t>
              <<<col_blocks, threadsPerBlock, 0, grad_reduced_feats.stream()>>>(
                  grad_feats.data<data_t>(), grad_reduced_feats.data<data_t>(),
                  reduce_from.data<int32_t>(), num_reduced, num_feats);
        }));
  }
  return {grad_feats};
}
