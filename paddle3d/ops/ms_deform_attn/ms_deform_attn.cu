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

/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_deform_attn_cuda_kernel.h"
#include "paddle/extension.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

template <typename scalar_t>
void ms_deformable_col2im_cuda(
    cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  const int num_threads =
      (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > 1024) {
    if ((channels & 1023) == 0) {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
             num_threads * 3 * sizeof(scalar_t), stream>>>(
              num_kernels, grad_col, data_value, data_spatial_shapes,
              data_level_start_index, data_sampling_loc, data_attn_weight,
              batch_size, spatial_size, num_heads, channels, num_levels,
              num_query, num_point, grad_value, grad_sampling_loc,
              grad_attn_weight);
    } else {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
    }
  } else {
    switch (channels) {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      1>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      2>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      4>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      8>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      16>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      32>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      64>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      128>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      256>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      512>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      default:
        if (channels < 64) {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                 num_threads * 3 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight);
        } else {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                 num_threads * 3 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

std::vector<paddle::Tensor> ms_deform_attn_forward_cuda(
    const paddle::Tensor &value, const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights, const int im2col_step) {
  CHECK_INPUT(value);
  CHECK_INPUT(spatial_shapes);
  CHECK_INPUT(level_start_index);
  CHECK_INPUT(sampling_locations);
  CHECK_INPUT(attention_weights);

  const int batch = value.shape()[0];
  const int spatial_size = value.shape()[1];
  const int num_heads = value.shape()[2];
  const int channels = value.shape()[3];

  const int num_levels = spatial_shapes.shape()[0];

  const int num_query = sampling_locations.shape()[1];
  const int num_point = sampling_locations.shape()[4];

  const int im2col_step_ = std::min(batch, im2col_step);

  PD_CHECK(batch % im2col_step_ == 0, "batch(", batch,
           ") must divide im2col_step(", im2col_step_, ")");

  auto output = paddle::full({batch, num_query, num_heads * channels}, 0,
                             value.type(), paddle::GPUPlace());

  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto per_output_size = num_query * num_heads * channels;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    const int num_kernels = im2col_step_ * per_output_size;
    const int num_actual_kernels = im2col_step_ * per_output_size;
    const int num_threads = CUDA_NUM_THREADS;

    PD_DISPATCH_FLOATING_TYPES(
        value.type(), "ms_deform_attn_forward_cuda", ([&] {
          ms_deformable_im2col_gpu_kernel<data_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
                 value.stream()>>>(
                  num_kernels,
                  value.data<data_t>() + n * im2col_step_ * per_value_size,
                  spatial_shapes.data<int64_t>(),
                  level_start_index.data<int64_t>(),
                  sampling_locations.data<data_t>() +
                      n * im2col_step_ * per_sample_loc_size,
                  attention_weights.data<data_t>() +
                      n * im2col_step_ * per_attn_weight_size,
                  im2col_step_, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point,
                  output.data<data_t>() + n * im2col_step_ * per_output_size);
        }));
  }

  return {output};
}

std::vector<paddle::Tensor> ms_deform_attn_backward_cuda(
    const paddle::Tensor &grad_out, const paddle::Tensor &value,
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights, const int im2col_step) {
  CHECK_INPUT(value);
  CHECK_INPUT(spatial_shapes);
  CHECK_INPUT(level_start_index);
  CHECK_INPUT(sampling_locations);
  CHECK_INPUT(attention_weights);
  CHECK_INPUT(grad_out);

  const int batch = value.shape()[0];
  const int spatial_size = value.shape()[1];
  const int num_heads = value.shape()[2];
  const int channels = value.shape()[3];

  const int num_levels = spatial_shapes.shape()[0];

  const int num_query = sampling_locations.shape()[1];
  const int num_point = sampling_locations.shape()[4];

  const int im2col_step_ = std::min(batch, im2col_step);

  const int batch_n = im2col_step_;
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;

  auto grad_value =
      paddle::full(value.shape(), 0, value.type(), paddle::GPUPlace());
  auto grad_sampling_loc =
      paddle::full(sampling_locations.shape(), 0, sampling_locations.type(),
                   paddle::GPUPlace());
  auto grad_attn_weight =
      paddle::full(attention_weights.shape(), 0, attention_weights.type(),
                   paddle::GPUPlace());

  for (int n = 0; n < batch / im2col_step_; ++n) {
    PD_DISPATCH_FLOATING_TYPES(
        value.type(), "ms_deform_attn_backward_cuda", ([&] {
          ms_deformable_col2im_cuda<data_t>(
              value.stream(),
              grad_out.data<data_t>() + n * im2col_step_ * per_value_size,
              value.data<data_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(),
              sampling_locations.data<data_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attention_weights.data<data_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              im2col_step_, spatial_size, num_heads, channels, num_levels,
              num_query, num_point,
              grad_value.data<data_t>() + n * im2col_step_ * per_value_size,
              grad_sampling_loc.data<data_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              grad_attn_weight.data<data_t>() +
                  n * im2col_step_ * per_attn_weight_size);
        }));
  }

  return {grad_value, grad_sampling_loc, grad_attn_weight};
  // return {grad_value};
}