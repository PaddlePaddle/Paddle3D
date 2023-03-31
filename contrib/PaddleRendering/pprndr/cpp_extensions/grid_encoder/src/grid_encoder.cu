#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "paddle/extension.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor")
#define CHECK_IS_INT(x)                             \
  PD_CHECK(x.dtype() == paddle::DataType::INT32, #x \
           " must be an int32 "                     \
           "tensor")
#define CHECK_IS_FLOATING(x)                             \
  PD_CHECK(x.dtype() == paddle::DataType::FLOAT32 ||     \
               x.dtype() == paddle::DataType::FLOAT16 || \
               x.dtype() == paddle::DataType::FLOAT64,   \
           #x " must be a floating tensor")

// just for compatability of half precision in
// AT_DISPATCH_FLOATING_TYPES_AND_HALF...
static inline __device__ paddle::float16 atomicAdd(paddle::float16 *address,
                                                   paddle::float16 val) {
  // requires CUDA >= 10 and ARCH >= 70
  // this is very slow compared to float or __half2, and never used.
  // return atomicAdd(reinterpret_cast<__half*>(address), val);
}

template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
  static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

  // While 1 is technically not a good prime for hashing (or a prime at all), it
  // helps memory coherence and is sufficient for our use case of obtaining a
  // uniformly colliding index from high-dimensional coordinates.
  constexpr uint32_t primes[7] = {1u,          2654435761u, 805459861u,
                                  3674653429u, 2097192037u, 1434869437u,
                                  2165219737u};

  uint32_t result = 0;
#pragma unroll
  for (uint32_t i = 0; i < D; ++i) {
    result ^= pos_grid[i] * primes[i];
  }

  return result;
}

template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const uint32_t gridtype,
                                   const bool align_corners, const uint32_t ch,
                                   const uint32_t hashmap_size,
                                   const uint32_t resolution,
                                   const uint32_t pos_grid[D]) {
  uint32_t stride = 1;
  uint32_t index = 0;

#pragma unroll
  for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
    index += pos_grid[d] * stride;
    stride *= align_corners ? resolution : (resolution + 1);
  }

  // NOTE: for NeRF, the hash is in fact not necessary. Check
  // https://github.com/NVlabs/instant-ngp/issues/97. gridtype: 0 == hash, 1 ==
  // tiled
  if (gridtype == 0 && stride > hashmap_size) {
    index = fast_hash<D>(pos_grid);
  }

  return (index % hashmap_size) * C + ch;
}

template <typename data_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(const float *__restrict__ inputs,
                            const data_t *__restrict__ grid,
                            const int *__restrict__ offsets,
                            data_t *__restrict__ outputs, const uint32_t B,
                            const uint32_t L, const float S, const uint32_t H,
                            data_t *__restrict__ dy_dx, const uint32_t gridtype,
                            const bool align_corners) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= B) return;

  const uint32_t level = blockIdx.y;

  // locate
  grid += (uint32_t)offsets[level] * C;
  inputs += b * D;
  outputs += level * B * C + b * C;

  // check input range (should be in [0, 1])
  bool flag_oob = false;
#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    if (inputs[d] < 0 || inputs[d] > 1) {
      flag_oob = true;
    }
  }
  // if input out of bound, just set output to 0
  if (flag_oob) {
#pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
      outputs[ch] = 0;
    }
    if (dy_dx) {
      dy_dx += b * D * L * C + level * D * C;  // B L D C
#pragma unroll
      for (uint32_t d = 0; d < D; d++) {
#pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
          dy_dx[d * C + ch] = 0;
        }
      }
    }
    return;
  }

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution = (uint32_t)ceil(scale) + 1;

  // calculate coordinate
  float pos[D];
  uint32_t pos_grid[D];

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    pos[d] -= (float)pos_grid[d];
  }

  // printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1],
  // pos_grid[0], pos_grid[1]);

  // interpolate
  data_t results[C] = {static_cast<data_t>(0)};  // temp results in register

#pragma unroll
  for (uint32_t idx = 0; idx < (1 << D); idx++) {
    float w = 1;
    uint32_t pos_grid_local[D];

#pragma unroll
    for (uint32_t d = 0; d < D; d++) {
      if ((idx & (1 << d)) == 0) {
        w *= 1 - pos[d];
        pos_grid_local[d] = pos_grid[d];
      } else {
        w *= pos[d];
        pos_grid_local[d] = pos_grid[d] + 1;
      }
    }

    uint32_t index = get_grid_index<D, C>(
        gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

// writing to register (fast)
#pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
      results[ch] +=
          static_cast<data_t>(w * static_cast<float>(grid[index + ch]));
    }

    // printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx,
    // index, w, grid[index]);
  }

// writing to global memory (slow)
#pragma unroll
  for (uint32_t ch = 0; ch < C; ch++) {
    outputs[ch] = results[ch];
  }

  // prepare dy_dx
  // differentiable (soft) indexing:
  // https://discuss.pytorch.org/t/differentiable-indexing/17647/9
  if (dy_dx) {
    dy_dx += b * D * L * C + level * D * C;  // B L D C

#pragma unroll
    for (uint32_t gd = 0; gd < D; gd++) {
      data_t results_grad[C] = {static_cast<data_t>(0)};

#pragma unroll
      for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
        float w = scale;
        uint32_t pos_grid_local[D];

#pragma unroll
        for (uint32_t nd = 0; nd < D - 1; nd++) {
          const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

          if ((idx & (1 << nd)) == 0) {
            w *= 1 - pos[d];
            pos_grid_local[d] = pos_grid[d];
          } else {
            w *= pos[d];
            pos_grid_local[d] = pos_grid[d] + 1;
          }
        }

        pos_grid_local[gd] = pos_grid[gd];
        uint32_t index_left =
            get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size,
                                 resolution, pos_grid_local);
        pos_grid_local[gd] = pos_grid[gd] + 1;
        uint32_t index_right =
            get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size,
                                 resolution, pos_grid_local);

#pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
          results_grad[ch] += static_cast<data_t>(
              w * static_cast<float>(grid[index_right + ch] -
                                     grid[index_left + ch]));
        }
      }

#pragma unroll
      for (uint32_t ch = 0; ch < C; ch++) {
        dy_dx[gd * C + ch] = results_grad[ch];
      }
    }
  }
}

template <typename data_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const data_t *__restrict__ grad, const float *__restrict__ inputs,
    const data_t *__restrict__ grid, const int *__restrict__ offsets,
    data_t *__restrict__ grad_grid, const uint32_t B, const uint32_t L,
    const float S, const uint32_t H, const uint32_t gridtype,
    const bool align_corners) {
  const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
  if (b >= B) return;

  const uint32_t level = blockIdx.y;
  const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

  // locate
  grad_grid += offsets[level] * C;
  inputs += b * D;
  grad += level * B * C + b * C + ch;  // L, B, C

  const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
  const float scale = exp2f(level * S) * H - 1.0f;
  const uint32_t resolution = (uint32_t)ceil(scale) + 1;

// check input range (should be in [0, 1])
#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    if (inputs[d] < 0 || inputs[d] > 1) {
      return;  // grad is init as 0, so we simply return.
    }
  }

  // calculate coordinate
  float pos[D];
  uint32_t pos_grid[D];

#pragma unroll
  for (uint32_t d = 0; d < D; d++) {
    pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
    pos_grid[d] = floorf(pos[d]);
    pos[d] -= (float)pos_grid[d];
  }

  data_t grad_cur[N_C] = {static_cast<data_t>(0)};  // fetch to register
#pragma unroll
  for (uint32_t c = 0; c < N_C; c++) {
    grad_cur[c] = grad[c];
  }

// interpolate
#pragma unroll
  for (uint32_t idx = 0; idx < (1 << D); idx++) {
    float w = 1;
    uint32_t pos_grid_local[D];

#pragma unroll
    for (uint32_t d = 0; d < D; d++) {
      if ((idx & (1 << d)) == 0) {
        w *= 1 - pos[d];
        pos_grid_local[d] = pos_grid[d];
      } else {
        w *= pos[d];
        pos_grid_local[d] = pos_grid[d] + 1;
      }
    }

    uint32_t index = get_grid_index<D, C>(
        gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

    // atomicAdd for __half is slow (especially for large values), so we use
    // __half2 if N_C % 2 == 0
    // TODO: use float which is better than __half, if N_C % 2 != 0
    if (std::is_same<data_t, paddle::float16>::value && N_C % 2 == 0) {
#pragma unroll
      for (uint32_t c = 0; c < N_C; c += 2) {
        // process two __half at once (by interpreting as a __half2)
        __half2 v = {(__half)(w * static_cast<float>(grad_cur[c])),
                     (__half)(w * static_cast<float>(grad_cur[c + 1]))};
        atomicAdd((__half2 *)&grad_grid[index + c], v);
      }
      // float, or __half when N_C % 2 != 0 (which means C == 1)
    } else {
#pragma unroll
      for (uint32_t c = 0; c < N_C; c++) {
        atomicAdd(&grad_grid[index + c],
                  static_cast<data_t>(w * static_cast<float>(grad_cur[c])));
      }
    }
  }
}

template <typename data_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(const data_t *__restrict__ grad,
                                      const data_t *__restrict__ dy_dx,
                                      data_t *__restrict__ grad_inputs,
                                      uint32_t B, uint32_t L) {
  const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
  if (t >= B * D) return;

  const uint32_t b = t / D;
  const uint32_t d = t - b * D;

  dy_dx += b * L * D * C;

  data_t result = static_cast<data_t>(0);

#pragma unroll
  for (int l = 0; l < L; l++) {
#pragma unroll
    for (int ch = 0; ch < C; ch++) {
      result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
    }
  }

  grad_inputs[t] = result;
}

template <typename data_t, uint32_t D>
void kernel_grid_wrapper(const float *inputs, const data_t *embeddings,
                         const int *offsets, data_t *outputs, const uint32_t B,
                         const uint32_t C, const uint32_t L, const float S,
                         const uint32_t H, data_t *dy_dx,
                         const uint32_t gridtype, const bool align_corners,
                         cudaStream_t stream) {
  static constexpr uint32_t N_THREAD = 512;
  const dim3 blocks_hashgrid = {div_round_up(B, N_THREAD), L, 1};
  switch (C) {
    case 1:
      kernel_grid<data_t, D, 1><<<blocks_hashgrid, N_THREAD, 0, stream>>>(
          inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype,
          align_corners);
      break;
    case 2:
      kernel_grid<data_t, D, 2><<<blocks_hashgrid, N_THREAD, 0, stream>>>(
          inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype,
          align_corners);
      break;
    case 4:
      kernel_grid<data_t, D, 4><<<blocks_hashgrid, N_THREAD, 0, stream>>>(
          inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype,
          align_corners);
      break;
    case 8:
      kernel_grid<data_t, D, 8><<<blocks_hashgrid, N_THREAD, 0, stream>>>(
          inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx, gridtype,
          align_corners);
      break;
    default:
      throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
  }
}

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [L, B, C], float (L first, so only one level of hashmap needs to fit
// into cache at a time.) H: base resolution dy_dx: [B, L * D * C]
template <typename data_t>
void grid_encode_forward_cuda(const float *inputs, const data_t *embeddings,
                              const int *offsets, data_t *outputs,
                              const uint32_t B, const uint32_t D,
                              const uint32_t C, const uint32_t L, const float S,
                              const uint32_t H, data_t *dy_dx,
                              const uint32_t gridtype, const bool align_corners,
                              cudaStream_t stream) {
  switch (D) {
    case 1:
      kernel_grid_wrapper<data_t, 1>(inputs, embeddings, offsets, outputs, B, C,
                                     L, S, H, dy_dx, gridtype, align_corners,
                                     stream);
      break;
    case 2:
      kernel_grid_wrapper<data_t, 2>(inputs, embeddings, offsets, outputs, B, C,
                                     L, S, H, dy_dx, gridtype, align_corners,
                                     stream);
      break;
    case 3:
      kernel_grid_wrapper<data_t, 3>(inputs, embeddings, offsets, outputs, B, C,
                                     L, S, H, dy_dx, gridtype, align_corners,
                                     stream);
      break;
    case 4:
      kernel_grid_wrapper<data_t, 4>(inputs, embeddings, offsets, outputs, B, C,
                                     L, S, H, dy_dx, gridtype, align_corners,
                                     stream);
      break;
    case 5:
      kernel_grid_wrapper<data_t, 5>(inputs, embeddings, offsets, outputs, B, C,
                                     L, S, H, dy_dx, gridtype, align_corners,
                                     stream);
      break;
    default:
      throw std::runtime_error{"GridEncoding: D must be 1, 2, 3, 4, or 5."};
  }
}

template <typename data_t, uint32_t D>
void kernel_grid_backward_wrapper(
    const data_t *grad, const float *inputs, const data_t *embeddings,
    const int *offsets, data_t *grad_embeddings, const uint32_t B,
    const uint32_t C, const uint32_t L, const float S, const uint32_t H,
    const data_t *dy_dx, data_t *grad_inputs, const uint32_t gridtype,
    const bool align_corners, cudaStream_t stream) {
  static constexpr uint32_t N_THREAD = 256;
  const uint32_t N_C = std::min(2u, C);  // n_features_per_thread
  const dim3 blocks_hashgrid = {div_round_up(B * C / N_C, N_THREAD), L, 1};
  switch (C) {
    case 1:
      kernel_grid_backward<data_t, D, 1, 1>
          <<<blocks_hashgrid, N_THREAD, 0, stream>>>(
              grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H,
              gridtype, align_corners);
      if (dy_dx)
        kernel_input_backward<data_t, D, 1>
            <<<div_round_up(B * D, N_THREAD), N_THREAD, 0, stream>>>(
                grad, dy_dx, grad_inputs, B, L);
      break;
    case 2:
      kernel_grid_backward<data_t, D, 2, 2>
          <<<blocks_hashgrid, N_THREAD, 0, stream>>>(
              grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H,
              gridtype, align_corners);
      if (dy_dx)
        kernel_input_backward<data_t, D, 2>
            <<<div_round_up(B * D, N_THREAD), N_THREAD, 0, stream>>>(
                grad, dy_dx, grad_inputs, B, L);
      break;
    case 4:
      kernel_grid_backward<data_t, D, 4, 2>
          <<<blocks_hashgrid, N_THREAD, 0, stream>>>(
              grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H,
              gridtype, align_corners);
      if (dy_dx)
        kernel_input_backward<data_t, D, 4>
            <<<div_round_up(B * D, N_THREAD), N_THREAD, 0, stream>>>(
                grad, dy_dx, grad_inputs, B, L);
      break;
    case 8:
      kernel_grid_backward<data_t, D, 8, 2>
          <<<blocks_hashgrid, N_THREAD, 0, stream>>>(
              grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H,
              gridtype, align_corners);
      if (dy_dx)
        kernel_input_backward<data_t, D, 8>
            <<<div_round_up(B * D, N_THREAD), N_THREAD, 0, stream>>>(
                grad, dy_dx, grad_inputs, B, L);
      break;
    default:
      throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
  }
}

// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution
template <typename data_t>
void grid_encode_backward_cuda(const data_t *grad, const float *inputs,
                               const data_t *embeddings, const int *offsets,
                               data_t *grad_embeddings, const uint32_t B,
                               const uint32_t D, const uint32_t C,
                               const uint32_t L, const float S,
                               const uint32_t H, const data_t *dy_dx,
                               data_t *grad_inputs, const uint32_t gridtype,
                               const bool align_corners, cudaStream_t stream) {
  switch (D) {
    case 1:
      kernel_grid_backward_wrapper<data_t, 1>(
          grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H,
          dy_dx, grad_inputs, gridtype, align_corners, stream);
      break;
    case 2:
      kernel_grid_backward_wrapper<data_t, 2>(
          grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H,
          dy_dx, grad_inputs, gridtype, align_corners, stream);
      break;
    case 3:
      kernel_grid_backward_wrapper<data_t, 3>(
          grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H,
          dy_dx, grad_inputs, gridtype, align_corners, stream);
      break;
    case 4:
      kernel_grid_backward_wrapper<data_t, 4>(
          grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H,
          dy_dx, grad_inputs, gridtype, align_corners, stream);
      break;
    case 5:
      kernel_grid_backward_wrapper<data_t, 5>(
          grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H,
          dy_dx, grad_inputs, gridtype, align_corners, stream);
      break;
    default:
      throw std::runtime_error{"GridEncoding: D must be 1, 2, 3, 4, or 5."};
  }
}

std::vector<paddle::Tensor> grid_encode_forward(
    const paddle::Tensor &inputs, const paddle::Tensor &embeddings,
    const paddle::Tensor &offsets, const int64_t input_dim,
    const int64_t level_dim, const int64_t num_levels, const float scale,
    const int64_t base_resolution, const int64_t gridtype,
    const bool align_corners, const bool inputs_stop_gradient) {
  CHECK_CUDA(inputs);
  CHECK_CUDA(embeddings);
  CHECK_CUDA(offsets);

  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(embeddings);
  CHECK_IS_INT(offsets);

  int64_t B = inputs.shape()[0];
  auto outputs = paddle::empty({num_levels, B, level_dim}, embeddings.dtype(),
                               paddle::GPUPlace());
  auto dy_dx = paddle::empty({B, num_levels * input_dim * base_resolution},
                             embeddings.dtype(), paddle::GPUPlace());

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      embeddings.dtype(), "grid_encode_forward_cuda", ([&] {
        grid_encode_forward_cuda<data_t>(
            inputs.data<float>(), embeddings.data<data_t>(),
            offsets.data<int>(), outputs.data<data_t>(), B, input_dim,
            level_dim, num_levels, scale, base_resolution,
            inputs_stop_gradient ? nullptr : dy_dx.data<data_t>(), gridtype,
            align_corners, inputs.stream());
      }));

  return {outputs, dy_dx};
}

std::vector<paddle::Tensor> grid_encode_backward(
    const paddle::Tensor &grad, const paddle::Tensor &inputs,
    const paddle::Tensor &embeddings, const paddle::Tensor &offsets,
    const paddle::Tensor &dy_dx, const int64_t input_dim,
    const int64_t level_dim, const int64_t num_levels, const float scale,
    const int64_t base_resolution, const int64_t gridtype,
    const bool align_corners, const bool inputs_stop_gradient) {
  CHECK_CUDA(grad);
  CHECK_CUDA(inputs);
  CHECK_CUDA(embeddings);
  CHECK_CUDA(offsets);
  CHECK_CUDA(dy_dx);

  CHECK_IS_FLOATING(grad);
  CHECK_IS_FLOATING(inputs);
  CHECK_IS_FLOATING(embeddings);
  CHECK_IS_INT(offsets);
  CHECK_IS_FLOATING(dy_dx);

  auto grad_embeddings = paddle::experimental::full_like(embeddings, 0.0);
  auto grad_inputs =
      paddle::experimental::full_like(inputs, 0.0, embeddings.dtype());
  int64_t B = inputs.shape()[0];

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      grad.dtype(), "grid_encode_backward_cuda", ([&] {
        grid_encode_backward_cuda<data_t>(
            grad.data<data_t>(), inputs.data<float>(),
            embeddings.data<data_t>(), offsets.data<int>(),
            grad_embeddings.data<data_t>(), B, input_dim, level_dim, num_levels,
            scale, base_resolution,
            inputs_stop_gradient ? nullptr : dy_dx.data<data_t>(),
            inputs_stop_gradient ? nullptr : grad_inputs.data<data_t>(),
            gridtype, align_corners, grad.stream());
      }));

  return {grad_embeddings, grad_inputs.cast(inputs.dtype())};
}
