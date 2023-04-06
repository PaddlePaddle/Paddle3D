// Kernels in this file are based on
// https://github.com/KAIR-BAIR/nerfacc/blob/master/nerfacc/cuda/csrc/ray_marching.cu

#include <vector>

#include "../include/contraction.h"
#include "helper_math.h"
#include "paddle/extension.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor")
#define CHECK_IS_FLOATING(x)                             \
  PD_CHECK(x.dtype() == paddle::DataType::FLOAT32 ||     \
               x.dtype() == paddle::DataType::FLOAT16 || \
               x.dtype() == paddle::DataType::FLOAT64,   \
           #x " must be a floating tensor")

inline __device__ __host__ float3 sign(float3 a) {
  return make_float3(copysignf(1.0f, a.x), copysignf(1.0f, a.y),
                     copysignf(1.0f, a.z));
}

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

inline __host__ __device__ void swapf(float& a, float& b) {
  float c = a;
  a = b;
  b = c;
}

inline __device__ __host__ float calc_dt(const float t, const float cone_angle,
                                         const float dt_min,
                                         const float dt_max) {
  return clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ __host__ int grid_idx_at(const float3 xyz_unit,
                                           const int3 grid_res) {
  // xyz should be always in [0, 1]^3.
  int3 ixyz = make_int3(xyz_unit * make_float3(grid_res));
  ixyz = clamp(ixyz, make_int3(0, 0, 0), grid_res - 1);
  int3 grid_offset = make_int3(grid_res.y * grid_res.z, grid_res.z, 1);
  int idx = dot(ixyz, grid_offset);
  return idx;
}

template <typename data_t>
inline __device__ __host__ data_t
grid_occupied_at(const float3 xyz, const float3 roi_min, const float3 roi_max,
                 ContractionType contraction_type, const int3 grid_res,
                 const data_t* grid_value) {
  if (contraction_type == ContractionType::AABB &&
      (xyz.x < roi_min.x || xyz.x > roi_max.x || xyz.y < roi_min.y ||
       xyz.y > roi_max.y || xyz.z < roi_min.z || xyz.z > roi_max.z)) {
    return false;
  }
  float3 xyz_unit = apply_contraction(xyz, roi_min, roi_max, contraction_type);
  int idx = grid_idx_at(xyz_unit, grid_res);
  return grid_value[idx];
}

// dda like step
inline __device__ __host__ float distance_to_next_voxel(
    const float3 xyz, const float3 dir, const float3 inv_dir,
    const float3 roi_min, const float3 roi_max, const int3 grid_res) {
  float3 _occ_res = make_float3(grid_res);
  float3 _xyz = roi_to_unit(xyz, roi_min, roi_max) * _occ_res;
  float3 txyz = ((floorf(_xyz + 0.5f + 0.5f * sign(dir)) - _xyz) * inv_dir) /
                _occ_res * (roi_max - roi_min);
  float t = min(min(txyz.x, txyz.y), txyz.z);
  return fmaxf(t, 0.0f);
}

inline __device__ __host__ float advance_to_next_voxel(
    const float t, const float dt_min, const float3 xyz, const float3 dir,
    const float3 inv_dir, const float3 roi_min, const float3 roi_max,
    const int3 grid_res, const float far) {
  // Regular stepping (may be slower but matches non-empty space)
  float t_target =
      t + distance_to_next_voxel(xyz, dir, inv_dir, roi_min, roi_max, grid_res);
  t_target = min(t_target, far);
  float _t = t;
  do {
    _t += dt_min;
  } while (_t < t_target);
  return _t;
}

// rays_o/d: [N, 3]
// nears/fars: [N]
// data_t should always be float in use.
template <typename data_t>
__global__ void kernel_near_far_from_aabb(const data_t* __restrict__ rays_o,
                                          const data_t* __restrict__ rays_d,
                                          const data_t* __restrict__ aabb,
                                          const uint32_t N, data_t* nears,
                                          data_t* fars) {
  // parallel per ray
  const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  // locate
  rays_o += n * 3;
  rays_d += n * 3;

  const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
  const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
  const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

  // get near far (assume cube scene)
  float near = (aabb[0] - ox) * rdx;
  float far = (aabb[3] - ox) * rdx;
  if (near > far) swapf(near, far);

  float near_y = (aabb[1] - oy) * rdy;
  float far_y = (aabb[4] - oy) * rdy;
  if (near_y > far_y) swapf(near_y, far_y);

  if (near > far_y || near_y > far) {
    nears[n] = fars[n] = 1e10f;
    return;
  }

  if (near_y > near) near = near_y;
  if (far_y < far) far = far_y;

  float near_z = (aabb[2] - oz) * rdz;
  float far_z = (aabb[5] - oz) * rdz;
  if (near_z > far_z) swapf(near_z, far_z);

  if (near > far_z || near_z > far) {
    nears[n] = fars[n] = 1e10f;
    return;
  }

  if (near_z > near) near = near_z;
  if (far_z < far) far = far_z;

  nears[n] = near;
  fars[n] = far;
}

__global__ void first_round_ray_marching_kernel(
    const uint32_t n_rays,
    const float* rays_o,  // shape (n_rays, 3)
    const float* rays_d,  // shape (n_rays, 3)
    const float* nears,   // shape (n_rays,)
    const float* fars,    // shape (n_rays,)
    const float* aabb, const int3 grid_res,
    const bool* grid_binary,  // shape (reso_x, reso_y, reso_z)
    const ContractionType contraction_type, const float step_size,
    const float cone_angle, int* num_steps) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  rays_o += i * 3;
  rays_d += i * 3;
  nears += i;
  fars += i;

  num_steps += i;

  const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
  const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
  const float3 inv_dir = 1.0f / dir;
  const float near = nears[0], far = fars[0];

  const float3 roi_min = make_float3(aabb[0], aabb[1], aabb[2]);
  const float3 roi_max = make_float3(aabb[3], aabb[4], aabb[5]);

  // TODO: compute dt_max from occ resolution.
  float dt_min = step_size;
  float dt_max = 1e10f;

  int j = 0;
  float t0 = near;
  float dt = calc_dt(t0, cone_angle, dt_min, dt_max);
  float t1 = t0 + dt;
  float t_mid = (t0 + t1) * 0.5f;

  while (t_mid < far) {
    // current center
    const float3 xyz = origin + t_mid * dir;
    if (grid_occupied_at(xyz, roi_min, roi_max, contraction_type, grid_res,
                         grid_binary)) {
      ++j;
      // march to next sample
      t0 = t1;
      t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max);
      t_mid = (t0 + t1) * 0.5f;
    } else {
      // march to next sample
      switch (contraction_type) {
        case ContractionType::AABB:
          // no contraction
          t_mid = advance_to_next_voxel(t_mid, dt_min, xyz, dir, inv_dir,
                                        roi_min, roi_max, grid_res, far);
          dt = calc_dt(t_mid, cone_angle, dt_min, dt_max);
          t0 = t_mid - dt * 0.5f;
          t1 = t_mid + dt * 0.5f;
          break;

        default:
          // any type of scene contraction does not work with DDA.
          t0 = t1;
          t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max);
          t_mid = (t0 + t1) * 0.5f;
          break;
      }
    }
  }

  *num_steps = j;

  return;
}

template <typename data_t>
__global__ void second_round_ray_marching_kernel(
    const uint32_t n_rays,
    const float* rays_o,  // shape (n_rays, 3)
    const float* rays_d,  // shape (n_rays, 3)
    const float* nears,   // shape (n_rays,)
    const float* fars,    // shape (n_rays,)
    const float* aabb, const int3 grid_res,
    const bool* grid_binary,  // shape (reso_x, reso_y, reso_z)
    const ContractionType contraction_type, const float step_size,
    const float cone_angle, const data_t* packed_info, float* t_starts,
    float* t_ends) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  rays_o += i * 3;
  rays_d += i * 3;
  nears += i;
  fars += i;

  int64_t base = packed_info[i * 2];
  int64_t steps = packed_info[i * 2 + 1];
  t_starts += base;
  t_ends += base;

  const float3 origin = make_float3(rays_o[0], rays_o[1], rays_o[2]);
  const float3 dir = make_float3(rays_d[0], rays_d[1], rays_d[2]);
  const float3 inv_dir = 1.0f / dir;
  const float near = nears[0], far = fars[0];

  const float3 roi_min = make_float3(aabb[0], aabb[1], aabb[2]);
  const float3 roi_max = make_float3(aabb[3], aabb[4], aabb[5]);

  // TODO: compute dt_max from occ resolution.
  float dt_min = step_size;
  float dt_max = 1e10f;

  int64_t j = 0;
  float t0 = near;
  float dt = calc_dt(t0, cone_angle, dt_min, dt_max);
  float t1 = t0 + dt;
  float t_mid = (t0 + t1) * 0.5f;

  while (t_mid < far) {
    // current center
    const float3 xyz = origin + t_mid * dir;
    if (grid_occupied_at(xyz, roi_min, roi_max, contraction_type, grid_res,
                         grid_binary)) {
      t_starts[j] = t0;
      t_ends[j] = t1;
      ++j;
      // march to next sample
      t0 = t1;
      t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max);
      t_mid = (t0 + t1) * 0.5f;
    } else {
      // march to next sample
      switch (contraction_type) {
        case ContractionType::AABB:
          // no contraction
          t_mid = advance_to_next_voxel(t_mid, dt_min, xyz, dir, inv_dir,
                                        roi_min, roi_max, grid_res, far);
          dt = calc_dt(t_mid, cone_angle, dt_min, dt_max);
          t0 = t_mid - dt * 0.5f;
          t1 = t_mid + dt * 0.5f;
          break;

        default:
          // any type of scene contraction does not work with DDA.
          t0 = t1;
          t1 = t0 + calc_dt(t0, cone_angle, dt_min, dt_max);
          t_mid = (t0 + t1) * 0.5f;
          break;
      }
    }
  }

  return;
}

template <typename data_t>
__global__ void pdf_ray_marching_kernel(
    const uint32_t n_rays,
    const int* packed_info,  // input ray & point indices.
    const data_t* starts,    // input start t
    const data_t* ends,      // input end t
    const data_t* weights,   // transmittance weights
    const int* resample_packed_info, data_t* resample_starts,
    data_t* resample_ends) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  const int base = packed_info[i * 2];       // point idx start.
  const int steps = packed_info[i * 2 + 1];  // point idx shift.
  const int resample_base = resample_packed_info[i * 2];
  const int resample_steps = resample_packed_info[i * 2 + 1];
  if (steps == 0) return;

  starts += base;
  ends += base;
  weights += base;
  resample_starts += resample_base;
  resample_ends += resample_base;

  // normalize weights **per ray**
  data_t weights_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < steps; j++) weights_sum += weights[j];
  data_t padding = fmaxf(1e-5f - weights_sum, 0.0f);
  data_t padding_step = padding / steps;
  weights_sum += padding;

  int num_bins = resample_steps + 1;
  data_t cdf_step_size = (1.0f - 1.0 / num_bins) / resample_steps;

  int idx = 0, j = 0;
  data_t cdf_prev = 0.0f,
         cdf_next = (weights[idx] + padding_step) / weights_sum;
  data_t cdf_u = 1.0 / (2 * num_bins);
  while (j < num_bins) {
    if (cdf_u < cdf_next) {
      // resample in this interval
      data_t scaling = (ends[idx] - starts[idx]) / (cdf_next - cdf_prev);
      data_t t = (cdf_u - cdf_prev) * scaling + starts[idx];
      if (j < num_bins - 1) resample_starts[j] = t;
      if (j > 0) resample_ends[j - 1] = t;
      // going further to next resample
      cdf_u += cdf_step_size;
      j += 1;
    } else {
      // going to next interval
      idx += 1;
      cdf_prev = cdf_next;
      cdf_next += (weights[idx] + padding_step) / weights_sum;
    }
  }
  if (j != num_bins) {
    printf("Error: %d %d %f\n", j, num_bins, weights_sum);
  }
  return;
}

template <typename data_t>
__global__ void query_occ_kernel(
    // rays info
    const uint32_t n_samples,
    const float* samples,  // shape (n_samples, 3)
    // occupancy grid & contraction
    const float* aabb, const int3 grid_res,
    const data_t* grid_value,  // shape (reso_x, reso_y, reso_z)
    const ContractionType contraction_type,
    // outputs
    data_t* occs) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_samples) return;

  // locate
  samples += i * 3;
  occs += i;

  const float3 roi_min = make_float3(aabb[0], aabb[1], aabb[2]);
  const float3 roi_max = make_float3(aabb[3], aabb[4], aabb[5]);
  const float3 xyz = make_float3(samples[0], samples[1], samples[2]);

  *occs = grid_occupied_at(xyz, roi_min, roi_max, contraction_type, grid_res,
                           grid_value);
  return;
}

__global__ void contract_kernel(
    // samples info
    const uint32_t n_samples,
    const float* samples,  // (n_samples, 3)
    // contraction
    const float* aabb, const ContractionType contraction_type,
    // outputs
    float* out_samples) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_samples) return;

  // locate
  samples += i * 3;
  out_samples += i * 3;

  const float3 roi_min = make_float3(aabb[0], aabb[1], aabb[2]);
  const float3 roi_max = make_float3(aabb[3], aabb[4], aabb[5]);
  const float3 xyz = make_float3(samples[0], samples[1], samples[2]);
  float3 xyz_unit = apply_contraction(xyz, roi_min, roi_max, contraction_type);

  out_samples[0] = xyz_unit.x;
  out_samples[1] = xyz_unit.y;
  out_samples[2] = xyz_unit.z;
  return;
}

__global__ void contract_inv_kernel(
    // samples info
    const uint32_t n_samples,
    const float* samples,  // (n_samples, 3)
    // contraction
    const float* aabb, const ContractionType type,
    // outputs
    float* out_samples) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_samples) return;

  // locate
  samples += i * 3;
  out_samples += i * 3;

  const float3 roi_min = make_float3(aabb[0], aabb[1], aabb[2]);
  const float3 roi_max = make_float3(aabb[3], aabb[4], aabb[5]);
  const float3 xyz_unit = make_float3(samples[0], samples[1], samples[2]);
  float3 xyz = apply_contraction_inv(xyz_unit, roi_min, roi_max, type);

  out_samples[0] = xyz.x;
  out_samples[1] = xyz.y;
  out_samples[2] = xyz.z;
  return;
}

template <typename data_t>
__global__ void transmittance_from_alpha_forward_kernel(
    const uint32_t n_rays,
    // inputs
    const data_t* packed_info, const float* alphas,
    // outputs
    float* transmittance) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  const data_t base = packed_info[i * 2];
  const data_t steps = packed_info[i * 2 + 1];
  if (steps == 0) return;

  alphas += base;
  transmittance += base;

  // accumulation
  float T = 1.0f;
#pragma unroll
  for (int64_t j = 0; j < steps; ++j) {
    transmittance[j] = T;
    T *= (1.0f - alphas[j]);
  }
  return;
}

template <typename data_t>
__global__ void transmittance_from_alpha_backward_kernel(
    const uint32_t n_rays,
    // inputs
    const data_t* packed_info, const float* alphas, const float* transmittance,
    const float* transmittance_grad,
    // outputs
    float* alphas_grad) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  const data_t base = packed_info[i * 2];
  const data_t steps = packed_info[i * 2 + 1];
  if (steps == 0) return;

  alphas += base;
  transmittance += base;
  transmittance_grad += base;
  alphas_grad += base;

  // accumulation
  float cumsum = 0.0f;
#pragma unroll
  for (int64_t j = steps - 1; j >= 0; --j) {
    alphas_grad[j] = cumsum / fmax(1.0f - alphas[j], 1e-10f);
    cumsum -= transmittance_grad[j] * transmittance[j];
  }
  return;
}

template <typename data_t>
__global__ void unpack_info_kernel(
    // input
    const int n_rays, const data_t* packed_info,
    // output
    data_t* ray_indices) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  const data_t base = packed_info[i * 2];       // point idx start.
  const data_t steps = packed_info[i * 2 + 1];  // point idx shift.
  if (steps == 0) return;

  ray_indices += base;
#pragma unroll
  for (int64_t j = 0; j < steps; ++j) {
    ray_indices[j] = i;
  }
}

template <typename int_data_t>
__global__ void weight_from_sigma_forward_kernel(
    const uint32_t n_rays, const int_data_t* packed_info, const float* starts,
    const float* ends, const float* sigmas,
    // outputs
    float* weights, const uint32_t n_samples = -1) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  int_data_t base;
  int_data_t steps;
  if (packed_info != nullptr) {
    base = packed_info[i * 2];
    steps = packed_info[i * 2 + 1];
  } else {
    base = i * n_samples;
    steps = n_samples;
  }
  if (steps == 0) return;

  starts += base;
  ends += base;
  sigmas += base;
  weights += base;

  // accumulation
  float T = 1.f;
#pragma unroll
  for (int64_t j = 0; j < steps; ++j) {
    const float delta = ends[j] - starts[j];
    const float alpha = 1.f - __expf(-sigmas[j] * delta);
    weights[j] = alpha * T;
    T *= (1.f - alpha);
  }
  return;
}

template <typename int_data_t>
__global__ void weight_from_sigma_backward_kernel(
    const uint32_t n_rays, const int_data_t* packed_info, const float* starts,
    const float* ends, const float* sigmas, const float* weights,
    const float* grad_weights,
    // outputs
    float* grad_sigmas, const uint32_t n_samples = -1) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  int_data_t base;
  int_data_t steps;
  if (packed_info != nullptr) {
    base = packed_info[i * 2];
    steps = packed_info[i * 2 + 1];
  } else {
    base = i * n_samples;
    steps = n_samples;
  }
  if (steps == 0) return;

  starts += base;
  ends += base;
  sigmas += base;
  weights += base;
  grad_weights += base;
  grad_sigmas += base;

  float accum = 0;
#pragma unroll
  for (int64_t j = 0; j < steps; ++j) {
    accum += grad_weights[j] * weights[j];
  }

  // accumulation
  float T = 1.f;
#pragma unroll
  for (int64_t j = 0; j < steps; ++j) {
    const float delta = ends[j] - starts[j];
    const float alpha = 1.f - __expf(-sigmas[j] * delta);
    grad_sigmas[j] = (grad_weights[j] * T - accum) * delta;
    accum -= grad_weights[j] * weights[j];
    T *= (1.f - alpha);
  }
  return;
}

std::vector<paddle::Tensor> near_far_from_aabb(const paddle::Tensor& rays_o,
                                               const paddle::Tensor& rays_d,
                                               const paddle::Tensor& aabb) {
  CHECK_CUDA(rays_o);
  CHECK_CUDA(rays_d);
  CHECK_CUDA(aabb);

  PD_CHECK(rays_o.shape().size() == 2 & rays_o.shape()[1] == 3);
  PD_CHECK(rays_d.shape().size() == 2 & rays_d.shape()[1] == 3);
  PD_CHECK(aabb.shape().size() == 1 & aabb.shape()[0] == 6);

  uint32_t n_rays = rays_o.shape()[0];
  auto nears = paddle::empty({n_rays, 1}, rays_o.dtype(), paddle::GPUPlace());
  auto fars = paddle::empty({n_rays, 1}, rays_o.dtype(), paddle::GPUPlace());

  static constexpr uint32_t N_THREAD = 256;

  PD_DISPATCH_FLOATING_TYPES(
      rays_o.dtype(), "kernel_near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<data_t>
            <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, rays_o.stream()>>>(
                rays_o.data<data_t>(), rays_d.data<data_t>(),
                aabb.data<data_t>(), n_rays, nears.data<data_t>(),
                fars.data<data_t>());
      }));

  return {nears, fars};
}

std::vector<paddle::Tensor> first_round_ray_marching(
    const paddle::Tensor& rays_o, const paddle::Tensor& rays_d,
    const paddle::Tensor& nears, const paddle::Tensor& fars,
    const paddle::Tensor& aabb, const paddle::Tensor& grid_binary,
    const int contraction_type_, const float step_size,
    const float cone_angle) {
  CHECK_CUDA(rays_o);
  CHECK_CUDA(rays_d);
  CHECK_CUDA(nears);
  CHECK_CUDA(fars);

  PD_CHECK(rays_o.shape().size() == 2 & rays_o.shape()[1] == 3);
  PD_CHECK(rays_d.shape().size() == 2 & rays_d.shape()[1] == 3);
  PD_CHECK(nears.shape().size() == 2 & nears.shape()[1] == 1);
  PD_CHECK(fars.shape().size() == 2 & fars.shape()[1] == 1);
  PD_CHECK(aabb.shape().size() == 1 & aabb.shape()[0] == 6);
  PD_CHECK(grid_binary.shape().size() == 3);

  ContractionType contraction_type =
      static_cast<ContractionType>(contraction_type_);

  uint32_t n_rays = rays_o.shape()[0];
  auto grid_binary_dims = grid_binary.shape();
  const int3 grid_res =
      make_int3(grid_binary_dims[0], grid_binary_dims[1], grid_binary_dims[2]);

  // helper counter
  auto num_steps =
      paddle::empty({n_rays}, paddle::DataType::INT32, paddle::GPUPlace());

  static constexpr uint32_t N_THREAD = 256;
  // count number of samples per ray
  first_round_ray_marching_kernel<<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                                    rays_o.stream()>>>(
      // rays
      n_rays, rays_o.data<float>(), rays_d.data<float>(), nears.data<float>(),
      fars.data<float>(),
      // occupancy grid & contraction
      aabb.data<float>(), grid_res, grid_binary.data<bool>(), contraction_type,
      // sampling
      step_size, cone_angle,
      // outputs
      num_steps.data<int>());

  return {num_steps};
}

std::vector<paddle::Tensor> second_round_ray_marching(
    const paddle::Tensor& rays_o, const paddle::Tensor& rays_d,
    const paddle::Tensor& nears, const paddle::Tensor& fars,
    const paddle::Tensor& aabb, const paddle::Tensor& grid_binary,
    const paddle::Tensor& packed_info, const int contraction_type_,
    const float step_size, const float cone_angle, const int64_t total_steps) {
  CHECK_CUDA(rays_o);
  CHECK_CUDA(rays_d);
  CHECK_CUDA(nears);
  CHECK_CUDA(fars);

  PD_CHECK(rays_o.shape().size() == 2 & rays_o.shape()[1] == 3);
  PD_CHECK(rays_d.shape().size() == 2 & rays_d.shape()[1] == 3);
  PD_CHECK(nears.shape().size() == 2 & nears.shape()[1] == 1);
  PD_CHECK(fars.shape().size() == 2 & fars.shape()[1] == 1);
  PD_CHECK(aabb.shape().size() == 1 & aabb.shape()[0] == 6);
  PD_CHECK(grid_binary.shape().size() == 3);

  ContractionType contraction_type =
      static_cast<ContractionType>(contraction_type_);

  uint32_t n_rays = rays_o.shape()[0];
  auto grid_binary_dims = grid_binary.shape();
  const int3 grid_res =
      make_int3(grid_binary_dims[0], grid_binary_dims[1], grid_binary_dims[2]);

  // output samples starts and ends
  auto t_starts =
      paddle::empty({total_steps, 1}, rays_o.dtype(), paddle::GPUPlace());
  auto t_ends =
      paddle::empty({total_steps, 1}, rays_o.dtype(), paddle::GPUPlace());

  static constexpr uint32_t N_THREAD = 256;

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      second_round_ray_marching_kernel<int>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, rays_o.stream()>>>(
              // rays
              n_rays, rays_o.data<float>(), rays_d.data<float>(),
              nears.data<float>(), fars.data<float>(),
              // occupancy grid & contraction
              aabb.data<float>(), grid_res, grid_binary.data<bool>(),
              contraction_type,
              // sampling
              step_size, cone_angle, packed_info.data<int>(),
              // outputs
              t_starts.data<float>(), t_ends.data<float>());
      break;
    case paddle::DataType::INT64:
      second_round_ray_marching_kernel<int64_t>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, rays_o.stream()>>>(
              // rays
              n_rays, rays_o.data<float>(), rays_d.data<float>(),
              nears.data<float>(), fars.data<float>(),
              // occupancy grid & contraction
              aabb.data<float>(), grid_res, grid_binary.data<bool>(),
              contraction_type,
              // sampling
              step_size, cone_angle, packed_info.data<int64_t>(),
              // outputs
              t_starts.data<float>(), t_ends.data<float>());
      break;
    default:
      PD_THROW(
          "function second_round_ray_marching_kernel is not implemented for "
          "packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {t_starts, t_ends};
}

std::vector<paddle::Tensor> pdf_ray_marching(
    const paddle::Tensor& starts, const paddle::Tensor& ends,
    const paddle::Tensor& weights, const paddle::Tensor& packed_info,
    const paddle::Tensor& resample_packed_info, const int64_t total_steps) {
  CHECK_CUDA(starts);
  CHECK_CUDA(ends);
  CHECK_CUDA(weights);
  CHECK_CUDA(packed_info);

  PD_CHECK(starts.shape().size() == 2 & starts.shape()[1] == 1);
  PD_CHECK(ends.shape().size() == 2 & ends.shape()[1] == 1);
  PD_CHECK((weights.shape().size() == 2 & weights.shape()[1] == 1) |
           (weights.shape().size() == 1));
  PD_CHECK(packed_info.shape().size() == 2 & packed_info.shape()[1] == 2);

  const uint32_t n_rays = packed_info.shape()[0];
  const uint32_t n_samples = weights.shape()[0];

  auto resample_starts =
      paddle::full({total_steps, 1}, 0, starts.dtype(), starts.place());
  auto resample_ends =
      paddle::full({total_steps, 1}, 0, ends.dtype(), ends.place());

  static constexpr uint32_t N_THREAD = 256;

  PD_DISPATCH_FLOATING_TYPES(
      weights.dtype(), "pdf_ray_marching_kernel", ([&] {
        pdf_ray_marching_kernel<data_t>
            <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, weights.stream()>>>(
                n_rays,
                // inputs
                packed_info.data<int>(), starts.data<data_t>(),
                ends.data<data_t>(), weights.data<data_t>(),
                resample_packed_info.data<int>(),
                // outputs
                resample_starts.data<data_t>(), resample_ends.data<data_t>());
      }));

  return {resample_starts, resample_ends};
}

std::vector<paddle::Tensor> grid_query(const paddle::Tensor& samples,
                                       const paddle::Tensor& aabb,
                                       const paddle::Tensor& grid_value,
                                       const int contraction_type_) {
  CHECK_CUDA(samples);

  ContractionType contraction_type =
      static_cast<ContractionType>(contraction_type_);

  uint32_t n_samples = samples.shape()[0];
  auto grid_value_dims = grid_value.shape();
  const int3 grid_res =
      make_int3(grid_value_dims[0], grid_value_dims[1], grid_value_dims[2]);

  static constexpr uint32_t N_THREAD = 256;

  auto occs =
      paddle::empty({n_samples}, grid_value.dtype(), paddle::GPUPlace());

  switch (occs.dtype()) {
    case paddle::DataType::FLOAT32:
      query_occ_kernel<float><<<div_round_up(n_samples, N_THREAD), N_THREAD, 0,
                                samples.stream()>>>(
          n_samples, samples.data<float>(), aabb.data<float>(), grid_res,
          grid_value.data<float>(), contraction_type, occs.data<float>());
      break;
    case paddle::DataType::FLOAT64:
      query_occ_kernel<double><<<div_round_up(n_samples, N_THREAD), N_THREAD, 0,
                                 samples.stream()>>>(
          n_samples, samples.data<float>(), aabb.data<float>(), grid_res,
          grid_value.data<double>(), contraction_type, occs.data<double>());
      break;
    case paddle::DataType::BOOL:
      query_occ_kernel<bool><<<div_round_up(n_samples, N_THREAD), N_THREAD, 0,
                               samples.stream()>>>(
          n_samples, samples.data<float>(), aabb.data<float>(), grid_res,
          grid_value.data<bool>(), contraction_type, occs.data<bool>());
      break;
  }

  return {occs};
}

std::vector<paddle::Tensor> contract(const paddle::Tensor& samples,
                                     const paddle::Tensor& aabb,
                                     const int contraction_type_) {
  CHECK_CUDA(samples);

  ContractionType contraction_type =
      static_cast<ContractionType>(contraction_type_);

  uint32_t n_samples = samples.shape()[0];
  auto out_samples =
      paddle::empty({n_samples, 3}, samples.dtype(), paddle::GPUPlace());
  static constexpr uint32_t N_THREAD = 256;
  contract_kernel<<<div_round_up(n_samples, N_THREAD), N_THREAD, 0,
                    samples.stream()>>>(n_samples, samples.data<float>(),
                                        aabb.data<float>(), contraction_type,
                                        out_samples.data<float>());

  return {out_samples};
}

std::vector<paddle::Tensor> contract_inv(const paddle::Tensor& samples,
                                         const paddle::Tensor& aabb,
                                         const int contraction_type_) {
  CHECK_CUDA(samples);

  ContractionType contraction_type =
      static_cast<ContractionType>(contraction_type_);

  uint32_t n_samples = samples.shape()[0];
  auto out_samples =
      paddle::empty({n_samples, 3}, samples.dtype(), paddle::GPUPlace());

  static constexpr uint32_t N_THREAD = 256;
  contract_inv_kernel<<<div_round_up(n_samples, N_THREAD), N_THREAD, 0,
                        samples.stream()>>>(
      n_samples, samples.data<float>(), aabb.data<float>(), contraction_type,
      out_samples.data<float>());

  return {out_samples};
}

std::vector<paddle::Tensor> transmittance_from_alpha_forward(
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas) {
  CHECK_CUDA(packed_info);
  CHECK_CUDA(alphas);

  PD_CHECK(packed_info.shape().size() == 2);
  PD_CHECK(alphas.shape().size() == 2 & alphas.shape()[1] == 1);

  uint32_t n_rays = packed_info.shape()[0];
  uint32_t n_samples = alphas.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  auto transmittance = paddle::empty_like(alphas);

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      transmittance_from_alpha_forward_kernel<int>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, alphas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int>(), alphas.data<float>(),
              // outputs
              transmittance.data<float>());
      break;
    case paddle::DataType::INT64:
      transmittance_from_alpha_forward_kernel<int64_t>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, alphas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int64_t>(), alphas.data<float>(),
              // outputs
              transmittance.data<float>());
      break;
    default:
      PD_THROW(
          "function transmittance_from_alpha_forward_kernel is not implemented "
          "for packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {transmittance};
}

std::vector<paddle::Tensor> transmittance_from_alpha_backward(
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas,
    const paddle::Tensor& transmittance,
    const paddle::Tensor& grad_transmittance) {
  CHECK_CUDA(packed_info);
  CHECK_CUDA(alphas);
  CHECK_CUDA(transmittance);
  CHECK_CUDA(grad_transmittance);

  PD_CHECK(packed_info.shape().size() == 2);
  PD_CHECK(transmittance.shape().size() == 2 & transmittance.shape()[1] == 1);
  PD_CHECK(grad_transmittance.shape().size() == 2 &
           grad_transmittance.shape()[1] == 1);

  uint32_t n_rays = packed_info.shape()[0];
  uint32_t n_samples = transmittance.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  auto grad_alphas = paddle::empty_like(alphas);

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      transmittance_from_alpha_backward_kernel<int>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, alphas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int>(), alphas.data<float>(),
              transmittance.data<float>(), grad_transmittance.data<float>(),
              // outputs
              grad_alphas.data<float>());
      break;
    case paddle::DataType::INT64:
      transmittance_from_alpha_backward_kernel<int64_t>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, alphas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int64_t>(), alphas.data<float>(),
              transmittance.data<float>(), grad_transmittance.data<float>(),
              // outputs
              grad_alphas.data<float>());
      break;
    default:
      PD_THROW(
          "function transmittance_from_alpha_backward_kernel is not "
          "implemented for packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {grad_alphas};
}

std::vector<paddle::Tensor> unpack_info(const paddle::Tensor& packed_info,
                                        const int n_samples) {
  CHECK_CUDA(packed_info);

  uint32_t n_rays = packed_info.shape()[0];

  PD_CHECK(packed_info.shape().size() == 2 & packed_info.shape()[1] == 2);

  static constexpr uint32_t N_THREAD = 256;

  auto ray_indices =
      paddle::empty({n_samples}, packed_info.dtype(), paddle::GPUPlace());

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      unpack_info_kernel<int><<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                                packed_info.stream()>>>(
          n_rays, packed_info.data<int>(), ray_indices.data<int>());
      break;
    case paddle::DataType::INT64:
      unpack_info_kernel<int64_t><<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                                    packed_info.stream()>>>(
          n_rays, packed_info.data<int64_t>(), ray_indices.data<int64_t>());
      break;
    default:
      PD_THROW(
          "function unpack_info_kernel is not implemented for packed_info's "
          "data type `",
          packed_info.dtype(), "`");
  }

  return {ray_indices};
}

std::vector<paddle::Tensor> weight_from_packed_sigma_forward(
    const paddle::Tensor& packed_info, const paddle::Tensor& starts,
    const paddle::Tensor& ends, const paddle::Tensor& sigmas) {
  CHECK_CUDA(packed_info);
  CHECK_CUDA(starts);
  CHECK_CUDA(ends);
  CHECK_CUDA(sigmas);

  PD_CHECK(packed_info.shape().size() == 2);
  PD_CHECK(starts.shape().size() == 2 & starts.shape()[1] == 1);
  PD_CHECK(ends.shape().size() == 2 & ends.shape()[1] == 1);
  PD_CHECK(sigmas.shape().size() == 2 & sigmas.shape()[1] == 1);

  uint32_t n_rays = packed_info.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  auto weights = paddle::empty_like(sigmas);

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      weight_from_sigma_forward_kernel<int>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, sigmas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int>(), starts.data<float>(), ends.data<float>(),
              sigmas.data<float>(),
              // outputs
              weights.data<float>());
      break;
    case paddle::DataType::INT64:
      weight_from_sigma_forward_kernel<int64_t>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, sigmas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int64_t>(), starts.data<float>(),
              ends.data<float>(), sigmas.data<float>(),
              // outputs
              weights.data<float>());
      break;
    default:
      PD_THROW(
          "function rendering_forward_kernel is not implemented for "
          "packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {weights};
}

std::vector<paddle::Tensor> weight_from_packed_sigma_backward(
    const paddle::Tensor& packed_info, const paddle::Tensor& starts,
    const paddle::Tensor& ends, const paddle::Tensor& sigmas,
    const paddle::Tensor& weights, const paddle::Tensor& grad_weights) {
  CHECK_CUDA(packed_info);
  CHECK_CUDA(starts);
  CHECK_CUDA(ends);
  CHECK_CUDA(sigmas);
  CHECK_CUDA(weights);
  CHECK_CUDA(grad_weights);

  PD_CHECK(packed_info.shape().size() == 2);
  PD_CHECK(starts.shape().size() == 2 & starts.shape()[1] == 1);
  PD_CHECK(ends.shape().size() == 2 & ends.shape()[1] == 1);
  PD_CHECK(sigmas.shape().size() == 2 & sigmas.shape()[1] == 1);
  PD_CHECK(weights.shape().size() == 2 & weights.shape()[1] == 1);
  PD_CHECK(grad_weights.shape().size() == 2 & grad_weights.shape()[1] == 1);

  uint32_t n_rays = packed_info.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  auto grad_sigmas = paddle::empty_like(sigmas);

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      weight_from_sigma_backward_kernel<int>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, sigmas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int>(), starts.data<float>(), ends.data<float>(),
              sigmas.data<float>(), weights.data<float>(),
              grad_weights.data<float>(),
              // outputs
              grad_sigmas.data<float>());
      break;
    case paddle::DataType::INT64:
      weight_from_sigma_backward_kernel<int64_t>
          <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, sigmas.stream()>>>(
              n_rays,
              // inputs
              packed_info.data<int64_t>(), starts.data<float>(),
              ends.data<float>(), sigmas.data<float>(), weights.data<float>(),
              grad_weights.data<float>(),
              // outputs
              grad_sigmas.data<float>());
      break;
    default:
      PD_THROW(
          "function rendering_forward_kernel is not implemented for "
          "packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {grad_sigmas};
}

std::vector<paddle::Tensor> weight_from_sigma_forward(
    const paddle::Tensor& starts, const paddle::Tensor& ends,
    const paddle::Tensor& sigmas) {
  CHECK_CUDA(starts);
  CHECK_CUDA(ends);
  CHECK_CUDA(sigmas);

  PD_CHECK(starts.shape().size() == 3 & starts.shape()[2] == 1);
  PD_CHECK(ends.shape().size() == 3 & ends.shape()[2] == 1);
  PD_CHECK(sigmas.shape().size() == 3 & sigmas.shape()[2] == 1);

  uint32_t n_rays = sigmas.shape()[0];
  uint32_t n_samples = sigmas.shape()[1];

  static constexpr uint32_t N_THREAD = 256;

  auto weights = paddle::empty_like(sigmas);

  weight_from_sigma_forward_kernel<uint32_t>
      <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, sigmas.stream()>>>(
          n_rays,
          // inputs
          nullptr, starts.data<float>(), ends.data<float>(),
          sigmas.data<float>(),
          // outputs
          weights.data<float>(), n_samples);

  return {weights};
}

std::vector<paddle::Tensor> weight_from_sigma_backward(
    const paddle::Tensor& starts, const paddle::Tensor& ends,
    const paddle::Tensor& sigmas, const paddle::Tensor& weights,
    const paddle::Tensor& grad_weights) {
  CHECK_CUDA(starts);
  CHECK_CUDA(ends);
  CHECK_CUDA(sigmas);
  CHECK_CUDA(weights);
  CHECK_CUDA(grad_weights);

  PD_CHECK(starts.shape().size() == 3 & starts.shape()[2] == 1);
  PD_CHECK(ends.shape().size() == 3 & ends.shape()[2] == 1);
  PD_CHECK(sigmas.shape().size() == 3 & sigmas.shape()[2] == 1);
  PD_CHECK(weights.shape().size() == 3 & weights.shape()[2] == 1);
  PD_CHECK(grad_weights.shape().size() == 3 & grad_weights.shape()[2] == 1);

  uint32_t n_rays = sigmas.shape()[0];
  uint32_t n_samples = sigmas.shape()[1];

  static constexpr uint32_t N_THREAD = 256;

  auto grad_sigmas = paddle::empty_like(sigmas);

  weight_from_sigma_backward_kernel<uint32_t>
      <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0, sigmas.stream()>>>(
          n_rays,
          // inputs
          nullptr, starts.data<float>(), ends.data<float>(),
          sigmas.data<float>(), weights.data<float>(),
          grad_weights.data<float>(),
          // outputs
          grad_sigmas.data<float>(), n_samples);

  return {grad_sigmas};
}