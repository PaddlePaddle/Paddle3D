// kernels in this file are based on
// https://github.com/KAIR-BAIR/nerfacc/blob/master/nerfacc/cuda/csrc/render_transmittance.cu

#include <vector>

#include "helper_math.h"
#include "paddle/extension.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor")

template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <typename data_t, typename int_data_t>
__global__ void rendering_forward_kernel(
    const uint32_t n_rays,
    const int_data_t *packed_info,  // input ray & point indices.
    const data_t *starts,           // input start t
    const data_t *ends,             // input end t
    const data_t *sigmas,           // input density after activation
    const data_t *alphas,           // input alpha (opacity) values.
    const float early_stop_eps,     // transmittance threshold for early stop
    const float alpha_thresh,       // alpha threshold for emtpy space
    // outputs: should be all-zero initialized
    int_data_t *num_steps,  // the number of valid steps for each ray
    data_t *weights,        // the number rendering weights for each sample
    bool
        *compact_selector  // the samples that we needs to compute the gradients
) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  const int_data_t base = packed_info[i * 2 + 0];   // point idx start.
  const int_data_t steps = packed_info[i * 2 + 1];  // point idx shift.
  if (steps == 0) return;

  if (alphas != nullptr) {
    // rendering with alpha
    alphas += base;
  } else {
    // rendering with density
    starts += base;
    ends += base;
    sigmas += base;
  }

  if (num_steps != nullptr) {
    num_steps += i;
  }
  if (weights != nullptr) {
    weights += base;
  }
  if (compact_selector != nullptr) {
    compact_selector += base;
  }

  // accumulated rendering
  data_t T = 1.f;
  int_data_t cnt = 0;
  for (int64_t j = 0; j < steps; ++j) {
    if (T < early_stop_eps) {
      break;
    }
    data_t alpha;
    if (alphas != nullptr) {
      // rendering with alpha
      alpha = alphas[j];
    } else {
      // rendering with density
      data_t delta = ends[j] - starts[j];
      alpha = 1.f - __expf(-sigmas[j] * delta);
    }
    if (alpha < alpha_thresh) {
      // empty space
      continue;
    }
    const data_t weight = alpha * T;
    T *= (1.f - alpha);
    if (weights != nullptr) {
      weights[j] = weight;
    }
    if (compact_selector != nullptr) {
      compact_selector[j] = true;
    }
    cnt += 1;
  }
  if (num_steps != nullptr) {
    *num_steps = cnt;
  }
  return;
}

template <typename data_t, typename int_data_t>
__global__ void rendering_backward_kernel(
    const uint32_t n_rays,
    const int_data_t *packed_info,  // input ray & point indices.
    const data_t *starts,           // input start t
    const data_t *ends,             // input end t
    const data_t *sigmas,           // input density after activation
    const data_t *alphas,           // input alpha (opacity) values.
    const float early_stop_eps,     // transmittance threshold for early stop
    const float alpha_thresh,       // alpha threshold for emtpy space
    const data_t *weights,          // forward output
    const data_t *grad_weights,     // input gradients
    // if alphas was given, we compute the gradients for alphas.
    // otherwise, we compute the gradients for sigmas.
    data_t *grad_sigmas,  // output gradients
    data_t *grad_alphas   // output gradients
) {
  const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n_rays) return;

  // locate
  const int_data_t base = packed_info[i * 2 + 0];   // point idx start.
  const int_data_t steps = packed_info[i * 2 + 1];  // point idx shift.
  if (steps == 0) return;

  if (alphas != nullptr) {
    // rendering with alpha
    alphas += base;
    grad_alphas += base;
  } else {
    // rendering with density
    starts += base;
    ends += base;
    sigmas += base;
    grad_sigmas += base;
  }

  weights += base;
  grad_weights += base;

  data_t accum = 0;
  for (int64_t j = 0; j < steps; ++j) {
    accum += grad_weights[j] * weights[j];
  }

  // backward of accumulated rendering
  data_t T = 1.f;
  for (int64_t j = 0; j < steps; ++j) {
    if (T < early_stop_eps) {
      break;
    }
    data_t alpha;
    if (alphas != nullptr) {
      // rendering with alpha
      alpha = alphas[j];
      if (alpha < alpha_thresh) {
        // empty space
        continue;
      }
      grad_alphas[j] =
          (grad_weights[j] * T - accum) / fmaxf(1.f - alpha, 1e-10f);
    } else {
      // rendering with density
      data_t delta = ends[j] - starts[j];
      alpha = 1.f - __expf(-sigmas[j] * delta);
      if (alpha < alpha_thresh) {
        // empty space
        continue;
      }
      grad_sigmas[j] = (grad_weights[j] * T - accum) * delta;
    }

    accum -= grad_weights[j] * weights[j];
    T *= (1.f - alpha);
  }
}

// -- rendering with alphas -- //

std::vector<paddle::Tensor> rendering_alphas_forward_compressed(
    const paddle::Tensor &packed_info, const paddle::Tensor &alphas,
    float early_stop_eps, float alpha_thresh) {
  CHECK_CUDA(packed_info);
  CHECK_CUDA(alphas);

  PD_CHECK(packed_info.shape().size() == 2 & packed_info.shape()[1] == 2);
  PD_CHECK(alphas.shape().size() == 2 & alphas.shape()[1] == 1);

  const uint32_t n_rays = packed_info.shape()[0];
  const uint32_t n_samples = alphas.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  // compress the samples to get rid of invisible ones.
  auto num_steps =
      paddle::full({n_rays}, 0, packed_info.dtype(), packed_info.place());
  auto compact_selector =
      paddle::full({n_samples}, 0, paddle::DataType::BOOL, alphas.place());

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      PD_DISPATCH_FLOATING_TYPES(
          alphas.dtype(), "rendering_forward_kernel", ([&] {
            rendering_forward_kernel<data_t, int>
                <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                   alphas.stream()>>>(n_rays,
                                      // inputs
                                      packed_info.data<int>(), nullptr, nullptr,
                                      nullptr, alphas.data<data_t>(),
                                      early_stop_eps, alpha_thresh,
                                      // outputs
                                      num_steps.data<int>(), nullptr,
                                      compact_selector.data<bool>());
          }));
      break;
    case paddle::DataType::INT64:
      PD_DISPATCH_FLOATING_TYPES(
          alphas.dtype(), "rendering_forward_kernel", ([&] {
            rendering_forward_kernel<data_t, int64_t>
                <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                   alphas.stream()>>>(n_rays,
                                      // inputs
                                      packed_info.data<int64_t>(), nullptr,
                                      nullptr, nullptr, alphas.data<data_t>(),
                                      early_stop_eps, alpha_thresh,
                                      // outputs
                                      num_steps.data<int64_t>(), nullptr,
                                      compact_selector.data<bool>());
          }));
      break;
    default:
      PD_THROW(
          "function rendering_forward_kernel is not implemented for "
          "packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {num_steps, compact_selector};
}

std::vector<paddle::Tensor> rendering_alphas_forward(
    const paddle::Tensor &packed_info, const paddle::Tensor &alphas,
    float early_stop_eps, float alpha_thresh) {
  CHECK_CUDA(packed_info);
  CHECK_CUDA(alphas);

  PD_CHECK(packed_info.shape().size() == 2 & packed_info.shape()[1] == 2);
  PD_CHECK(alphas.shape().size() == 2 & alphas.shape()[1] == 1);

  const uint32_t n_rays = packed_info.shape()[0];
  const uint32_t n_samples = alphas.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  // compress the samples to get rid of invisible ones.
  auto weights = paddle::full({n_samples}, 0, alphas.dtype(), alphas.place());

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      PD_DISPATCH_FLOATING_TYPES(
          alphas.dtype(), "rendering_forward_kernel", ([&] {
            rendering_forward_kernel<data_t, int>
                <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                   alphas.stream()>>>(n_rays,
                                      // inputs
                                      packed_info.data<int>(), nullptr, nullptr,
                                      nullptr, alphas.data<data_t>(),
                                      early_stop_eps, alpha_thresh,
                                      // outputs
                                      nullptr, weights.data<data_t>(), nullptr);
          }));
      break;
    case paddle::DataType::INT64:
      PD_DISPATCH_FLOATING_TYPES(
          alphas.dtype(), "rendering_forward_kernel", ([&] {
            rendering_forward_kernel<data_t, int64_t>
                <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                   alphas.stream()>>>(n_rays,
                                      // inputs
                                      packed_info.data<int64_t>(), nullptr,
                                      nullptr, nullptr, alphas.data<data_t>(),
                                      early_stop_eps, alpha_thresh,
                                      // outputs
                                      nullptr, weights.data<data_t>(), nullptr);
          }));
      break;
    default:
      PD_THROW(
          "function rendering_forward_kernel is not implemented for "
          "packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {weights};
}

std::vector<paddle::Tensor> rendering_alphas_backward(
    const paddle::Tensor &weights, const paddle::Tensor &grad_weights,
    const paddle::Tensor &packed_info, const paddle::Tensor &alphas,
    float early_stop_eps, float alpha_thresh) {
  const uint32_t n_rays = packed_info.shape()[0];
  const uint32_t n_samples = alphas.shape()[0];

  static constexpr uint32_t N_THREAD = 256;

  // outputs
  auto grad_alphas = paddle::experimental::full_like(alphas, 0);

  switch (packed_info.dtype()) {
    case paddle::DataType::INT32:
      PD_DISPATCH_FLOATING_TYPES(
          alphas.dtype(), "rendering_backward_kernel", ([&] {
            rendering_backward_kernel<data_t, int>
                <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                   alphas.stream()>>>(
                    n_rays,
                    // inputs
                    packed_info.data<int>(), nullptr, nullptr, nullptr,
                    alphas.data<data_t>(), early_stop_eps, alpha_thresh,
                    weights.data<data_t>(), grad_weights.data<data_t>(),
                    // outputs
                    nullptr, grad_alphas.data<data_t>());
          }));
      break;
    case paddle::DataType::INT64:
      PD_DISPATCH_FLOATING_TYPES(
          alphas.dtype(), "rendering_backward_kernel", ([&] {
            rendering_backward_kernel<data_t, int64_t>
                <<<div_round_up(n_rays, N_THREAD), N_THREAD, 0,
                   alphas.stream()>>>(
                    n_rays,
                    // inputs
                    packed_info.data<int64_t>(), nullptr, nullptr, nullptr,
                    alphas.data<data_t>(), early_stop_eps, alpha_thresh,
                    weights.data<data_t>(), grad_weights.data<data_t>(),
                    // outputs
                    nullptr, grad_alphas.data<data_t>());
          }));
      break;
    default:
      PD_THROW(
          "function rendering_backward_kernel is not implemented for "
          "packed_info's data type `",
          packed_info.dtype(), "`");
  }

  return {grad_alphas};
}
