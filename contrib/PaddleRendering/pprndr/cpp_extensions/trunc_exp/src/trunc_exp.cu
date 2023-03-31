#include <cmath>
#include <vector>

#include "paddle/extension.h"

#define CHECK_CUDA_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> trunc_exp_cuda_forward(const paddle::Tensor& x) {
  CHECK_CUDA_INPUT(x);

  return {paddle::exp(x)};
}

std::vector<paddle::Tensor> trunc_exp_cuda_backward(
    const paddle::Tensor& x, const paddle::Tensor& grad_out) {
  CHECK_CUDA_INPUT(x);
  CHECK_CUDA_INPUT(grad_out);

  return {
      paddle::multiply(paddle::exp(paddle::clip(x, -15.0f, 15.0f)), grad_out)};
}
