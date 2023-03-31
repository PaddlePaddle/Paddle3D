#include <cmath>
#include <vector>

#include "paddle/extension.h"

#define CHECK_CPU_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

#define CHECK_IS_FLOATING(x)                           \
  PD_CHECK(x.dtype() == paddle::DataType::FLOAT32 ||   \
               x.dtype() == paddle::DataType::FLOAT64, \
           #x " must be a floating tensor")

std::vector<paddle::Tensor> trunc_exp_cpu_forward(const paddle::Tensor& x) {
  CHECK_CPU_INPUT(x);

  return {paddle::exp(x)};
}

std::vector<paddle::Tensor> trunc_exp_cpu_backward(
    const paddle::Tensor& x, const paddle::Tensor& grad_out) {
  CHECK_CPU_INPUT(x);
  CHECK_CPU_INPUT(grad_out);

  return {
      paddle::multiply(paddle::exp(paddle::clip(x, -15.0f, 15.0f)), grad_out)};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> trunc_exp_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> trunc_exp_cuda_backward(
    const paddle::Tensor& x, const paddle::Tensor& grad_out);
#endif

std::vector<paddle::Tensor> TruncExpForward(const paddle::Tensor& x) {
  CHECK_IS_FLOATING(x);
  if (x.is_cpu()) {
    return trunc_exp_cpu_forward(x);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return trunc_exp_cuda_forward(x);
#endif
  } else {
    PD_THROW(
        "Unsupported device type for forward function of trunc_exp "
        "operator.");
  }
}

std::vector<paddle::Tensor> TruncExpBackward(const paddle::Tensor& x,
                                             const paddle::Tensor& grad_out) {
  CHECK_IS_FLOATING(x);
  CHECK_IS_FLOATING(grad_out);
  if (x.is_cpu()) {
    return trunc_exp_cpu_backward(x, grad_out);
#ifdef PADDLE_WITH_CUDA
  } else if (x.is_gpu()) {
    return trunc_exp_cuda_backward(x, grad_out);
#endif
  } else {
    PD_THROW(
        "Unsupported device type for backward function of trunc_exp "
        "operator.");
  }
}

std::vector<std::vector<int64_t>> TruncExpInferShape(
    std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> TruncExpInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> TruncExpBackwardInferShape(
    std::vector<int64_t> x_shape, std::vector<int64_t> grad_out_shape) {
  return {x_shape};
}

PD_BUILD_OP(trunc_exp)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(TruncExpForward))
    .SetInferShapeFn(PD_INFER_SHAPE(TruncExpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TruncExpInferDtype));

PD_BUILD_GRAD_OP(trunc_exp)
    .Inputs({"X", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(TruncExpBackward));
