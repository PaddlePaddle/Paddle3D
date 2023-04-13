#include <vector>

#include "../include/contraction.h"
#include "paddle/extension.h"

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> rendering_alphas_forward_compressed(
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas,
    float early_stop_eps, float alpha_thresh);

std::vector<paddle::Tensor> rendering_alphas_forward(
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas,
    float early_stop_eps, float alpha_thresh);

std::vector<paddle::Tensor> rendering_alphas_backward(
    const paddle::Tensor& weights, const paddle::Tensor& grad_weights,
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas,
    float early_stop_eps, float alpha_thresh);
#endif

// rendering_alphas_compressed (no grad)
std::vector<std::vector<int64_t>> RenderingAlphasCompressedInferShape(
    std::vector<int64_t> packed_info_shape, std::vector<int64_t> alphas_shape) {
  return {{packed_info_shape[0]}, {alphas_shape[0]}};
}

std::vector<paddle::DataType> RenderingAlphasCompressedInferDtype(
    paddle::DataType packed_info_dtype, paddle::DataType alphas_dtype) {
  return {packed_info_dtype, paddle::DataType::BOOL};
}

PD_BUILD_OP(rendering_alphas_compressed)
    .Inputs({"packed_info", "alphas"})
    .Outputs({"num_steps", "compact_selector"})
    .Attrs({"early_stop_eps: float", "alpha_thresh: float"})
    .SetKernelFn(PD_KERNEL(rendering_alphas_forward_compressed))
    .SetInferShapeFn(PD_INFER_SHAPE(RenderingAlphasCompressedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RenderingAlphasCompressedInferDtype));

// rendering_alphas
std::vector<std::vector<int64_t>> RenderingAlphasInferShape(
    std::vector<int64_t> packed_info_shape, std::vector<int64_t> alphas_shape) {
  return {{alphas_shape[0]}};
}

std::vector<paddle::DataType> RenderingAlphasInferDtype(
    paddle::DataType packed_info_dtype, paddle::DataType alphas_dtype) {
  return {alphas_dtype};
}

PD_BUILD_OP(rendering_alphas)
    .Inputs({"packed_info", "alphas"})
    .Outputs({"weights"})
    .Attrs({"early_stop_eps: float", "alpha_thresh: float"})
    .SetKernelFn(PD_KERNEL(rendering_alphas_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(RenderingAlphasInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RenderingAlphasInferDtype));

PD_BUILD_GRAD_OP(rendering_alphas)
    .Inputs({"weights", paddle::Grad("weights"), "packed_info", "alphas"})
    .Outputs({paddle::Grad("alphas")})
    .Attrs({"early_stop_eps: float", "alpha_thresh: float"})
    .SetKernelFn(PD_KERNEL(rendering_alphas_backward));
