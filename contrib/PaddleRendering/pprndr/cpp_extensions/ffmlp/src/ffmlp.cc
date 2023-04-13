#pragma once

#include <stdint.h>

#include <cstddef>
#include <vector>

#include "paddle/extension.h"

#ifdef PADDLE_WITH_CUDA
// activation: should have been enum, here we just use int.
std::vector<paddle::Tensor> ffmlp_forward(const paddle::Tensor& inputs,
                                          const paddle::Tensor& weights,
                                          const int64_t output_dim,
                                          const int64_t hidden_dim,
                                          const int64_t num_layers,
                                          const int activation_,
                                          const int output_activation_);

std::vector<paddle::Tensor> ffmlp_inference(const paddle::Tensor& inputs,
                                            const paddle::Tensor& weights,
                                            const int64_t output_dim,
                                            const int64_t hidden_dim,
                                            const int64_t num_layers,
                                            const int activation_,
                                            const int output_activation_);

std::vector<paddle::Tensor> ffmlp_backward(
    const paddle::Tensor& inputs, const paddle::Tensor& weights,
    const paddle::Tensor& forward_buffer, const paddle::Tensor& grad,
    const int64_t output_dim, const int64_t hidden_dim,
    const int64_t num_layers, const int activation_,
    const int output_activation_);

void allocate_splitk(size_t size);

void free_splitk();

#endif

std::vector<std::vector<int64_t>> FFMLPInferShape(
    std::vector<int64_t> inputs_shape, std::vector<int64_t> weights_shape,
    const int64_t output_dim, const int64_t hidden_dim,
    const int64_t num_layers, const int activation_,
    const int output_activation_) {
  const int64_t B = inputs_shape[0];
  return {{B, output_dim}, {B, hidden_dim}};
}

std::vector<paddle::DataType> FFMLPInferDtype(paddle::DataType inputs_dtype,
                                              paddle::DataType weights_dtype) {
  return {paddle::DataType::FLOAT16, paddle::DataType::FLOAT16};
}

std::vector<std::vector<int64_t>> FFMLPInferenceInferShape(
    std::vector<int64_t> inputs_shape, std::vector<int64_t> weights_shape,
    const int64_t output_dim, const int64_t hidden_dim,
    const int64_t num_layers, const int activation_,
    const int output_activation_) {
  const int64_t B = inputs_shape[0];
  return {{B, output_dim}};
}

std::vector<paddle::DataType> FFMLPInferenceInferDtype(
    paddle::DataType inputs_dtype, paddle::DataType weights_dtype) {
  return {paddle::DataType::FLOAT16};
}

PD_BUILD_OP(ffmlp_op)
    .Inputs({"inputs", "weights"})
    .Outputs({"outputs", "forward_buffer"})
    .Attrs({"output_dim: int64_t", "hidden_dim: int64_t", "num_layers: int64_t",
            "activation_: int", "output_activation_: int"})
    .SetKernelFn(PD_KERNEL(ffmlp_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(FFMLPInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FFMLPInferDtype));

PD_BUILD_GRAD_OP(ffmlp_op)
    .Inputs({"inputs", "weights", "forward_buffer", paddle::Grad("outputs")})
    .Outputs({paddle::Grad("inputs"), paddle::Grad("weights")})
    .Attrs({"output_dim: int64_t", "hidden_dim: int64_t", "num_layers: int64_t",
            "activation_: int", "output_activation_: int"})
    .SetKernelFn(PD_KERNEL(ffmlp_backward));

PD_BUILD_OP(ffmlp_infer_op)
    .Inputs({"inputs", "weights"})
    .Outputs({"outputs"})
    .Attrs({"output_dim: int64_t", "hidden_dim: int64_t", "num_layers: int64_t",
            "activation_: int", "output_activation_: int"})
    .SetKernelFn(PD_KERNEL(ffmlp_inference))
    .SetInferShapeFn(PD_INFER_SHAPE(FFMLPInferenceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FFMLPInferDtype));

PYBIND11_MODULE(ffmlp, m) {
  m.def("allocate_splitk", &allocate_splitk, "allocate k cuda streams");
  m.def("free_splitk", &free_splitk, "free cuda streams");
}
