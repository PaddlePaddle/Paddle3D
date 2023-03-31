#include <vector>

#include "paddle/extension.h"

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> sh_encode_forward(const paddle::Tensor &inputs,
                                              const int64_t degree,
                                              const bool stop_gradient);

std::vector<paddle::Tensor> sh_encode_backward(const paddle::Tensor &grad,
                                               const paddle::Tensor &inputs,
                                               const paddle::Tensor &dy_dx,
                                               const int64_t degree,
                                               const bool stop_gradient);
#endif

std::vector<std::vector<int64_t>> SHEncoderInferShape(
    std::vector<int64_t> inputs_shape, const int64_t degree,
    const bool stop_gradient) {
  int64_t B = inputs_shape[0];
  int64_t input_dim = inputs_shape[1];
  int64_t output_dim = degree * degree;
  return {{B, output_dim}, {B, input_dim, output_dim}};
}

std::vector<paddle::DataType> SHEncoderInferDtype(
    paddle::DataType inputs_dtype) {
  return {inputs_dtype, inputs_dtype};
}

PD_BUILD_OP(sh_encode)
    .Inputs({"inputs"})
    .Outputs({"outputs", "dy_dx"})
    .Attrs({"degree: int64_t", "stop_gradient: bool"})
    .SetKernelFn(PD_KERNEL(sh_encode_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(SHEncoderInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SHEncoderInferDtype));

PD_BUILD_GRAD_OP(sh_encode)
    .Inputs({paddle::Grad("outputs"), "inputs", "dy_dx"})
    .Outputs({paddle::Grad("inputs")})
    .Attrs({"degree: int64_t", "stop_gradient: bool"})
    .SetKernelFn(PD_KERNEL(sh_encode_backward));
