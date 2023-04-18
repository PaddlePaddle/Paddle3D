#include <vector>

#include "paddle/extension.h"

#ifdef PADDLE_WITH_CUDA
// inputs: [batch_size, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [batch_size, L * C], float
// H: base resolution
std::vector<paddle::Tensor> grid_encode_forward(
    const paddle::Tensor& inputs, const paddle::Tensor& embeddings,
    const paddle::Tensor& offsets, const int64_t input_dim,
    const int64_t level_dim, const int64_t num_levels, const float scale,
    const int64_t base_resolution, const int64_t gridtype,
    const bool align_corners, const bool inputs_stop_gradient);
std::vector<paddle::Tensor> grid_encode_backward(
    const paddle::Tensor& grad, const paddle::Tensor& inputs,
    const paddle::Tensor& embeddings, const paddle::Tensor& offsets,
    const paddle::Tensor& dy_dx, const int64_t input_dim,
    const int64_t level_dim, const int64_t num_levels, const float scale,
    const int64_t base_resolution, const int64_t gridtype,
    const bool align_corners, const bool inputs_stop_gradient);
#endif

std::vector<std::vector<int64_t>> GridEncoderInferShape(
    std::vector<int64_t> inputs_shape, std::vector<int64_t> embeddings_shape,
    std::vector<int64_t> offsets_shape, const int64_t input_dim,
    const int64_t level_dim, const int64_t num_levels, const float scale,
    const int64_t base_resolution, const int64_t gridtype,
    const bool align_corners, const bool inputs_stop_gradient) {
  int64_t B = inputs_shape[0];
  return {{num_levels, B, level_dim}, {B, num_levels * input_dim * level_dim}};
}

std::vector<paddle::DataType> GridEncoderInferDtype(
    paddle::DataType inputs_dtype, paddle::DataType embeddings_dtype,
    paddle::DataType offsets_dtype) {
  return {embeddings_dtype, embeddings_dtype};
}

PD_BUILD_OP(grid_encode)
    .Inputs({"inputs", "embeddings", "offsets"})
    .Outputs({"outputs", "dy_dx"})
    .Attrs({"input_dim: int64_t", "level_dim: int64_t", "num_levels: int64_t",
            "log2_per_level_scale: float", "base_resolution: int64_t",
            "gridtype: int64_t", "align_corners: bool",
            "inputs_stop_gradient: bool"})
    .SetKernelFn(PD_KERNEL(grid_encode_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(GridEncoderInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GridEncoderInferDtype));

PD_BUILD_GRAD_OP(grid_encode)
    .Inputs({paddle::Grad("outputs"), "inputs", "embeddings", "offsets",
             "dy_dx"})
    .Outputs({paddle::Grad("embeddings"), paddle::Grad("inputs")})
    .Attrs({"input_dim: int64_t", "level_dim: int64_t", "num_levels: int64_t",
            "log2_per_level_scale: float", "base_resolution: int64_t",
            "gridtype: int64_t", "align_corners: bool",
            "inputs_stop_gradient: bool"})
    .SetKernelFn(PD_KERNEL(grid_encode_backward));