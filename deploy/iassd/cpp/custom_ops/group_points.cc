#include <vector>

#include "paddle/include/experimental/ext_all.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

// cuda launcher declaration
void group_points_cuda_launcher(const int b, const int c, const int n,
                                const int npoints, const int nsample,
                                const float *points, const int *idx,
                                float *out);
void group_points_grad_cuda_launcher(const int b, const int c, const int n,
                                     const int npoints, const int nsample,
                                     const float *grad_out, const int *idx,
                                     float *grad_points);

// op forward wrapper
std::vector<paddle::Tensor> group_points_cuda_forward(
    const paddle::Tensor &points_tensor, const paddle::Tensor &idx_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT(idx_tensor);
  const int b = points_tensor.shape()[0];
  const int c = points_tensor.shape()[1];
  const int n = points_tensor.shape()[2];
  const int npoints = idx_tensor.shape()[1];
  const int nsample = idx_tensor.shape()[2];

  auto *points = points_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();
  auto out_tensor = paddle::empty(
      {b, c, npoints, nsample}, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto *out = out_tensor.data<float>();

  group_points_cuda_launcher(b, c, n, npoints, nsample, points, idx, out);

  return {out_tensor};
}

// op backward wrapper
std::vector<paddle::Tensor> group_points_cuda_backward(
    const paddle::Tensor &grad_out_tensor, const paddle::Tensor &idx_tensor,
    const paddle::Tensor &points_tensor) {
  CHECK_INPUT(grad_out_tensor);
  CHECK_INPUT(idx_tensor);
  const int b = grad_out_tensor.shape()[0];
  const int c = grad_out_tensor.shape()[1];
  const int npoints = grad_out_tensor.shape()[2];
  const int nsample = grad_out_tensor.shape()[3];
  const int n = points_tensor.shape()[2];

  auto *grad_out = grad_out_tensor.data<float>();
  auto *idx = idx_tensor.data<int>();
  auto grad_points_tensor = paddle::full(
      {b, c, n}, 0.0, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto *grad_points = grad_points_tensor.data<float>();

  group_points_grad_cuda_launcher(b, c, n, npoints, nsample, grad_out, idx,
                                  grad_points);

  return {grad_points_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> GroupInferShape(
    std::vector<int64_t> points_shape, std::vector<int64_t> idx_shape) {
  const int b = points_shape[0];
  const int c = points_shape[1];
  const int npoints = idx_shape[1];
  const int nsample = idx_shape[2];
  return {{b, c, npoints, nsample}};
}

// data type infer
std::vector<paddle::DataType> GroupInferDtype(paddle::DataType points_dtype,
                                              paddle::DataType idx_dtype) {
  return {points_dtype};
}

// build forward op
PD_BUILD_OP(group_operation)
    .Inputs({"points_tensor", "idx_tensor"})
    .Outputs({"out_tensor"})
    .SetKernelFn(PD_KERNEL(group_points_cuda_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(GroupInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GroupInferDtype));

// build backward op
PD_BUILD_GRAD_OP(group_operation)
    .Inputs({paddle::Grad("out_tensor"), "idx_tensor", "points_tensor"})
    .Outputs({paddle::Grad("points_tensor")})
    .SetKernelFn(PD_KERNEL(group_points_cuda_backward));