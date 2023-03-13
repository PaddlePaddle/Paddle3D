#include <vector>

#include "paddle/include/experimental/ext_all.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

// cuda launcher declaration
void ball_query_cuda_launcher(const int b, const int n, const int m,
                              const float radius, const int nsample,
                              const float *new_xyz, const float *xyz, int *idx);

// op forward wrapper
std::vector<paddle::Tensor> ball_query_cuda_forward(
    const paddle::Tensor &new_xyz_tensor, const paddle::Tensor &xyz_tensor,
    const float &radius, const int &nsample) {
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor);
  const int b = new_xyz_tensor.shape()[0];
  const int m = new_xyz_tensor.shape()[1];
  const int n = xyz_tensor.shape()[1];
  auto *new_xyz = new_xyz_tensor.data<float>();
  auto *xyz = xyz_tensor.data<float>();
  auto idx_tensor = paddle::empty({b, m, nsample}, paddle::DataType::INT32,
                                  paddle::GPUPlace());
  auto *idx = idx_tensor.data<int>();

  ball_query_cuda_launcher(b, n, m, radius, nsample, new_xyz, xyz, idx);

  return {idx_tensor};
}

// shape infer
std::vector<std::vector<int64_t>> BallQueryInferShape(
    std::vector<int64_t> new_xyz_shape, std::vector<int64_t> xyz_shape,
    const float &radius, const int &nsample) {
  return {{new_xyz_shape[0], new_xyz_shape[1], nsample}};
}

// data type infer
std::vector<paddle::DataType> BallQueryInferDtype(paddle::DataType t1,
                                                  paddle::DataType t2) {
  return {paddle::DataType::INT32};
}

// build forward op
PD_BUILD_OP(ball_query)
    .Inputs({"new_xyz_tensor", "xyz_tensor"})
    .Outputs({"idx"})
    .Attrs({"radius: float", "nsample: int"})
    .SetKernelFn(PD_KERNEL(ball_query_cuda_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(BallQueryInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BallQueryInferDtype));
