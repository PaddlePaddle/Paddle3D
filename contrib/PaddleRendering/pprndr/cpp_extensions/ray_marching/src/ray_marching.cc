#include <vector>

#include "../include/contraction.h"
#include "paddle/extension.h"

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> near_far_from_aabb(const paddle::Tensor& rays_o,
                                               const paddle::Tensor& rays_d,
                                               const paddle::Tensor& aabb);

std::vector<paddle::Tensor> first_round_ray_marching(
    const paddle::Tensor& rays_o, const paddle::Tensor& rays_d,
    const paddle::Tensor& nears, const paddle::Tensor& fars,
    const paddle::Tensor& roi, const paddle::Tensor& grid_binary,
    const int contraction_type_, const float step_size, const float cone_angle);

std::vector<paddle::Tensor> second_round_ray_marching(
    const paddle::Tensor& rays_o, const paddle::Tensor& rays_d,
    const paddle::Tensor& nears, const paddle::Tensor& fars,
    const paddle::Tensor& aabb, const paddle::Tensor& grid_binary,
    const paddle::Tensor& packed_info, const int contraction_type_,
    const float step_size, const float cone_angle, const int64_t total_steps);

std::vector<paddle::Tensor> pdf_ray_marching(
    const paddle::Tensor& starts, const paddle::Tensor& ends,
    const paddle::Tensor& weights, const paddle::Tensor& packed_info,
    const paddle::Tensor& resample_packed_info, const int64_t total_steps);

std::vector<paddle::Tensor> grid_query(const paddle::Tensor& samples,
                                       const paddle::Tensor& aabb,
                                       const paddle::Tensor& grid_value,
                                       const int contraction_type_);

std::vector<paddle::Tensor> contract(const paddle::Tensor& samples,
                                     const paddle::Tensor& aabb,
                                     const int contraction_type_);

std::vector<paddle::Tensor> contract_inv(const paddle::Tensor& samples,
                                         const paddle::Tensor& aabb,
                                         const int contraction_type_);

std::vector<paddle::Tensor> transmittance_from_alpha_forward(
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas);

std::vector<paddle::Tensor> transmittance_from_alpha_backward(
    const paddle::Tensor& packed_info, const paddle::Tensor& alphas,
    const paddle::Tensor& transmittance,
    const paddle::Tensor& grad_transmittance);

std::vector<paddle::Tensor> unpack_info(const paddle::Tensor& packed_info,
                                        const int n_samples);

std::vector<paddle::Tensor> weight_from_packed_sigma_forward(
    const paddle::Tensor& packed_info, const paddle::Tensor& starts,
    const paddle::Tensor& ends, const paddle::Tensor& sigmas);

std::vector<paddle::Tensor> weight_from_packed_sigma_backward(
    const paddle::Tensor& packed_info, const paddle::Tensor& starts,
    const paddle::Tensor& ends, const paddle::Tensor& sigmas,
    const paddle::Tensor& weights, const paddle::Tensor& grad_weights);

std::vector<paddle::Tensor> weight_from_sigma_forward(
    const paddle::Tensor& starts, const paddle::Tensor& ends,
    const paddle::Tensor& sigmas);

std::vector<paddle::Tensor> weight_from_sigma_backward(
    const paddle::Tensor& starts, const paddle::Tensor& ends,
    const paddle::Tensor& sigmas, const paddle::Tensor& weights,
    const paddle::Tensor& grad_weights);
#endif

// near_far_from_aabb
std::vector<std::vector<int64_t>> NearFarFromAABBInferShape(
    std::vector<int64_t> rays_o_shape, std::vector<int64_t> rays_d_shape,
    std::vector<int64_t> aabb_shape) {
  int64_t n_rays = rays_o_shape[0];
  return {{n_rays, 1}, {n_rays, 1}};
}

std::vector<paddle::DataType> NearFarFromAABBInferDtype(
    paddle::DataType rays_o_dtype, paddle::DataType rays_d_dtype) {
  return {rays_o_dtype, rays_o_dtype};
}

PD_BUILD_OP(near_far_from_aabb)
    .Inputs({"rays_origins", "rays_directions", "aabb"})
    .Outputs({"nears", "fars"})
    .SetKernelFn(PD_KERNEL(near_far_from_aabb))
    .SetInferShapeFn(PD_INFER_SHAPE(NearFarFromAABBInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(NearFarFromAABBInferDtype));

// first_round_ray_marching
std::vector<std::vector<int64_t>> FirstRoundRayMarchingInferShape(
    std::vector<int64_t> rays_o_shape, std::vector<int64_t> rays_d_shape,
    std::vector<int64_t> nears_shape, std::vector<int64_t> fars_shape,
    std::vector<int64_t> roi_shape, std::vector<int64_t> grid_binary_shape) {
  return {{rays_o_shape[0]}};
}

std::vector<paddle::DataType> FirstRoundRayMarchingInferDtype(
    paddle::DataType rays_o_dtype, paddle::DataType rays_d_dtype,
    paddle::DataType nears_dtype, paddle::DataType fars_dtype,
    paddle::DataType roi_dtype, paddle::DataType grid_binary_dtype) {
  return {paddle::DataType::INT32};
}

PD_BUILD_OP(first_round_ray_marching)
    .Inputs({"rays_origins", "rays_directions", "nears", "fars", "roi",
             "grid_binary"})
    .Outputs({"num_steps"})
    .Attrs({"contraction_type_: int", "step_size: float", "cone_angle: float"})
    .SetKernelFn(PD_KERNEL(first_round_ray_marching))
    .SetInferShapeFn(PD_INFER_SHAPE(FirstRoundRayMarchingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FirstRoundRayMarchingInferDtype));

// second_round_ray_marching
std::vector<std::vector<int64_t>> SecondRoundRayMarchingInferShape(
    std::vector<int64_t> rays_o_shape, std::vector<int64_t> rays_d_shape,
    std::vector<int64_t> nears_shape, std::vector<int64_t> fars_shape,
    std::vector<int64_t> aabb_shape, std::vector<int64_t> grid_binary_shape,
    std::vector<int64_t> packed_info_shape, const int contraction_type_,
    const float step_size, const float cone_angle, const int64_t total_steps) {
  return {{total_steps, 1}, {total_steps, 1}};
}

std::vector<paddle::DataType> SecondRoundRayMarchingInferDtype(
    paddle::DataType rays_o_dtype, paddle::DataType rays_d_dtype,
    paddle::DataType nears_dtype, paddle::DataType fars_dtype,
    paddle::DataType aabb_dtype, paddle::DataType grid_binary_dtype,
    paddle::DataType packed_info_dtype) {
  return {rays_o_dtype, rays_o_dtype};
}

PD_BUILD_OP(second_round_ray_marching)
    .Inputs({"rays_origins", "rays_directions", "nears", "fars", "aabb",
             "grid_binary", "packed_info"})
    .Outputs({"t_starts", "t_ends"})
    .Attrs({"contraction_type_: int", "step_size: float", "cone_angle: float",
            "total_steps: int64_t"})
    .SetKernelFn(PD_KERNEL(second_round_ray_marching))
    .SetInferShapeFn(PD_INFER_SHAPE(SecondRoundRayMarchingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SecondRoundRayMarchingInferDtype));

// pdf_ray_marching
std::vector<std::vector<int64_t>> PDFRayMarchingInferShape(
    std::vector<int64_t> starts_shape, std::vector<int64_t> ends_shape,
    std::vector<int64_t> weights_shape, std::vector<int64_t> packed_info_shape,
    std::vector<int64_t> resample_packed_info_shape,
    const int64_t total_steps) {
  return {{total_steps, 1}, {total_steps, 1}};
}

std::vector<paddle::DataType> PDFRayMarchingInferDtype(
    paddle::DataType starts_dtype, paddle::DataType ends_dtype,
    paddle::DataType weights_dtype, paddle::DataType packed_info_dtype,
    paddle::DataType resample_packed_info_dtype) {
  return {starts_dtype, ends_dtype};
}

PD_BUILD_OP(pdf_ray_marching)
    .Inputs({"t_starts", "t_ends", "weights", "packed_info",
             "resample_packed_info"})
    .Outputs({"resample_starts", "resample_ends"})
    .Attrs({"total_steps: int64_t"})
    .SetKernelFn(PD_KERNEL(pdf_ray_marching))
    .SetInferShapeFn(PD_INFER_SHAPE(PDFRayMarchingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PDFRayMarchingInferDtype));

// grid_query
std::vector<std::vector<int64_t>> GridQueryInferShape(
    std::vector<int64_t> samples_shape, std::vector<int64_t> aabb_shape,
    std::vector<int64_t> grid_value_shape) {
  return {{samples_shape[0]}};
}

std::vector<paddle::DataType> GridQueryInferDtype(
    paddle::DataType samples_dtype, paddle::DataType aabb_dtype,
    paddle::DataType grid_value_dtype) {
  return {grid_value_dtype};
}

PD_BUILD_OP(grid_query)
    .Inputs({"samples", "aabb", "grid_value"})
    .Outputs({"occupancies"})
    .Attrs({"contraction_type_: int"})
    .SetKernelFn(PD_KERNEL(grid_query))
    .SetInferShapeFn(PD_INFER_SHAPE(GridQueryInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GridQueryInferDtype));

// contract
std::vector<std::vector<int64_t>> ContractInferShape(
    std::vector<int64_t> samples_shape, std::vector<int64_t> aabb_shape) {
  return {{samples_shape[0], 3}};
}

std::vector<paddle::DataType> ContractInferDtype(paddle::DataType samples_dtype,
                                                 paddle::DataType aabb_dtype) {
  return {samples_dtype};
}

PD_BUILD_OP(contract)
    .Inputs({"samples", "aabb"})
    .Outputs({"contracted_samples"})
    .Attrs({"contraction_type_: int"})
    .SetKernelFn(PD_KERNEL(contract))
    .SetInferShapeFn(PD_INFER_SHAPE(ContractInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ContractInferDtype));

// contract_inv
std::vector<std::vector<int64_t>> ContractInvInferShape(
    std::vector<int64_t> samples_shape, std::vector<int64_t> aabb_shape) {
  return {{samples_shape[0], 3}};
}

std::vector<paddle::DataType> ContractInvInferDtype(
    paddle::DataType samples_dtype, paddle::DataType aabb_dtype) {
  return {samples_dtype};
}

PD_BUILD_OP(contract_inv)
    .Inputs({"samples", "aabb"})
    .Outputs({"contracted_samples"})
    .Attrs({"contraction_type_: int"})
    .SetKernelFn(PD_KERNEL(contract_inv))
    .SetInferShapeFn(PD_INFER_SHAPE(ContractInvInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ContractInvInferDtype));

// transmittance_from_alpha
std::vector<std::vector<int64_t>> TransmittanceFromAlphaInferShape(
    std::vector<int64_t> packed_info_shape, std::vector<int64_t> alphas_shape) {
  return {alphas_shape};
}

std::vector<paddle::DataType> TransmittanceFromAlphaInferDtype(
    paddle::DataType packed_info_dtype, paddle::DataType alphas_dtype) {
  return {alphas_dtype};
}

PD_BUILD_OP(transmittance_from_alpha)
    .Inputs({"packed_info", "alphas"})
    .Outputs({"transmittance"})
    .SetKernelFn(PD_KERNEL(transmittance_from_alpha_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(TransmittanceFromAlphaInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TransmittanceFromAlphaInferDtype));

PD_BUILD_GRAD_OP(transmittance_from_alpha)
    .Inputs({"packed_info", "alphas", "transmittance",
             paddle::Grad("transmittance")})
    .Outputs({paddle::Grad("alphas")})
    .SetKernelFn(PD_KERNEL(transmittance_from_alpha_backward));

// unpack_info
std::vector<std::vector<int64_t>> UnpackInfoInferShape(
    std::vector<int64_t> packed_info_shape, const int n_samples) {
  return {{n_samples}};
}

std::vector<paddle::DataType> UnpackInfoInferDtype(
    paddle::DataType packed_info_dtype) {
  return {packed_info_dtype};
}

PD_BUILD_OP(unpack_info)
    .Inputs({"packed_info"})
    .Outputs({"ray_indices"})
    .Attrs({"n_samples: int"})
    .SetKernelFn(PD_KERNEL(unpack_info))
    .SetInferShapeFn(PD_INFER_SHAPE(UnpackInfoInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(UnpackInfoInferDtype));

// weight_from_packed_sigma
std::vector<std::vector<int64_t>> WeightFromPackedSigmaInferShape(
    std::vector<int64_t> packed_info_shape, std::vector<int64_t> starts_shape,
    std::vector<int64_t> ends_shape, std::vector<int64_t> sigmas_shape) {
  return {sigmas_shape};
}

std::vector<paddle::DataType> WeightFromPackedSigmaInferDtype(
    paddle::DataType packed_info_dtype, paddle::DataType starts_dtype,
    paddle::DataType ends_dtype, paddle::DataType sigmas_dtype) {
  return {sigmas_dtype};
}

PD_BUILD_OP(weight_from_packed_sigma)
    .Inputs({"packed_info", "starts", "ends", "sigmas"})
    .Outputs({"weights"})
    .SetKernelFn(PD_KERNEL(weight_from_packed_sigma_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightFromPackedSigmaInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightFromPackedSigmaInferDtype));

PD_BUILD_GRAD_OP(weight_from_packed_sigma)
    .Inputs({"packed_info", "starts", "ends", "sigmas", "weights",
             paddle::Grad("weights")})
    .Outputs({paddle::Grad("sigmas")})
    .SetKernelFn(PD_KERNEL(weight_from_packed_sigma_backward));

// weight_from_sigma
std::vector<std::vector<int64_t>> WeightFromSigmaInferShape(
    std::vector<int64_t> starts_shape, std::vector<int64_t> ends_shape,
    std::vector<int64_t> sigmas_shape) {
  return {sigmas_shape};
}

std::vector<paddle::DataType> WeightFromSigmaInferDtype(
    paddle::DataType starts_dtype, paddle::DataType ends_dtype,
    paddle::DataType sigmas_dtype) {
  return {sigmas_dtype};
}

PD_BUILD_OP(weight_from_sigma)
    .Inputs({"starts", "ends", "sigmas"})
    .Outputs({"weights"})
    .SetKernelFn(PD_KERNEL(weight_from_sigma_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightFromSigmaInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightFromSigmaInferDtype));

PD_BUILD_GRAD_OP(weight_from_sigma)
    .Inputs({"starts", "ends", "sigmas", "weights", paddle::Grad("weights")})
    .Outputs({paddle::Grad("sigmas")})
    .SetKernelFn(PD_KERNEL(weight_from_sigma_backward));
