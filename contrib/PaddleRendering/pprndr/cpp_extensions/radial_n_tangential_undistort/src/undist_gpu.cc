#include <paddle/extension.h>

#include <vector>

/**
 * @brief cpu distortion correction template function
 *
 * @param [in] xy_coords input xy coordinate (in camera coordinate system) Nx2
 * @param [in] input_distortion_coeffs distortion coefficient, must be 6, order
 * k1 k2 k3 k4 p1 p2
 * @param [in] xy_undist distortion corrected coordinates (in camera coordinate
 * system) Nx2
 * @param [in] input_numel number of pixels
 * @param [in] max_iterations iterations
 * @param [in] eps minimum resolution error
 */
template <typename data_t>
void cv_undistort_cpu_kernel(const data_t *xy_coords,
                             const data_t *distortion_coeffs, data_t *xy_undist,
                             float eps, int max_iterations, int input_numel) {
  float k1, k2, k3, k4, p1, p2;
  k1 = distortion_coeffs[0];
  k2 = distortion_coeffs[1];

  k3 = distortion_coeffs[2];
  k4 = distortion_coeffs[3];

  p1 = distortion_coeffs[4];
  p2 = distortion_coeffs[5];

  for (int64_t xy_number = 0; xy_number < input_numel; xy_number += 2) {
    float x = xy_coords[xy_number];
    float y = xy_coords[xy_number + 1];
    for (int iteration = 0; iteration < max_iterations; iteration++) {
      float xd, yd;
      xd = xy_coords[xy_number];
      yd = xy_coords[xy_number + 1];
      float fx, fy, fx_x, fx_y, fy_x, fy_y, r, d, d_r, d_x, d_y;

      r = x * x + y * y;
      d = (float)1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)));

      fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd;
      fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd;

      //# Compute derivative of d over [x, y]
      d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4));
      d_x = 2.0 * x * d_r;
      d_y = 2.0 * y * d_r;

      //# Compute derivative of fx over x and y.
      fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x;
      fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y;

      //# Compute derivative of fy over x and y.
      fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x;
      fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y;

      //
      float denominator, x_numerator, y_numerator, step_x, step_y;
      denominator = fy_x * fx_y - fx_x * fy_y;
      x_numerator = fx * fy_y - fy * fx_y;
      y_numerator = fy * fx_x - fx * fy_x;

      if (abs(denominator) > eps) {
        step_x = x_numerator / denominator;
        step_y = y_numerator / denominator;
      } else {
        step_x = 0.0;
        step_y = 0.0;
      }
      x += step_x;
      y += step_y;
    }
    xy_undist[xy_number] = x;
    xy_undist[xy_number + 1] = y;
  }
}

/**
 * @brief cpu version distortion correction function
 *
 * @param [in] xy_coords input xy coordinate (in camera coordinate system) Nx2
 * @param [in] input_distortion_coeffs distortion coefficient, must be 6, order
 * k1 k2 k3 k4 p1 p2
 * @param [in] max_iterations iterations
 * @param [in] eps
 * @return xy_undist distortion corrected coordinates (in camera coordinate
 * system) Nx2
 */
std::vector<paddle::Tensor> cv_undistort_cpu(
    const paddle::Tensor &xy_coords,
    const paddle::Tensor &input_distortion_coeffs, int max_iterations,
    float eps) {
  auto xy_undist =
      paddle::Tensor(paddle::PlaceType::kCPU, xy_coords.shape());  // output
  int input_numel = xy_coords.size();  // input data size

  PD_CHECK(input_distortion_coeffs.size() == 6,
           "PD_CHECK returns input_distortion_coeffs.size() !=6.");

  PD_DISPATCH_FLOATING_TYPES(
      xy_coords.type(), "cv_undistort_cpu_kernel", ([&] {
        cv_undistort_cpu_kernel<data_t>(
            xy_coords.data<data_t>(), input_distortion_coeffs.data<data_t>(),
            xy_undist.data<data_t>(), eps, max_iterations, input_numel);
      }));
  return {xy_undist};
}

/**
 * @brief gpu distortion correction interface
 *
 * @param [in] xy_coords input xy coordinate (in camera coordinate system) Nx2
 * @param [in] input_distortion_coeffs distortion coefficient, must be 6, order
 * k1 k2 k3 k4 p1 p2
 * @param [in] max_iterations iterations
 * @param [in] eps
 * @return Distortion corrected coordinates (in camera coordinate system) Nx2
 */
std::vector<paddle::Tensor> cv_undistort_cuda(
    const paddle::Tensor &xy_coords,
    const paddle::Tensor &input_distortion_coeffs, int max_iterations,
    float eps);

/**
 * @brief distortion correction interface, supports automatic selection to run
 * on cpu or gpu according to the input tensor type
 *
 * @param [in] xy_coords input xy coordinate (in camera coordinate system) Nx2
 * @param [in] input_distortion_coeffs distortion coefficient, must be 6, order
 * k1 k2 k3 k4 p1 p2
 * @return Distortion corrected coordinates (in camera coordinate system) Nx2
 */
std::vector<paddle::Tensor> cv_undist(
    const paddle::Tensor &xy_coords,
    const paddle::Tensor &input_distortion_coeffs, int max_iterations = 10,
    float eps = 1e-9) {
  if (xy_coords.is_cpu()) {
    PD_CHECK(input_distortion_coeffs.is_cpu(),
             "input_distortion_coeffs and xy_coords not in the same deivce");
    return cv_undistort_cpu(xy_coords, input_distortion_coeffs, max_iterations,
                            eps);

#ifdef PADDLE_WITH_CUDA
  } else if (xy_coords.is_gpu()) {
    PD_CHECK(input_distortion_coeffs.is_gpu(),
             "input_distortion_coeffs and xy_coords not in the same deivce");
    return cv_undistort_cuda(xy_coords, input_distortion_coeffs, max_iterations,
                             eps);
#endif

  } else {
    PD_THROW(
        "Unsupported device type for forward function of custom relu "
        "operator.");
  }
}

PD_BUILD_OP(radial_n_tangential_undistort)
    .Inputs({"xy_coords", "input_distortion_coeffs"})
    .Outputs({"output1"})
    .Attrs({"max_iterations: int", "eps: float"})
    .SetKernelFn(PD_KERNEL(cv_undist));
