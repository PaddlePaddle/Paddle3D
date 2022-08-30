/*
This code is based on https://github.com/CVMI-Lab/PAConv/blob/main/obj_cls/cuda_lib/src/gpu/assign_score_withk_gpu.cu
Ths copyright of CVMI-Lab/PAConv is as follows:
Apache-2.0 License [see LICENSE for details].
*/
#include "paddle/extension.h"
#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

static const int SUM = 0;
static const int AVG = 1;
static const int MAX = 2;

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> assign_score_withk_backward_cuda(
                             const paddle::Tensor &scores,
                             const paddle::Tensor &points,
                             const paddle::Tensor &centers,
                             const paddle::Tensor &knn_idx,
                             const paddle::Tensor &output,
                             const paddle::Tensor &output_grad
                             );

std::vector<paddle::Tensor> assign_score_withk_forward_cuda(
                                        const paddle::Tensor &scores,
                                        const paddle::Tensor &points,
                                        const paddle::Tensor &centers,
                                        const paddle::Tensor &knn_idx);
#endif

template<typename data_t>
void assign_score_withk_forward_cpu_kernel(
    const int nthreads,
    const int B,
    const int N,
    const int M,
    const int K,
    const int O,
    const int aggregate,
    const data_t *points,
    const data_t *centers,
    const data_t *scores,
    const int64_t *knn_idx,
    data_t *output) {

  for (long i = 0; i < nthreads; i++) {
    for (int k = 0; k < K; k++) {
      // ------- loop for M ----------
      for (int m = 0; m < M; m++) {
        int b = static_cast<int>(i / (O * N));
        int n = static_cast<int>(i % (O * N) / O);
        int o = static_cast<int>(i % O);
        int kn = static_cast<int>(knn_idx[b * K * N + n * K + k]);

        if (aggregate == SUM) {
          output[b * N * O + o * N + n] += points[b * N * M * O + kn * M * O + m * O + o] *
              scores[b * N * K * M + n * K * M + k * M + m];
          output[b * N * O + o * N + n] -= centers[b * N * M * O + n * M * O + m * O + o] *
              scores[b * N * K * M + n * K * M + k * M + m];
        } else if (aggregate == AVG) {
          output[o * N + n] += 2 * points[kn * M * O + m * O + o] * scores[n * K * M + k * M + m] / K;
          output[o * N + n] -= points[n * M * O + m * O + o] * scores[n * K * M + k * M + m] / K;
        } else if (aggregate == MAX) {

        }
      }
    }
  }
}

template<typename data_t>
void assign_score_withk_backward_points_cpu_kernel(
    const int nthreads, const int B, const int N, const int M,
    const int K, const int O, const int aggregate,
    const data_t *grad_out,
    const data_t *scores,
    const int64_t *knn_idx,
    data_t *grad_points,
    data_t *grad_centers) {

  for (int i = 0; i < nthreads; i++) {
    int b = static_cast<int>(i / (M * O));
    int m = static_cast<int>(i % (M * O) / O);
    int o = static_cast<int>(i % O);

    // ----- loop for N,K ---------
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        int kn = knn_idx[b * N * K + n * K + k];
        grad_points[b * N * M * O + kn * M * O + m * O + o] +=
            scores[b * N * K * M + n * K * M + k * M + m] * grad_out[b * O * N + o * N + n];
        grad_centers[b * N * M * O + n * M * O + m * O + o] -=
            scores[b * N * K * M + n * K * M + k * M + m] * grad_out[b * O * N + o * N + n];
      }
    }
  }
}

template<typename data_t>
void assign_score_withk_backward_scores_cpu_kernel(
    const int nthreads, const int B, const int N, const int M,
    const int K, const int O, const int aggregate,
    const data_t *grad_out,
    const data_t *points,
    const data_t *centers,
    const int64_t *knn_idx,
    data_t *grad_scores) {

  for (int i = 0; i < nthreads; i++) {
    int b = static_cast<int>(i / (N * M * K));
    int n = static_cast<int>(i % (N * M * K) / M / K);
    int k = static_cast<int>(i % (M * K) / M);
    int m = static_cast<int>(i % M);
    int kn = knn_idx[b * N * K + n * K + k];

    for (int o = 0; o < O; o++) {
      grad_scores[b * N * K * M + n * K * M + k * M + m] += (points[b * N * M * O + kn * M * O + m * O + o]
          - centers[b * N * M * O + n * M * O + m * O + o]) *
          grad_out[b * O * N + o * N + n];
    }
  }
}

std::vector<paddle::Tensor> assign_score_withk_forward_cpu(
    const paddle::Tensor &scores,
    const paddle::Tensor &points,
    const paddle::Tensor &centers,
    const paddle::Tensor &knn_idx) {

  auto aggregate = SUM;
  auto B = points.shape()[0];
  auto N = points.shape()[1];
  auto M = points.shape()[2];
  auto O = points.shape()[3];

  auto K = scores.shape()[2];

  auto output = paddle::full({B, O, N}, 0, paddle::DataType::FLOAT32, paddle::CPUPlace());

  int nthreads = B * N * O;

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_score_withk_forward_cpu_kernel", ([&] {
        assign_score_withk_forward_cpu_kernel<data_t>(
            nthreads,
            B, N, M, K, O, aggregate,
            points.data<data_t>(),
            centers.data<data_t>(),
            scores.data<data_t>(),
            knn_idx.data<int64_t>(),
            output.data<data_t>()
        );
      })
  );

  return {output};
}

std::vector<paddle::Tensor> assign_score_withk_backward_cpu(
    const paddle::Tensor &scores,
    const paddle::Tensor &points,
    const paddle::Tensor &centers,
    const paddle::Tensor &knn_idx,
    const paddle::Tensor &output,
    const paddle::Tensor &output_grad
) {

  auto scores_grad = paddle::full(scores.shape(), 0, paddle::DataType::FLOAT32, paddle::CPUPlace());
  auto points_grad = paddle::full(points.shape(), 0, paddle::DataType::FLOAT32, paddle::CPUPlace());
  auto centers_grad = paddle::full(centers.shape(), 0, paddle::DataType::FLOAT32, paddle::CPUPlace());

  auto aggregate = SUM;
  auto B = points.shape()[0];
  auto N = points.shape()[1];
  auto M = points.shape()[2];
  auto O = points.shape()[3];

  auto K = scores.shape()[2];

  int nthreads_1 = B * M * O;
  int nthreads_2 = B * N * K * M;

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_score_withk_backward_points_cpu_kernel", ([&] {
        assign_score_withk_backward_points_cpu_kernel<data_t>(
            nthreads_1, B, N, M, K, O, aggregate,
            output_grad.data<data_t>(),
            scores.data<data_t>(),
            knn_idx.data<int64_t>(),
            points_grad.data<data_t>(),
            centers_grad.data<data_t>()
        );
      })
  );

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_score_withk_backward_scores_cpu_kernel", ([&] {
        assign_score_withk_backward_scores_cpu_kernel<data_t>(
            nthreads_2, B, N, M, K, O, aggregate,
            output_grad.data<data_t>(),
            points.data<data_t>(),
            centers.data<data_t>(),
            knn_idx.data<int64_t>(),
            scores_grad.data<data_t>()
        );
      })
  );
  return {scores_grad, points_grad, centers_grad};
}

std::vector<paddle::Tensor> assign_score_withk_forward(
    const paddle::Tensor &scores,
    const paddle::Tensor &points,
    const paddle::Tensor &centers,
    const paddle::Tensor &knn_idx) {

  if (scores.is_cpu()) {
    return assign_score_withk_forward_cpu(scores, points, centers, knn_idx);
#ifdef PADDLE_WITH_CUDA
    } else if (scores.is_gpu()) {
        return assign_score_withk_forward_cuda(scores, points, centers, knn_idx);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom  operator.");
  }
}

std::vector<paddle::Tensor> assign_score_withk_backward(
    const paddle::Tensor &scores,
    const paddle::Tensor &points,
    const paddle::Tensor &centers,
    const paddle::Tensor &knn_idx,
    const paddle::Tensor &output,
    const paddle::Tensor &output_grad
) {

  if (scores.is_cpu()) {
    return assign_score_withk_backward_cpu(scores, points, centers, knn_idx, output, output_grad);
#ifdef PADDLE_WITH_CUDA
    } else if (scores.is_gpu()) {
      return assign_score_withk_backward_cuda(scores, points,centers, knn_idx, output, output_grad);
#endif
  } else {
    PD_THROW("Unsupported device type for backward function of custom  operator.");
  }
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> scores_shape,
                                             std::vector<int64_t> points_shape,
                                             std::vector<int64_t> centers_shape,
                                             std::vector<int64_t> knn_idx_shape) {
  auto B = points_shape[0];
  auto N = points_shape[1];
  auto O = points_shape[3];

  return {{B, O, N}};
}

std::vector<paddle::DataType>
InferDtype(paddle::DataType t1, paddle::DataType t2, paddle::DataType t3, paddle::DataType t4) {
  return {t1};
}

PD_BUILD_OP(assign_score_withk)
    .Inputs({"scores", "points", "centers", "knn_idx"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(assign_score_withk_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));

PD_BUILD_GRAD_OP(assign_score_withk)
    .Inputs({"scores", "points", "centers", "knn_idx", "output",paddle::Grad("output")})
    .Outputs({paddle::Grad("scores"), paddle::Grad("points"), paddle::Grad("centers")})
    .SetKernelFn(PD_KERNEL(assign_score_withk_backward));