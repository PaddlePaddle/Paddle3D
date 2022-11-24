/*
This code is based on https://github.com/CVMI-Lab/PAConv/blob/main/obj_cls/cuda_lib/src/gpu/assign_score_withk_gpu.cu
Ths copyright of CVMI-Lab/PAConv is as follows:
Apache-2.0 License [see LICENSE for details].
*/
#include "paddle/extension.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK 512

const int SUM = 0;
const int AVG = 1;
const int MAX = 2;

template<typename data_t>
__global__ void assign_score_withk_forward_kernel(
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
  long gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i = gid; i < nthreads; i += blockDim.x * gridDim.x) {
    for (int k = 0; k < K; k++) {
      // ------- loop for M ----------
      for (int m = 0; m < M; m++) {
        int b = static_cast<int>(i / (O * N));
        int n = static_cast<int>(i % (O * N) / O);
        int o = static_cast<int>(i % O);
        int kn = static_cast<int>(knn_idx[b * K * N + n * K + k]);

        if (aggregate == SUM) {
          // feature concat
          atomicAdd(output + b * N * O + o * N + n,
                    points[b * N * M * O + kn * M * O + m * O + o] *
                        scores[b * N * K * M + n * K * M + k * M + m]
                        - centers[b * N * M * O + n * M * O + m * O + o] *
                            scores[b * N * K * M + n * K * M + k * M + m]);
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
__global__ void assign_score_withk_backward_scores_kernel(
    const int nthreads, const int B, const int N, const int M,
    const int K, const int O, const int aggregate,
    const data_t *grad_out,
    const data_t *points,
    const data_t *centers,
    const int64_t *knn_idx,
    data_t *grad_scores) {
  long gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i = gid; i < nthreads; i += blockDim.x * gridDim.x) {
    int b = static_cast<int>(i / (N * M * K));
    int n = static_cast<int>(i % (N * M * K) / M / K);
    int k = static_cast<int>(i % (M * K) / M);
    int m = static_cast<int>(i % M);
    int kn = knn_idx[b * N * K + n * K + k];

    for (int o = 0; o < O; o++) {
      atomicAdd(grad_scores + b * N * K * M + n * K * M + k * M + m,
                (points[b * N * M * O + kn * M * O + m * O + o]
                    - centers[b * N * M * O + n * M * O + m * O + o]) * grad_out[b * O * N + o * N + n]);
    }
  }
}

template<typename data_t>
__global__ void assign_score_withk_backward_points_kernel(
    const int nthreads, const int B, const int N, const int M,
    const int K, const int O, const int aggregate,
    const data_t *grad_out,
    const data_t *scores,
    const int64_t *knn_idx,
    data_t *grad_points,
    data_t *grad_centers) {
  long gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i = gid; i < nthreads; i += blockDim.x * gridDim.x) {
    int b = static_cast<int>(i / (M * O));
    int m = static_cast<int>(i % (M * O) / O);
    int o = static_cast<int>(i % O);

    // ----- loop for N,K ---------
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        int kn = knn_idx[b * N * K + n * K + k];
        atomicAdd(grad_points + b * N * M * O + kn * M * O + m * O + o,
                  scores[b * N * K * M + n * K * M + k * M + m] * grad_out[b * O * N + o * N + n]);
        atomicAdd(grad_centers + b * N * M * O + n * M * O + m * O + o,
                  -scores[b * N * K * M + n * K * M + k * M + m] * grad_out[b * O * N + o * N + n]);
      }
    }
  }
}

std::vector<paddle::Tensor> assign_score_withk_forward_cuda(
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

  auto output = paddle::full({B, O, N}, 0, paddle::DataType::FLOAT32, paddle::GPUPlace());

  int nthreads = B * N * O;

  int grid = (nthreads + BLOCK - 1) / BLOCK;

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_score_withk_forward_kernel", ([&] {
        assign_score_withk_forward_kernel<data_t><<<grid, BLOCK, 0, points.stream()>>>(
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

std::vector<paddle::Tensor> assign_score_withk_backward_cuda(
    const paddle::Tensor &scores,
    const paddle::Tensor &points,
    const paddle::Tensor &centers,
    const paddle::Tensor &knn_idx,
    const paddle::Tensor &output,
    const paddle::Tensor &output_grad
) {
  auto scores_grad = paddle::full(scores.shape(), 0, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto points_grad = paddle::full(points.shape(), 0, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto centers_grad = paddle::full(centers.shape(), 0, paddle::DataType::FLOAT32, paddle::GPUPlace());

  auto aggregate = SUM;
  auto B = points.shape()[0];
  auto N = points.shape()[1];
  auto M = points.shape()[2];
  auto O = points.shape()[3];

  auto K = scores.shape()[2];

  int nthreads_1 = B * M * O;
  int nthreads_2 = B * N * K * M;

  int grid1 = (nthreads_1 + BLOCK - 1) / BLOCK;
  int grid2 = (nthreads_2 + BLOCK - 1) / BLOCK;

  PD_DISPATCH_FLOATING_TYPES(
      scores.type(), "assign_score_withk_backward_points_kernel", ([&] {
        assign_score_withk_backward_points_kernel<data_t><<<grid1, BLOCK, 0, scores.stream()>>>(
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
      points.type(), "assign_score_withk_backward_scores_kernel", ([&] {
        assign_score_withk_backward_scores_kernel<data_t><<<grid2, BLOCK, 0, points.stream()>>>(
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
