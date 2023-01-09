// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

// This version iou3d_nms_v2 takes 3d bboxes in format of [x1, y1, x2, y2, angle]
// which is different from iou3d_nms

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <paddle/extension.h>

#include <vector>

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

const int THREADS_PER_BLOCK_NMS = sizeof(int64_t) * 8;

void boxesoverlapLauncher(const int num_a, const float *boxes_a,
                          const int num_b, const float *boxes_b,
                          float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b,
                         const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, int64_t *mask, int boxes_num,
                 float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, int64_t *mask, int boxes_num,
                       float nms_overlap_thresh);

std::vector<paddle::Tensor> boxes_overlap_bev_gpu(
    const paddle::Tensor &boxes_a, const paddle::Tensor &boxes_b) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, angle]
  // params boxes_b: (M, 5) [x1, y1, x2, y2, angle]
  // params ans_overlap: (N, M)
  int num_a = boxes_a.shape()[0];
  int num_b = boxes_b.shape()[0];

  const float *boxes_a_data = boxes_a.data<float>();
  const float *boxes_b_data = boxes_b.data<float>();
  auto ans_overlap = paddle::empty({num_a, num_b}, paddle::DataType::FLOAT32,
                                   paddle::GPUPlace());
  float *ans_overlap_data = ans_overlap.data<float>();

  boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data,
                       ans_overlap_data);

  return {ans_overlap};
}

std::vector<paddle::Tensor> boxes_iou_bev_gpu(
    const paddle::Tensor &boxes_a_tensor,
    const paddle::Tensor &boxes_b_tensor) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, angle]
  // params boxes_b: (M, 5) [x1, y1, x2, y2, angle]
  // params ans_overlap: (N, M)

  int num_a = boxes_a_tensor.shape()[0];
  int num_b = boxes_b_tensor.shape()[0];

  const float *boxes_a_data = boxes_a_tensor.data<float>();
  const float *boxes_b_data = boxes_b_tensor.data<float>();
  auto ans_iou_tensor = paddle::empty({num_a, num_b}, paddle::DataType::FLOAT32,
                                      paddle::GPUPlace());
  float *ans_iou_data = ans_iou_tensor.data<float>();

  boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

  return {ans_iou_tensor};
}

std::vector<paddle::Tensor> nms_gpu(const paddle::Tensor &boxes,
                                    float nms_overlap_thresh) {
  // params boxes: (N, 5) [x1, y1, x2, y2, angle]
  auto keep = paddle::empty({boxes.shape()[0]}, paddle::DataType::INT32,
                            paddle::CPUPlace());
  auto num_to_keep_tensor =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());
  int *num_to_keep_data = num_to_keep_tensor.data<int>();

  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  int *keep_data = keep.data<int>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  // int64_t *mask_data = NULL;
  // CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks *
  // sizeof(int64_t)));
  auto mask = paddle::empty({boxes_num * col_blocks}, paddle::DataType::INT64,
                            paddle::GPUPlace());
  int64_t *mask_data = mask.data<int64_t>();
  nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // std::vector<int64_t> mask_cpu(boxes_num * col_blocks);

  // CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks *
  // sizeof(int64_t),
  //                       cudaMemcpyDeviceToHost));
  const paddle::Tensor mask_cpu_tensor = mask.copy_to(paddle::CPUPlace(), true);
  const int64_t *mask_cpu = mask_cpu_tensor.data<int64_t>();
  // cudaFree(mask_data);

  int64_t remv_cpu[col_blocks];
  memset(remv_cpu, 0, col_blocks * sizeof(int64_t));

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      const int64_t *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }

  num_to_keep_data[0] = num_to_keep;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return {keep, num_to_keep_tensor};
}

std::vector<paddle::Tensor> nms_normal_gpu(const paddle::Tensor &boxes,
                                           float nms_overlap_thresh) {
  // params boxes: (N, 5) [x1, y1, x2, y2, angle]
  // params keep: (N)

  auto keep = paddle::empty({boxes.shape()[0]}, paddle::DataType::INT32,
                            paddle::CPUPlace());
  auto num_to_keep_tensor =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());
  int *num_to_keep_data = num_to_keep_tensor.data<int>();
  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  int *keep_data = keep.data<int>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  // int64_t *mask_data = NULL;
  // CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks *
  // sizeof(int64_t)));
  auto mask = paddle::empty({boxes_num * col_blocks}, paddle::DataType::INT64,
                            paddle::GPUPlace());
  int64_t *mask_data = mask.data<int64_t>();
  nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // int64_t mask_cpu[boxes_num * col_blocks];
  // int64_t *mask_cpu = new int64_t [boxes_num * col_blocks];
  // std::vector<int64_t> mask_cpu(boxes_num * col_blocks);

  // CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks *
  // sizeof(int64_t),
  //                       cudaMemcpyDeviceToHost));

  // cudaFree(mask_data);

  const paddle::Tensor mask_cpu_tensor = mask.copy_to(paddle::CPUPlace(), true);
  const int64_t *mask_cpu = mask_cpu_tensor.data<int64_t>();

  int64_t remv_cpu[col_blocks];
  memset(remv_cpu, 0, col_blocks * sizeof(int64_t));

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      const int64_t *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }

  num_to_keep_data[0] = num_to_keep;
  if (cudaSuccess != cudaGetLastError()) {
    printf("Error!\n");
  }
  return {keep, num_to_keep_tensor};
}

std::vector<paddle::DataType> NmsInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>> NmsInferShape(
    std::vector<int64_t> boxes_shape) {
  return {{boxes_shape[0]}, {1}};
}

std::vector<paddle::DataType> NmsNormalInferDtype(
    paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>> NmsNormalInferShape(
    std::vector<int64_t> boxes_shape) {
  return {{boxes_shape[0]}, {1}};
}

std::vector<paddle::DataType> BoxesIouBevGpuInferDtype(
    paddle::DataType boxes_a_dtype, paddle::DataType boxes_b_dtype) {
  return {boxes_a_dtype};
}

std::vector<std::vector<int64_t>> BoxesIouBevGpuInferShape(
    std::vector<int64_t> boxes_a_shape, std::vector<int64_t> boxes_b_shape) {
  return {{boxes_a_shape[0], boxes_b_shape[0]}};
}

std::vector<paddle::DataType> BoxesOverlapBevGpuInferDtype(
    paddle::DataType boxes_a_dtype, paddle::DataType boxes_b_dtype) {
  return {boxes_a_dtype};
}

std::vector<std::vector<int64_t>> BoxesOverlapBevGpuInferShape(
    std::vector<int64_t> boxes_a_shape, std::vector<int64_t> boxes_b_shape) {
  return {{boxes_a_shape[0], boxes_b_shape[0]}};
}

PD_BUILD_OP(boxes_iou_bev_gpu)
    .Inputs({"boxes_a_tensor", " boxes_b_tensor"})
    .Outputs({"ans_iou_tensor"})
    .SetKernelFn(PD_KERNEL(boxes_iou_bev_gpu))
    .SetInferDtypeFn(PD_INFER_DTYPE(BoxesIouBevGpuInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(BoxesIouBevGpuInferShape));

PD_BUILD_OP(boxes_overlap_bev_gpu)
    .Inputs({"boxes_a", " boxes_b"})
    .Outputs({"ans_overlap"})
    .SetKernelFn(PD_KERNEL(boxes_overlap_bev_gpu))
    .SetInferDtypeFn(PD_INFER_DTYPE(BoxesOverlapBevGpuInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(BoxesOverlapBevGpuInferShape));

PD_BUILD_OP(nms_gpu)
    .Inputs({"boxes"})
    .Outputs({"keep", "num_to_keep"})
    .Attrs({"nms_overlap_thresh: float"})
    .SetKernelFn(PD_KERNEL(nms_gpu))
    .SetInferDtypeFn(PD_INFER_DTYPE(NmsInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(NmsInferShape));

PD_BUILD_OP(nms_normal_gpu)
    .Inputs({"boxes"})
    .Outputs({"keep", "num_to_keep"})
    .Attrs({"nms_overlap_thresh: float"})
    .SetInferShapeFn(PD_INFER_SHAPE(NmsNormalInferShape))
    .SetKernelFn(PD_KERNEL(nms_normal_gpu))
    .SetInferDtypeFn(PD_INFER_DTYPE(NmsNormalInferDtype));
