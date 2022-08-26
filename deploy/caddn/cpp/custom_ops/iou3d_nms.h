#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

// #include <paddle/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

#include "paddle/include/experimental/ext_all.h"

int boxes_overlap_bev_gpu(paddle::Tensor boxes_a, paddle::Tensor boxes_b,
                          paddle::Tensor ans_overlap);
int boxes_iou_bev_gpu(paddle::Tensor boxes_a, paddle::Tensor boxes_b,
                      paddle::Tensor ans_iou);
std::vector<paddle::Tensor> nms_gpu(const paddle::Tensor& boxes,
                                    float nms_overlap_thresh);
int nms_normal_gpu(paddle::Tensor boxes, paddle::Tensor keep,
                   float nms_overlap_thresh);

#endif
