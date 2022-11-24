#ifndef IOU3D_CPU_H
#define IOU3D_CPU_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

#include "paddle/include/experimental/ext_all.h"

int boxes_iou_bev_cpu(paddle::Tensor boxes_a_tensor,
                      paddle::Tensor boxes_b_tensor,
                      paddle::Tensor ans_iou_tensor);

#endif
