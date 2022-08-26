#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>

#include "iou3d_cpu.h"
#include "iou3d_nms.h"
#include "paddle/include/experimental/ext_all.h"

std::vector<std::vector<int64_t>> NMSInferShape(
    std::vector<int64_t> boxes_shape) {
  int64_t keep_num = 1;
  return {{boxes_shape[0]}, {keep_num}};
}

std::vector<paddle::DataType> NMSInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::INT64};
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// 	m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes
// overlap");
// 	m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
// 	m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
// 	m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
// 	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");
// }

PD_BUILD_OP(nms_gpu)
    .Inputs({"boxes"})
    .Outputs({"keep", "num_to_keep"})
    .Attrs({"nms_overlap_thresh: float"})
    .SetKernelFn(PD_KERNEL(nms_gpu))
    .SetInferDtypeFn(PD_INFER_DTYPE(NMSInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(NMSInferShape));
