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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

DEFINE_string(model_file, "", "Path of a inference model");
DEFINE_string(params_file, "", "Path of a inference params");
DEFINE_string(lidar_file, "", "Path of a lidar file to be predicted");
DEFINE_int32(num_point_dim, 4, "Dimension of a point in the lidar file");
DEFINE_string(point_cloud_range, "",
              "Range of point cloud for voxelize operation");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_int32(use_trt, 0,
             "Whether to use tensorrt to accelerate when using gpu");
DEFINE_int32(trt_precision, 0,
             "Precision type of tensorrt, 0: kFloat32, 1: kHalf");
DEFINE_int32(
    trt_use_static, 0,
    "Whether to load the tensorrt graph optimization from a disk path");
DEFINE_string(trt_static_dir, "",
              "Path of a tensorrt graph optimization directory");
DEFINE_int32(collect_shape_info, 0,
             "Whether to collect dynamic shape before using tensorrt");
DEFINE_string(dynamic_shape_file, "",
              "Path of a dynamic shape file for tensorrt");

void parse_string_to_vector(const std::string &str, std::vector<float> *vec) {
  std::stringstream ss(str);
  float number;
  while (ss >> number) {
    vec->push_back(number);
  }
}

bool read_point(const std::string &file_path, const int num_point_dim,
                void **buffer, int *num_points) {
  std::ifstream file_in(file_path, std::ios::in | std::ios::binary);
  if (num_point_dim < 4) {
    LOG(ERROR) << "Point dimension must not be less than 4, but received "
               << "num_point_dim is " << num_point_dim << ".\n";
  }

  if (!file_in) {
    LOG(ERROR) << "Failed to read file: " << file_path << "\n";
    return false;
  }

  std::streampos file_size;
  file_in.seekg(0, std::ios::end);
  file_size = file_in.tellg();
  file_in.seekg(0, std::ios::beg);

  *buffer = malloc(file_size);
  if (*buffer == nullptr) {
    LOG(ERROR) << "Failed to malloc memory of size: " << file_size << "\n";
    return false;
  }
  file_in.read(reinterpret_cast<char *>(*buffer), file_size);
  file_in.close();

  if (file_size / sizeof(float) % num_point_dim != 0) {
    LOG(ERROR) << "Loaded file size (" << file_size
               << ") is not evenly divisible by num_point_dim ("
               << num_point_dim << ")\n";
    return false;
  }
  *num_points = file_size / sizeof(float) / num_point_dim;
  return true;
}

void mask_points_outside_range(const float *points, const int num_points,
                               const std::vector<float> &point_cloud_range,
                               const int num_point_dim,
                               std::vector<float> *selected_points) {
  for (int i = 0; i < num_points; i += num_point_dim) {
    float pt_x = points[i];
    float pt_y = points[i + 1];
    // in [-x, x] and [-y, y] range
    if ((pt_x >= point_cloud_range[0]) && (pt_x <= point_cloud_range[3]) &&
        (pt_y >= point_cloud_range[1]) && (pt_y <= point_cloud_range[4])) {
      for (int d = 0; d < num_point_dim; ++d) {
        selected_points->push_back(points[i + d]);
      }
    }
  }
}

bool preprocess(const std::string &file_path, const int num_point_dim,
                const std::vector<float> &point_cloud_range,
                std::vector<int> *points_shape,
                std::vector<float> *points_data) {
  void *buffer = nullptr;
  int num_points = 0;
  if (!read_point(file_path, num_point_dim, &buffer, &num_points)) {
    return false;
  }
  float *points = static_cast<float *>(buffer);

  std::vector<float> masked_points;
  mask_points_outside_range(points, num_points, point_cloud_range,
                            num_point_dim, &masked_points);

  points_data->assign(masked_points.begin(), masked_points.end());
  points_shape->push_back(masked_points.size() / num_point_dim);
  points_shape->push_back(num_point_dim);

  free(points);
  return true;
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(
    const std::string &model_path, const std::string &params_path,
    const int gpu_id, const int use_trt, const int trt_precision,
    const int trt_use_static, const std::string trt_static_dir,
    const int collect_shape_info, const std::string dynamic_shape_file) {
  paddle::AnalysisConfig config;
  config.EnableUseGpu(1000, gpu_id);
  config.SetModel(model_path, params_path);
  if (use_trt) {
    paddle::AnalysisConfig::Precision precision;
    if (trt_precision == 0) {
      precision = paddle_infer::PrecisionType::kFloat32;
    } else if (trt_precision == 1) {
      precision = paddle_infer::PrecisionType::kHalf;
    } else {
      LOG(ERROR) << "Tensorrt type can only support 0 or 1, but received is"
                 << trt_precision << "\n";
      return nullptr;
    }
    config.EnableTensorRtEngine(1 << 30, 1, 20, precision, trt_use_static,
                                false);

    if (dynamic_shape_file == "") {
      LOG(ERROR) << "dynamic_shape_file should be set, but received is "
                 << dynamic_shape_file << "\n";
      return nullptr;
    }
    if (collect_shape_info) {
      config.CollectShapeRangeInfo(dynamic_shape_file);
    } else {
      config.EnableTunedTensorRtDynamicShape(dynamic_shape_file, true);
    }

    if (trt_use_static) {
      if (trt_static_dir == "") {
        LOG(ERROR) << "trt_static_dir should be set, but received is "
                   << trt_static_dir << "\n";
        return nullptr;
      }
      config.SetOptimCacheDir(trt_static_dir);
    }
  }
  config.SwitchIrOptim(true);
  return paddle_infer::CreatePredictor(config);
}

void run(Predictor *predictor, const std::vector<int> &points_shape,
         const std::vector<float> &points_data, std::vector<float> *box3d_lidar,
         std::vector<int64_t> *label_preds, std::vector<float> *scores) {
  auto input_names = predictor->GetInputNames();
  for (const auto &tensor_name : input_names) {
    auto in_tensor = predictor->GetInputHandle(tensor_name);
    if (tensor_name == "data") {
      in_tensor->Reshape(points_shape);
      in_tensor->CopyFromCpu(points_data.data());
    }
  }

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  for (size_t i = 0; i != output_names.size(); i++) {
    auto output = predictor->GetOutputHandle(output_names[i]);
    std::vector<int> output_shape = output->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    if (i == 0) {
      box3d_lidar->resize(out_num);
      output->CopyToCpu(box3d_lidar->data());
    } else if (i == 1) {
      scores->resize(out_num);
      output->CopyToCpu(scores->data());
    } else if (i == 2) {
      label_preds->resize(out_num);
      output->CopyToCpu(label_preds->data());
    }
  }
}

bool parse_result(const std::vector<float> &box3d_lidar,
                  const std::vector<int64_t> &label_preds,
                  const std::vector<float> &scores) {
  int num_bbox3d = scores.size();
  for (size_t box_idx = 0; box_idx != num_bbox3d; ++box_idx) {
    // filter fake results:  score = -1
    if (scores[box_idx] < 0) {
      continue;
    }
    LOG(INFO) << "Score: " << scores[box_idx]
              << " Label: " << label_preds[box_idx] << " ";

    LOG(INFO) << "Box (x_c, y_c, z_c, w, l, h, -rot): "
              << box3d_lidar[box_idx * 7 + 0] << " "
              << box3d_lidar[box_idx * 7 + 1] << " "
              << box3d_lidar[box_idx * 7 + 2] << " "
              << box3d_lidar[box_idx * 7 + 3] << " "
              << box3d_lidar[box_idx * 7 + 4] << " "
              << box3d_lidar[box_idx * 7 + 5] << " "
              << box3d_lidar[box_idx * 7 + 6] << "\n";
  }

  return true;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_file == "" || FLAGS_params_file == "" ||
      FLAGS_lidar_file == "" || FLAGS_point_cloud_range == "") {
    LOG(INFO) << "Missing required parameter"
              << "\n";
    LOG(INFO) << "Usage: " << std::string(argv[0])
              << " --model_file ${MODEL_FILE} "
              << "--params_file ${PARAMS_FILE} "
              << "--lidar_file ${LIDAR_FILE}"
              << "--point_cloud_range ${POINT_CLOUD_RANGE} "
              << "\n";
    return -1;
  }

  auto predictor = create_predictor(
      FLAGS_model_file, FLAGS_params_file, FLAGS_gpu_id, FLAGS_use_trt,
      FLAGS_trt_precision, FLAGS_trt_use_static, FLAGS_trt_static_dir,
      FLAGS_collect_shape_info, FLAGS_dynamic_shape_file);
  if (predictor == nullptr) {
    return 0;
  }

  std::vector<float> point_cloud_range;
  parse_string_to_vector(FLAGS_point_cloud_range, &point_cloud_range);

  std::vector<int> points_shape;
  std::vector<float> points_data;
  if (!preprocess(FLAGS_lidar_file, FLAGS_num_point_dim, point_cloud_range,
                  &points_shape, &points_data)) {
    LOG(ERROR) << "Failed to preprocess!\n";
    return 0;
  }

  std::vector<float> box3d_lidar;
  std::vector<int64_t> label_preds;
  std::vector<float> scores;
  run(predictor.get(), points_shape, points_data, &box3d_lidar, &label_preds,
      &scores);

  parse_result(box3d_lidar, label_preds, scores);

  return 0;
}
