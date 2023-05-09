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
#include <sstream>
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
DEFINE_string(voxel_size, "", "Size of voxels for voxelize operation");
DEFINE_int32(max_points_in_voxel, 100, "Maximum number of points in a voxel");
DEFINE_int32(max_voxel_num, 12000, "Maximum number of voxels");
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

bool read_point(const std::string &file_path, const int num_point_dim,
                void **buffer, int *num_points) {
  std::ifstream file_in(file_path, std::ios::in | std::ios::binary);

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

bool hard_voxelize(const float point_cloud_range_x_min,
                   const float point_cloud_range_y_min,
                   const float point_cloud_range_z_min,
                   const float voxel_size_x, const float voxel_size_y,
                   const float voxel_size_z, const int grid_size_x,
                   const int grid_size_y, const int grid_size_z,
                   const int max_num_points_in_voxel, const int max_voxels,
                   const float *points, const int num_point_dim,
                   const int num_points, float *voxels, int *coords,
                   int *num_points_per_voxel, int *voxel_num) {
  voxel_num[0] = 0;
  int voxel_idx, grid_idx, curr_num_point;
  int coord_x, coord_y, coord_z;
  int *grid_idx_to_voxel_idx = new int[grid_size_x * grid_size_y * grid_size_z];
  memset(grid_idx_to_voxel_idx, -1,
         sizeof(int) * grid_size_x * grid_size_y * grid_size_z);
  for (int point_idx = 0; point_idx < num_points; ++point_idx) {
    coord_x = floor(
        (points[point_idx * num_point_dim + 0] - point_cloud_range_x_min) /
        voxel_size_x);
    coord_y = floor(
        (points[point_idx * num_point_dim + 1] - point_cloud_range_y_min) /
        voxel_size_y);
    coord_z = floor(
        (points[point_idx * num_point_dim + 2] - point_cloud_range_z_min) /
        voxel_size_z);

    if (coord_x < 0 || coord_x > grid_size_x || coord_x == grid_size_x) {
      continue;
    }
    if (coord_y < 0 || coord_y > grid_size_y || coord_y == grid_size_y) {
      continue;
    }
    if (coord_z < 0 || coord_z > grid_size_z || coord_z == grid_size_z) {
      continue;
    }

    grid_idx =
        coord_z * grid_size_y * grid_size_x + coord_y * grid_size_x + coord_x;
    voxel_idx = grid_idx_to_voxel_idx[grid_idx];
    if (voxel_idx == -1) {
      voxel_idx = voxel_num[0];
      if (voxel_num[0] == max_voxels || voxel_num[0] > max_voxels) {
        continue;
      }
      voxel_num[0]++;
      grid_idx_to_voxel_idx[grid_idx] = voxel_idx;
      coords[voxel_idx * 3 + 0] = coord_z;
      coords[voxel_idx * 3 + 1] = coord_y;
      coords[voxel_idx * 3 + 2] = coord_x;
    }
    curr_num_point = num_points_per_voxel[voxel_idx];
    if (curr_num_point < max_num_points_in_voxel) {
      for (int j = 0; j < num_point_dim; ++j) {
        voxels[voxel_idx * max_num_points_in_voxel * num_point_dim +
               curr_num_point * num_point_dim + j] =
            points[point_idx * num_point_dim + j];
      }
      num_points_per_voxel[voxel_idx] = curr_num_point + 1;
    }
  }
  delete[] grid_idx_to_voxel_idx;
  return true;
}

bool preprocess(const std::string &file_path, const int num_point_dim,
                const float point_cloud_range_x_min,
                const float point_cloud_range_y_min,
                const float point_cloud_range_z_min, const float voxel_size_x,
                const float voxel_size_y, const float voxel_size_z,
                const int grid_size_x, const int grid_size_y,
                const int grid_size_z, const int max_num_points_in_voxel,
                const int max_voxels, std::vector<int> *voxels_shape,
                std::vector<float> *voxels_data,
                std::vector<int> *num_points_shape,
                std::vector<int> *num_points_data,
                std::vector<int> *coords_shape, std::vector<int> *coords_data) {
  void *buffer = nullptr;
  int num_points;
  if (!read_point(file_path, num_point_dim, &buffer, &num_points)) {
    return false;
  }
  float *points = static_cast<float *>(buffer);

  float *voxels_ptr =
      new float[max_voxels * max_num_points_in_voxel * num_point_dim]();
  int *num_points_ptr = new int[max_voxels]();
  int *coords_ptr = new int[max_voxels * 3]();
  int *voxel_num_ptr = new int[1]();
  hard_voxelize(
      point_cloud_range_x_min, point_cloud_range_y_min, point_cloud_range_z_min,
      voxel_size_x, voxel_size_y, voxel_size_z, grid_size_x, grid_size_y,
      grid_size_z, max_num_points_in_voxel, max_voxels, points, num_point_dim,
      num_points, voxels_ptr, coords_ptr, num_points_ptr, voxel_num_ptr);
  free(points);

  voxels_data->assign(
      voxels_ptr,
      voxels_ptr + voxel_num_ptr[0] * max_num_points_in_voxel * num_point_dim);
  num_points_data->assign(num_points_ptr, num_points_ptr + voxel_num_ptr[0]);
  coords_data->assign(coords_ptr, coords_ptr + voxel_num_ptr[0] * 3);
  voxels_shape->push_back(voxel_num_ptr[0]);
  voxels_shape->push_back(max_num_points_in_voxel);
  voxels_shape->push_back(num_point_dim);
  num_points_shape->push_back(voxel_num_ptr[0]);
  coords_shape->push_back(voxel_num_ptr[0]);
  coords_shape->push_back(3);  // batch_id, z, y, x

  delete[] voxels_ptr;
  delete[] num_points_ptr;
  delete[] coords_ptr;
  delete[] voxel_num_ptr;

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
    config.EnableTensorRtEngine(1 << 30, 1, 10, precision, trt_use_static,
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

void run(Predictor *predictor, const std::vector<int> &voxels_shape,
         const std::vector<float> &voxels_data,
         const std::vector<int> &coords_shape,
         const std::vector<int> &coords_data,
         const std::vector<int> &num_points_shape,
         const std::vector<int> &num_points_data,
         std::vector<float> *box3d_lidar, std::vector<int64_t> *label_preds,
         std::vector<float> *scores) {
  auto input_names = predictor->GetInputNames();
  for (const auto &tensor_name : input_names) {
    auto in_tensor = predictor->GetInputHandle(tensor_name);
    if (tensor_name == "voxels") {
      in_tensor->Reshape(voxels_shape);
      in_tensor->CopyFromCpu(voxels_data.data());
    } else if (tensor_name == "coords") {
      in_tensor->Reshape(coords_shape);
      in_tensor->CopyFromCpu(coords_data.data());
    } else if (tensor_name == "num_points_per_voxel") {
      in_tensor->Reshape(num_points_shape);
      in_tensor->CopyFromCpu(num_points_data.data());
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
      label_preds->resize(out_num);
      output->CopyToCpu(label_preds->data());
    } else if (i == 2) {
      scores->resize(out_num);
      output->CopyToCpu(scores->data());
    }
  }
}

bool parse_result(const std::vector<float> &box3d_lidar,
                  const std::vector<int64_t> &label_preds,
                  const std::vector<float> &scores) {
  int num_bbox3d = scores.size();
  int bbox3d_dims = box3d_lidar.size() / num_bbox3d;
  for (size_t box_idx = 0; box_idx != num_bbox3d; ++box_idx) {
    // filter fake results:  score = -1
    if (scores[box_idx] < 0) {
      continue;
    }
    LOG(INFO) << "Score: " << scores[box_idx]
              << " Label: " << label_preds[box_idx] << " ";
    if (bbox3d_dims == 7) {
      LOG(INFO) << "Box (x_c, y_c, z_c, w, l, h, -rot): "
                << box3d_lidar[box_idx * 7 + 0] << " "
                << box3d_lidar[box_idx * 7 + 1] << " "
                << box3d_lidar[box_idx * 7 + 2] << " "
                << box3d_lidar[box_idx * 7 + 3] << " "
                << box3d_lidar[box_idx * 7 + 4] << " "
                << box3d_lidar[box_idx * 7 + 5] << " "
                << box3d_lidar[box_idx * 7 + 6] << "\n";
    }
  }

  return true;
}

void parse_string_to_vector(const std::string &str, std::vector<float> *vec) {
  std::stringstream ss(str);
  float number;
  while (ss >> number) {
    vec->push_back(number);
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_file == "" || FLAGS_params_file == "" ||
      FLAGS_lidar_file == "" || FLAGS_point_cloud_range == "" ||
      FLAGS_voxel_size == "") {
    LOG(INFO) << "Missing required parameter"
              << "\n";
    LOG(INFO) << "Usage: " << std::string(argv[0])
              << " --model_file ${MODEL_FILE} "
              << "--params_file ${PARAMS_FILE} "
              << "--lidar_file ${LIDAR_FILE} "
              << "--point_cloud_range ${POINT_CLOUD_RANGE} "
              << "--voxel_size ${VOXEL_SIZE} "
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
  std::vector<float> voxel_size;
  parse_string_to_vector(FLAGS_voxel_size, &voxel_size);

  const float point_cloud_range_x_min = point_cloud_range[0];
  const float point_cloud_range_y_min = point_cloud_range[1];
  const float point_cloud_range_z_min = point_cloud_range[2];
  const float voxel_size_x = voxel_size[0];
  const float voxel_size_y = voxel_size[1];
  const float voxel_size_z = voxel_size[2];
  int grid_size_x = static_cast<int>(
      round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size_x));
  int grid_size_y = static_cast<int>(
      round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size_y));
  int grid_size_z = static_cast<int>(
      round((point_cloud_range[5] - point_cloud_range[2]) / voxel_size_z));

  std::vector<int> voxels_shape;
  std::vector<float> voxels_data;
  std::vector<int> num_points_shape;
  std::vector<int> num_points_data;
  std::vector<int> coords_shape;
  std::vector<int> coords_data;

  if (!preprocess(FLAGS_lidar_file, FLAGS_num_point_dim,
                  point_cloud_range_x_min, point_cloud_range_y_min,
                  point_cloud_range_z_min, voxel_size_x, voxel_size_y,
                  voxel_size_z, grid_size_x, grid_size_y, grid_size_z,
                  FLAGS_max_points_in_voxel, FLAGS_max_voxel_num, &voxels_shape,
                  &voxels_data, &num_points_shape, &num_points_data,
                  &coords_shape, &coords_data)) {
    LOG(ERROR) << "Failed to preprocess!\n";
    return 0;
  }

  std::vector<float> box3d_lidar;
  std::vector<int64_t> label_preds;
  std::vector<float> scores;
  run(predictor.get(), voxels_shape, voxels_data, coords_shape, coords_data,
      num_points_shape, num_points_data, &box3d_lidar, &label_preds, &scores);

  parse_result(box3d_lidar, label_preds, scores);

  return 0;
}
