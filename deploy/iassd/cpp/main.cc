#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using paddle_infer::Predictor;

DEFINE_string(model_file, "", "Path of inference model");
DEFINE_string(params_file, "", "Path of inference params");
DEFINE_string(lidar_file, "", "Path of lidar file");
DEFINE_string(run_mode, "", "Run mode, could be fp32, trt_fp32, trt_fp16");
DEFINE_int32(gpu_id, 0, "GPU Id");

bool read_point(const std::string &file_path, const int &num_point_dim,
                void **buffer, int *num_point) {
  std::ifstream file_in(file_path, std::ios::in | std::ios::binary);

  if (!file_in) {
    LOG(ERROR) << "Failed to read file: " << file_path << "\n";
    return false;
  }

  // get file size
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
  *num_point = file_size / sizeof(float) / num_point_dim;
  return true;
}

void mask_points_outside_range(const std::vector<float> &points,
                               const std::vector<float> &point_cloud_range,
                               const int &num_point_dim,
                               std::vector<float> *selected_points) {
  for (int i = 0; i < points.size(); i += num_point_dim) {
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

void sample_points(const std::vector<float> &points, const int &dst_num_points,
                   const int &num_point_dim,
                   std::vector<float> *selected_points) {
  int src_num_points = points.size() / num_point_dim;
  std::vector<int> far_idx_choice;
  std::vector<int> near_idx;
  std::vector<int> choice;
  std::random_device rd;
  std::mt19937 g(rd());
  g.seed(1024);
  if (dst_num_points < src_num_points) {
    for (int i = 0; i < src_num_points; ++i) {
      float pt_x = points[i * num_point_dim];
      float pt_y = points[i * num_point_dim + 1];
      float pt_z = points[i * num_point_dim + 2];
      float dist = sqrt(pt_x * pt_x + pt_y * pt_y + pt_z * pt_z);
      if (dist < 40.0) {
        near_idx.push_back(i);
      } else {
        far_idx_choice.push_back(i);
      }
    }
    if (dst_num_points > far_idx_choice.size()) {
      // shuffle near_idx
      std::shuffle(near_idx.begin(), near_idx.end(), g);
      choice.insert(choice.begin(), near_idx.begin(),
                    near_idx.begin() + dst_num_points - far_idx_choice.size());
      if (far_idx_choice.size() > 0) {
        choice.insert(choice.end(), far_idx_choice.begin(),
                      far_idx_choice.end());
      }
    } else {
      std::vector<int> src_idx(src_num_points);
      for (int v = 0; v < src_num_points; ++v) {
        src_idx[v] = v;
      }
      // shuffle src_idx
      std::shuffle(src_idx.begin(), src_idx.end(), g);
      choice.insert(choice.begin(), src_idx.begin(),
                    src_idx.begin() + dst_num_points);
    }
  } else {
    std::vector<int> src_idx(src_num_points);
    for (int v = 0; v < src_num_points; ++v) {
      src_idx[v] = v;
    }
    choice.insert(choice.begin(), src_idx.begin(), src_idx.end());
    if (dst_num_points > src_num_points) {
      for (int i = src_num_points; i < dst_num_points; ++i) {
        std::uniform_int_distribution<int> uniform_dist(0, src_num_points - 1);
        choice.push_back(uniform_dist(g));
      }
    }
  }
  // sample points by selected choice
  for (int i = 0; i < choice.size(); ++i) {
    int idx = choice[i];
    for (int d = 0; d < num_point_dim; ++d) {
      selected_points->push_back(points[idx * num_point_dim + d]);
    }
  }
}

std::shared_ptr<Predictor> init_predictor(const std::string &model_file,
                                          const std::string &params_file,
                                          const std::string &run_mode,
                                          const int &gpu_id) {
  // init config
  Config config;
  config.SetModel(model_file, params_file);
  config.EnableUseGpu(1000, gpu_id);

  // trt setting
  if (run_mode == "trt_fp32") {
    config.EnableTensorRtEngine(1 << 30, 1, 15, PrecisionType::kFloat32, false,
                                false);
  } else if (run_mode == "trt_fp16") {
    config.EnableTensorRtEngine(1 << 30, 1, 15, PrecisionType::kHalf, false,
                                false);
  }

  // memory optim
  config.EnableMemoryOptim();

  return CreatePredictor(config);
}

void run(Predictor *predictor, std::vector<float> &points,
         std::vector<int> &input_shape, std::vector<float> *boxes,
         std::vector<long> *labels, std::vector<float> *scores) {
  // setup input points handle
  auto input_names = predictor->GetInputNames();
  auto points_handle =
      predictor->GetInputHandle(input_names[0]);  // just one input: point cloud
  points_handle->Reshape(input_shape);
  points_handle->CopyFromCpu(points.data());

  // do infer
  CHECK(predictor->Run());

  // fetch predict boxes, labels, scores
  auto output_names = predictor->GetOutputNames();
  for (int i = 0; i < output_names.size(); ++i) {
    auto output = predictor->GetOutputHandle(output_names[i]);
    std::vector<int> output_shape = output->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    if (i == 0) {
      boxes->resize(out_num);
      output->CopyToCpu(boxes->data());
    } else if (i == 1) {
      labels->resize(out_num);
      output->CopyToCpu(labels->data());
    } else if (i == 2) {
      scores->resize(out_num);
      output->CopyToCpu(scores->data());
    }
  }
}

void print_results(const std::vector<float> &boxes,
                   const std::vector<long> &labels,
                   const std::vector<float> &scores) {
  int num_boxes = scores.size();
  int boxes_dim = boxes.size() / num_boxes;
  for (int box_idx = 0; box_idx != num_boxes; ++box_idx) {
    // filter fake results:  label = -1
    if (labels[box_idx] < 0.0) {
      continue;
    }
    LOG(INFO) << "Score: " << scores[box_idx] << " Label: " << labels[box_idx]
              << " ";
    if (boxes_dim == 7) {
      LOG(INFO) << "Box (x_c, y_c, z_c, w, l, h, rot): "
                << boxes[box_idx * 7 + 0] << " " << boxes[box_idx * 7 + 1]
                << " " << boxes[box_idx * 7 + 2] << " "
                << boxes[box_idx * 7 + 3] << " " << boxes[box_idx * 7 + 4]
                << " " << boxes[box_idx * 7 + 5] << " "
                << boxes[box_idx * 7 + 6] << "\n";
    }
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = init_predictor(FLAGS_model_file, FLAGS_params_file,
                                  FLAGS_run_mode, FLAGS_gpu_id);

  // input handle and settings
  const int num_sampled_point = 16384;
  const int num_point_dim = 4;  // xyz + intensity
  std::vector<float> point_cloud_range = {0, -40, -3, 70.4, 40, 1};
  std::vector<int> input_shape = {num_sampled_point, num_point_dim};

  // read points
  int num_point;
  void *buffer = nullptr;
  if (!read_point(FLAGS_lidar_file, num_point_dim, &buffer, &num_point)) {
    return false;
  }
  float *points = static_cast<float *>(buffer);
  std::vector<float> input_data(points, points + num_point * num_point_dim);

  // preprocess
  std::vector<float> masked_points;
  mask_points_outside_range(input_data, point_cloud_range, num_point_dim,
                            &masked_points);
  std::vector<float> selected_points;
  sample_points(masked_points, num_sampled_point, num_point_dim,
                &selected_points);

  // output handle
  std::vector<float> boxes;
  std::vector<long> labels;
  std::vector<float> scores;

  // run infer
  run(predictor.get(), selected_points, input_shape, &boxes, &labels, &scores);

  // print results
  print_results(boxes, labels, scores);

  return 0;
}
