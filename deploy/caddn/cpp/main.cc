/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <time.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_file, "", "Path of a inference model");
DEFINE_string(params_file, "", "Path of a inference params");
DEFINE_string(image_file, "", "Path of a image file to be predicted");
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

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

const std::string shape_range_info = "deeplab_model/shape_range_info.pbtxt";

paddle_infer::PrecisionType GetPrecisionType(const std::string &ptype) {
  if (ptype == "trt_fp32") return paddle_infer::PrecisionType::kFloat32;
  if (ptype == "trt_fp16") return paddle_infer::PrecisionType::kHalf;
  return paddle_infer::PrecisionType::kFloat32;
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(
    const std::string &model_path, const std::string &params_path,
    const int gpu_id, const int use_trt, const int trt_precision,
    const int trt_use_static, const std::string trt_static_dir,
    const int collect_shape_info, const std::string dynamic_shape_file) {
  paddle::AnalysisConfig config;
  config.EnableUseGpu(1000, gpu_id);
  config.SetModel(model_path, params_path);
  config.EnableMemoryOptim();
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
    config.EnableTensorRtEngine(1 << 30, 1, 12, precision, trt_use_static,
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

void normalize(cv::Mat *im, const std::vector<float> &mean,
               const std::vector<float> &std, float &scale) {
  if (scale) {
    (*im).convertTo(*im, CV_32FC3, scale);
  }
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) / std[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) / std[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) / std[2];
    }
  }
}

void mat_to_vec(const cv::Mat *im, float *data) {
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

void run(Predictor *predictor, const std::vector<int> &images_shape,
         const std::vector<float> &images_data,
         const std::vector<int> &cam_shape, const std::vector<float> &cam_data,
         const std::vector<int> &lidar_shape,
         const std::vector<float> &lidar_data, std::vector<float> *boxes,
         std::vector<float> *labels, std::vector<float> *scores) {
  auto input_names = predictor->GetInputNames();
  for (const auto &tensor_name : input_names) {
    auto in_tensor = predictor->GetInputHandle(tensor_name);
    if (tensor_name == "images") {
      in_tensor->Reshape(images_shape);
      in_tensor->CopyFromCpu(images_data.data());
    } else if (tensor_name == "trans_cam_to_img") {
      in_tensor->Reshape(cam_shape);
      in_tensor->CopyFromCpu(cam_data.data());
    } else if (tensor_name == "trans_lidar_to_cam") {
      in_tensor->Reshape(lidar_shape);
      in_tensor->CopyFromCpu(lidar_data.data());
    }
  }

  for (int i = 0; i < 100; i++) {
    auto start_time = std::chrono::steady_clock::now();
    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    for (size_t i = 0; i != output_names.size(); i++) {
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
    auto end_time = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                                   start_time)
                  .count() /
              1000000.0;
    LOG(INFO) << "time per file: " << tt << "(ms).\n";
  }
}

void resize(const cv::Mat &img, cv::Mat &resize_img, int resized_h,
            int resized_w) {
  cv::resize(img, resize_img, cv::Size(resized_h, resized_w), 0, 0,
             cv::INTER_LINEAR);
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_file == "" || FLAGS_params_file == "" ||
      FLAGS_lidar_file == "") {
    LOG(INFO) << "Missing required parameter"
              << "\n";
    LOG(INFO) << "Usage: " << std::string(argv[0])
              << " --model_file ${MODEL_FILE} "
              << "--params_file ${PARAMS_FILE} "
              << "--lidar_file ${LIDAR_FILE}"
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

  cv::Mat img_resized;
  std::vector<float> input_data(1 * 3 * 640 * 960, 0.0f);
  cv::Mat img = imread(FLAGS_image_file, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

  resize(img, img_resized, 960, 640);

  img_resized.convertTo(img_resized, CV_32F, 1.0f / 255.0f);

  mat_to_vec(&img_resized, input_data.data());

  std::vector<int> images_shape = {1, 3, 640, 960};
  std::vector<int> cam_shape = {1, 3, 4};
  std::vector<float> cam_data{7.183351e+02, 0.000000e+00,  6.003891e+02,
                              4.450382e+01, 0.000000e+00,  7.183351e+02,
                              1.815122e+02, -5.951107e-01, 0.000000e+00,
                              0.000000e+00, 1.000000e+00,  2.616315e-03};

  std::vector<int> lidar_shape = {1, 4, 4};
  std::vector<float> lidar_data = {
      0.0048523,   -0.9999298, -0.01081266, -0.00711321,
      -0.00302069, 0.01079808, -0.99993706, -0.06176636,
      0.99998367,  0.00488465, -0.00296808, -0.26739058,
      0.,          0.,         0.,          1.};

  std::vector<float> boxes;
  std::vector<float> labels;
  std::vector<float> scores;
  run(predictor.get(), images_shape, input_data, cam_shape, cam_data,
      lidar_shape, lidar_data, &boxes, &labels, &scores);
  // boxes 7个数据
  std::cout << "boxes"
            << "\n";
  for (auto e : boxes) {
    LOG(INFO) << e;
  }
  // labels 1个数据
  std::cout << "labels"
            << "\n";
  for (auto e : labels) {
    LOG(INFO) << e;
  }

  // scores：1个数据
  std::cout << "scores"
            << "\n";
  for (auto e : scores) {
    LOG(INFO) << e;
  }
  return 0;
}