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
DEFINE_string(
    image_files, "",
    "list Path of a image file to be predicted, which split by comma");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_bool(use_trt, false,
            "Whether to use tensorrt to accelerate when using gpu");
DEFINE_int32(trt_precision, 0,
             "Precision type of tensorrt, 0: kFloat32, 1: kHalf");
DEFINE_bool(trt_use_static, false,
            "Whether to load the tensorrt graph optimization from a disk path");
DEFINE_string(trt_static_dir, "",
              "Path of a tensorrt graph optimization directory");
DEFINE_bool(collect_shape_info, false,
            "Whether to collect dynamic shape before using tensorrt");
DEFINE_string(dynamic_shape_file, "petr_shape_info.txt",
              "Path of a dynamic shape file for tensorrt");
DEFINE_bool(with_timestamp, false,
            "Whether model with timestamp input(for petrv2)");
using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

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
         const std::vector<float> &images_data, const std::vector<int> &k_shape,
         const std::vector<float> &k_data,
         const std::vector<int> &timestamp_shape,
         const std::vector<float> &timestamp_data, std::vector<float> *boxes,
         std::vector<float> *scores, std::vector<int64_t> *labels,
         const bool with_timestamp) {
  auto input_names = predictor->GetInputNames();

  auto in_tensor0 = predictor->GetInputHandle(input_names[0]);
  in_tensor0->Reshape(images_shape);
  in_tensor0->CopyFromCpu(images_data.data());

  auto in_tensor1 = predictor->GetInputHandle(input_names[1]);
  in_tensor1->Reshape(k_shape);
  in_tensor1->CopyFromCpu(k_data.data());

  if (with_timestamp) {
    auto in_tensor2 = predictor->GetInputHandle(input_names[2]);
    in_tensor2->Reshape(timestamp_shape);
    in_tensor2->CopyFromCpu(timestamp_data.data());
  }

  for (int i = 0; i < 1; i++) {
    auto start_time = std::chrono::steady_clock::now();
    CHECK(predictor->Run());
    std::cout << "finish run!!!!" << std::endl;
    auto output_names = predictor->GetOutputNames();
    for (size_t i = 0; i != output_names.size(); i++) {
      auto output = predictor->GetOutputHandle(output_names[i]);

      std::vector<int> output_shape = output->shape();
      int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());
      if (i == 0) {
        std::cout << "get bbox out size: " << out_num << std::endl;
        boxes->resize(out_num);
        output->CopyToCpu(boxes->data());
      } else if (i == 1) {
        std::cout << "get scores out size: " << out_num << std::endl;
        scores->resize(out_num);
        output->CopyToCpu(scores->data());
      } else if (i == 2) {
        std::cout << "get labels out size: " << out_num << std::endl;
        labels->resize(out_num);
        output->CopyToCpu(labels->data());
        std::cout << "finish get labels out size: " << out_num << std::endl;
      }
    }
    // std::cout << "get out: " << i << std::endl;

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

std::vector<std::string> split_by_comma(std::string image_files) {
  std::vector<std::string> vecs;
  std::stringstream image_files_ss(image_files);

  while (image_files_ss.good()) {
    std::string substr;
    getline(image_files_ss, substr, ',');
    vecs.push_back(substr);
  }
  return vecs;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_file == "" || FLAGS_params_file == "" ||
      FLAGS_image_files == "") {
    LOG(INFO) << "Missing required parameter"
              << "\n";
    LOG(INFO) << "Usage: " << std::string(argv[0])
              << " --model_file ${MODEL_FILE} "
              << "--params_file ${PARAMS_FILE} "
              << "--image_files ${IMAGE_FILES}"
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

  int num_cams = 6;

  if (FLAGS_with_timestamp) {
    num_cams = 12;
  }

  std::vector<float> input_data(num_cams * 3 * 320 * 800, 0.0f);

  std::vector<cv::Mat> imgs;
  auto filenames = split_by_comma(FLAGS_image_files);

  for (auto filename : filenames) {
    cv::Mat img = imread(filename, cv::IMREAD_COLOR);
    imgs.push_back(img);
  }
  std::cout << "imgs size: " << imgs.size() << std::endl;
  std::vector<cv::Mat> cropped_imgs;

  std::vector<float> mean{103.530, 116.280, 123.675};
  std::vector<float> std{57.375, 57.120, 58.395};
  float scale = 1.0f;
  for (auto img : imgs) {
    cv::Mat img_resized;
    resize(img, img_resized, 800, 450);
    auto crop_img = img_resized(cv::Range(130, 450), cv::Range(0, 800));
    normalize(&crop_img, mean, std, scale);
    cropped_imgs.push_back(crop_img);
  }

  for (int i = 0; i < num_cams; i++) {
    mat_to_vec(&cropped_imgs[i], input_data.data() + i * (3 * 320 * 800));
  }

  std::vector<int> images_shape = {1, num_cams, 3, 320, 800};
  std::vector<int> k_shape = {1, num_cams, 4, 4};
  std::vector<int> timestamp_shape = {1, num_cams};

  /* clang-format off */
  std::vector<float> k_data{
    -1.40307297e-03, 9.07780395e-06, 4.84838307e-01, -5.43047376e-02,
                            -1.40780103e-04,
                            1.25770375e-05,
                            1.04126692e+00,
                            7.67668605e-01,
                            -1.02884378e-05,
                            -1.41007011e-03,
                            1.02823459e-01,
                            -3.07415128e-01,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            1.00000000e+00,
                            -9.39000631e-04,
                            -7.65239349e-07,
                            1.14073277e+00,
                            4.46270645e-01,
                            1.04998052e-03,
                            1.91798881e-05,
                            2.06218868e-01,
                            7.42717385e-01,
                            1.48074005e-05,
                            -1.40855671e-03,
                            7.45946690e-02,
                            -3.16081315e-01,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            1.00000000e+00,
                            -7.0699735e-04,
                            4.2389297e-07,
                            -5.5183989e-01,
                            -5.3276348e-01,
                            -1.2281288e-03,
                            2.5626015e-05,
                            1.0212017e+00,
                            6.1102939e-01,
                            -2.2421273e-05,
                            -1.4170362e-03,
                            9.3639769e-02,
                            -3.0863306e-01,
                            0.0000000e+00,
                            0.0000000e+00,
                            0.0000000e+00,
                            1.0000000e+00,
                            2.2227580e-03,
                            2.5312484e-06,
                            -9.7261822e-01,
                            9.0684637e-02,
                            1.9360810e-04,
                            2.1347081e-05,
                            -1.0779887e+00,
                            -7.9227984e-01,
                            4.3742721e-06,
                            -2.2310747e-03,
                            1.0842450e-01,
                            -2.9406491e-01,
                            0.0000000e+00,
                            0.0000000e+00,
                            0.0000000e+00,
                            1.0000000e+00,
                            5.97175560e-04,
                            -5.88774265e-06,
                            -1.15893924e+00,
                            -4.49921310e-01,
                            -1.28312141e-03,
                            3.58297058e-07,
                            1.48300052e-01,
                            1.14334166e-01,
                            -2.80917516e-06,
                            -1.41527120e-03,
                            8.37693438e-02,
                            -2.36765608e-01,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            1.00000000e+00,
                            3.6048229e-04,
                            3.8333174e-06,
                            7.9871160e-01,
                            4.3321830e-01,
                            1.3671946e-03,
                            6.7484652e-06,
                            -8.4722507e-01,
                            1.9411178e-01,
                            7.5027779e-06,
                            -1.4139183e-03,
                            8.2083985e-02,
                            -2.4505949e-01,
                            0.0000000e+00,
                            0.0000000e+00,
                            0.0000000e+00,
                            1.0000000e+00

  };
  /* clang-format on */
  if (FLAGS_with_timestamp) {
    for (int i = 0; i < num_cams / 2 * 4 * 4; ++i) {
      k_data.push_back(k_data[i]);
    }
  }

  std::vector<float> timestamp(num_cams, 0.0f);

  // petrv2 inference, this is a fake input, you need to input real timestamp
  // timestampe will only affect Velocity predict.
  if (FLAGS_with_timestamp) {
    for (int i = num_cams / 2; i < num_cams; ++i) {
      timestamp[i] = 1.0f;
    }
  }

  std::vector<float> boxes;
  std::vector<int64_t> labels;
  std::vector<float> scores;
  run(predictor.get(), images_shape, input_data, k_shape, k_data,
      timestamp_shape, timestamp, &boxes, &scores, &labels,
      FLAGS_with_timestamp);
  // boxes 9个数据
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