#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using paddle_infer::Predictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(image, "", "Image path to be predicted.");
DEFINE_bool(use_gpu, false, "Whether to use gpu.");
DEFINE_bool(use_trt, false, "Whether to use trt.");
DEFINE_int32(trt_precision, 0,
             "Precision type of tensorrt, 0: kFloat32, 1: kInt8, 2: kHalf");
DEFINE_bool(collect_dynamic_shape_info, false,
            "Whether to collect dynamic shape before using tensorrt");
DEFINE_string(dynamic_shape_file, "dynamic_shape_info.txt",
              "Path of a dynamic shape file for tensorrt");

void get_image(const std::string &image, float *data) {
  cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);

  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, cv::Size(1280, 384));
  // Normalize
  img.convertTo(img, CV_32F, 1.0 / 255, 0);

  std::vector<float> mean_values{0.485, 0.456, 0.406};
  std::vector<float> std_values{0.229, 0.224, 0.225};

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(img, rgbChannels);

  for (int i = 0; i < 3; ++i) {
    rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1 / std_values[i],
                             (0.0 - mean_values[i]) / std_values[i]);
  }

  cv::merge(rgbChannels, img);

  // from hwc to chw
  int rows = img.rows;
  int cols = img.cols;
  int chs = img.channels();

  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(
        img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  config.SetModel(FLAGS_model_file, FLAGS_params_file);

  if (FLAGS_use_gpu) {
    config.EnableUseGpu(1000, 0);
  } else {
    config.EnableMKLDNN();
  }

  if (FLAGS_collect_dynamic_shape_info) {
    config.CollectShapeRangeInfo(FLAGS_dynamic_shape_file);
  } else if (FLAGS_use_trt) {
    config.EnableTensorRtEngine(
        1 << 30, 1, 5, PrecisionType(FLAGS_trt_precision), false, false);
    config.EnableTunedTensorRtDynamicShape(FLAGS_dynamic_shape_file, true);
  }

  config.SwitchIrOptim(true);
  config.EnableMemoryOptim();

  return CreatePredictor(config);
}

void run(Predictor *predictor, const std::vector<float> &input_im_data,
         const std::vector<int> &input_im_shape,
         const std::vector<float> &input_K_data,
         const std::vector<int> &input_K_shape,
         const std::vector<float> &input_ratio_data,
         const std::vector<int> &input_ratio_shape,
         std::vector<float> &out_data) {
  int input_num = std::accumulate(input_im_shape.begin(), input_im_shape.end(),
                                  1, std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();

  auto input_im_handle = predictor->GetInputHandle(input_names[0]);
  input_im_handle->Reshape(input_im_shape);
  input_im_handle->CopyFromCpu(input_im_data.data());

  auto input_K_handle = predictor->GetInputHandle(input_names[1]);
  input_K_handle->Reshape(input_K_shape);
  input_K_handle->CopyFromCpu(input_K_data.data());

  auto input_ratio_handle = predictor->GetInputHandle(input_names[2]);
  input_ratio_handle->Reshape(input_ratio_shape);
  input_ratio_handle->CopyFromCpu(input_ratio_data.data());

  CHECK(predictor->Run());
  auto output_t = predictor->GetOutputHandle(output_names[0]);

  auto outshape = output_t->shape();
  int outsize = std::accumulate(outshape.begin(), outshape.end(), 1,
                                std::multiplies<int>());
  out_data.resize(outsize);

  output_t->CopyToCpu(out_data.data());
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_file == "" || FLAGS_params_file == "" || FLAGS_image == "") {
    std::cout << "Missing required parameter" << std::endl;
    std::cout << "Usage: " << std::string(argv[0])
              << " --model_file ${MODEL_FILE}"
              << "--params_file ${PARAMS_FILE} "
              << "--image ${TEST_IMAGE}" << std::endl;
    return -1;
  }

  auto predictor = InitPredictor();

  std::vector<int> input_im_shape = {1, 3, 384, 1280};
  std::vector<float> input_im_data(1 * 3 * 384 * 1280);
  get_image(FLAGS_image, input_im_data.data());

  std::vector<int> input_K_shape = {1, 3, 3};
  // Listed below are camera intrinsic parameter of the kitti dataset
  // If the model is trained on other datasets, please replace the relevant data
  std::vector<float> input_K_data = {
      721.53771973, 0., 609.55932617, 0., 721.53771973, 172.85400391, 0, 0, 1};
  std::vector<int> input_ratio_shape = {1, 2};
  std::vector<float> input_ratio_data(4, 4);

  std::vector<float> out_data;
  run(predictor.get(), input_im_data, input_im_shape, input_K_data,
      input_K_shape, input_ratio_data, input_ratio_shape, out_data);

  std::vector<std::vector<float>> results;
  for (int i = 0; i < out_data.size(); i += 14) {
    // item 1       :  class
    // item 2       :  observation angle α
    // item 3 ~ 6   :  box2d x1, y1, x2, y2
    // item 7 ~ 9   :  box3d h, w, l
    // item 10 ~ 12 :  box3d bottom center x, y, z
    // item 13      :  box3d yaw angle
    // item 14      :  score
    std::vector<float> vec(out_data.begin() + i, out_data.begin() + i + 14);
    results.push_back(vec);
  }

  for (const auto &res : results) {
    // Filter predictions with low scores
    if (res[13] <= 0.25) continue;
    for (const auto &item : res) {
      std::cout << item << " ";
    }
    std::cout << std::endl;
  }
}
