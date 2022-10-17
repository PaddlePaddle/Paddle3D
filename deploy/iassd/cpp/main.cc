#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

// #include "paddle/include/paddle_inference_api.h"

// using paddle_infer::Config;
// using paddle_infer::CreatePredictor;
// using paddle_infer::Predictor;

// DEFINE_string(lidar_file, "", "Path of lidar file");

bool read_point(const std::string &file_path, const int &num_point_dim,
                void **buffer, int *num_points) {
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
  *num_points = file_size / sizeof(float) / num_point_dim;
  return true;
}