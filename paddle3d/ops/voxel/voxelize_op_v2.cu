#include <vector>

#include "paddle/extension.h"

#define CHECK_INPUT(x) \
  PD_CHECK(x.is_gpu() || x.is_gpu_pinned(), #x " must be a GPU Tensor.")

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0) {
      c_x = 0;
    } else if (c_x >= grid_x) {
      c_x = grid_x - 1;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0) {
      c_y = 0;
    } else if (c_y >= grid_y) {
      c_y = grid_y - 1;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0) {
      c_z = 0;
    } else if (c_z >= grid_z) {
      c_z = grid_z - 1;
    }

    coors_offset[0] = c_z;
    coors_offset[1] = c_y;
    coors_offset[2] = c_x;
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          return;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    // if (coor[i][0] == -1)
    //    continue;
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

std::vector<paddle::Tensor> hard_voxelize_gpu(
    const paddle::Tensor& points, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);

  const int NDim = 3;
  const int num_points = points.shape()[0];
  const int num_features = points.shape()[1];

  auto voxels = paddle::full({max_voxels, max_points, num_features}, 0.0,
                             paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto coors = paddle::full({max_voxels, 3}, 0.0, paddle::DataType::INT32,
                            paddle::GPUPlace());
  auto num_points_per_voxel = paddle::full(
      {max_voxels}, 0.0, paddle::DataType::INT32, paddle::GPUPlace());

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  auto temp_coors = paddle::full({num_points, NDim}, 0, paddle::DataType::INT32,
                                 paddle::GPUPlace());

  dim3 grid(std::min(1 + ((num_points - 1) / 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<data_t, int>
            <<<grid, block, 0, points.stream()>>>(
                points.data<data_t>(), temp_coors.data<int>(), voxel_x, voxel_y,
                voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
                coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
                num_features, NDim);
      }));

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx = paddle::full(
      {
          num_points,
      },
      -1, paddle::DataType::INT32, paddle::GPUPlace());
  auto point_to_voxelidx = paddle::full(
      {
          num_points,
      },
      -1, paddle::DataType::INT32, paddle::GPUPlace());

  dim3 map_grid(std::min(1 + ((num_points - 1) / 512), 4096));
  dim3 map_block(512);
  PD_DISPATCH_INTEGRAL_TYPES(
      temp_coors.type(), "determin_duplicate", ([&] {
        point_to_voxelidx_kernel<int>
            <<<map_grid, map_block, 0, temp_coors.stream()>>>(
                temp_coors.data<int>(), point_to_voxelidx.data<int>(),
                point_to_pointidx.data<int>(), max_points, max_voxels,
                num_points, NDim);
      }));

  // 3. determin voxel num and voxel's coor index
  // make the logic in the CUDA device could accelerate about 10 times
  auto coor_to_voxelidx = paddle::full(
      {
          num_points,
      },
      -1, paddle::DataType::INT32, paddle::GPUPlace());
  auto voxel_num = paddle::full(
      {
          1,
      },
      0, paddle::DataType::INT32,
      paddle::GPUPlace());  // must be zero from the begining

  PD_DISPATCH_INTEGRAL_TYPES(
      temp_coors.type(), "determin_duplicate", ([&] {
        determin_voxel_num<int><<<1, 1, 0, temp_coors.stream()>>>(
            num_points_per_voxel.data<int>(), point_to_voxelidx.data<int>(),
            point_to_pointidx.data<int>(), coor_to_voxelidx.data<int>(),
            voxel_num.data<int>(), max_points, max_voxels, num_points);
      }));

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(1 + ((pts_output_size - 1) / 512), 4096));
  dim3 cp_block(512);
  PD_DISPATCH_FLOATING_TYPES(points.type(), "assign_point_to_voxel", ([&] {
                               assign_point_to_voxel<float, int>
                                   <<<cp_grid, cp_block, 0, points.stream()>>>(
                                       pts_output_size, points.data<float>(),
                                       point_to_voxelidx.data<int>(),
                                       coor_to_voxelidx.data<int>(),
                                       voxels.data<float>(), max_points,
                                       num_features, num_points, NDim);
                             }));

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(std::min(1 + ((coors_output_size - 1) / 512), 4096));
  dim3 coors_cp_block(512);
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_point_to_voxel", ([&] {
        assign_voxel_coors<float, int>
            <<<coors_cp_grid, coors_cp_block, 0, points.stream()>>>(
                coors_output_size, temp_coors.data<int>(),
                point_to_voxelidx.data<int>(), coor_to_voxelidx.data<int>(),
                coors.data<int>(), num_points, NDim);
      }));

  return {voxels, coors, num_points_per_voxel, voxel_num};
}
