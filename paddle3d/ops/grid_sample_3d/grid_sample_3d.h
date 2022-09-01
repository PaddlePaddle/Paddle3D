//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
//

#ifndef GRID_SAMPLE_3D_H
#define GRID_SAMPLE_3D_H

#include <cassert>
#include <cmath>
#include <vector>

#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__

enum class Mode { bilinear, nearest };

enum class PaddingMode { zeros, border, reflect };

namespace {}
#endif
