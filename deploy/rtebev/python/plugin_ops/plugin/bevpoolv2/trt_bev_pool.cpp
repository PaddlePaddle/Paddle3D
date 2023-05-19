// Copyright (c) OpenMMLab. All rights reserved.
#include "trt_bev_pool.hpp"

#include <assert.h>

#include <chrono>

#include "trt_bev_pool_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"bev_pool_v2"};
}  // namespace

TRTBEVPoolV2::TRTBEVPoolV2(const std::string &name, int outWidth, int outHeight)
    : TRTPluginBase(name), mOutWidth(outWidth), mOutHeight(outHeight) {}

TRTBEVPoolV2::TRTBEVPoolV2(const std::string name, const void *data,
                           size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mOutWidth);
  deserialize_value(&data, &length, &mOutHeight);
}

nvinfer1::IPluginV2DynamicExt *TRTBEVPoolV2::clone() const TRT_NOEXCEPT {
  TRTBEVPoolV2 *plugin = new TRTBEVPoolV2(mLayerName, mOutWidth, mOutHeight);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTBEVPoolV2::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // input[0] == depth
  // input[1] == feat
  // input[2] == ranks_depth
  // input[3] == ranks_feat
  // input[4] == ranks_bev
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = exprBuilder.constant(1);  // Todo support batch>1
  ret.d[1] = exprBuilder.constant(mOutHeight);
  ret.d[2] = exprBuilder.constant(mOutWidth);
  ret.d[3] = inputs[1].d[3];
  return ret;
}

bool TRTBEVPoolV2::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  // input[0] == depth->kFLOAT
  // input[1] == feat->kFLOAT
  // input[2] == ranks_depth->kINT32
  // input[3] == ranks_feat->kINT32
  // input[4] == ranks_bev->kINT32
  // input[5] == interval_starts->kINT32
  // input[6] == interval_lengths->kINT32
  // output[0] == bev_feat->kFLOAT
  if (pos == 0 || pos == 1 || pos == 7) {
    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return (ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  }
}

void TRTBEVPoolV2::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments

  ASSERT(nbInputs == 7);
  ASSERT(nbOutputs == 1);
}

size_t TRTBEVPoolV2::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                      int nbInputs,
                                      const nvinfer1::PluginTensorDesc *outputs,
                                      int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int TRTBEVPoolV2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                          const nvinfer1::PluginTensorDesc *outputDesc,
                          const void *const *inputs, void *const *outputs,
                          void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  nvinfer1::Dims feat_dims = inputDesc[1].dims;      // bnhwc
  nvinfer1::Dims interval_dims = inputDesc[5].dims;  // n
  nvinfer1::Dims out_dims = outputDesc[0].dims;      // bhwc
  auto data_type = inputDesc[0].type;
  int num_points =
      out_dims.d[0] * out_dims.d[1] * out_dims.d[2] * out_dims.d[3];
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      bev_pool_v2_set_zero(num_points, (float *)outputs[0]);
      bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (float *)inputs[0],
                  (float *)inputs[1], (int *)inputs[2], (int *)inputs[3],
                  (int *)inputs[4], (int *)inputs[5], (int *)inputs[6],
                  (float *)outputs[0], stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TRTBEVPoolV2::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTBEVPoolV2::getPluginType() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TRTBEVPoolV2::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TRTBEVPoolV2::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTBEVPoolV2::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mOutWidth) + serialized_size(mOutHeight);
}

void TRTBEVPoolV2::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mOutWidth);
  serialize_value(&buffer, mOutHeight);
}

////////////////////// creator /////////////////////////////

TRTBEVPoolV2Creator::TRTBEVPoolV2Creator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>(
      {nvinfer1::PluginField("output_z"),
       nvinfer1::PluginField("output_height"),
       nvinfer1::PluginField("output_width")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTBEVPoolV2Creator::getPluginName() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TRTBEVPoolV2Creator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  int outWidth = 128;
  int outHeight = 128;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("output_height") == 0) {
      outHeight = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("output_width") == 0) {
      outWidth = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }
  ASSERT(outHeight > 0);
  ASSERT(outWidth > 0);

  TRTBEVPoolV2 *plugin = new TRTBEVPoolV2(name, outWidth, outHeight);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TRTBEVPoolV2(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTBEVPoolV2Creator);
}  // namespace mmdeploy
