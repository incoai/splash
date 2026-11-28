#include "model/QwenVision.hpp"
#include "Checked.hpp"

#include <string>
#include <utility>

namespace splash::model {
namespace {

ops::DenseWeights readAffine(WeightFile &file, uint32_t outputSize,
                             uint32_t inputSize, std::string_view label) {
  return {
      file.section(checkedMultiply<WeightStoreError>(
                       checkedMultiply<WeightStoreError>(
                           outputSize, inputSize, "vision weight elements"),
                       kBFloat16Bytes, "vision weight bytes"),
                   label),
      file.section(checkedMultiply<WeightStoreError>(
                       outputSize, kBFloat16Bytes, "vision bias bytes"),
                   label),
  };
}

ops::LayerNormWeights readNorm(WeightFile &file, uint32_t width,
                               std::string_view label) {
  const uint64_t bytes = checkedMultiply<WeightStoreError>(
      width, kBFloat16Bytes, "vision norm bytes");
  return {file.section(bytes, label), file.section(bytes, label)};
}

} // namespace

ops::VisionWeights readQwenVision(WeightFile &file, const ops::VisionLayout &layout) {
  ops::VisionWeights result;
  result.layout = layout;
  result.blocks.reserve(layout.depth);
  result.patchEmbedding =
      readAffine(file, layout.hiddenSize, layout.patchDimension, "patch-embed");
  result.positionTable = file.section(
      checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(
              checkedMultiply<WeightStoreError>(layout.positionGridSide,
                                                layout.positionGridSide,
                                                "vision position count"),
              layout.hiddenSize, "vision position elements"),
          kBFloat16Bytes, "vision position bytes"),
      "position-table");
  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::VisionBlock block;
    block.norm1 = readNorm(file, layout.hiddenSize, "norm1");
    block.qkv = readAffine(file, 3 * layout.hiddenSize, layout.hiddenSize, "qkv");
    block.projection =
        readAffine(file, layout.hiddenSize, layout.hiddenSize, "proj");
    block.norm2 = readNorm(file, layout.hiddenSize, "norm2");
    block.upProjection = readAffine(file, layout.paddedIntermediateSize,
                                    layout.hiddenSize, "fc1");
    block.downProjection = readAffine(file, layout.hiddenSize,
                                      layout.paddedIntermediateSize, "fc2");
    result.blocks.push_back(std::move(block));
  }
  result.mergerNorm = readNorm(file, layout.hiddenSize, "merger-norm");
  result.mergerUpProjection = readAffine(
      file, layout.mergedHiddenSize, layout.mergedHiddenSize, "merger-fc1");
  result.mergerDownProjection = readAffine(
      file, layout.outputHiddenSize, layout.mergedHiddenSize, "merger-fc2");
  return result;
}

} // namespace splash::model
