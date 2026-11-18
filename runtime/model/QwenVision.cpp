#include "model/QwenVision.hpp"
#include "Checked.hpp"
#include "model/VisionLoader.hpp"

#include <string>
#include <utility>

namespace splash::model {
namespace {

ops::VisionAffine readAffine(WeightFile &file, uint32_t outputSize,
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

ops::VisionNorm readNorm(WeightFile &file, uint32_t width,
                         std::string_view label) {
  const uint64_t bytes = checkedMultiply<WeightStoreError>(
      width, kBFloat16Bytes, "vision norm bytes");
  return {file.section(bytes, label), file.section(bytes, label)};
}

// A bf16 tensor of rows x columns values.
metal::MetalBuffer readTensor(WeightFile &file, uint32_t rows, uint32_t columns, std::string_view label) {
  return file.section(checkedMultiply<WeightStoreError>(
                          checkedMultiply<WeightStoreError>(rows, columns, "vision tensor elements"), kBFloat16Bytes,
                          "vision tensor bytes"),
                      label);
}

// RADIO's tensors in the order VisionLoader.cpp plans them (planRadio): no
// patch or merger biases, the prefix tokens after the positions, the
// merger's RMS weight alone, and the video patch embedding last.
void readRadio(WeightFile &file, const ops::VisionLayout &layout, ops::VisionWeights &tensors) {
  tensors.patchEmbedding = {readTensor(file, layout.hiddenSize, layout.patchDimension, "patch-embed"), {}};
  tensors.positionTable =
      readTensor(file, layout.positionGridSide * layout.positionGridSide, layout.hiddenSize, "position-table");
  tensors.prefixTokens = readTensor(file, layout.prefixTokens, layout.hiddenSize, "prefix-tokens");
  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::VisionBlock block;
    block.norm1 = readNorm(file, layout.hiddenSize, "norm1");
    block.qkv = readAffine(file, 3 * layout.hiddenSize, layout.hiddenSize, "qkv");
    block.projection = readAffine(file, layout.hiddenSize, layout.hiddenSize, "proj");
    block.norm2 = readNorm(file, layout.hiddenSize, "norm2");
    block.upProjection = readAffine(file, layout.intermediateSize, layout.hiddenSize, "fc1");
    block.downProjection = readAffine(file, layout.hiddenSize, layout.intermediateSize, "fc2");
    tensors.blocks.push_back(std::move(block));
  }
  if (layout.finalNorm) tensors.finalNorm = readNorm(file, layout.hiddenSize, "final-norm");
  tensors.mergerNorm = {readTensor(file, layout.mergedHiddenSize, 1, "merger-norm"), {}};
  tensors.mergerUpProjection = {
      readTensor(file, layout.mergerIntermediateSize, layout.mergedHiddenSize, "merger-fc1"), {}};
  tensors.mergerDownProjection = {
      readTensor(file, layout.outputHiddenSize, layout.mergerIntermediateSize, "merger-fc2"), {}};
  tensors.videoPatchEmbedding = {readTensor(file, layout.hiddenSize, 2 * layout.patchDimension, "video-patch-embed"), {}};
}

QwenVisionWeights readVision(metal::MetalBackend &backend, WeightImages &images,
                             ImagePlan image, const ops::VisionLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  QwenVisionWeights result;
  result.tensors.layout = layout;
  result.tensors.blocks.reserve(layout.depth);

  WeightFile file = images.load(std::move(image));
  if (ops::isRadioVision(layout)) {
    readRadio(file, layout, result.tensors);
    file.finish();
    result.files.push_back(file.record());
    result.actualAllocatedBytes = metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
    return result;
  }
  result.tensors.patchEmbedding =
      readAffine(file, layout.hiddenSize, layout.patchDimension, "patch-embed");
  result.tensors.positionTable = file.section(
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
    result.tensors.blocks.push_back(std::move(block));
  }
  result.tensors.mergerNorm = readNorm(file, layout.hiddenSize, "merger-norm");
  result.tensors.mergerUpProjection = readAffine(
      file, layout.mergedHiddenSize, layout.mergedHiddenSize, "merger-fc1");
  result.tensors.mergerDownProjection = readAffine(
      file, layout.outputHiddenSize, layout.mergedHiddenSize, "merger-fc2");
  file.finish();
  result.files.push_back(file.record());

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

// The loader checked its layout when it was built.
QwenVisionWeights loadQwenVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                        const VisionLoader &source) {
  return readVision(backend, images, source.image(), source.layout());
}

} // namespace splash::model
