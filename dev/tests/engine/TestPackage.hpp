#pragma once

// Synthetic packed model packages: the weight files of a package of given
// layouts, each section at its aligned offset and zero, which
// loadModelPackage loads as it loads a Splash package.

#include "TestChecks.hpp"
#include "model/DFlashDraft.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightLayout.hpp"
#include "ops/Vision.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::test {

inline constexpr std::string_view kDraftLayerMagic = "MDFD0004";
inline constexpr std::string_view kTargetEmbeddingMagic = "MDFE0001";
inline constexpr std::string_view kTargetHeadMagic = "MDFL0002";
inline constexpr std::string_view kTargetLayerMagic = "MDFL0006";
inline constexpr std::string_view kVisionMagic = "MDFV0001";

inline uint64_t alignPacked(uint64_t value) {
  return (value + model::kWeightFileAlignment - 1) &
         ~(model::kWeightFileAlignment - 1);
}

inline uint64_t checkedProduct(uint64_t left, uint64_t right) {
  require(!left || right <= std::numeric_limits<uint64_t>::max() / left,
          "synthetic layout size overflow");
  return left * right;
}

inline uint64_t q4Bytes(uint32_t outputSize, uint32_t inputSize) {
  return checkedProduct(outputSize, inputSize) * 9 / 16;
}

inline void storeLittleEndian32(uint8_t *destination, uint32_t value) {
  destination[0] = static_cast<uint8_t>(value);
  destination[1] = static_cast<uint8_t>(value >> 8);
  destination[2] = static_cast<uint8_t>(value >> 16);
  destination[3] = static_cast<uint8_t>(value >> 24);
}

// Writes a weight file of the given magic, layer and type whose sections
// have the given sizes, and returns its size.
inline uint64_t writeWeightFile(const std::filesystem::path &path,
                                std::string_view magic, uint32_t layer,
                                uint32_t type,
                                std::span<const uint64_t> sections) {
  require(magic.size() == 8, "synthetic magic has the wrong size");
  std::filesystem::create_directories(path.parent_path());
  int descriptor = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                        0600);
  require(descriptor >= 0, "unable to create synthetic packed file");

  std::array<uint8_t, 16> header{};
  std::memcpy(header.data(), magic.data(), magic.size());
  storeLittleEndian32(header.data() + 8, layer);
  storeLittleEndian32(header.data() + 12, type);
  ssize_t written = pwrite(descriptor, header.data(), header.size(), 0);
  if (written != static_cast<ssize_t>(header.size())) {
    close(descriptor);
    throw std::runtime_error("unable to write synthetic packed header");
  }

  uint64_t offset = header.size();
  for (uint64_t bytes : sections) {
    require(bytes > 0, "synthetic section is empty");
    offset = alignPacked(offset) + bytes;
  }
  uint64_t fileBytes = alignPacked(offset);
  if (fileBytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
      ftruncate(descriptor, static_cast<off_t>(fileBytes)) != 0) {
    close(descriptor);
    throw std::runtime_error("unable to size synthetic packed file");
  }
  close(descriptor);
  return fileBytes;
}

inline std::vector<uint64_t>
targetLayerSections(const model::Qwen3_8Layout &layout, bool full) {
  constexpr uint64_t bf16 = 2;
  std::vector<uint64_t> result{
      uint64_t(layout.hiddenSize) * bf16,
      q4Bytes(full ? layout.packedFullWidth : layout.packedGdnWidth,
              layout.hiddenSize),
  };
  if (full) {
    result.insert(result.end(), {uint64_t(layout.attentionHeadDimension) * bf16,
                                 uint64_t(layout.attentionHeadDimension) * bf16,
                                 q4Bytes(layout.hiddenSize, layout.attentionWidth)});
  } else {
    result.insert(result.end(), {uint64_t(layout.convolutionDimension) * 4 * bf16,
                                 uint64_t(layout.gdnValueHeads) * 4,
                                 uint64_t(layout.gdnValueHeads) * bf16,
                                 uint64_t(layout.gdnHeadDimension) * bf16,
                                 q4Bytes(layout.hiddenSize, layout.attentionWidth)});
  }
  result.insert(result.end(), {uint64_t(layout.hiddenSize) * bf16,
                               q4Bytes(layout.intermediateSize, layout.hiddenSize),
                               q4Bytes(layout.intermediateSize, layout.hiddenSize),
                               q4Bytes(layout.hiddenSize, layout.intermediateSize)});
  return result;
}

inline std::vector<uint64_t>
draftLayerSections(const model::DFlashDraftLayout &layout) {
  constexpr uint64_t bf16 = 2;
  return {
      uint64_t(layout.hiddenSize) * bf16,
      uint64_t(4) * layout.hiddenSize * bf16,
      q4Bytes(layout.dynamicSize, layout.hiddenSize),
      q4Bytes(layout.qkvSize, layout.hiddenSize),
      uint64_t(layout.attentionHeadDimension) * bf16,
      uint64_t(layout.attentionHeadDimension) * bf16,
      q4Bytes(layout.hiddenSize, layout.attentionSize),
      uint64_t(layout.hiddenSize) * bf16,
      uint64_t(4) * layout.hiddenSize * bf16,
      q4Bytes(layout.dynamicSize, layout.hiddenSize),
      q4Bytes(layout.intermediateSize, layout.hiddenSize),
      q4Bytes(layout.intermediateSize, layout.hiddenSize),
      q4Bytes(layout.hiddenSize, layout.intermediateSize),
  };
}

inline std::vector<uint64_t> visionSections(const ops::VisionLayout &layout) {
  constexpr uint64_t bf16 = 2;
  auto affine = [&](uint64_t outputSize, uint64_t inputSize,
                    std::vector<uint64_t> &sections) {
    sections.push_back(outputSize * inputSize * bf16);
    sections.push_back(outputSize * bf16);
  };
  auto norm = [&](std::vector<uint64_t> &sections) {
    sections.push_back(uint64_t(layout.hiddenSize) * bf16);
    sections.push_back(uint64_t(layout.hiddenSize) * bf16);
  };
  std::vector<uint64_t> result;
  affine(layout.hiddenSize, layout.patchDimension, result);
  result.push_back(uint64_t(layout.positionGridSide) *
                   layout.positionGridSide * layout.hiddenSize * bf16);
  for (uint32_t block = 0; block < layout.depth; ++block) {
    norm(result);
    affine(uint64_t(3) * layout.hiddenSize, layout.hiddenSize, result);
    affine(layout.hiddenSize, layout.hiddenSize, result);
    norm(result);
    affine(layout.paddedIntermediateSize, layout.hiddenSize, result);
    affine(layout.hiddenSize, layout.paddedIntermediateSize, result);
  }
  norm(result);
  affine(layout.mergedHiddenSize, layout.mergedHiddenSize, result);
  affine(layout.outputHiddenSize, layout.mergedHiddenSize, result);
  return result;
}

// The bytes of each role's files.
struct SyntheticAccounting {
  uint64_t targetBytes = 0;
  uint64_t draftBytes = 0;
  uint64_t visionBytes = 0;
};

// Writes the files of a package of these layouts under root.
inline SyntheticAccounting
writeSyntheticPackage(const std::filesystem::path &root,
                      const model::Qwen3_8Layout &target,
                      const model::DFlashDraftLayout &draft,
                      const ops::VisionLayout &vision) {
  SyntheticAccounting result;
  for (uint32_t layer = 0; layer < target.layers; ++layer) {
    bool full = target.isFullAttentionLayer(layer);
    auto sections = targetLayerSections(target, full);
    result.targetBytes += writeWeightFile(
        root / "target" / ("layer-" + std::to_string(layer) + ".bin"),
        kTargetLayerMagic, layer, full ? 1U : 0U, sections);
  }
  std::array<uint64_t, 2> headSections{
      uint64_t(target.hiddenSize) * 2,
      q4Bytes(target.vocabularySize, target.hiddenSize),
  };
  result.targetBytes += writeWeightFile(root / "target/head.bin",
                                        kTargetHeadMagic, target.layers, 2,
                                        headSections);
  uint64_t embeddingElements =
      uint64_t(target.vocabularySize) * target.hiddenSize;
  std::array<uint64_t, 3> embeddingSections{
      embeddingElements / 2,
      embeddingElements / 32,
      embeddingElements / 32,
  };
  result.targetBytes += writeWeightFile(
      root / "target/embedding.bin", kTargetEmbeddingMagic,
      target.vocabularySize, target.hiddenSize, embeddingSections);

  for (uint32_t layer = 0; layer < draft.layers; ++layer) {
    auto sections = draftLayerSections(draft);
    result.draftBytes += writeWeightFile(
        root / "draft" / ("layer-" + std::to_string(layer) + ".bin"),
        kDraftLayerMagic, layer, 0, sections);
  }
  uint64_t codebookBytes =
      uint64_t(draft.vocabularySize) * draft.selectorRank * 2;
  std::array<uint64_t, 6> modelSections{
      q4Bytes(draft.hiddenSize, draft.targetHiddenSize),
      uint64_t(draft.hiddenSize) * 2,
      uint64_t(draft.hiddenSize) * 2,
      q4Bytes(draft.selectorRank, draft.hiddenSize),
      codebookBytes,
      codebookBytes,
  };
  result.draftBytes += writeWeightFile(root / "draft/model.bin",
                                       kDraftLayerMagic, draft.layers, 1,
                                       modelSections);
  auto sections = visionSections(vision);
  result.visionBytes += writeWeightFile(root / "vision/model.bin",
                                        kVisionMagic, vision.depth, 0,
                                        sections);
  return result;
}

} // namespace splash::test
