#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <vector>

namespace splash::ops {

struct VisionLayout final {
  uint32_t depth = 27;
  uint32_t hiddenSize = 1152;
  uint32_t patchDimension = 1536;
  uint32_t intermediateSize = 4304;
  uint32_t paddedIntermediateSize = 4352;
  uint32_t mergedHiddenSize = 4608;
  uint32_t outputHiddenSize = 5120;
  uint32_t heads = 16;
  uint32_t headDimension = 72;
  uint32_t positionGridSide = 48;
  uint32_t patchSize = 16;
  uint32_t spatialMerge = 2;

  bool operator==(const VisionLayout &) const = default;
};

struct VisionAffine final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
};

struct VisionNorm final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
};

struct VisionBlock final {
  VisionNorm norm1;
  VisionAffine qkv;
  VisionAffine projection;
  VisionNorm norm2;
  VisionAffine upProjection;
  VisionAffine downProjection;
};

struct VisionWeights final {
  VisionLayout layout;
  VisionAffine patchEmbedding;
  metal::MetalBuffer positionTable;
  std::vector<VisionBlock> blocks;
  VisionNorm mergerNorm;
  VisionAffine mergerUpProjection;
  VisionAffine mergerDownProjection;
};

// Input geometry shared by the frontend and runtime: RGB patch size and the
// maximum number of patches per image.
inline constexpr uint32_t kImagePatchPixels = 16;
inline constexpr uint32_t kImageChannels = 3;
inline constexpr uint32_t kMaximumImagePatches = 16384;
[[nodiscard]] constexpr uint64_t imagePixelBytes(uint32_t gridHeight,
                                                 uint32_t gridWidth) noexcept {
  return uint64_t{gridHeight} * kImagePatchPixels * gridWidth *
         kImagePatchPixels * kImageChannels;
}
// Patch-grid sides are multiples of the 2x2 spatial merge. Each merged token
// covers four patches and follows row-major order over the merged grid.
struct ImageGrid final {
  uint32_t height = 0; // patches
  uint32_t width = 0;

  [[nodiscard]] bool valid() const noexcept {
    return height >= 2 && width >= 2 && height % 2 == 0 && width % 2 == 0;
  }
  // Exact for any sides, so a grid past a patch limit cannot wrap under it.
  [[nodiscard]] uint64_t patches() const noexcept {
    return uint64_t{height} * width;
  }
  [[nodiscard]] uint32_t mergedTokens() const noexcept {
    return (height / 2) * (width / 2);
  }
  [[nodiscard]] uint64_t pixelBytes() const noexcept {
    return imagePixelBytes(height, width);
  }
};

// GPU vision encoder with one reusable scratch arena sized for maximumPatches.
// encode() appends one image's dispatches, from resized uint8 RGB pixels to
// bf16 language embeddings in the caller's buffer, so the arena can be
// released after encoding while later chunks copy the image's rows into their
// hidden rows (ops::RowCopy).
class Vision final {
public:
  // Scratch footprint before construction, so the caller can reserve it.
  [[nodiscard]] static uint64_t
  scratchBytes(const VisionLayout &layout, uint32_t maximumPatches);
  // Embedding buffers are written in whole merger tiles: the caller allocates
  // this many rows of layout.outputHiddenSize bf16 values per image.
  [[nodiscard]] static uint32_t embeddingRows(ImageGrid grid) noexcept;

  // Throws unless `model` holds every weight of the tower its layout names.
  Vision(metal::MetalBackend &backend, const VisionWeights &model,
         uint32_t maximumPatches);
  Vision(const Vision &) = delete;
  Vision &operator=(const Vision &) = delete;

  [[nodiscard]] uint64_t arenaBytes() const noexcept {
    return arena_.sizeBytes();
  }
  // The most patches an image it encodes may have.
  [[nodiscard]] uint32_t maximumPatches() const noexcept {
    return maximumPatches_;
  }

  void encode(metal::CommandGraph &graph, ImageGrid grid,
              const metal::MetalBuffer &pixels,
              const metal::MetalBuffer &embeddings) const;

private:
  enum class Scratch : uint32_t {
    Patches,
    RopeCos,
    RopeSin,
    Hidden,
    Normalized,
    Qkv,
    Intermediate,
    Queries,
    Keys,
    Values,
    Attention,
    Context,
    Count,
  };

  [[nodiscard]] const metal::MetalBuffer &scratch(Scratch tensor) const noexcept {
    return scratch_[static_cast<uint32_t>(tensor)];
  }
  void addGemm(metal::CommandGraph &graph, const char *pipeline,
               const metal::MetalBuffer &input,
               const VisionAffine &weights,
               const metal::MetalBuffer &output,
               const metal::MetalBuffer &residual, uint32_t outputSize,
               uint32_t inputSize, uint32_t rows, uint32_t tileRows,
               uint32_t tileColumns) const;
  void addNorm(metal::CommandGraph &graph, const metal::MetalBuffer &input,
               const VisionNorm &weights,
               const metal::MetalBuffer &output, uint32_t rows) const;

  const VisionWeights &model_;
  uint32_t maximumPatches_ = 0;
  metal::MetalBuffer arena_;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

} // namespace splash::ops
