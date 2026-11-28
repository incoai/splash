#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/DenseLinear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace splash::ops {

// The vision towers whose kernels Splash holds, each behind its encoder
// (ImageEncoder): Qwen3.5's (ops::Vision), a ViT over 16-pixel patches of two
// frames whose merger has a LayerNorm of its own; and Inkling 2's
// (ops::InklingVision), a ViT with a class token and LayerScale over
// 20-pixel patches of one frame, merged into a projector without biases.
enum class VisionTower : uint8_t { Qwen, Inkling };

// A vision tower's geometry. The defaults are Qwen3.5's 27 blocks;
// outputHiddenSize, the width of the text model its merger projects into, is
// each family's, its target's hidden size (visionLayout), and a family's
// layout names its tower (QwenHybridLayout::visionTower).
// paddedPatchDimension is the patch embedding's input width as its image
// stores it, padded with zero columns to whole GEMM chunks of 128.
struct VisionLayout final {
  VisionTower tower = VisionTower::Qwen;
  uint32_t depth = 27;
  uint32_t hiddenSize = 1152;
  uint32_t patchDimension = 1536;
  uint32_t paddedPatchDimension = 1536;
  uint32_t intermediateSize = 4304;
  uint32_t paddedIntermediateSize = 4352;
  uint32_t mergedHiddenSize = 4608;
  uint32_t outputHiddenSize = 0;
  uint32_t heads = 16;
  uint32_t headDimension = 72;
  uint32_t positionGridSide = 48;
  uint32_t patchSize = 16;
  uint32_t spatialMerge = 2;

  bool operator==(const VisionLayout &) const = default;
};

// Inkling 2's tower (clip.vision.projector_type inkling2v): 23 blocks of
// width 1024, 16 heads of 64, an MLP of 4096, 20-pixel RGB patches (1200
// values, stored padded to 1280), a learned 32x32 position grid, and a 2x2
// merge into a projector whose hidden width is the merged one, 4096.
inline constexpr VisionLayout kInklingVisionLayout{.tower = VisionTower::Inkling,
                                                   .depth = 23,
                                                   .hiddenSize = 1024,
                                                   .patchDimension = 1200,
                                                   .paddedPatchDimension = 1280,
                                                   .intermediateSize = 4096,
                                                   .paddedIntermediateSize = 4096,
                                                   .mergedHiddenSize = 4096,
                                                   .heads = 16,
                                                   .headDimension = 64,
                                                   .positionGridSide = 32,
                                                   .patchSize = 20,
                                                   .spatialMerge = 2};

// The layout of `tower` whose merger projects into a text model of
// `outputHiddenSize`.
[[nodiscard]] constexpr VisionLayout visionLayout(VisionTower tower, uint32_t outputHiddenSize) noexcept {
  VisionLayout layout = tower == VisionTower::Inkling ? kInklingVisionLayout : VisionLayout{};
  layout.outputHiddenSize = outputHiddenSize;
  return layout;
}

// A block's weights, bf16, or an Inkling projector's projections Q8_0
// (DenseWeights::quantized). Inkling's tower scales its attention's
// and MLP's outputs per channel before each residual add (LayerScale, its
// output projections' DenseWeights::scale); Qwen's has no scales.
struct VisionBlock final {
  LayerNormWeights norm1;
  DenseWeights qkv;
  DenseWeights projection;
  LayerNormWeights norm2;
  DenseWeights upProjection;
  DenseWeights downProjection;
};

// A tower's weights, bf16 but for an Inkling projector's Q8_0 projections.
// Qwen's merger norms the blocks' output
// (mergerNorm). Inkling's prepends a class token, its embedding and its
// position as two rows (classToken), and norms the embeddings before the
// blocks (inputNorm); its merger projections have zero biases, as its patch
// embedding has, and its image stores them so.
struct VisionWeights final {
  VisionLayout layout;
  DenseWeights patchEmbedding;
  metal::MetalBuffer positionTable;
  std::vector<VisionBlock> blocks;
  LayerNormWeights mergerNorm;
  DenseWeights mergerUpProjection;
  DenseWeights mergerDownProjection;
  metal::MetalBuffer classToken{};
  LayerNormWeights inputNorm{};
};

// Input geometry shared by the frontend and runtime: RGB pixels in square
// patches of the tower's side (Qwen's 16, the protocol's default, or
// Inkling's 20), and the maximum number of patches per image.
inline constexpr uint32_t kImagePatchPixels = 16;
inline constexpr uint32_t kImageChannels = 3;
inline constexpr uint32_t kMaximumImagePatches = 16384;
[[nodiscard]] constexpr uint64_t imagePixelBytes(uint32_t gridHeight, uint32_t gridWidth,
                                                 uint32_t patchPixels) noexcept {
  return uint64_t{gridHeight} * patchPixels * gridWidth * patchPixels * kImageChannels;
}
// Patch-grid sides are multiples of the 2x2 spatial merge. Each merged token
// covers four patches and follows row-major order over the merged grid.
struct ImageGrid final {
  uint32_t height = 0; // patches
  uint32_t width = 0;
  // The image's own pixels at its top left, before the padding to whole
  // tokens (zero: the whole grid), which an Inkling tower's encoder takes as
  // its reference pads them.
  uint32_t contentHeight = 0;
  uint32_t contentWidth = 0;

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
  // The bytes of its pixels in patches of `patchPixels` a side.
  [[nodiscard]] uint64_t pixelBytes(uint32_t patchPixels) const noexcept {
    return imagePixelBytes(height, width, patchPixels);
  }
};

// A GPU vision encoder with one reusable scratch arena sized for
// maximumPatches. encode() appends one image's dispatches, from its uint8 RGB
// pixels in the tower's patches to bf16 language embeddings in the caller's
// buffer, a row of layout.outputHiddenSize per merged token, so the arena can
// be released after encoding while later chunks copy the image's rows into
// their hidden rows (ops::RowCopy). The runtime holds whichever tower its
// model has behind this interface.
class ImageEncoder {
public:
  virtual ~ImageEncoder() = default;

  // The scratch footprint of the layout's tower's encoder before
  // construction, so the caller can reserve it.
  [[nodiscard]] static uint64_t scratchBytes(const VisionLayout &layout, uint32_t maximumPatches);
  // The encoder of the tower the weights' layout names; throws unless they
  // hold every weight of it.
  [[nodiscard]] static std::shared_ptr<ImageEncoder> make(metal::MetalBackend &backend, const VisionWeights &model,
                                                          uint32_t maximumPatches);

  // Its scratch arena as Metal allocates it (MetalBuffer::allocatedBytes).
  [[nodiscard]] virtual uint64_t arenaBytes() const noexcept = 0;
  // The most patches an image it encodes may have.
  [[nodiscard]] virtual uint32_t maximumPatches() const noexcept = 0;
  virtual void encode(metal::CommandGraph &graph, ImageGrid grid, const metal::MetalBuffer &pixels,
                      const metal::MetalBuffer &embeddings) const = 0;
};

// Qwen3.5's tower, from resized pixels. Its projections, LayerNorms and
// attention are the shared dense operators (ops/DenseLinear.hpp,
// ops/Normalization.hpp, ops/DenseAttention.hpp).
class Vision final : public ImageEncoder {
public:
  // Scratch footprint before construction, so the caller can reserve it.
  [[nodiscard]] static uint64_t
  scratchBytes(const VisionLayout &layout, uint32_t maximumPatches);

  // Throws unless `model` holds every weight of the tower its layout names.
  Vision(metal::MetalBackend &backend, const VisionWeights &model,
         uint32_t maximumPatches);
  Vision(const Vision &) = delete;
  Vision &operator=(const Vision &) = delete;

  [[nodiscard]] uint64_t arenaBytes() const noexcept override {
    return arena_.allocatedBytes();
  }
  [[nodiscard]] uint32_t maximumPatches() const noexcept override {
    return maximumPatches_;
  }

  void encode(metal::CommandGraph &graph, ImageGrid grid,
              const metal::MetalBuffer &pixels,
              const metal::MetalBuffer &embeddings) const override;

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
    Context,
    Count,
  };

  [[nodiscard]] const metal::MetalBuffer &scratch(Scratch tensor) const noexcept {
    return scratch_[static_cast<uint32_t>(tensor)];
  }
  const VisionWeights &model_;
  uint32_t maximumPatches_ = 0;
  DenseLinear dense_;
  metal::MetalBuffer arena_;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

} // namespace splash::ops
