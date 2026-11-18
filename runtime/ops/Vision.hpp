#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <vector>

namespace splash::ops {

// A vision tower's geometry. The kernels compile two (kCompiledVision,
// Vision.cpp): the defaults, Qwen3.5's 27 blocks, and kRadioVisionLayout,
// Nemotron 3.5 Bolt's RADIO. outputHiddenSize, the width of the text model
// its merger projects into, is each family's, stated beside its target
// (Qwen3_8.hpp, Qwen3_6Moe.hpp).
struct VisionLayout final {
  uint32_t depth = 27;
  uint32_t hiddenSize = 1152;
  uint32_t patchDimension = 1536;
  uint32_t intermediateSize = 4304;
  uint32_t paddedIntermediateSize = 4352;
  uint32_t mergedHiddenSize = 4608;
  uint32_t outputHiddenSize = 0;
  uint32_t heads = 16;
  uint32_t headDimension = 72;
  uint32_t positionGridSide = 48;
  uint32_t patchSize = 16;
  uint32_t spatialMerge = 2;
  // Learned tokens the tower attends with the patches and drops before the
  // merger (RADIO's cls and registers).
  uint32_t prefixTokens = 0;
  // The merger's up-projection width.
  uint32_t mergerIntermediateSize = 4608;
  // RADIO's final LayerNorm (v.post_ln) on the residual before the merger
  // (Nemotron 3.5 Super's tower; Bolt's has none).
  bool finalNorm = false;

  bool operator==(const VisionLayout &) const = default;
};

// RADIO (C-RADIO ViT-H/16): 32 blocks 1280 wide, 16 heads of 80, MLP 5120,
// single-frame 768-value patches, a 128x128 position table, 10 prefix tokens,
// and a merger of RMS norm, 5120 -> 20480, ReLU squared, -> output.
inline constexpr VisionLayout kRadioVisionLayout{
    32, 1280, 768, 5120, 5120, 5120, 0, 16, 80, 128, 16, 2, 10, 20480};
// Whether a layout is RADIO's, whatever text model it projects into.
[[nodiscard]] constexpr bool isRadioVision(VisionLayout layout) noexcept {
  layout.outputHiddenSize = 0;
  layout.finalNorm = false;
  return layout == kRadioVisionLayout;
}

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

// bf16 weights. RADIO's patch embedding and merger have no biases (their bias
// buffers stay empty), and its merger norm is an RMS weight per merged value
// (mergerNorm.weight, no bias).
struct VisionWeights final {
  VisionLayout layout;
  VisionAffine patchEmbedding;
  metal::MetalBuffer positionTable;
  std::vector<VisionBlock> blocks;
  VisionNorm mergerNorm;
  VisionNorm finalNorm = {}; // RADIO's, when its layout has one
  // RADIO's video patch embedding, two frames of patches wide (no bias).
  VisionAffine videoPatchEmbedding = {};
  VisionAffine mergerUpProjection;
  VisionAffine mergerDownProjection;
  metal::MetalBuffer prefixTokens = {};
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
// covers four patches and follows row-major order over the merged grid. A
// video's tubelet is two frames of one grid, encoded into one grid's tokens.
struct ImageGrid final {
  uint32_t height = 0; // patches
  uint32_t width = 0;
  uint32_t frames = 1;

  [[nodiscard]] bool valid() const noexcept {
    return height >= 2 && width >= 2 && height % 2 == 0 && width % 2 == 0 && (frames == 1 || frames == 2);
  }
  // Exact for any sides, so a grid past a patch limit cannot wrap under it.
  [[nodiscard]] uint64_t patches() const noexcept {
    return uint64_t{height} * width;
  }
  [[nodiscard]] uint32_t mergedTokens() const noexcept {
    return (height / 2) * (width / 2);
  }
  [[nodiscard]] uint64_t pixelBytes() const noexcept {
    return imagePixelBytes(height, width) * frames;
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
  bool radio_ = false;
  metal::MetalBuffer arena_;
  // The bias of RADIO's bias-free GEMMs, zero for the widest of them: the
  // arena's end, which scratchBytes counts.
  metal::MetalBuffer zeros_;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

// EVS over a pruned video's tubelets (model::ImageSpan::pruned), from their
// encoded rows (Vision::encode, Vision::embeddingRows(grid) rows apart) and
// the bf16 rows of the text between them to the span's rows: the `kept`
// least redundant tubelet rows in order, each tubelet's after the text that
// precedes it (`cuts`, a u32 per tubelet), as the processor drops the pruned
// rows of the prompt it widened.
struct VideoPruningShape final {
  uint32_t tubelets = 0;
  ImageGrid grid;
  uint32_t width = 0;
  uint32_t kept = 0;
  uint32_t textTokens = 0;
};

class VideoPruning final {
public:
  // The f32 dissimilarities and u32 keep flags, one per tubelet row.
  [[nodiscard]] static uint64_t flagBytes(const VideoPruningShape &shape) noexcept;
  static void add(metal::CommandGraph &graph, const VideoPruningShape &shape,
                  const metal::MetalBuffer &encoded, const metal::MetalBuffer &text,
                  const metal::MetalBuffer &cuts, const metal::MetalBuffer &dissimilarity,
                  const metal::MetalBuffer &keep, const metal::MetalBuffer &output);
};

} // namespace splash::ops
