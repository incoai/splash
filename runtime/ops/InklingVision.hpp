#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Vision.hpp"

#include <cstdint>

namespace splash::ops {

// Inkling 2's vision tower (kInklingVisionLayout), from an image's pixels
// padded right and bottom to whole 40-pixel merge units (the frontend's
// preparation, server/images.py) to bf16 language embeddings:
//  - patchify: 20-pixel RGB patches normalized x / 127.5 - 1, one row each;
//  - their embedding (no bias) plus the learned 32x32 position table
//    resampled to the grid (bilinear, half-pixel), and a class token, its
//    embedding plus its position, after them;
//  - a LayerNorm (epsilon 1e-5) that starts the fp32 residual stream;
//  - 23 pre-norm blocks: attention of 16 heads of 64 with an axial 2D RoPE
//    (interleaved pairs, columns then rows, theta 10000) and an exact-GELU
//    MLP of 4096, each output scaled per channel (LayerScale) before its
//    residual add;
//  - the patch rows without the class token's, four neighbours (2x2) to a
//    row of 4096, into the projector: 4096 -> GELU -> the text width.
// Every matrix is bf16 (a Q8_0 projector's, rounded to it when its image is
// written), activations bf16 between fp32 epilogues, the residual fp32. Its
// own kernels are shared/inkling_vision.metal's; its projections, LayerNorms
// and attention are the shared dense operators, as Qwen's tower's.
class InklingVision final : public ImageEncoder {
public:
  [[nodiscard]] static uint64_t scratchBytes(const VisionLayout &layout, uint32_t maximumPatches);

  // Throws unless `model` holds every weight of the tower its layout names.
  InklingVision(metal::MetalBackend &backend, const VisionWeights &model, uint32_t maximumPatches);
  InklingVision(const InklingVision &) = delete;
  InklingVision &operator=(const InklingVision &) = delete;

  [[nodiscard]] uint64_t arenaBytes() const noexcept override { return arena_.sizeBytes(); }
  [[nodiscard]] uint32_t maximumPatches() const noexcept override { return maximumPatches_; }

  void encode(metal::CommandGraph &graph, ImageGrid grid, const metal::MetalBuffer &pixels,
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
