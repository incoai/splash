#include "Vision.hpp"

#include "Checked.hpp"
#include "metal/abi/Vision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace splash::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint32_t kGemmRowTile = 64;
constexpr uint32_t kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32;
constexpr uint32_t kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
constexpr uint32_t kQueryTile = 64;
constexpr uint32_t kQkDimension = 80; // head dimension 72 padded to 16
constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;

// The one vision tower the kernels are specialized for.
constexpr VisionLayout kTower{};
static_assert(kTower.headDimension + 8 == kQkDimension && kTower.hiddenSize % kGemmColumnTile == 0 &&
                  kTower.paddedIntermediateSize % kGemmColumnTile == 0 &&
                  kTower.mergedHiddenSize % kMergerColumnTile == 0 &&
                  kTower.patchDimension % kGemmColumnTile == 0,
              "vision kernels' tile assumptions");

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

// Scratch tensor byte sizes for one encoder sized to maximumPatches. GEMM
// tiles read and write whole 64-row tiles, attention reads whole 128-key
// tiles, and the merger reads the normalized rows as (patches / 4, 4608).
std::array<uint64_t, 12> scratchLayout(const VisionLayout &layout,
                                       uint32_t maximumPatches) {
  const uint64_t rows = roundUp(maximumPatches, kGemmRowTile);
  const uint64_t padded = roundUp(maximumPatches, kKeyTile);
  const uint64_t mergedRows = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  return {
      rows * layout.patchDimension * kBf16Bytes,                 // Patches
      uint64_t{maximumPatches} * layout.headDimension * kFloatBytes, // RopeCos
      uint64_t{maximumPatches} * layout.headDimension * kFloatBytes, // RopeSin
      rows * hidden * kFloatBytes,                               // Hidden
      std::max(rows * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Qkv
      rows * layout.paddedIntermediateSize * kBf16Bytes,         // Intermediate
      headRows * kQkDimension * kBf16Bytes,                      // Queries
      headRows * kQkDimension * kBf16Bytes,                      // Keys
      headRows * layout.headDimension * kBf16Bytes,              // Values
      headRows * layout.headDimension * kBf16Bytes,              // Attention
      rows * hidden * kBf16Bytes,                                // Context
  };
}

void requireLayout(const VisionLayout &layout) {
  // Packages share the same vision tower; only the language-space projection
  // width varies with the text model.
  VisionLayout tower = layout;
  tower.outputHiddenSize = kTower.outputHiddenSize;
  if (tower != kTower || !layout.outputHiddenSize || layout.outputHiddenSize % kMergerColumnTile) {
    throw std::invalid_argument(
        "vision kernels are specialized for the Qwen3.5 27-block tower");
  }
}

} // namespace

uint64_t Vision::scratchBytes(const VisionLayout &layout,
                              uint32_t maximumPatches) {
  requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4) {
    throw std::invalid_argument(
        "vision scratch must cover a positive multiple of four patches");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vision scratch byte count overflows");
    }
    total += aligned;
  }
  return total;
}

uint32_t Vision::embeddingRows(ImageGrid grid) noexcept {
  return roundUp(grid.mergedTokens(), kMergerRowTile);
}

Vision::Vision(metal::MetalBackend &backend, const VisionWeights &model,
               uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  // New backend buffers are zero-filled, so padding rows read by whole tiles
  // start finite.
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared,
                                  "vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (cursor != total)
    throw std::logic_error("vision scratch arena mismatch");
}

void Vision::addGemm(CommandGraph &graph, const char *pipeline,
                     const MetalBuffer &input, const VisionAffine &weights,
                     const MetalBuffer &output, const MetalBuffer &residual,
                     uint32_t outputSize, uint32_t inputSize, uint32_t rows,
                     uint32_t tileRows, uint32_t tileColumns) const {
  graph.add(pipeline, {input, weights.weight, weights.bias, output, residual},
            VisionGemmParams{outputSize, inputSize},
            {(rows + tileRows - 1) / tileRows, outputSize / tileColumns, 1});
}

void Vision::addNorm(CommandGraph &graph, const MetalBuffer &input,
                     const VisionNorm &weights, const MetalBuffer &output,
                     uint32_t rows) const {
  graph.add("vision_layer_norm", {input, weights.weight, weights.bias, output},
            VisionNormParams{model_.layout.hiddenSize}, {rows, 1, 1});
}

void Vision::encode(CommandGraph &graph, ImageGrid grid,
                    const MetalBuffer &pixels,
                    const MetalBuffer &embeddings) const {
  const VisionLayout &layout = model_.layout;
  if (!grid.valid() || grid.patches() > maximumPatches_) {
    throw std::invalid_argument("image grid exceeds the vision encoder");
  }
  const auto tokens = static_cast<uint32_t>(grid.patches());
  if (!pixels || pixels.sizeBytes() < grid.pixelBytes()) {
    throw std::invalid_argument("image pixels do not cover the grid");
  }
  const uint64_t embeddingBytes = uint64_t{embeddingRows(grid)} *
                                  layout.outputHiddenSize * kBf16Bytes;
  if (!embeddings || embeddings.sizeBytes() < embeddingBytes) {
    throw std::invalid_argument("embedding buffer is too small for the grid");
  }

  const uint32_t padded = roundUp(tokens, kKeyTile);
  const uint32_t merged = grid.mergedTokens();
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionQkvParams qkvParams{tokens, padded};
  const VisionAttentionParams attentionParams{
      tokens, padded, 1.0F / std::sqrt(static_cast<float>(layout.headDimension))};
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &qkv = scratch(Scratch::Qkv);
  const MetalBuffer &context = scratch(Scratch::Context);

  graph.add("vision_patchify", {pixels, scratch(Scratch::Patches)}, gridParams,
            {tokens, 1, 1});
  // The positions start the residual stream the patch embedding adds to.
  graph.add("vision_prepare_positions",
            {model_.positionTable, hidden, scratch(Scratch::RopeCos),
             scratch(Scratch::RopeSin)},
            gridParams, {roundUp(tokens, kGemmRowTile), 1, 1});
  addGemm(graph, "vision_gemm_m64n128_residual", scratch(Scratch::Patches),
          model_.patchEmbedding, hidden, hidden, layout.hiddenSize,
          layout.patchDimension, tokens, kGemmRowTile, kGemmColumnTile);

  for (const VisionBlock &block : model_.blocks) {
    addNorm(graph, hidden, block.norm1, normalized, tokens);
    addGemm(graph, "vision_gemm_m64n128", normalized, block.qkv, qkv, hidden,
            3 * layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    graph.add("vision_qkv_prepare",
              {qkv, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin),
               scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values)},
              qkvParams, {padded, 1, 1});
    graph.add("vision_attention",
              {scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values), scratch(Scratch::Attention)},
              attentionParams,
              {(tokens + kQueryTile - 1) / kQueryTile, layout.heads, 1});
    graph.add("vision_attention_pack", {scratch(Scratch::Attention), context},
              qkvParams, {tokens, 1, 1});
    addGemm(graph, "vision_gemm_m64n128_residual", context, block.projection, hidden,
            hidden, layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    addNorm(graph, hidden, block.norm2, normalized, tokens);
    addGemm(graph, "vision_gemm_m64n128_gelu_tanh", normalized, block.upProjection,
            scratch(Scratch::Intermediate), hidden,
            layout.paddedIntermediateSize, layout.hiddenSize, tokens,
            kGemmRowTile, kGemmColumnTile);
    addGemm(graph, "vision_gemm_m64n128_residual", scratch(Scratch::Intermediate),
            block.downProjection, hidden, hidden, layout.hiddenSize,
            layout.paddedIntermediateSize, tokens, kGemmRowTile,
            kGemmColumnTile);
  }

  // The merger reads the normalized rows as (merged, 4608): four consecutive
  // block-major patches form one merged token.
  addNorm(graph, hidden, model_.mergerNorm, normalized, tokens);
  addGemm(graph, "vision_gemm_m32n256_gelu_erf", normalized,
          model_.mergerUpProjection,
          qkv, hidden, layout.mergedHiddenSize, layout.mergedHiddenSize, merged,
          kMergerRowTile, kMergerColumnTile);
  addGemm(graph, "vision_gemm_m32n256", qkv, model_.mergerDownProjection,
          embeddings,
          hidden, layout.outputHiddenSize, layout.mergedHiddenSize, merged,
          kMergerRowTile, kMergerColumnTile);
}

void Vision::inject(CommandGraph &graph, const MetalBuffer &embeddings,
                    const MetalBuffer &packedHidden, uint32_t hiddenSize,
                    uint32_t sourceRow, uint32_t destinationRow,
                    uint32_t rows) {
  if (!rows || !hiddenSize)
    throw std::invalid_argument("vision injection requires rows and a width");
  const VisionInjectParams params{sourceRow, destinationRow, rows, hiddenSize};
  graph.add("vision_inject_embeddings", {embeddings, packedHidden}, params,
            {std::min<uint64_t>((uint64_t{rows} * hiddenSize + 255) / 256,
                                1024),
             1, 1});
}

} // namespace splash::ops
