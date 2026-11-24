#include "Vision.hpp"

#include "Checked.hpp"
#include "metal/abi/DenseAttention.h"
#include "metal/abi/Vision.h"
#include "ops/BufferExtent.hpp"
#include "ops/DenseAttention.hpp"
#include "ops/InklingVision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <unistd.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;

// The one vision tower the kernels are specialized for; its projections'
// sizes are DenseLinear's to check (requireWeights).
constexpr VisionLayout kTower{};
// Queries and keys of the tower's heads as the dense attention reads them.
constexpr uint32_t kQkDimension = splash_dense_attention_padded(kTower.headDimension);

// Scratch tensor byte sizes for one encoder sized to maximumPatches. The
// projections read whole row tiles, attention reads whole key tiles, and the
// merger reads the normalized rows as (patches / 4, 4608).
std::array<uint64_t, 11> scratchLayout(const VisionLayout &layout,
                                       uint32_t maximumPatches) {
  const uint64_t rows = DenseLinear::paddedRows(maximumPatches);
  const uint64_t padded = DenseAttention::paddedKeys(maximumPatches);
  const uint64_t mergedRows = DenseLinear::paddedRows(maximumPatches / 4);
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
      rows * hidden * kBf16Bytes,                                // Context
  };
}

void requireLayout(const VisionLayout &layout) {
  // The targets share the same vision tower; only the language-space
  // projection width varies with the text model.
  VisionLayout tower = layout;
  tower.outputHiddenSize = kTower.outputHiddenSize;
  if (tower != kTower || !layout.outputHiddenSize) {
    throw std::invalid_argument(
        "vision kernels are specialized for the Qwen3.5 27-block tower");
  }
}

// Every weight of the tower the kernels read, in each of its layout's blocks:
// bf16 [output][input] matrices with a bias per output, a LayerNorm's weight
// and bias per hidden value, and the learned position of every cell of the
// grid.
void requireWeights(const VisionWeights &model) {
  const VisionLayout &layout = model.layout;
  if (model.blocks.size() != layout.depth) {
    throw std::invalid_argument("vision tower holds " + std::to_string(model.blocks.size()) + " blocks, needs " +
                                std::to_string(layout.depth));
  }
  const uint32_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  DenseLinear::requireWeights(model.patchEmbedding, hidden, layout.patchDimension, "vision patch embedding");
  requireBytes(model.positionTable,
               uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden * kBf16Bytes,
               "vision position table");
  for (const VisionBlock &block : model.blocks) {
    Normalization::requireLayer(block.norm1, hidden, "vision attention norm");
    DenseLinear::requireWeights(block.qkv, 3 * hidden, hidden, "vision q/k/v");
    DenseLinear::requireWeights(block.projection, hidden, hidden, "vision attention projection");
    Normalization::requireLayer(block.norm2, hidden, "vision MLP norm");
    DenseLinear::requireWeights(block.upProjection, layout.paddedIntermediateSize, hidden, "vision up projection");
    DenseLinear::requireWeights(block.downProjection, hidden, layout.paddedIntermediateSize, "vision down projection");
  }
  Normalization::requireLayer(model.mergerNorm, hidden, "vision merger norm");
  DenseLinear::requireWeights(model.mergerUpProjection, merged, merged, "vision merger up projection");
  DenseLinear::requireWeights(model.mergerDownProjection, layout.outputHiddenSize, merged,
                              "vision merger down projection");
}

} // namespace

// As Metal allocates the arena: whole pages, as arenaBytes counts them.
uint64_t ImageEncoder::scratchBytes(const VisionLayout &layout, uint32_t maximumPatches) {
  const uint64_t bytes = layout.tower == VisionTower::Inkling ? InklingVision::scratchBytes(layout, maximumPatches)
                                                              : Vision::scratchBytes(layout, maximumPatches);
  const uint64_t page = static_cast<uint64_t>(getpagesize());
  return (bytes + page - 1) / page * page;
}

std::shared_ptr<ImageEncoder> ImageEncoder::make(metal::MetalBackend &backend, const VisionWeights &model,
                                                 uint32_t maximumPatches) {
  if (model.layout.tower == VisionTower::Inkling)
    return std::make_shared<InklingVision>(backend, model, maximumPatches);
  return std::make_shared<Vision>(backend, model, maximumPatches);
}

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

Vision::Vision(metal::MetalBackend &backend, const VisionWeights &model,
               uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches), dense_(backend.capabilities()) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  requireWeights(model);
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

void Vision::encode(CommandGraph &graph, ImageGrid grid,
                    const MetalBuffer &pixels,
                    const MetalBuffer &embeddings) const {
  const VisionLayout &layout = model_.layout;
  if (!grid.valid() || grid.patches() > maximumPatches_) {
    throw std::invalid_argument("image grid exceeds the vision encoder");
  }
  const auto tokens = static_cast<uint32_t>(grid.patches());
  requireBytes(pixels, grid.pixelBytes(layout.patchSize), "image pixel");
  requireBytes(embeddings, uint64_t{grid.mergedTokens()} * layout.outputHiddenSize * kBf16Bytes,
               "image embedding");

  const uint32_t padded = DenseAttention::paddedKeys(tokens);
  const uint32_t merged = grid.mergedTokens();
  const uint32_t hiddenSize = layout.hiddenSize;
  const float epsilon = static_cast<float>(SPLASH_VISION_NORM_EPSILON);
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionQkvParams qkvParams{tokens, padded};
  // vision_qkv_prepare's per-head planes of every padded token: queries and
  // keys padded to 80 dimensions, values of 72. The output goes straight to
  // the context rows the projection reads, each head's 72 dimensions after
  // the heads before it.
  const auto plane = [&](Scratch tensor, uint32_t width) {
    return HeadTensor{scratch(tensor), padded * width, width};
  };
  const DenseAttentionTensors attention{plane(Scratch::Queries, kQkDimension), plane(Scratch::Keys, kQkDimension),
                                        plane(Scratch::Values, layout.headDimension),
                                        {scratch(Scratch::Context), layout.headDimension, layout.hiddenSize}};
  const float scale = 1.0F / std::sqrt(static_cast<float>(layout.headDimension));
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
            gridParams, {tokens, 1, 1});
  dense_.add(graph, model_.patchEmbedding, hiddenSize, layout.patchDimension, scratch(Scratch::Patches),
             hidden, tokens, DenseEpilogue::Residual);

  for (const VisionBlock &block : model_.blocks) {
    Normalization::addLayer(graph, hidden, block.norm1, normalized, hiddenSize, tokens, epsilon);
    dense_.add(graph, block.qkv, 3 * hiddenSize, hiddenSize, normalized, qkv, tokens);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    graph.add("vision_qkv_prepare",
              {qkv, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin),
               scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values)},
              qkvParams, {padded, 1, 1});
    // Every patch attends every patch of the image.
    DenseAttention::add(graph, attention, layout.heads, layout.headDimension, tokens, scale);
    dense_.add(graph, block.projection, hiddenSize, hiddenSize, context, hidden, tokens, DenseEpilogue::Residual);
    Normalization::addLayer(graph, hidden, block.norm2, normalized, hiddenSize, tokens, epsilon);
    dense_.add(graph, block.upProjection, layout.paddedIntermediateSize, hiddenSize, normalized,
               scratch(Scratch::Intermediate), tokens, DenseEpilogue::GeluTanh);
    dense_.add(graph, block.downProjection, hiddenSize, layout.paddedIntermediateSize,
               scratch(Scratch::Intermediate), hidden, tokens, DenseEpilogue::Residual);
  }

  // The merger reads the normalized rows as (merged, 4608): four consecutive
  // block-major patches form one merged token.
  Normalization::addLayer(graph, hidden, model_.mergerNorm, normalized, hiddenSize, tokens, epsilon);
  dense_.add(graph, model_.mergerUpProjection, layout.mergedHiddenSize, layout.mergedHiddenSize, normalized,
             qkv, merged, DenseEpilogue::GeluErf);
  dense_.add(graph, model_.mergerDownProjection, layout.outputHiddenSize, layout.mergedHiddenSize, qkv,
             embeddings, merged);
}

} // namespace splash::ops
