#include "InklingVision.hpp"

#include "Checked.hpp"
#include "metal/abi/Vision.h"
#include "ops/BufferExtent.hpp"
#include "ops/DenseAttention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;

// The one tower inkling_vision.metal is specialized for; its projections'
// sizes are DenseLinear's to check (requireWeights). Its heads of 64 need no
// padding for the dense attention.
constexpr VisionLayout kTower = kInklingVisionLayout;
static_assert(kTower.headDimension == 64 && kTower.heads * kTower.headDimension == kTower.hiddenSize &&
                  kTower.mergedHiddenSize == kTower.hiddenSize * kTower.spatialMerge * kTower.spatialMerge,
              "Inkling vision kernels' assumptions");

// Scratch tensor byte sizes for one encoder sized to maximumPatches, whose
// images have a token more, their class token's. The projections read whole
// row tiles, attention reads whole key tiles, and the merger reads the patch
// rows as (patches / 4, 4096), which the normalized rows (its input) and the
// q/k/v rows (its hidden rows) hold.
std::array<uint64_t, 11> scratchLayout(const VisionLayout &layout, uint32_t maximumPatches) {
  const uint32_t tokens = maximumPatches + 1;
  const uint64_t rows = DenseLinear::paddedRows(tokens);
  const uint64_t padded = DenseAttention::paddedKeys(tokens);
  const uint64_t mergedRows = DenseLinear::paddedRows(maximumPatches / 4);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  return {
      rows * layout.paddedPatchDimension * kBf16Bytes,                                // Patches
      uint64_t{tokens} * layout.headDimension * kFloatBytes,                          // RopeCos
      uint64_t{tokens} * layout.headDimension * kFloatBytes,                          // RopeSin
      uint64_t{tokens} * hidden * kFloatBytes,                                        // Hidden
      std::max(rows * hidden, mergedRows * layout.mergedHiddenSize) * kBf16Bytes,     // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergedHiddenSize) * kBf16Bytes, // Qkv
      rows * layout.intermediateSize * kBf16Bytes,                                    // Intermediate
      headRows * layout.headDimension * kBf16Bytes,                                   // Queries
      headRows * layout.headDimension * kBf16Bytes,                                   // Keys
      headRows * layout.headDimension * kBf16Bytes,                                   // Values
      rows * hidden * kBf16Bytes,                                                     // Context
  };
}

void requireLayout(const VisionLayout &layout) {
  if (layout != visionLayout(VisionTower::Inkling, layout.outputHiddenSize) || !layout.outputHiddenSize)
    throw std::invalid_argument("Inkling vision kernels are specialized for its 23-block tower");
}

// Every weight the kernels read: bf16 [output][input] matrices with a bias
// per output (the patch embedding's and the projector's zero), LayerNorm
// weights and biases, the LayerScales of the blocks' output projections,
// the class token's two rows and the learned position of every cell of the
// grid.
void requireWeights(const VisionWeights &model) {
  const VisionLayout &layout = model.layout;
  if (model.blocks.size() != layout.depth)
    throw std::invalid_argument("vision tower holds " + std::to_string(model.blocks.size()) + " blocks, needs " +
                                std::to_string(layout.depth));
  const uint32_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  DenseLinear::requireWeights(model.patchEmbedding, hidden, layout.paddedPatchDimension, "vision patch embedding");
  requireBytes(model.positionTable,
               uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden * kBf16Bytes,
               "vision position table");
  requireBytes(model.classToken, 2 * uint64_t{hidden} * kBf16Bytes, "vision class token");
  Normalization::requireLayer(model.inputNorm, hidden, "vision input norm");
  for (const VisionBlock &block : model.blocks) {
    Normalization::requireLayer(block.norm1, hidden, "vision attention norm");
    DenseLinear::requireWeights(block.qkv, 3 * hidden, hidden, "vision q/k/v");
    DenseLinear::requireWeights(block.projection, hidden, hidden, "vision attention projection");
    requireBytes(block.projection.scale, uint64_t{hidden} * kBf16Bytes, "vision attention projection scale");
    Normalization::requireLayer(block.norm2, hidden, "vision MLP norm");
    DenseLinear::requireWeights(block.upProjection, layout.intermediateSize, hidden, "vision up projection");
    DenseLinear::requireWeights(block.downProjection, hidden, layout.intermediateSize, "vision down projection");
    requireBytes(block.downProjection.scale, uint64_t{hidden} * kBf16Bytes, "vision down projection scale");
  }
  DenseLinear::requireWeights(model.mergerUpProjection, merged, merged, "vision merger up projection");
  DenseLinear::requireWeights(model.mergerDownProjection, layout.outputHiddenSize, merged,
                              "vision merger down projection");
}

} // namespace

uint64_t InklingVision::scratchBytes(const VisionLayout &layout, uint32_t maximumPatches) {
  requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4)
    throw std::invalid_argument("vision scratch must cover a positive multiple of four patches");
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total)
      throw std::overflow_error("vision scratch byte count overflows");
    total += aligned;
  }
  return total;
}

InklingVision::InklingVision(metal::MetalBackend &backend, const VisionWeights &model, uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches), dense_(backend.capabilities()) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  requireWeights(model);
  // New backend buffers are zero-filled, so padding rows read by whole tiles
  // start finite.
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared, "vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (cursor != total) throw std::logic_error("vision scratch arena mismatch");
}

void InklingVision::encode(CommandGraph &graph, ImageGrid grid, const MetalBuffer &pixels,
                           const MetalBuffer &embeddings) const {
  const VisionLayout &layout = model_.layout;
  if (!grid.valid() || grid.patches() > maximumPatches_)
    throw std::invalid_argument("image grid exceeds the vision encoder");
  requireBytes(pixels, grid.pixelBytes(layout.patchSize), "image pixel");
  requireBytes(embeddings, uint64_t{grid.mergedTokens()} * layout.outputHiddenSize * kBf16Bytes, "image embedding");

  const auto patches = static_cast<uint32_t>(grid.patches());
  // The patches and the class token after them.
  const uint32_t tokens = patches + 1;
  const uint32_t padded = DenseAttention::paddedKeys(tokens);
  const uint32_t merged = grid.mergedTokens();
  const uint32_t hiddenSize = layout.hiddenSize;
  const float epsilon = static_cast<float>(SPLASH_INKLING_VISION_NORM_EPSILON);
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionQkvParams qkvParams{tokens, padded};
  // inkling_vision_qkv_prepare's per-head planes of every padded token, of
  // 64 dimensions each; the output goes straight to the context rows the
  // projection reads, each head's 64 dimensions after the heads before it.
  const auto plane = [&](Scratch tensor) {
    return HeadTensor{scratch(tensor), padded * layout.headDimension, layout.headDimension};
  };
  const DenseAttentionTensors attention{plane(Scratch::Queries), plane(Scratch::Keys), plane(Scratch::Values),
                                        {scratch(Scratch::Context), layout.headDimension, hiddenSize}};
  const float scale = 1.0F / std::sqrt(static_cast<float>(layout.headDimension));
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &qkv = scratch(Scratch::Qkv);
  const MetalBuffer &context = scratch(Scratch::Context);

  graph.add("inkling_vision_patchify", {pixels, scratch(Scratch::Patches)}, gridParams, {patches, 1, 1});
  // The positions and the class token start the rows the patch embedding
  // (of the patches' rows; it has no bias) adds to; the LayerNorm after it
  // starts the residual stream.
  graph.add("inkling_vision_prepare_positions",
            {model_.positionTable, model_.classToken, hidden, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin)},
            gridParams, {tokens, 1, 1});
  dense_.add(graph, model_.patchEmbedding, hiddenSize, layout.paddedPatchDimension, scratch(Scratch::Patches), hidden,
             patches, DenseEpilogue::Residual);
  Normalization::addLayer(graph, hidden, model_.inputNorm, hidden, hiddenSize, tokens, epsilon,
                          FloatOutput::Float32);

  for (const VisionBlock &block : model_.blocks) {
    Normalization::addLayer(graph, hidden, block.norm1, normalized, hiddenSize, tokens, epsilon);
    dense_.add(graph, block.qkv, 3 * hiddenSize, hiddenSize, normalized, qkv, tokens);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    graph.add("inkling_vision_qkv_prepare",
              {qkv, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin), scratch(Scratch::Queries),
               scratch(Scratch::Keys), scratch(Scratch::Values)},
              qkvParams, {padded, 1, 1});
    // Every token attends every token of the image, the class token's too.
    DenseAttention::add(graph, attention, layout.heads, layout.headDimension, tokens, scale);
    dense_.add(graph, block.projection, hiddenSize, hiddenSize, context, hidden, tokens,
               DenseEpilogue::ScaledResidual);
    Normalization::addLayer(graph, hidden, block.norm2, normalized, hiddenSize, tokens, epsilon);
    dense_.add(graph, block.upProjection, layout.intermediateSize, hiddenSize, normalized,
               scratch(Scratch::Intermediate), tokens, DenseEpilogue::GeluErf);
    dense_.add(graph, block.downProjection, hiddenSize, layout.intermediateSize, scratch(Scratch::Intermediate), hidden,
               tokens, DenseEpilogue::ScaledResidual);
  }

  // The merger reads the patch rows as (merged, 4096): four consecutive
  // block-major patches form one merged token.
  graph.add("inkling_vision_patch_rows", {hidden, normalized}, {patches, 1, 1});
  dense_.add(graph, model_.mergerUpProjection, layout.mergedHiddenSize, layout.mergedHiddenSize, normalized, qkv,
             merged, DenseEpilogue::GeluErf);
  dense_.add(graph, model_.mergerDownProjection, layout.outputHiddenSize, layout.mergedHiddenSize, qkv, embeddings,
             merged);
}

} // namespace splash::ops
