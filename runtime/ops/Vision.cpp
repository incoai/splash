#include "Vision.hpp"

#include "Checked.hpp"
#include "metal/abi/Vision.h"
#include "ops/BufferExtent.hpp"

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

constexpr uint32_t kGemmRowTile = 64;
constexpr uint32_t kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32;
constexpr uint32_t kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;

// The towers the kernels compile (kernels/shared/vision.metal): Qwen3.5's,
// and RADIO's, whose kernels are named _radio and _d80.
constexpr std::array kCompiledVision{VisionLayout{}, kRadioVisionLayout};
static_assert(std::all_of(kCompiledVision.begin(), kCompiledVision.end(),
                          [](const VisionLayout &tower) {
                            return tower.hiddenSize % kGemmColumnTile == 0 &&
                                   tower.paddedIntermediateSize % kGemmColumnTile == 0 &&
                                   tower.mergedHiddenSize % kMergerColumnTile == 0 &&
                                   tower.mergerIntermediateSize % kMergerColumnTile == 0 &&
                                   tower.patchDimension % kGemmColumnTile == 0;
                          }),
              "vision kernels' tile assumptions");

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

// Query and key rows hold the head dimension padded with zeros to the 16 a
// matmul's K extent takes: Qwen3.5's 72 to 80, RADIO's 80 as it is.
uint32_t qkDimension(const VisionLayout &layout) noexcept {
  return roundUp(layout.headDimension, 16);
}

// The bias RADIO's bias-free GEMMs read: zeros for the widest of them, at
// the arena's end after the scratch tensors.
uint64_t zeroBiasBytes(const VisionLayout &layout) noexcept {
  return uint64_t{std::max({layout.hiddenSize, layout.mergerIntermediateSize, layout.outputHiddenSize})} *
         kBf16Bytes;
}

// Scratch tensor byte sizes for one encoder sized to maximumPatches. The
// tower's tokens are the patches and its prefix; GEMM tiles read and write
// whole 64-row tiles, attention reads whole 128-key tiles, and the merger
// reads the normalized rows as (patches / 4, merged) and writes its up
// projection's (patches / 4, mergerIntermediate) over the q/k/v rows. RADIO
// has no rope, so its rope tables are a placeholder value, and its patch
// rows are two frames wide for a video's tubelet.
std::array<uint64_t, 12> scratchLayout(const VisionLayout &layout,
                                       uint32_t maximumPatches, bool radio) {
  const uint32_t tokens = maximumPatches + layout.prefixTokens;
  const uint64_t rows = roundUp(tokens, kGemmRowTile);
  const uint64_t padded = roundUp(tokens, kKeyTile);
  const uint64_t mergedRows = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  const uint64_t rope = radio ? kFloatBytes : uint64_t{maximumPatches} * layout.headDimension * kFloatBytes;
  return {
      rows * layout.patchDimension * (radio ? 2 : 1) * kBf16Bytes, // Patches
      rope,                                                      // RopeCos
      rope,                                                      // RopeSin
      rows * hidden * kFloatBytes,                               // Hidden
      std::max(rows * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergerIntermediateSize) *
          kBf16Bytes,                                            // Qkv
      rows * layout.paddedIntermediateSize * kBf16Bytes,         // Intermediate
      headRows * qkDimension(layout) * kBf16Bytes,               // Queries
      headRows * qkDimension(layout) * kBf16Bytes,               // Keys
      headRows * layout.headDimension * kBf16Bytes,              // Values
      headRows * layout.headDimension * kBf16Bytes,              // Attention
      rows * hidden * kBf16Bytes,                                // Context
  };
}

// Whether the layout is RADIO's; throws unless the kernels compile its tower.
// Only the language-space projection width varies with the text model, and
// whether RADIO has its final norm.
bool requireLayout(const VisionLayout &layout) {
  VisionLayout tower = layout;
  tower.outputHiddenSize = 0;
  tower.finalNorm = false;
  if (std::find(kCompiledVision.begin(), kCompiledVision.end(), tower) == kCompiledVision.end() ||
      !layout.outputHiddenSize || layout.outputHiddenSize % kMergerColumnTile) {
    throw std::invalid_argument("no vision kernels for the tower");
  }
  if (layout.finalNorm && !isRadioVision(tower)) throw std::invalid_argument("only RADIO has a final vision norm");
  return isRadioVision(tower);
}

// Every weight of the tower the kernels read, in each of its layout's blocks:
// bf16 [output][input] matrices with a bias per output, a LayerNorm's weight
// and bias per hidden value, and the learned position of every cell of the
// grid. RADIO's patch embedding and merger read no bias, it reads its prefix
// tokens, and its merger norm is an RMS weight per merged value.
void requireWeights(const VisionWeights &model, bool radio) {
  const VisionLayout &layout = model.layout;
  if (model.blocks.size() != layout.depth) {
    throw std::invalid_argument("vision tower holds " + std::to_string(model.blocks.size()) + " blocks, needs " +
                                std::to_string(layout.depth));
  }
  const uint64_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  const uint64_t mergerIntermediate = layout.mergerIntermediateSize;
  const auto requireAffine = [](const VisionAffine &affine, uint64_t output, uint64_t input,
                                const std::string &name, bool bias = true) {
    requireBytes(affine.weight, output * input * kBf16Bytes, name + " weight");
    if (bias) requireBytes(affine.bias, output * kBf16Bytes, name + " bias");
  };
  const auto requireNorm = [&](const VisionNorm &norm, const std::string &name) {
    requireBytes(norm.weight, hidden * kBf16Bytes, name + " weight");
    requireBytes(norm.bias, hidden * kBf16Bytes, name + " bias");
  };
  requireAffine(model.patchEmbedding, hidden, layout.patchDimension, "vision patch embedding", !radio);
  if (radio) requireAffine(model.videoPatchEmbedding, hidden, 2 * layout.patchDimension, "vision video patch embedding", false);
  requireBytes(model.positionTable,
               uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden * kBf16Bytes,
               "vision position table");
  if (radio) requireBytes(model.prefixTokens, uint64_t{layout.prefixTokens} * hidden * kBf16Bytes, "vision prefix tokens");
  for (const VisionBlock &block : model.blocks) {
    requireNorm(block.norm1, "vision attention norm");
    requireAffine(block.qkv, 3 * hidden, hidden, "vision q/k/v");
    requireAffine(block.projection, hidden, hidden, "vision attention projection");
    requireNorm(block.norm2, "vision MLP norm");
    requireAffine(block.upProjection, layout.paddedIntermediateSize, hidden, "vision up projection");
    requireAffine(block.downProjection, hidden, layout.paddedIntermediateSize, "vision down projection");
  }
  if (layout.finalNorm) requireNorm(model.finalNorm, "vision final norm");
  if (radio)
    requireBytes(model.mergerNorm.weight, merged * kBf16Bytes, "vision merger norm weight");
  else
    requireNorm(model.mergerNorm, "vision merger norm");
  requireAffine(model.mergerUpProjection, mergerIntermediate, merged, "vision merger up projection", !radio);
  requireAffine(model.mergerDownProjection, layout.outputHiddenSize, mergerIntermediate,
                "vision merger down projection", !radio);
}

} // namespace

uint64_t Vision::scratchBytes(const VisionLayout &layout,
                              uint32_t maximumPatches) {
  const bool radio = requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4) {
    throw std::invalid_argument(
        "vision scratch must cover a positive multiple of four patches");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches, radio)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vision scratch byte count overflows");
    }
    total += aligned;
  }
  if (radio)
    total += alignUp(zeroBiasBytes(layout));
  return total;
}

uint32_t Vision::embeddingRows(ImageGrid grid) noexcept {
  return roundUp(grid.mergedTokens(), kMergerRowTile);
}

Vision::Vision(metal::MetalBackend &backend, const VisionWeights &model,
               uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches), radio_(requireLayout(model.layout)) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  requireWeights(model, radio_);
  // New backend buffers are zero-filled, so padding rows read by whole tiles
  // start finite.
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared,
                                  "vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches, radio_);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (radio_) {
    zeros_ = backend.view(arena_, cursor, zeroBiasBytes(model.layout));
    cursor += alignUp(zeroBiasBytes(model.layout));
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
  if (grid.frames != 1 && !radio_) throw std::invalid_argument("only RADIO encodes video");
  const auto patches = static_cast<uint32_t>(grid.patches());
  const uint32_t tokens = patches + layout.prefixTokens;
  requireBytes(pixels, grid.pixelBytes(), "image pixel");
  requireBytes(embeddings, uint64_t{embeddingRows(grid)} * layout.outputHiddenSize * kBf16Bytes,
               "image embedding");

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

  if (radio_) {
    const bool video = grid.frames == 2;
    graph.add(video ? "vision_patchify_radio_video" : "vision_patchify_radio", {pixels, scratch(Scratch::Patches)},
              gridParams, {patches, 1, 1});
    graph.add("vision_prepare_positions_radio", {model_.positionTable, hidden}, gridParams, {patches, 1, 1});
    addGemm(graph, "vision_gemm_m64n128_residual", scratch(Scratch::Patches),
            {(video ? model_.videoPatchEmbedding : model_.patchEmbedding).weight, zeros_}, hidden, hidden,
            layout.hiddenSize, grid.frames * layout.patchDimension, patches, kGemmRowTile, kGemmColumnTile);
    // The prefix rows, past the embedding's last tile over them.
    graph.add("vision_prefix_radio", {model_.prefixTokens, hidden}, gridParams,
              {roundUp(tokens, kGemmRowTile) - patches, 1, 1});
  } else {
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
  }

  for (const VisionBlock &block : model_.blocks) {
    addNorm(graph, hidden, block.norm1, normalized, tokens);
    addGemm(graph, "vision_gemm_m64n128", normalized, block.qkv, qkv, hidden,
            3 * layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    if (radio_)
      graph.add("vision_qkv_prepare_radio",
                {qkv, scratch(Scratch::Queries), scratch(Scratch::Keys), scratch(Scratch::Values)}, qkvParams,
                {padded, 1, 1});
    else
      graph.add("vision_qkv_prepare",
                {qkv, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin),
                 scratch(Scratch::Queries), scratch(Scratch::Keys),
                 scratch(Scratch::Values)},
                qkvParams, {padded, 1, 1});
    graph.add(radio_ ? "vision_attention_d80" : "vision_attention",
              {scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values), scratch(Scratch::Attention)},
              attentionParams,
              {(tokens + SPLASH_VISION_ATTENTION_ROWS - 1) /
                   SPLASH_VISION_ATTENTION_ROWS,
               layout.heads, 1},
              {SPLASH_VISION_ATTENTION_THREADS, 1, 1});
    graph.add(radio_ ? "vision_attention_gather_d80" : "vision_attention_gather", {scratch(Scratch::Attention), context},
              qkvParams, {tokens, 1, 1});
    addGemm(graph, "vision_gemm_m64n128_residual", context, block.projection, hidden,
            hidden, layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    addNorm(graph, hidden, block.norm2, normalized, tokens);
    addGemm(graph, radio_ ? "vision_gemm_m64n128_gelu_erf" : "vision_gemm_m64n128_gelu_tanh", normalized,
            block.upProjection,
            scratch(Scratch::Intermediate), hidden,
            layout.paddedIntermediateSize, layout.hiddenSize, tokens,
            kGemmRowTile, kGemmColumnTile);
    addGemm(graph, "vision_gemm_m64n128_residual", scratch(Scratch::Intermediate),
            block.downProjection, hidden, hidden, layout.hiddenSize,
            layout.paddedIntermediateSize, tokens, kGemmRowTile,
            kGemmColumnTile);
  }

  // The merger reads the normalized rows as (merged, mergedHiddenSize): four
  // consecutive block-major patches form one merged token. Qwen3.5 norms
  // each patch's row before, RADIO the merged row after.
  if (radio_) {
    if (layout.finalNorm)
      graph.add("vision_layer_norm_f32", {hidden, model_.finalNorm.weight, model_.finalNorm.bias, hidden},
                VisionNormParams{layout.hiddenSize}, {patches, 1, 1});
    graph.add("vision_rms_norm", {hidden, model_.mergerNorm.weight, normalized},
              VisionNormParams{layout.mergedHiddenSize}, {merged, 1, 1});
    addGemm(graph, "vision_gemm_m32n256_relu2", normalized, {model_.mergerUpProjection.weight, zeros_}, qkv,
            hidden, layout.mergerIntermediateSize, layout.mergedHiddenSize, merged, kMergerRowTile,
            kMergerColumnTile);
    addGemm(graph, "vision_gemm_m32n256", qkv, {model_.mergerDownProjection.weight, zeros_}, embeddings, hidden,
            layout.outputHiddenSize, layout.mergerIntermediateSize, merged, kMergerRowTile, kMergerColumnTile);
    return;
  }
  addNorm(graph, hidden, model_.mergerNorm, normalized, tokens);
  addGemm(graph, "vision_gemm_m32n256_gelu_erf", normalized,
          model_.mergerUpProjection,
          qkv, hidden, layout.mergerIntermediateSize, layout.mergedHiddenSize, merged,
          kMergerRowTile, kMergerColumnTile);
  addGemm(graph, "vision_gemm_m32n256", qkv, model_.mergerDownProjection,
          embeddings,
          hidden, layout.outputHiddenSize, layout.mergerIntermediateSize, merged,
          kMergerRowTile, kMergerColumnTile);
}

uint64_t VideoPruning::flagBytes(const VideoPruningShape &shape) noexcept {
  return uint64_t{shape.tubelets} * shape.grid.mergedTokens() * kFloatBytes;
}

void VideoPruning::add(CommandGraph &graph, const VideoPruningShape &shape, const MetalBuffer &encoded,
                       const MetalBuffer &text, const MetalBuffer &cuts, const MetalBuffer &dissimilarity,
                       const MetalBuffer &keep, const MetalBuffer &output) {
  const uint32_t rows = shape.grid.mergedTokens();
  const uint64_t total = uint64_t{shape.tubelets} * rows;
  if (!shape.tubelets || shape.grid.frames != 2 || !shape.width || !shape.textTokens || shape.kept < rows ||
      shape.kept > total || total + shape.textTokens > UINT32_MAX)
    throw std::invalid_argument("invalid video pruning");
  const uint32_t stride = Vision::embeddingRows(shape.grid);
  requireBytes(encoded, uint64_t{shape.tubelets} * stride * shape.width * kBf16Bytes, "video tubelet rows");
  requireBytes(text, uint64_t{shape.textTokens} * shape.width * kBf16Bytes, "video text rows");
  requireBytes(cuts, uint64_t{shape.tubelets} * sizeof(uint32_t), "video text cuts");
  requireBytes(dissimilarity, flagBytes(shape), "video dissimilarity");
  requireBytes(keep, flagBytes(shape), "video keep flag");
  requireBytes(output, (uint64_t{shape.textTokens} + shape.kept) * shape.width * kBf16Bytes, "video rows");
  const VideoPruningParams params{shape.tubelets, rows, stride, shape.width, shape.kept, shape.textTokens};
  const auto groups = static_cast<uint32_t>(total);
  graph.add("video_pruning_dissimilarity", {encoded, dissimilarity}, params, {groups, 1, 1});
  graph.add("video_pruning_keep", {dissimilarity, keep}, params,
            {(groups + CommandGraph::kDefaultThreads - 1) / CommandGraph::kDefaultThreads, 1, 1});
  graph.add("video_pruning_place", {encoded, text, cuts, keep, output}, params,
            {groups + shape.textTokens, 1, 1});
}

} // namespace splash::ops
