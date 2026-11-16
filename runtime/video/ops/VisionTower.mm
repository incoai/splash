#include "video/ops/VisionTower.hpp"

#include "metal/abi/Vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::video::ops {
namespace {

constexpr uint32_t kGemmRowTile = 64, kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32, kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
constexpr uint32_t kHeads = 16, kHeadDimension = 72, kQkDimension = 80;
constexpr uint64_t kHalf = 2, kFloat = 4;
// Qwen3-VL's processor (preprocessor_config.json of MiniMax-H3's processor): smart_resize's factor and pixel range.
constexpr uint32_t kResizeFactor = 32;
constexpr uint64_t kMinimumPixels = 65536, kMaximumPixels = 16777216;

static_assert(VisionTower::kHidden % kGemmColumnTile == 0 && VisionTower::kPaddedIntermediate % kGemmColumnTile == 0 &&
                  VisionTower::kMergedHidden % kMergerColumnTile == 0 && VisionTower::kOutput % kMergerColumnTile == 0 &&
                  VisionTower::kPatchDimension % 128 == 0 && VisionTower::kHidden % 128 == 0 &&
                  VisionTower::kPaddedIntermediate % 128 == 0 && VisionTower::kMergedHidden % 128 == 0,
              "the tower kernels' tile and K-chunk assumptions");

uint32_t roundUp(uint32_t value, uint32_t multiple) { return (value + multiple - 1) / multiple * multiple; }
uint64_t align(uint64_t bytes) { return (bytes + 255) / 256 * 256; }

} // namespace

VisionGrid VisionTower::grid(uint32_t pixelHeight, uint32_t pixelWidth) {
  const uint64_t pixels = uint64_t{pixelHeight} * pixelWidth;
  if (!pixelHeight || !pixelWidth || pixelHeight % kResizeFactor || pixelWidth % kResizeFactor ||
      pixels < kMinimumPixels || pixels > kMaximumPixels)
    throw std::invalid_argument("a vision image of " + std::to_string(pixelWidth) + "x" + std::to_string(pixelHeight) +
                                " would be resized by the Qwen3-VL processor; the engine takes sides that are "
                                "multiples of 32 and 65,536 to 16,777,216 pixels");
  return {pixelHeight / kPatchPixels, pixelWidth / kPatchPixels};
}

uint32_t VisionTower::outputRows(VisionGrid grid) { return roundUp(grid.merged(), kMergerRowTile); }

VisionTower::VisionTower(metal::MetalBackend &backend, uint32_t maximumPatches)
    : backend_(backend), maximumPatches_(maximumPatches) {
  if (!maximumPatches || maximumPatches % 4) throw std::invalid_argument("vision tower scratch needs 4 n patches");
  // GEMMs read and write whole 64-row tiles, attention whole 128-key tiles, the mergers 32-row tiles of
  // (patches / 4, 4608).
  const uint64_t rows = roundUp(maximumPatches, kGemmRowTile);
  const uint64_t padded = roundUp(maximumPatches, kKeyTile);
  const uint64_t merged = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t headRows = uint64_t{kHeads} * padded;
  const std::array<uint64_t, Count> bytes{
      rows * kPatchDimension * kHalf,                                          // Patches
      std::max(rows * kHidden, merged * kMergedHidden) * kHalf,                // Normalized
      std::max(rows * 3 * kHidden, merged * kMergedHidden) * kHalf,            // Qkv (and the mergers' fc1 rows)
      rows * kPaddedIntermediate * kHalf,                                      // Intermediate
      headRows * kQkDimension * kHalf,                                         // Queries
      headRows * kQkDimension * kHalf,                                         // Keys
      headRows * kHeadDimension * kHalf,                                       // Values
      headRows * kHeadDimension * kHalf,                                       // Attention
      rows * kHidden * kHalf,                                                  // Context
  };
  uint64_t total = 0;
  for (uint64_t b : bytes) total += align(b);
  // New backend buffers are zero-filled: padding rows read by whole tiles start finite.
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared, "vision-tower-scratch");
  uint64_t cursor = 0;
  for (uint32_t i = 0; i < Count; ++i) {
    scratch_[i] = backend.view(arena_, cursor, bytes[i]);
    cursor += align(bytes[i]);
  }
}

VisionTowerImage VisionTower::image(VisionGrid grid, const uint8_t *rgb, const uint8_t *rgb2) const {
  if (!grid.height || !grid.width || grid.height % 2 || grid.width % 2 || grid.patches() > maximumPatches_)
    throw std::invalid_argument("vision image grid exceeds the tower's scratch");
  VisionTowerImage image;
  image.grid = grid;
  const uint64_t pixelBytes = uint64_t{grid.patches()} * kPatchPixels * kPatchPixels * 3;
  image.pixels = backend_.allocateBuffer(pixelBytes, metal::BufferStorage::Shared, "vision-tower-pixels");
  std::memcpy(image.pixels.contents(), rgb, pixelBytes);
  if (rgb2) {
    image.pixels2 = backend_.allocateBuffer(pixelBytes, metal::BufferStorage::Shared, "vision-tower-pixels-2");
    std::memcpy(image.pixels2.contents(), rgb2, pixelBytes);
  }
  const uint64_t rows = roundUp(grid.patches(), kGemmRowTile);
  image.hidden = backend_.allocateBuffer(rows * kHidden * kFloat, metal::BufferStorage::Shared, "vision-tower-hidden");
  const uint64_t rope = uint64_t{grid.patches()} * kHeadDimension * kFloat;
  image.ropeCos = backend_.allocateBuffer(rope, metal::BufferStorage::Shared, "vision-tower-rope-cos");
  image.ropeSin = backend_.allocateBuffer(rope, metal::BufferStorage::Shared, "vision-tower-rope-sin");
  for (auto &output : image.outputs)
    output = backend_.allocateBuffer(uint64_t{outputRows(grid)} * kOutput * kFloat, metal::BufferStorage::Shared,
                                     "vision-tower-output");
  return image;
}

namespace {

void gemm(metal::CommandGraph &graph, const char *pipeline, const metal::MetalBuffer &input,
          const VisionTowerAffine &weights, const metal::MetalBuffer &output, const metal::MetalBuffer &residual,
          uint32_t outputSize, uint32_t inputSize, uint32_t rows, uint32_t tileRows, uint32_t tileColumns) {
  graph.add(pipeline, {input, weights.weight, weights.bias, output, residual}, VisionGemmParams{outputSize, inputSize},
            {(rows + tileRows - 1) / tileRows, outputSize / tileColumns, 1});
}

void norm(metal::CommandGraph &graph, const metal::MetalBuffer &input, const VisionTowerNorm &weights,
          const metal::MetalBuffer &output, uint32_t width, uint32_t rows) {
  graph.add("h3_vision_layer_norm", {input, weights.weight, weights.bias, output}, VisionNormParams{width},
            {rows, 1, 1});
}

} // namespace

void VisionTower::addEmbedding(metal::CommandGraph &graph, const VisionTowerEmbedding &weights,
                               const VisionTowerImage &image) const {
  const VisionGridParams params{image.grid.height, image.grid.width};
  const uint32_t tokens = image.grid.patches();
  if (image.pixels2)
    graph.add("h3_vision_patchify_pair", {image.pixels, image.pixels2, scratch_[Patches]}, params, {tokens, 1, 1});
  else
    graph.add("h3_vision_patchify", {image.pixels, scratch_[Patches]}, params, {tokens, 1, 1});
  // The positions start the residual stream the patch embedding adds to; padding rows of the last tile are zeroed.
  graph.add("h3_vision_prepare_positions", {weights.positions, image.hidden, image.ropeCos, image.ropeSin}, params,
            {roundUp(tokens, kGemmRowTile), 1, 1});
  gemm(graph, "h3_vision_gemm_m64n128_residual", scratch_[Patches], weights.patch, image.hidden, image.hidden, kHidden,
       kPatchDimension, tokens, kGemmRowTile, kGemmColumnTile);
}

void VisionTower::addBlock(metal::CommandGraph &graph, const VisionTowerBlock &weights,
                           const VisionTowerImage &image) const {
  const uint32_t tokens = image.grid.patches();
  const uint32_t padded = roundUp(tokens, kKeyTile);
  const VisionQkvParams qkvParams{tokens, padded};
  const VisionAttentionParams attentionParams{tokens, padded, 1.0F / std::sqrt(float(kHeadDimension))};
  norm(graph, image.hidden, weights.norm1, scratch_[Normalized], kHidden, tokens);
  gemm(graph, "h3_vision_gemm_m64n128", scratch_[Normalized], weights.qkv, scratch_[Qkv], scratch_[Qkv], 3 * kHidden,
       kHidden, tokens, kGemmRowTile, kGemmColumnTile);
  graph.add("h3_vision_qkv_prepare",
            {scratch_[Qkv], image.ropeCos, image.ropeSin, scratch_[Queries], scratch_[Keys], scratch_[Values]},
            qkvParams, {padded, 1, 1});
  graph.add("h3_vision_attention", {scratch_[Queries], scratch_[Keys], scratch_[Values], scratch_[Attention]},
            attentionParams, {(tokens + SPLASH_VISION_ATTENTION_ROWS - 1) / SPLASH_VISION_ATTENTION_ROWS, kHeads, 1},
            {SPLASH_VISION_ATTENTION_THREADS, 1, 1});
  // The shared tower's gather: a 16-bit copy of the head-major rows, so it moves fp16 as it moves bf16.
  graph.add("vision_attention_gather", {scratch_[Attention], scratch_[Context]}, qkvParams, {tokens, 1, 1});
  gemm(graph, "h3_vision_gemm_m64n128_residual", scratch_[Context], weights.projection, image.hidden, image.hidden,
       kHidden, kHidden, tokens, kGemmRowTile, kGemmColumnTile);
  norm(graph, image.hidden, weights.norm2, scratch_[Normalized], kHidden, tokens);
  gemm(graph, "h3_vision_gemm_m64n128_gelu_tanh", scratch_[Normalized], weights.up, scratch_[Intermediate],
       scratch_[Intermediate], kPaddedIntermediate, kHidden, tokens, kGemmRowTile, kGemmColumnTile);
  gemm(graph, "h3_vision_gemm_m64n128_residual", scratch_[Intermediate], weights.down, image.hidden, image.hidden,
       kHidden, kPaddedIntermediate, tokens, kGemmRowTile, kGemmColumnTile);
}

void VisionTower::addMerger(metal::CommandGraph &graph, const VisionTowerMerger &weights,
                            const VisionTowerImage &image, uint32_t output) const {
  if (output >= image.outputs.size()) throw std::invalid_argument("vision tower has four merger outputs");
  const uint32_t merged = image.grid.merged();
  // The merger normalizes each patch, then reads four consecutive (block-major) patches as one 4608-wide row; a
  // deepstack merger normalizes the 4608 values of a merged token as one row.
  if (output == 0)
    norm(graph, image.hidden, weights.norm, scratch_[Normalized], kHidden, image.grid.patches());
  else
    norm(graph, image.hidden, weights.norm, scratch_[Normalized], kMergedHidden, merged);
  gemm(graph, "h3_vision_gemm_m32n256_gelu_erf", scratch_[Normalized], weights.up, scratch_[Qkv], scratch_[Qkv],
       kMergedHidden, kMergedHidden, merged, kMergerRowTile, kMergerColumnTile);
  gemm(graph, "h3_vision_gemm_m32n256_f32", scratch_[Qkv], weights.down, image.outputs[output],
       image.outputs[output], kOutput, kMergedHidden, merged, kMergerRowTile, kMergerColumnTile);
}

} // namespace splash::video::ops
