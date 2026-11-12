#pragma once

// The text encoder's vision tower (kernels/video/vision_tower.metal): Qwen3-VL-32B's ViT, which turns an image of
// the presentation into the language-space rows of its vision block, as Qwen3VLVisionModel does for one image:
//   patchify (16 x 16 patches over the temporal patch of 2: an image's frame twice, a video block's pair of frames;
//   spatial-merge-block-major order)
//   -> patch embedding + the bilinear resample of the 48 x 48 position table (fp32 residual stream)
//   -> 27 blocks: LayerNorm -> QKV -> 2D RoPE -> full attention over the image -> projection (+ residual),
//      LayerNorm -> fc1 + GELU (tanh) -> fc2 (+ residual)
//   -> after blocks 8, 16 and 24 a deepstack merger (LayerNorm over a merged token's 4 x 1152 values -> fc1 + GELU
//      (erf) -> fc2), after the last block the merger (LayerNorm per patch, then the same MLP): [merged][5120] each.
// fp16 weights (converted from the checkpoint's bf16 by model/VisionTowerWeights) and activations, fp32 accumulation,
// residual stream and LayerNorm statistics; the mergers' rows stay fp32.
//
// The caller streams the weights: one command per stage (embedding, each block, each merger), each with the weights
// of that stage only, for every image of the request at once (state per image: VisionTowerImage).
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <array>
#include <cstdint>

namespace splash::video::ops {

struct VisionTowerAffine {
  metal::MetalBuffer weight;  // fp16 [output][input]
  metal::MetalBuffer bias;    // fp16 [output]
};
struct VisionTowerNorm {
  metal::MetalBuffer weight, bias;  // fp16 [width]
};
struct VisionTowerEmbedding {
  VisionTowerAffine patch;       // [1152][1536]: input features (channel, temporal, row, column)
  metal::MetalBuffer positions;  // fp16 [48 x 48][1152]
};
struct VisionTowerBlock {
  VisionTowerNorm norm1;
  VisionTowerAffine qkv, projection;
  VisionTowerNorm norm2;
  VisionTowerAffine up, down;  // fc1 / fc2, the intermediate zero-padded from 4304 to 4352
};
struct VisionTowerMerger {
  VisionTowerNorm norm;         // 1152 (the merger) or 4608 (a deepstack merger)
  VisionTowerAffine up, down;  // [4608][4608], [5120][4608]
};

// An image's patch grid, in 16-pixel patches: both sides even (the 2 x 2 merge).
struct VisionGrid {
  uint32_t height = 0, width = 0;
  [[nodiscard]] uint32_t patches() const { return height * width; }
  [[nodiscard]] uint32_t merged() const { return patches() / 4; }
};

// One image's state through the tower: its pixels (pixels2: a video block's second frame, empty for an image), its
// fp32 residual stream and RoPE tables, and the four outputs (the merger's rows, then the three deepstack mergers'),
// fp32 [outputRows][5120] each, rows past merged() padding. A video block (Qwen3-VL's video, one merged frame pair,
// grid t = 1) runs as an image does: Qwen3-VL attends within each frame pair (cu_seqlens per t) and repeats the
// position embedding and 2D RoPE per t, so only its patchify differs.
struct VisionTowerImage {
  VisionGrid grid;
  metal::MetalBuffer pixels, pixels2, hidden, ropeCos, ropeSin;
  std::array<metal::MetalBuffer, 4> outputs;
};

class VisionTower final {
public:
  static constexpr uint32_t kDepth = 27;
  static constexpr uint32_t kHidden = 1152;
  static constexpr uint32_t kPatchDimension = 1536;
  static constexpr uint32_t kIntermediate = 4304;
  static constexpr uint32_t kPaddedIntermediate = 4352;
  static constexpr uint32_t kMergedHidden = 4608;
  static constexpr uint32_t kOutput = 5120;
  static constexpr uint32_t kPositionSide = 48;
  static constexpr uint32_t kPatchPixels = 16;
  // The deepstack mergers follow these blocks, in order (deepstack_visual_indexes).
  static constexpr std::array<uint32_t, 3> kDeepstack{8, 16, 24};

  // The grid of an image of these pixels: the Qwen3-VL processor's smart_resize (factor 32, 65,536 to 16,777,216
  // pixels) must leave it as it is, which holds for every canvas (multiples of 32). Throws std::invalid_argument
  // otherwise.
  [[nodiscard]] static VisionGrid grid(uint32_t pixelHeight, uint32_t pixelWidth);
  [[nodiscard]] static uint32_t outputRows(VisionGrid grid);

  // Scratch for images of up to maximumPatches patches, shared by the images of a command in turn.
  VisionTower(metal::MetalBackend &backend, uint32_t maximumPatches);
  VisionTower(const VisionTower &) = delete;
  VisionTower &operator=(const VisionTower &) = delete;

  // An image's state, its pixels (uint8 RGB [16 h][16 w][3]) copied in; with rgb2, a video block's state, rgb the
  // first and rgb2 the second frame of its temporal pair.
  [[nodiscard]] VisionTowerImage image(VisionGrid grid, const uint8_t *rgb, const uint8_t *rgb2 = nullptr) const;
  [[nodiscard]] uint64_t scratchBytes() const { return arena_.sizeBytes(); }

  void addEmbedding(metal::CommandGraph &graph, const VisionTowerEmbedding &weights, const VisionTowerImage &image) const;
  void addBlock(metal::CommandGraph &graph, const VisionTowerBlock &weights, const VisionTowerImage &image) const;
  // output 0: the merger after the last block; 1 + j: deepstack merger j after block kDeepstack[j].
  void addMerger(metal::CommandGraph &graph, const VisionTowerMerger &weights, const VisionTowerImage &image,
                 uint32_t output) const;

private:
  enum Scratch : uint32_t { Patches, Normalized, Qkv, Intermediate, Queries, Keys, Values, Attention, Context, Count };
  metal::MetalBackend &backend_;
  uint32_t maximumPatches_ = 0;
  metal::MetalBuffer arena_;
  std::array<metal::MetalBuffer, Count> scratch_;
};

} // namespace splash::video::ops
