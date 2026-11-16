#pragma once

// The text encoder of the video model: MiniMax-H3 conditions on
// hidden_states[50] of Qwen3-VL-32B, the unnormalized residual stream after
// decoder layer 50 of 64, of every prompt token. It is the language model of
// an MLX affine checkpoint (4 bits in groups of 64), run as a prefill-only
// encoder through those layers: the later layers, the final norm and the head
// are never read. Its token table, norms and projections load as the
// engine's block images of the checkpoint's own quantization and run the
// engine's token gather, RMS norms and prefill linears (ops/); only the
// attention is its own (kernels/video/text_encoder.metal): Q/K RMS norms,
// the rotary embedding of a text prompt, whose M-RoPE positions are the same
// on every axis and so rotate as one, and causal GQA attention over the
// prompt in one pass. A presentation with images adds the vision tower and
// the interleaved M-RoPE of its rows (kernels/video/vision_tower.metal); one
// with reference videos (Ref2VA) adds their frame pairs' vision blocks, and a
// presentation longer than one prefill pass runs its linears a pass of rows at
// a time (the attention and norms still over every row at once).

#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::video::model {

class TextEncoder final {
public:
  // hidden_states[kLayers] rows of kHiddenSize values.
  static constexpr uint32_t kLayers = 50;
  static constexpr uint32_t kHiddenSize = 5120;
  // One prefill pass: the engine's prefill linears' row bound.
  static constexpr uint32_t kMaximumTokens = SPLASH_PREFILL_TOKEN_BUDGET;
  // A presentation's bound, the prompt bound the engine announces: past kMaximumTokens rows the linears run a pass
  // of kMaximumTokens rows at a time.
  // MiniMax-H3's Ref2VA presentation of a 5 s reference video and two images takes 6,393 rows at 480p and 21,045
  // at 768p.
  static constexpr uint32_t kMaximumPresentationTokens = 16 * SPLASH_PREFILL_TOKEN_BUDGET;

  // Plans the checkpoint in `directory` (its config.json and safetensors
  // shards) and checks its configuration; loads no weight. Throws
  // std::invalid_argument for a checkpoint of another model.
  TextEncoder(metal::MetalBackend &backend, const std::filesystem::path &directory);
  ~TextEncoder();
  TextEncoder(const TextEncoder &) = delete;
  TextEncoder &operator=(const TextEncoder &) = delete;

  // hidden_states[kLayers] of a prompt of 1 to kMaximumPresentationTokens token ids,
  // f32 [tokens][kHiddenSize] of bf16 values. The weights stream through the
  // call a layer at a time, each loaded before its command and freed after
  // it: the encoder holds memory only while it runs, and then one layer's
  // weights and the activations. Throws std::invalid_argument for a prompt it
  // cannot encode.
  //
  // A presentation with images (MiniMax-H3's FL2VA / i2va presentation:
  // "<Picture i>: ", then <|vision_start|>, one <|image_pad|> per merged
  // token of image i, <|vision_end|>, per keyframe, then the prompt) also
  // runs the checkpoint's vision tower (video/ops/VisionTower.hpp, its weights
  // streamed a stage at a time before the text layers, model/
  // VisionTowerWeights.hpp) on each image, as Qwen3-VL does: the merger's
  // rows replace the pad rows' embeddings, the three deepstack mergers' rows
  // are added to the pad rows after decoder layers 0, 1 and 2, and the
  // rotary positions are Qwen3-VL's M-RoPE of the presentation (positions).
  // images[i] is the image of the presentation's i-th vision block. Without
  // images and vision tokens the encode is the text-only one, command for
  // command.
  //
  // MiniMax-H3's Ref2VA presentation (diffusers MiniMaxH3Ref2VATextEncoderStep)
  // also holds video blocks: per reference video "<Video k>: " and per merged
  // frame pair "<t seconds>" + <|vision_start|>, one <|video_pad|> per merged
  // token of a frame, <|vision_end|>. A video block is one Image whose rgb is
  // the pair's first sampled frame and rgb2 its second (videoBlockFrames); the
  // tower runs it as Qwen3-VL runs a video (attention within the frame pair,
  // the same position embedding and 2D RoPE), and its M-RoPE positions are
  // those of an image block (get_rope_index splits a video grid per merged
  // frame; the timestamps between them are text). images lists every vision
  // block, image or video, in token order.
  static constexpr uint32_t kVisionStart = 151652, kVisionEnd = 151653, kImagePad = 151655, kVideoPad = 151656;
  struct Image {
    const uint8_t *rgb = nullptr;  // uint8 RGB [height][width][3]
    uint32_t width = 0, height = 0;
    const uint8_t *rgb2 = nullptr;  // second frame of a video block's temporal pair (null: the frame twice)
  };
  [[nodiscard]] std::vector<float> encode(std::span<const uint32_t> tokens, std::span<const Image> images = {});

  // The presentation's vision blocks: per block its first pad row, pad count and whether its pads are
  // <|video_pad|>. Throws std::invalid_argument for vision tokens outside a well-formed block (<|vision_start|>, one
  // or more pads of one kind, <|vision_end|>).
  struct VisionBlock {
    uint32_t start = 0, pads = 0;
    bool video = false;
  };
  [[nodiscard]] static std::vector<VisionBlock> visionBlocks(std::span<const uint32_t> tokens);
  // The DiT's tags of the presentation's rows: 1 text, 0 a vision block's rows (its start, pads and end), as
  // MiniMax-H3's presentation tags them.
  [[nodiscard]] static std::vector<uint8_t> tags(std::span<const uint32_t> tokens);
  // The pad tokens of an image of these pixels (Qwen3-VL's processor; throws for an image it would resize). A video
  // block of frames of these pixels takes as many (the video processor's smart_resize leaves 480x832 and 768x1344
  // frames as they are).
  [[nodiscard]] static uint32_t imageTokens(uint32_t width, uint32_t height);

  // A reference video's vision blocks (diffusers _sample_video_condition_frames, Qwen3-VL's temporal patch of 2):
  // of `frames` normalized frames at `fps`, every (fps / sampleFps)-th, Python-rounded and deduplicated, padded to an
  // even count by repeating the last; per block the indices of its pair's two frames and its timestamp, the mean of
  // the pair's sample times (sample i at i / sampleFps), which the presentation renders "<{:.1f} seconds>". At 24
  // fps, 124 frames: (0, 12) 0.25, (24, 36) 1.25, ..., (120, 120) 5.0. Throws std::invalid_argument for a video of
  // fewer than two samples.
  static constexpr double kReferenceFps = 24.0, kVideoSampleFps = 2.0;
  struct VideoBlockFrames {
    uint32_t first = 0, second = 0;
    double seconds = 0;
  };
  [[nodiscard]] static std::vector<VideoBlockFrames> videoBlockFrames(uint32_t frames, double fps = kReferenceFps,
                                                                      double sampleFps = kVideoSampleFps);
  // Qwen3-VL's get_rope_index for one presentation: (t, h, w) per row [tokens][3]; text rows advance one position
  // on all three axes, an image block's pads take (s, s + row, s + column) of its merged grid and advance the next
  // row by max(rows, columns).
  [[nodiscard]] static std::vector<uint32_t> positions(std::span<const uint32_t> tokens, std::span<const Image> images);

  // The last encode's vision tower: seconds (weights read and converted, and the GPU), the Metal memory it allocated
  // at its peak above what was allocated before it, and its rows.
  struct VisionTiming {
    double seconds = 0, loadSeconds = 0;
    uint64_t peakBytes = 0;
    uint32_t rows = 0, images = 0;
  };
  [[nodiscard]] const VisionTiming &lastVision() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::video::model
