#pragma once

// One request as it passes through the video runtime's stages (SGLang Diffusion's Req): what the client asked for,
// and what each stage leaves for the next.
#include "video/model/Prep.hpp"
#include "video/ops/Geometry.hpp"

#include <cstdint>
#include <vector>

namespace splash::video {

enum class VaePrecision : int32_t { W8 = 0, F16 = 1 };

// How the engine renders a clip: the client's variant, length and size, and the model's fixed execution choices.
struct ClipOptions {
  model::Variant variant = model::Variant::Turbo;
  int32_t frames = 73;  // 17 n + 5 at 24 fps: 3 s = 73, 10 s = 243, 15 s = 362
  int32_t width = 1344, height = 768;
  int32_t windowChunk = 5, windowRadius = 1;  // window softmax; anchor frames first and last (always)
  bool qkfix = true;                           // fp16 Q / K in the QKFIX blocks
  std::vector<int32_t> qkfixBlocks{45, 48, 49};
  // The DiT reads its block weights from SSD into a ring of streamSlots slots: a 15 s clip fits in about 15 GB
  // instead of 32 GB, at the same speed.
  bool stream = true;
  int32_t streamSlots = 8;
  VaePrecision vaePrecision = VaePrecision::W8;
};

struct VideoBatch {
  uint64_t requestId = 0;
  ClipOptions options;
  ops::VideoGeometry geometry;
  std::vector<uint32_t> promptTokens;
  int32_t steps = 0;  // 0: the variant's
  uint64_t seed = 0;
  // Keyframes on the canvas, RGB [height][width][3] (empty: none): the video VAE encoder's input as is.
  std::vector<uint8_t> firstFrame, lastFrame;
  bool preview = false, audio = false;

  std::vector<float> hidden;                      // TextEncodingStage: [rows][5120]
  model::Prepared prepared;                       // PrepStage, until DenoisingStage is done with it
  std::vector<float> videoLatents, audioLatents;  // DenoisingStage: [video rows][96], [audio rows][32]
  std::vector<uint8_t> frames;                    // DecodingStage: uint8 RGB [frames][height][width][3]
  std::vector<float> wav;                         // AudioDecodingStage: [2][samples] at 32 kHz, or empty
  std::vector<uint8_t> mp4;                       // MuxStage
};

} // namespace splash::video
