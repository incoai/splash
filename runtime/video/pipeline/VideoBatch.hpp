#pragma once

// One request as it passes through the video runtime's stages (SGLang Diffusion's Req): what the client asked for,
// and what each stage leaves for the next.
#include "video/model/Prep.hpp"
#include "video/model/References.hpp"
#include "video/ops/Geometry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace splash::video {

// The clip the client asked for: its DiT, length and size.
struct ClipOptions {
  std::string dit;  // the DiT's name: its weights are dit-<dit>/ (model::ditDirectory)
  int32_t frames = 73;  // 17 n + 5 at 24 fps: 3 s = 73, 5 s = 124, 10 s = 243, 15 s = 362
  int32_t width = 1344, height = 768;  // a canvas of the 768p or 480p tier (ops::canvasTier)
};

// A Ref2VA reference as the server normalized it (protocol::VideoReference): an image, a video's frames at 24 fps on its
// tier canvas with its soundtrack, or an audio reference's samples alone.
struct ClipReference {
  ops::ReferenceShape::Kind kind = ops::ReferenceShape::Kind::Image;
  int32_t frames = 1, width = 0, height = 0;
  std::vector<uint8_t> rgb;  // uint8 RGB [frames][height][width][3] (empty: audio)
  std::vector<float> wav;    // fp32 [2][samples] at 32 kHz, or empty
};

struct VideoBatch {
  uint64_t requestId = 0;
  ClipOptions options;
  ops::VideoGeometry geometry;
  std::vector<uint32_t> promptTokens;
  int32_t steps = 0;  // a VDN-H3 DiT's steps; a FastH3 DiT's 0 or its rungs' count
  int32_t maxForwards = 0;  // diagnostics (video-golden --forwards N): stop the denoising after N forwards; 0: all
  uint64_t seed = 0;
  // Keyframes on the canvas, RGB [height][width][3] (empty: none): the video VAE encoder's input as is.
  std::vector<uint8_t> firstFrame, lastFrame;
  // Ref2VA references in presentation order (none with keyframes): the text encoder's vision blocks and PrepStage's
  // reference encoders take them; PrepStage releases them once encoded.
  std::vector<ClipReference> references;
  bool preview = false, audio = false;

  std::vector<float> hidden;                      // TextEncodingStage: [rows][5120]
  std::vector<uint8_t> textTags;                  // TextEncodingStage: [rows] 1 text, 0 vision rows; empty: all text
  model::Prepared prepared;                       // PrepStage, until DenoisingStage is done with it
  std::vector<float> videoLatents, audioLatents;  // DenoisingStage: [video rows][96], [audio rows][32]
  std::vector<float> step0Video, step0Audio;      // DenoisingStage: the first forward's velocity, as the latents
  std::vector<uint8_t> frames;                    // DecodingStage: uint8 RGB [frames][height][width][3]
  std::vector<float> wav;                         // AudioDecodingStage: [2][samples] at 32 kHz, or empty
  std::vector<uint8_t> mp4;                       // MuxStage

  // The references as the engine's encoders take them: views into references.
  [[nodiscard]] std::vector<model::RequestReference> requestReferences() const {
    std::vector<model::RequestReference> out;
    for (const ClipReference &r : references) {
      model::RequestReference v;
      v.kind = r.kind;
      v.rgb = r.rgb.empty() ? nullptr : r.rgb.data();
      v.frames = r.frames;
      v.width = r.width;
      v.height = r.height;
      v.wav = r.wav.empty() ? nullptr : r.wav.data();
      v.samples = static_cast<int64_t>(r.wav.size() / 2);
      out.push_back(v);
    }
    return out;
  }
};

} // namespace splash::video
