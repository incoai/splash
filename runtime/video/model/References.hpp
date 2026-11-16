#pragma once
// A Ref2VA request's references through the engine's encoders (Pika H3 Flex's conditioning.encode, stage 2): per
// reference in presentation order, a video's frames (normalized by the host: 24 fps, the target's frame count, the
// tier canvas) trimmed down to 17 k + 5 for the video VAE (video/ops/VaeEncoder.hpp encodeVideoRgb), an image at its tier
// size (encodeRgb), each posterior sampled under a fresh seed-42 CPU generator (video/model/ReferenceLatents.hpp), and a
// soundtrack, or an audio reference's waveform, through the audio VAE encoder (video/ops/AudioEncoder.hpp). Each encoder's weights load before its first
// encode and are released after its last, so nothing of them is resident when the DiT loads. Also the text
// encoder's vision blocks of the references (video/model/TextEncoder.hpp: an image block per image, a frame-pair block per
// videoBlockFrames entry of a video's normalized frames).
#include "metal/MetalBackend.hpp"
#include "video/model/Prep.hpp"
#include "video/model/TextEncoder.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace splash::video::model {

// One reference of a request, as the host normalized it (non-owning).
struct RequestReference {
  ops::ReferenceShape::Kind kind = ops::ReferenceShape::Kind::Image;  // Video, Image or Audio
  const uint8_t *rgb = nullptr;  // uint8 RGB [frames][height][width][3] (null for Audio)
  int32_t frames = 1, width = 0, height = 0;
  // fp32 [2][samples] at 32 kHz, truncated to the clip: a video's soundtrack (null: silent) or an audio reference's
  const float *wav = nullptr;
  int64_t samples = 0;
};

struct EncodedReferences {
  std::vector<std::vector<float>> video, audio;  // per reference: [24][F][h][w] / [T][64] (empty when absent)
  std::vector<ReferenceInput> inputs;            // views into video / audio, for PrepInputs::references
  std::vector<int32_t> encodedFrames;            // per reference: the frames the video VAE encoded (17 k + 5)
  double videoS = 0, imageS = 0, audioS = 0, loadS = 0, totalS = 0;
  int64_t nonfinite = 0;
};

// The largest 17 k + 5 <= frames (G.align_num_frames_down); throws for fewer than 22 frames.
[[nodiscard]] int32_t referenceVideoFrames(int32_t frames);

// Encodes every reference (seed: the posterior sample's generator seed, Pika's KEYFRAME_ENCODE_SEED 42).
void encodeReferences(metal::MetalBackend &backend, const std::string &weights,
                      const std::vector<RequestReference> &references, EncodedReferences &out, uint64_t seed = 42);

// The presentation's vision blocks of these references, in token order (views into the references' pixels).
[[nodiscard]] std::vector<TextEncoder::Image> referenceVisionImages(const std::vector<RequestReference> &references);

} // namespace splash::video::model
