#pragma once

// The clip's mp4 (the last stage): an MP4 muxer on AVFoundation (container and the system H.264 / AAC encoders; no
// compute of the engine runs here): uint8 RGB frames -> H.264 High (VideoToolbox, BT.709 tags) at 24 fps, 19.8 Mb/s at
// 1344 x 768 (at least the libx264 CRF 18 quality of the former ffmpeg mux), the fp32 stereo waveform, when there is
// one -> AAC-LC 192 kb/s at 32 kHz, one AVAssetWriter. Every frame is written (no -shortest truncation); both tracks
// start at 0. Bad frames throw std::invalid_argument, a failed write std::runtime_error.
#include "video/pipeline/PipelineStage.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::video {

// frames: uint8 RGB [n][height][width][3]; wav: [2][samples] fp32 in [-1, 1] at 32 kHz, or empty.
[[nodiscard]] std::vector<uint8_t> muxMp4(std::span<const uint8_t> frames, int32_t width, int32_t height,
                                          std::span<const float> wav, uint64_t requestId);

class MuxStage final : public PipelineStage {
public:
  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Mux; }
  void forward(VideoBatch &batch, StageEvents &events) override;
};

} // namespace splash::video
