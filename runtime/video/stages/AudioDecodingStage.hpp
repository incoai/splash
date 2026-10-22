#pragma once

// The audio's decoding, when the request asks for it: the final audio latents through the audio VAE's BigVGAN decoder
// (ops/AudioDecoder.hpp) into the clip's stereo waveform at 32 kHz.
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class AudioDecodingStage final : public PipelineStage {
public:
  // weights: the converted weights' root.
  AudioDecodingStage(metal::MetalBackend &backend, std::filesystem::path weights)
      : backend_(backend), weights_(std::move(weights)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Audio; }
  [[nodiscard]] bool runs(const VideoBatch &batch) const override { return batch.audio; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
};

} // namespace splash::video
