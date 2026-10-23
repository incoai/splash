#pragma once

// The video's decoding (SGLang Diffusion's DecodingStage): the final video latents, de-normalised, through the video
// VAE decoder (ops/VaeDecoder.hpp) into the clip's uint8 frames.
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class DecodingStage final : public PipelineStage {
public:
  // weights: the converted weights' root.
  DecodingStage(metal::MetalBackend &backend, std::filesystem::path weights)
      : backend_(backend), weights_(std::move(weights)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Decode; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
};

} // namespace splash::video
