#pragma once

// Request preparation (SGLang Diffusion's conditioning, timestep and latent preparation in one stage,
// model/Prep.hpp): the packed layout and its RoPE, the refined text, the schedule and its adaLN tables, the noise,
// and with keyframes their conditioning rows (the video VAE encoder).
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class PrepStage final : public PipelineStage {
public:
  // weights: the converted weights' root.
  PrepStage(metal::MetalBackend &backend, std::filesystem::path weights)
      : backend_(backend), weights_(std::move(weights)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Prepare; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
};

} // namespace splash::video
