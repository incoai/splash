#pragma once

// Request preparation (SGLang Diffusion's conditioning, timestep and latent preparation in one stage,
// video/model/Prep.hpp): the packed layout and its RoPE, the refined text, the schedule and its adaLN tables, the noise,
// and with keyframes their conditioning rows (the video VAE encoder).
#include "video/model/Adapter.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace splash::video {

class PrepStage final : public PipelineStage {
public:
  // weights: the converted weights' root; adapters: in place of the DiT manifest's (model::adapterSpecs).
  PrepStage(metal::MetalBackend &backend, std::filesystem::path weights,
            std::optional<std::vector<model::AdapterSpec>> adapters = std::nullopt)
      : backend_(backend), weights_(std::move(weights)), adapters_(std::move(adapters)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Prepare; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
  std::optional<std::vector<model::AdapterSpec>> adapters_;
};

} // namespace splash::video
