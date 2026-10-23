#pragma once

// The TAEH3 preview, when the request asks for it: the final video latents through the TAEH3 decoder
// (ops/PreviewDecoder.hpp), sent as an mp4 output of their own before the video decodes.
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class PreviewStage final : public PipelineStage {
public:
  // weights: the converted weights' root.
  PreviewStage(metal::MetalBackend &backend, std::filesystem::path weights)
      : backend_(backend), weights_(std::move(weights)) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Preview; }
  [[nodiscard]] bool runs(const VideoBatch &batch) const override { return batch.preview; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  std::filesystem::path weights_;
};

} // namespace splash::video
