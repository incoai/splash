#pragma once

// The TAEH3 preview, when the request asks for it: the final video latents through the TAEH3 decoder
// (video/ops/PreviewDecoder.hpp), sent as an mp4 output of their own before the video decodes.
#include "video/metal/DeviceProfile.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class PreviewStage final : public PipelineStage {
public:
  // weights: the converted weights' root.
  // profile: the device's row, which must outlive the stage.
  PreviewStage(metal::MetalBackend &backend, const device::Profile &profile, std::filesystem::path weights)
      : backend_(backend), profile_(&profile), weights_(std::move(weights)) {}

  [[nodiscard]] wire::VideoPhase phase() const override { return wire::VideoPhase::Preview; }
  [[nodiscard]] bool runs(const VideoBatch &batch) const override { return batch.preview; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  const device::Profile *profile_;
  std::filesystem::path weights_;
};

} // namespace splash::video
