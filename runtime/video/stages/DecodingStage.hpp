#pragma once

// The video's decoding (SGLang Diffusion's DecodingStage): the final video latents, de-normalised, through the video
// VAE decoder (video/ops/VaeDecoder.hpp) into the clip's uint8 frames.
#include "video/metal/DeviceProfile.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class DecodingStage final : public PipelineStage {
public:
  // weights: the converted weights' root.
  // profile: the device's row, which must outlive the stage.
  DecodingStage(metal::MetalBackend &backend, const device::Profile &profile, std::filesystem::path weights)
      : backend_(backend), profile_(&profile), weights_(std::move(weights)) {}

  [[nodiscard]] wire::VideoPhase phase() const override { return wire::VideoPhase::Decode; }
  void forward(VideoBatch &batch, StageEvents &events) override;

private:
  metal::MetalBackend &backend_;
  const device::Profile *profile_;
  std::filesystem::path weights_;
};

} // namespace splash::video
