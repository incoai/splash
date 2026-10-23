#include "video/stages/DecodingStage.hpp"

#include "StderrLine.hpp"
#include "metal/abi/video/Layout.h"
#include "video/model/Latents.hpp"
#include "video/model/VaeWeights.hpp"

#include <cstring>
#include <format>

namespace splash::video {

void DecodingStage::forward(VideoBatch &batch, StageEvents &) {
  const ops::VideoGeometry &g = batch.geometry;
  const ops::VaePlan plan = ops::planVaeDecode(g);
  const char *tag = batch.options.vaePrecision == VaePrecision::W8 ? "w8" : "f16";
  ops::VaeDecoder decoder(backend_, model::loadVaeDecoder(backend_, (weights_ / "vae").string(), tag));
  // de-normalised channels-last latents, the last frame repeated over the plan's pad tokens
  const size_t frameFloats = size_t(g.latentHeight) * g.latentWidth * H3_VAE_LATENT;
  std::vector<float> z(frameFloats * (g.latentFrames + plan.padTokens), 0.0f);
  float mean[24], std[24];
  for (int c = 0; c < 24; ++c) {
    mean[c] = static_cast<float>(model::kVaeLatentMean[c]);
    std[c] = static_cast<float>(model::kVaeLatentStd[c]);
  }
  model::unpatchifyVideo(batch.videoLatents.data(), g, std, mean, z.data());
  for (int32_t k = 0; k < plan.padTokens; ++k)
    std::memcpy(z.data() + (g.latentFrames + k) * frameFloats, z.data() + (g.latentFrames - 1) * frameFloats,
                frameFloats * 4);
  batch.frames.assign(size_t(g.width) * g.height * 3 * plan.outputFrames(), 0);
  ops::DecodeStats stats;
  decoder.decode(z.data(), g.latentFrames + plan.padTokens, g, plan, batch.frames.data(), stats);
  splash::logLine(std::format("H3 decode {} frames wall {:.1f}s gpu {:.1f}s nonfinite {} device_MB {:.0f}",
                              plan.outputFrames(), stats.wallS, stats.gpuS, stats.nonfinite, stats.deviceMb));
}

} // namespace splash::video
