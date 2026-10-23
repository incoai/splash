#include "video/stages/PreviewStage.hpp"

#include "StderrLine.hpp"
#include "metal/abi/video/Layout.h"
#include "video/model/Latents.hpp"
#include "video/model/PreviewWeights.hpp"
#include "video/stages/MuxStage.hpp"

#include <format>

namespace splash::video {

void PreviewStage::forward(VideoBatch &batch, StageEvents &events) {
  const ops::VideoGeometry &g = batch.geometry;
  ops::PreviewDecoder decoder(backend_, model::loadPreviewDecoder(backend_, (weights_ / "preview").string()),
                              g.latentHeight, g.latentWidth);
  std::vector<float> z(size_t(g.latentFrames) * g.latentHeight * g.latentWidth * H3_VAE_LATENT);
  model::unpatchifyVideo(batch.videoLatents.data(), g, nullptr, nullptr, z.data());
  const int32_t count = ops::previewFrames(g.latentFrames);
  std::vector<uint8_t> frames(size_t(g.width) * g.height * 3 * count);
  ops::PreviewStats stats;
  decoder.decode(z.data(), g.latentFrames, frames.data(), nullptr, stats);
  splash::logLine(std::format("H3 preview {} frames wall {:.2f}s gpu {:.2f}s nonfinite {}", stats.frames, stats.wallS,
                              stats.gpuS, stats.nonfinite));
  events.output(protocol::VideoOutputKind::Preview, static_cast<uint32_t>(count),
                muxMp4(frames, g.width, g.height, {}, batch.requestId));
}

} // namespace splash::video
