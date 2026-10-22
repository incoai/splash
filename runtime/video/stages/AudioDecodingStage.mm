#include "video/stages/AudioDecodingStage.hpp"

#include "StderrLine.hpp"
#include "video/model/AudioWeights.hpp"

#include <format>

namespace splash::video {

void AudioDecodingStage::forward(VideoBatch &batch, StageEvents &) {
  ops::AudioDecoder decoder(backend_, model::loadAudioDecoder(backend_, (weights_ / "audio").string()));
  const int32_t latents = batch.geometry.audioRows / 2;
  batch.wav.assign(size_t(2) * ops::AudioDecoder::samples(latents), 0.0f);
  ops::AudioStats stats;
  decoder.decode(batch.audioLatents.data(), latents, batch.wav.data(), stats);
  splash::logLine(std::format("H3 audio {} samples x 2 wall {:.2f}s gpu {:.2f}s nonfinite {} device_MB {:.0f}",
                              stats.samples, stats.wallS, stats.gpuS, stats.nonfinite, stats.deviceMb));
}

} // namespace splash::video
