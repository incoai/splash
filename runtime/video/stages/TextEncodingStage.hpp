#pragma once

// The prompt's text encoding (SGLang Diffusion's TextEncodingStage): Qwen3-VL-32B's prefill through decoder layer 50
// (model/TextEncoder.hpp), whose hidden states condition the DiT.
#include "video/model/TextEncoder.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include <filesystem>

namespace splash::video {

class TextEncodingStage final : public PipelineStage {
public:
  // directory: the encoder's checkpoint, its config.json and shards.
  TextEncodingStage(metal::MetalBackend &backend, const std::filesystem::path &directory)
      : encoder_(backend, directory) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Encode; }
  void forward(VideoBatch &batch, StageEvents &) override { batch.hidden = encoder_.encode(batch.promptTokens); }

private:
  model::TextEncoder encoder_;
};

} // namespace splash::video
