#pragma once

// The prompt's text encoding (SGLang Diffusion's TextEncodingStage): Qwen3-VL-32B's prefill through decoder layer 50
// (video/model/TextEncoder.hpp), whose hidden states condition the DiT.
#include "video/model/TextEncoder.hpp"
#include "video/pipeline/PipelineStage.hpp"

#include "StderrLine.hpp"

#include <filesystem>
#include <format>
#include <vector>

namespace splash::video {

class TextEncodingStage final : public PipelineStage {
public:
  // directory: the encoder's checkpoint, its config.json and shards.
  TextEncodingStage(metal::MetalBackend &backend, const std::filesystem::path &directory)
      : encoder_(backend, directory) {}

  [[nodiscard]] protocol::VideoPhase phase() const override { return protocol::VideoPhase::Encode; }
  // A presentation with vision blocks (the server's for a clip from keyframes: model::TextEncoder::encode) encodes
  // the keyframes' images in its blocks' order, the first frame's and then the last frame's, on the canvas as the
  // VAE encoder takes them; its vision rows are tagged 0 for the DiT (batch.textTags). A Ref2VA presentation's blocks
  // take the references' images and video frame pairs (model::referenceVisionImages), in presentation order.
  void forward(VideoBatch &batch, StageEvents &) override {
    const auto blocks = model::TextEncoder::visionBlocks(batch.promptTokens);
    std::vector<model::TextEncoder::Image> images;
    if (!batch.references.empty())
      images = model::referenceVisionImages(batch.requestReferences());
    else if (!blocks.empty())
      for (const std::vector<uint8_t> *frame : {&batch.firstFrame, &batch.lastFrame})
        if (!frame->empty())
          images.push_back({frame->data(), static_cast<uint32_t>(batch.options.width),
                            static_cast<uint32_t>(batch.options.height)});
    batch.hidden = encoder_.encode(batch.promptTokens, images);
    batch.textTags = blocks.empty() ? std::vector<uint8_t>{} : model::TextEncoder::tags(batch.promptTokens);
    if (!blocks.empty()) {
      const auto &v = encoder_.lastVision();
      splash::logLine(std::format("H3 encode vision: {} images, {} rows, tower {:.2f}s (weights {:.2f}s), tower peak "
                                  "{:.0f} MB above the encode's start",
                                  v.images, v.rows, v.seconds, v.loadSeconds, v.peakBytes / 1048576.0));
    }
  }

private:
  model::TextEncoder encoder_;
};

} // namespace splash::video
