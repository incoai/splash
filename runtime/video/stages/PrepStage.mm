#include "video/stages/PrepStage.hpp"

#include "StderrLine.hpp"
#include "video/model/FastH3.hpp"
#include "video/model/TextEncoder.hpp"

#include <format>
#include <stdexcept>

namespace splash::video {

void PrepStage::forward(VideoBatch &batch, StageEvents &) {
  auto keyframe = [](const std::vector<uint8_t> &rgb) { return rgb.empty() ? nullptr : rgb.data(); };
  model::PrepInputs inputs;
  inputs.hidden = batch.hidden.data();
  inputs.textRows = static_cast<int32_t>(batch.hidden.size() / model::TextEncoder::kHiddenSize);
  if (!batch.textTags.empty()) {
    if (batch.textTags.size() != size_t(inputs.textRows))
      throw std::invalid_argument("text tags do not match the hidden states' rows");
    inputs.textTags = batch.textTags.data();
  }
  inputs.steps = batch.steps;
  inputs.seed = batch.seed;
  inputs.firstFrame = keyframe(batch.firstFrame);
  inputs.lastFrame = keyframe(batch.lastFrame);
  inputs.adapters = model::adapterSpecs(weights_ / model::ditDirectory(batch.options.dit), adapters_);
  // Ref2VA: the references through the video and audio VAEs' encoders (posterior sample of seed 42, Pika's
  // KEYFRAME_ENCODE_SEED), whose weights are released before the DiT's load; their pixels go once encoded.
  model::EncodedReferences references;
  if (!batch.references.empty()) {
    model::encodeReferences(backend_, weights_.string(), batch.requestReferences(), references, 42);
    inputs.references = references.inputs;
    splash::logLine(std::format("H3 references: {} encoded in {:.1f}s (video {:.1f}s, images {:.1f}s, audio {:.1f}s, "
                                "weights {:.1f}s)",
                                batch.references.size(), references.totalS, references.videoS, references.imageS,
                                references.audioS, references.loadS));
  }
  model::Prepared &out = batch.prepared;
  model::Prep::run(backend_, weights_.string(), batch.options.dit, batch.geometry, inputs, out);
  std::vector<ClipReference>().swap(batch.references);
  splash::logLine(std::format("H3 prep M={} L={} C={} steps={} distinct_t={} total {:.1f}s (layout {:.2f}s, keyframe "
                              "encode {:.2f}s, text+noise {:.2f}s, adaLN tables {:.1f}s reading {:.1f} GB) "
                              "device_MB={:.0f}",
                              out.rows, out.textRows, out.condRows, out.steps, out.table.size(), out.totalS,
                              out.layoutS, out.encodeS, out.textS, out.adalnS, out.adalnReadGb, deviceMb(backend_)));
}

} // namespace splash::video
