#include "video/stages/PrepStage.hpp"

#include "StderrLine.hpp"
#include "video/model/TextEncoder.hpp"

#include <format>

namespace splash::video {

void PrepStage::forward(VideoBatch &batch, StageEvents &) {
  auto keyframe = [](const std::vector<uint8_t> &rgb) { return rgb.empty() ? nullptr : rgb.data(); };
  model::PrepInputs inputs;
  inputs.hidden = batch.hidden.data();
  inputs.textRows = static_cast<int32_t>(batch.hidden.size() / model::TextEncoder::kHiddenSize);
  inputs.steps = batch.steps;
  inputs.seed = batch.seed;
  inputs.firstFrame = keyframe(batch.firstFrame);
  inputs.lastFrame = keyframe(batch.lastFrame);
  model::Prepared &out = batch.prepared;
  model::Prep::run(backend_, weights_.string(), batch.options.variant, batch.geometry, inputs, out);
  splash::logLine(std::format("H3 prep M={} L={} C={} steps={} distinct_t={} total {:.1f}s (layout {:.2f}s, keyframe "
                              "encode {:.2f}s, text+noise {:.2f}s, adaLN tables {:.1f}s reading {:.1f} GB) "
                              "device_MB={:.0f}",
                              out.rows, out.textRows, out.condRows, out.steps, out.table.size(), out.totalS,
                              out.layoutS, out.encodeS, out.textS, out.adalnS, out.adalnReadGb, deviceMb(backend_)));
}

} // namespace splash::video
