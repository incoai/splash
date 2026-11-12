#pragma once

// A stage of the video runtime (SGLang Diffusion's PipelineStage): one phase of a request, which reads what the
// stages before it left in the batch and leaves its results there. The runtime (VideoRuntime.hpp) runs its stages in
// order and announces each stage's phase as it starts and as it completes; a stage reports the steps in between and
// its outputs, and stops at a step boundary once the request is cancelled.
#include "engine/Protocol.hpp"
#include "video/pipeline/VideoBatch.hpp"

#include <cstdint>
#include <vector>

namespace splash::video {

// A request stopped by its Cancel.
struct Cancelled final {};

// What the request's client hears while a stage runs.
class StageEvents {
public:
  virtual ~StageEvents() = default;
  // `step` of the stage's steps are done.
  virtual void progress(uint32_t step) = 0;
  virtual void output(protocol::VideoOutputKind kind, uint32_t frames, std::vector<uint8_t> mp4) = 0;
  [[nodiscard]] virtual bool cancelled() const = 0;
};

class PipelineStage {
public:
  virtual ~PipelineStage() = default;
  [[nodiscard]] virtual protocol::VideoPhase phase() const = 0;
  // Whether the stage runs for the request, and the steps its phase reports.
  [[nodiscard]] virtual bool runs(const VideoBatch &) const { return true; }
  [[nodiscard]] virtual uint32_t steps(const VideoBatch &) const { return 1; }
  virtual void forward(VideoBatch &batch, StageEvents &events) = 0;
};

} // namespace splash::video
