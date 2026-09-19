#pragma once

#include "model/Model.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace splash::engine {

enum class RequestPriority : uint8_t {
  Foreground = 0,
  Normal = 1,
  Background = 2,
};

struct StepResult final {
  uint64_t requestId = 0;
  uint32_t consumedPromptTokens = 0;
  bool finished = false;
  DecodeStage nextDecodeStage = DecodeStage::Regular;
};

enum class EngineFinishReason : uint8_t { Stop, Length, Cancelled };
enum class EngineCacheStatus : uint8_t { Miss, PrefixHit };

struct EngineRequest final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  BatchCohort cohort = BatchCohort::Greedy;
  std::vector<uint32_t> prompt;
  std::vector<ImageSpan> images;
  std::vector<uint8_t> imagePixels;
  uint32_t maxNewTokens = 0;
  SamplingParameters sampling;
  ConstraintMode constraint = ConstraintMode::None;
  double deadlineMilliseconds = 0.0;
  bool returnProgress = false;
  // Nonempty selects score-only mode: prefill runs to completion, no token is
  // generated, and the raw final-position logits at these ids are returned in
  // the completion callback. maxNewTokens must be zero.
  std::vector<uint32_t> scoreTokens{};

  [[nodiscard]] ModelRequest modelView() const noexcept {
    return {id,        cohort,   prompt,     images, imagePixels,
            maxNewTokens, sampling, constraint, scoreTokens};
  }
};

class EngineEventSink {
public:
  virtual ~EngineEventSink() = default;
  virtual void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, double) = 0;
  virtual void started(uint64_t requestId, EngineCacheStatus cacheStatus,
                       uint32_t matchedTokens, uint32_t stateSlot) = 0;
  virtual void promptProgress(uint64_t, uint32_t) {}
  virtual void tokens(uint64_t requestId, std::span<const uint32_t> tokens) = 0;
  virtual void maskRequested(uint64_t requestId,
                             std::span<const uint32_t> simulationTokens) = 0;
  virtual void completed(uint64_t requestId, EngineFinishReason reason,
                         uint32_t promptTokens, uint32_t completionTokens,
                         std::span<const float> optionLogits) = 0;
  virtual void failed(uint64_t requestId, std::string code, std::string message,
                      bool retryable) = 0;
  virtual void capacityExhausted(uint64_t requestId, uint32_t requiredKvPages,
                                 uint32_t availableKvPages,
                                 uint64_t retryAfterMicros) = 0;
};

} // namespace splash::engine
