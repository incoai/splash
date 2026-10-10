#pragma once

#include "model/Model.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::engine {

// Values are the native request frame's priority byte.
enum class RequestPriority : uint8_t {
  Foreground = 0,
  Normal = 1,
  Background = 2,
};

// A score request's full-vocabulary readout at its final position, beside
// its option logits: log-sum-exp over the vocabulary and the highest logits.
struct ScoreReadout final {
  double logNormalizer = 0.0;
  std::vector<uint32_t> topIds;
  std::vector<float> topLogits;
};

struct StepResult final {
  uint64_t requestId = 0;
  uint32_t consumedPromptTokens = 0;
  bool finished = false;
  DecodeStage nextDecodeStage = DecodeStage::Regular;
};

// Values are the Done event's reason byte.
enum class EngineFinishReason : uint8_t {
  Stop = 0,
  Length = 1,
  Cancelled = 2,
};

struct EngineRequest final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  std::vector<uint32_t> prompt;
  // Trailing prompt tokens a later request may not share (a chat template's
  // generation prompt, which the next turn may render differently), so
  // reusable state is kept before them. Zero when unknown; it must leave at
  // least one prompt token.
  uint32_t generationPromptTokens = 0;
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
  // RequestFlag bits.
  uint32_t flags = 0;

  [[nodiscard]] ModelRequest modelView() const noexcept {
    return {id,       prompt,     images,      imagePixels, maxNewTokens,
            sampling, constraint, scoreTokens, flags};
  }
};

// How a request ends without finishing: decided by the engine, reported by
// the protocol adapter from this one table.
enum class LaneOutcome : uint8_t {
  // Completes with EngineFinishReason::Cancelled; never sent as an error.
  Cancelled,
  DeadlineExceeded,
  ResourceTimeout,
  // A lone lane that cannot fit even after every cached prefix went:
  // retrying the same request fails the same way.
  CapacityExhausted,
  ModelResultInvalid,
  InvalidMask,
  // The server left a token-mask request unanswered (Engine.cpp's limit).
  MaskTimeout,
};

struct LaneOutcomeWire final {
  std::string_view code;
  bool retryable;
};

[[nodiscard]] constexpr LaneOutcomeWire
laneOutcomeWire(LaneOutcome outcome) noexcept {
  switch (outcome) {
  case LaneOutcome::Cancelled:
    return {"cancelled", false};
  case LaneOutcome::DeadlineExceeded:
    return {"deadline_exceeded", false};
  case LaneOutcome::ResourceTimeout:
    return {"resource_timeout", true};
  case LaneOutcome::CapacityExhausted:
    return {"capacity_exhausted", false};
  case LaneOutcome::ModelResultInvalid:
    return {"model_result_invalid", false};
  case LaneOutcome::InvalidMask:
    return {"invalid_mask_response", false};
  case LaneOutcome::MaskTimeout:
    return {"mask_timeout", true};
  }
}

inline constexpr std::string_view kDeadlineExceededMessage =
    "request deadline exceeded";

class EngineEventSink {
public:
  virtual ~EngineEventSink() = default;
  // Kind, width, input, output, drafted and accepted tokens, then the
  // command's wall time and the engine's cycle for it: from the previous
  // command's retirement, or from its plan when the engine was idle, to its
  // own retirement.
  virtual void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, double, double) = 0;
  virtual void started(uint64_t requestId, uint32_t matchedTokens,
                       uint32_t lane) = 0;
  virtual void promptProgress(uint64_t, uint32_t) {}
  virtual void tokens(uint64_t requestId, std::span<const uint32_t> tokens) = 0;
  virtual void maskRequested(uint64_t requestId,
                             std::span<const uint32_t> simulationTokens) = 0;
  virtual void completed(uint64_t requestId, EngineFinishReason reason,
                         uint32_t promptTokens, uint32_t completionTokens,
                         std::span<const float> optionLogits,
                         const ScoreReadout &score) = 0;
  virtual void failed(uint64_t requestId, LaneOutcome outcome,
                      std::string message) = 0;
};

} // namespace splash::engine
