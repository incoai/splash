#include "engine/NativeRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace splash::engine {
namespace {

static_assert(protocol::kMaximumScoreOptions ==
                  model::ExecutionLimits::maximumScoreOptions,
              "native protocol and model score option bounds must match");
static_assert(protocol::kMinimumScoreOptions ==
                  model::ExecutionLimits::minimumScoreOptions,
              "native protocol and model score option bounds must match");

RequestPriority mapPriority(protocol::RequestPriority priority) {
  switch (priority) {
  case protocol::RequestPriority::Foreground:
    return RequestPriority::Foreground;
  case protocol::RequestPriority::Normal:
    return RequestPriority::Normal;
  case protocol::RequestPriority::Background:
    return RequestPriority::Background;
  }
  throw std::invalid_argument("invalid protocol priority");
}

BatchCohort mapCohort(protocol::Cohort cohort) {
  switch (cohort) {
  case protocol::Cohort::Greedy:
    return BatchCohort::Greedy;
  case protocol::Cohort::Sampling:
    return BatchCohort::Sampling;
  case protocol::Cohort::Constrained:
    return BatchCohort::Constrained;
  }
  throw std::invalid_argument("invalid protocol cohort");
}

ConstraintMode mapConstraint(protocol::ConstraintMode constraint) {
  switch (constraint) {
  case protocol::ConstraintMode::None:
    return ConstraintMode::None;
  case protocol::ConstraintMode::TokenMask:
    return ConstraintMode::TokenMask;
  }
  throw std::invalid_argument("invalid protocol constraint mode");
}

protocol::CacheDisposition mapCacheDisposition(EngineCacheStatus status) {
  switch (status) {
  case EngineCacheStatus::Miss:
    return protocol::CacheDisposition::Miss;
  case EngineCacheStatus::PrefixHit:
    return protocol::CacheDisposition::PrefixHit;
  }
  throw std::logic_error("invalid engine cache status");
}

protocol::FinishReason mapFinishReason(EngineFinishReason reason) {
  switch (reason) {
  case EngineFinishReason::Stop:
    return protocol::FinishReason::Stop;
  case EngineFinishReason::Length:
    return protocol::FinishReason::Length;
  case EngineFinishReason::Cancelled:
    return protocol::FinishReason::Cancelled;
  }
  throw std::logic_error("invalid engine finish reason");
}

} // namespace

NativeRuntime::NativeRuntime(NativeLoopConfig config, engine::Cache &cache,
                             model::Model &model, ByteSink output,
                             StatusProvider statusProvider,
                             NativeLoopClocks clocks,
                             protocol::ProtocolLimits limits)
    : config_(std::move(config)), output_(std::move(output)),
      statusProvider_(std::move(statusProvider)), clocks_(std::move(clocks)),
      limits_(limits), parser_(limits_),
      core_(config_.engine, cache, model, *this) {
  if (!config_.engineInstanceId || !config_.maskWordsPerToken || !output_ ||
      !statusProvider_) {
    throw std::invalid_argument("invalid native engine loop config");
  }
  NativeLoopClocks defaults = defaultClocks();
  if (!clocks_.unixMicros) {
    clocks_.unixMicros = std::move(defaults.unixMicros);
  }
  if (!clocks_.monotonicMilliseconds) {
    clocks_.monotonicMilliseconds = std::move(defaults.monotonicMilliseconds);
  }
}

bool NativeRuntime::receive(std::span<const uint8_t> bytes) {
  if (closeConnection_)
    return false;
  size_t offset = 0;
  while (offset < bytes.size() && !closeConnection_) {
    protocol::ParseStep step = parser_.consume(bytes.subspan(offset));
    offset += step.consumedBytes;
    if (step.issue)
      return handleIssue(std::move(*step.issue));
    if (step.frame) {
      // The parser is already past a frame it yielded, so a request-scoped
      // decode failure leaves the frames behind it to be processed.
      auto decoded = protocol::decodeFrame(*step.frame, limits_);
      if (decoded) {
        if (!handle(*decoded.value))
          return false;
      } else if (step.frame->type == protocol::FrameType::MaskResponse &&
                 decoded.issue->failureClass ==
                     protocol::FailureClass::RequestError) {
        if (!handleMaskIssue(std::move(*decoded.issue)))
          return false;
      } else {
        if (step.frame->type == protocol::FrameType::Request &&
            telemetry_.contains(decoded.issue->requestId)) {
          decoded.issue->failureClass = protocol::FailureClass::ProtocolFatal;
        }
        if (!handleIssue(std::move(*decoded.issue)))
          return false;
      }
    } else if (!step.consumedBytes) {
      engineError("protocol_stalled", "protocol parser made no input progress");
      return false;
    }
  }
  return !closeConnection_;
}

bool NativeRuntime::finishInput() {
  if (closeConnection_)
    return false;
  if (auto issue = parser_.finish()) {
    return handleIssue(std::move(*issue));
  }
  return true;
}

bool NativeRuntime::tick() {
  if (closeConnection_ || !engineHealthy_)
    return false;
  try {
    return core_.tick(clocks_.monotonicMilliseconds());
  } catch (...) {
    executionFailed(std::current_exception());
  }
  return false;
}

bool NativeRuntime::runControl(const std::function<bool()> &control) {
  if (closeConnection_ || !engineHealthy_)
    return false;
  try {
    return control ? control() : false;
  } catch (...) {
    executionFailed(std::current_exception());
  }
  return false;
}

void NativeRuntime::executionFailed(std::exception_ptr failure) {
  try {
    std::rethrow_exception(failure);
  } catch (const metal::MetalBackendError &error) {
    if (config_.metrics)
      config_.metrics->metalFailed();
    engineError("metal_execution_failed", error.what());
  } catch (const std::exception &error) {
    engineError("engine_execution_failed", error.what());
  } catch (...) {
    engineError("engine_execution_failed", "unknown engine exception");
  }
}

void NativeRuntime::announceReady() {
  if (closeConnection_ || !engineHealthy_) {
    throw std::logic_error("unhealthy engine cannot become ready");
  }
  if (ready_)
    throw std::logic_error("ready was already announced");
  if (!send(protocol::ReadyEvent{config_.engineInstanceId,
                                 model::ExecutionLimits::maximumBatchWidth,
                                 config_.engine.maxContext,
                                 protocol::kNativeFeatureBits})) {
    throw std::runtime_error("failed to serialize ready event");
  }
  ready_ = true;
}

std::optional<double> NativeRuntime::millisecondsUntilNextWakeup() const {
  auto wakeup = core_.nextWakeupMilliseconds();
  if (!wakeup)
    return std::nullopt;
  double now = clocks_.monotonicMilliseconds();
  if (!std::isfinite(now))
    return 0.0;
  return std::max(0.0, *wakeup - now);
}

bool NativeRuntime::handle(protocol::Message &message) {
  return std::visit(
      [&](auto &typed) -> bool {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, protocol::RequestFrame>) {
          return handleRequest(typed);
        } else if constexpr (std::is_same_v<T, protocol::CancelFrame>) {
          return handleCancel(typed);
        } else if constexpr (std::is_same_v<T, protocol::MaskResponseFrame>) {
          return handleMask(typed);
        } else if constexpr (std::is_same_v<T, protocol::StatusRequestFrame>) {
          return handleStatus(typed);
        } else {
          protocol::ProtocolIssue issue;
          issue.failureClass = protocol::FailureClass::ProtocolFatal;
          issue.code = protocol::IssueCode::InvalidEnumValue;
          issue.message = "client sent a server-only native protocol message";
          return handleIssue(std::move(issue));
        }
      },
      message);
}

bool NativeRuntime::handleRequest(protocol::RequestFrame &request) {
  if (telemetry_.contains(request.requestId)) {
    return handleIssue({protocol::FailureClass::ProtocolFatal,
                        protocol::IssueCode::InvalidRequestId,
                        request.requestId, "request id is already active"});
  }
  if (!ready_) {
    requestError(request.requestId, "engine_not_ready",
                 "engine warmup has not completed", true);
    return true;
  }
  uint64_t nowUnix = clocks_.unixMicros();
  double nowMonotonic = clocks_.monotonicMilliseconds();
  if (!std::isfinite(nowMonotonic) ||
      request.absoluteDeadlineUnixMicros <= nowUnix) {
    requestError(request.requestId, "deadline_exceeded",
                 "request deadline elapsed before admission");
    return true;
  }
  uint64_t absoluteRemaining = request.absoluteDeadlineUnixMicros - nowUnix;
  uint64_t remaining =
      std::min(absoluteRemaining, request.remainingDeadlineMicros);
  if (!remaining) {
    requestError(request.requestId, "deadline_exceeded",
                 "request deadline elapsed before admission");
    return true;
  }

  // The frame dies with this handler, so its large payloads (prompt
  // tokens, image pixels) move into the engine request. Error paths below
  // only read the request id.
  try {
    EngineRequest engineRequest;
    engineRequest.id = request.requestId;
    engineRequest.priority = mapPriority(request.priority);
    engineRequest.cohort = mapCohort(request.cohort);
    engineRequest.prompt = std::move(request.promptTokens);
    engineRequest.images.reserve(request.imageSpans.size());
    for (const protocol::ImageSpanFrame &span : request.imageSpans) {
      engineRequest.images.push_back({span.offset, span.tokens, span.gridHeight,
                                      span.gridWidth, span.digestLo,
                                      span.digestHi});
    }
    engineRequest.imagePixels = std::move(request.imagePixels);
    engineRequest.maxNewTokens = request.logicalMaxOutputTokens;
    engineRequest.scoreTokens = std::move(request.scoreTokens);
    engineRequest.sampling = {request.sampling.temperature,
                              request.sampling.topP, request.sampling.topK,
                              request.seed};
    engineRequest.constraint = mapConstraint(request.constraint);
    engineRequest.returnProgress = request.returnProgress;
    engineRequest.deadlineMilliseconds =
        nowMonotonic + double(remaining) / 1000.0;
    core_.submit(std::move(engineRequest));
  } catch (const std::invalid_argument &error) {
    requestError(request.requestId, "invalid_request", error.what());
    return true;
  } catch (const std::exception &error) {
    engineError("request_admission_failed", error.what());
    return false;
  } catch (...) {
    engineError("request_admission_failed",
                "unknown request admission exception");
    return false;
  }
  try {
    auto [_, inserted] = telemetry_.emplace(
        request.requestId,
        RequestTelemetry{.arrivedMilliseconds = nowMonotonic});
    if (!inserted) {
      throw std::logic_error("accepted request already has native telemetry");
    }
  } catch (const std::exception &error) {
    // Core admission has already committed. Any failure in the matching
    // native registry is an engine invariant failure; treating
    // it as a bad client request would continue with split ownership.
    engineError("request_admission_failed", error.what());
    return false;
  } catch (...) {
    engineError("request_admission_failed",
                "unknown request telemetry exception");
    return false;
  }
  return true;
}

bool NativeRuntime::handleCancel(const protocol::CancelFrame &cancel) {
  if (!telemetry_.contains(cancel.requestId)) {
    // Cancel can arrive after the terminal event; treat it as a no-op.
    return true;
  }
  try {
    core_.cancel(cancel.requestId);
  } catch (const std::exception &error) {
    // A decoded cancel for live telemetry has no remaining client-side
    // semantic failure. Core cancellation/release exceptions indicate
    // inconsistent engine ownership and must stop this process.
    engineError("cancel_failed", error.what());
    return false;
  } catch (...) {
    engineError("cancel_failed", "unknown cancellation exception");
    return false;
  }
  return true;
}

bool NativeRuntime::handleMask(const protocol::MaskResponseFrame &mask) {
  if (!telemetry_.contains(mask.requestId)) {
    // A CPU mask calculation can finish after the request ends.
    return true;
  }
  auto failMaskRequest = [&](std::string code, std::string message) -> bool {
    try {
      core_.failRequest(mask.requestId, std::move(code), std::move(message));
      return true;
    } catch (const std::exception &error) {
      engineError("mask_response_failure", error.what());
    } catch (...) {
      engineError("mask_response_failure", "unknown mask terminal exception");
    }
    return false;
  };

  auto found = pendingMasks_.find(mask.requestId);
  if (found == pendingMasks_.end() ||
      found->second.maskRequestId != mask.maskRequestId ||
      found->second.expectedWords != mask.maskWords.size()) {
    return failMaskRequest("invalid_mask_response",
                           "mask response does not match the pending request");
  }
  try {
    core_.provideMask(mask.requestId, mask.maskWords);
    pendingMasks_.erase(found);
  } catch (const std::invalid_argument &error) {
    // Correctly framed mask contents (for example an all-zero row) are a
    // request-scoped semantic error. Internal state/allocator failures
    // are handled below as engine-unhealthy.
    return failMaskRequest("invalid_mask_response", error.what());
  } catch (const std::exception &error) {
    engineError("mask_response_failure", error.what());
    return false;
  } catch (...) {
    engineError("mask_response_failure", "unknown token-mask exception");
    return false;
  }
  return true;
}

bool NativeRuntime::handleStatus(const protocol::StatusRequestFrame &status) {
  try {
    std::string json = statusProvider_();
    if (json.empty())
      throw std::runtime_error("empty status document");
    return send(protocol::StatusJsonEvent{
        status.correlationId, protocol::kStatusSchemaVersion, std::move(json)});
  } catch (const std::exception &error) {
    engineError("status_failed", error.what());
    return false;
  }
}

bool NativeRuntime::handleMaskIssue(protocol::ProtocolIssue issue) {
  if (!issue.requestId)
    return handleIssue(std::move(issue));
  if (!telemetry_.contains(issue.requestId)) {
    // Once framing establishes the request id, ignore late mask responses
    // for requests that have already ended, including invalid mask contents.
    return true;
  }
  try {
    core_.failRequest(issue.requestId,
                      std::string(protocol::issueCodeName(issue.code)),
                      std::move(issue.message));
  } catch (const std::exception &error) {
    engineError("mask_response_failure", error.what());
    return false;
  } catch (...) {
    engineError("mask_response_failure",
                "unknown malformed mask response failure");
    return false;
  }
  return !closeConnection_;
}

bool NativeRuntime::handleIssue(protocol::ProtocolIssue issue) {
  protocol::FailureClass classification = issue.failureClass;
  uint64_t requestId = issue.requestId;
  if (classification == protocol::FailureClass::RequestError && !requestId) {
    classification = protocol::FailureClass::ProtocolFatal;
  }
  send(protocol::ErrorEvent{
      classification,
      classification == protocol::FailureClass::RequestError ? requestId : 0,
      false, std::string(protocol::issueCodeName(issue.code)),
      std::move(issue.message)});
  if (protocol::connectionMustClose(classification)) {
    closeConnection_ = true;
    if (classification == protocol::FailureClass::EngineUnhealthy) {
      engineHealthy_ = false;
    }
    return false;
  }
  return true;
}

void NativeRuntime::requestError(uint64_t requestId, std::string code,
                                 std::string message, bool retryable) {
  send(protocol::ErrorEvent{protocol::FailureClass::RequestError, requestId,
                            retryable, std::move(code), std::move(message)});
}

void NativeRuntime::engineError(std::string code, std::string message) {
  send(protocol::ErrorEvent{protocol::FailureClass::EngineUnhealthy, 0, false,
                            std::move(code), std::move(message)});
  engineHealthy_ = false;
  closeConnection_ = true;
}

bool NativeRuntime::send(protocol::Message message) {
  if (closeConnection_)
    return false;
  auto serialized = protocol::serializeMessage(message, limits_);
  if (!serialized) {
    // An event the engine cannot put on the wire is an engine defect. Report
    // it once and stop the stream, so the client sees the cause instead of a
    // later frame that contradicts the missing one.
    engineHealthy_ = false;
    closeConnection_ = true;
    auto report = protocol::serializeMessage(
        protocol::ErrorEvent{protocol::FailureClass::EngineUnhealthy, 0, false,
                             "protocol_encode_failed",
                             serialized.issue->message},
        limits_);
    if (report) {
      try {
        output_(*report.value);
      } catch (...) {
      }
    }
    return false;
  }
  try {
    output_(*serialized.value);
  } catch (...) {
    engineHealthy_ = false;
    closeConnection_ = true;
    return false;
  }
  return true;
}

void NativeRuntime::batchCompleted(WorkKind kind, uint32_t width,
                                   uint32_t inputTokens, uint32_t outputTokens,
                                   uint32_t draftedTokens,
                                   uint32_t acceptedDraftTokens,
                                   double wallMilliseconds) {
  if (config_.metrics) {
    config_.metrics->batchCompleted(kind, width, inputTokens, outputTokens,
                                    draftedTokens, acceptedDraftTokens,
                                    wallMilliseconds);
  }
}

void NativeRuntime::started(uint64_t requestId, EngineCacheStatus cacheStatus,
                            uint32_t matchedTokens, uint32_t stateSlot) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  telemetry.startedMilliseconds = clocks_.monotonicMilliseconds();
  send(protocol::StartEvent{requestId, mapCacheDisposition(cacheStatus),
                            static_cast<int32_t>(stateSlot), matchedTokens,
                            config_.engine.maxContext});
}

void NativeRuntime::promptProgress(uint64_t requestId,
                                   uint32_t processedTokens) {
  const auto &telemetry = telemetry_.at(requestId);
  send(protocol::PromptProgressEvent{
      requestId, processedTokens,
      durationMicros(telemetry.startedMilliseconds,
                     clocks_.monotonicMilliseconds())});
}

void NativeRuntime::tokens(uint64_t requestId,
                           std::span<const uint32_t> values) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  double now = clocks_.monotonicMilliseconds();
  if (!telemetry.firstTokenMilliseconds) {
    telemetry.firstTokenMilliseconds = now;
  }
  uint32_t offset = telemetry.emittedTokens;
  telemetry.emittedTokens += static_cast<uint32_t>(values.size());
  if (config_.metrics) {
    config_.metrics->tokens(telemetry.arrivedMilliseconds,
                            telemetry.lastTokenMilliseconds,
                            static_cast<uint32_t>(values.size()), now);
  }
  telemetry.lastTokenMilliseconds = now;
  send(protocol::TokensEvent{
      requestId, offset, std::vector<uint32_t>(values.begin(), values.end())});
}

void NativeRuntime::maskRequested(uint64_t requestId,
                                  std::span<const uint32_t> simulationTokens) {
  if (pendingMasks_.contains(requestId)) {
    throw std::logic_error("request already has a pending token mask");
  }
  uint64_t maskRequestId = nextMaskRequestId_++;
  if (!maskRequestId)
    maskRequestId = nextMaskRequestId_++;
  uint64_t maskRows = uint64_t(simulationTokens.size()) + 1;
  uint64_t expectedWords = uint64_t(config_.maskWordsPerToken) * maskRows;
  if (!expectedWords || expectedWords > limits_.maxMaskWords) {
    throw std::length_error("token mask dimensions exceed wire limits");
  }
  pendingMasks_.emplace(requestId, PendingMask{maskRequestId, expectedWords});
  send(protocol::MaskRequestEvent{
      requestId, maskRequestId, config_.maskWordsPerToken,
      std::vector<uint32_t>(simulationTokens.begin(), simulationTokens.end())});
}

void NativeRuntime::completed(uint64_t requestId, EngineFinishReason reason,
                              uint32_t promptTokens, uint32_t completionTokens,
                              std::span<const float> optionLogits) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  double now = clocks_.monotonicMilliseconds();
  double started = telemetry.startedMilliseconds > 0.0
                       ? telemetry.startedMilliseconds
                       : telemetry.arrivedMilliseconds;
  double first = telemetry.firstTokenMilliseconds.value_or(now);
  send(protocol::DoneEvent{
      requestId, mapFinishReason(reason), promptTokens, completionTokens,
      durationMicros(started, first),
      telemetry.firstTokenMilliseconds ? durationMicros(first, now) : 0,
      durationMicros(telemetry.arrivedMilliseconds, now),
      std::vector<float>(optionLogits.begin(), optionLogits.end())});
  pendingMasks_.erase(requestId);
  telemetry_.erase(requestId);
}

void NativeRuntime::failed(uint64_t requestId, std::string code,
                           std::string message, bool retryable) {
  requestError(requestId, std::move(code), std::move(message), retryable);
  pendingMasks_.erase(requestId);
  telemetry_.erase(requestId);
}

void NativeRuntime::capacityExhausted(uint64_t requestId,
                                      uint32_t requiredKvPages,
                                      uint32_t availableKvPages,
                                      uint64_t retryAfterMicros) {
  send(protocol::CapacityExhaustedEvent{requestId, requiredKvPages,
                                        availableKvPages, retryAfterMicros});
  if (config_.metrics) {
    config_.metrics->capacityFailed();
  }
  pendingMasks_.erase(requestId);
  telemetry_.erase(requestId);
}

NativeLoopClocks NativeRuntime::defaultClocks() {
  return {
      [] {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(now).count());
      },
      [] {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::milli>(now).count();
      }};
}

uint64_t NativeRuntime::durationMicros(double startMilliseconds,
                                       double endMilliseconds) {
  if (!std::isfinite(startMilliseconds) || !std::isfinite(endMilliseconds) ||
      endMilliseconds <= startMilliseconds) {
    return 0;
  }
  double micros = (endMilliseconds - startMilliseconds) * 1000.0;
  if (micros >= double(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(micros);
}

} // namespace splash::engine
