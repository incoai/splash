#include "engine/NativeRuntime.hpp"
#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace splash::engine {

uint64_t systemUnixMicros() noexcept {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

NativeRuntime::NativeRuntime(NativeLoopConfig config, engine::Cache &cache,
                             model::Model &model, ByteSink output,
                             protocol::ProtocolLimits limits)
    : config_(std::move(config)), output_(std::move(output)), limits_(limits),
      parser_(limits_), core_(config_.engine, cache, model, *this) {
  if (!output_ || !config_.engine.weights ||
      !config_.unixMicros || !config_.monotonicMilliseconds)
    throw std::invalid_argument("the native engine loop lacks a component it needs");
  if (auto issue = protocol::validateLimits(limits_))
    throw std::invalid_argument(issue->describe());
  core_.startIdleClock(config_.monotonicMilliseconds());
}

void NativeRuntime::serveStatus(StatusProvider provider) {
  if (!provider)
    throw std::invalid_argument("the native engine loop was given no status source");
  statusProvider_ = std::move(provider);
}

bool NativeRuntime::receive(std::span<const uint8_t> bytes) {
  if (closeConnection_)
    return false;
  try {
    size_t offset = 0;
    while (offset < bytes.size() && !closeConnection_) {
      protocol::ParseStep step = parser_.consume(bytes.subspan(offset));
      offset += step.consumedBytes;
      if (step.issue)
        return handleIssue(std::move(*step.issue));
      if (!step.frame)
        continue;
      // The parser is already past a frame it yielded, so a request-scoped
      // decode failure leaves the frames behind it to be processed.
      const protocol::FrameType type = step.frame->type;
      auto decoded = protocol::decodeFrame(std::move(*step.frame), limits_);
      if (decoded) {
        if (!handle(*decoded.value))
          return false;
      } else if (type == protocol::FrameType::MaskResponse &&
                 decoded.issue->failureClass ==
                     protocol::FailureClass::RequestError) {
        if (!handleMaskIssue(std::move(*decoded.issue)))
          return false;
      } else {
        if (type == protocol::FrameType::Request &&
            telemetry_.contains(decoded.issue->requestId)) {
          decoded.issue->failureClass = protocol::FailureClass::ProtocolFatal;
        }
        if (!handleIssue(std::move(*decoded.issue)))
          return false;
      }
    }
  } catch (...) {
    executionFailed(std::current_exception());
    return false;
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
    return core_.tick(config_.monotonicMilliseconds());
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

bool NativeRuntime::flushRestorePoints() {
  if (!engineHealthy_)
    return false;
  try {
    return core_.flushRestorePoints();
  } catch (...) {
    executionFailed(std::current_exception());
  }
  return false;
}

void NativeRuntime::releaseIdleWeights() {
  static_cast<void>(core_.releaseIdleWeights(config_.monotonicMilliseconds()));
}

void NativeRuntime::executionFailed(std::exception_ptr failure) {
  if (config_.engineFailed)
    config_.engineFailed();
  try {
    std::rethrow_exception(failure);
  } catch (const metal::MetalBackendError &error) {
    metrics_.metalFailed();
    // A command short of GPU memory says nothing about the requests it ran
    // (server/supervision.py Quarantine counts the other failures).
    engineError(error.outOfMemory() ? "metal_out_of_memory" : "metal_execution_failed",
                error.what());
  } catch (const std::exception &error) {
    engineError("engine_execution_failed", error.what());
  } catch (...) {
    engineError("engine_execution_failed", "unknown engine exception");
  }
}

void NativeRuntime::announceWarming(const protocol::WarmingEvent &progress) {
  // A send that fails closes the connection, which announceReady reports.
  static_cast<void>(send(progress));
}

void NativeRuntime::announceReady(const protocol::ReadyEvent &ready) {
  if (!engineHealthy_)
    throw std::logic_error("unhealthy engine cannot become ready");
  if (ready_)
    throw std::logic_error("ready was already announced");
  // A healthy engine's send fails only when a write failed and closed the
  // connection; one that cannot encode the event is no longer healthy.
  if (!send(ready)) {
    throw std::runtime_error(engineHealthy_ ? "the connection closed before Ready"
                                            : "failed to serialize ready event");
  }
  ready_ = true;
  // The idle release counts from here: warmup is not idleness.
  core_.startIdleClock(config_.monotonicMilliseconds());
}

std::optional<double> NativeRuntime::millisecondsUntilNextWakeup() const {
  auto wakeup = core_.nextWakeupMilliseconds();
  if (!wakeup)
    return std::nullopt;
  return std::max(0.0, *wakeup - config_.monotonicMilliseconds());
}

bool NativeRuntime::handle(protocol::ClientMessage &message) {
  return std::visit(
      [&](auto &typed) -> bool {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, protocol::RequestFrame>) {
          return handleRequest(typed);
        } else if constexpr (std::is_same_v<T, protocol::CancelFrame>) {
          return handleCancel(typed);
        } else if constexpr (std::is_same_v<T, protocol::MaskResponseFrame>) {
          return handleMask(typed);
        } else if constexpr (std::is_same_v<T, protocol::ReleaseRequestFrame>) {
          return handleRelease(typed);
        } else if constexpr (std::is_same_v<T, protocol::StopFrame>) {
          return handleStop(typed);
        } else {
          return handleStatus(typed);
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
  const uint64_t nowUnix = config_.unixMicros();
  const double nowMonotonic = config_.monotonicMilliseconds();
  const uint64_t remaining =
      request.absoluteDeadlineUnixMicros > nowUnix
          ? std::min(request.absoluteDeadlineUnixMicros - nowUnix,
                     request.remainingDeadlineMicros)
          : 0;
  if (!remaining) {
    const LaneOutcomeWire deadline =
        laneOutcomeWire(LaneOutcome::DeadlineExceeded);
    requestError(request.requestId, std::string(deadline.code),
                 std::string(kDeadlineExceededMessage), deadline.retryable);
    return true;
  }

  // The frame dies with this handler, so its large payloads (prompt
  // tokens, image pixels) move into the engine request. Error paths below
  // only read the request id.
  try {
    EngineRequest engineRequest;
    engineRequest.id = request.requestId;
    engineRequest.priority = request.priority;
    engineRequest.prompt = std::move(request.promptTokens);
    engineRequest.generationPromptTokens = request.generationPromptTokens;
    engineRequest.images = std::move(request.imageSpans);
    engineRequest.imagePixels = std::move(request.imagePixels);
    engineRequest.maxNewTokens = request.logicalMaxOutputTokens;
    engineRequest.scoreTokens = std::move(request.scoreTokens);
    engineRequest.sampling = request.sampling;
    engineRequest.constraint = request.constraint;
    engineRequest.flags = request.flags & kRequestFlagBits;
    engineRequest.thinkingBudget = request.thinkingBudget;
    engineRequest.cacheSalt = request.cacheSalt;
    engineRequest.returnProgress =
        request.flags & protocol::kReturnProgressFlag;
    engineRequest.deadlineMilliseconds =
        nowMonotonic + double(remaining) / 1000.0;
    core_.submit(std::move(engineRequest));
  } catch (const std::invalid_argument &error) {
    requestError(request.requestId, "invalid_request", error.what());
    return true;
  }
  telemetry_.emplace(request.requestId,
                     RequestTelemetry{.arrivedMilliseconds = nowMonotonic});
  if (telemetry_.size() == 1 && config_.holdingRequests)
    config_.holdingRequests(true);
  return true;
}

bool NativeRuntime::handleCancel(const protocol::CancelFrame &cancel) {
  // Cancel can arrive after the terminal event; it is then a no-op.
  if (telemetry_.contains(cancel.requestId))
    core_.cancel(cancel.requestId);
  return true;
}

bool NativeRuntime::handleStop(const protocol::StopFrame &stop) {
  // Stop can arrive after the terminal event; it is then a no-op.
  if (telemetry_.contains(stop.requestId))
    core_.stop(stop.requestId);
  return true;
}

bool NativeRuntime::handleMask(const protocol::MaskResponseFrame &mask) {
  if (!telemetry_.contains(mask.requestId)) {
    // A CPU mask calculation can finish after the request ends.
    return true;
  }
  auto found = pendingMasks_.find(mask.requestId);
  if (found == pendingMasks_.end() ||
      found->second.maskRequestId != mask.maskRequestId ||
      found->second.expectedWords != mask.maskWords.size()) {
    core_.failRequest(mask.requestId, LaneOutcome::InvalidMask,
                      "mask response does not match the pending request");
    return true;
  }
  // Erased before the engine reads the mask: unusable contents fail the
  // request there, and its failure event erases the same entry.
  pendingMasks_.erase(found);
  core_.provideMask(mask.requestId, mask.maskWords);
  return true;
}

bool NativeRuntime::handleStatus(const protocol::StatusRequestFrame &status) {
  std::string json = statusProvider_();
  if (json.empty())
    throw std::runtime_error("empty status document");
  return send(protocol::StatusJsonEvent{status.correlationId, std::move(json)});
}

// The weights go back now, as at the end of the idle release, when the engine
// holds no request and no restore is under way, with the idle release on or
// off; otherwise nothing changes. The answer is the status document either way,
// whose weights section tells the caller which: `released` and not `restoring`
// when they are given back, as a restore leaves them `released` until its last
// image is written back.
bool NativeRuntime::handleRelease(const protocol::ReleaseRequestFrame &release) {
  static_cast<void>(core_.releaseWeights(config_.monotonicMilliseconds()));
  return handleStatus(protocol::StatusRequestFrame{release.correlationId});
}

bool NativeRuntime::handleMaskIssue(protocol::ProtocolIssue issue) {
  if (!telemetry_.contains(issue.requestId)) {
    // Once framing establishes the request id, ignore late mask responses
    // for requests that have already ended, including invalid mask contents.
    return true;
  }
  core_.failRequest(issue.requestId, LaneOutcome::InvalidMask,
                    std::string(protocol::issueCodeName(issue.code)) + ": " +
                        issue.message);
  return !closeConnection_;
}

bool NativeRuntime::handleIssue(protocol::ProtocolIssue issue) {
  const protocol::FailureClass classification = issue.failureClass;
  if (classification == protocol::FailureClass::EngineUnhealthy) {
    engineError(std::string(protocol::issueCodeName(issue.code)),
                std::move(issue.message));
    return false;
  }
  send(protocol::ErrorEvent{
      classification,
      classification == protocol::FailureClass::RequestError ? issue.requestId
                                                             : 0,
      false, std::string(protocol::issueCodeName(issue.code)),
      std::move(issue.message)});
  if (protocol::connectionMustClose(classification)) {
    closeConnection_ = true;
    return false;
  }
  return true;
}

void NativeRuntime::requestError(uint64_t requestId, std::string code,
                                 std::string message, bool retryable,
                                 uint32_t retryAfterSeconds) {
  send(protocol::ErrorEvent{protocol::FailureClass::RequestError, requestId,
                            retryable, std::move(code), std::move(message),
                            retryAfterSeconds});
}

void NativeRuntime::engineError(std::string code, std::string message) {
  if (engineFailure_.empty())
    engineFailure_ = code + ": " + message;
  send(protocol::ErrorEvent{protocol::FailureClass::EngineUnhealthy, 0, false,
                            std::move(code), std::move(message)});
  engineHealthy_ = false;
  closeConnection_ = true;
}

bool NativeRuntime::send(const protocol::EngineEvent &event) {
  if (closeConnection_)
    return false;
  auto serialized = protocol::serializeEvent(event, limits_);
  if (!serialized) {
    // An event the engine cannot put on the wire is an engine defect. Report
    // it once and stop the stream, so the client sees the cause instead of a
    // later frame that contradicts the missing one.
    if (config_.engineFailed)
      config_.engineFailed();
    engineHealthy_ = false;
    closeConnection_ = true;
    if (engineFailure_.empty())
      engineFailure_ = "protocol_encode_failed: " + serialized.issue->message;
    auto report = protocol::serializeEvent(
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
    return true;
  } catch (...) {
    // A write that fails ends the connection, not the engine: the transport
    // knows whether the peer closed, is stopping the engine or failed.
    closeConnection_ = true;
    return false;
  }
}

void NativeRuntime::batchCompleted(WorkKind kind, uint32_t width,
                                   uint32_t inputTokens, uint32_t outputTokens,
                                   uint32_t draftedTokens,
                                   uint32_t acceptedDraftTokens,
                                   double wallMilliseconds,
                                   double cycleMilliseconds) {
  metrics_.batchCompleted(kind, width, inputTokens, outputTokens,
                                  draftedTokens, acceptedDraftTokens,
                                  wallMilliseconds, cycleMilliseconds);
}

void NativeRuntime::started(uint64_t requestId, uint32_t matchedTokens,
                            uint32_t lane) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  telemetry.startedMilliseconds = config_.monotonicMilliseconds();
  send(protocol::StartEvent{requestId, lane, matchedTokens});
}

void NativeRuntime::promptProgress(uint64_t requestId,
                                   uint32_t processedTokens) {
  const auto &telemetry = telemetry_.at(requestId);
  send(protocol::PromptProgressEvent{
      requestId, processedTokens,
      durationMicros(telemetry.startedMilliseconds.value(),
                     config_.monotonicMilliseconds())});
}

engine::EngineSnapshot NativeRuntime::statusSnapshot() const {
  engine::EngineSnapshot result = core_.snapshot();
  const double now = config_.monotonicMilliseconds();
  for (engine::ActiveRequestSnapshot &request : result.activeRequests)
    if (const auto found = telemetry_.find(request.id); found != telemetry_.end())
      request.ageMilliseconds = now - found->second.arrivedMilliseconds;
  return result;
}

void NativeRuntime::firstTokenParts(uint64_t requestId, const TtftParts &parts) {
  telemetry_.at(requestId).ttftParts = parts;
}

void NativeRuntime::tokens(uint64_t requestId,
                           std::span<const uint32_t> values) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  double now = config_.monotonicMilliseconds();
  if (!telemetry.firstTokenMilliseconds) {
    telemetry.firstTokenMilliseconds = now;
    // What held it: the engine's parts, and the rest of its TTFT (the
    // host, admission, idle).
    TtftParts &parts = telemetry.ttftParts;
    const double ttft = now - telemetry.arrivedMilliseconds;
    double counted = 0.0;
    for (size_t part = 0; part < kTtftParts; ++part)
      if (part != static_cast<size_t>(TtftPart::Rest))
        counted += parts[part];
    parts[static_cast<size_t>(TtftPart::Rest)] = std::max(0.0, ttft - counted);
    metrics_.ttftParts(parts);
    logLine("TTFT of request ", requestId, ": ", ttft, " ms: own prefill ", parts[0], ", claim wait ", parts[1],
            ", decode ", parts[2], ", other prefill ", parts[3], ", rest ", parts[4]);
  }
  uint32_t offset = telemetry.emittedTokens;
  telemetry.emittedTokens += static_cast<uint32_t>(values.size());
  metrics_.tokens(telemetry.arrivedMilliseconds,
                          telemetry.lastTokenMilliseconds,
                          static_cast<uint32_t>(values.size()), now);
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
  const uint32_t wordsPerToken =
      model::maskWordsPerToken(config_.engine.vocabularySize);
  uint64_t maskRows = uint64_t(simulationTokens.size()) + 1;
  uint64_t expectedWords = uint64_t(wordsPerToken) * maskRows;
  if (!expectedWords || expectedWords > limits_.maxMaskWords) {
    throw std::length_error("token mask dimensions exceed wire limits");
  }
  pendingMasks_.emplace(requestId, PendingMask{maskRequestId, expectedWords});
  send(protocol::MaskRequestEvent{
      requestId, maskRequestId, wordsPerToken,
      std::vector<uint32_t>(simulationTokens.begin(), simulationTokens.end())});
}

void NativeRuntime::completed(uint64_t requestId, EngineFinishReason reason, bool bounded,
                              uint32_t promptTokens, uint32_t completionTokens,
                              uint32_t deduplicatedTokens,
                              std::span<const float> optionLogits,
                              const SpeculativeCounts &speculation) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  double now = config_.monotonicMilliseconds();
  // A request that ends before it starts counts from its arrival.
  double started =
      telemetry.startedMilliseconds.value_or(telemetry.arrivedMilliseconds);
  double first = telemetry.firstTokenMilliseconds.value_or(now);
  send(protocol::DoneEvent{
      requestId, reason, promptTokens, completionTokens,
      durationMicros(started, first),
      telemetry.firstTokenMilliseconds ? durationMicros(first, now) : 0,
      durationMicros(telemetry.arrivedMilliseconds, now),
      std::vector<float>(optionLogits.begin(), optionLogits.end()),
      deduplicatedTokens, speculation, bounded});
  ended(requestId);
}

void NativeRuntime::failed(uint64_t requestId, LaneOutcome outcome,
                           std::string message, uint32_t retryAfterSeconds) {
  const LaneOutcomeWire wire = laneOutcomeWire(outcome);
  if (outcome == LaneOutcome::CapacityExhausted)
    metrics_.capacityFailed();
  requestError(requestId, std::string(wire.code), std::move(message),
               wire.retryable, retryAfterSeconds);
  ended(requestId);
}

void NativeRuntime::ended(uint64_t requestId) {
  pendingMasks_.erase(requestId);
  telemetry_.erase(requestId);
  if (telemetry_.empty() && config_.holdingRequests)
    config_.holdingRequests(false);
}

uint64_t NativeRuntime::durationMicros(double startMilliseconds,
                                       double endMilliseconds) {
  if (endMilliseconds <= startMilliseconds)
    return 0;
  double micros = (endMilliseconds - startMilliseconds) * 1000.0;
  if (micros >= double(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(micros);
}

} // namespace splash::engine
