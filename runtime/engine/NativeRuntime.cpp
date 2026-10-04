#include "engine/NativeRuntime.hpp"
#include "StderrLine.hpp"
#include "TestConfig.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace splash::engine {

NativeRuntime::NativeRuntime(NativeLoopConfig config, engine::Cache &cache,
                             model::Model &model, ByteSink output,
                             StatusProvider statusProvider,
                             protocol::ProtocolLimits limits)
    : config_(std::move(config)), output_(std::move(output)),
      statusProvider_(std::move(statusProvider)), clocks_(clocks()),
      limits_(limits), parser_(limits_),
      core_(config_.engine, cache, model, *this),
      weightKeepAliveMilliseconds_(
          1000.0 * testConfig().residencyKeepAliveSeconds.value_or(
                       metal::kResidencyKeepAliveSeconds)),
      lastRequestMilliseconds_(clocks_.monotonicMilliseconds()) {
  if (!output_ || !statusProvider_) {
    throw std::invalid_argument("invalid native engine loop config");
  }
  if (auto issue = protocol::validateLimits(limits_))
    throw std::invalid_argument(issue->describe());
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
    // The engine runs no request until its weights are back, written an
    // image per tick so that frames are answered between them.
    if (restoreStarted_) {
      restoreWeights();
      return true;
    }
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
  if (!config_.weights || config_.weights->released() || !core_.idle() ||
      clocks_.monotonicMilliseconds() - lastRequestMilliseconds_ <
          weightKeepAliveMilliseconds_)
    return;
  config_.weights->release();
  std::ostringstream message;
  message << "Weights released after " << weightKeepAliveMilliseconds_ / 1000.0
          << " s without a request; the next request restores them";
  writeStderrLine(message.str());
}

void NativeRuntime::restoreWeights() {
  if (!config_.weights->restore())
    return;
  lastRequestMilliseconds_ = clocks_.monotonicMilliseconds();
  std::ostringstream message;
  message << "Weights restored in " << std::fixed << std::setprecision(2)
          << (lastRequestMilliseconds_ - *restoreStarted_) / 1000.0 << " s";
  restoreStarted_.reset();
  writeStderrLine(message.str());
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
  if (!send(protocol::ReadyEvent{model::ExecutionLimits::maximumBatchWidth,
                                 config_.engine.maxContext,
                                 config_.engine.maxImagePatches != 0})) {
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
  const uint64_t nowUnix = clocks_.unixMicros();
  const double nowMonotonic = clocks_.monotonicMilliseconds();
  const uint64_t remaining =
      request.absoluteDeadlineUnixMicros > nowUnix
          ? std::min(request.absoluteDeadlineUnixMicros - nowUnix,
                     request.remainingDeadlineMicros)
          : 0;
  if (!std::isfinite(nowMonotonic) || !remaining) {
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
    engineRequest.flags = request.flags;
    engineRequest.returnProgress = request.returnProgress;
    engineRequest.deadlineMilliseconds =
        nowMonotonic + double(remaining) / 1000.0;
    core_.submit(std::move(engineRequest));
  } catch (const std::invalid_argument &error) {
    requestError(request.requestId, "invalid_request", error.what());
    return true;
  }
  telemetry_.emplace(request.requestId,
                     RequestTelemetry{.arrivedMilliseconds = nowMonotonic});
  // Released weights are written back before the engine runs the request
  // (tick()); a failure to restore them stops the engine.
  if (config_.weights && config_.weights->released() && !restoreStarted_)
    restoreStarted_ = nowMonotonic;
  return true;
}

bool NativeRuntime::handleCancel(const protocol::CancelFrame &cancel) {
  // Cancel can arrive after the terminal event; it is then a no-op.
  if (telemetry_.contains(cancel.requestId))
    core_.cancel(cancel.requestId);
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

bool NativeRuntime::handleMaskIssue(protocol::ProtocolIssue issue) {
  if (!issue.requestId)
    return handleIssue(std::move(issue));
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
  protocol::FailureClass classification = issue.failureClass;
  uint64_t requestId = issue.requestId;
  if (classification == protocol::FailureClass::EngineUnhealthy) {
    engineError(std::string(protocol::issueCodeName(issue.code)),
                std::move(issue.message));
    return false;
  }
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
  std::string failure;
  try {
    output_(*serialized.value);
    return true;
  } catch (const std::exception &error) {
    failure = error.what();
  } catch (...) {
    failure = "unknown output exception";
  }
  if (engineFailure_.empty())
    engineFailure_ = "output_write_failed: " + failure;
  engineHealthy_ = false;
  closeConnection_ = true;
  return false;
}

void NativeRuntime::batchCompleted(WorkKind kind, uint32_t width,
                                   uint32_t inputTokens, uint32_t outputTokens,
                                   uint32_t draftedTokens,
                                   uint32_t acceptedDraftTokens,
                                   double wallMilliseconds,
                                   double cycleMilliseconds) {
  if (config_.metrics) {
    config_.metrics->batchCompleted(kind, width, inputTokens, outputTokens,
                                    draftedTokens, acceptedDraftTokens,
                                    wallMilliseconds, cycleMilliseconds);
  }
}

void NativeRuntime::started(uint64_t requestId, uint32_t matchedTokens,
                            uint32_t lane) {
  RequestTelemetry &telemetry = telemetry_.at(requestId);
  telemetry.startedMilliseconds = clocks_.monotonicMilliseconds();
  send(protocol::StartEvent{requestId, lane, matchedTokens});
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
      requestId, reason, promptTokens, completionTokens,
      durationMicros(started, first),
      telemetry.firstTokenMilliseconds ? durationMicros(first, now) : 0,
      durationMicros(telemetry.arrivedMilliseconds, now),
      std::vector<float>(optionLogits.begin(), optionLogits.end())});
  pendingMasks_.erase(requestId);
  telemetry_.erase(requestId);
  lastRequestMilliseconds_ = now;
}

void NativeRuntime::failed(uint64_t requestId, LaneOutcome outcome,
                           std::string message) {
  const LaneOutcomeWire wire = laneOutcomeWire(outcome);
  if (config_.metrics && outcome == LaneOutcome::CapacityExhausted)
    config_.metrics->capacityFailed();
  requestError(requestId, std::string(wire.code), std::move(message),
               wire.retryable);
  pendingMasks_.erase(requestId);
  telemetry_.erase(requestId);
  lastRequestMilliseconds_ = clocks_.monotonicMilliseconds();
}

NativeRuntime::Clocks NativeRuntime::clocks() {
  Clocks result{testConfig().unixMicros, testConfig().monotonicMilliseconds};
  if (!result.unixMicros) {
    result.unixMicros = [] {
      auto now = std::chrono::system_clock::now().time_since_epoch();
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(now).count());
    };
  }
  if (!result.monotonicMilliseconds) {
    result.monotonicMilliseconds = [] {
      auto now = std::chrono::steady_clock::now().time_since_epoch();
      return std::chrono::duration<double, std::milli>(now).count();
    };
  }
  return result;
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
