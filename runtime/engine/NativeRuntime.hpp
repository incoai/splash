#pragma once

#include "AwakeClock.hpp"
#include "engine/Engine.hpp"
#include "engine/Protocol.hpp"
#include "engine/RuntimeMetrics.hpp"

#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

namespace splash::engine {

// The clocks a native loop reads unless told others: the system clock in
// Unix microseconds, which a request's absolute deadline is on, and the awake
// clock in milliseconds (awakeMilliseconds), which it measures every duration
// on.
[[nodiscard]] uint64_t systemUnixMicros() noexcept;

// The loop's constructor refuses a config without the engine's weights
// (EngineConfig::weights) or its clocks.
struct NativeLoopConfig {
  engine::EngineConfig engine;
  // Told true when the engine takes a request while it holds none, and false
  // when its last request ends; the process keeps the Mac from idle sleep in
  // between (main.mm). It must not throw. Empty where nothing needs to know.
  std::function<void(bool)> holdingRequests;
  // Told when the engine's own work fails (its failure boundary below caught
  // an exception, or an event it made does not fit the wire), before the
  // failure is reported. It must not throw. Empty where nothing needs to know.
  std::function<void()> engineFailed;
  std::function<uint64_t()> unixMicros = systemUnixMicros;
  std::function<double()> monotonicMilliseconds = awakeMilliseconds;
};

// Translates native protocol messages and events at the Engine boundary.
//
// The engine's failure boundary. Every received frame, tick(), runControl()
// and flushRestorePoints() run inside it: an exception is reported once as an
// EngineUnhealthy ErrorEvent (metal_execution_failed for MetalBackendError,
// else engine_execution_failed), the connection closes and the process exits.
// An output write that fails closes the connection and leaves the engine
// healthy.
// Request-scoped problems never arrive as exceptions here except
// std::invalid_argument from Engine::submit.
class NativeRuntime final : private EngineEventSink {
public:
  using ByteSink = std::function<void(std::span<const uint8_t>)>;
  using StatusProvider = std::function<std::string()>;

  NativeRuntime(NativeLoopConfig config, engine::Cache &cache, model::Model &model,
                ByteSink output, protocol::ProtocolLimits limits);

  // Answers status requests with `provider`'s document. The serving runtime
  // connects itself once it is built, before any frame arrives; a status
  // request without a source fails the engine. Throws std::invalid_argument
  // for an empty provider.
  void serveStatus(StatusProvider provider);

  // Processes every complete frame in bytes. False means the connection
  // must close. Request-scoped errors return true and preserve framing.
  bool receive(std::span<const uint8_t> bytes);
  bool finishInput();

  // Executes at most one explicit GPU BatchPlan, or takes back one part of
  // the released weights (Engine::tick).
  bool tick();
  // Command-free control work uses the same failure boundary as execution.
  bool runControl(const std::function<bool()> &control);
  // At a clean stop (Engine::flushRestorePoints). False until no restore
  // point is left, and once the engine has failed.
  bool flushRestorePoints();
  // Releases the weights once the engine has held no request for the idle
  // release since it became ready (Engine::releaseIdleWeights). Runs between
  // commands, in the control pass.
  void releaseIdleWeights();
  void setCompletionNotifier(std::function<void()> notifier) {
    core_.setCompletionNotifier(std::move(notifier));
  }

  // Sends Warming, before announceReady, once the engine holds its weights.
  void announceWarming();
  // Sends `ready` (readyEventFor), once, and starts the idle release's count.
  void announceReady(const protocol::ReadyEvent &ready);

  [[nodiscard]] bool connectionMustClose() const noexcept {
    return closeConnection_;
  }
  [[nodiscard]] bool engineHealthy() const noexcept { return engineHealthy_; }
  // Code and message of the failure that stopped the engine, for the log.
  [[nodiscard]] const std::string &engineFailure() const noexcept {
    return engineFailure_;
  }
  [[nodiscard]] bool commandInFlight() const noexcept {
    return core_.commandInFlight();
  }
  // How long FdTransport may block in poll(2): until the engine's next timed
  // event (Engine::nextWakeupMilliseconds), a refused restore's retry among
  // them, without a limit when it has none.
  [[nodiscard]] std::optional<double> millisecondsUntilNextWakeup() const;
  // The snapshot /status reports: each live request's age counted from its
  // arrival, on the runtime's clock.
  [[nodiscard]] engine::EngineSnapshot statusSnapshot() const;
  [[nodiscard]] WeightsSnapshot weightsSnapshot() const { return core_.weightsSnapshot(); }
  // The requests' latencies and the batches' throughput, which only the
  // loop records, from its engine's events.
  [[nodiscard]] RuntimeMetricsSnapshot metricsSnapshot() const { return metrics_.snapshot(); }
  // The engine's (Engine::resourceWaitSnapshot), on the runtime's clock.
  [[nodiscard]] engine::ResourceWaitSnapshot resourceWaitSnapshot() const {
    return core_.resourceWaitSnapshot(config_.monotonicMilliseconds());
  }
  [[nodiscard]] double monotonicMilliseconds() const {
    return config_.monotonicMilliseconds();
  }
  // The control pass's engine side, between commands
  // (Engine::collectLandings).
  void collectLandings(double nowMilliseconds) { core_.collectLandings(nowMilliseconds); }

private:
  struct RequestTelemetry {
    double arrivedMilliseconds = 0.0;
    // Empty until the engine starts the request (EngineEventSink::started).
    std::optional<double> startedMilliseconds;
    std::optional<double> firstTokenMilliseconds;
    std::optional<double> lastTokenMilliseconds;
    uint32_t emittedTokens = 0;
    // What held it before its first token, as the engine counted it
    // (EngineEventSink::firstTokenParts).
    TtftParts ttftParts{};
  };

  struct PendingMask {
    uint64_t maskRequestId = 0;
    uint64_t expectedWords = 0;
  };

  bool handle(protocol::ClientMessage &message);
  bool handleRequest(protocol::RequestFrame &request);
  bool handleCancel(const protocol::CancelFrame &cancel);
  bool handleStop(const protocol::StopFrame &stop);
  bool handleMask(const protocol::MaskResponseFrame &mask);
  bool handleStatus(const protocol::StatusRequestFrame &status);
  bool handleRelease(const protocol::ReleaseRequestFrame &release);
  bool handleMaskIssue(protocol::ProtocolIssue issue);
  bool handleIssue(protocol::ProtocolIssue issue);
  void requestError(uint64_t requestId, std::string code, std::string message,
                    bool retryable = false, uint32_t retryAfterSeconds = 0);
  void engineError(std::string code, std::string message);
  void executionFailed(std::exception_ptr error);
  bool send(const protocol::EngineEvent &event);
  // After a request's terminal event: the engine no longer holds it.
  void ended(uint64_t requestId);

  void started(uint64_t requestId, uint32_t matchedTokens,
               uint32_t lane) override;
  void batchCompleted(WorkKind kind, uint32_t width, uint32_t inputTokens,
                      uint32_t outputTokens, uint32_t draftedTokens,
                      uint32_t acceptedDraftTokens, double wallMilliseconds,
                      double cycleMilliseconds) override;
  void promptProgress(uint64_t requestId, uint32_t processedTokens) override;
  void firstTokenParts(uint64_t requestId, const TtftParts &parts) override;
  void tokens(uint64_t requestId, std::span<const uint32_t> values) override;
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulationTokens) override;
  void completed(uint64_t requestId, EngineFinishReason reason, bool bounded,
                 uint32_t promptTokens, uint32_t completionTokens,
                 uint32_t deduplicatedTokens,
                 std::span<const float> optionLogits,
                 const SpeculativeCounts &speculation) override;
  void failed(uint64_t requestId, LaneOutcome outcome,
              std::string message, uint32_t retryAfterSeconds) override;

  static uint64_t durationMicros(double startMilliseconds,
                                 double endMilliseconds);

  NativeLoopConfig config_;
  ByteSink output_;
  StatusProvider statusProvider_;
  protocol::ProtocolLimits limits_;
  protocol::FrameParser parser_;
  // Before core_, which reports to it through this loop's EngineEventSink.
  RuntimeMetrics metrics_;
  engine::Engine core_;
  std::unordered_map<uint64_t, RequestTelemetry> telemetry_;
  std::unordered_map<uint64_t, PendingMask> pendingMasks_;
  uint64_t nextMaskRequestId_ = 1;
  bool ready_ = false;
  bool closeConnection_ = false;
  bool engineHealthy_ = true;
  std::string engineFailure_;
};

} // namespace splash::engine
