#pragma once

#include "engine/Engine.hpp"
#include "engine/Protocol.hpp"
#include "engine/Status.hpp"
#include "model/WeightMemory.hpp"

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
// clock (AwakeClock) in milliseconds, which it measures every duration on.
[[nodiscard]] uint64_t systemUnixMicros() noexcept;
[[nodiscard]] double awakeMilliseconds() noexcept;

// The loop's constructor refuses a config without its metrics, its weights,
// their admission or its clocks.
struct NativeLoopConfig {
  engine::EngineConfig engine;
  RuntimeMetrics *metrics = nullptr;
  // What the engine gives back while idle (ReleasableMemory): released once
  // the engine has held no request for the idle release, and taken back, a
  // part per tick, before the engine runs the next request.
  model::WeightMemory *weights = nullptr;
  // Admits the memory the weights take back
  // (MemoryGovernor::allocationAdmission), as memory the requests waiting for
  // them need (EngineConfig::serving).
  metal::AllocationAdmission weightAdmission;
  // Told true when the engine takes a request while it holds none, and false
  // when its last request ends; the process keeps the Mac from idle sleep in
  // between (main.mm). It must not throw. Empty where nothing needs to know.
  std::function<void(bool)> holdingRequests;
  std::function<uint64_t()> unixMicros = systemUnixMicros;
  std::function<double()> monotonicMilliseconds = awakeMilliseconds;
};

// Translates native protocol messages and events at the Engine boundary.
//
// The engine's failure boundary. Every received frame, tick(), runControl()
// and flushRestorePoints() run inside it: an exception is reported once as an
// EngineUnhealthy ErrorEvent (metal_execution_failed for MetalBackendError,
// else engine_execution_failed), the connection closes and the process exits.
// Request-scoped problems never arrive as exceptions here except
// std::invalid_argument from Engine::submit.
class NativeRuntime final : private EngineEventSink {
public:
  using ByteSink = std::function<void(std::span<const uint8_t>)>;
  using StatusProvider = std::function<std::string()>;

  // idleReleaseSeconds is the engine's idle release
  // (RuntimeResourcesConfig::idleReleaseSeconds): positive, and infinite
  // where the weights stay.
  NativeRuntime(NativeLoopConfig config, double idleReleaseSeconds,
                engine::Cache &cache, model::Model &model, ByteSink output,
                StatusProvider statusProvider, protocol::ProtocolLimits limits);

  // Processes every complete frame in bytes. False means the connection
  // must close. Request-scoped errors return true and preserve framing.
  bool receive(std::span<const uint8_t> bytes);
  bool finishInput();

  // Executes at most one explicit GPU BatchPlan, or takes back one part of
  // the released weights (model::WeightMemory::restore). While admission
  // refuses that part, the tick makes no progress and the requests wait for
  // memory, retrying after kResourceRetryBackoffMilliseconds, until
  // EngineConfig::resourceWaitTimeoutMilliseconds has passed since the first
  // refusal after the last part that came back. Then, or at once when no
  // request waits, the weights give back the parts that came back and the
  // requests fail with LaneOutcome::ResourceTimeout.
  bool tick();
  // Command-free control work uses the same failure boundary as execution.
  bool runControl(const std::function<bool()> &control);
  // At a clean stop (Engine::flushRestorePoints). False until no restore
  // point is left, and once the engine has failed.
  bool flushRestorePoints();
  // Releases the weights once the engine has held no request for the idle
  // release since it became ready. Runs between commands, in the control pass.
  void releaseIdleWeights();
  void setCompletionNotifier(std::function<void()> notifier) {
    core_.setCompletionNotifier(std::move(notifier));
  }

  void observePrefill(uint32_t rows, double wallMilliseconds) {
    core_.observePrefill(rows, wallMilliseconds);
  }

  void announceReady();

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
  // How long FdTransport may block in poll(2): until a refused restore tries
  // again, or the engine's next timed event (Engine::nextWakeupMilliseconds),
  // without a limit when it has none.
  [[nodiscard]] std::optional<double> millisecondsUntilNextWakeup() const;
  [[nodiscard]] engine::EngineSnapshot snapshot() const {
    return core_.snapshot();
  }
  // The snapshot /status reports: each live request's age counted from its
  // arrival, on the runtime's clock.
  [[nodiscard]] engine::EngineSnapshot statusSnapshot() const;
  [[nodiscard]] WeightsSnapshot weightsSnapshot() const {
    return {idleReleaseSeconds_, config_.weights->released(), weightRestores_,
            weightRestoreFailures_};
  }
  // The engine's (Engine::resourceWaitSnapshot), with the requests a refused
  // restore holds waiting for memory.
  [[nodiscard]] engine::ResourceWaitSnapshot resourceWaitSnapshot() const;
  [[nodiscard]] double monotonicMilliseconds() const {
    return config_.monotonicMilliseconds();
  }
  [[nodiscard]] MemoryReclaimResult
  reclaimMemory(const MemoryReclaimDirective &directive) {
    return core_.reclaimMemory(directive);
  }

private:
  struct RequestTelemetry {
    double arrivedMilliseconds = 0.0;
    // Empty until the engine starts the request (EngineEventSink::started).
    std::optional<double> startedMilliseconds;
    std::optional<double> firstTokenMilliseconds;
    std::optional<double> lastTokenMilliseconds;
    uint32_t emittedTokens = 0;
  };

  struct PendingMask {
    uint64_t maskRequestId = 0;
    uint64_t expectedWords = 0;
  };

  bool handle(protocol::ClientMessage &message);
  bool handleRequest(protocol::RequestFrame &request);
  bool restoreWeights();
  bool weightsRefused(metal::AllocationFailure failure);
  bool handleCancel(const protocol::CancelFrame &cancel);
  bool handleMask(const protocol::MaskResponseFrame &mask);
  bool handleStatus(const protocol::StatusRequestFrame &status);
  bool handleMaskIssue(protocol::ProtocolIssue issue);
  bool handleIssue(protocol::ProtocolIssue issue);
  void requestError(uint64_t requestId, std::string code, std::string message,
                    bool retryable = false);
  void engineError(std::string code, std::string message);
  void executionFailed(std::exception_ptr error);
  bool send(const protocol::EngineEvent &event);
  // After a request's terminal event: the engine no longer holds it.
  void ended(uint64_t requestId, double now);

  void started(uint64_t requestId, uint32_t matchedTokens,
               uint32_t lane) override;
  void batchCompleted(WorkKind kind, uint32_t width, uint32_t inputTokens,
                      uint32_t outputTokens, uint32_t draftedTokens,
                      uint32_t acceptedDraftTokens, double wallMilliseconds,
                      double cycleMilliseconds) override;
  void promptProgress(uint64_t requestId, uint32_t processedTokens) override;
  void tokens(uint64_t requestId, std::span<const uint32_t> values) override;
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulationTokens) override;
  void completed(uint64_t requestId, EngineFinishReason reason,
                 uint32_t promptTokens, uint32_t completionTokens,
                 std::span<const float> optionLogits,
                 const ScoreReadout &score) override;
  void failed(uint64_t requestId, LaneOutcome outcome,
              std::string message) override;

  static uint64_t durationMicros(double startMilliseconds,
                                 double endMilliseconds);

  NativeLoopConfig config_;
  double idleReleaseSeconds_;
  ByteSink output_;
  StatusProvider statusProvider_;
  protocol::ProtocolLimits limits_;
  protocol::FrameParser parser_;
  engine::Engine core_;
  std::unordered_map<uint64_t, RequestTelemetry> telemetry_;
  std::unordered_map<uint64_t, PendingMask> pendingMasks_;
  uint64_t nextMaskRequestId_ = 1;
  // When the engine's idle period began: at Ready, when a request ended, or
  // when the weights were restored for one.
  double idleSinceMilliseconds_;
  // When the weights began to be written back for a request, until they are.
  std::optional<double> restoreStarted_;
  // Since when admission refuses the next part of the weights to the
  // requests waiting: from the first refusal after the last part that came
  // back.
  std::optional<double> restoreRefusedSince_;
  // The times the weights were written back, each for a request.
  uint64_t weightRestores_ = 0;
  // The restores that gave up for want of memory.
  uint64_t weightRestoreFailures_ = 0;
  bool ready_ = false;
  bool closeConnection_ = false;
  bool engineHealthy_ = true;
  std::string engineFailure_;
};

} // namespace splash::engine
