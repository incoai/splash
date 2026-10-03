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

struct NativeLoopConfig {
  engine::EngineConfig engine;
  RuntimeMetrics *metrics = nullptr;
  // The model's weights: released once the engine has held no request for
  // the residency keep-alive (metal::kResidencyKeepAliveSeconds), and written
  // back, an image per tick, before the engine runs the next request. Null
  // where they stay, as in tests of other behavior.
  model::WeightMemory *weights = nullptr;
};

// Translates native protocol messages and events at the Engine boundary.
//
// The engine's failure boundary. Every received frame, tick() and
// runControl() run inside it: an exception is reported once as an
// EngineUnhealthy ErrorEvent (metal_execution_failed for MetalBackendError,
// else engine_execution_failed), the connection closes and the process exits.
// Request-scoped problems never arrive as exceptions here except
// std::invalid_argument from Engine::submit.
class NativeRuntime final : private EngineEventSink {
public:
  using ByteSink = std::function<void(std::span<const uint8_t>)>;
  using StatusProvider = std::function<std::string()>;

  NativeRuntime(NativeLoopConfig config, engine::Cache &cache,
                model::Model &model, ByteSink output,
                StatusProvider statusProvider, protocol::ProtocolLimits limits);

  // Processes every complete frame in bytes. False means the connection
  // must close. Request-scoped errors return true and preserve framing.
  bool receive(std::span<const uint8_t> bytes);
  bool finishInput();

  // Executes at most one explicit GPU BatchPlan, or writes back one image of
  // released weights.
  bool tick();
  // Command-free control work uses the same failure boundary as execution.
  bool runControl(const std::function<bool()> &control);
  // Releases the weights once the engine has held no request for the
  // residency keep-alive. Runs between commands, in the control pass.
  void releaseIdleWeights();
  void setCompletionNotifier(std::function<void()> notifier) {
    core_.setCompletionNotifier(std::move(notifier));
  }

  void observePrefill(uint32_t rows, double wallMilliseconds) {
    core_.observePrefill(rows, wallMilliseconds);
  }

  void announceReady();

  [[nodiscard]] bool ready() const noexcept { return ready_; }
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
  // Used by the fd event host to block in poll(2) without periodic sleeps,
  // waking for either a request deadline or a deferred resource retry.
  [[nodiscard]] std::optional<double> millisecondsUntilNextWakeup() const;
  [[nodiscard]] engine::EngineSnapshot snapshot() const {
    return core_.snapshot();
  }
  [[nodiscard]] engine::ResourceWaitSnapshot resourceWaitSnapshot() const {
    return core_.resourceWaitSnapshot(clocks_.monotonicMilliseconds());
  }
  [[nodiscard]] double monotonicMilliseconds() const {
    return clocks_.monotonicMilliseconds();
  }
  [[nodiscard]] MemoryReclaimResult
  reclaimMemory(const MemoryReclaimDirective &directive) {
    return core_.reclaimMemory(directive);
  }

private:
  struct RequestTelemetry {
    double arrivedMilliseconds = 0.0;
    double startedMilliseconds = 0.0;
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
  void restoreWeights();
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
                 std::span<const float> optionLogits) override;
  void failed(uint64_t requestId, LaneOutcome outcome,
              std::string message) override;

  // The system clock in microseconds and the steady clock in milliseconds,
  // or the test seam's (TestConfig).
  struct Clocks {
    std::function<uint64_t()> unixMicros;
    std::function<double()> monotonicMilliseconds;
  };
  static Clocks clocks();
  static uint64_t durationMicros(double startMilliseconds,
                                 double endMilliseconds);

  NativeLoopConfig config_;
  ByteSink output_;
  StatusProvider statusProvider_;
  Clocks clocks_;
  protocol::ProtocolLimits limits_;
  protocol::FrameParser parser_;
  engine::Engine core_;
  std::unordered_map<uint64_t, RequestTelemetry> telemetry_;
  std::unordered_map<uint64_t, PendingMask> pendingMasks_;
  uint64_t nextMaskRequestId_ = 1;
  // metal::kResidencyKeepAliveSeconds, or the test seam's (TestConfig), and
  // when the engine last held a request: when one ended, or when weights were
  // restored for one.
  double weightKeepAliveMilliseconds_;
  double lastRequestMilliseconds_;
  // When the weights began to be written back for a request, until they are.
  std::optional<double> restoreStarted_;
  bool ready_ = false;
  bool closeConnection_ = false;
  bool engineHealthy_ = true;
  std::string engineFailure_;
};

} // namespace splash::engine
