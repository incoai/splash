#pragma once

#include "engine/Engine.hpp"
#include "engine/Protocol.hpp"
#include "engine/Status.hpp"

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
  uint64_t engineInstanceId = 1;
  uint32_t maskWordsPerToken = 1;
  RuntimeMetrics *metrics = nullptr;
};

struct NativeLoopClocks {
  std::function<uint64_t()> unixMicros;
  std::function<double()> monotonicMilliseconds;
};

// Translates native protocol messages and events at the Engine boundary.
class NativeRuntime final : private EngineEventSink {
public:
  using ByteSink = std::function<void(std::span<const uint8_t>)>;
  using StatusProvider = std::function<std::string()>;

  NativeRuntime(NativeLoopConfig config, engine::Cache &cache,
                model::Model &model, ByteSink output,
                StatusProvider statusProvider, NativeLoopClocks clocks = {},
                protocol::ProtocolLimits limits = {});

  // Processes every complete frame in bytes. False means the connection
  // must close. Request-scoped errors return true and preserve framing.
  bool receive(std::span<const uint8_t> bytes);
  bool finishInput();

  // Executes at most one explicit GPU BatchPlan.
  bool tick();
  // Command-free control work uses the same failure boundary as execution.
  bool runControl(const std::function<bool()> &control);
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
  [[nodiscard]] bool idle() const { return core_.idle(); }
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
  [[nodiscard]] uint64_t
  reclaimMemory(const MemoryReclaimDirective &directive) {
    return core_.reclaimMemory(directive);
  }
  [[nodiscard]] bool reclaimDeferred() const noexcept {
    return core_.reclaimDeferred();
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

  bool handle(protocol::Message &message);
  bool handleRequest(protocol::RequestFrame &request);
  bool handleCancel(const protocol::CancelFrame &cancel);
  bool handleMask(const protocol::MaskResponseFrame &mask);
  bool handleStatus(const protocol::StatusRequestFrame &status);
  bool handleMaskIssue(protocol::ProtocolIssue issue);
  bool handleIssue(protocol::ProtocolIssue issue);
  void requestError(uint64_t requestId, std::string code, std::string message,
                    bool retryable = false);
  void engineError(std::string code, std::string message);
  void executionFailed(std::exception_ptr error);
  bool send(protocol::Message message);

  void started(uint64_t requestId, EngineCacheStatus cacheStatus,
               uint32_t matchedTokens, uint32_t stateSlot) override;
  void batchCompleted(WorkKind kind, uint32_t width, uint32_t inputTokens,
                      uint32_t outputTokens, uint32_t draftedTokens,
                      uint32_t acceptedDraftTokens,
                      double wallMilliseconds) override;
  void promptProgress(uint64_t requestId, uint32_t processedTokens) override;
  void tokens(uint64_t requestId, std::span<const uint32_t> values) override;
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulationTokens) override;
  void completed(uint64_t requestId, EngineFinishReason reason,
                 uint32_t promptTokens, uint32_t completionTokens) override;
  void failed(uint64_t requestId, std::string code, std::string message,
              bool retryable) override;
  void capacityExhausted(uint64_t requestId, uint32_t requiredKvPages,
                         uint32_t availableKvPages,
                         uint64_t retryAfterMicros) override;

  static NativeLoopClocks defaultClocks();
  static uint64_t durationMicros(double startMilliseconds,
                                 double endMilliseconds);

  NativeLoopConfig config_;
  ByteSink output_;
  StatusProvider statusProvider_;
  NativeLoopClocks clocks_;
  protocol::ProtocolLimits limits_;
  protocol::FrameParser parser_;
  engine::Engine core_;
  std::unordered_map<uint64_t, RequestTelemetry> telemetry_;
  std::unordered_map<uint64_t, PendingMask> pendingMasks_;
  uint64_t nextMaskRequestId_ = 1;
  bool ready_ = false;
  bool closeConnection_ = false;
  bool engineHealthy_ = true;
};

} // namespace splash::engine
