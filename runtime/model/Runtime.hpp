#pragma once

#include "model/ModelFactory.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace splash::model {

class RuntimeInstrumentation;

class Runtime final : public RuntimeModel {
public:
  explicit Runtime(RuntimeContext context);
  ~Runtime() override;
  void checkHealth() override;
  [[nodiscard]] std::optional<uint32_t> statesToActivate() const noexcept override;
  [[nodiscard]] uint64_t idleBufferBytes(uint32_t evictedStates, bool afterStart,
                                         bool keepLane) const noexcept override;
  [[nodiscard]] uint64_t startStateBytes() const noexcept override;
  [[nodiscard]] CommandLimits commandLimits() const noexcept override;
  [[nodiscard]] CommandWork prefillWork(std::span<const PrefillLane> lanes, uint32_t first,
                                        uint32_t passesLeft) const override;
  [[nodiscard]] std::span<ElasticPool *const> pools() noexcept override;

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Direct native-oracle entry point. Production admission uses
  // begin() and installs its cache-aware plan explicitly.
  void beginColdRequest(const ModelRequest &request, uint32_t stateLane);
  [[nodiscard]] StateAdmission
  begin(const ModelRequest &request) override;
  void suspend(uint64_t requestId) override;
  [[nodiscard]] StateAdmission
  resume(const ModelRequest &request) override;
  [[nodiscard]] std::unique_ptr<StateRestore> beginRestore(
      uint64_t requestId, uint32_t boundary,
      std::shared_ptr<const CompositeState> state, bool restoreDraft,
      std::function<void()> completion) override;
  void setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) override;
  [[nodiscard]] std::vector<ModelStepResult>
  prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override;
  [[nodiscard]] std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::shared_ptr<StateReservation>
  reserveState(uint64_t requestId, uint32_t tokens, metal::AllocationFailure &refusal) override;
  [[nodiscard]] std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) override;
  [[nodiscard]] bool canSnapshotToDisk(bool inUse) const noexcept override;
  [[nodiscard]] std::unique_ptr<StateOffload>
  snapshotToDisk(uint64_t requestId, std::function<void()> completion) override;
  [[nodiscard]] uint64_t stagingShortfall(uint64_t requestId, uint32_t pages) const override;
  [[nodiscard]] uint64_t hotFloorBytes(uint32_t lanes) const override;
  [[nodiscard]] uint64_t hotFloorShortfall(std::span<const uint64_t> requestIds) const override;
  [[nodiscard]] ModelMemoryAdmission reserveSpilledReads(uint64_t requestId, SpilledReadNeed need) override;
  [[nodiscard]] uint64_t releaseIdleBuffer(bool keepLane) noexcept override;
  [[nodiscard]] std::optional<std::string>
  provideMask(uint64_t requestId, std::span<const uint32_t> words) override;
  void end(uint64_t requestId) override;

  [[nodiscard]] WarmupStepResult warmupPrefill(uint32_t rows) override;
  [[nodiscard]] WarmupStepResult
  warmupDecodeBatch(uint32_t width) override;
  [[nodiscard]] WarmupStepResult
  warmupCompositeStateRestore() override;
  [[nodiscard]] ModelMemoryActual
  actualRuntimeMemory() const override;

  [[nodiscard]] ModelTelemetry
  telemetry() const noexcept override;

private:
  // The state lane of a resident request; the storage checks that its
  // committed state can be cached.
  [[nodiscard]] uint32_t residentLane(uint64_t requestId) const;
  void finishRestore(uint64_t requestId, uint32_t boundary, bool restoreDraft);
  void prepareWarmupDecode(uint64_t requestId, uint32_t anchor);
  [[nodiscard]] StateAdmission beginAt(const ModelRequest &request,
                                       uint32_t stateLane);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  prefillAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
               std::function<void()> completion);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  decodeAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion);

  friend class RuntimeInstrumentation;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
