#pragma once

#include "model/ModelFactory.hpp"

#include <cstdint>
#include <memory>

namespace splash::model {

class Runtime final : public RuntimeModel {
public:
  explicit Runtime(RuntimeContext context);
  ~Runtime() override;
  void checkHealth() override;
  [[nodiscard]] bool needsHealthCheck() const noexcept override;

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Direct native-oracle entry point. Production admission uses
  // begin() and installs its cache-aware plan explicitly.
  void beginColdRequest(const ModelRequest &request, uint32_t stateSlot);
  [[nodiscard]] StateAdmission
  begin(const ModelRequest &request) override;
  void suspend(uint64_t requestId) override;
  [[nodiscard]] StateAdmission
  resume(const ModelRequest &request) override;
  void restore(uint64_t requestId, uint32_t restoredPrefixLength,
                     std::shared_ptr<const CompositeState> restoredState,
                     bool restoreDraftState) override;
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
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  submitTransfers(std::function<void()> completion) override;
  [[nodiscard]] std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) override;
  [[nodiscard]] bool canSnapshotToDisk() const noexcept override;
  [[nodiscard]] std::unique_ptr<StateOffload>
  snapshotToDisk(uint64_t requestId, std::function<void()> completion) override;
  [[nodiscard]] uint64_t reclaimIdleState() noexcept override;
  void provideMask(uint64_t requestId,
                   std::span<const uint32_t> words) override;
  void end(uint64_t requestId) override;

  [[nodiscard]] WarmupStepResult warmupPrefill(uint32_t rows) override;
  [[nodiscard]] WarmupStepResult
  warmupDecodeBatch(uint32_t width) override;
  [[nodiscard]] WarmupStepResult warmupDraftVerifyCommit() override;
  [[nodiscard]] WarmupStepResult
  warmupCompositeStateRestore() override;
  [[nodiscard]] ModelMemoryActual
  actualRuntimeMemory() const override;

  [[nodiscard]] ModelTelemetry
  telemetry() const noexcept override;

private:
  // The state slot of a resident request whose committed state can be
  // snapshotted: page-aligned, with a complete draft window.
  [[nodiscard]] uint32_t committedStateSlot(uint64_t requestId);
  void finishRestore(uint64_t requestId, uint32_t boundary, bool restoreDraft);
  void prepareWarmupDecode(uint64_t requestId, uint32_t anchor);
  [[nodiscard]] metal::AllocationResult beginAt(const ModelRequest &request,
                                       uint32_t stateSlot);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  prefillAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
               std::function<void()> completion);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  decodeAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
