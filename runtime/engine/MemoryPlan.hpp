#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace splash::engine {

inline constexpr uint64_t kMiB = 1024ULL * 1024;
inline constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

// The pages of the KV runway in extents of extentPages pages: the whole
// extents that hold the pages startup warmup runs on, which the KV pool
// allocates when it is built.
[[nodiscard]] constexpr uint64_t kvRunwayPages(uint32_t extentPages) noexcept {
  return (uint64_t{model::ExecutionLimits::warmupKvPages} + extentPages - 1) /
         extentPages * extentPages;
}

// What one request needs at the least: the fixed bytes, one lane's state and
// the KV runway in the layout's smallest extents; nullopt on overflow.
[[nodiscard]] std::optional<uint64_t>
minimumRequiredBytes(uint64_t fixedBytes, uint64_t laneStateBytes,
                     const kv::Layout &layout) noexcept;

// Inputs that the memory planner needs from a loaded model. Model tensor and
// KV geometry stay with their owners; the planner receives only identity,
// capacity, and measured allocation sizes.
struct ModelMemoryFootprint final {
  uint64_t targetWeightsBytes = 0;
  uint64_t draftWeightsBytes = 0;
  uint64_t visionWeightsBytes = 0;
  model::ModelMemoryPlan runtime;
  // The buffer a state's write to the disk tier stages through, set aside
  // when the tier's state file opened (a quota that holds one state); zero
  // otherwise.
  uint64_t stateStagingBytes = 0;
};

struct ModelMemoryProfile final {
  std::string name;
  uint32_t maximumContextTokens = 0;
  kv::Layout targetKvLayout;
  ModelMemoryFootprint footprint;

  [[nodiscard]] std::optional<std::string> validationError() const;
  [[nodiscard]] uint64_t fixedRuntimeBytes() const;
};

struct EngineMemoryPolicy {
  // recommendedMaxWorkingSetSize already describes Metal's performance-safe
  // working set. Keep only a small runtime/measurement margin below it; the
  // separate host reserve below protects the rest of unified memory.
  static constexpr uint64_t minimumWorkingSetMarginBytes = 1 * kGiB;
  static constexpr uint32_t workingSetMarginPercent = 2;

  [[nodiscard]] static constexpr uint64_t
  workingSetMarginBytes(uint64_t recommendedWorkingSetBytes) noexcept {
    uint64_t proportional =
        (recommendedWorkingSetBytes / 100) * workingSetMarginPercent +
        ((recommendedWorkingSetBytes % 100) * workingSetMarginPercent) / 100;
    return proportional > minimumWorkingSetMarginBytes
               ? proportional
               : minimumWorkingSetMarginBytes;
  }

  [[nodiscard]] static constexpr uint64_t
  hardBudgetBytes(uint64_t recommendedWorkingSetBytes,
                  uint64_t maximumMemoryBytes) noexcept {
    const uint64_t margin = workingSetMarginBytes(recommendedWorkingSetBytes);
    const uint64_t automatic = recommendedWorkingSetBytes > margin
                                   ? recommendedWorkingSetBytes - margin
                                   : 0;
    return maximumMemoryBytes && maximumMemoryBytes < automatic
               ? maximumMemoryBytes
               : automatic;
  }

  // A bounded host cushion, independent of the engine's Metal capacity.
  [[nodiscard]] static constexpr uint64_t
  hostAvailableReserveBytes(uint64_t physicalMemoryBytes) noexcept {
    return std::min<uint64_t>(physicalMemoryBytes / 10, 2 * kGiB);
  }
};

enum class BudgetErrorCode {
  None,
  InvalidDeviceCapabilities,
  InvalidModelSpec,
  WorkingSetTooSmall,
  ArithmeticOverflow,
  KvPoolDoesNotFit,
};

struct EngineMemoryBreakdown {
  uint64_t physicalMemoryBytes = 0;
  uint64_t recommendedWorkingSetBytes = 0;
  // Optional user ceiling. Zero means the automatic safe working-set
  // ceiling. A higher value never overrides the OS-safe ceiling.
  uint64_t configuredMemoryLimitBytes = 0;
  uint64_t workingSetMarginBytes = 0;
  uint64_t hardBudgetBytes = 0;

  uint64_t targetWeightsBytes = 0;
  uint64_t draftWeightsBytes = 0;
  uint64_t visionWeightsBytes = 0;
  uint32_t maximumBatchWidth = model::ExecutionLimits::maximumBatchWidth;
  uint64_t laneStateBytes = 0;
  uint64_t sharedPrefillBytes = 0;
  uint64_t sharedDecodeBytes = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
  uint64_t stateStagingBytes = 0;
  uint64_t fixedRuntimeBytes = 0;

  // All lanes' state, cached composite states, and KV extents grow from this
  // one governor-controlled byte budget; none is preallocated.
  uint64_t dynamicBudgetBytes = 0;

  uint32_t kvPageTokens = 0;
  uint64_t kvPageBytes = 0;
  // The pool grows and shrinks in extents of kvExtentPages pages, a size
  // chosen for this pool (kv::Layout::extentPagesFor). This is allocation
  // geometry, never the cache block size.
  uint32_t kvExtentPages = 0;
  uint64_t kvExtentBytes = 0;
  // One request's KV capacity: the whole extents of the budget's pages left
  // after one lane's state. The pool's page ids cover more (RuntimeResources
  // sizes them by the hard budget). Request context and four-lane execution
  // are independent policy limits.
  uint32_t kvCapacityPages = 0;
  uint64_t kvCapacityBytes = 0;
  uint64_t kvCapacityTokens = 0;

  uint64_t minimumDynamicBytes = 0;
  uint64_t minimumRequiredBytes = 0;
  uint64_t deficitBytes = 0;

  [[nodiscard]] std::string toStatusJson() const;
  [[nodiscard]] std::string describe() const;
};

struct BudgetValidationStatus {
  bool valid = false;
  BudgetErrorCode code = BudgetErrorCode::None;
  std::string message;
  EngineMemoryBreakdown breakdown;

  [[nodiscard]] std::string describe() const;
};

struct EngineMemoryPlanResult;

class EngineMemoryPlan {
public:
  [[nodiscard]] const EngineMemoryBreakdown &breakdown() const noexcept {
    return breakdown_;
  }
  // Stable per-request ceiling advertised by the runtime: never more than the
  // model supports or one request's KV capacity holds (less the speculative
  // scratch); requests share the budget dynamically.
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept;
  // The ceiling this plan would advertise with at most memoryBytes, within
  // its configured limit; zero when one request cannot fit there.
  [[nodiscard]] uint32_t contextTokensWithin(uint64_t memoryBytes) const;

  [[nodiscard]] std::string toStatusJson() const;

private:
  EngineMemoryPlan(DeviceCapabilities device, ModelMemoryProfile model,
                   EngineMemoryBreakdown breakdown);

  DeviceCapabilities device_;
  ModelMemoryProfile model_;
  EngineMemoryBreakdown breakdown_;

  friend EngineMemoryPlanResult
  evaluateEngineMemoryPlan(const DeviceCapabilities &,
                           const ModelMemoryProfile &,
                           uint64_t);
};

struct EngineMemoryPlanResult {
  std::optional<EngineMemoryPlan> plan;
  BudgetValidationStatus status;
};

// Pure planning without resource allocation.
[[nodiscard]] EngineMemoryPlanResult
evaluateEngineMemoryPlan(const DeviceCapabilities &device,
                         const ModelMemoryProfile &model,
                         uint64_t maximumMemoryBytes);

} // namespace splash::engine
