#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace splash::engine {

[[nodiscard]] std::string
deviceStatusJson(const DeviceCapabilities &device);

inline constexpr uint64_t kMiB = 1024ULL * 1024;
inline constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

// Inputs that the memory planner needs from a loaded model. Model tensor and
// KV geometry stay with their owners; the planner receives only identity,
// capacity, and measured allocation sizes.
struct ModelMemoryFootprint final {
  uint64_t targetWeightsBytes = 0;
  uint64_t draftWeightsBytes = 0;
  uint64_t visionWeightsBytes = 0;
  uint64_t activeStateCellBytes = 0;
  uint64_t sharedPrefillBytes = 0;
  uint64_t sharedDecodeBytes = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
};

struct ModelMemoryProfile final {
  std::string name;
  uint32_t maximumContextTokens = 0;
  kv::Layout targetKvLayout;
  ModelMemoryFootprint footprint;

  [[nodiscard]] std::optional<std::string> validationError() const;
  [[nodiscard]] uint64_t fixedRuntimeBytes() const;
};

[[nodiscard]] std::string modelStatusJson(const ModelMemoryProfile &model);

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
                  uint64_t maximumMemoryBytes = 0) noexcept {
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

[[nodiscard]] std::string_view budgetErrorCodeName(BudgetErrorCode code);

struct EngineMemoryBreakdown {
  uint64_t physicalMemoryBytes = 0;
  uint64_t recommendedWorkingSetBytes = 0;
  // Optional user ceiling. Zero means the automatic safe working-set
  // ceiling. A higher value never overrides the OS-safe ceiling.
  uint64_t configuredMemoryLimitBytes = 0;
  uint64_t headroomBytes = 0;
  uint64_t hardBudgetBytes = 0;

  uint64_t targetWeightsBytes = 0;
  uint64_t draftWeightsBytes = 0;
  uint64_t visionWeightsBytes = 0;
  uint32_t maximumBatchWidth = model::ExecutionLimits::maximumBatchWidth;
  uint64_t activeStateCellBytes = 0;
  uint64_t sharedPrefillBytes = 0;
  uint64_t sharedDecodeBytes = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
  uint64_t fixedRuntimeBytes = 0;

  // All active state cells, cached composite states, and physical KV
  // extents grow from this one governor-controlled byte budget. None is
  // preallocated merely because the address space exists.
  uint64_t dynamicBudgetBytes = 0;

  uint32_t kvPageTokens = 0;
  uint64_t kvPageBytes = 0;
  // Number of logical Page32 blocks mapped in one Metal sparse-buffer
  // operation. This is allocation alignment, never the cache block size.
  uint32_t kvSparseMappingBatchPages = 0;
  uint32_t kvExtentPages = 0;
  uint64_t kvExtentBytes = 0;
  // Derived from the device's maximum Metal buffer length. Request context
  // and four-lane execution are independent policy limits.
  uint32_t maximumKvPages = 0;
  uint32_t kvVirtualPages = 0;
  uint64_t kvVirtualBytes = 0;
  uint64_t kvVirtualTokens = 0;

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

  [[nodiscard]] std::string toStatusJson() const;
  [[nodiscard]] std::string describe() const;
};

struct EngineMemoryPlanResult;

class EngineMemoryPlan {
public:
  [[nodiscard]] const EngineMemoryBreakdown &breakdown() const noexcept {
    return breakdown_;
  }
  // Stable per-request ceiling advertised by the runtime. The physical KV
  // pool is shared dynamically, but one admitted request is never promised
  // more than either the model supports or the complete pool can hold.
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept;

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
                         uint64_t maximumMemoryBytes = 0);

[[nodiscard]] EngineMemoryPlan
requireEngineMemoryPlan(const DeviceCapabilities &device,
                        const ModelMemoryProfile &model,
                        uint64_t maximumMemoryBytes = 0);

} // namespace splash::engine
