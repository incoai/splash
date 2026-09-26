#pragma once

#include "engine/MemoryPlan.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace splash::engine {

enum class MemoryAuditError {
    None,
    MissingMeasurement,
    CategoryExceedsPlan,
    BackendAccountingMismatch,
    RuntimeReserveExceeded,
    HardBudgetExceeded,
    WarmupEstimateDeviation,
    ArithmeticOverflow,
};

[[nodiscard]] std::string_view memoryAuditErrorName(
    MemoryAuditError error);

struct ActualMemoryReport {
    uint64_t targetWeightsBytes = 0;
    uint64_t draftWeightsBytes = 0;
    uint64_t visionWeightsBytes = 0;
    // Unique physical GDN/draft allocations across active lanes, cached
    // states and idle pooled buffers.
    uint64_t stateResidentBytes = 0;
    uint64_t sharedPrefillBytes = 0;
    uint64_t sharedDecodeBytes = 0;
    uint64_t kvResidentBytes = 0;
    // The disk tier's KV staging ring and copy table; zero without the tier.
    uint64_t kvStagingBytes = 0;

    uint64_t backendAllocatedBytes = 0;
    uint64_t deviceCurrentAllocatedBytes = 0;
    uint64_t devicePeakAllocatedBytes = 0;
    uint64_t estimatedWarmupPeakBytes = 0;
};

struct MemoryAuditResult {
    bool valid = false;
    MemoryAuditError error = MemoryAuditError::None;
    std::string message;
    ActualMemoryReport actual;
    uint64_t categorizedBytes = 0;
    uint64_t backendUnclassifiedBytes = 0;
    uint64_t deviceUntrackedBytes = 0;
    uint32_t warmupPeakDeviationBasisPoints = 0;
    uint64_t actualHeadroomBytes = 0;

    [[nodiscard]] std::string toStatusJson() const;
    [[nodiscard]] std::string describe() const;
};

inline constexpr uint32_t kMaximumWarmupDeviationBasisPoints = 500;

// Validates actual MTLResource.allocatedSize category totals and Metal's
// process-wide peak against the immutable plan.
[[nodiscard]] MemoryAuditResult auditActualMemory(
    const EngineMemoryPlan &plan, ActualMemoryReport actual);

}  // namespace splash::engine
