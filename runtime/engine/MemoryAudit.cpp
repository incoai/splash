#include "engine/MemoryAudit.hpp"
#include "engine/Checked.hpp"

#include <limits>
#include <sstream>
#include <utility>

namespace splash::engine {
namespace {

MemoryAuditResult fail(MemoryAuditError error, std::string message,
                       ActualMemoryReport actual) {
  MemoryAuditResult result;
  result.error = error;
  result.message = std::move(message);
  result.actual = std::move(actual);
  return result;
}

} // namespace

std::string_view memoryAuditErrorName(MemoryAuditError error) {
  switch (error) {
  case MemoryAuditError::None:
    return "none";
  case MemoryAuditError::MissingMeasurement:
    return "missing_measurement";
  case MemoryAuditError::CategoryExceedsPlan:
    return "category_exceeds_plan";
  case MemoryAuditError::BackendAccountingMismatch:
    return "backend_accounting_mismatch";
  case MemoryAuditError::RuntimeReserveExceeded:
    return "runtime_reserve_exceeded";
  case MemoryAuditError::HardBudgetExceeded:
    return "hard_budget_exceeded";
  case MemoryAuditError::WarmupEstimateDeviation:
    return "warmup_estimate_deviation";
  case MemoryAuditError::ArithmeticOverflow:
    return "arithmetic_overflow";
  }
  return "unknown";
}

MemoryAuditResult auditActualMemory(const EngineMemoryPlan &plan,
                                    ActualMemoryReport actual) {
  const EngineMemoryBreakdown &budget = plan.breakdown();
  // Vision weights are absent without a vision tower. Loading already
  // requires them for a model with vision.
  if (!actual.targetWeightsBytes || !actual.draftWeightsBytes ||
      !actual.stateResidentBytes || !actual.sharedPrefillBytes ||
      !actual.sharedDecodeBytes || !actual.kvResidentBytes ||
      !actual.backendAllocatedBytes || !actual.deviceCurrentAllocatedBytes ||
      !actual.devicePeakAllocatedBytes || !actual.estimatedWarmupPeakBytes) {
    return fail(MemoryAuditError::MissingMeasurement,
                "warmup memory report is incomplete", actual);
  }

  // The plan takes the weight categories from what loaded, so they are
  // counted but have no bound of their own. A buffer a loader does not report
  // is unclassified backend memory, which the pipeline and runtime reserves
  // bound; only the arenas, KV storage and the disk tier's KV staging have
  // planned category bounds.
  uint64_t categorized = 0;
  for (const uint64_t weights : {actual.targetWeightsBytes,
                                 actual.draftWeightsBytes,
                                 actual.visionWeightsBytes}) {
    if (!checkedAdd(categorized, weights, categorized)) {
      return fail(MemoryAuditError::ArithmeticOverflow,
                  "categorized memory sum overflowed", actual);
    }
  }
  struct Category {
    const char *name;
    uint64_t actual;
    uint64_t planned;
  };
  const Category categories[] = {
      {"shared prefill", actual.sharedPrefillBytes, budget.sharedPrefillBytes},
      {"shared decode", actual.sharedDecodeBytes, budget.sharedDecodeBytes},
      {"Q8 virtual storage", actual.kvResidentBytes, budget.kvVirtualBytes},
      {"KV staging", actual.kvStagingBytes, budget.kvStagingBytes},
  };
  for (const Category &category : categories) {
    if (category.actual > category.planned) {
      std::ostringstream message;
      message << category.name << " actual bytes " << category.actual
              << " exceed planned bytes " << category.planned;
      return fail(MemoryAuditError::CategoryExceedsPlan, message.str(), actual);
    }
    if (!checkedAdd(categorized, category.actual, categorized)) {
      return fail(MemoryAuditError::ArithmeticOverflow,
                  "categorized memory sum overflowed", actual);
    }
  }
  uint64_t dynamic = 0;
  if (!checkedAdd(actual.stateResidentBytes, actual.kvResidentBytes, dynamic) ||
      !checkedAdd(categorized, actual.stateResidentBytes, categorized)) {
    return fail(MemoryAuditError::ArithmeticOverflow,
                "elastic memory sum overflowed", actual);
  }
  if (dynamic > budget.dynamicBudgetBytes) {
    return fail(MemoryAuditError::CategoryExceedsPlan,
                "resident state and KV exceed the "
                "unified dynamic budget",
                actual);
  }
  if (categorized > actual.backendAllocatedBytes ||
      actual.backendAllocatedBytes > actual.deviceCurrentAllocatedBytes ||
      actual.deviceCurrentAllocatedBytes > actual.devicePeakAllocatedBytes) {
    return fail(MemoryAuditError::BackendAccountingMismatch,
                "current backend/category totals are inconsistent with Metal",
                actual);
  }

  uint64_t backendUnclassified = actual.backendAllocatedBytes - categorized;
  uint64_t deviceUntracked =
      actual.deviceCurrentAllocatedBytes - actual.backendAllocatedBytes;
  if (actual.deviceCurrentAllocatedBytes > budget.hardBudgetBytes ||
      actual.devicePeakAllocatedBytes > budget.hardBudgetBytes ||
      actual.estimatedWarmupPeakBytes > budget.hardBudgetBytes) {
    return fail(MemoryAuditError::HardBudgetExceeded,
                "actual or estimated Metal footprint exceeds hard budget",
                actual);
  }
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  uint64_t unclassified = 0;
  if (!checkedAdd(backendUnclassified, deviceUntracked, unclassified)) {
    return fail(MemoryAuditError::ArithmeticOverflow,
                "unclassified memory sum overflowed", actual);
  }
  if (unclassified > reserves) {
    return fail(
        MemoryAuditError::RuntimeReserveExceeded,
        "pipeline and runtime allocations exceed their explicit reserve",
        actual);
  }
  // The warmup estimate adds the reserves as the bound on unclassified
  // memory; the device peak holds that memory as it is. Compare the two with
  // the measured unclassified bytes in place of the reserves, so that the
  // rule measures the estimate rather than the reserves' unused part.
  if (actual.estimatedWarmupPeakBytes <= reserves) {
    return fail(MemoryAuditError::WarmupEstimateDeviation,
                "warmup estimate does not include the runtime reserves",
                actual);
  }
  const uint64_t predicted =
      actual.estimatedWarmupPeakBytes - reserves + unclassified;
  uint64_t difference = actual.devicePeakAllocatedBytes > predicted
                            ? actual.devicePeakAllocatedBytes - predicted
                            : predicted - actual.devicePeakAllocatedBytes;
  uint64_t basisPoints =
      difference > std::numeric_limits<uint64_t>::max() / 10'000
          ? std::numeric_limits<uint64_t>::max()
          : difference * 10'000 / predicted;
  if (basisPoints > kMaximumWarmupDeviationBasisPoints) {
    return fail(
        MemoryAuditError::WarmupEstimateDeviation,
        "actual Metal warmup peak differs from estimate by more than 5%",
        actual);
  }

  MemoryAuditResult result;
  result.valid = true;
  result.message = "actual Metal memory fits immutable plan";
  result.actual = actual;
  result.categorizedBytes = categorized;
  result.backendUnclassifiedBytes = backendUnclassified;
  result.deviceUntrackedBytes = deviceUntracked;
  result.warmupPeakDeviationBasisPoints = static_cast<uint32_t>(basisPoints);
  result.actualHeadroomBytes =
      budget.hardBudgetBytes - actual.devicePeakAllocatedBytes;
  return result;
}

std::string MemoryAuditResult::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"scope\":\"startup_warmup\","
      << "\"valid\":" << (valid ? "true" : "false") << ',' << "\"error\":\""
      << memoryAuditErrorName(error) << "\","
      << "\"categorized_bytes\":" << categorizedBytes << ','
      << "\"backend_unclassified_bytes\":" << backendUnclassifiedBytes << ','
      << "\"device_untracked_bytes\":" << deviceUntrackedBytes << ','
      << "\"warmup_peak_deviation_basis_points\":"
      << warmupPeakDeviationBasisPoints << ','
      << "\"actual_headroom_bytes\":" << actualHeadroomBytes << ','
      << "\"backend_allocated_bytes\":" << actual.backendAllocatedBytes << ','
      << "\"device_current_allocated_bytes\":"
      << actual.deviceCurrentAllocatedBytes << ','
      << "\"device_peak_allocated_bytes\":" << actual.devicePeakAllocatedBytes
      << ','
      << "\"estimated_warmup_peak_bytes\":" << actual.estimatedWarmupPeakBytes
      << '}';
  return out.str();
}

std::string MemoryAuditResult::describe() const {
  std::ostringstream out;
  out << (valid ? "memory audit valid" : "memory audit failed") << " ["
      << memoryAuditErrorName(error) << "]: " << message
      << "; peak=" << actual.devicePeakAllocatedBytes
      << "; estimate=" << actual.estimatedWarmupPeakBytes
      << "; headroom=" << actualHeadroomBytes;
  return out.str();
}

} // namespace splash::engine
