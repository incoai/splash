#include "TestModel.hpp"
#include "engine/MemoryAudit.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

EngineMemoryPlan plan(uint64_t visionBytes = kGiB,
                      uint64_t kvStagingBytes = 0) {
  DeviceCapabilities device;
  device.deviceName = "test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = 32 * kGiB;
  device.recommendedMaxWorkingSetBytes = 24 * kGiB;
  device.maxBufferLengthBytes = 16 * kGiB;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;
  ModelMemoryProfile model =
      test::modelMemoryProfile(2 * kGiB, 1 * kGiB, visionBytes);
  model.footprint.kvStagingBytes = kvStagingBytes;
  return requireEngineMemoryPlan(device, model);
}

// A consistent warmup report; `unclassifiedBytes` are backend buffers no
// loader reported.
ActualMemoryReport report(const EngineMemoryPlan &memoryPlan,
                          uint64_t unclassifiedBytes = 0) {
  const auto &b = memoryPlan.breakdown();
  ActualMemoryReport result;
  result.targetWeightsBytes = b.targetWeightsBytes;
  result.draftWeightsBytes = b.draftWeightsBytes;
  result.visionWeightsBytes = b.visionWeightsBytes;
  result.stateResidentBytes = b.activeStateCellBytes * 3;
  result.sharedPrefillBytes = b.sharedPrefillBytes;
  result.sharedDecodeBytes = b.sharedDecodeBytes;
  result.kvResidentBytes = b.kvExtentBytes;
  result.backendAllocatedBytes =
      result.targetWeightsBytes + result.draftWeightsBytes +
      result.visionWeightsBytes + result.stateResidentBytes +
      result.sharedPrefillBytes + result.sharedDecodeBytes +
      result.kvResidentBytes + unclassifiedBytes;
  result.deviceCurrentAllocatedBytes = result.backendAllocatedBytes;
  result.devicePeakAllocatedBytes = result.backendAllocatedBytes + 16 * kMiB;
  // Warmup estimates the categories' peak and adds the reserves.
  result.estimatedWarmupPeakBytes = result.devicePeakAllocatedBytes -
                                    unclassifiedBytes + b.pipelineReserveBytes +
                                    b.runtimeOverheadReserveBytes;
  return result;
}

void testUnifiedDynamicAudit() {
  EngineMemoryPlan memoryPlan = plan();
  ActualMemoryReport actual = report(memoryPlan);
  auto valid = auditActualMemory(memoryPlan, actual);
  require(valid.valid && valid.error == MemoryAuditError::None,
          "valid elastic memory report failed audit");
  require(valid.toStatusJson().find("\"scope\":\"startup_warmup\"") !=
              std::string::npos,
          "memory audit status does not identify its startup scope");

  ActualMemoryReport overflow = actual;
  overflow.backendAllocatedBytes -= overflow.stateResidentBytes;
  overflow.stateResidentBytes = memoryPlan.breakdown().dynamicBudgetBytes;
  overflow.backendAllocatedBytes += overflow.stateResidentBytes;
  overflow.deviceCurrentAllocatedBytes = overflow.backendAllocatedBytes;
  overflow.devicePeakAllocatedBytes = overflow.backendAllocatedBytes;
  overflow.estimatedWarmupPeakBytes = overflow.backendAllocatedBytes;
  auto rejected = auditActualMemory(memoryPlan, overflow);
  require(!rejected.valid &&
              rejected.error == MemoryAuditError::CategoryExceedsPlan,
          "dynamic state/KV budget overflow was accepted");
}

void testOptionalVisionAudit() {
  const auto textOnly = plan(0);
  require(auditActualMemory(textOnly, report(textOnly)).valid,
          "text-only warmup requires nonexistent vision weights");
}

// Weights are planned from what loaded, so their rows have no bound of their
// own; a buffer a loader does not report counts against the reserves.
void testUnreportedAllocationsCountAgainstReserves() {
  const auto memoryPlan = plan();
  const auto &budget = memoryPlan.breakdown();
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  const auto withinReserves =
      auditActualMemory(memoryPlan, report(memoryPlan, reserves));
  require(withinReserves.valid &&
              withinReserves.backendUnclassifiedBytes == reserves,
          "unreported allocations that fit the reserves were rejected or "
          "not counted");
  require(auditActualMemory(memoryPlan, report(memoryPlan, reserves + 1))
                  .error == MemoryAuditError::RuntimeReserveExceeded,
          "unreported allocations beyond the reserves were accepted");
}

// The plan sets the disk tier's KV staging aside beside the reserves, so the
// audit bounds it by that plan: it is neither charged to the reserves nor
// counted a second time beside the warmup estimate that includes it.
void testKvStagingHasItsOwnBound() {
  const uint64_t ring = 130 * kMiB;
  const auto memoryPlan = plan(kGiB, ring);
  const auto &budget = memoryPlan.breakdown();
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  const auto audit = [&](uint64_t stagingBytes, uint64_t unclassifiedBytes) {
    ActualMemoryReport actual = report(memoryPlan, unclassifiedBytes);
    actual.kvStagingBytes = stagingBytes;
    actual.backendAllocatedBytes += stagingBytes;
    actual.deviceCurrentAllocatedBytes += stagingBytes;
    actual.devicePeakAllocatedBytes += stagingBytes;
    actual.estimatedWarmupPeakBytes += stagingBytes;
    return auditActualMemory(memoryPlan, actual);
  };
  const auto staged = audit(ring, reserves);
  require(staged.valid && staged.backendUnclassifiedBytes == reserves &&
              staged.warmupPeakDeviationBasisPoints == 0,
          "KV staging was charged to the reserves or counted twice");
  require(audit(ring + 1, 0).error == MemoryAuditError::CategoryExceedsPlan,
          "KV staging beyond its plan was accepted");
  require(audit(0, 0).valid,
          "a plan with KV staging failed without a started disk tier");
}

void testFixedCategoryAndPeakFailures() {
  EngineMemoryPlan memoryPlan = plan();
  ActualMemoryReport actual = report(memoryPlan);
  actual.sharedDecodeBytes = memoryPlan.breakdown().sharedDecodeBytes + 1;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::CategoryExceedsPlan,
          "fixed arena overflow was accepted");

  actual = report(memoryPlan);
  actual.devicePeakAllocatedBytes = memoryPlan.breakdown().hardBudgetBytes + 1;
  actual.estimatedWarmupPeakBytes = actual.devicePeakAllocatedBytes;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::HardBudgetExceeded,
          "hard budget overflow was accepted");

  actual = report(memoryPlan);
  actual.estimatedWarmupPeakBytes = actual.devicePeakAllocatedBytes / 2;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::WarmupEstimateDeviation,
          "bad warmup estimate was accepted");
}

// The reserves bound the memory no category covers; they do not predict it.
// A correct warmup passes whatever part of them that memory takes, however
// small the model, and a real estimate error still fails.
void testWarmupDeviationExcludesReserves() {
  DeviceCapabilities device;
  device.deviceName = "test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = 64 * kGiB;
  device.recommendedMaxWorkingSetBytes = 48 * kGiB;
  device.maxBufferLengthBytes = 48 * kGiB;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;
  for (const uint64_t weightsMiB : {16'589, 12'288, 9'216, 6'144}) {
    for (const uint64_t untrackedMiB : {100, 300}) {
      const auto memoryPlan = requireEngineMemoryPlan(
          device, test::modelMemoryProfile(weightsMiB * kMiB, 1, 0));
      const auto &b = memoryPlan.breakdown();
      ActualMemoryReport actual;
      actual.targetWeightsBytes = b.targetWeightsBytes;
      actual.draftWeightsBytes = b.draftWeightsBytes;
      actual.stateResidentBytes = 4 * b.activeStateCellBytes;
      actual.sharedPrefillBytes = b.sharedPrefillBytes;
      actual.sharedDecodeBytes = b.sharedDecodeBytes;
      actual.kvResidentBytes = b.kvExtentBytes;
      actual.backendAllocatedBytes =
          actual.targetWeightsBytes + actual.draftWeightsBytes +
          actual.stateResidentBytes + actual.sharedPrefillBytes +
          actual.sharedDecodeBytes + actual.kvResidentBytes;
      actual.deviceCurrentAllocatedBytes =
          actual.backendAllocatedBytes + untrackedMiB * kMiB;
      actual.devicePeakAllocatedBytes = actual.deviceCurrentAllocatedBytes;
      actual.estimatedWarmupPeakBytes = actual.backendAllocatedBytes +
                                        b.pipelineReserveBytes +
                                        b.runtimeOverheadReserveBytes;
      const std::string model = std::to_string(weightsMiB) +
                                " MiB of weights, " +
                                std::to_string(untrackedMiB) + " MiB untracked";
      const auto audit = auditActualMemory(memoryPlan, actual);
      require(audit.valid && audit.warmupPeakDeviationBasisPoints == 0,
              (model + ": correct warmup failed the estimate gate").c_str());
      actual.devicePeakAllocatedBytes +=
          actual.deviceCurrentAllocatedBytes / 16;
      require(auditActualMemory(memoryPlan, actual).error ==
                  MemoryAuditError::WarmupEstimateDeviation,
              (model + ": a 6% estimate error passed").c_str());
    }
  }
}

} // namespace

int main() {
  try {
    testUnifiedDynamicAudit();
    testOptionalVisionAudit();
    testUnreportedAllocationsCountAgainstReserves();
    testKvStagingHasItsOwnBound();
    testFixedCategoryAndPeakFailures();
    testWarmupDeviationExcludesReserves();
    std::cout << "elastic memory audit tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory audit tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
