#include "TestChecks.hpp"
#include "TestModel.hpp"
#include "engine/MemoryAudit.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

EngineMemoryPlan plan(uint64_t visionBytes = kGiB,
                      uint64_t stateStagingBytes = 0, uint64_t aneFfnBytes = 0) {
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
  ModelMemoryProfile model =
      test::modelMemoryProfile(2 * kGiB, 1 * kGiB, visionBytes);
  model.footprint.stateStagingBytes = stateStagingBytes;
  model.footprint.aneFfnBytes = aneFfnBytes;
  return test::requireMemoryPlan(device, model);
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
  result.stateAllocatedBytes = b.laneStateBytes * 3;
  result.sharedPrefillBytes = b.sharedPrefillBytes;
  result.sharedDecodeBytes = b.sharedDecodeBytes;
  result.kvAllocatedBytes = b.kvExtentBytes;
  result.backendAllocatedBytes =
      result.targetWeightsBytes + result.draftWeightsBytes +
      result.visionWeightsBytes + result.stateAllocatedBytes +
      result.sharedPrefillBytes + result.sharedDecodeBytes +
      result.kvAllocatedBytes + unclassifiedBytes;
  result.deviceCurrentAllocatedBytes = result.backendAllocatedBytes;
  // Warmup peaked 16 MiB higher, in the backend's buffers.
  result.devicePeakAllocatedBytes = result.backendAllocatedBytes + 16 * kMiB;
  result.backendPeakAllocatedBytes = result.devicePeakAllocatedBytes;
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
  overflow.backendAllocatedBytes -= overflow.stateAllocatedBytes;
  overflow.stateAllocatedBytes = memoryPlan.breakdown().dynamicBudgetBytes;
  overflow.backendAllocatedBytes += overflow.stateAllocatedBytes;
  overflow.deviceCurrentAllocatedBytes = overflow.backendAllocatedBytes;
  overflow.devicePeakAllocatedBytes = overflow.backendAllocatedBytes;
  overflow.backendPeakAllocatedBytes = overflow.backendAllocatedBytes;
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

// The plan sets the disk tier's state staging aside beside the reserves, so the
// audit bounds it by that plan: it is neither charged to the reserves nor
// counted a second time beside the backend's peak that includes it.
void testStateStagingHasItsOwnBound() {
  // One Qwen3.8-27B state (DEVELOPMENT.md, SSD cache).
  const uint64_t staging = 187 * kMiB;
  const auto memoryPlan = plan(kGiB, staging);
  const auto &budget = memoryPlan.breakdown();
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  const auto audit = [&](uint64_t stagingBytes, uint64_t unclassifiedBytes) {
    ActualMemoryReport actual = report(memoryPlan, unclassifiedBytes);
    actual.stateStagingBytes = stagingBytes;
    actual.backendAllocatedBytes += stagingBytes;
    actual.deviceCurrentAllocatedBytes += stagingBytes;
    actual.devicePeakAllocatedBytes += stagingBytes;
    actual.backendPeakAllocatedBytes += stagingBytes;
    return auditActualMemory(memoryPlan, actual);
  };
  const auto staged = audit(staging, reserves);
  require(staged.valid && staged.backendUnclassifiedBytes == reserves &&
              staged.devicePeakDeviationBasisPoints == 0,
          "state staging was charged to the reserves or counted twice");
  require(audit(staging + 1, 0).error == MemoryAuditError::CategoryExceedsPlan,
          "state staging beyond its plan was accepted");
  require(audit(0, 0).valid,
          "a plan with state staging failed without a started disk tier");
}

// The prefill FFN's Neural Engine split is planned as a category of its own:
// the audit bounds it by that plan, apart from the reserves, and passes a
// start that ran without it.
void testNeuralEngineSplitHasItsOwnBound() {
  // Qwen3.8-27B's split at the M6's share (DEVELOPMENT.md, Neural Engine prefill).
  const uint64_t surfaces = 508 * kMiB;
  const auto memoryPlan = plan(kGiB, 0, surfaces);
  const auto &budget = memoryPlan.breakdown();
  const uint64_t reserves =
      budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
  const auto audit = [&](uint64_t splitBytes, uint64_t unclassifiedBytes) {
    ActualMemoryReport actual = report(memoryPlan, unclassifiedBytes);
    actual.aneFfnBytes = splitBytes;
    actual.backendAllocatedBytes += splitBytes;
    actual.deviceCurrentAllocatedBytes += splitBytes;
    actual.devicePeakAllocatedBytes += splitBytes;
    actual.backendPeakAllocatedBytes += splitBytes;
    return auditActualMemory(memoryPlan, actual);
  };
  const auto split = audit(surfaces, reserves);
  require(split.valid && split.backendUnclassifiedBytes == reserves,
          "the split was charged to the reserves or counted twice");
  require(audit(surfaces + 1, 0).error == MemoryAuditError::CategoryExceedsPlan,
          "a split beyond its plan was accepted");
  require(audit(0, 0).valid, "a start without the split failed its audit");
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
  actual.backendPeakAllocatedBytes = actual.devicePeakAllocatedBytes;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::HardBudgetExceeded,
          "hard budget overflow was accepted");

  actual = report(memoryPlan);
  actual.backendPeakAllocatedBytes = actual.devicePeakAllocatedBytes / 2;
  require(auditActualMemory(memoryPlan, actual).error ==
              MemoryAuditError::DevicePeakDeviation,
          "a device peak far above the backend's was accepted");
}

// The reserves bound the memory no category covers; they do not predict it.
// A correct warmup passes whatever part of them that memory takes, however
// small the model, and a device peak the accounting misses still fails.
void testDevicePeakDeviationExcludesReserves() {
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
  for (const uint64_t weightsMiB : {16'589, 12'288, 9'216, 6'144}) {
    for (const uint64_t untrackedMiB : {100, 300}) {
      const auto memoryPlan = test::requireMemoryPlan(
          device, test::modelMemoryProfile(weightsMiB * kMiB, 1, 0));
      const auto &b = memoryPlan.breakdown();
      ActualMemoryReport actual;
      actual.targetWeightsBytes = b.targetWeightsBytes;
      actual.draftWeightsBytes = b.draftWeightsBytes;
      actual.stateAllocatedBytes = 4 * b.laneStateBytes;
      actual.sharedPrefillBytes = b.sharedPrefillBytes;
      actual.sharedDecodeBytes = b.sharedDecodeBytes;
      actual.kvAllocatedBytes = b.kvExtentBytes;
      actual.backendAllocatedBytes =
          actual.targetWeightsBytes + actual.draftWeightsBytes +
          actual.stateAllocatedBytes + actual.sharedPrefillBytes +
          actual.sharedDecodeBytes + actual.kvAllocatedBytes;
      actual.deviceCurrentAllocatedBytes =
          actual.backendAllocatedBytes + untrackedMiB * kMiB;
      actual.devicePeakAllocatedBytes = actual.deviceCurrentAllocatedBytes;
      actual.backendPeakAllocatedBytes = actual.backendAllocatedBytes;
      const std::string model = std::to_string(weightsMiB) +
                                " MiB of weights, " +
                                std::to_string(untrackedMiB) + " MiB untracked";
      const auto audit = auditActualMemory(memoryPlan, actual);
      require(audit.valid && audit.devicePeakDeviationBasisPoints == 0,
              (model + ": correct warmup failed the device peak gate").c_str());
      actual.devicePeakAllocatedBytes +=
          actual.deviceCurrentAllocatedBytes / 16;
      require(auditActualMemory(memoryPlan, actual).error ==
                  MemoryAuditError::DevicePeakDeviation,
              (model + ": a 6% device peak deviation passed").c_str());
    }
  }
}

} // namespace

int main() {
  try {
    testUnifiedDynamicAudit();
    testOptionalVisionAudit();
    testUnreportedAllocationsCountAgainstReserves();
    testStateStagingHasItsOwnBound();
    testNeuralEngineSplitHasItsOwnBound();
    testFixedCategoryAndPeakFailures();
    testDevicePeakDeviationExcludesReserves();
    std::cout << "elastic memory audit tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory audit tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
