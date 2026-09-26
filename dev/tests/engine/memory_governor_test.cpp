#include "TestModel.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

// The governor reads nothing from the backend but its memory statistics, so
// this stand-in lets the tests set them without a GPU.
namespace splash::metal {
namespace {
MetalMemoryStats statistics;
} // namespace

struct MetalBackend::Impl {};
MetalBackend::MetalBackend(std::string, double, uint32_t, double)
    : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() = default;
MetalMemoryStats MetalBackend::memoryStats() const noexcept {
  return statistics;
}
MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
  return statistics;
}
} // namespace splash::metal

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}

// Available host memory is what macOS can hand out without swapping: free
// pages and pageable file-backed and purgeable pages, and with compression
// what compressing the anonymous pages frees.
void testHostAvailabilityCountsReclaimablePages() {
  constexpr uint64_t pageSize = 16384;
  auto availablePages = [](const HostMemoryPages &pages) {
    return estimateHostAvailableMemory(pages, pageSize) / pageSize;
  };
  // free_count includes the speculative pages; they are file-backed too.
  HostMemoryPages pages{
      .free = 15, .speculative = 5, .fileBacked = 35, .purgeable = 5};
  require(availablePages(pages) == 50,
          "host availability does not match macOS reclaimable accounting");
  pages.free += 5;
  pages.speculative += 5;
  require(availablePages(pages) == 50,
          "speculative file pages were counted twice");
  pages.free -= 10;
  require(availablePages(pages) == 40,
          "anonymous or compressed pages did not consume capacity");
  pages.purgeable = 0;
  require(availablePages(pages) == 35,
          "non-purgeable backing received reclaimable credit");
  // Wired file pages leave external_page_count: GPU pinning must reduce
  // available memory, rather than crediting hot weights for KV growth.
  pages.fileBacked -= 10;
  require(availablePages(pages) == 25,
          "wired weights remained available for new allocations");

  // Reading a file into clean cache does not require a second full copy
  // when that same immutable file is mapped again on the next startup.
  require(availablePages({.free = 75}) == 75 &&
              availablePages({.free = 35, .fileBacked = 40}) == 75,
          "cached weights reduced model reload capacity");
  // A 64 GB M5 Pro, whose hw.memsize less its VM queues left 1.2 GiB more
  // (the firmware carve-out, tag storage) that no allocation can have.
  require(estimateHostAvailableMemory(
              {.free = 2'349'632, .speculative = 90'428,
               .fileBacked = 417'802, .purgeable = 23'495},
              pageSize) == 44'245'008'384ULL,
          "unexpected available memory for a 64 GB snapshot");
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(estimateHostAvailableMemory({.free = maximum, .fileBacked = 1}, 1) ==
                  0 &&
              estimateHostAvailableMemory(
                  {.fileBacked = maximum, .purgeable = 1}, 1) == 0 &&
              estimateHostAvailableMemory({.free = maximum}, pageSize) == 0 &&
              estimateHostAvailableMemory({.free = 1, .speculative = 2}, 1) ==
                  0 &&
              estimateHostAvailableMemory({.free = maximum}, 1) == maximum &&
              estimateHostAvailableMemory(pages, 0) == 0,
          "invalid host counters or arithmetic overflow did not fail closed");
  // With compression (below critical system pressure), compressing the
  // anonymous pages frees what the compressor would not keep: at its
  // present ratio, at most 2:1, and 2:1 while it holds nothing.
  const auto compressedPages = [](uint64_t compressor, uint64_t compressed) {
    return estimateHostAvailableMemory({.free = 10, .fileBacked = 20, .anonymous = 40,
                                        .compressor = compressor, .compressed = compressed},
                                       1, true);
  };
  require(estimateHostAvailableMemory({.free = 10, .fileBacked = 20, .anonymous = 40}, 1) == 30 &&
              compressedPages(0, 0) == 50 && compressedPages(10, 15) == 44 && compressedPages(10, 40) == 50 &&
              compressedPages(10, 10) == 30 && compressedPages(10, 5) == 30,
          "compression credit is not the anonymous pages' savings at the compressor's ratio, at most 2:1");
  require(estimateHostAvailableMemory({.free = maximum, .anonymous = 2}, 1, true) == 0,
          "compression credit overflowed");
  require(EngineMemoryPolicy::hostAvailableReserveBytes(16 * kGiB) ==
                  16 * kGiB / 10 &&
              EngineMemoryPolicy::hostAvailableReserveBytes(48 * kGiB) ==
                  2 * kGiB &&
              EngineMemoryPolicy::hostAvailableReserveBytes(128 * kGiB) ==
                  2 * kGiB,
          "the macOS reserve is a tenth of a small machine, 2 GiB above 20 GiB");
}

// A Mac whose recommended working set is numerator/denominator of its memory,
// as Metal reports it (2/3 at 32 GB, 3/4 at 36 GB).
DeviceCapabilities mac(uint64_t physicalGiB, uint64_t numerator,
                       uint64_t denominator) {
  DeviceCapabilities device;
  device.deviceName = "governor-test";
  device.appleGpuFamily = 9;
  device.macosMajor = 26;
  device.macosMinor = 4;
  device.physicalMemoryBytes = physicalGiB * kGiB;
  device.recommendedMaxWorkingSetBytes =
      device.physicalMemoryBytes / denominator * numerator;
  device.maxBufferLengthBytes = device.recommendedMaxWorkingSetBytes;
  device.maxThreadgroupMemoryBytes = 32 * 1024;
  device.maxThreadgroupWidth = 1024;
  device.hasUnifiedMemory = true;
  device.supportsPlacementSparse = true;
  return device;
}

// Grows one request's KV extent by extent, as PageStorage maps it, and
// returns the pages the governor granted.
uint32_t grantKvPages(MemoryGovernor &governor,
                      const EngineMemoryBreakdown &budget) {
  uint32_t granted = 0;
  while (granted < budget.kvVirtualPages) {
    const uint32_t pages =
        std::min(budget.kvExtentPages, budget.kvVirtualPages - granted);
    const uint64_t bytes = uint64_t{pages} * budget.kvPageBytes;
    auto reservation = governor.tryReserve(bytes);
    if (!reservation)
      break;
    metal::statistics.sparseResidentBytes += bytes;
    metal::statistics.deviceCurrentAllocatedBytes += bytes;
    reservation->commit();
    granted += pages;
  }
  return granted;
}

// The plan budgets the memory Metal holds outside the backend's buffers
// (pipelines, driver allocations) inside its reserves and advertises the
// context the rest of the budget holds. One request must reach that context
// whatever part of the reserves this memory takes; memory beyond them is
// still charged, and the device never exceeds the hard budget.
void testAdvertisedContextIsGrantable() {
  struct Machine {
    const char *name;
    uint64_t physicalGiB, numerator, denominator;
    kv::Format format;
    uint32_t advertisedTokens;
  };
  for (const Machine &machine :
       {Machine{"32 GB INT8", 32, 2, 3, kv::Format::Int8, 69'625},
        Machine{"36 GB INT8", 36, 3, 4, kv::Format::Int8, 253'945},
        Machine{"36 GB BF16", 36, 3, 4, kv::Format::BFloat16, 129'241}}) {
    // The 27B with its draft and vision tower: 16.2 GiB of weights.
    ModelMemoryProfile model =
        test::modelMemoryProfile(15 * kGiB, kGiB / 2, 7 * kGiB / 10);
    model.targetKvLayout.format = machine.format;
    const EngineMemoryPlan plan = requireEngineMemoryPlan(
        mac(machine.physicalGiB, machine.numerator, machine.denominator),
        model);
    const EngineMemoryBreakdown &budget = plan.breakdown();
    require(plan.maximumContextTokens() == machine.advertisedTokens,
            std::string(machine.name) + ": unexpected advertised context");
    const uint64_t reserves =
        budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes;
    for (const uint64_t untracked :
         {uint64_t{0}, 64 * kMiB, 150 * kMiB, 300 * kMiB, reserves,
          reserves + 256 * kMiB}) {
      // After warmup the weights, the arenas and the request's state cell
      // are the backend's buffers; Metal holds the untracked bytes besides.
      metal::statistics = {};
      metal::statistics.allocatedBytes =
          budget.targetWeightsBytes + budget.draftWeightsBytes +
          budget.visionWeightsBytes + budget.sharedPrefillBytes +
          budget.sharedDecodeBytes + budget.activeStateCellBytes;
      metal::statistics.deviceCurrentAllocatedBytes =
          metal::statistics.allocatedBytes + untracked;
      metal::MetalBackend backend("unused");
      // Configured as RuntimeResources configures it.
      MemoryGovernor governor(
          backend, budget.hardBudgetBytes, 2 * kGiB,
          [] { return std::optional<uint64_t>(200 * kGiB); }, reserves);
      const uint32_t granted = grantKvPages(governor, budget);
      const std::string context = std::string(machine.name) + ", " +
                                  std::to_string(untracked / kMiB) +
                                  " MiB untracked: granted " +
                                  std::to_string(granted) + " of " +
                                  std::to_string(budget.kvVirtualPages) +
                                  " KV pages";
      require(metal::statistics.deviceCurrentAllocatedBytes <=
                  budget.hardBudgetBytes,
              context + ", beyond the hard budget");
      if (untracked <= reserves)
        require(granted == budget.kvVirtualPages,
                context + ", short of the advertised context");
      else
        require(granted < budget.kvVirtualPages,
                context + ", memory beyond the reserves was not charged");
    }
  }
}

// A request can need more than the host headroom above the warning margin
// while the idle headroom still clears it. Its refusal must start the paced
// reclaim it waits for, after which it fits.
void testHostRefusalStartsReclaim() {
  metal::statistics = {};
  metal::statistics.allocatedBytes = 20 * kGiB;
  metal::statistics.deviceCurrentAllocatedBytes = 20 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  const uint64_t stateCell = 350'224'384;
  std::optional<uint64_t> available = hostReserve + kGiB + 200 * kMiB;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve,
                          [&available] { return available; });
  metal::AllocationFailure failure;
  require(!governor.tryReserve(40 * kGiB, &failure) &&
              failure == metal::AllocationFailure::EngineBudget &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "an engine budget refusal was taken for host pressure");
  require(!governor.tryReserve(stateCell, &failure) &&
              failure == metal::AllocationFailure::HostPressure,
          "a request beyond the host headroom was admitted");
  const MemoryGovernorSnapshot refused = governor.snapshot();
  MemoryPressurePolicy policy;
  const MemoryReclaimDirective directive = policy.update(refused, 0.0, true);
  require(refused.pressure == MemoryPressure::Warning &&
              !refused.hostGrowthAllowed && directive.reclaimEmptyKvExtents &&
              !directive.evictAllUnpinnedPrefixes &&
              !directive.keepResumePoint &&
              directive.targetBytes == kGiB - 200 * kMiB,
          "a request-sized host refusal did not start the paced reclaim");
  // The reclaim reaches the recovery margin, and the request fits.
  *available += directive.targetBytes;
  require(governor.tryReserve(stateCell).has_value() &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "the waiting request did not fit after the reclaim");
}

// The paced passes up to the next measurement continue what transfers held
// back of a pass's target, less what each releases, until it is met. A new
// measurement replaces it, and every critical pass evicts everything afresh.
void testPolicyContinuesHeldBackTarget() {
  MemoryPressurePolicy policy;
  MemoryGovernorSnapshot pressure{.pressure = MemoryPressure::Warning,
                                  .hostMeasurementValid = true,
                                  .hostHeadroomBytes = kHostRecoveryMarginBytes - 300};
  const auto pass = [&](double now, MemoryReclaimResult result) {
    const MemoryReclaimDirective directive = policy.update(pressure, now, true);
    policy.reclaimed(directive, result);
    return directive.targetBytes;
  };
  constexpr MemoryReclaimResult none{};
  require(pass(0.0, {100, ReclaimOutcome::Pending}) == 300 &&
              pass(100.0, {50, ReclaimOutcome::Pending}) == 200 &&
              pass(200.0, {0, ReclaimOutcome::Met}) == 150 && pass(300.0, none) == 0,
          "the paced passes did not continue a held-back target until it was met");
  require(pass(1000.0, {0, ReclaimOutcome::Pending}) == 300, "the pass was not measured");
  pressure.hostHeadroomBytes = kHostRecoveryMarginBytes - 100;
  require(pass(2000.0, none) == 100 && pass(2100.0, none) == 0,
          "a measurement did not replace the held-back target");
  pressure.pressure = MemoryPressure::Critical;
  static_cast<void>(pass(2200.0, {0, ReclaimOutcome::Pending}));
  pressure.pressure = MemoryPressure::Warning;
  require(pass(2300.0, none) == 0, "evicting everything was continued after critical pressure");
}

// With nothing left to reclaim, the hold for the recovery margin could only be
// lifted by other applications: every request, however small, would wait.
// While reclaim reports that, growth that clears the warning margin proceeds
// and a request beyond it holds no other. A pass that finds memory again, or
// a new episode of host pressure, brings the hold back.
void testExhaustedReclaimWaivesTheHold() {
  metal::statistics = {};
  metal::statistics.allocatedBytes = 12 * kGiB;
  metal::statistics.deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB + kGiB / 2;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, [&available] { return available; });
  metal::AllocationFailure failure;
  require(!governor.tryReserve(kGiB, &failure) && failure == metal::AllocationFailure::HostPressure,
          "growth past the warning margin was admitted");
  governor.reclaimed(ReclaimOutcome::Untargeted);
  require(!governor.tryReserve(100 * kMiB) && !governor.snapshot().hostGrowthAllowed,
          "the host refusal did not hold growth for the recovery margin");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  require(governor.snapshot().hostGrowthAllowed && governor.tryReserve(100 * kMiB).has_value(),
          "growth within the warning margin still waited after reclaim was exhausted");
  require(!governor.tryReserve(kGiB, &failure) && failure == metal::AllocationFailure::HostPressure &&
              governor.snapshot().pressure == MemoryPressure::Warning &&
              governor.tryReserve(100 * kMiB).has_value(),
          "a request past the warning margin was admitted or held the others");
  governor.reclaimed(ReclaimOutcome::Pending);
  require(!governor.tryReserve(100 * kMiB), "the hold did not return with memory to reclaim");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().pressure == MemoryPressure::Normal, "the host did not recover");
  available = hostReserve + kGiB + kGiB / 2;
  require(!governor.tryReserve(kGiB) && !governor.tryReserve(100 * kMiB),
          "an earlier episode's exhausted reclaim waived the hold");
}

} // namespace

int main() {
  try {
    testHostAvailabilityCountsReclaimablePages();
    testAdvertisedContextIsGrantable();
    testHostRefusalStartsReclaim();
    testPolicyContinuesHeldBackTarget();
    testExhaustedReclaimWaivesTheHold();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
