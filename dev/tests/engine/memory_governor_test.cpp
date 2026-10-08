#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "TestModel.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

// Admits bytes through the governor's one admission path, running allocate
// while they are reserved.
metal::AllocationResult admit(MemoryGovernor &governor, uint64_t bytes,
                              const std::function<void()> &allocate = [] {}) {
  return governor.allocationAdmission()(bytes, allocate);
}

// Charges bytes to the backend, as an allocation does.
void allocate(uint64_t bytes) {
  test::metalStatistics().allocatedBytes += bytes;
  test::metalStatistics().deviceCurrentAllocatedBytes += bytes;
}

// Available host memory is what macOS can hand out without swapping: free
// pages and pageable file-backed and purgeable pages, and with compression
// what compressing the anonymous pages frees.
void testHostAvailabilityCountsReclaimablePages() {
  constexpr uint64_t pageSize = 16384;
  auto availablePages = [](const HostMemoryPages &pages) {
    return estimateHostAvailableMemory(pages, pageSize, false) / pageSize;
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
  // Wired file pages leave external_page_count: pinned pages must reduce
  // available memory rather than count as reclaimable.
  pages.fileBacked -= 10;
  require(availablePages(pages) == 25,
          "wired file pages remained available for new allocations");

  // Clean file cache is reclaimable: reading a file takes no capacity from
  // a later allocation.
  require(availablePages({.free = 75}) == 75 &&
              availablePages({.free = 35, .fileBacked = 40}) == 75,
          "clean file cache reduced allocation capacity");
  // A 64 GB M5 Pro, whose hw.memsize less its VM queues left 1.2 GiB more
  // (the firmware carve-out, tag storage) that no allocation can have.
  require(estimateHostAvailableMemory(
              {.free = 2'349'632, .speculative = 90'428,
               .fileBacked = 417'802, .purgeable = 23'495},
              pageSize, false) == 44'245'008'384ULL,
          "unexpected available memory for a 64 GB snapshot");
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(estimateHostAvailableMemory({.free = maximum, .fileBacked = 1}, 1, false) ==
                  0 &&
              estimateHostAvailableMemory(
                  {.fileBacked = maximum, .purgeable = 1}, 1, false) == 0 &&
              estimateHostAvailableMemory({.free = maximum}, pageSize, false) == 0 &&
              estimateHostAvailableMemory({.free = 1, .speculative = 2}, 1, false) ==
                  0 &&
              estimateHostAvailableMemory({.free = maximum}, 1, false) == maximum &&
              estimateHostAvailableMemory(pages, 0, false) == 0,
          "invalid host counters or arithmetic overflow did not fail closed");
  // With compression (below critical system pressure), compressing the
  // anonymous pages frees what the compressor would not keep: at its
  // present ratio, at most 2:1, and 2:1 while it holds nothing.
  const auto compressedPages = [](uint64_t compressor, uint64_t compressed) {
    return estimateHostAvailableMemory({.free = 10, .fileBacked = 20, .anonymous = 40,
                                        .compressor = compressor, .compressed = compressed},
                                       1, true);
  };
  require(estimateHostAvailableMemory({.free = 10, .fileBacked = 20, .anonymous = 40}, 1, false) == 30 &&
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
  return device;
}

// Grows one request's KV extent by extent, as PageStorage allocates it, and
// returns the pages the governor granted.
uint32_t grantKvPages(MemoryGovernor &governor,
                      const EngineMemoryBreakdown &budget) {
  uint32_t granted = 0;
  while (granted < budget.kvCapacityPages) {
    const uint32_t pages = budget.kvExtentPages;
    const uint64_t bytes = uint64_t{pages} * budget.kvPageBytes;
    if (!admit(governor, bytes, [bytes] { allocate(bytes); }))
      break;
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
        Machine{"36 GB BF16", 36, 3, 4, kv::Format::BFloat16, 129'049}}) {
    // The 27B with its draft and vision tower: 16.2 GiB of weights.
    ModelMemoryProfile model =
        test::modelMemoryProfile(15 * kGiB, kGiB / 2, 7 * kGiB / 10);
    model.targetKvLayout.format = machine.format;
    const EngineMemoryPlan plan = test::requireMemoryPlan(
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
      // After warmup the weights, the arenas and the request's lane state
      // are the backend's buffers; Metal holds the untracked bytes besides.
      test::metalStatistics() = {};
      test::metalStatistics().allocatedBytes =
          budget.targetWeightsBytes + budget.draftWeightsBytes +
          budget.visionWeightsBytes + budget.sharedPrefillBytes +
          budget.sharedDecodeBytes + budget.laneStateBytes;
      test::metalStatistics().deviceCurrentAllocatedBytes =
          test::metalStatistics().allocatedBytes + untracked;
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
                                  std::to_string(budget.kvCapacityPages) +
                                  " KV pages";
      require(test::metalStatistics().deviceCurrentAllocatedBytes <=
                  budget.hardBudgetBytes,
              context + ", beyond the hard budget");
      if (untracked <= reserves)
        require(granted == budget.kvCapacityPages,
                context + ", short of the advertised context");
      else
        require(granted < budget.kvCapacityPages,
                context + ", memory beyond the reserves was not charged");
    }
  }
}

// A reservation holds exactly its bytes under the limit while its
// allocation runs. An admission whose allocation the driver denies reports
// that and releases its reservation; a backend defect propagates and
// releases it too.
void testReservationsAndAdmissionClasses() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  MemoryGovernor governor(backend, limit, hostReserve, [] {
    return std::optional<uint64_t>(hostReserve + 3 * kGiB);
  }, 0);
  const metal::AllocationResult held = admit(governor, limit - 1, [&] {
    require(static_cast<bool>(admit(governor, 1)),
            "memory governor held more than the reserved bytes");
    const metal::AllocationResult over = admit(governor, 2);
    require(!over && over.failure == metal::AllocationFailure::EngineBudget &&
                governor.snapshot().deniedReservations == 1,
            "memory governor oversold its limit");
  });
  require(static_cast<bool>(held), "memory governor refused bytes under its limit");
  const metal::AllocationResult driverDenied = admit(governor, 1024, [] {
    throw metal::MetalAllocationError("injected driver allocation denial");
  });
  require(!driverDenied &&
              driverDenied.failure == metal::AllocationFailure::DriverRejected &&
              admit(governor, limit),
          "driver allocation denial did not release its reservation");
  rejects(
      [&] {
        static_cast<void>(admit(governor, 1024, [] {
          throw metal::MetalBackendError("injected backend defect");
        }));
      },
      "injected backend defect", "admission swallowed a backend defect");
  require(static_cast<bool>(admit(governor, limit)),
          "a backend defect leaked its reservation");
}

// Critical pressure stops all growth. Host headroom below the warning margin
// stops it too, but as warning pressure, which sheds cache in paced passes,
// and only crossing the recovery margin reopens it. Exhausting the engine's
// limit is not host pressure, and missing telemetry fails closed without
// discarding valid cache.
void testHostHeadroomHysteresis() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + 3 * kGiB;
  MemoryGovernor governor(backend, limit, hostReserve,
                          [&available] { return available; }, 0);
  // Whether one byte more is admitted.
  const auto admitsMore = [&governor] { return static_cast<bool>(admit(governor, 1)); };
  governor.setPressure(MemoryPressure::Critical);
  require(!admitsMore(), "critical pressure did not stop new growth");
  governor.setPressure(MemoryPressure::Normal);
  require(admitsMore(), "normal pressure did not reopen admission");
  available = hostReserve;
  require(!admitsMore(), "host reserve did not stop unified-memory growth");
  // Reaching the reserve is a warning that sheds cache in paced passes;
  // only the OS critical verdict drops every evictable entry.
  require(governor.snapshot().pressure == MemoryPressure::Warning,
          "reaching the host reserve was treated as critical");
  available = hostReserve / 2;
  require(governor.snapshot().pressure == MemoryPressure::Warning && !admitsMore(),
          "low availability escalated to destructive system pressure");
  available = hostReserve + kGiB / 2;
  require(governor.snapshot().pressure == MemoryPressure::Warning &&
              !governor.snapshot().hostGrowthAllowed && !admitsMore(),
          "low host headroom did not request proactive cache reclaim");
  available = hostReserve + 3 * kGiB / 2;
  require(governor.snapshot().pressure == MemoryPressure::Warning,
          "warning pressure recovered without crossing the hysteresis");
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().pressure == MemoryPressure::Normal &&
              governor.snapshot().hostGrowthAllowed,
          "host recovery did not reopen admission");
  const metal::AllocationResult full = admit(governor, limit, [&] {
    const MemoryGovernorSnapshot snapshot = governor.snapshot();
    require(snapshot.headroomBytes == 0 && snapshot.hostGrowthAllowed,
            "engine budget exhaustion was confused with host pressure");
  });
  require(static_cast<bool>(full), "engine capacity reservation failed");
  available.reset();
  require(!admitsMore() && !governor.snapshot().hostMeasurementValid,
          "missing host memory telemetry did not fail closed");
  MemoryPressurePolicy missingPolicy;
  const auto missing = governor.snapshot();
  const auto missingDirective = missingPolicy.update(missing, 0.0, false);
  require(missing.pressure == MemoryPressure::Warning && missingDirective &&
              !missingDirective->critical && missingDirective->targetBytes == 0,
          "missing telemetry discarded valid cache");
  governor.setPressure(MemoryPressure::Critical);
  require(missingPolicy.update(governor.snapshot(), 1.0, false).value().critical,
          "missing telemetry hid critical system pressure");
  governor.setPressure(MemoryPressure::Normal);
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().hostHeadroomBytes == 3 * kGiB,
          "host memory headroom accounting is wrong");
}

// Low reclaimable memory must not reopen growth or suppress the pressure
// reclaimer; reclaimable file cache can reopen it. A system warning alone
// does not block growth the host headroom clears.
void testReclaimablePagesReopenGrowth() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + 3 * kGiB;
  MemoryGovernor governor(backend, limit, hostReserve,
                          [&available] { return available; }, 0);
  const auto admitsMore = [&governor] { return static_cast<bool>(admit(governor, 1)); };
  HostMemoryPages pressurePages{.free = hostReserve, .fileBacked = kGiB / 2};
  available = estimateHostAvailableMemory(pressurePages, 1, false);
  MemoryPressurePolicy hostPolicy;
  const auto hostDirective = hostPolicy.update(governor.snapshot(), 0.0, false);
  require(!admitsMore() && governor.snapshot().pressure == MemoryPressure::Warning &&
              hostDirective && !hostDirective->critical &&
              hostDirective->targetBytes == kGiB,
          "low reclaimable memory bypassed bounded pressure recovery");
  pressurePages.fileBacked = 3 * kGiB;
  available = estimateHostAvailableMemory(pressurePages, 1, false);
  require(governor.snapshot().hostGrowthAllowed && admitsMore() &&
              !hostPolicy.update(governor.snapshot(), 1000.0, false),
          "reclaimable host recovery did not reopen normal admission");
  governor.setPressure(MemoryPressure::Warning);
  require(governor.snapshot().pressure == MemoryPressure::Warning &&
              governor.snapshot().hostGrowthAllowed && admitsMore(),
          "system warning blocked growth despite sufficient host headroom");
  available = hostReserve + kGiB / 2;
  require(!admitsMore(), "system warning bypassed insufficient host headroom");
}

// Reading the governor moves nothing: /status and the engine's growth
// queries see what it would decide, and only a reservation or the control
// pass moves the host's hysteresis. A recovery that only a read saw leaves
// growth held once the host falls back inside the recovery margin.
void testSnapshotLeavesTheHysteresis() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 64 * 1024, hostReserve,
                          [&available] { return available; }, 0);
  require(!admit(governor, 1), "growth inside the warning margin was admitted");
  available = hostReserve + 3 * kGiB;
  require(governor.snapshot().hostGrowthAllowed, "a read did not see the host recover");
  available = hostReserve + 3 * kGiB / 2;
  require(!governor.snapshot().hostGrowthAllowed && !admit(governor, 1),
          "a read moved the host's hysteresis");
  available = hostReserve + 3 * kGiB;
  require(governor.evaluate().hostGrowthAllowed,
          "the control pass did not see the host recover");
  available = hostReserve + 3 * kGiB / 2;
  require(governor.snapshot().hostGrowthAllowed && admit(governor, 1),
          "the control pass did not move the host's hysteresis");
}

// Warning pressure asks for what measured headroom lacks of the recovery
// margin, at most the warning margin, once per settled measurement; critical
// pressure evicts everything.
void testPressurePolicyDirectives() {
  MemoryPressurePolicy policy;
  MemoryGovernorSnapshot policySnapshot;
  policySnapshot.pressure = MemoryPressure::Warning;
  policySnapshot.hostMeasurementValid = true;
  policySnapshot.systemPressure = MemoryPressure::Warning;
  policySnapshot.hostHeadroomBytes = 3 * kGiB / 2;
  const auto firstDirective = policy.update(policySnapshot, 0.0, false);
  require(firstDirective && !firstDirective->critical &&
              firstDirective->targetBytes == kGiB / 2,
          "warning pressure ignored measured headroom");
  require(policy.update(policySnapshot, 500.0, false).value().targetBytes == 0 &&
              policy.update(policySnapshot, 999.0, false).value().targetBytes == 0,
          "warning pressure reclaimed again before telemetry settled");
  // Another application consumed more memory in the SAME warning episode.
  // Earlier reclaimed bytes must not offset this new deficit.
  policySnapshot.hostHeadroomBytes = kGiB / 4;
  require(policy.update(policySnapshot, 1000.0, false).value().targetBytes == kGiB,
          "persistent warning did not request a new bounded shrink pass");
  policySnapshot.systemPressure = MemoryPressure::Normal;
  policySnapshot.hostHeadroomBytes = 7 * kGiB / 4;
  require(policy.update(policySnapshot, 2000.0, false).value().targetBytes == kGiB / 4,
          "pressure recovery ignored the current smaller deficit");
  policySnapshot.pressure = MemoryPressure::Normal;
  require(!policy.update(policySnapshot, 2100.0, false),
          "normal pressure requested cache reclaim");
  policySnapshot.pressure = MemoryPressure::Warning;
  require(policy.update(policySnapshot, 2101.0, false).value().targetBytes == kGiB / 4,
          "a new pressure episode inherited an old cooldown");
  policySnapshot.systemPressure = MemoryPressure::Warning;
  policySnapshot.hostHeadroomBytes = 3 * kGiB;
  const auto advisory = policy.update(policySnapshot, 3101.0, false);
  require(advisory && !advisory->critical && advisory->targetBytes == 0,
          "system warning discarded live cache despite sufficient headroom");
  // The newest publication is what a follow-up resumes from; rebuilding it
  // costs a whole prefill, so a shrink nothing is waiting for leaves it and
  // takes the rest. A waiting request outranks it, and Critical takes all.
  policySnapshot.hostHeadroomBytes = kGiB / 4;
  const auto speculative = policy.update(policySnapshot, 4101.0, false);
  require(speculative && speculative->targetBytes == kGiB && speculative->keepResumePoint,
          "a speculative shrink discarded the resume point");
  const auto demanded = policy.update(policySnapshot, 5101.0, true);
  require(demanded && demanded->targetBytes == kGiB && !demanded->keepResumePoint,
          "a waiting request could not reach the resume point");
  policySnapshot.pressure = MemoryPressure::Critical;
  const auto criticalDirective = policy.update(policySnapshot, 2102.0, false);
  require(criticalDirective && criticalDirective->critical &&
              !criticalDirective->keepResumePoint,
          "critical pressure did not request aggressive reclaim");
}

// A request can need more than the host headroom above the warning margin
// while the idle headroom still clears it. Its refusal must start the paced
// reclaim it waits for, after which it fits.
void testHostRefusalStartsReclaim() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 20 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 20 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  const uint64_t laneState = 350'224'384;
  std::optional<uint64_t> available = hostReserve + 64 * kGiB;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve,
                          [&available] { return available; }, 0);
  const metal::AllocationResult beyondLimit = admit(governor, 40 * kGiB);
  require(!beyondLimit && beyondLimit.failure == metal::AllocationFailure::EngineBudget,
          "an engine budget refusal was taken for host pressure");
  // The host refuses the same probe once it has 1.2 GiB of room, and that
  // is the cause reported; a probe beyond the limit holds no host pressure.
  available = hostReserve + kGiB + 200 * kMiB;
  const metal::AllocationResult shared = admit(governor, 40 * kGiB);
  require(!shared && shared.failure == metal::AllocationFailure::HostPressure &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "a refusal the host shares was reported as the engine's");
  const metal::AllocationResult beyondHost = admit(governor, laneState);
  require(!beyondHost && beyondHost.failure == metal::AllocationFailure::HostPressure,
          "a request beyond the host headroom was admitted");
  const MemoryGovernorSnapshot refused = governor.snapshot();
  MemoryPressurePolicy policy;
  const std::optional<MemoryReclaimDirective> directive = policy.update(refused, 0.0, true);
  require(refused.pressure == MemoryPressure::Warning &&
              !refused.hostGrowthAllowed && directive && !directive->critical &&
              !directive->keepResumePoint &&
              directive->targetBytes == kGiB - 200 * kMiB,
          "a request-sized host refusal did not start the paced reclaim");
  // The reclaim reaches the recovery margin, and the request fits.
  *available += directive->targetBytes;
  require(static_cast<bool>(admit(governor, laneState)) &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "the waiting request did not fit after the reclaim");
}

// A refusal the host shares with the engine's limit is the host's: it lifts
// with the host's pressure, the limit's only once memory is freed. For a
// request in service the host refuses only under critical pressure.
void testHostRefusalComesBeforeTheEngineLimit() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 14 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 14 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 15 * kGiB, hostReserve, [&available] { return available; }, 0);
  const metal::AllocationResult shared = admit(governor, 2 * kGiB);
  require(!shared && shared.failure == metal::AllocationFailure::HostPressure,
          "a refusal the host shares was reported as the engine's");
  governor.setServing(true);
  const metal::AllocationResult inService = admit(governor, 2 * kGiB);
  require(!inService && inService.failure == metal::AllocationFailure::EngineBudget,
          "the host's margins refused a request in service");
  governor.setPressure(MemoryPressure::Critical);
  const metal::AllocationResult critical = admit(governor, 2 * kGiB);
  require(!critical && critical.failure == metal::AllocationFailure::HostPressure,
          "critical pressure was reported as the engine's limit");
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
    const MemoryReclaimDirective directive = policy.update(pressure, now, true).value();
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
  require(policy.update(pressure, 2150.0, true).value().critical,
          "critical pressure kept the serving footprint");
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
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB + kGiB / 2;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, [&available] { return available; }, 0);
  const metal::AllocationResult pastMargin = admit(governor, kGiB);
  require(!pastMargin && pastMargin.failure == metal::AllocationFailure::HostPressure,
          "growth past the warning margin was admitted");
  governor.reclaimed(ReclaimOutcome::Untargeted);
  require(!admit(governor, 100 * kMiB) && !governor.snapshot().hostGrowthAllowed,
          "the host refusal did not hold growth for the recovery margin");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  require(governor.snapshot().hostGrowthAllowed && admit(governor, 100 * kMiB),
          "growth within the warning margin still waited after reclaim was exhausted");
  const metal::AllocationResult waived = admit(governor, kGiB);
  require(!waived && waived.failure == metal::AllocationFailure::HostPressure &&
              governor.snapshot().pressure == MemoryPressure::Warning &&
              admit(governor, 100 * kMiB),
          "a request past the warning margin was admitted or held the others");
  governor.reclaimed(ReclaimOutcome::Pending);
  require(!admit(governor, 100 * kMiB), "the hold did not return with memory to reclaim");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  available = hostReserve + 3 * kGiB;
  require(governor.evaluate().pressure == MemoryPressure::Normal, "the host did not recover");
  available = hostReserve + kGiB + kGiB / 2;
  require(!admit(governor, kGiB) && !admit(governor, 100 * kMiB),
          "an earlier episode's exhausted reclaim waived the hold");
}

// Host pressure holds back growth that nothing in flight depends on. What a
// request in service needs is granted while the host is short, into its
// reserve too: holding it back would strand the request and the memory it
// already has. Only the engine's limit and critical pressure refuse it, and
// without the mark the margins apply as before.
void testRequestInServiceGrowsThroughHostPressure() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 14 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 14 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 15 * kGiB, hostReserve, [&available] { return available; }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  const metal::AllocationResult idle = grow(400 * kMiB);
  require(!idle && idle.failure == metal::AllocationFailure::HostPressure &&
              !governor.snapshot().hostGrowthAllowed,
          "growth that nothing in service needs cleared no warning margin");

  governor.setServing(true);
  require(grow(400 * kMiB) && !governor.snapshot().hostGrowthAllowed,
          "a request in service waited for the warning margin");
  available = hostReserve / 2;
  require(static_cast<bool>(grow(100 * kMiB)),
          "a request in service waited for the host's reserve");
  const metal::AllocationResult pastLimit = grow(kGiB);
  require(!pastLimit && pastLimit.failure == metal::AllocationFailure::EngineBudget,
          "a request in service grew past the engine's limit");
  governor.setPressure(MemoryPressure::Critical);
  const metal::AllocationResult critical = grow(100 * kMiB);
  require(!critical && critical.failure == metal::AllocationFailure::HostPressure,
          "critical pressure admitted a request in service");
  governor.setPressure(MemoryPressure::Normal);
  require(static_cast<bool>(grow(100 * kMiB)),
          "a request in service stayed refused after critical pressure");

  governor.setServing(false);
  available = hostReserve + kGiB / 2;
  const metal::AllocationResult afterService = grow(100 * kMiB);
  require(!afterService && afterService.failure == metal::AllocationFailure::HostPressure,
          "the mark of a request in service outlived it");
}

// The guard fires when what the compressor holds grows by the warning margin
// within the window while the engine grows, and owes the host's room above
// its reserve and that growth for the hold. Growth spread wider than the
// window, compression while the engine holds still or shrinks, and growth
// across a restart do not fire it. A fire within the hold renews it; the hold
// then ends on its own.
void testCompressionGuardFiresOnTheEnginesGrowth() {
  using Event = CompressionGuard::Event;
  CompressionGuard guard;
  double now = 0.0;
  uint64_t compressed = 4 * kGiB;
  uint64_t charged = 20 * kGiB;
  const uint64_t room = 30 * kGiB;
  // Advances the clock by `step` and samples.
  const auto sample = [&](double step) {
    now += step;
    return guard.update(now, CompressionGuard::Sample{compressed, charged, room});
  };
  // A pass without a sample: the window starts again.
  const auto restart = [&] { return guard.update(now, std::nullopt); };
  require(sample(0.0) == Event::None, "a first sample fired");
  // Twenty samples half a second apart: the compressor grows short of the
  // margin within the window while the engine grows.
  for (int i = 0; i < 20; ++i) {
    compressed += (kHostWarningMarginBytes - 1) / 20;
    charged += kMiB;
    require(sample(500.0) == Event::None, "growth short of the warning margin fired");
  }
  // The same pace for longer: no window holds the margin.
  for (int i = 0; i < 20; ++i) {
    compressed += (kHostWarningMarginBytes - 1) / 21;
    charged += kMiB;
    require(sample(550.0) == Event::None, "growth spread past the window fired");
  }
  require(restart() == Event::None && sample(0.0) == Event::None, "a restart fired");
  // The margin within the window, the engine still: another program's.
  compressed += kHostWarningMarginBytes;
  require(sample(500.0) == Event::None, "growth while the engine held still fired");
  compressed += kHostWarningMarginBytes;
  charged -= kGiB;
  require(sample(500.0) == Event::None, "growth while the engine shrank fired");
  compressed -= kHostWarningMarginBytes;
  require(sample(500.0) == Event::None, "a falling compressor fired");
  // A pass without a sample drops what came before it.
  require(restart() == Event::None, "a restart fired");
  compressed += kHostWarningMarginBytes;
  charged += kGiB;
  require(sample(500.0) == Event::None, "growth across a restart fired");
  // The margin within the window while the engine grows.
  compressed += kHostWarningMarginBytes;
  charged += 200 * kMiB;
  require(sample(500.0) == Event::Fired && guard.growthBytes() == kHostWarningMarginBytes &&
              guard.debtBytes() == room + kHostWarningMarginBytes,
          "the guard missed the engine's growth paid for by compression");
  // The fire paid for that growth: the window starts again.
  require(sample(500.0) == Event::None, "one growth fired twice");
  // A fire within the hold renews it.
  now += kCompressionHoldMilliseconds / 2;
  compressed += 2 * kHostWarningMarginBytes;
  charged += 200 * kMiB;
  require(sample(0.0) == Event::None, "a first sample after a pause fired");
  compressed += 2 * kHostWarningMarginBytes;
  charged += 200 * kMiB;
  require(sample(500.0) == Event::Fired && guard.growthBytes() == 2 * kHostWarningMarginBytes &&
              guard.debtBytes() == room + 2 * kHostWarningMarginBytes,
          "a fire within the hold did not renew it");
  const double renewed = now;
  now = renewed + kCompressionHoldMilliseconds - 1.0;
  require(sample(0.0) == Event::None && guard.debtBytes() != 0, "the hold ended early");
  // The hold ends also on a pass without a sample.
  now = renewed + kCompressionHoldMilliseconds;
  require(restart() == Event::Expired && guard.debtBytes() == 0, "the hold did not end");
  require(sample(500.0) == Event::None, "an ended hold ended again");

  // The growth counts from the compressor's least point in the window, not
  // its oldest, and the engine must have grown since that point.
  const auto at = [&](uint64_t content, uint64_t engine, uint64_t hostRoom = 30 * kGiB) {
    now += 500.0;
    return guard.update(now, CompressionGuard::Sample{content, engine, hostRoom});
  };
  static_cast<void>(restart());
  require(at(5 * kGiB, 20 * kGiB) == Event::None && at(4 * kGiB, 20 * kGiB) == Event::None &&
              at(5 * kGiB, 20 * kGiB + kMiB) == Event::Fired &&
              guard.debtBytes() == 30 * kGiB + kGiB,
          "growth from the compressor's least point did not fire");
  now += kCompressionHoldMilliseconds;
  static_cast<void>(restart());
  // The engine shrank since the compressor's least point and grew a little
  // after: the compression is not its doing.
  require(at(4 * kGiB, 20 * kGiB) == Event::None &&
              at(4 * kGiB + kGiB / 2, 15 * kGiB) == Event::None &&
              at(5 * kGiB, 15 * kGiB + 200 * kMiB) == Event::None,
          "compression while the engine shrank fired after a small regrowth");
  // A fire within the hold owes at least what the earlier one did, even
  // once requests in service took some of the room.
  static_cast<void>(restart());
  require(at(4 * kGiB, 20 * kGiB) == Event::None &&
              at(5 * kGiB, 20 * kGiB + kMiB) == Event::Fired &&
              guard.debtBytes() == 31 * kGiB,
          "a fire after a restart did not owe the room and the growth");
  require(at(5 * kGiB, 40 * kGiB, 10 * kGiB) == Event::None &&
              at(6 * kGiB, 40 * kGiB + kMiB, 10 * kGiB) == Event::Fired &&
              guard.debtBytes() == 31 * kGiB,
          "a renewal owed less than the fire before it");
}

// The debt comes off the host's headroom: growth no request in service needs
// stops and the paced reclaim asks for the recovery margin, while a request
// in service still grows. Memory freed after the fire counts in full, and
// growth resumes once the headroom clears the margins again.
void testHostDebtHoldsGrowthOutsideService() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 20 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 20 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + 30 * kGiB;
  MemoryGovernor governor(backend, 100 * kGiB, hostReserve, [&available] { return available; }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  require(grow(kMiB) && governor.evaluate().hostDebtBytes == 0, "growth was held without a debt");
  governor.setHostDebt(30 * kGiB + kGiB);
  const MemoryGovernorSnapshot owed = governor.evaluate();
  const metal::AllocationResult held = grow(kMiB);
  require(!held && held.failure == metal::AllocationFailure::HostPressure &&
              owed.hostDebtBytes == 31 * kGiB && owed.hostAvailableBytes == hostReserve + 30 * kGiB &&
              owed.hostHeadroomBytes == 0 && !owed.hostGrowthAllowed &&
              owed.pressure == MemoryPressure::Warning,
          "a debt did not hold growth outside service");
  MemoryPressurePolicy policy;
  const auto directive = policy.update(owed, 0.0, false);
  require(directive && !directive->critical && directive->targetBytes == kHostWarningMarginBytes &&
              directive->keepResumePoint,
          "a debt did not start the paced reclaim");
  governor.setServing(true);
  require(static_cast<bool>(grow(kMiB)), "a debt held a request in service");
  governor.setServing(false);
  // The engine returns what it owes and the recovery margin.
  available = hostReserve + 30 * kGiB + kGiB + kHostRecoveryMarginBytes;
  const MemoryGovernorSnapshot repaid = governor.evaluate();
  require(repaid.hostHeadroomBytes == kHostRecoveryMarginBytes && repaid.hostGrowthAllowed &&
              repaid.pressure == MemoryPressure::Normal && grow(kMiB),
          "growth did not resume once the debt was repaid");
  governor.setHostDebt(0);
  require(governor.evaluate().hostHeadroomBytes == 30 * kGiB + kGiB + kHostRecoveryMarginBytes,
          "a cleared debt still held the headroom");
}

} // namespace

int main() {
  try {
    testHostAvailabilityCountsReclaimablePages();
    testAdvertisedContextIsGrantable();
    testReservationsAndAdmissionClasses();
    testHostHeadroomHysteresis();
    testReclaimablePagesReopenGrowth();
    testSnapshotLeavesTheHysteresis();
    testPressurePolicyDirectives();
    testHostRefusalStartsReclaim();
    testHostRefusalComesBeforeTheEngineLimit();
    testPolicyContinuesHeldBackTarget();
    testExhaustedReclaimWaivesTheHold();
    testRequestInServiceGrowsThroughHostPressure();
    testCompressionGuardFiresOnTheEnginesGrowth();
    testHostDebtHoldsGrowthOutsideService();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
