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

// A host whose spare memory is all its available memory: no anonymous
// pages to compress.
std::optional<HostMemory> hostMemory(std::optional<uint64_t> bytes) {
  if (!bytes)
    return std::nullopt;
  return HostMemory{*bytes, *bytes};
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
    return estimateHostMemory(pages, pageSize, false).available / pageSize;
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
  require(estimateHostMemory(
              {.free = 2'349'632, .speculative = 90'428,
               .fileBacked = 417'802, .purgeable = 23'495},
              pageSize, false).available == 44'245'008'384ULL,
          "unexpected available memory for a 64 GB snapshot");
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  const auto none = [](const HostMemory &host) { return !host.available && !host.spare; };
  require(none(estimateHostMemory({.free = maximum, .fileBacked = 1}, 1, false)) &&
              none(estimateHostMemory({.fileBacked = maximum, .purgeable = 1}, 1, false)) &&
              none(estimateHostMemory({.free = maximum}, pageSize, false)) &&
              none(estimateHostMemory({.free = 1, .speculative = 2}, 1, false)) &&
              estimateHostMemory({.free = maximum}, 1, false).available == maximum &&
              none(estimateHostMemory(pages, 0, false)),
          "invalid host counters or arithmetic overflow did not fail closed");
  // With compression (below critical system pressure), compressing the
  // anonymous pages frees what the compressor would not keep: at its
  // present ratio, at most 2:1, and 2:1 while it holds nothing. The spare
  // memory counts half of that credit.
  const auto compressedPages = [](uint64_t compressor, uint64_t compressed) {
    return estimateHostMemory({.free = 10, .fileBacked = 20, .anonymous = 40,
                               .compressor = compressor, .compressed = compressed},
                              1, true);
  };
  const auto both = [](const HostMemory &host, uint64_t available, uint64_t spare) {
    return host.available == available && host.spare == spare;
  };
  require(both(estimateHostMemory({.free = 10, .fileBacked = 20, .anonymous = 40}, 1, false), 30, 30) &&
              both(compressedPages(0, 0), 50, 40) && both(compressedPages(10, 15), 44, 37) &&
              both(compressedPages(10, 40), 50, 40) && both(compressedPages(10, 10), 30, 30) &&
              both(compressedPages(10, 5), 30, 30),
          "compression credit is not the anonymous pages' savings at the compressor's ratio, at most 2:1, "
          "and half of it in the spare memory");
  require(none(estimateHostMemory({.free = maximum, .anonymous = 2}, 1, true)),
          "compression credit overflowed");
  require(EngineMemoryPolicy::hostAvailableReserveBytes(16 * kGiB) ==
                  16 * kGiB / 10 &&
              EngineMemoryPolicy::hostAvailableReserveBytes(48 * kGiB) ==
                  2 * kGiB &&
              EngineMemoryPolicy::hostAvailableReserveBytes(128 * kGiB) ==
                  2 * kGiB,
          "the macOS reserve is a tenth of a small machine, 2 GiB above 20 GiB");
  require(EngineMemoryPolicy::hostSpareReserveBytes(16 * kGiB) == 2 * kGiB &&
              EngineMemoryPolicy::hostSpareReserveBytes(24 * kGiB) == 2 * kGiB &&
              EngineMemoryPolicy::hostSpareReserveBytes(64 * kGiB) == 64 * kGiB / 100 * 7 &&
              EngineMemoryPolicy::hostSpareReserveBytes(128 * kGiB) == 7 * kGiB,
          "the spare reserve is 7% of the Mac's memory, from 2 to 7 GiB");
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
// through the plan's governor whatever part of the reserves this memory
// takes; memory beyond them is still charged, and the device never exceeds
// the hard budget.
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
      // The plan's governor, as a start and the runtime oracle build it.
      MemoryGovernor governor(backend, budget, [] { return hostMemory(200 * kGiB); });
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
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [] { return hostMemory(hostReserve + 3 * kGiB); }, 0);
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

// Host memory the engine holds outside Metal (a streamed table's row cache)
// is charged in full beside the backend's buffers.
void testHostHeldBytesAreCharged() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024, held = 16 * 1024;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve, [] {
    return hostMemory(hostReserve + 3 * kGiB);
  }, 0, held);
  require(static_cast<bool>(admit(governor, limit - held)) && !admit(governor, limit - held + 1),
          "memory governor did not charge the host memory the engine holds");
  require(governor.snapshot().chargedBytes == held, "the host memory the engine holds is not in its charge");
}

// Critical pressure stops all growth. Host headroom below the warning margin
// stops it too, as warning pressure, which sheds cache in paced passes and
// lasts until the headroom crosses the recovery margin; with no request
// waiting for memory, growth that leaves the warning margin free resumes
// before then. Exhausting the engine's limit is not host pressure, and
// missing telemetry fails closed without discarding valid cache.
void testHostHeadroomHysteresis() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + 3 * kGiB;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
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
// reclaimer; reclaimable file cache can reopen it.
void testReclaimablePagesReopenGrowth() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  const uint64_t limit = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + 3 * kGiB;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  const auto admitsMore = [&governor] { return static_cast<bool>(admit(governor, 1)); };
  HostMemoryPages pressurePages{.free = hostReserve, .fileBacked = kGiB / 2};
  available = estimateHostMemory(pressurePages, 1, false).available;
  MemoryPressurePolicy hostPolicy;
  const auto hostDirective = hostPolicy.update(governor.snapshot(), 0.0, false);
  require(!admitsMore() && governor.snapshot().pressure == MemoryPressure::Warning &&
              hostDirective && !hostDirective->critical &&
              hostDirective->targetBytes == kGiB / 2,
          "low reclaimable memory bypassed bounded pressure recovery");
  pressurePages.fileBacked = 3 * kGiB;
  available = estimateHostMemory(pressurePages, 1, false).available;
  require(governor.snapshot().hostGrowthAllowed && admitsMore() &&
              !hostPolicy.update(governor.snapshot(), 1000.0, false),
          "reclaimable host recovery did not reopen normal admission");
}

// Reading the governor moves nothing: /status and the engine's growth
// queries see what it would decide, and only a reservation or the control
// pass moves the host's hysteresis. A recovery that only a read saw leaves
// growth held for a waiting request once the host falls back inside the
// recovery margin.
void testSnapshotLeavesTheHysteresis() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 64 * 1024;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 64 * 1024, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  governor.setRequestWaiting(true);
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

// The host's pressure asks, once per settled measurement and for at most the
// warning margin, for what measured headroom lacks of the warning margin, or
// of the recovery margin while a request waits for memory; critical pressure
// evicts everything.
void testPressurePolicyDirectives() {
  MemoryPressurePolicy policy;
  MemoryGovernorSnapshot policySnapshot;
  policySnapshot.pressure = MemoryPressure::Warning;
  policySnapshot.hostMeasurementValid = true;
  policySnapshot.hostHeadroomBytes = kGiB / 2;
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
  require(policy.update(policySnapshot, 1000.0, false).value().targetBytes == 3 * kGiB / 4,
          "persistent warning did not request a new bounded shrink pass");
  policySnapshot.hostHeadroomBytes = 3 * kGiB / 4;
  require(policy.update(policySnapshot, 2000.0, false).value().targetBytes == kGiB / 4,
          "pressure recovery ignored the current smaller deficit");
  policySnapshot.pressure = MemoryPressure::Normal;
  require(!policy.update(policySnapshot, 2100.0, false),
          "normal pressure requested cache reclaim");
  policySnapshot.pressure = MemoryPressure::Warning;
  require(policy.update(policySnapshot, 2101.0, false).value().targetBytes == kGiB / 4,
          "a new pressure episode inherited an old cooldown");
  // Past the warning margin the hysteresis still holds the pressure. A pass
  // nothing waits for then evicts nothing; one a request waits for frees
  // what the recovery margin lacks.
  policySnapshot.hostHeadroomBytes = 3 * kGiB / 2;
  const auto idle = policy.update(policySnapshot, 3101.0, false);
  require(idle && !idle->critical && idle->targetBytes == 0,
          "a shrink nothing waited for went past the warning margin");
  const auto waited = policy.update(policySnapshot, 4101.0, true);
  require(waited && waited->targetBytes == kGiB / 2,
          "a waiting request's shrink stopped short of the recovery margin");
  // The newest publication is what a follow-up resumes from; rebuilding it
  // costs a whole prefill, so a shrink nothing is waiting for leaves it and
  // takes the rest short of what is in use, which includes the replay point
  // a running request publishes for its follow-up. A waiting request
  // outranks both, and Critical takes all.
  policySnapshot.hostHeadroomBytes = kGiB / 4;
  const auto speculative = policy.update(policySnapshot, 6101.0, false);
  require(speculative && speculative->targetBytes == 3 * kGiB / 4 &&
              speculative->keepResumePoint && speculative->upTo == ReclaimClass::Ordinary,
          "a speculative shrink discarded the resume point or took what is in use");
  const auto demanded = policy.update(policySnapshot, 7101.0, true);
  require(demanded && demanded->targetBytes == kGiB && !demanded->keepResumePoint &&
              demanded->upTo == ReclaimClass::InUse,
          "a waiting request could not reach the resume point or what is in use");
  policySnapshot.pressure = MemoryPressure::Critical;
  const auto criticalDirective = policy.update(policySnapshot, 7102.0, false);
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
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
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
  // The control pass finds the refused request waiting.
  governor.setRequestWaiting(true);
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
  MemoryGovernor governor(backend, 15 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
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
// a new episode of host pressure, brings the hold back; a pass that macOS's
// warning alone asked for, while the host had room, waives nothing.
void testExhaustedReclaimWaivesTheHold() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB + kGiB / 2;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
  governor.setRequestWaiting(true);
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
  available = hostReserve + 3 * kGiB;
  governor.setPressure(MemoryPressure::Warning);
  require(governor.evaluate().pressure == MemoryPressure::Warning, "macOS's warning was ignored");
  governor.reclaimed(ReclaimOutcome::Exhausted);
  governor.setPressure(MemoryPressure::Normal);
  available = hostReserve + kGiB / 2;
  require(governor.evaluate().pressure == MemoryPressure::Warning, "the host did not run short");
  available = hostReserve + kGiB + kGiB / 2;
  require(!admit(governor, 100 * kMiB),
          "a pass that only macOS's warning asked for waived the next episode's hold");
}

// Growth waits for the recovery margin only while a request waits for
// memory, so that what reclaim frees goes to it. With none waiting, paced
// reclaim only restores the warning margin, and growth that clears that
// margin proceeds while the host's headroom sits between the margins.
void testOnlyAWaitingRequestHoldsGrowth() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + kGiB / 2;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
  require(!admit(governor, 100 * kMiB), "growth inside the warning margin was admitted");
  available = hostReserve + kGiB + kGiB / 2;
  require(governor.evaluate().pressure == MemoryPressure::Warning &&
              governor.snapshot().hostGrowthAllowed && admit(governor, 100 * kMiB),
          "growth that cleared the warning margin waited with no request waiting");
  governor.setRequestWaiting(true);
  require(!governor.snapshot().hostGrowthAllowed && !admit(governor, 100 * kMiB),
          "growth did not wait for the recovery margin while a request waited");
  governor.setRequestWaiting(false);
  require(governor.snapshot().hostGrowthAllowed && admit(governor, 100 * kMiB),
          "the hold outlived the waiting request");
}

// A pass with no target leaves nothing to settle: a request that starts
// waiting right after it gets its own pass at once, and that one, which has
// a target, waits for the host's counters before the next.
void testOnlyATargetedPassWaitsToSettle() {
  MemoryPressurePolicy policy;
  const MemoryGovernorSnapshot between{.pressure = MemoryPressure::Warning,
                                       .hostMeasurementValid = true,
                                       .hostHeadroomBytes = kGiB + kGiB / 2};
  const auto idle = policy.update(between, 0.0, false);
  const auto waited = policy.update(between, 10.0, true);
  require(idle && idle->targetBytes == 0 && waited && waited->targetBytes == kGiB / 2,
          "an empty pass made a waiting request wait for its own");
  require(policy.update(between, 20.0, true).value().targetBytes == 0 &&
              policy.update(between, 1010.0, true).value().targetBytes == kGiB / 2,
          "a pass that freed memory did not wait for the host to settle");
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
  MemoryGovernor governor(backend, 15 * kGiB, hostReserve, hostReserve, [&available] { return hostMemory(available); }, 0);
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

// A Mac shared with other applications: their anonymous memory makes the
// available memory, which counts compressing it 2:1, much larger than the
// spare memory, which counts half of that credit. Growth that no request in
// service needs keeps the warning margin above the spare reserve of the spare
// memory, and the paced reclaim returns cache toward it, while memory a
// request in service needs still grows and the governor starts against the
// smaller reserve macOS keeps of the available memory.
void testOptionalGrowthKeepsTheSpareReserve() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 20 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 20 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB, spareReserve = 7 * kGiB;
  // 4 GiB free and file-backed, and 40 GiB of other applications' anonymous
  // memory: 20 GiB more if compressed 2:1, of which spare counts 10.
  HostMemory host{24 * kGiB, 14 * kGiB};
  MemoryGovernor governor(backend, 100 * kGiB, hostReserve, spareReserve, [&host] {
    return std::optional<HostMemory>(host);
  }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  MemoryGovernorSnapshot snapshot = governor.snapshot();
  require(snapshot.hostAvailableBytes == 24 * kGiB && snapshot.hostSpareBytes == 14 * kGiB &&
              snapshot.hostSpareReserveBytes == spareReserve &&
              snapshot.hostHeadroomBytes == 7 * kGiB && snapshot.hostGrowthAllowed,
          "the headroom is not the spare memory above the spare reserve");
  require(static_cast<bool>(grow(5 * kGiB)), "growth within the spare headroom was refused");
  // The growth came out of the host's free memory.
  host = {19 * kGiB, 9 * kGiB};
  require(!grow(kGiB + kGiB / 2),
          "growth no request needs did not keep the warning margin above the spare reserve");
  host = {17 * kGiB + kGiB / 2, 7 * kGiB + kGiB / 2};
  snapshot = governor.evaluate();
  MemoryPressurePolicy policy;
  const auto directive = policy.update(snapshot, 0.0, false);
  require(snapshot.pressure == MemoryPressure::Warning && !snapshot.hostGrowthAllowed &&
              directive && directive->targetBytes == kGiB - snapshot.hostHeadroomBytes,
          "the paced reclaim did not restore the warning margin above the spare reserve");
  governor.setServing(true);
  require(static_cast<bool>(grow(4 * kGiB)),
          "a request in service waited for the spare reserve");
  governor.setServing(false);
  // A start needs only the reserve macOS keeps of the available memory.
  host = {3 * kGiB, 0};
  MemoryGovernor starting(backend, 100 * kGiB, hostReserve, spareReserve, [&host] {
    return std::optional<HostMemory>(host);
  }, 0);
  host = {hostReserve, 0};
  rejects([&] { MemoryGovernor refused(backend, 100 * kGiB, hostReserve, spareReserve, [&host] {
            return std::optional<HostMemory>(host);
          }, 0); },
          "host available memory does not satisfy the system reserve",
          "a start below the reserve macOS keeps was admitted");
}

// macOS's warning asks caches to give memory back. While it lasts, growth
// that no request in service needs stops whatever the host's headroom, and
// the snapshot says so, while what a request in service needs still grows.
// Every paced pass returns the warning margin's worth of cache and keeps the
// resume point, even for a waiting request, which the warning holds back
// too; only a pass a request waits for takes what is in use. Once the level
// is normal again, growth resumes and no pass runs.
void testSystemWarningGivesCacheBack() {
  test::metalStatistics() = {};
  test::metalStatistics().allocatedBytes = 12 * kGiB;
  test::metalStatistics().deviceCurrentAllocatedBytes = 12 * kGiB;
  metal::MetalBackend backend("unused");
  const uint64_t hostReserve = 2 * kGiB;
  const std::optional<uint64_t> available = hostReserve + 16 * kGiB;
  MemoryGovernor governor(backend, 40 * kGiB, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  const auto grow = [&](uint64_t bytes) {
    return admit(governor, bytes, [bytes] { allocate(bytes); });
  };
  governor.setPressure(MemoryPressure::Warning);
  const MemoryGovernorSnapshot warned = governor.evaluate();
  const metal::AllocationResult cache = grow(100 * kMiB);
  require(!cache && cache.failure == metal::AllocationFailure::HostPressure &&
              warned.pressure == MemoryPressure::Warning && !warned.hostGrowthAllowed,
          "growth that no request in service needs went on while macOS warned");
  governor.setServing(true);
  require(static_cast<bool>(grow(100 * kMiB)), "macOS's warning refused a request in service");
  governor.setServing(false);

  MemoryPressurePolicy policy;
  const auto first = policy.update(warned, 0.0, false);
  require(first && !first->critical && first->targetBytes == kHostWarningMarginBytes &&
              first->keepResumePoint && first->upTo == ReclaimClass::Ordinary,
          "a pass under macOS's warning did not return the warning margin's worth of cache, "
          "keeping the resume point and what is in use");
  require(policy.update(warned, 500.0, false).value().targetBytes == 0 &&
              policy.update(warned, 1000.0, false).value().targetBytes == kHostWarningMarginBytes,
          "the passes under macOS's warning did not keep the pace");
  const auto waited = policy.update(warned, 2000.0, true);
  require(waited && waited->targetBytes == kHostWarningMarginBytes && waited->keepResumePoint &&
              waited->upTo == ReclaimClass::InUse,
          "a pass under macOS's warning dropped the resume point for a request it holds back, "
          "or kept what is in use from it");

  governor.setPressure(MemoryPressure::Normal);
  const MemoryGovernorSnapshot normal = governor.evaluate();
  require(normal.pressure == MemoryPressure::Normal && normal.hostGrowthAllowed &&
              grow(100 * kMiB) && !policy.update(normal, 3000.0, false),
          "growth or the passes did not follow macOS back to normal pressure");
}

} // namespace

int main() {
  try {
    testHostAvailabilityCountsReclaimablePages();
    testAdvertisedContextIsGrantable();
    testReservationsAndAdmissionClasses();
    testHostHeldBytesAreCharged();
    testHostHeadroomHysteresis();
    testReclaimablePagesReopenGrowth();
    testSnapshotLeavesTheHysteresis();
    testPressurePolicyDirectives();
    testHostRefusalStartsReclaim();
    testHostRefusalComesBeforeTheEngineLimit();
    testPolicyContinuesHeldBackTarget();
    testExhaustedReclaimWaivesTheHold();
    testOnlyAWaitingRequestHoldsGrowth();
    testOnlyATargetedPassWaitsToSettle();
    testRequestInServiceGrowsThroughHostPressure();
    testOptionalGrowthKeepsTheSpareReserve();
    testSystemWarningGivesCacheBack();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
