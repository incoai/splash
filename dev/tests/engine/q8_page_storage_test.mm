#include "ops/Q8PageStorage.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void run(const std::string &metallib) {
    constexpr uint64_t pageSize = 16384;
    auto availablePages = [](const HostMemoryPages &pages) {
        return estimateHostAvailableMemory(pages, pageSize, 100 * pageSize) /
               pageSize;
    };
    HostMemoryPages pages{.active = 50, .inactive = 20, .speculative = 5,
                          .wired = 10, .compressor = 5,
                          .fileBacked = 35, .purgeable = 5};
    require(availablePages(pages) == 50,
            "host availability does not match macOS reclaimable accounting");
    pages.active += pages.inactive;
    pages.inactive = 0;
    require(availablePages(pages) == 50,
            "active/inactive transitions changed reclaimable capacity");
    pages.active -= 5;
    pages.speculative += 5;
    require(availablePages(pages) == 50,
            "speculative file pages were counted twice");
    pages.inactive += 10;
    require(availablePages(pages) == 40,
            "inactive anonymous allocations did not consume capacity");
    pages.purgeable = 0;
    require(availablePages(pages) == 35,
            "non-purgeable backing received reclaimable credit");
    pages.compressor += 5;
    require(availablePages(pages) == 30,
            "compressor physical memory was not charged");
    // Wired file pages leave external_page_count: GPU pinning must reduce
    // available memory, rather than crediting hot weights for KV growth.
    pages.active -= 10;
    pages.fileBacked -= 10;
    pages.wired += 10;
    require(availablePages(pages) == 20,
            "wired weights remained available for new allocations");

    // Reading a file into clean cache does not require a second full copy
    // when that same immutable file is mapped again on the next startup.
    HostMemoryPages uncached{.active = 10, .wired = 10, .compressor = 5};
    HostMemoryPages cached = uncached;
    cached.active += 40;
    cached.fileBacked = 40;
    require(availablePages(uncached) == 75 && availablePages(cached) == 75,
            "cached weights reduced model reload capacity");
    require(estimateHostAvailableMemory(
                {.active = 1'298'324, .inactive = 1'281'896,
                 .speculative = 53'613, .wired = 259'122,
                 .compressor = 182'452, .fileBacked = 1'745'483,
                 .purgeable = 22'868}, pageSize, 48ULL << 30) == 30'124'802'048ULL,
            "unexpected available memory for warm-restart snapshot");
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    require(estimateHostAvailableMemory(
                {.active = maximum, .inactive = 1}, 1, maximum) == 0 &&
                estimateHostAvailableMemory(
                    {.active = maximum}, pageSize, maximum) == 0 &&
                estimateHostAvailableMemory({.fileBacked = 1}, 1, maximum) == 0 &&
                estimateHostAvailableMemory({.purgeable = 1}, 1, maximum) == 0 &&
                estimateHostAvailableMemory(
                    {.active = maximum}, 1, maximum - 1) == 0 &&
                estimateHostAvailableMemory({}, 1, maximum) == maximum &&
                estimateHostAvailableMemory(pages, 0, maximum) == 0 &&
                estimateHostAvailableMemory(pages, pageSize, 0) == 0,
            "invalid host counters or arithmetic overflow did not fail closed");
    require(EngineMemoryPolicy::hostAvailableReserveBytes(16 * (1ULL << 30)) ==
                16 * (1ULL << 30) / 10 &&
            EngineMemoryPolicy::hostAvailableReserveBytes(48 * (1ULL << 30)) ==
                2 * (1ULL << 30) &&
            EngineMemoryPolicy::hostAvailableReserveBytes(128 * (1ULL << 30)) ==
                2 * (1ULL << 30),
            "the macOS reserve is a tenth of a small machine, 2 GiB above 20 GiB");
    constexpr kv::Q8Layout q8Layout{16, 4, 256};
    constexpr kv::Q8Layout compactLayout{10, 2, 256};
    // 64 KiB sparse tiles: the 512-byte-per-page scale buffers force
    // 128-page mapping batches for four KV heads and 256 for two.
    static_assert(kv::kSparseMappingAlignmentBytes == 64 * 1024);
    static_assert(q8Layout.bytesPerModelPage() == 1'064'960);
    static_assert(q8Layout.sparseMappingBatchPages() == 128);
    static_assert(q8Layout.backingExtentPages() == 128);
    static_assert(compactLayout.bytesPerModelPage() == 332'800);
    static_assert(compactLayout.sparseMappingBatchPages() == 256);
    static_assert(compactLayout.backingExtentPages() == 512);
    metal::MetalBackend backend(metallib);
    auto baseline = backend.memoryStats();
    uint64_t observed = std::max(
        baseline.allocatedBytes, baseline.deviceCurrentAllocatedBytes);
    constexpr uint64_t giB = 1ULL << 30;
    constexpr uint64_t hostReserve = 64 * 1024;
    std::optional<uint64_t> fakeHostAvailable = hostReserve + 3 * giB;
    MemoryGovernor bounded(
        backend, observed + 64 * 1024, hostReserve,
        [&fakeHostAvailable] {
            return fakeHostAvailable;
        });
    auto reservation = bounded.tryReserve(64 * 1024);
    require(reservation.has_value() &&
                bounded.snapshot().reservedBytes == 64 * 1024,
            "memory governor did not reserve exact growth bytes");
    metal::AllocationFailure failure;
    require(!bounded.tryReserve(1, &failure).has_value() &&
                failure == metal::AllocationFailure::EngineBudget &&
                bounded.snapshot().deniedReservations == 1,
            "memory governor oversold its hard ceiling");
    reservation->commit();
    auto admit = bounded.allocationAdmission();
    const auto driverDenied = admit(1024, [] {
      throw metal::MetalAllocationError("injected driver allocation denial");
    });
    require(!driverDenied &&
                driverDenied.failure == metal::AllocationFailure::DriverRejected &&
                bounded.snapshot().reservedBytes == 0,
            "driver allocation denial did not release its reservation");
    bool defectPropagated = false;
    try {
      (void)admit(1024, [] {
        throw metal::MetalBackendError("injected backend defect");
      });
    } catch (const metal::MetalBackendError &) {
      defectPropagated = true;
    }
    require(defectPropagated && bounded.snapshot().reservedBytes == 0,
            "admission swallowed a backend defect or leaked its reservation");
    require(admit(1024, [] {}) && bounded.snapshot().reservedBytes == 0,
            "driver allocation denial poisoned later admission");
    // A request may cross the reserve while the idle pressure snapshot is
    // still Normal. Its exact refusal reason must remain retryable.
    fakeHostAvailable = hostReserve + giB + 512;
    bool allocated = false;
    const auto hostDenied = admit(1024, [&] { allocated = true; });
    require(!hostDenied && !allocated &&
                hostDenied.failure == metal::AllocationFailure::HostPressure &&
                bounded.snapshot().pressure == MemoryPressure::Normal &&
                bounded.snapshot().reservedBytes == 0,
            "request-sized host refusal lost its cause or ran allocation");
    fakeHostAvailable = hostReserve + 3 * giB;
    bounded.setPressure(MemoryPressure::Critical);
    require(!bounded.tryReserve(1).has_value(),
            "critical pressure did not stop new growth");
    bounded.setPressure(MemoryPressure::Normal);
    require(bounded.tryReserve(1).has_value(),
            "normal pressure did not reopen physical admission");
    fakeHostAvailable = hostReserve;
    require(!bounded.tryReserve(1).has_value(),
            "host reserve did not stop unified-memory growth");
    // Reaching the reserve is a warning that sheds cache in paced passes;
    // only the OS critical verdict drops every evictable entry.
    require(bounded.snapshot().pressure == MemoryPressure::Warning,
            "reaching the host reserve was treated as critical");
    fakeHostAvailable = hostReserve / 2;
    require(bounded.snapshot().pressure == MemoryPressure::Warning &&
                !bounded.tryReserve(1).has_value(),
            "low availability escalated to destructive system pressure");
    fakeHostAvailable = hostReserve + giB / 2;
    require(bounded.snapshot().pressure == MemoryPressure::Warning &&
                !bounded.snapshot().growthAllowed &&
                !bounded.tryReserve(1).has_value(),
            "low host headroom did not request proactive cache reclaim");
    fakeHostAvailable = hostReserve + 3 * giB / 2;
    require(bounded.snapshot().pressure == MemoryPressure::Warning,
            "warning pressure recovered without crossing the hysteresis");
    fakeHostAvailable = hostReserve + 3 * giB;
    require(bounded.snapshot().pressure == MemoryPressure::Normal &&
                bounded.snapshot().growthAllowed,
            "host recovery did not reopen physical admission");
    {
        auto reservation = bounded.tryReserve(64 * 1024);
        require(reservation.has_value(), "engine capacity reservation failed");
        const auto full = bounded.snapshot();
        require(!full.growthAllowed && full.hostGrowthAllowed,
                "engine budget exhaustion was confused with host pressure");
    }
    fakeHostAvailable.reset();
    require(!bounded.tryReserve(1).has_value() &&
                !bounded.snapshot().hostMeasurementValid,
            "missing host memory telemetry did not fail closed");
    MemoryPressurePolicy missingPolicy;
    auto missing = bounded.snapshot();
    auto missingDirective = missingPolicy.update(missing, 0.0, false);
    require(missing.pressure == MemoryPressure::Warning &&
                !missingDirective.evictAllUnpinnedPrefixes &&
                missingDirective.targetBytes == 0,
            "missing telemetry discarded valid cache");
    bounded.setPressure(MemoryPressure::Critical);
    require(missingPolicy.update(bounded.snapshot(), 1.0, false).evictAllUnpinnedPrefixes,
            "missing telemetry hid critical system pressure");
    bounded.setPressure(MemoryPressure::Normal);
    fakeHostAvailable = hostReserve + 3 * giB;
    require(bounded.snapshot().hostHeadroomBytes == 3 * giB,
            "host memory headroom accounting is wrong");

    // Lots of inactive anonymous memory must not reopen physical growth or
    // suppress the existing pressure reclaimer. Reclaimable file cache can.
    HostMemoryPages pressurePages{
        .inactive = 24 * giB, .fileBacked = giB / 2};
    fakeHostAvailable = estimateHostAvailableMemory(
        pressurePages, 1, hostReserve + 24 * giB);
    MemoryPressurePolicy hostPolicy;
    auto hostDirective = hostPolicy.update(bounded.snapshot(), 0.0, false);
    require(!bounded.tryReserve(1).has_value() &&
                bounded.snapshot().pressure == MemoryPressure::Warning &&
                hostDirective.reclaimEmptyKvExtents &&
                !hostDirective.evictAllUnpinnedPrefixes &&
                hostDirective.targetBytes == giB,
            "inactive anonymous credit bypassed bounded pressure recovery");
    pressurePages.fileBacked = 3 * giB;
    fakeHostAvailable = estimateHostAvailableMemory(
        pressurePages, 1, hostReserve + 24 * giB);
    require(bounded.snapshot().growthAllowed &&
                bounded.tryReserve(1).has_value() &&
                !hostPolicy.update(bounded.snapshot(), 1000.0, false).reclaimEmptyKvExtents,
            "reclaimable host recovery did not reopen normal admission");
    bounded.setPressure(MemoryPressure::Warning);
    require(bounded.snapshot().pressure == MemoryPressure::Warning &&
                bounded.snapshot().growthAllowed &&
                bounded.tryReserve(1).has_value(),
            "system warning blocked growth despite sufficient host headroom");
    fakeHostAvailable = hostReserve + giB / 2;
    require(!bounded.tryReserve(1).has_value(),
            "system warning bypassed insufficient host headroom");
    fakeHostAvailable = hostReserve + 3 * giB;
    bounded.setPressure(MemoryPressure::Normal);

    MemoryPressurePolicy policy;
    MemoryGovernorSnapshot policySnapshot;
    policySnapshot.pressure = MemoryPressure::Warning;
    policySnapshot.hostMeasurementValid = true;
    policySnapshot.systemPressure = MemoryPressure::Warning;
    policySnapshot.hostHeadroomBytes = 3 * giB / 2;
    auto firstDirective = policy.update(policySnapshot, 0.0, false);
    require(firstDirective.reclaimEmptyKvExtents &&
                !firstDirective.evictAllUnpinnedPrefixes &&
                firstDirective.targetBytes == giB / 2,
            "warning pressure ignored measured headroom");
    require(policy.update(policySnapshot, 500.0, false).targetBytes == 0 &&
                policy.update(policySnapshot, 999.0, false).targetBytes == 0,
            "warning pressure reclaimed again before telemetry settled");
    // Another application consumed more memory in the SAME warning episode.
    // Earlier reclaimed bytes must not offset this new deficit.
    policySnapshot.hostHeadroomBytes = giB / 4;
    require(policy.update(policySnapshot, 1000.0, false).targetBytes == giB,
            "persistent warning did not request a new bounded shrink pass");
    policySnapshot.systemPressure = MemoryPressure::Normal;
    policySnapshot.hostHeadroomBytes = 7 * giB / 4;
    require(policy.update(policySnapshot, 2000.0, false).targetBytes == giB / 4,
            "pressure recovery ignored the current smaller deficit");
    policySnapshot.pressure = MemoryPressure::Normal;
    require(!policy.update(policySnapshot, 2100.0, false).reclaimEmptyKvExtents,
            "normal pressure requested cache reclaim");
    policySnapshot.pressure = MemoryPressure::Warning;
    require(policy.update(policySnapshot, 2101.0, false).targetBytes == giB / 4,
            "a new pressure episode inherited an old cooldown");
    policySnapshot.systemPressure = MemoryPressure::Warning;
    policySnapshot.hostHeadroomBytes = 3 * giB;
    const auto advisory = policy.update(policySnapshot, 3101.0, false);
    require(advisory.reclaimEmptyKvExtents && !advisory.evictAllUnpinnedPrefixes &&
                advisory.targetBytes == 0,
            "system warning discarded live cache despite sufficient headroom");
    // The newest publication is what a follow-up resumes from; rebuilding it
    // costs a whole prefill, so a shrink nothing is waiting for leaves it and
    // takes the rest. A waiting request outranks it, and Critical takes all.
    policySnapshot.systemPressure = MemoryPressure::Warning;
    policySnapshot.hostHeadroomBytes = giB / 4;
    const auto speculative = policy.update(policySnapshot, 4101.0, false);
    require(speculative.targetBytes == giB && speculative.keepResumePoint,
            "a speculative shrink discarded the resume point");
    const auto demanded = policy.update(policySnapshot, 5101.0, true);
    require(demanded.targetBytes == giB && !demanded.keepResumePoint,
            "a waiting request could not reach the resume point");
    policySnapshot.pressure = MemoryPressure::Critical;
    auto criticalDirective = policy.update(policySnapshot, 2102.0, false);
    require(criticalDirective.reclaimEmptyKvExtents &&
                criticalDirective.evictAllUnpinnedPrefixes &&
                !criticalDirective.keepResumePoint,
            "critical pressure did not request aggressive reclaim");

    std::optional<uint64_t> elasticHostAvailable = 2ULL * 1024 * 1024 * 1024;
    MemoryGovernor hostGated(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes,
        128ULL * 1024 * 1024,
        [&elasticHostAvailable] { return elasticHostAvailable; });
    kv::Q8PageStorage hostGatedStorage(
        backend, hostGated.allocationAdmission(), q8Layout, 256);
    if (hostGatedStorage.residentPages() != 128) {
        throw std::runtime_error(
            "elastic Q8 storage started with " +
            std::to_string(hostGatedStorage.residentPages()) +
            " resident blocks instead of 128");
    }
    elasticHostAvailable = 128ULL * 1024 * 1024;
    require(!hostGatedStorage.ensureResident(128) &&
                hostGatedStorage.residentPages() == 128,
            "host pressure did not reject the next KV extent transactionally");
    elasticHostAvailable = 4ULL * 1024 * 1024 * 1024;
    require(hostGatedStorage.ensureResident(128) &&
                hostGatedStorage.residentPages() == 256,
            "KV growth did not recover after host memory became available");

    MemoryGovernor governor(
        backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
    kv::Q8PageStorage storage(backend, governor.allocationAdmission(),
                              q8Layout,
                              q8Layout.sparseMappingBatchPages());
    require(storage.declaredBytes() ==
                q8Layout.sparseMappingBatchPages() *
                    q8Layout.bytesPerModelPage(),
            "declared Q8 pool bytes are wrong");
    require(storage.actualAllocatedBytes() == storage.declaredBytes(),
            "granularity-aligned Q8 pool has hidden Metal rounding");
    require(storage.residentPages() == q8Layout.sparseMappingBatchPages() &&
                storage.isResident(0),
            "initial Q8 runway residency is incorrect");
    const auto residentBefore = backend.memoryStats().sparseResidentBytes;
    require(storage.releaseReady(), "idle storage reported a pending release");
    require(storage.releaseBackingForPage(0) &&
                storage.actualAllocatedBytes() == 0 &&
                storage.residentPages() == 0 && !storage.isResident(3),
            "empty Q8 extent did not return its physical heap");
    // The heap is released asynchronously; the backing reports readiness
    // only after the sparse queue has completed the unmap.
    storage.awaitRelease();
    require(storage.releaseReady() &&
                backend.memoryStats().sparseResidentBytes + storage.declaredBytes() ==
                    residentBefore,
            "completed Q8 release did not free its heap");
    require(storage.ensureResident(3) && storage.isResident(0) &&
                storage.actualAllocatedBytes() == storage.declaredBytes(),
            "Q8 extent could not be remapped on demand");
    // Remap immediately behind a release without waiting: the queue orders
    // the new mapping after the unmap of the same range.
    require(storage.releaseBackingForPage(0) && storage.ensureResident(0) &&
                storage.actualAllocatedBytes() == storage.declaredBytes(),
            "Q8 extent could not be remapped behind an in-flight release");
    storage.awaitRelease();
    require(storage.releaseReady(), "drained Q8 storage still reports a pending release");

    kv::Q8PageStorage compactStorage(
        backend, governor.allocationAdmission(), compactLayout,
        compactLayout.sparseMappingBatchPages());
    require(static_cast<bool>(compactStorage.layer(9).keyData) &&
                compactStorage.declaredBytes() ==
                    uint64_t{compactLayout.sparseMappingBatchPages()} *
                        compactLayout.bytesPerModelPage() &&
                compactStorage.residentPages() ==
                    compactLayout.sparseMappingBatchPages(),
            "model-provided compact Q8 geometry was not honored");

    std::cout << "q8 page storage tests passed\n";
}

}  // namespace

int main(int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "usage: q8_page_storage_test METALLIB\n";
        return EXIT_FAILURE;
    }
    try {
        run(argv[1]);
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "q8 page storage test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
