#import "MetalBackend.hpp"
#include "CommandWatchdog.hpp"
#include "DeviceQueries.hpp"
#include "MetalEvent.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CommonCrypto/CommonDigest.h>
#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <unistd.h>

namespace splash::metal {
namespace {

// The accelerator entry that backs a Metal device publishes gpu-core-count.
// The device's registry ID names that entry or a child of it; the first
// IOAccelerator service is the fallback, since Apple silicon Macs have one
// GPU. Zero means the property was not found anywhere.
uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
    uint32_t count = 0;
    const auto read = [&](io_registry_entry_t entry) {
        if (!entry) return false;
        CFTypeRef value = IORegistryEntryCreateCFProperty(
            entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int64_t number = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value),
                                 kCFNumberSInt64Type, &number) &&
                number > 0 && number <= 4096) {
                count = static_cast<uint32_t>(number);
            }
            CFRelease(value);
        }
        return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryId));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
        io_registry_entry_t parent = MACH_PORT_NULL;
        if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
            KERN_SUCCESS) {
            parent = MACH_PORT_NULL;
        }
        IOObjectRelease(entry);
        entry = parent;
    }
    if (entry) IOObjectRelease(entry);
    if (!count) {
        io_registry_entry_t accelerator = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
        if (accelerator) {
            read(accelerator);
            IOObjectRelease(accelerator);
        }
    }
    return count;
}

std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}

std::string errorDescription(NSError *error) {
    if (!error) return "unknown Metal error";
    std::string result = stringFromNSString(error.localizedDescription);
    return result.empty() ? "unknown Metal error" : result;
}

NSUInteger checkedNSUInteger(uint64_t value, std::string_view field) {
    if (value > std::numeric_limits<NSUInteger>::max()) {
        throw MetalBackendError(std::string(field) + " exceeds NSUInteger");
    }
    return static_cast<NSUInteger>(value);
}

MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(checkedNSUInteger(size.x, field),
                       checkedNSUInteger(size.y, field),
                       checkedNSUInteger(size.z, field));
}

bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

constexpr uint64_t kPlacementSparsePageBytes = MetalBackend::kPlacementSparsePageBytes;
constexpr MTLSparsePageSize kPlacementSparsePageSize = MTLSparsePageSize64;
constexpr NSUInteger kSparseUnmapTimeoutMilliseconds = 30000;
constexpr NSUInteger kSparseMapTimeoutMilliseconds = 30000;

MTLSparsePageSize metalSparsePageSize(uint64_t bytes) {
    if (bytes != kPlacementSparsePageBytes) {
        throw MetalBackendError(
            "placement-sparse page size must be exactly 64 KiB");
    }
    return kPlacementSparsePageSize;
}

double steadySeconds() noexcept {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char *commandStatusName(MTLCommandBufferStatus status) noexcept {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return "not_enqueued";
    case MTLCommandBufferStatusEnqueued: return "enqueued";
    case MTLCommandBufferStatusCommitted: return "committed";
    case MTLCommandBufferStatusScheduled: return "scheduled";
    case MTLCommandBufferStatusCompleted: return "completed";
    case MTLCommandBufferStatusError: return "error";
    }
    return "unknown";
}

[[maybe_unused]] MachSteadyClockBridge sampleMachSteadyClockBridge() noexcept {
    // Lazy initialization is reached only by enabled command profiling. Normal
    // serving performs no extra Mach-clock or timebase queries.
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t value{};
        if (mach_timebase_info(&value) != KERN_SUCCESS)
            return mach_timebase_info_data_t{};
        return value;
    }();
    MachSteadyClockBridge bridge;
    bridge.timebaseNumer = timebase.numer;
    bridge.timebaseDenom = timebase.denom;
    bridge.beganSteadySeconds = steadySeconds();
    bridge.machAbsoluteTimestamp = mach_absolute_time();
    bridge.endedSteadySeconds = steadySeconds();
    if (!timebase.numer || !timebase.denom || !bridge.machAbsoluteTimestamp ||
        !std::isfinite(bridge.beganSteadySeconds) ||
        !std::isfinite(bridge.endedSteadySeconds) ||
        bridge.endedSteadySeconds < bridge.beganSteadySeconds)
        return bridge;
    bridge.machSeconds = static_cast<double>(
        static_cast<long double>(bridge.machAbsoluteTimestamp) * timebase.numer /
        timebase.denom / 1.0e9L);
    bridge.uncertaintySeconds =
        (bridge.endedSteadySeconds - bridge.beganSteadySeconds) * 0.5;
    bridge.steadyMinusMachSeconds =
        bridge.beganSteadySeconds + bridge.uncertaintySeconds - bridge.machSeconds;
    bridge.valid = std::isfinite(bridge.machSeconds) &&
        std::isfinite(bridge.steadyMinusMachSeconds);
    return bridge;
}

template <typename T>
void raisePeak(std::atomic<T> &peak, T value) noexcept {
    T current = peak.load(std::memory_order_relaxed);
    while (value > current &&
           !peak.compare_exchange_weak(current, value,
                                       std::memory_order_relaxed)) {}
}

NSString *checkedNSString(std::string_view value, std::string_view field) {
    NSString *result = [[NSString alloc]
        initWithBytes:value.data()
        length:value.size()
        encoding:NSUTF8StringEncoding];
    if (!result) {
        throw MetalBackendError(std::string(field) + " is not UTF-8");
    }
    return result;
}

}  // namespace

struct AllocationAccounting {
    std::atomic<uint64_t> allocatedBytes{0};
    std::atomic<uint64_t> peakAllocatedBytes{0};
    std::atomic<uint64_t> sparseVirtualBytes{0};
    std::atomic<uint64_t> sparseResidentBytes{0};
    std::atomic<uint64_t> peakSparseResidentBytes{0};
    std::atomic<uint64_t> residentBytes{0};
    std::atomic<uint64_t> peakResidentBytes{0};

    void addResident(uint64_t bytes) noexcept {
        raisePeak(peakResidentBytes,
                  residentBytes.fetch_add(bytes, std::memory_order_relaxed) +
                      bytes);
    }
};

struct MetalAllocation {
    // Own the host mapping for our views as well as the Metal deallocator.
    // Validation wrappers may not retain the supplied deallocator block.
    std::shared_ptr<void> externalOwner;
    __strong id<MTLBuffer> buffer = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    uint64_t bytes = 0;
    uint64_t sparseVirtualBytes = 0;
    bool placementSparse = false;
    BufferStorage storage = BufferStorage::Shared;
    // Pointer argument buffers own source allocations independently of their
    // caller's views. No source bytes or accounting charges are duplicated.
    std::vector<std::shared_ptr<MetalAllocation>> readOnlyIndirectAllocations;

    ~MetalAllocation() {
        if (accounting && bytes) {
            accounting->allocatedBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
            accounting->residentBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        }
        if (accounting && sparseVirtualBytes) {
            accounting->sparseVirtualBytes.fetch_sub(
                sparseVirtualBytes, std::memory_order_relaxed);
        }
    }
};

struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct SparseHeap::Impl {
    __strong id<MTLHeap> heap = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    uint64_t bytes = 0;

    ~Impl() {
        if (accounting && bytes) {
            accounting->sparseResidentBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
            accounting->residentBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        }
    }
};

namespace {

struct WeightResidencyRegistration {
    std::vector<std::shared_ptr<MetalAllocation>> allocations;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLResidencySet> set = nil;
    uint64_t bytes = 0;
    bool attached = false;
    bool requested = false;

    // Called only at quiescent shutdown, or while unwinding startup before any
    // command can be submitted. Keeping the queue and C++ allocations alive is
    // independent of Metal's undocumented allocation-retention behavior.
    void finish() noexcept {
        if (requested) {
            requested = false;
            @try {
                [set endResidency];
            } @catch (NSException *) {
                // Destruction cannot throw. The set/queue still own their normal
                // Metal lifetimes, and there is no outstanding GPU work here.
            }
        }
        if (attached) {
            attached = false;
            @try {
                [queue removeResidencySet:set];
            } @catch (NSException *) {
            }
        }
    }

    ~WeightResidencyRegistration() { finish(); }
};

// Counter storage is vendor-private. Small chunks avoid assuming that the
// resolved timestamp's eight bytes describe the backing allocation's size.
// Buffer creation remains the authoritative capacity check. The SDK limits a
// process to 32 sample buffers; profiling never exceeds that count per command.
constexpr size_t kProfileDispatchesPerBuffer = 256;
constexpr size_t kMaxProfileCounterBuffers = 32;
constexpr size_t kMaxRetainedCommandProfiles = 8;

struct CounterProfileChunk {
    __strong id<MTLCounterSampleBuffer> buffer = nil;
    size_t firstDispatch = 0;
    NSUInteger sampleCount = 0;
};

struct AtomicDeviceMemorySampleTiming {
    std::atomic<double> began{0.0};
    std::atomic<double> ended{0.0};

    [[nodiscard]] DeviceMemorySampleTiming snapshot() const noexcept {
        DeviceMemorySampleTiming timing;
        // Publishing the end also publishes the earlier begin. Read the end
        // first so a just-started read remains explicitly incomplete.
        timing.endedSteadySeconds = ended.load(std::memory_order_acquire);
        timing.beganSteadySeconds = began.load(std::memory_order_acquire);
        timing.valid = timing.beganSteadySeconds > 0.0 &&
            timing.endedSteadySeconds >= timing.beganSteadySeconds;
        if (timing.valid)
            timing.seconds = timing.endedSteadySeconds - timing.beganSteadySeconds;
        return timing;
    }
};

// Small retained command record, independent of optional dispatch profiling.
// Atomic end-before-begin publication follows the existing memory span rules.
struct CommandHostTimingRecording {
    double submissionStart = 0.0, encodingStart = 0.0, encodingEnd = 0.0;
    std::atomic<double> commitBegin{0.0}, commitEnd{0.0}, scheduled{0.0},
        completed{0.0}, wallEnd{0.0}, submissionReturn{0.0}, dependencyWait{0.0};
    AtomicDeviceMemorySampleTiming preCommitMemory, postCommitMemory,
        scheduledMemory, completedMemory;

    [[nodiscard]] CommandHostTiming snapshot() const noexcept {
        CommandHostTiming result;
        if (!(submissionStart > 0.0 && encodingStart >= submissionStart && encodingEnd >= encodingStart))
            return result;
        result.timedCommands = 1;
        result.preparationSeconds = encodingStart - submissionStart;
        result.encodingSeconds = encodingEnd - encodingStart;
        const double end = commitEnd.load(std::memory_order_acquire);
        const double begin = commitBegin.load(std::memory_order_acquire);
        if (begin >= encodingEnd && end >= begin) {
            result.commitSamples = 1;
            result.beforeCommitSeconds = begin - encodingEnd;
            result.commitSeconds = end - begin;
        }
        const double scheduledAt = scheduled.load(std::memory_order_acquire);
        if (begin > 0.0 && scheduledAt >= begin) {
            result.scheduledCallbackSamples = 1;
            result.commitToScheduledCallbackSeconds = scheduledAt - begin;
        }
        const double completedAt = completed.load(std::memory_order_acquire);
        if (begin > 0.0 && completedAt >= begin) {
            result.completedCallbackSamples = 1;
            result.commitToCompletedCallbackSeconds = completedAt - begin;
        }
        const double wallEndedAt = wallEnd.load(std::memory_order_acquire);
        if (completedAt > 0.0 && wallEndedAt >= completedAt)
            result.completionCallbackBeforeWallEndSeconds = wallEndedAt - completedAt;
        const double returnedAt = submissionReturn.load(std::memory_order_acquire);
        if (returnedAt >= submissionStart) result.submissionReturnSeconds = returnedAt - submissionStart;
        result.dependencyWaitSeconds = dependencyWait.load(std::memory_order_acquire);
        const auto copyMemory = [](const AtomicDeviceMemorySampleTiming &record,
                                   uint64_t &samples, double &seconds) {
            const auto span = record.snapshot();
            if (span.valid) { samples = 1; seconds = span.seconds; }
        };
        copyMemory(preCommitMemory, result.preCommitMemorySamples, result.preCommitMemorySampleSeconds);
        copyMemory(postCommitMemory, result.postCommitMemorySamples, result.postCommitMemorySampleSeconds);
        copyMemory(scheduledMemory, result.scheduledMemorySamples, result.scheduledMemorySampleSeconds);
        copyMemory(completedMemory, result.completedMemorySamples, result.completedMemorySampleSeconds);
        return result;
    }
};

struct CounterProfileRecording {
    mutable std::mutex mutex;
    CommandDispatchProfile profile;
    std::vector<CounterProfileChunk> chunks;
    std::atomic<double> scheduledSeconds{0.0};
    std::atomic<double> commitBeginSeconds{0.0};
    std::atomic<double> commitEndSeconds{0.0};
    AtomicDeviceMemorySampleTiming preCommitMemorySample;
    AtomicDeviceMemorySampleTiming postCommitMemorySample;
    AtomicDeviceMemorySampleTiming scheduledMemorySample;
    AtomicDeviceMemorySampleTiming completedMemorySample;
    MachSteadyClockBridge commitBridge;
    MachSteadyClockBridge completedBridge;

    [[nodiscard]] bool sampling() const noexcept {
        return profile.status == CommandDispatchProfileStatus::Pending &&
               !chunks.empty();
    }

    CommandDispatchProfile snapshot(CommandTiming timing,
                                    const std::string &failure) {
        std::lock_guard lock(mutex);
        profile.timing = timing;
        profile.hostScheduledSeconds =
            scheduledSeconds.load(std::memory_order_acquire);
        profile.hostCommitBeginSeconds =
            commitBeginSeconds.load(std::memory_order_acquire);
        profile.hostCommitEndSeconds =
            commitEndSeconds.load(std::memory_order_acquire);
        profile.hostCommitTimingValid = profile.hostCommitBeginSeconds > 0.0 &&
            profile.hostCommitEndSeconds >= profile.hostCommitBeginSeconds;
        if (profile.hostCommitTimingValid)
            profile.hostCommitSeconds = profile.hostCommitEndSeconds -
                profile.hostCommitBeginSeconds;
        profile.hostReadySeconds = steadySeconds();
        profile.preCommitDeviceMemorySample = preCommitMemorySample.snapshot();
        profile.postCommitDeviceMemorySample = postCommitMemorySample.snapshot();
        profile.scheduledDeviceMemorySample = scheduledMemorySample.snapshot();
        profile.completedDeviceMemorySample = completedMemorySample.snapshot();
        profile.commitClockBridge = commitBridge;
        profile.completedClockBridge = completedBridge;
        if (!failure.empty()) {
            profile.status = CommandDispatchProfileStatus::CommandFailed;
            profile.reason = failure;
            for (auto &dispatch : profile.dispatches)
                dispatch.timestampsValid = false;
        }
        // A consumed but still-live ticket must not keep the process's limited
        // sample-buffer slots occupied. The command has completed or was never
        // committed, and resolution is finished before reaching this point.
        chunks.clear();
        return std::move(profile);
    }
};

template <class Sample>
uint64_t sampleDeviceMemoryProfiled(
    const std::shared_ptr<CounterProfileRecording> &record,
    AtomicDeviceMemorySampleTiming CounterProfileRecording::*span,
    Sample &&sample) noexcept {
    if (record)
        (record.get()->*span).began.store(steadySeconds(), std::memory_order_release);
    const uint64_t bytes = sample();
    if (record)
        (record.get()->*span).ended.store(steadySeconds(), std::memory_order_release);
    return bytes;
}

template <class Sample>
uint64_t sampleDeviceMemoryTimed(
    AtomicDeviceMemorySampleTiming &normal,
    const std::shared_ptr<CounterProfileRecording> &profile,
    AtomicDeviceMemorySampleTiming CounterProfileRecording::*span,
    Sample &&sample) noexcept {
    normal.began.store(steadySeconds(), std::memory_order_release);
    const uint64_t bytes = sampleDeviceMemoryProfiled(profile, span, std::forward<Sample>(sample));
    normal.ended.store(steadySeconds(), std::memory_order_release);
    return bytes;
}

// Resolving is CPU work after command completion, never an extra GPU command or
// wait. Invalid/missing samples remain explicitly unavailable and do not poison
// an otherwise successful inference command.
void resolveCounterProfile(const std::shared_ptr<CounterProfileRecording> &record,
                           id<MTLCommandBuffer> command,
                           double completedSeconds) {
    if (!record) return;
    std::lock_guard lock(record->mutex);
    CommandDispatchProfile &profile = record->profile;
    profile.hostCompletedSeconds = completedSeconds;
    profile.commandGpuStartSeconds = command.GPUStartTime;
    profile.commandGpuEndSeconds = command.GPUEndTime;
    profile.commandKernelStartSeconds = command.kernelStartTime;
    profile.commandKernelEndSeconds = command.kernelEndTime;
    profile.commandKernelTimingValid =
        std::isfinite(profile.commandKernelStartSeconds) &&
        std::isfinite(profile.commandKernelEndSeconds) &&
        profile.commandKernelStartSeconds > 0.0 &&
        profile.commandKernelEndSeconds >= profile.commandKernelStartSeconds;
    if (!record->sampling() ||
        command.status != MTLCommandBufferStatusCompleted)
        return;
    [command.device sampleTimestamps:&profile.calibrationCpuEnd
                       gpuTimestamp:&profile.calibrationGpuEnd];
    const auto unavailable = [&](CommandDispatchProfileStatus status,
                                 std::string reason) {
        profile.status = status;
        profile.reason = std::move(reason);
        for (auto &dispatch : profile.dispatches)
            dispatch.timestampsValid = false;
    };
    if (profile.calibrationCpuEnd <= profile.calibrationCpuStart ||
        profile.calibrationGpuEnd <= profile.calibrationGpuStart) {
        unavailable(CommandDispatchProfileStatus::InvalidTimestamps,
                    "CPU/GPU clock calibration did not advance");
        return;
    }
    const long double secondsPerGpuTick =
        static_cast<long double>(profile.calibrationCpuEnd -
                                 profile.calibrationCpuStart) /
        static_cast<long double>(profile.calibrationGpuEnd -
                                 profile.calibrationGpuStart) / 1.0e9L;
    for (const CounterProfileChunk &chunk : record->chunks) {
        NSData *resolved = [chunk.buffer
            resolveCounterRange:NSMakeRange(0, chunk.sampleCount)];
        if (!resolved ||
            resolved.length != chunk.sampleCount * sizeof(MTLCounterResultTimestamp)) {
            unavailable(CommandDispatchProfileStatus::ResolveFailed,
                        "timestamp counter resolve returned an unexpected byte count");
            return;
        }
        for (NSUInteger sample = 0; sample < chunk.sampleCount; sample += 2) {
            std::array<MTLCounterResultTimestamp, 2> pair{};
            std::memcpy(pair.data(),
                        static_cast<const uint8_t *>(resolved.bytes) +
                            sample * sizeof(MTLCounterResultTimestamp),
                        sizeof(pair));
            auto &dispatch = profile.dispatches[chunk.firstDispatch + sample / 2];
            dispatch.gpuStartTimestamp = pair[0].timestamp;
            dispatch.gpuEndTimestamp = pair[1].timestamp;
            if (!pair[0].timestamp || !pair[1].timestamp ||
                pair[0].timestamp == MTLCounterErrorValue ||
                pair[1].timestamp == MTLCounterErrorValue ||
                pair[1].timestamp < pair[0].timestamp ||
                pair[0].timestamp < profile.calibrationGpuStart ||
                pair[1].timestamp > profile.calibrationGpuEnd) {
                unavailable(CommandDispatchProfileStatus::InvalidTimestamps,
                            "missing, unordered or uncalibrated GPU timestamp samples");
                return;
            }
            const auto calibrated = [&](uint64_t timestamp) {
                return static_cast<double>(
                    static_cast<long double>(profile.calibrationCpuStart) / 1.0e9L +
                    static_cast<long double>(timestamp - profile.calibrationGpuStart) *
                        secondsPerGpuTick);
            };
            dispatch.calibratedStartSeconds = calibrated(pair[0].timestamp);
            dispatch.calibratedEndSeconds = calibrated(pair[1].timestamp);
            dispatch.gpuSeconds = static_cast<double>(
                static_cast<long double>(pair[1].timestamp - pair[0].timestamp) *
                secondsPerGpuTick);
            if (!std::isfinite(dispatch.calibratedStartSeconds) ||
                !std::isfinite(dispatch.calibratedEndSeconds) ||
                !std::isfinite(dispatch.gpuSeconds) || dispatch.gpuSeconds < 0.0) {
                unavailable(CommandDispatchProfileStatus::InvalidTimestamps,
                            "GPU clock conversion produced a nonfinite timing");
                return;
            }
        }
    }
    for (auto &dispatch : profile.dispatches)
        dispatch.timestampsValid = true;
    profile.status = CommandDispatchProfileStatus::Complete;
}

[[maybe_unused]] void resolveCounterProfileSafely(
    const std::shared_ptr<CounterProfileRecording> &record,
    id<MTLCommandBuffer> command, double completedSeconds) {
    if (!record) return;
    const auto failed = [&](std::string reason) {
        std::lock_guard lock(record->mutex);
        record->profile.status = CommandDispatchProfileStatus::ResolveFailed;
        record->profile.reason = std::move(reason);
        for (auto &dispatch : record->profile.dispatches)
            dispatch.timestampsValid = false;
    };
    const auto failedWithoutAllocation = [&]() noexcept {
        // Even constructing an error description can fail under host pressure.
        // The enum remains an explicit failure without allocating another string.
        try {
            std::lock_guard lock(record->mutex);
            record->profile.status = CommandDispatchProfileStatus::ResolveFailed;
            record->profile.reason.clear();
            for (auto &dispatch : record->profile.dispatches)
                dispatch.timestampsValid = false;
        } catch (...) {
        }
    };
    @try {
        try {
            resolveCounterProfile(record, command, completedSeconds);
        } catch (const std::exception &error) {
            try {
                failed("timestamp profiling failed: " + std::string(error.what()));
            } catch (...) {
                failedWithoutAllocation();
            }
        } catch (...) {
            try {
                failed("timestamp profiling failed with an unknown CPU exception");
            } catch (...) {
                failedWithoutAllocation();
            }
        }
    } @catch (NSException *exception) {
        try {
            failed("timestamp profiling failed: " +
                   stringFromNSString(exception.reason));
        } catch (...) {
            failedWithoutAllocation();
        }
    }
}

} // namespace

struct ResidencyLease::Impl {
    std::shared_ptr<WeightResidencyRegistration> registration;
};

struct BackendAsyncState {
    __strong id<MTLDevice> device = nil;
    mutable std::atomic<uint64_t> deviceCurrentAllocatedBytes{0};
    mutable std::atomic<uint64_t> devicePeakAllocatedBytes{0};
    std::atomic<bool> healthy{true};
    mutable std::mutex healthMutex;
    std::string healthReason;
    mutable std::mutex gateMutex;
    uint64_t nextSequence = 0;
    uint64_t activeSequence = 0;
    size_t activeDispatchCount = 0;
    __weak id<MTLCommandBuffer> activeCommand = nil;
    std::function<void(id<MTLCommandBuffer>)> activeCompletion;
    CommandWatchdog commandWatchdog;
    std::stop_source stopping;
    std::atomic<uint64_t> mapWaitEvent{0};
    std::atomic<double> mapWaitStarted{0.0};
    std::atomic<double> lastMapWaitSeconds{0.0};
    std::atomic<double> maxMapWaitSeconds{0.0};
    mutable std::mutex profileMutex;
    std::vector<CommandDispatchProfile> commandProfiles;
    uint64_t droppedCommandProfiles = 0;
    // Guarded by gateMutex. Only stop + consumption of the outstanding ticket
    // can detach this startup registration, even if the public lease is gone.
    std::shared_ptr<WeightResidencyRegistration> weightResidency;

    void publishProfile(CommandDispatchProfile profile) noexcept {
        std::lock_guard lock(profileMutex);
        if (commandProfiles.size() == kMaxRetainedCommandProfiles) {
            commandProfiles.erase(commandProfiles.begin());
            ++droppedCommandProfiles;
        }
        profile.droppedProfilesBefore = droppedCommandProfiles;
        try {
            commandProfiles.push_back(std::move(profile));
        } catch (...) {
            // Optional instrumentation must never suppress ticket completion.
            ++droppedCommandProfiles;
        }
    }

    void dropProfile() noexcept {
        std::lock_guard lock(profileMutex);
        ++droppedCommandProfiles;
    }

    uint64_t sampleDeviceMemory() const noexcept {
        if (!device) return 0;
        uint64_t current = static_cast<uint64_t>(device.currentAllocatedSize);
        deviceCurrentAllocatedBytes.store(current, std::memory_order_relaxed);
        raisePeak(devicePeakAllocatedBytes, current);
        return current;
    }

    void ensureHealthy() const {
        if (healthy.load(std::memory_order_acquire)) return;
        std::lock_guard lock(healthMutex);
        throw MetalBackendError("Metal backend is unhealthy: " + healthReason);
    }

    void markUnhealthy(std::string reason) {
        {
            std::lock_guard lock(healthMutex);
            if (healthReason.empty()) healthReason = std::move(reason);
        }
        healthy.store(false, std::memory_order_release);
    }

    uint64_t beginSubmission(size_t dispatchCount) {
        ensureHealthy();
        std::lock_guard lock(gateMutex);
        if (stopping.stop_requested())
            throw MetalBackendError("Metal backend is stopping");
        if (activeSequence) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        if (nextSequence == std::numeric_limits<uint64_t>::max()) {
            throw MetalBackendError("Metal command sequence exhausted");
        }
        activeSequence = ++nextSequence;
        activeDispatchCount = dispatchCount;
        return activeSequence;
    }

    bool commitSubmission(uint64_t sequence, id<MTLCommandBuffer> command,
                          CommandHostTimingRecording &hostTiming,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        if (stopping.stop_requested()) return false;
        activeCommand = command;
        activeCompletion = std::move(completion);
        commandWatchdog.start(sequence, steadySeconds());
        hostTiming.commitBegin.store(steadySeconds(), std::memory_order_release);
        [command commit];
        // Completion's gateMutex rendezvous sees this even for a command that
        // finishes before commit() returns. Instrumentation adds no new wait.
        hostTiming.commitEnd.store(steadySeconds(), std::memory_order_release);
        return true;
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::shared_ptr<WeightResidencyRegistration> retiring;
        {
            std::lock_guard lock(gateMutex);
            commandWatchdog.complete(sequence);
            if (activeSequence == sequence) {
                activeSequence = 0;
                activeCommand = nil;
                activeCompletion = {};
            }
            if (!activeSequence && stopping.stop_requested())
                retiring = std::move(weightResidency);
        }
        if (retiring) retiring->finish();
    }

    void stop() noexcept {
        std::shared_ptr<WeightResidencyRegistration> retiring;
        {
            std::lock_guard lock(gateMutex);
            stopping.request_stop();
            if (!activeSequence) retiring = std::move(weightResidency);
        }
        if (retiring) retiring->finish();
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
    }

    void checkCommandHealth() {
        id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> complete;
        {
            std::lock_guard lock(gateMutex);
            if (commandWatchdog.expired(steadySeconds())) {
                command = activeCommand;
                const auto status = command ? command.status
                                            : MTLCommandBufferStatusNotEnqueued;
                // Recover terminal results even if the driver has not delivered
                // its callback. Finish outside the gate: it takes the ticket lock.
                if (command && (status == MTLCommandBufferStatusCompleted ||
                                status == MTLCommandBufferStatusError)) {
                    complete = activeCompletion;
                } else {
                    std::ostringstream message;
                    message << "Metal command completion timed out after "
                            << commandWatchdog.timeoutSeconds()
                            << " seconds (sequence=" << activeSequence
                            << ", status=" << (command ? commandStatusName(status)
                                                       : "unavailable")
                            << ", dispatches=" << activeDispatchCount << ')';
                    markUnhealthy(message.str());
                }
            }
        }
        if (complete) complete(command);
        ensureHealthy();
    }

    [[nodiscard]] bool hasActiveSubmission() const noexcept {
        std::lock_guard lock(gateMutex);
        return activeSequence != 0;
    }
};

struct CommandTicket::State {
    std::shared_ptr<BackendAsyncState> backend;
    std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    CommandCompletion completion;
    std::shared_ptr<CounterProfileRecording> counterProfile;
    CommandHostTimingRecording hostTiming;
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    std::chrono::steady_clock::time_point wallStart;
    uint64_t sparseEventValue = 0;
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        auto wallEnd = std::chrono::steady_clock::now();
        CommandTiming timing;
        timing.gpuSeconds =
            command.GPUEndTime - command.GPUStartTime;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();

        std::string error;
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::ostringstream message;
            message << "Metal command " << sequence
                    << " failed (sparse event " << sparseEventValue << ')';
            if (command.error) {
                message << ": " << errorDescription(command.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
    }

    void finish(CommandTiming result, std::string failure = {}) {
        CommandCompletion notify;
        {
            std::lock_guard lock(mutex);
            // Host recovery, late callbacks, and discarded commands all share
            // this completion path; only the first result may publish or notify.
            if (completed) return;
            backend->completeSubmission(sequence);
            if (!failure.empty()) backend->markUnhealthy(failure);
            timing = result;
            error = std::move(failure);
            if (counterProfile) {
                try {
                    backend->publishProfile(counterProfile->snapshot(timing, error));
                } catch (...) {
                    backend->dropProfile();
                }
            }
            completed = true;
            notify = completion;
        }
        if (notify) {
            try {
                notify(sequence);
            } catch (...) {
                backend->markUnhealthy(
                    "Metal completion callback threw an exception");
            }
        }
        condition.notify_all();
    }

    void release() noexcept {
        bool shouldRelease = false;
        {
            std::lock_guard lock(mutex);
            if (!released) {
                released = true;
                retainedAllocations.clear();
                shouldRelease = true;
            }
        }
        if (shouldRelease && backend) {
            // Refresh admission telemetry on the consuming thread after GPU
            // completion, before allowing the next submission.
            if (backend->healthy.load(std::memory_order_acquire))
                backend->sampleDeviceMemory();
            backend->releaseSubmission(sequence);
        }
    }

    void abandon() noexcept {
        {
            std::unique_lock lock(mutex);
            condition.wait(lock, [this] { return completed; });
        }
        release();
    }
};

struct MetalBackend::Impl {
    std::function<bool()> cancelled;
    void checkCancellation() const {
        if (cancelled && cancelled())
            throw MetalBackendError("Metal operation cancelled");
    }
    std::function<void()> operationGuard;

    std::atomic<bool> dispatchProfiling{false};
    std::vector<DispatchTiming> dispatchProfile;
    std::atomic<CommandDispatchProfilingMode> commandDispatchProfiling{
        CommandDispatchProfilingMode::Off};
    CommandDispatchProfilingCapability profilingCapability;
    __strong id<MTLCounterSet> timestampCounterSet = nil;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTL4CommandQueue> sparseQueue = nil;
    __strong id<MTLSharedEvent> sparseEvent = nil;
    __strong id<MTLLibrary> library = nil;
    __strong NSMutableDictionary<NSString *, id<MTLComputePipelineState>>
        *pipelines = nil;

    DeviceCapabilities capabilities;
    std::array<uint8_t, 32> metallibSha256{};
    std::shared_ptr<AllocationAccounting> accounting =
        std::make_shared<AllocationAccounting>();
    std::shared_ptr<BackendAsyncState> asyncState =
        std::make_shared<BackendAsyncState>();
    mutable std::mutex commandMutex;
    uint64_t nextSparseEventValue = 0;
    uint64_t pendingSparseEventValue = 0;

    // The one outstanding asynchronous unmap; guarded by commandMutex. Its
    // heap stays alive, and counted resident, until the queue signals.
    struct PendingSparseUnmap {
        uint64_t eventValue = 0;
        SparseHeap heap;
        std::chrono::steady_clock::time_point issued;
    };
    std::optional<PendingSparseUnmap> pendingUnmap;
    std::atomic<uint64_t> pendingUnmapCount{0};
    std::atomic<uint64_t> completedUnmaps{0};
    std::atomic<double> lastUnmapSeconds{0.0};
    std::atomic<double> maxUnmapSeconds{0.0};
    std::atomic<double> pendingUnmapIssuedSeconds{0.0};

    ~Impl() {
        // Teardown must not wait for a stalled mapping queue. Keep its backing
        // alive until the driver acknowledges the pending unmap instead.
        if (pendingUnmap && sparseEvent.signaledValue < pendingUnmap->eventValue) {
            auto retainedHeap = std::make_shared<SparseHeap>(std::move(pendingUnmap->heap));
            id<MTL4CommandQueue> retainedQueue = sparseQueue;
            id<MTLSharedEvent> retainedEvent = sparseEvent;
            [sparseEvent notifyListener:[MTLSharedEventListener sharedListener]
                atValue:pendingUnmap->eventValue block:^(id<MTLSharedEvent>, uint64_t) {
                    (void)retainedHeap;
                    (void)retainedQueue;
                    (void)retainedEvent;
                }];
        }
    }

    // Requires commandMutex. Releases the heap of a completed unmap.
    bool reapSparseUnmapsLocked() noexcept {
        if (!pendingUnmap) return false;
        if (sparseEvent.signaledValue < pendingUnmap->eventValue) return false;
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pendingUnmap->issued).count();
        lastUnmapSeconds.store(seconds, std::memory_order_relaxed);
        raisePeak(maxUnmapSeconds, seconds);
        completedUnmaps.fetch_add(1, std::memory_order_relaxed);
        pendingUnmap.reset();
        pendingUnmapCount.store(0, std::memory_order_release);
        sampleDeviceMemory();
        return true;
    }

    // Requires commandMutex. Blocks until the outstanding unmap completes.
    void awaitSparseUnmapLocked() {
        if (!pendingUnmap) return;
        if (![sparseEvent waitUntilSignaledValue:pendingUnmap->eventValue
                                       timeoutMS:kSparseUnmapTimeoutMilliseconds]) {
            std::ostringstream details;
            details << "sparse unmapping timed out: event="
                    << pendingUnmap->eventValue
                    << " signaled=" << sparseEvent.signaledValue
                    << " pending_map=" << pendingSparseEventValue
                    << " waited_ms=" << kSparseUnmapTimeoutMilliseconds;
            std::string message = details.str();
            markUnhealthy(message);
            throw MetalBackendError(message);
        }
        static_cast<void>(reapSparseUnmapsLocked());
    }

    uint64_t sampleDeviceMemory() const noexcept {
        return asyncState->sampleDeviceMemory();
    }

    void ensureHealthy() const {
        asyncState->ensureHealthy();
    }

    void markUnhealthy(std::string reason) {
        asyncState->markUnhealthy(std::move(reason));
    }

    MetalBuffer wrap(std::shared_ptr<MetalAllocation> allocation) {
        auto result = std::make_shared<MetalBuffer::Impl>();
        result->lengthBytes = allocation->buffer.length;
        result->allocation = std::move(allocation);
        return MetalBuffer(std::move(result));
    }

    MetalBuffer registerBuffer(id<MTLBuffer> buffer, BufferStorage storage,
                               std::shared_ptr<void> externalOwner = {}) {
        auto allocation = std::make_shared<MetalAllocation>();
        allocation->externalOwner = std::move(externalOwner);
        allocation->buffer = buffer;
        allocation->accounting = accounting;
        allocation->bytes = buffer.allocatedSize;
        allocation->storage = storage;
        raisePeak(accounting->peakAllocatedBytes,
                  accounting->allocatedBytes.fetch_add(
                      allocation->bytes, std::memory_order_relaxed) +
                      allocation->bytes);
        accounting->addResident(allocation->bytes);
        sampleDeviceMemory();
        return wrap(std::move(allocation));
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLComputePipelineState> cached = [pipelines objectForKey:key];
        if (cached) return cached;

        id<MTLFunction> function = [library newFunctionWithName:key];
        if (!function) {
            throw MetalBackendError(
                "missing Metal function: " + std::string(name));
        }
        NSError *error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
        }
        [pipelines setObject:result forKey:key];
        sampleDeviceMemory();
        return result;
    }

    id<MTLArgumentEncoder> readOnlyArgumentEncoder(
        std::string_view name, uint32_t bufferIndex,
        std::vector<MTLPointerType *> *pointerLayout = nullptr) {
        if (device.argumentBuffersSupport != MTLArgumentBuffersTier2)
            throw MetalBackendError("Metal Tier 2 argument buffers are unavailable");
        if (name.empty() || bufferIndex >= 31)
            throw MetalBackendError("invalid Metal argument-buffer function binding");
        id<MTLFunction> function = [library newFunctionWithName:
            checkedNSString(name, "argument-buffer function")];
        if (!function)
            throw MetalBackendError("missing Metal function: " + std::string(name));
        if (pointerLayout) {
            NSError *error = nil;
            MTLComputePipelineReflection *reflection = nil;
            id<MTLComputePipelineState> reflectedPipeline = [device
                newComputePipelineStateWithFunction:function
                options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                reflection:&reflection error:&error];
            if (!reflectedPipeline || !reflection)
                throw MetalBackendError("unable to reflect Metal argument buffer: " +
                    errorDescription(error));
            MTLStructType *layout = nil;
            for (id<MTLBinding> binding in reflection.bindings)
                if (binding.type == MTLBindingTypeBuffer && binding.index == bufferIndex) {
                    id<MTLBufferBinding> bufferBinding = (id<MTLBufferBinding>)binding;
                    if (binding.access != MTLBindingAccessReadOnly ||
                        !bufferBinding.bufferPointerType.elementIsArgumentBuffer)
                        throw MetalBackendError("Metal source argument buffer must be read-only");
                    layout = bufferBinding.bufferStructType;
                    if (!layout) layout = bufferBinding.bufferPointerType.elementStructType;
                    break;
                }
            if (!layout || !layout.members.count)
                throw MetalBackendError("Metal argument buffer has no reflected pointer layout");
            const auto appendStruct = [&](auto &&self, MTLStructType *structure,
                                          uint64_t baseIndex, uint32_t depth) -> void {
                if (!structure || !structure.members.count || depth > 16)
                    throw MetalBackendError("invalid Metal argument-buffer reflected structure");
                for (MTLStructMember *member in structure.members) {
                    if (member.argumentIndex > UINT32_MAX - baseIndex)
                        throw MetalBackendError("Metal argument-buffer index extent overflows");
                    const uint64_t index = baseIndex + member.argumentIndex;
                    // metal::array is reflected as a struct with one __elems
                    // array member. Its child argument IDs are relative to the
                    // wrapper's base ID, rather than already globally offset.
                    if (member.dataType == MTLDataTypeStruct) {
                        self(self, member.structType, index, depth + 1);
                        continue;
                    }
                    MTLPointerType *pointer = nil;
                    uint64_t count = 1, stride = 1;
                    if (member.dataType == MTLDataTypePointer) {
                        pointer = member.pointerType;
                    } else if (member.dataType == MTLDataTypeArray &&
                               member.arrayType.elementType == MTLDataTypePointer) {
                        pointer = member.arrayType.elementPointerType;
                        count = member.arrayType.arrayLength;
                        stride = member.arrayType.argumentIndexStride;
                    }
                    if (!pointer || pointer.access != MTLBindingAccessReadOnly ||
                        pointer.elementIsArgumentBuffer || !count || stride != 1 ||
                        index != pointerLayout->size() ||
                        count > UINT32_MAX - pointerLayout->size())
                        throw MetalBackendError("Metal argument buffer requires dense read-only source pointers");
                    pointerLayout->insert(pointerLayout->end(), count, pointer);
                }
            };
            appendStruct(appendStruct, layout, 0, 0);
            [pipelines setObject:reflectedPipeline forKey:checkedNSString(name, "pipeline name")];
            sampleDeviceMemory();
        }
        id<MTLArgumentEncoder> encoder = nil;
        @try {
            encoder = [function newArgumentEncoderWithBufferIndex:bufferIndex];
        } @catch (NSException *exception) {
            throw MetalBackendError("invalid Metal argument-buffer layout: " +
                std::string(exception.reason.UTF8String ?: "unknown exception"));
        }
        if (!encoder || !encoder.encodedLength ||
            encoder.encodedLength > capabilities.maxBufferLengthBytes)
            throw MetalBackendError("invalid Metal argument-buffer byte extent");
        return encoder;
    }
};

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;

MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalBuffer::operator bool() const noexcept {
    return impl_ && impl_->allocation && impl_->allocation->buffer;
}

uint64_t MetalBuffer::sizeBytes() const noexcept {
    return impl_ ? impl_->lengthBytes : 0;
}

bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
    if (impl_ == other.impl_) return true;
    return impl_ && other.impl_ &&
           impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes == other.impl_->offsetBytes &&
           impl_->lengthBytes == other.impl_->lengthBytes;
}

BufferStorage MetalBuffer::storage() const noexcept {
    return impl_ && impl_->allocation ? impl_->allocation->storage
                                      : BufferStorage::Shared;
}

void *MetalBuffer::contents() const noexcept {
    if (!impl_ || !impl_->allocation ||
        impl_->allocation->storage != BufferStorage::Shared) {
        return nullptr;
    }
    void *base = impl_->allocation->buffer.contents;
    if (!base) return nullptr;
    return static_cast<uint8_t *>(base) + impl_->offsetBytes;
}

SparseHeap::SparseHeap() = default;
SparseHeap::~SparseHeap() = default;
SparseHeap::SparseHeap(SparseHeap &&) noexcept = default;
SparseHeap &SparseHeap::operator=(SparseHeap &&) noexcept = default;

SparseHeap::SparseHeap(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SparseHeap::operator bool() const noexcept {
    return impl_ && impl_->heap;
}

uint64_t SparseHeap::sizeBytes() const noexcept {
    return impl_ ? impl_->bytes : 0;
}

ResidencyLease::ResidencyLease() = default;
ResidencyLease::~ResidencyLease() = default;
ResidencyLease::ResidencyLease(ResidencyLease &&) noexcept = default;
ResidencyLease &ResidencyLease::operator=(ResidencyLease &&) noexcept = default;
ResidencyLease::ResidencyLease(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ResidencyLease::operator bool() const noexcept {
    return impl_ && impl_->registration;
}

uint64_t ResidencyLease::bufferCount() const noexcept {
    return impl_ && impl_->registration
        ? impl_->registration->allocations.size() : 0;
}

uint64_t ResidencyLease::byteCount() const noexcept {
    return impl_ && impl_->registration ? impl_->registration->bytes : 0;
}

CommandTicket::CommandTicket() = default;

CommandTicket::CommandTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

CommandTicket::~CommandTicket() {
    if (state_) state_->abandon();
}

CommandTicket::CommandTicket(CommandTicket &&) noexcept = default;

CommandTicket &CommandTicket::operator=(CommandTicket &&other) noexcept {
    if (this == &other) return *this;
    if (state_) state_->abandon();
    state_ = std::move(other.state_);
    return *this;
}

CommandTicket::operator bool() const noexcept {
    return static_cast<bool>(state_);
}

uint64_t CommandTicket::sequence() const noexcept {
    return state_ ? state_->sequence : 0;
}

bool CommandTicket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

CommandTiming CommandTicket::wait() {
    if (!state_) throw MetalBackendError("Metal command ticket is empty");
    CommandTiming timing;
    std::string error;
    const double waitStarted = steadySeconds();
    {
        std::unique_lock lock(state_->mutex);
        state_->condition.wait(lock, [this] { return state_->completed; });
        timing = state_->timing;
        error = state_->error;
    }
    const double waitEnded = steadySeconds();
    const auto recorded = state_->hostTiming.snapshot();
    // Synthetic legacy-replay tickets already contain accumulated subcommands.
    if (recorded.timedCommands) timing.host = recorded;
    ++timing.host.ticketWaitCalls;
    timing.host.ticketBlockingWaitSeconds += std::max(0.0, waitEnded - waitStarted);
    state_->release();
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
}

MetalBackend::MetalBackend(std::string metallibPath, double commandTimeoutSeconds)
    : impl_(std::make_unique<Impl>()) {
    impl_->asyncState->commandWatchdog = CommandWatchdog(commandTimeoutSeconds);
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first.
        const NSOperatingSystemVersion os =
            NSProcessInfo.processInfo.operatingSystemVersion;
        const auto component = [](NSInteger value) {
            return value > 0 ? static_cast<uint32_t>(value) : 0U;
        };
        impl_->capabilities.macosMajor = component(os.majorVersion);
        impl_->capabilities.macosMinor = component(os.minorVersion);
        impl_->capabilities.macosPatch = component(os.patchVersion);
        if (!impl_->capabilities.meetsMinimumMacos()) {
            throw MetalBackendError(
                "Splash requires macOS " +
                std::to_string(DeviceCapabilities::kMinimumMacosMajor) + '.' +
                std::to_string(DeviceCapabilities::kMinimumMacosMinor) +
                " or newer; this Mac runs macOS " +
                impl_->capabilities.macosVersion());
        }
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw MetalBackendError("Metal device unavailable");
        }
        impl_->asyncState->device = impl_->device;
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }
        impl_->profilingCapability.dispatchBoundary = [impl_->device
            supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
        impl_->profilingCapability.stageBoundary = [impl_->device
            supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        for (id<MTLCounterSet> counterSet in impl_->device.counterSets) {
            if ([counterSet.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                for (id<MTLCounter> counter in counterSet.counters) {
                    if ([counter.name isEqualToString:MTLCommonCounterTimestamp]) {
                        impl_->timestampCounterSet = counterSet;
                        impl_->profilingCapability.timestampCounterSet = true;
                        break;
                    }
                }
                break;
            }
        }
        if (!impl_->timestampCounterSet)
            impl_->profilingCapability.reason = "timestamp counter set unavailable";
        else if (!impl_->profilingCapability.dispatchBoundary &&
                 !impl_->profilingCapability.stageBoundary)
            impl_->profilingCapability.reason =
                "neither dispatch nor stage counter sampling is supported";

        NSString *path = checkedNSString(metallibPath, "metallib path");
        NSError *error = nil;
        NSData *fileData = [NSData dataWithContentsOfFile:path
                                                 options:0
                                                   error:&error];
        if (!fileData) {
            throw MetalBackendError(
                "unable to read metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        if (!fileData.length ||
            fileData.length > std::numeric_limits<CC_LONG>::max()) {
            throw MetalBackendError("metallib is empty or too large to hash: " +
                                    metallibPath);
        }
        // DEFAULT copies into immutable dispatch-owned storage. Hash the same
        // contiguous data passed to Metal, never a second read of the path.
        dispatch_data_t data = dispatch_data_create(
            fileData.bytes, fileData.length, nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (!data)
            throw MetalBackendError("unable to copy metallib data: " + metallibPath);
        const void *bytes = nullptr;
        size_t byteCount = 0;
        dispatch_data_t mapped = dispatch_data_create_map(data, &bytes, &byteCount);
        if (!mapped || !bytes || byteCount != fileData.length ||
            !CC_SHA256(bytes, static_cast<CC_LONG>(byteCount),
                       impl_->metallibSha256.data())) {
            throw MetalBackendError("unable to hash metallib data: " + metallibPath);
        }
        error = nil;
        impl_->library =
            [impl_->device newLibraryWithData:mapped error:&error];
        if (!impl_->library) {
            throw MetalBackendError(
                "unable to load metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        impl_->pipelines = [NSMutableDictionary dictionary];
        if (!impl_->pipelines) {
            throw MetalBackendError("unable to create Metal pipeline cache");
        }
        impl_->sampleDeviceMemory();

        DeviceCapabilities &capabilities = impl_->capabilities;
        capabilities.deviceName = stringFromNSString(impl_->device.name);
        capabilities.gpuCoreCount = gpuCoreCountForDevice(impl_->device.registryID);
        for (uint32_t family = 10; family >= 7; --family) {
            if ([impl_->device supportsFamily:
                    static_cast<MTLGPUFamily>(1000 + family)]) {
                capabilities.appleGpuFamily = family;
                break;
            }
        }
        capabilities.physicalMemoryBytes =
            NSProcessInfo.processInfo.physicalMemory;
        capabilities.recommendedMaxWorkingSetBytes =
            impl_->device.recommendedMaxWorkingSetSize;
        capabilities.maxBufferLengthBytes = impl_->device.maxBufferLength;
        capabilities.maxThreadgroupMemoryBytes =
            impl_->device.maxThreadgroupMemoryLength;
        MTLSize maximumThreads = impl_->device.maxThreadsPerThreadgroup;
        capabilities.maxThreadgroupWidth = maximumThreads.width;
        capabilities.hasUnifiedMemory = impl_->device.hasUnifiedMemory;

        // Query sparse support and exercise the private-buffer/placement-heap ABI.
        if (@available(macOS 26.4, *)) {
            if (queryPlacementSparseSupport(impl_->device)) {
                impl_->sparseQueue = [impl_->device newMTL4CommandQueue];
                impl_->sparseEvent = [impl_->device newSharedEvent];
                if (!impl_->sparseQueue || !impl_->sparseEvent) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its queue or event");
                }

                id<MTLBuffer> canaryBuffer = [impl_->device
                    newBufferWithLength:kPlacementSparsePageBytes
                    options:MTLResourceStorageModePrivate
                    placementSparsePageSize:kPlacementSparsePageSize];
                MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
                if (!descriptor) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its heap descriptor");
                }
                descriptor.type = MTLHeapTypePlacement;
                descriptor.storageMode = MTLStorageModePrivate;
                descriptor.size = kPlacementSparsePageBytes;
                descriptor.maxCompatiblePlacementSparsePageSize =
                    kPlacementSparsePageSize;
                id<MTLHeap> canaryHeap =
                    [impl_->device newHeapWithDescriptor:descriptor];
                if (!canaryBuffer || !canaryHeap) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its buffer or heap");
                }
                MTLSharedEventListener *listener =
                    [MTLSharedEventListener sharedListener];
                if (!listener) {
                    throw MetalAllocationError(
                        "placement-sparse probe could not allocate its completion listener");
                }
                MTL4UpdateSparseBufferMappingOperation operation{};
                operation.mode = MTLSparseTextureMappingModeMap;
                operation.bufferRange = NSMakeRange(0, 1);
                operation.heapOffset = 0;
                [impl_->sparseQueue updateBufferMappings:canaryBuffer
                                                   heap:canaryHeap
                                             operations:&operation
                                                  count:1];
                [impl_->sparseQueue signalEvent:impl_->sparseEvent value:1];
                BOOL mapped = [impl_->sparseEvent
                    waitUntilSignaledValue:1 timeoutMS:5000];

                operation.mode = MTLSparseTextureMappingModeUnmap;
                [impl_->sparseQueue updateBufferMappings:canaryBuffer
                                                   heap:nil
                                             operations:&operation
                                                  count:1];
                [impl_->sparseQueue signalEvent:impl_->sparseEvent value:2];
                BOOL unmapped = [impl_->sparseEvent
                    waitUntilSignaledValue:2 timeoutMS:5000];
                if (!unmapped) {
                    // Only an unfinished probe needs asynchronous ownership.
                    id<MTL4CommandQueue> probeQueue = impl_->sparseQueue;
                    id<MTLSharedEvent> probeEvent = impl_->sparseEvent;
                    [impl_->sparseEvent notifyListener:listener atValue:2
                        block:^(id<MTLSharedEvent>, uint64_t) {
                            (void)canaryBuffer;
                            (void)canaryHeap;
                            (void)probeQueue;
                            (void)probeEvent;
                        }];
                }
                if (!mapped || !unmapped) {
                    throw MetalBackendError(
                        std::string("placement-sparse probe timed out after 5000 ms waiting for ") +
                        (!mapped ? "mapping" : "unmapping") +
                        " (last signaled event=" +
                        std::to_string(impl_->sparseEvent.signaledValue) + ')');
                }
                capabilities.supportsPlacementSparse = true;
                impl_->nextSparseEventValue = 2;
            }
        }
    }
    impl_->sampleDeviceMemory();
}

MetalBackend::~MetalBackend() { stop(); }

void MetalBackend::stop() noexcept {
    if (!impl_) return;
    impl_->asyncState->stop();
}
void MetalBackend::setCancellationProbe(std::function<bool()> probe) {
    impl_->cancelled = std::move(probe);
}

MetalBackend::MetalBackend(MetalBackend &&) noexcept = default;
MetalBackend &MetalBackend::operator=(MetalBackend &&other) noexcept {
    if (this != &other) {
        stop();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

const DeviceCapabilities &MetalBackend::capabilities() const noexcept {
    return impl_->capabilities;
}

const std::array<uint8_t, 32> &MetalBackend::metallibSha256() const noexcept {
    return impl_->metallibSha256;
}

void MetalBackend::checkOperation() const {
    impl_->ensureHealthy();
    if (impl_->operationGuard) impl_->operationGuard();
}

void MetalBackend::setOperationGuard(std::function<void()> guard) {
    impl_->operationGuard = std::move(guard);
}

MetalBuffer MetalBackend::allocateBuffer(uint64_t bytes,
                                         BufferStorage storage,
                                         std::string_view label) {
    checkOperation();
    if (!bytes) throw MetalBackendError("Metal buffer size must be positive");
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }

    MTLResourceOptions options = storage == BufferStorage::Shared
        ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
    id<MTLBuffer> buffer = [impl_->device
        newBufferWithLength:checkedNSUInteger(bytes, "buffer size")
        options:options];
    if (!buffer) throw MetalAllocationError("Metal buffer allocation failed");
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, storage);
}

MetalBuffer MetalBackend::allocatePlacementSparseBuffer(
    uint64_t virtualBytes, uint64_t sparsePageBytes, std::string_view label) {
    checkOperation();
    const MTLSparsePageSize pageSize = metalSparsePageSize(sparsePageBytes);
    if (!impl_->capabilities.supportsPlacementSparse) {
        throw MetalBackendError("placement-sparse Metal is unavailable");
    }
    if (!virtualBytes || virtualBytes % sparsePageBytes) {
        throw MetalBackendError(
            "placement-sparse buffer size must be tile-aligned");
    }
    if (virtualBytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError(
            "placement-sparse buffer exceeds maxBufferLength");
    }

    id<MTLBuffer> buffer = [impl_->device
        newBufferWithLength:checkedNSUInteger(virtualBytes, "sparse buffer size")
        options:MTLResourceStorageModePrivate
        placementSparsePageSize:pageSize];
    if (!buffer) {
        throw MetalAllocationError(
            "placement-sparse buffer creation failed");
    }
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");

    auto allocation = std::make_shared<MetalAllocation>();
    allocation->buffer = buffer;
    allocation->accounting = impl_->accounting;
    allocation->sparseVirtualBytes = virtualBytes;
    allocation->placementSparse = true;
    allocation->storage = BufferStorage::Private;
    impl_->accounting->sparseVirtualBytes.fetch_add(
        virtualBytes, std::memory_order_relaxed);
    impl_->sampleDeviceMemory();
    return impl_->wrap(std::move(allocation));
}

SparseHeap MetalBackend::allocatePlacementHeap(
    uint64_t physicalBytes, uint64_t sparsePageBytes, std::string_view label) {
    impl_->ensureHealthy();
    const MTLSparsePageSize pageSize = metalSparsePageSize(sparsePageBytes);
    if (!impl_->capabilities.supportsPlacementSparse) {
        throw MetalBackendError("placement-sparse Metal is unavailable");
    }
    if (!physicalBytes || physicalBytes % sparsePageBytes) {
        throw MetalBackendError(
            "placement heap size must be tile-aligned");
    }

    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = checkedNSUInteger(physicalBytes, "placement heap size");
    descriptor.maxCompatiblePlacementSparsePageSize = pageSize;
    id<MTLHeap> heap = [impl_->device newHeapWithDescriptor:descriptor];
    if (!heap) {
        throw MetalAllocationError("placement heap allocation failed");
    }
    if (!label.empty()) heap.label = checkedNSString(label, "heap label");

    auto result = std::make_shared<SparseHeap::Impl>();
    result->heap = heap;
    const uint64_t heapBytes = static_cast<uint64_t>(heap.size);
    if (heapBytes < physicalBytes || heapBytes % sparsePageBytes) {
        throw MetalBackendError("placement heap has unexpected size");
    }
    result->accounting = impl_->accounting;
    result->bytes = heapBytes;
    raisePeak(impl_->accounting->peakSparseResidentBytes,
              impl_->accounting->sparseResidentBytes.fetch_add(
                  result->bytes, std::memory_order_relaxed) + result->bytes);
    impl_->accounting->addResident(result->bytes);
    impl_->sampleDeviceMemory();
    return SparseHeap(std::move(result));
}

void MetalBackend::mapSparse(
    const SparseHeap &heap, std::span<const SparseMapping> mappings) {
    if (!heap.impl_ || !heap.impl_->heap ||
        heap.impl_->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("placement heap belongs to another backend");
    }
    if (mappings.empty()) {
        throw MetalBackendError("sparse mapping list must not be empty");
    }

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    if (impl_->asyncState->hasActiveSubmission()) {
        throw MetalBackendError(
            "cannot map sparse memory while a command is in flight");
    }
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    const uint64_t tileBytes = kPlacementSparsePageBytes;
    for (const SparseMapping &mapping : mappings) {
        if (!mapping.buffer.impl_ ||
            !mapping.buffer.impl_->allocation ||
            mapping.buffer.impl_->allocation->accounting.get() !=
                impl_->accounting.get() ||
            !mapping.buffer.impl_->allocation->placementSparse) {
            throw MetalBackendError("invalid placement-sparse buffer");
        }
        if (!mapping.sizeBytes ||
            mapping.bufferOffsetBytes % tileBytes ||
            mapping.sizeBytes % tileBytes ||
            mapping.heapOffsetBytes % tileBytes ||
            mapping.bufferOffsetBytes > mapping.buffer.sizeBytes() ||
            mapping.sizeBytes >
                mapping.buffer.sizeBytes() - mapping.bufferOffsetBytes ||
            mapping.heapOffsetBytes > heap.impl_->bytes ||
            mapping.sizeBytes > heap.impl_->bytes - mapping.heapOffsetBytes) {
            throw MetalBackendError("sparse mapping range is invalid");
        }
    }
    if (impl_->nextSparseEventValue ==
        std::numeric_limits<uint64_t>::max()) {
        throw MetalBackendError("sparse event sequence exhausted");
    }

    // A range released moments ago may be mapped again to a new heap. Make
    // the map depend on the in-flight unmap explicitly rather than relying
    // on queue order alone; compute submission follows both completions.
    if (impl_->pendingUnmap) {
        [impl_->sparseQueue waitForEvent:impl_->sparseEvent
                                 value:impl_->pendingUnmap->eventValue];
    }
    for (const SparseMapping &mapping : mappings) {
        MTL4UpdateSparseBufferMappingOperation operation{};
        operation.mode = MTLSparseTextureMappingModeMap;
        operation.bufferRange = NSMakeRange(
            checkedNSUInteger(mapping.bufferOffsetBytes / tileBytes,
                              "sparse buffer tile offset"),
            checkedNSUInteger(mapping.sizeBytes / tileBytes,
                              "sparse mapping tile count"));
        operation.heapOffset = checkedNSUInteger(
            mapping.heapOffsetBytes / tileBytes, "sparse heap tile offset");
        [impl_->sparseQueue
            updateBufferMappings:mapping.buffer.impl_->allocation->buffer
                             heap:heap.impl_->heap
                       operations:&operation
                            count:1];
    }
    const uint64_t eventValue = ++impl_->nextSparseEventValue;
    // A failed dependency wait must not release backing still being mapped.
    auto retainedHeap = heap.impl_;
    std::vector<SparseMapping> retainedMappings(mappings.begin(), mappings.end());
    id<MTL4CommandQueue> retainedQueue = impl_->sparseQueue;
    id<MTLSharedEvent> retainedEvent = impl_->sparseEvent;
    [impl_->sparseEvent notifyListener:[MTLSharedEventListener sharedListener]
        atValue:eventValue block:^(id<MTLSharedEvent>, uint64_t) {
            (void)retainedHeap;
            (void)retainedMappings;
            (void)retainedQueue;
            (void)retainedEvent;
        }];
    [impl_->sparseQueue signalEvent:impl_->sparseEvent value:eventValue];
    impl_->pendingSparseEventValue = eventValue;
}

void MetalBackend::unmapSparse(
    std::span<const SparseMapping> mappings, SparseHeap &&heap) {
    if (mappings.empty()) {
        throw MetalBackendError("sparse unmapping list must not be empty");
    }
    if (!heap.impl_ || !heap.impl_->heap ||
        heap.impl_->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError(
            "sparse unmapping requires the mapped placement heap");
    }

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    if (impl_->asyncState->hasActiveSubmission()) {
        throw MetalBackendError(
            "cannot unmap sparse memory while a command is in flight");
    }
    const uint64_t tileBytes = kPlacementSparsePageBytes;
    for (const SparseMapping &mapping : mappings) {
        if (!mapping.buffer.impl_ ||
            !mapping.buffer.impl_->allocation ||
            mapping.buffer.impl_->allocation->accounting.get() !=
                impl_->accounting.get() ||
            !mapping.buffer.impl_->allocation->placementSparse ||
            !mapping.sizeBytes ||
            mapping.bufferOffsetBytes % tileBytes ||
            mapping.sizeBytes % tileBytes ||
            mapping.bufferOffsetBytes > mapping.buffer.sizeBytes() ||
            mapping.sizeBytes >
                mapping.buffer.sizeBytes() - mapping.bufferOffsetBytes) {
            throw MetalBackendError("sparse unmapping range is invalid");
        }
    }
    if (impl_->nextSparseEventValue ==
        std::numeric_limits<uint64_t>::max()) {
        throw MetalBackendError("sparse event sequence exhausted");
    }

    // One outstanding unmap at a time keeps the kernel's per-tile teardown
    // paced; the caller normally checks sparseUnmapPending() first.
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    impl_->awaitSparseUnmapLocked();

    // Allocation rollback may unmap before compute has consumed the map
    // event. Order that dependent update explicitly on the Metal 4 queue.
    if (impl_->pendingSparseEventValue) {
        [impl_->sparseQueue waitForEvent:impl_->sparseEvent
                                 value:impl_->pendingSparseEventValue];
    }
    for (const SparseMapping &mapping : mappings) {
        MTL4UpdateSparseBufferMappingOperation operation{};
        operation.mode = MTLSparseTextureMappingModeUnmap;
        operation.bufferRange = NSMakeRange(
            checkedNSUInteger(mapping.bufferOffsetBytes / tileBytes,
                              "sparse buffer tile offset"),
            checkedNSUInteger(mapping.sizeBytes / tileBytes,
                              "sparse unmapping tile count"));
        [impl_->sparseQueue
            updateBufferMappings:mapping.buffer.impl_->allocation->buffer
                             heap:nil
                       operations:&operation
                            count:1];
    }
    const uint64_t eventValue = ++impl_->nextSparseEventValue;
    [impl_->sparseQueue signalEvent:impl_->sparseEvent value:eventValue];
    const auto issued = std::chrono::steady_clock::now();
    impl_->pendingUnmap.emplace();
    impl_->pendingUnmap->eventValue = eventValue;
    impl_->pendingUnmap->heap = std::move(heap);
    impl_->pendingUnmap->issued = issued;
    impl_->pendingUnmapIssuedSeconds.store(
        std::chrono::duration<double>(issued.time_since_epoch()).count(),
        std::memory_order_relaxed);
    impl_->pendingUnmapCount.store(1, std::memory_order_release);
    impl_->sampleDeviceMemory();
}

bool MetalBackend::sparseUnmapPending() noexcept {
    // Reap opportunistically; a command being encoded on another thread
    // must not stall the caller, which is often the reclaim pacing loop.
    if (std::unique_lock commandLock(impl_->commandMutex, std::try_to_lock);
        commandLock.owns_lock()) {
        static_cast<void>(impl_->reapSparseUnmapsLocked());
    }
    if (impl_->pendingUnmapCount.load(std::memory_order_acquire) == 0)
        return false;
    // An unmap outstanding for longer than the drain's bounded wait is the
    // same fault the drain would report, observed here without blocking the
    // serving loop: the backend marks itself unhealthy and the supervisor
    // replaces the engine.
    const double issued =
        impl_->pendingUnmapIssuedSeconds.load(std::memory_order_relaxed);
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (issued > 0.0 &&
        (now - issued) * 1000.0 > double(kSparseUnmapTimeoutMilliseconds)) {
        try {
            impl_->markUnhealthy(
                "sparse unmapping exceeded " +
                std::to_string(kSparseUnmapTimeoutMilliseconds) +
                " ms without completing");
        } catch (...) {
        }
    }
    return true;
}

void MetalBackend::drainSparseUnmaps() {
    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    impl_->awaitSparseUnmapLocked();
}

MetalBuffer MetalBackend::wrapSharedMemory(
    void *address, uint64_t bytes, std::shared_ptr<void> lifetime,
    std::string_view label) {
    checkOperation();
    if (!address || !bytes) {
        throw MetalBackendError("shared memory address and size are required");
    }
    if (!lifetime) {
        throw MetalBackendError("shared memory lifetime token is required");
    }
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("shared memory exceeds maxBufferLength");
    }
    long systemPageSize = sysconf(_SC_PAGESIZE);
    if (systemPageSize <= 0) {
        throw MetalBackendError("unable to determine system page size");
    }
    uint64_t pageSize = static_cast<uint64_t>(systemPageSize);
    if (reinterpret_cast<uintptr_t>(address) % pageSize || bytes % pageSize) {
        throw MetalBackendError(
            "shared memory address and size must be page-aligned");
    }

    id<MTLBuffer> buffer = [impl_->device
        newBufferWithBytesNoCopy:address
        length:checkedNSUInteger(bytes, "shared memory size")
        options:MTLResourceStorageModeShared
        deallocator:^(void *, NSUInteger) {
            // Metal may retain the buffer beyond our last C++ view/ticket,
            // including while a completed command's handler is returning.
            // Keep its backing owner until Metal actually releases it.
            (void)lifetime;
        }];
    if (!buffer) {
        throw MetalBackendError("zero-copy Metal buffer creation failed");
    }
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, BufferStorage::Shared,
                                 std::move(lifetime));
}

MetalBuffer MetalBackend::view(const MetalBuffer &base,
                               uint64_t offsetBytes,
                               uint64_t lengthBytes) const {
    impl_->ensureHealthy();
    if (!base.impl_ || !base.impl_->allocation) {
        throw MetalBackendError("cannot view an empty Metal buffer");
    }
    if (base.impl_->allocation->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("Metal buffer belongs to another backend");
    }
    if (!lengthBytes || offsetBytes > base.impl_->lengthBytes ||
        lengthBytes > base.impl_->lengthBytes - offsetBytes) {
        std::ostringstream message;
        message << "Metal buffer view is out of range: offset=" << offsetBytes
                << " length=" << lengthBytes
                << " base_length=" << base.impl_->lengthBytes;
        throw MetalBackendError(message.str());
    }
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->allocation = base.impl_->allocation;
    result->offsetBytes = base.impl_->offsetBytes + offsetBytes;
    result->lengthBytes = lengthBytes;
    return MetalBuffer(std::move(result));
}

bool MetalBackend::supportsArgumentBuffersTier2() const noexcept {
    return impl_->device.argumentBuffersSupport == MTLArgumentBuffersTier2;
}

uint64_t MetalBackend::readOnlyArgumentBufferByteCount(
    std::string_view pipelineName, uint32_t bufferIndex) {
    impl_->checkCancellation();
    impl_->ensureHealthy();
    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    return impl_->readOnlyArgumentEncoder(pipelineName, bufferIndex).encodedLength;
}

MetalBuffer MetalBackend::makeReadOnlyArgumentBuffer(
    std::string_view pipelineName, uint32_t bufferIndex,
    std::span<const BufferBinding> resources, std::string_view label) {
    impl_->checkCancellation();
    impl_->ensureHealthy();
    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    if (impl_->asyncState->hasActiveSubmission())
        throw MetalBackendError("argument-buffer creation requires a safe point");
    if (resources.empty() || resources.size() > UINT32_MAX)
        throw MetalBackendError("argument-buffer resources must be nonempty");
    std::vector<const BufferBinding *> ordered(resources.size(), nullptr);
    std::vector<std::shared_ptr<MetalAllocation>> retained;
    retained.reserve(resources.size());
    std::unordered_set<const MetalAllocation *> seen;
    for (const auto &resource : resources) {
        if (resource.index >= resources.size() || ordered[resource.index])
            throw MetalBackendError("argument-buffer indices must be unique and dense");
        if (!resource.buffer.impl_ || !resource.buffer.impl_->allocation)
            throw MetalBackendError("argument-buffer resource is empty");
        const auto &view = *resource.buffer.impl_;
        const auto &allocation = view.allocation;
        if (allocation->accounting.get() != impl_->accounting.get())
            throw MetalBackendError("argument-buffer resource belongs to another backend");
        if (allocation->placementSparse ||
            !allocation->readOnlyIndirectAllocations.empty())
            throw MetalBackendError("argument-buffer resource must be an ordinary source buffer");
        if (!view.lengthBytes || view.offsetBytes > allocation->buffer.length ||
            view.lengthBytes > allocation->buffer.length - view.offsetBytes)
            throw MetalBackendError("argument-buffer resource view is out of range");
        ordered[resource.index] = &resource;
        if (seen.insert(allocation.get()).second) retained.push_back(allocation);
    }
    std::vector<MTLPointerType *> pointerLayout;
    id<MTLArgumentEncoder> encoder = impl_->readOnlyArgumentEncoder(
        pipelineName, bufferIndex, &pointerLayout);
    if (resources.size() != pointerLayout.size())
        throw MetalBackendError("argument-buffer resource count differs from its reflected layout");
    for (uint64_t i = 0; i < ordered.size(); ++i) {
        const auto &view = *ordered[i]->buffer.impl_;
        const auto pointer = pointerLayout[i];
        if (!pointer.alignment || view.offsetBytes % pointer.alignment ||
            view.lengthBytes < pointer.dataSize)
            throw MetalBackendError("argument-buffer source view violates its reflected pointer type");
    }
    auto result = allocateBuffer(encoder.encodedLength, BufferStorage::Shared,
                                  label);
    std::memset(result.contents(), 0, result.sizeBytes());
    @try {
        [encoder setArgumentBuffer:result.impl_->allocation->buffer offset:0];
        for (const auto *resource : ordered) {
            const auto &view = *resource->buffer.impl_;
            [encoder setBuffer:view.allocation->buffer
                        offset:checkedNSUInteger(view.offsetBytes,
                                                 "argument-buffer source offset")
                       atIndex:resource->index];
        }
    } @catch (NSException *exception) {
        throw MetalBackendError("invalid Metal argument-buffer pointer binding: " +
            std::string(exception.reason.UTF8String ?: "unknown exception"));
    }
    result.impl_->allocation->readOnlyIndirectAllocations = std::move(retained);
    return result;
}

ResidencyLease MetalBackend::requestWeightResidency(
    std::span<const MetalBuffer> weights, std::string_view label) {
    impl_->ensureHealthy();
    if (weights.empty())
        throw MetalBackendError("weight residency requires immutable weight buffers");
    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    const auto ensureStartup = [&] {
        if (impl_->asyncState->stopping.stop_requested())
            throw MetalBackendError("Metal backend is stopping");
        if (impl_->asyncState->activeSequence)
            throw MetalBackendError("weight residency requires no outstanding command ticket");
        if (impl_->asyncState->weightResidency)
            throw MetalBackendError("weight residency was already requested for this backend");
    };
    {
        std::lock_guard gateLock(impl_->asyncState->gateMutex);
        ensureStartup();
    }
    if (![impl_->device supportsFamily:MTLGPUFamilyApple6])
        throw MetalBackendError("device does not support model weight residency sets");

    auto registration = std::make_shared<WeightResidencyRegistration>();
    registration->queue = impl_->queue;
    registration->allocations.reserve(weights.size());
    std::vector<id<MTLAllocation>> nativeAllocations;
    nativeAllocations.reserve(weights.size());
    std::unordered_set<const void *> seen;
    for (const MetalBuffer &weight : weights) {
        if (!weight.impl_ || !weight.impl_->allocation || !weight.sizeBytes())
            throw MetalBackendError("weight residency contains an empty buffer");
        const auto &allocation = weight.impl_->allocation;
        if (allocation->accounting.get() != impl_->accounting.get())
            throw MetalBackendError("weight residency buffer belongs to another backend");
        if (allocation->placementSparse)
            throw MetalBackendError("weight residency does not include sparse backing");
        const void *identity = (__bridge const void *)allocation->buffer;
        if (!seen.insert(identity).second) continue;
        if (allocation->bytes >
            std::numeric_limits<uint64_t>::max() - registration->bytes)
            throw MetalBackendError("weight residency byte count overflows");
        registration->bytes += allocation->bytes;
        registration->allocations.push_back(allocation);
        nativeAllocations.push_back(allocation->buffer);
    }
    // Allocate the wrapper before requesting residency; a subsequent CPU
    // allocation failure cannot leave a successful request unowned.
    auto lease = std::make_shared<ResidencyLease::Impl>();
    lease->registration = registration;
    @autoreleasepool {
        MTLResidencySetDescriptor *descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = checkedNSUInteger(nativeAllocations.size(),
                                                       "weight residency count");
        descriptor.label = label.empty() ? @"Splash immutable model weights"
                                        : checkedNSString(label, "weight residency label");
        NSError *error = nil;
        registration->set = [impl_->device newResidencySetWithDescriptor:descriptor
                                                                  error:&error];
        if (!registration->set)
            throw MetalBackendError("unable to create weight residency set: " +
                                    errorDescription(error));
        // Set methods are not thread-safe. Startup and stop share this gate;
        // no command is outstanding or can start during the setup sequence.
        std::lock_guard gateLock(impl_->asyncState->gateMutex);
        ensureStartup();
        @try {
            [registration->set addAllocations:nativeAllocations.data()
                                        count:nativeAllocations.size()];
            [registration->set commit];
            [registration->queue addResidencySet:registration->set];
            registration->attached = true;
            [registration->set requestResidency];
            registration->requested = true;
        } @catch (NSException *exception) {
            throw MetalBackendError("weight residency setup failed: " +
                                    stringFromNSString(exception.reason));
        }
        impl_->asyncState->weightResidency = registration;
    }
    // Startup is quiescent; expose any driver metadata allocation without
    // charging already-accounted weight backing a second time.
    impl_->sampleDeviceMemory();
    return ResidencyLease(std::move(lease));
}

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
    return submitAsync(dispatch).wait();
}

CommandTiming MetalBackend::submitCommand(
    std::span<const ComputeDispatch> dispatches) {
    return submitCommandAsync(dispatches).wait();
}

CommandTicket MetalBackend::submitAsync(
    const ComputeDispatch &dispatch, CommandCompletion completion) {
    return submitCommandAsync(
        std::span<const ComputeDispatch>(&dispatch, 1),
        std::move(completion));
}

void MetalBackend::setDispatchProfiling(bool enabled) {
    std::lock_guard lock(impl_->commandMutex);
    if (impl_->asyncState->hasActiveSubmission())
        throw MetalBackendError("cannot change profiling with an outstanding command ticket");
    if (enabled && impl_->commandDispatchProfiling.load(std::memory_order_acquire) !=
                       CommandDispatchProfilingMode::Off)
        throw MetalBackendError("legacy replay and counter profiling cannot both be enabled");
    impl_->dispatchProfiling.store(enabled, std::memory_order_release);
}

std::vector<DispatchTiming> MetalBackend::takeDispatchProfile() {
    return std::exchange(impl_->dispatchProfile, {});
}

CommandDispatchProfilingCapability
MetalBackend::commandDispatchProfilingCapability() const {
    return impl_->profilingCapability;
}

void MetalBackend::setCommandDispatchProfiling(CommandDispatchProfilingMode mode) {
    switch (mode) {
    case CommandDispatchProfilingMode::Off:
    case CommandDispatchProfilingMode::DispatchBoundary:
    case CommandDispatchProfilingMode::StagePerDispatch:
    case CommandDispatchProfilingMode::Command:
        break;
    default:
        throw MetalBackendError("unknown command dispatch profiling mode");
    }
    std::lock_guard lock(impl_->commandMutex);
    if (impl_->asyncState->hasActiveSubmission())
        throw MetalBackendError("cannot change profiling with an outstanding command ticket");
    if (impl_->dispatchProfiling && mode != CommandDispatchProfilingMode::Off)
        throw MetalBackendError("legacy replay and counter profiling cannot both be enabled");
    impl_->commandDispatchProfiling.store(mode, std::memory_order_release);
}

std::vector<CommandDispatchProfile> MetalBackend::takeCommandDispatchProfiles() {
    std::lock_guard lock(impl_->asyncState->profileMutex);
    return std::exchange(impl_->asyncState->commandProfiles, {});
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion) {
    checkOperation();
    if (dispatches.empty()) {
        throw MetalBackendError("Metal command must contain a dispatch");
    }
    const auto profileMode = impl_->commandDispatchProfiling.load(
        std::memory_order_acquire);
    const double profileStart = profileMode == CommandDispatchProfilingMode::Off
                                    ? 0.0 : steadySeconds();
    if (impl_->dispatchProfiling && profileMode != CommandDispatchProfilingMode::Off)
        throw MetalBackendError("legacy replay and counter profiling cannot both be enabled");
    if (impl_->dispatchProfiling && dispatches.size() > 1) {
        // Replay serially, one command per dispatch, then hand back an
        // already-completed ticket carrying the summed timing so callers
        // observe the usual asynchronous contract.
        CommandTiming total;
        for (const ComputeDispatch &dispatch : dispatches) {
            CommandTiming timing = submitAsync(dispatch).wait();
            impl_->dispatchProfile.push_back(
                {dispatch.pipelineName, timing.gpuSeconds});
            total.gpuSeconds += timing.gpuSeconds;
            total.wallSeconds += timing.wallSeconds;
            total.host.add(timing.host);
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = impl_->asyncState;
        ticketState->sequence = impl_->asyncState->beginSubmission(dispatches.size());
        ticketState->timing = total;
        ticketState->completed = true;
        if (completion) completion(ticketState->sequence);
        return CommandTicket(std::move(ticketState));
    }
    struct PreparedDispatch {
        const ComputeDispatch *source = nullptr;
        MTLSize groups{};
        MTLSize threads{};
        uint64_t threadCount = 0;
        __strong id<MTLComputePipelineState> pipeline = nil;
        std::vector<id<MTLResource>> readOnlyIndirectResources;
    };
    std::vector<PreparedDispatch> prepared;
    prepared.reserve(dispatches.size());
#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
    // Retain the exact views bound by every experimental projection. One head
    // can be used repeatedly in a command, and all tiles share sticky status.
    std::vector<MetalBuffer> experimentalDiagnostics;
    std::unordered_set<const void *> experimentalDiagnosticAddresses;
#endif
    for (const ComputeDispatch &dispatch : dispatches) {
        PreparedDispatch item;
        item.source = &dispatch;
        item.groups = metalSize(dispatch.threadgroups, "threadgroups");
        item.threads = metalSize(
            dispatch.threadsPerThreadgroup, "threadsPerThreadgroup");
        if (multiplyOverflows(dispatch.threadsPerThreadgroup.x,
                              dispatch.threadsPerThreadgroup.y) ||
            multiplyOverflows(dispatch.threadsPerThreadgroup.x *
                                  dispatch.threadsPerThreadgroup.y,
                              dispatch.threadsPerThreadgroup.z)) {
            throw MetalBackendError("threadsPerThreadgroup size overflows");
        }
        item.threadCount = dispatch.threadsPerThreadgroup.x *
            dispatch.threadsPerThreadgroup.y *
            dispatch.threadsPerThreadgroup.z;

        std::unordered_set<uint32_t> indices;
        std::unordered_set<const MetalAllocation *> indirectResources;
        for (const BufferBinding &binding : dispatch.buffers) {
            if (!binding.buffer.impl_ || !binding.buffer.impl_->allocation) {
                std::ostringstream message;
                message << "compute dispatch '" << dispatch.pipelineName
                        << "' contains an empty buffer at index "
                        << binding.index;
                throw MetalBackendError(message.str());
            }
            if (binding.buffer.impl_->allocation->accounting.get() !=
                impl_->accounting.get()) {
                throw MetalBackendError(
                    "compute dispatch buffer belongs to another backend");
            }
            if (!indices.insert(binding.index).second) {
                throw MetalBackendError("duplicate compute binding index");
            }
            for (const auto &source :
                 binding.buffer.impl_->allocation->readOnlyIndirectAllocations) {
                if (source->accounting.get() != impl_->accounting.get())
                    throw MetalBackendError("indirect compute resource belongs to another backend");
                if (indirectResources.insert(source.get()).second)
                    item.readOnlyIndirectResources.push_back(source->buffer);
            }
        }
        for (const BytesBinding &binding : dispatch.bytes) {
            if (!binding.data || !binding.sizeBytes) {
                throw MetalBackendError("compute byte binding is empty");
            }
            checkedNSUInteger(binding.sizeBytes, "byte binding size");
            if (!indices.insert(binding.index).second) {
                throw MetalBackendError("duplicate compute binding index");
            }
        }
#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
        const bool fp8Projection =
            dispatch.pipelineName.starts_with("experimental_fp8_decode_") ||
            dispatch.pipelineName.starts_with("experimental_fp8_prefill_");
        const bool int8Projection =
            dispatch.pipelineName.starts_with("experimental_int8_decode_") ||
            dispatch.pipelineName.starts_with("experimental_int8_prefill_");
        const bool int8Quantization =
            dispatch.pipelineName == "experimental_int8_row_scale" ||
            dispatch.pipelineName == "experimental_int8_quantize" ||
            dispatch.pipelineName == "experimental_int8_partial_peak512" ||
            dispatch.pipelineName == "experimental_int8_reduce_quantize512";
        if (fp8Projection || int8Projection || int8Quantization) {
            const uint32_t diagnosticIndex = int8Quantization ? 3
                                           : int8Projection ? 8 : 7;
            const auto diagnostic = std::find_if(
                dispatch.buffers.begin(), dispatch.buffers.end(),
                [diagnosticIndex](const BufferBinding &binding) {
                    return binding.index == diagnosticIndex;
                });
            if (diagnostic == dispatch.buffers.end() ||
                diagnostic->buffer.storage() != BufferStorage::Shared ||
                diagnostic->buffer.sizeBytes() < 4 * sizeof(uint32_t) ||
                !diagnostic->buffer.contents() ||
                diagnostic->buffer.impl_->offsetBytes % alignof(uint32_t)) {
                throw MetalBackendError(
                    "experimental projection requires aligned shared diagnostics");
            }
            if (experimentalDiagnosticAddresses.insert(
                    diagnostic->buffer.contents()).second)
                experimentalDiagnostics.push_back(diagnostic->buffer);
        }
#endif
        prepared.push_back(item);
    }

    std::lock_guard commandLock(impl_->commandMutex);
    impl_->ensureHealthy();
    static_cast<void>(impl_->reapSparseUnmapsLocked());
    for (PreparedDispatch &item : prepared) {
        item.pipeline = impl_->pipeline(item.source->pipelineName);
        if (item.threadCount >
            item.pipeline.maxTotalThreadsPerThreadgroup) {
            throw MetalBackendError(
                "threadsPerThreadgroup exceeds pipeline capability");
        }
    }

    std::shared_ptr<CounterProfileRecording> profileRecording;
    if (profileMode != CommandDispatchProfilingMode::Off) {
        profileRecording = std::make_shared<CounterProfileRecording>();
        auto &profile = profileRecording->profile;
        profile.mode = profileMode;
        profile.hostSubmissionStartSeconds = profileStart;
        profile.dispatchCount = prepared.size();
        const size_t metadataCount = std::min(prepared.size(),
            kMaxProfileCounterBuffers * kProfileDispatchesPerBuffer);
        profile.dispatchMetadataTruncated = metadataCount != prepared.size();
        if (profile.dispatchMetadataTruncated)
            profile.reason = "dispatch metadata truncated at 8192 entries";
        profile.dispatches.reserve(metadataCount);
        for (size_t i = 0; i < metadataCount; ++i) {
            const PreparedDispatch &item = prepared[i];
            const ComputeDispatch &dispatch = *item.source;
            CommandDispatchTimestamp metadata;
            metadata.index = profile.dispatches.size();
            metadata.pipelineName = dispatch.pipelineName;
            metadata.threadgroups = dispatch.threadgroups;
            metadata.threadsPerThreadgroup = dispatch.threadsPerThreadgroup;
            metadata.executionWidth = item.pipeline.threadExecutionWidth;
            metadata.maxTotalThreadsPerThreadgroup =
                item.pipeline.maxTotalThreadsPerThreadgroup;
            metadata.staticThreadgroupMemoryBytes =
                item.pipeline.staticThreadgroupMemoryLength;
            metadata.bindings.reserve(dispatch.buffers.size() + dispatch.bytes.size());
            for (const BufferBinding &binding : dispatch.buffers)
                metadata.bindings.push_back(
                    {binding.index, binding.buffer.sizeBytes(), false});
            for (const BytesBinding &binding : dispatch.bytes)
                metadata.bindings.push_back({binding.index, binding.sizeBytes, true});
            profile.dispatches.push_back(std::move(metadata));
        }
        // Only attachment zero is guaranteed when the timestamp set is the
        // device's sole counter set. Native sampling uses one sample buffer and
        // never splits the original encoder; creation validates its capacity.
        const size_t dispatchesPerBuffer =
            profileMode == CommandDispatchProfilingMode::DispatchBoundary
                ? prepared.size() : kProfileDispatchesPerBuffer;
        if (profileMode == CommandDispatchProfilingMode::Command) {
            profile.status = CommandDispatchProfileStatus::Complete;
        } else if (!impl_->profilingCapability.supports(profileMode)) {
            profile.status = CommandDispatchProfileStatus::Unsupported;
            profile.reason = !impl_->profilingCapability.timestampCounterSet
                ? "timestamp counter set unavailable"
                : profileMode == CommandDispatchProfilingMode::DispatchBoundary
                    ? "device does not support counter sampling at dispatch boundaries"
                    : "device does not support counter sampling at stage boundaries";
        } else if (prepared.size() >
                   kMaxProfileCounterBuffers * kProfileDispatchesPerBuffer) {
            profile.status = CommandDispatchProfileStatus::SampleLimitExceeded;
            profile.reason = "dispatch count exceeds profiling limit of 8192; metadata truncated";
        } else {
            for (size_t first = 0; first < prepared.size();
                 first += dispatchesPerBuffer) {
                MTLCounterSampleBufferDescriptor *descriptor =
                    [MTLCounterSampleBufferDescriptor new];
                descriptor.counterSet = impl_->timestampCounterSet;
                descriptor.storageMode = MTLStorageModeShared;
                const size_t count = std::min(dispatchesPerBuffer,
                                             prepared.size() - first);
                descriptor.sampleCount = count * 2;
                descriptor.label = @"Splash optional dispatch timestamps";
                NSError *error = nil;
                id<MTLCounterSampleBuffer> samples = [impl_->device
                    newCounterSampleBufferWithDescriptor:descriptor error:&error];
                if (!samples) {
                    profile.status = CommandDispatchProfileStatus::AllocationFailed;
                    profile.reason = "unable to create timestamp counter buffer: " +
                        errorDescription(error);
                    profileRecording->chunks.clear();
                    break;
                }
                profileRecording->chunks.push_back(
                    {samples, first, descriptor.sampleCount});
            }
            if (profileRecording->sampling()) {
                profile.encoderBoundariesAltered =
                    profileMode == CommandDispatchProfilingMode::StagePerDispatch;
                profile.samplingBarriers =
                    profileMode == CommandDispatchProfilingMode::DispatchBoundary;
            }
        }
    }

    auto ticketState = std::make_shared<CommandTicket::State>();
    ticketState->backend = impl_->asyncState;
    ticketState->completion = std::move(completion);
    ticketState->counterProfile = profileRecording;
    std::unordered_set<const MetalAllocation *> retained;
    for (const ComputeDispatch &dispatch : dispatches) {
        for (const BufferBinding &binding : dispatch.buffers) {
            const auto &allocation = binding.buffer.impl_->allocation;
            if (retained.insert(allocation.get()).second) {
                ticketState->retainedAllocations.push_back(allocation);
            }
            for (const auto &source : allocation->readOnlyIndirectAllocations)
                if (retained.insert(source.get()).second)
                    ticketState->retainedAllocations.push_back(source);
        }
    }
    ticketState->sequence = impl_->asyncState->beginSubmission(dispatches.size());
    if (profileRecording)
        profileRecording->profile.sequence = ticketState->sequence;

    auto failBeforeCommit = [&](std::string message) {
        impl_->markUnhealthy(message);
        impl_->asyncState->releaseSubmission(ticketState->sequence);
        throw MetalBackendError(std::move(message));
    };

    const double submissionStarted = steadySeconds();
    auto wallStart = std::chrono::steady_clock::now();
    ticketState->hostTiming.submissionStart = submissionStarted;
    ticketState->hostTiming.encodingStart =
        std::chrono::duration<double>(wallStart.time_since_epoch()).count();
    if (profileRecording) {
        profileRecording->profile.hostEncodingStartSeconds =
            std::chrono::duration<double>(wallStart.time_since_epoch()).count();
        profileRecording->profile.hostPreparationSeconds =
            profileRecording->profile.hostEncodingStartSeconds - profileStart;
        if (profileRecording->sampling())
            [impl_->device sampleTimestamps:
                &profileRecording->profile.calibrationCpuStart gpuTimestamp:
                &profileRecording->profile.calibrationGpuStart];
    }
    id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
    if (!command) {
        failBeforeCommit("unable to create Metal command buffer");
    }
    const uint64_t sparseEventValue = impl_->pendingSparseEventValue;
    ticketState->wallStart = wallStart;
    ticketState->sparseEventValue = sparseEventValue;
    if (sparseEventValue) {
        // Keep the queue dependency explicit; the CPU resolves it before commit.
        [command encodeWaitForEvent:impl_->sparseEvent value:sparseEventValue];
    }
    // Encoders can remain autoreleased after their command has completed.
    // The serving loop is long-lived, so bound their temporary ownership to
    // encoding; the command retains everything needed for GPU execution.
    @autoreleasepool {
        const bool sampled = profileRecording && profileRecording->sampling();
        const auto encodeDispatch = [&](id<MTLComputeCommandEncoder> encoder,
                                        const PreparedDispatch &item) {
            const ComputeDispatch &dispatch = *item.source;
            [encoder setComputePipelineState:item.pipeline];
            if (!item.readOnlyIndirectResources.empty())
                [encoder useResources:item.readOnlyIndirectResources.data()
                                count:item.readOnlyIndirectResources.size()
                                usage:MTLResourceUsageRead];
            for (const BufferBinding &binding : dispatch.buffers) {
                const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
                [encoder setBuffer:buffer.allocation->buffer
                            offset:checkedNSUInteger(buffer.offsetBytes,
                                                     "buffer offset")
                           atIndex:binding.index];
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                [encoder setBytes:binding.data
                           length:checkedNSUInteger(binding.sizeBytes,
                                                    "byte binding size")
                          atIndex:binding.index];
            }
            [encoder dispatchThreadgroups:item.groups
                     threadsPerThreadgroup:item.threads];
        };
        try {
            if (sampled && profileMode == CommandDispatchProfilingMode::StagePerDispatch) {
                for (size_t i = 0; i < prepared.size(); ++i) {
                    const auto &chunk = profileRecording->chunks[
                        i / kProfileDispatchesPerBuffer];
                    MTLComputePassDescriptor *descriptor =
                        [MTLComputePassDescriptor computePassDescriptor];
                    descriptor.dispatchType = MTLDispatchTypeSerial;
                    auto attachment = descriptor.sampleBufferAttachments[0];
                    attachment.sampleBuffer = chunk.buffer;
                    attachment.startOfEncoderSampleIndex =
                        (i - chunk.firstDispatch) * 2;
                    attachment.endOfEncoderSampleIndex =
                        (i - chunk.firstDispatch) * 2 + 1;
                    id<MTLComputeCommandEncoder> encoder =
                        [command computeCommandEncoderWithDescriptor:descriptor];
                    if (!encoder)
                        failBeforeCommit("unable to create sampled Metal compute encoder");
                    encodeDispatch(encoder, prepared[i]);
                    [encoder endEncoding];
                }
            } else {
                id<MTLComputeCommandEncoder> encoder = nil;
                if (sampled) {
                    MTLComputePassDescriptor *descriptor =
                        [MTLComputePassDescriptor computePassDescriptor];
                    descriptor.dispatchType = MTLDispatchTypeSerial;
                    for (size_t i = 0; i < profileRecording->chunks.size(); ++i) {
                        auto attachment = descriptor.sampleBufferAttachments[i];
                        attachment.sampleBuffer = profileRecording->chunks[i].buffer;
                        attachment.startOfEncoderSampleIndex = MTLCounterDontSample;
                        attachment.endOfEncoderSampleIndex = MTLCounterDontSample;
                    }
                    encoder = [command computeCommandEncoderWithDescriptor:descriptor];
                } else {
                    encoder = [command computeCommandEncoder];
                }
                if (!encoder)
                    failBeforeCommit("unable to create Metal compute encoder");
                if (sampled) {
                    for (size_t i = 0; i < prepared.size(); ++i) {
                        const auto &chunk = profileRecording->chunks.front();
                        const NSUInteger sample = (i - chunk.firstDispatch) * 2;
                        [encoder sampleCountersInBuffer:chunk.buffer
                                          atSampleIndex:sample withBarrier:YES];
                        encodeDispatch(encoder, prepared[i]);
                        [encoder sampleCountersInBuffer:chunk.buffer
                                          atSampleIndex:sample + 1 withBarrier:YES];
                    }
                } else {
                    for (const PreparedDispatch &item : prepared)
                        encodeDispatch(encoder, item);
                }
                [encoder endEncoding];
            }
        } catch (...) {
            impl_->asyncState->releaseSubmission(ticketState->sequence);
            throw;
        }
    }
    ticketState->hostTiming.encodingEnd = steadySeconds();
    if (profileRecording) {
        profileRecording->profile.hostEncodingEndSeconds = steadySeconds();
        profileRecording->profile.hostEncodingSeconds =
            profileRecording->profile.hostEncodingEndSeconds -
            profileRecording->profile.hostEncodingStartSeconds;
    }

    // Driver callbacks only complete the ticket. Device-wide memory telemetry
    // is sampled on the host before submission and when consuming the result.
    std::shared_ptr<BackendAsyncState> observer = impl_->asyncState;
    if (profileRecording) {
        [command addScheduledHandler:^(id<MTLCommandBuffer>) {
            const double scheduled = steadySeconds();
            ticketState->hostTiming.scheduled.store(scheduled, std::memory_order_release);
            profileRecording->scheduledSeconds.store(scheduled, std::memory_order_release);
        }];
    }
    [command addCompletedHandler:^(id<MTLCommandBuffer> completedCommand) {
        if (profileRecording) {
            const double completed = steadySeconds();
            ticketState->hostTiming.completed.store(completed, std::memory_order_release);
            const auto bridge = sampleMachSteadyClockBridge();
            {
                std::lock_guard lock(profileRecording->mutex);
                profileRecording->completedBridge = bridge;
            }
            resolveCounterProfileSafely(profileRecording, completedCommand, completed);
        }
        ticketState->finishCommand(completedCommand);
    }];
    sampleDeviceMemoryTimed(ticketState->hostTiming.preCommitMemory, profileRecording,
        &CounterProfileRecording::preCommitMemorySample,
        [&] { return impl_->sampleDeviceMemory(); });
    id<MTLSharedEvent> event = impl_->sparseEvent;
    const bool pendingMap =
        sparseEventValue && event.signaledValue < sparseEventValue;
    const double mapWaitStart = steadySeconds();
    if (pendingMap) {
        observer->mapWaitStarted.store(mapWaitStart, std::memory_order_relaxed);
        observer->mapWaitEvent.store(sparseEventValue, std::memory_order_release);
    }
    afterMetalEvent(event, sparseEventValue, kSparseMapTimeoutMilliseconds,
        [command, event, observer, ticketState, sparseEventValue,
         pendingMap, mapWaitStart, wallStart, profileRecording](bool signaled) {
            if (pendingMap) {
                const double waited = steadySeconds() - mapWaitStart;
                ticketState->hostTiming.dependencyWait.store(waited, std::memory_order_release);
                observer->lastMapWaitSeconds.store(waited, std::memory_order_relaxed);
                raisePeak(observer->maxMapWaitSeconds, waited);
                observer->mapWaitEvent.store(0, std::memory_order_release);
                if (profileRecording)
                    profileRecording->profile.sparseDependencyWaitSeconds = waited;
            }
            if (observer->stopping.stop_requested()) {
                ticketState->finish({}, "Metal backend stopped before command submission");
                return;
            }
            if (!signaled || !observer->healthy.load(std::memory_order_acquire)) {
                std::ostringstream message;
                message << "sparse mapping dependency failed before Metal command "
                        << ticketState->sequence << ": event " << sparseEventValue
                        << ", signaled " << event.signaledValue;
                if (!signaled)
                    message << ", wait exceeded "
                            << kSparseMapTimeoutMilliseconds << " ms";
                CommandTiming timing;
                timing.wallSeconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wallStart).count();
                ticketState->finish(timing, message.str());
                return;
            }
            if (profileRecording) {
                const auto bridge = sampleMachSteadyClockBridge();
                std::lock_guard lock(profileRecording->mutex);
                profileRecording->commitBridge = bridge;
                profileRecording->commitBeginSeconds.store(steadySeconds(),
                                                          std::memory_order_release);
            }
            const bool committed = observer->commitSubmission(ticketState->sequence, command,
                    ticketState->hostTiming,
                    [weakTicket = std::weak_ptr(ticketState)](id<MTLCommandBuffer> completed) {
                        if (auto ticket = weakTicket.lock()) ticket->finishCommand(completed);
                    });
            if (profileRecording)
                profileRecording->commitEndSeconds.store(steadySeconds(),
                                                         std::memory_order_release);
            if (!committed) {
                ticketState->finish({}, "Metal backend stopped before command submission");
                return;
            }
        }, observer->stopping.get_token());
    impl_->pendingSparseEventValue = 0;
    ticketState->hostTiming.submissionReturn.store(steadySeconds(), std::memory_order_release);
    return CommandTicket(std::move(ticketState));
}

MetalMemoryStats MetalBackend::memoryStats() const noexcept {
    // Reading MTLDevice.currentAllocatedSize can synchronize with an active
    // command on some Apple GPUs. Every allocation and command lifecycle
    // boundary already samples it, so status must use the cached atomic value
    // rather than turning a control-plane query into a GPU barrier.
    uint64_t deviceCurrent =
        impl_->asyncState->deviceCurrentAllocatedBytes.load(
            std::memory_order_relaxed);
    const uint64_t pendingUnmaps =
        impl_->pendingUnmapCount.load(std::memory_order_acquire);
    return {
        impl_->accounting->allocatedBytes.load(std::memory_order_relaxed),
        impl_->accounting->peakAllocatedBytes.load(std::memory_order_relaxed),
        deviceCurrent,
        impl_->asyncState->devicePeakAllocatedBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->sparseVirtualBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->sparseResidentBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->peakSparseResidentBytes.load(
            std::memory_order_relaxed),
        impl_->accounting->peakResidentBytes.load(std::memory_order_relaxed),
        kPlacementSparsePageBytes,
        pendingUnmaps,
        impl_->completedUnmaps.load(std::memory_order_relaxed),
        impl_->lastUnmapSeconds.load(std::memory_order_relaxed),
        impl_->maxUnmapSeconds.load(std::memory_order_relaxed),
        pendingUnmaps
            ? std::max(0.0, steadySeconds() -
                                impl_->pendingUnmapIssuedSeconds.load(
                                    std::memory_order_relaxed))
            : 0.0,
        impl_->asyncState->mapWaitEvent.load(std::memory_order_acquire),
        impl_->asyncState->mapWaitEvent.load(std::memory_order_acquire)
            ? std::max(0.0, steadySeconds() -
                impl_->asyncState->mapWaitStarted.load(std::memory_order_relaxed))
            : 0.0,
        impl_->asyncState->lastMapWaitSeconds.load(std::memory_order_relaxed),
        impl_->asyncState->maxMapWaitSeconds.load(std::memory_order_relaxed),
    };
}

MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
    {
        // A completed unmap releases its heap here without ever waiting
        // behind an active encode or mapping call.
        std::unique_lock lock(impl_->commandMutex, std::try_to_lock);
        if (lock.owns_lock())
            static_cast<void>(impl_->reapSparseUnmapsLocked());
    }
    impl_->sampleDeviceMemory();
    return memoryStats();
}

uint64_t MetalBackend::submissionCount() const noexcept {
    std::lock_guard lock(impl_->asyncState->gateMutex);
    return impl_->asyncState->nextSequence;
}

size_t MetalBackend::pipelineCount() const noexcept {
    std::lock_guard lock(impl_->commandMutex);
    return impl_->pipelines.count;
}

void MetalBackend::checkHealth() {
    impl_->asyncState->checkCommandHealth();
    if (impl_->pendingUnmapCount.load(std::memory_order_acquire)) {
        static_cast<void>(sparseUnmapPending());
        impl_->ensureHealthy();
    }
}

bool MetalBackend::needsHealthCheck() const noexcept {
    return impl_->asyncState->hasActiveSubmission() ||
           impl_->pendingUnmapCount.load(std::memory_order_acquire) != 0;
}

bool MetalBackend::healthy() const noexcept {
    return impl_->asyncState->healthy.load(std::memory_order_acquire);
}

std::string MetalBackend::unhealthyReason() const {
    std::lock_guard lock(impl_->asyncState->healthMutex);
    return impl_->asyncState->healthReason;
}

}  // namespace splash::metal
