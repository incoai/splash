#import "MetalBackend.hpp"
#include "CommandWatchdog.hpp"
#include "Residency.hpp"
#include "TestConfig.hpp"
#ifdef SPLASH_BACKEND_INSTRUMENTATION
#include "BackendInstrumentation.hpp"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string_view>
#include <unordered_map>
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

void readMacosVersion(DeviceCapabilities &capabilities) {
    const NSOperatingSystemVersion os =
        NSProcessInfo.processInfo.operatingSystemVersion;
    const auto component = [](NSInteger value) {
        return value > 0 ? static_cast<uint32_t>(value) : 0U;
    };
    capabilities.macosMajor = component(os.majorVersion);
    capabilities.macosMinor = component(os.minorVersion);
    capabilities.macosPatch = component(os.patchVersion);
}

// The backend and probeDeviceCapabilities() share one reading of the device,
// so the probe judges a Mac by the values the engine validates.
void readDeviceCapabilities(id<MTLDevice> device,
                            DeviceCapabilities &capabilities) {
    capabilities.deviceName = stringFromNSString(device.name);
    capabilities.gpuCoreCount = gpuCoreCountForDevice(device.registryID);
    // Apple GPU families nest, so the device's is the last one supported
    // counting up from Apple7.
    uint32_t family = 0;
    for (uint32_t next = 7;
         [device supportsFamily:static_cast<MTLGPUFamily>(1000 + next)]; ++next)
        family = next;
    capabilities.appleGpuFamily = family;
    capabilities.physicalMemoryBytes = NSProcessInfo.processInfo.physicalMemory;
    capabilities.recommendedMaxWorkingSetBytes =
        device.recommendedMaxWorkingSetSize;
    capabilities.maxBufferLengthBytes = device.maxBufferLength;
    capabilities.maxThreadgroupMemoryBytes = device.maxThreadgroupMemoryLength;
    MTLSize maximumThreads = device.maxThreadsPerThreadgroup;
    capabilities.maxThreadgroupWidth = maximumThreads.width;
    capabilities.hasUnifiedMemory = device.hasUnifiedMemory;
}

// Every uint64_t size, offset and length passes to Metal as it is.
static_assert(sizeof(NSUInteger) == sizeof(uint64_t),
              "Splash builds for arm64 only");

MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(size.x, size.y, size.z);
}

bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

// Entries of a kernel's buffer argument table on every Apple GPU family.
constexpr uint32_t kBufferArgumentEntries = 31;
// How long a ticket waits for its command before it asks the watchdog.
constexpr auto kTicketWaitSlice = std::chrono::seconds(1);
// The queue's outstanding command buffers; a command takes one per event
// signal and one more.
constexpr NSUInteger kMaximumCommandBuffers = 512;

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

template <typename T>
void raisePeak(std::atomic<T> &peak, T value) noexcept {
    T current = peak.load(std::memory_order_relaxed);
    while (value > current &&
           !peak.compare_exchange_weak(current, value,
                                       std::memory_order_relaxed)) {}
}

// Hashes pipeline names as views, so a cache lookup builds no string.
struct PipelineNameHash {
    using is_transparent = void;
    size_t operator()(std::string_view name) const noexcept {
        return std::hash<std::string_view>{}(name);
    }
};

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
};

struct MetalAllocation {
    // Own the host mapping for our views as well as the Metal deallocator.
    // Validation wrappers may not retain the supplied deallocator block.
    std::shared_ptr<void> externalOwner;
    __strong id<MTLBuffer> buffer = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    uint64_t bytes = 0;
    BufferStorage storage = BufferStorage::Shared;
    // The residency set the buffer belongs to, held weakly as allocations
    // may outlive the backend. The set retains the buffer, and with it its
    // memory, so the last view takes it out.
    std::weak_ptr<Residency> residency;

    ~MetalAllocation() {
        if (auto kept = residency.lock()) kept->remove(buffer);
        if (accounting && bytes) {
            accounting->allocatedBytes.fetch_sub(
                bytes, std::memory_order_relaxed);
        }
    }
};

// Every Impl has an allocation with a buffer: only registerBuffer and view
// create one.
struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct SharedEvent::Impl {
    __strong id<MTLSharedEvent> event = nil;
};

struct BackendAsyncState {
    explicit BackendAsyncState(double commandTimeoutSeconds)
        : commandWatchdog(commandTimeoutSeconds) {}

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
    bool stopping = false;
    // MetalBackend::setWaitInterrupt's predicate.
    std::function<bool()> waitInterrupt;

    void sampleDeviceMemory() const noexcept {
        if (!device) return;
        uint64_t current = static_cast<uint64_t>(device.currentAllocatedSize);
        deviceCurrentAllocatedBytes.store(current, std::memory_order_relaxed);
        raisePeak(devicePeakAllocatedBytes, current);
    }

    [[noreturn]] void throwUnhealthy() const {
        std::lock_guard lock(healthMutex);
        throw MetalBackendError("Metal backend is unhealthy: " + healthReason);
    }

    void ensureHealthy() const {
        if (!healthy.load(std::memory_order_acquire)) throwUnhealthy();
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
        if (stopping)
            throw MetalBackendError("Metal backend is stopping");
        if (activeSequence) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        activeSequence = ++nextSequence;
        activeDispatchCount = dispatchCount;
        return activeSequence;
    }

    void commitSubmission(uint64_t sequence, const std::vector<id<MTLCommandBuffer>> &leading,
                          id<MTLCommandBuffer> command,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        activeCommand = command;
        activeCompletion = std::move(completion);
        commandWatchdog.start(sequence, steadySeconds());
        for (id<MTLCommandBuffer> earlier : leading) [earlier commit];
        [command commit];
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
        if (activeSequence == sequence) {
            activeSequence = 0;
            activeCommand = nil;
            activeCompletion = {};
        }
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
    }

    // Runs the command watchdog. A terminal command whose callback is late
    // is completed here; one still running past its timeout marks the
    // backend unhealthy, and the answer is then true: the backend gave up on
    // that command.
    [[nodiscard]] bool commandAbandoned() noexcept {
        id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> complete;
        {
            std::lock_guard lock(gateMutex);
            if (!commandWatchdog.expired(steadySeconds())) return false;
            command = activeCommand;
            const auto status = command ? command.status
                                        : MTLCommandBufferStatusNotEnqueued;
            // Recover terminal results even if the driver has not delivered
            // its callback. Finish outside the gate: it takes the ticket lock.
            if (command && (status == MTLCommandBufferStatusCompleted ||
                            status == MTLCommandBufferStatusError)) {
                complete = activeCompletion;
            } else {
                // Waits that must not throw run this too: the reason drops
                // its details when they cannot be formatted.
                std::string reason = "Metal command completion timed out";
                try {
                    std::ostringstream message;
                    message << reason << " after "
                            << commandWatchdog.timeoutSeconds()
                            << " seconds (sequence=" << activeSequence
                            << ", status=" << (command ? commandStatusName(status)
                                                       : "unavailable")
                            << ", dispatches=" << activeDispatchCount << ')';
                    reason = message.str();
                } catch (const std::bad_alloc &) {
                }
                markUnhealthy(std::move(reason));
                return true;
            }
        }
        if (complete) complete(command);
        return false;
    }

    void checkCommandHealth() {
        static_cast<void>(commandAbandoned());
        ensureHealthy();
    }

    // True when the process is shutting down; the waiter decides whether
    // that gives its command up.
    [[nodiscard]] bool waitInterrupted() const noexcept {
        return waitInterrupt && waitInterrupt();
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
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    std::chrono::steady_clock::time_point wallStart;
    // Command buffers committed before the last one, split at event signals.
    std::vector<id<MTLCommandBuffer>> leadingCommands;
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        auto wallEnd = std::chrono::steady_clock::now();
        CommandTiming timing;
        const double gpuStart = leadingCommands.empty()
            ? command.GPUStartTime : leadingCommands.front().GPUStartTime;
        timing.gpuSeconds = command.GPUEndTime - gpuStart;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();

        std::string error;
        id<MTLCommandBuffer> failed = command.status != MTLCommandBufferStatusCompleted ? command : nil;
        for (id<MTLCommandBuffer> earlier : leadingCommands) {
            if (earlier.status != MTLCommandBufferStatusCompleted) {
                failed = earlier;
                break;
            }
        }
        if (failed) {
            std::ostringstream message;
            message << "Metal command " << sequence << " failed";
            if (failed.error) {
                message << ": " << errorDescription(failed.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
    }

    void finish(CommandTiming result, std::string failure) {
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
            completed = true;
            notify = completion;
        }
        if (notify) {
            try {
                notify();
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
            // Refresh the cached device telemetry, which status reports, on
            // the consuming thread after GPU completion; admission samples
            // its own (refreshMemoryStats).
            if (backend->healthy.load(std::memory_order_acquire))
                backend->sampleDeviceMemory();
            backend->releaseSubmission(sequence);
        }
    }

    // Waits for the command in kTicketWaitSlice slices. Between them, outside
    // `mutex` (the watchdog may finish this ticket through finishCommand), it
    // asks the backend whether to stop, and with honorShutdown also whether
    // the process is shutting down. False when the backend gave up on a
    // command that never completed: the GPU may still use the retained
    // allocations, which the command's completion handler keeps alive with
    // this state.
    [[nodiscard]] bool awaitCompletion(bool honorShutdown) noexcept {
        std::unique_lock lock(mutex);
        while (!condition.wait_for(lock, kTicketWaitSlice,
                                   [this] { return completed; })) {
            lock.unlock();
            const bool abandoned = backend->commandAbandoned();
            const bool interrupted =
                !abandoned && honorShutdown && backend->waitInterrupted();
            lock.lock();
            if (completed) break;
            // A shutdown gives the command up only here, where the lock shows
            // it unfinished: one that completed meanwhile leaves the backend
            // healthy.
            if (interrupted) backend->markUnhealthy(shutdownReason());
            if (abandoned || interrupted) return false;
        }
        return true;
    }

    // Why the backend is unhealthy once a shutdown gave this command up.
    [[nodiscard]] std::string shutdownReason() const noexcept {
        std::string reason = "shutdown requested while waiting for a Metal command";
        try {
            reason = "shutdown requested while waiting for Metal command " +
                     std::to_string(sequence);
        } catch (const std::bad_alloc &) {
        }
        return reason;
    }

    // An abandoned command keeps its allocations until its completion
    // handler lets go of this state; the unhealthy backend admits no more.
    // A shutdown does not give up this wait: teardown waits for the command,
    // as long as the watchdog lets it.
    void abandon() noexcept {
        if (awaitCompletion(false)) release();
    }
};

struct MetalBackend::Impl {
    // One dispatch of a command, validated and resolved to its pipeline.
    struct PreparedDispatch {
        const ComputeDispatch *source = nullptr;
        MTLSize groups{};
        MTLSize threads{};
        uint64_t threadCount = 0;
        // The argument table entries its buffers take, one bit each.
        uint32_t bufferIndices = 0;
        __strong id<MTLComputePipelineState> pipeline = nil;
    };

    // A command as submission encodes it: its dispatches and every
    // allocation they bind, which its ticket retains.
    struct PreparedCommand {
        std::vector<PreparedDispatch> dispatches;
        std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    };

    std::function<void()> operationGuard;

#ifdef SPLASH_BACKEND_INSTRUMENTATION
    bool dispatchProfiling = false;
    std::vector<DispatchTiming> dispatchProfile;
#endif
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    // Allocations hold it weakly: they may outlive the backend.
    std::shared_ptr<Residency> residency;
    __strong id<MTLLibrary> library = nil;
    // Looked up for every dispatch when its command is prepared, which the
    // GPU may be waiting for; a hit allocates nothing. Used only by the
    // submitting thread.
    std::unordered_map<std::string, id<MTLComputePipelineState>,
                       PipelineNameHash, std::equal_to<>>
        pipelines;

    DeviceCapabilities capabilities;
    std::shared_ptr<AllocationAccounting> accounting =
        std::make_shared<AllocationAccounting>();
    std::shared_ptr<BackendAsyncState> asyncState =
        std::make_shared<BackendAsyncState>(
            testConfig().commandTimeoutSeconds.value_or(kCommandTimeoutSeconds));

    void sampleDeviceMemory() const noexcept {
        asyncState->sampleDeviceMemory();
    }

    void ensureHealthy() const {
        asyncState->ensureHealthy();
    }

    void markUnhealthy(std::string reason) {
        asyncState->markUnhealthy(std::move(reason));
    }

    MetalBuffer registerBuffer(id<MTLBuffer> buffer, BufferStorage storage,
                               std::shared_ptr<void> externalOwner = {}) {
        auto allocation = std::make_shared<MetalAllocation>();
        allocation->externalOwner = std::move(externalOwner);
        allocation->buffer = buffer;
        allocation->accounting = accounting;
        allocation->bytes = buffer.allocatedSize;
        allocation->storage = storage;
        residency->add(buffer);
        allocation->residency = residency;
        raisePeak(accounting->peakAllocatedBytes,
                  accounting->allocatedBytes.fetch_add(
                      allocation->bytes, std::memory_order_relaxed) +
                      allocation->bytes);
        sampleDeviceMemory();
        auto result = std::make_shared<MetalBuffer::Impl>();
        result->lengthBytes = buffer.length;
        result->allocation = std::move(allocation);
        return MetalBuffer(std::move(result));
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        if (const auto cached = pipelines.find(name); cached != pipelines.end())
            return cached->second;
        id<MTLComputePipelineState> result = newPipeline(name);
        pipelines.emplace(name, result);
        sampleDeviceMemory();
        return result;
    }

    id<MTLComputePipelineState> newPipeline(std::string_view name) {
        NSString *key = checkedNSString(name, "pipeline name");
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
        return result;
    }

    PreparedCommand prepare(std::span<const ComputeDispatch> dispatches) {
        if (dispatches.empty()) {
            throw MetalBackendError("Metal command must contain a dispatch");
        }
        const auto signals = std::count_if(dispatches.begin(), dispatches.end(),
            [](const ComputeDispatch &dispatch) { return dispatch.event && dispatch.event->signal; });
        if (static_cast<NSUInteger>(signals) >= kMaximumCommandBuffers) {
            throw MetalBackendError("Metal command signals more events than its queue holds command buffers");
        }
        PreparedCommand command;
        command.dispatches.reserve(dispatches.size());
        size_t bindings = 0;
        for (const ComputeDispatch &dispatch : dispatches) {
            PreparedDispatch item;
            item.source = &dispatch;
            if (dispatch.event) {
                if (!dispatch.event->event || !dispatch.event->value) {
                    throw MetalBackendError("event step requires an event and a value");
                }
                command.dispatches.push_back(item);
                continue;
            }
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

            // Each binding takes its own entry of the argument table.
            uint32_t indices = 0;
            const auto claim = [&](uint32_t index) {
                if (index >= kBufferArgumentEntries) {
                    throw MetalBackendError(
                        "compute binding index exceeds the argument table");
                }
                if (indices & (uint32_t{1} << index)) {
                    throw MetalBackendError("duplicate compute binding index");
                }
                indices |= uint32_t{1} << index;
            };
            for (const BufferBinding &binding : dispatch.buffers) {
                if (!binding.buffer.impl_) {
                    std::ostringstream message;
                    message << "compute dispatch '" << dispatch.pipelineName
                            << "' contains an empty buffer at index "
                            << binding.index;
                    throw MetalBackendError(message.str());
                }
                if (binding.buffer.impl_->allocation->accounting.get() !=
                    accounting.get()) {
                    throw MetalBackendError(
                        "compute dispatch buffer belongs to another backend");
                }
                claim(binding.index);
                item.bufferIndices |= uint32_t{1} << binding.index;
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                if (!binding.data || !binding.sizeBytes) {
                    throw MetalBackendError("compute byte binding is empty");
                }
                claim(binding.index);
            }
            bindings += dispatch.buffers.size();
            command.dispatches.push_back(item);
        }

        for (PreparedDispatch &item : command.dispatches) {
            if (item.source->event) continue;
            item.pipeline = pipeline(item.source->pipelineName);
            if (item.threadCount >
                item.pipeline.maxTotalThreadsPerThreadgroup) {
                throw MetalBackendError(
                    "threadsPerThreadgroup exceeds pipeline capability");
            }
        }

        // Each allocation the command binds, once.
        std::vector<const std::shared_ptr<MetalAllocation> *> bound;
        bound.reserve(bindings);
        for (const ComputeDispatch &dispatch : dispatches) {
            for (const BufferBinding &binding : dispatch.buffers)
                bound.push_back(&binding.buffer.impl_->allocation);
        }
        const auto allocation =
            [](const std::shared_ptr<MetalAllocation> *owner) {
                return owner->get();
            };
        std::ranges::sort(bound, {}, allocation);
        const auto repeated = std::ranges::unique(bound, {}, allocation);
        bound.erase(repeated.begin(), repeated.end());
        command.retainedAllocations.reserve(bound.size());
        for (const std::shared_ptr<MetalAllocation> *owner : bound)
            command.retainedAllocations.push_back(*owner);
        return command;
    }

    // Encodes and commits prepared dispatches; the ticket retains
    // `retained` until it is consumed.
    CommandTicket commit(std::span<const PreparedDispatch> dispatches,
                         std::vector<std::shared_ptr<MetalAllocation>> retained,
                         CommandCompletion completion) {
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        ticketState->retainedAllocations = std::move(retained);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size());

        auto failBeforeCommit = [&](std::string message) {
            markUnhealthy(message);
            asyncState->releaseSubmission(ticketState->sequence);
            throw MetalBackendError(std::move(message));
        };

        auto wallStart = std::chrono::steady_clock::now();
        // Metal may autorelease the command and its encoder, and the serving
        // loop's pool never drains, so their temporary ownership ends with
        // this submission (under the validation layer an autoreleased
        // command holds every member of the residency set). The command
        // retains everything the GPU still needs.
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) {
                failBeforeCommit("unable to create Metal command buffer");
            }
            ticketState->wallStart = wallStart;
            // An event signal ends a command buffer; the rest continue in the
            // next.
            std::vector<id<MTLCommandBuffer>> leading;
            id<MTLComputeCommandEncoder> encoder = nil;
            // Indexed by argument table entry. The ticket and the dispatches
            // keep the buffers alive.
            __unsafe_unretained id<MTLBuffer> buffers[kBufferArgumentEntries];
            NSUInteger offsets[kBufferArgumentEntries];
            for (const PreparedDispatch &item : dispatches) {
                const ComputeDispatch &dispatch = *item.source;
                if (dispatch.event) {
                    if (encoder) {
                        [encoder endEncoding];
                        encoder = nil;
                    }
                    id<MTLSharedEvent> event =
                        (__bridge id<MTLSharedEvent>)dispatch.event->event.nativeHandle();
                    if (!dispatch.event->signal) {
                        [command encodeWaitForEvent:event value:dispatch.event->value];
                        continue;
                    }
                    [command encodeSignalEvent:event value:dispatch.event->value];
                    leading.push_back(command);
                    command = [queue commandBuffer];
                    if (!command) {
                        failBeforeCommit("unable to create Metal command buffer");
                    }
                    continue;
                }
                if (!encoder) {
                    encoder = [command computeCommandEncoder];
                    if (!encoder) {
                        failBeforeCommit("unable to create Metal compute encoder");
                    }
                }
                [encoder setComputePipelineState:item.pipeline];
                for (const BufferBinding &binding : dispatch.buffers) {
                    const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
                    buffers[binding.index] = buffer.allocation->buffer;
                    offsets[binding.index] = buffer.offsetBytes;
                }
                // One call per run of consecutive entries: a command graph's
                // dispatch binds a single run.
                for (uint32_t unbound = item.bufferIndices; unbound;) {
                    const uint32_t first = std::countr_zero(unbound);
                    const uint32_t count = std::countr_one(unbound >> first);
                    [encoder setBuffers:buffers + first
                                offsets:offsets + first
                              withRange:NSMakeRange(first, count)];
                    unbound &= ~(((uint32_t{1} << count) - 1) << first);
                }
                for (const BytesBinding &binding : dispatch.bytes) {
                    [encoder setBytes:binding.data
                               length:binding.sizeBytes
                              atIndex:binding.index];
                }
                [encoder dispatchThreadgroups:item.groups
                         threadsPerThreadgroup:item.threads];
            }
            if (encoder) [encoder endEncoding];
            ticketState->leadingCommands = leading;

            // Driver callbacks only complete the ticket. Device-wide memory
            // telemetry is sampled on the host when consuming the result. The
            // handler holds the ticket's state strongly: once a waiter gives
            // up on the command, it keeps the retained allocations until the
            // GPU ends.
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                ticketState->finishCommand(completed);
            }];
            residency->use();
            asyncState->commitSubmission(ticketState->sequence, leading, command,
                [weakTicket = std::weak_ptr(ticketState)](
                    id<MTLCommandBuffer> completed) {
                    if (auto ticket = weakTicket.lock())
                        ticket->finishCommand(completed);
                });
        }
        return CommandTicket(std::move(ticketState));
    }

#ifdef SPLASH_BACKEND_INSTRUMENTATION
    // Commits every dispatch of the command as its own command and waits
    // for it, then hands back an already-completed ticket with the summed
    // timing, so callers observe the usual asynchronous contract.
    CommandTicket submitProfiled(std::span<const ComputeDispatch> dispatches,
                                 CommandCompletion completion) {
        if (std::any_of(dispatches.begin(), dispatches.end(),
                [](const ComputeDispatch &dispatch) { return dispatch.event.has_value(); })) {
            throw MetalBackendError("dispatch profiling cannot replay event steps");
        }
        const PreparedCommand command = prepare(dispatches);
        CommandTiming total;
        for (const PreparedDispatch &item : command.dispatches) {
            const CommandTiming timing =
                commit({&item, 1}, command.retainedAllocations, {}).wait();
            dispatchProfile.push_back(
                {item.source->pipelineName, timing.gpuSeconds});
            total.gpuSeconds += timing.gpuSeconds;
            total.wallSeconds += timing.wallSeconds;
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->sequence =
            asyncState->beginSubmission(command.dispatches.size());
        ticketState->timing = total;
        ticketState->completed = true;
        if (completion) completion();
        return CommandTicket(std::move(ticketState));
    }
#endif
};

SharedEvent::SharedEvent() = default;
SharedEvent::~SharedEvent() = default;
SharedEvent::SharedEvent(const SharedEvent &) = default;
SharedEvent &SharedEvent::operator=(const SharedEvent &) = default;
SharedEvent::SharedEvent(SharedEvent &&) noexcept = default;
SharedEvent &SharedEvent::operator=(SharedEvent &&) noexcept = default;
SharedEvent::SharedEvent(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

SharedEvent::operator bool() const noexcept { return impl_ && impl_->event; }

void *SharedEvent::nativeHandle() const noexcept {
    return impl_ ? (__bridge void *)impl_->event : nullptr;
}

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;

MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalBuffer::operator bool() const noexcept {
    return impl_ != nullptr;
}

uint64_t MetalBuffer::sizeBytes() const noexcept {
    return impl_ ? impl_->lengthBytes : 0;
}

uint64_t MetalBuffer::allocatedBytes() const noexcept {
    return impl_ ? impl_->allocation->bytes : 0;
}

bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
    if (impl_ == other.impl_) return true;
    return impl_ && other.impl_ &&
           impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes == other.impl_->offsetBytes &&
           impl_->lengthBytes == other.impl_->lengthBytes;
}

BufferStorage MetalBuffer::storage() const noexcept {
    return impl_ ? impl_->allocation->storage : BufferStorage::Shared;
}

void *MetalBuffer::contents() const noexcept {
    if (!impl_ || impl_->allocation->storage != BufferStorage::Shared) {
        return nullptr;
    }
    return static_cast<uint8_t *>(impl_->allocation->buffer.contents) +
           impl_->offsetBytes;
}

uint64_t MetalBuffer::gpuAddress() const noexcept {
    if (!impl_) return 0;
    return impl_->allocation->buffer.gpuAddress + impl_->offsetBytes;
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

bool CommandTicket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

CommandTiming CommandTicket::wait() {
    if (!state_) throw MetalBackendError("Metal command ticket is empty");
    if (!state_->awaitCompletion(true)) {
        // Let go first, so that unwinding does not wait again.
        auto backend = state_->backend;
        state_.reset();
        backend->throwUnhealthy();
    }
    CommandTiming timing;
    std::string error;
    {
        std::lock_guard lock(state_->mutex);
        timing = state_->timing;
        error = state_->error;
    }
    state_->release();
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
}

MetalBackend::MetalBackend(std::string metallibPath)
    : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first.
        readMacosVersion(impl_->capabilities);
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
        // A command splits into one Metal command buffer per event signal
        // (EventStep), all created before any commits: the default limit of
        // 64 outstanding would block a 64-layer prefill with an ANE step each.
        impl_->queue = [impl_->device
            newCommandQueueWithMaxCommandBufferCount:kMaximumCommandBuffers];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }

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
        // The library keeps the bytes read here, whatever later replaces the
        // path; the dispatch data retains them rather than copying them.
        dispatch_data_t data = dispatch_data_create(
            fileData.bytes, fileData.length, nullptr, ^{ (void)fileData; });
        error = nil;
        impl_->library =
            [impl_->device newLibraryWithData:data error:&error];
        if (!impl_->library) {
            throw MetalBackendError(
                "unable to load metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // Ending residency dispatches a kernel built here, so no pipeline or
        // driver program is compiled when a keep-alive lapses.
        impl_->residency = std::make_shared<Residency>(
            impl_->device, impl_->queue,
            impl_->newPipeline(Residency::kKickPipeline),
            testConfig().residencyKeepAliveSeconds.value_or(
                kResidencyKeepAliveSeconds));

        readDeviceCapabilities(impl_->device, impl_->capabilities);
    }
    impl_->sampleDeviceMemory();
}

MetalBackend::~MetalBackend() = default;

void MetalBackend::stop() noexcept {
    std::lock_guard lock(impl_->asyncState->gateMutex);
    impl_->asyncState->stopping = true;
}

const DeviceCapabilities &MetalBackend::capabilities() const noexcept {
    return impl_->capabilities;
}

DeviceCapabilities probeDeviceCapabilities() {
    @autoreleasepool {
        DeviceCapabilities capabilities;
        readMacosVersion(capabilities);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw MetalBackendError("Metal device unavailable");
        readDeviceCapabilities(device, capabilities);
        return capabilities;
    }
}

void MetalBackend::checkOperation() const {
    impl_->ensureHealthy();
    if (impl_->operationGuard) impl_->operationGuard();
}

void MetalBackend::setOperationGuard(std::function<void()> guard) {
    impl_->operationGuard = std::move(guard);
}

void MetalBackend::setWaitInterrupt(std::function<bool()> shuttingDown) {
    impl_->asyncState->waitInterrupt = std::move(shuttingDown);
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
        newBufferWithLength:bytes
        options:options];
    if (!buffer) throw MetalAllocationError("Metal buffer allocation failed");
    if (!label.empty()) buffer.label = checkedNSString(label, "buffer label");
    return impl_->registerBuffer(buffer, storage);
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
        length:bytes
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
    if (!base.impl_) {
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

SharedEvent MetalBackend::newSharedEvent() {
    checkOperation();
    auto result = std::make_shared<SharedEvent::Impl>();
    result->event = [impl_->device newSharedEvent];
    if (!result->event) {
        throw MetalBackendError("unable to create Metal shared event");
    }
    return SharedEvent(std::move(result));
}

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
    return submitAsync(dispatch).wait();
}

CommandTiming MetalBackend::submitCommand(
    std::span<const ComputeDispatch> dispatches) {
    return submitCommandAsync(dispatches).wait();
}

CommandTicket MetalBackend::submitAsync(const ComputeDispatch &dispatch) {
    return submitCommandAsync(std::span<const ComputeDispatch>(&dispatch, 1));
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion) {
    checkOperation();
#ifdef SPLASH_BACKEND_INSTRUMENTATION
    if (impl_->dispatchProfiling)
        return impl_->submitProfiled(dispatches, std::move(completion));
#endif
    Impl::PreparedCommand command = impl_->prepare(dispatches);
    return impl_->commit(command.dispatches,
                         std::move(command.retainedAllocations),
                         std::move(completion));
}

void MetalBackend::preparePipelines(
    std::span<const ComputeDispatch> dispatches) {
    checkOperation();
    static_cast<void>(impl_->prepare(dispatches));
}

MetalMemoryStats MetalBackend::memoryStats() const noexcept {
    // Reading MTLDevice.currentAllocatedSize can synchronize with an active
    // command on some Apple GPUs. Every allocation and command lifecycle
    // boundary already samples it, so status must use the cached atomic value
    // rather than turning a control-plane query into a GPU barrier.
    return {
        impl_->accounting->allocatedBytes.load(std::memory_order_relaxed),
        impl_->accounting->peakAllocatedBytes.load(std::memory_order_relaxed),
        impl_->asyncState->deviceCurrentAllocatedBytes.load(
            std::memory_order_relaxed),
        impl_->asyncState->devicePeakAllocatedBytes.load(
            std::memory_order_relaxed),
    };
}

MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
    impl_->sampleDeviceMemory();
    return memoryStats();
}

bool MetalBackend::commandInFlight() const noexcept {
    return impl_->asyncState->hasActiveSubmission();
}

void MetalBackend::checkHealth() {
    impl_->asyncState->checkCommandHealth();
}

bool MetalBackend::healthy() const noexcept {
    return impl_->asyncState->healthy.load(std::memory_order_acquire);
}

std::string MetalBackend::unhealthyReason() const {
    std::lock_guard lock(impl_->asyncState->healthMutex);
    return impl_->asyncState->healthReason;
}

#ifdef SPLASH_BACKEND_INSTRUMENTATION
uint64_t BackendInstrumentation::submittedCommands(
    const MetalBackend &backend) {
    std::lock_guard lock(backend.impl_->asyncState->gateMutex);
    return backend.impl_->asyncState->nextSequence;
}

size_t BackendInstrumentation::cachedPipelines(const MetalBackend &backend) {
    return backend.impl_->pipelines.size();
}

void BackendInstrumentation::setDispatchProfiling(MetalBackend &backend,
                                                  bool enabled) {
    backend.impl_->dispatchProfiling = enabled;
}

std::vector<DispatchTiming>
BackendInstrumentation::takeDispatchProfile(MetalBackend &backend) {
    return std::exchange(backend.impl_->dispatchProfile, {});
}
#endif

}  // namespace splash::metal
