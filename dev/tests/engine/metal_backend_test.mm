#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/metal/DeviceQueries.hpp"
#include "../../../runtime/ops/PagedKv.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <CommonCrypto/CommonDigest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using splash::metal::AllocationFailure;
using splash::metal::MetalAllocationError;
using splash::metal::BufferBinding;
using splash::metal::BufferStorage;
using splash::metal::BytesBinding;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBackendError;
using splash::metal::MetalBuffer;
using splash::metal::SparseMapping;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void awaitSparseRelease(MetalBackend &backend, uint64_t residentBytes,
                        std::optional<uint64_t> virtualBytes = std::nullopt) {
    // The driver event and its resource-retention callback complete separately.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (true) {
        const auto stats = backend.memoryStats();
        if (stats.sparseResidentBytes == residentBytes &&
            (!virtualBytes || stats.sparseVirtualBytes == *virtualBytes)) return;
        require(std::chrono::steady_clock::now() < deadline,
                "completed sparse mapping did not release retained resources");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

struct TemporaryMetallib final {
    TemporaryMetallib()
        : path((std::filesystem::temp_directory_path() /
                "splash-metal-backend.XXXXXX").string()) {
        const int descriptor = ::mkstemp(path.data());
        if (descriptor < 0)
            throw std::runtime_error("could not create temporary metallib");
        ::close(descriptor);
    }
    ~TemporaryMetallib() { ::unlink(path.c_str()); }
    std::string path;
};

std::array<uint8_t, 32> libraryDigest(NSData *data) {
    require(data && data.length &&
                data.length <= std::numeric_limits<CC_LONG>::max(),
            "invalid test library data");
    std::array<uint8_t, 32> digest{};
    require(CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length),
                      digest.data()) != nullptr,
            "test library SHA-256 failed");
    return digest;
}

template <typename Function>
void requireBackendError(Function &&function, const std::string &message) {
    try {
        function();
    } catch (const MetalBackendError &) {
        return;
    }
    fail(message);
}

class MethodReplacement final {
public:
    MethodReplacement(id object, SEL selector, IMP replacement) {
        method_ = class_getInstanceMethod(object_getClass(object), selector);
        require(method_ != nullptr, "probe fault method is missing");
        original = method_setImplementation(method_, replacement);
    }
    ~MethodReplacement() { method_setImplementation(method_, original); }
    IMP original = nullptr;
private:
    Method method_ = nullptr;
};

thread_local bool inCompletionHandler = false;
IMP originalCompletedHandler = nullptr;
IMP originalAllocatedSize = nullptr;
std::promise<void> completedOnGpu;
std::promise<void> completionReturned;
std::shared_future<void> releaseMemoryQuery;
std::atomic<unsigned> completionMemoryQueries{0};
std::atomic<unsigned> memoryQueries{0};

void observeCompletion(id command, SEL selector, MTLCommandBufferHandler handler) {
    reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
        originalCompletedHandler)(command, selector, ^(id<MTLCommandBuffer> completed) {
        completedOnGpu.set_value();
        inCompletionHandler = true;
        handler(completed);
        inCompletionHandler = false;
        completionReturned.set_value();
    });
}

NSUInteger delayedCompletionMemoryQuery(id device, SEL selector) {
    ++memoryQueries;
    if (inCompletionHandler) {
        ++completionMemoryQueries;
        releaseMemoryQuery.wait();
    }
    return reinterpret_cast<NSUInteger (*)(id, SEL)>(originalAllocatedSize)(device, selector);
}

void completionDoesNotWaitForMemoryTelemetry(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, 0.1);
    auto buffer = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    std::promise<void> release;
    releaseMemoryQuery = release.get_future().share();
    auto gpuDone = completedOnGpu.get_future();
    auto callbackDone = completionReturned.get_future();
    MethodReplacement completion(command, @selector(addCompletedHandler:),
                                 reinterpret_cast<IMP>(observeCompletion));
    originalCompletedHandler = completion.original;
    MethodReplacement memory(device, @selector(currentAllocatedSize),
                             reinterpret_cast<IMP>(delayedCompletionMemoryQuery));
    originalAllocatedSize = memory.original;
    auto ticket = backend.submitAsync(dispatch);
    const bool completed = gpuDone.wait_for(std::chrono::seconds(5)) ==
                           std::future_status::ready;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool ready = ticket.ready();
    bool healthy = true;
    try { backend.checkHealth(); }
    catch (const MetalBackendError &) { healthy = false; }
    release.set_value();
    const unsigned queriesBeforeConsumption = memoryQueries;
    (void)ticket.wait();
    require(callbackDone.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "completion handler did not drain after telemetry was released");
    require(completed, "test GPU command did not complete");
    require(ready && healthy && completionMemoryQueries == 0,
            "completed GPU work depends on memory telemetry and can trip the watchdog");
    require(memoryQueries > queriesBeforeConsumption,
            "consuming a completed command did not refresh admission telemetry");
    const unsigned queriesAfterConsumption = memoryQueries;
    (void)ticket.wait();
    require(memoryQueries == queriesAfterConsumption,
            "an already-released ticket queried device memory again");
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "completion telemetry test produced the wrong result");
    std::cout << "PASS GPU completion independent of memory telemetry\n";
}

id<MTLSharedEvent> commandWatchdogGate = nil;
std::promise<void> delayedCompletionStarted;
std::promise<void> delayedCompletionReturned;
std::shared_future<void> releaseCompletionNotification;
IMP originalCommandStatus = nullptr;
std::atomic<void *> failedCommand{nullptr};
bool injectCommandFailure = false;
std::atomic<bool> delayNextCompletion{false};

IMP originalCommandCommit = nullptr;
void commitBehindWatchdogGate(id command, SEL selector) {
    [command encodeWaitForEvent:commandWatchdogGate value:1];
    reinterpret_cast<void (*)(id, SEL)>(originalCommandCommit)(command, selector);
}

MTLCommandBufferStatus terminalCommandStatus(id command, SEL selector) {
    if ((__bridge void *)command == failedCommand.load())
        return MTLCommandBufferStatusError;
    return reinterpret_cast<MTLCommandBufferStatus (*)(id, SEL)>(
        originalCommandStatus)(command, selector);
}

void delayCompletionNotification(id command, SEL selector, MTLCommandBufferHandler handler) {
    if (!delayNextCompletion.exchange(false)) {
        reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
            originalCompletedHandler)(command, selector, handler);
        return;
    }
    reinterpret_cast<void (*)(id, SEL, MTLCommandBufferHandler)>(
        originalCompletedHandler)(command, selector, ^(id<MTLCommandBuffer> completed) {
        require(completed.status == MTLCommandBufferStatusCompleted,
                "delayed notification test did not complete on the GPU");
        if (injectCommandFailure)
            failedCommand.store((__bridge void *)completed);
        delayedCompletionStarted.set_value();
        releaseCompletionNotification.wait();
        handler(completed);
        delayedCompletionReturned.set_value();
    });
}

void terminalCommandRecovers(const std::string &metallibPath, bool failed,
                                   bool pendingNext = false) {
    MetalBackend backend(metallibPath, 0.1);
    auto buffer = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    id<MTLCommandQueue> queue = [MTLCreateSystemDefaultDevice() newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    std::promise<void> release;
    releaseCompletionNotification = release.get_future().share();
    delayedCompletionStarted = std::promise<void>{};
    delayedCompletionReturned = std::promise<void>{};
    auto gpuDone = delayedCompletionStarted.get_future();
    auto callbackDone = delayedCompletionReturned.get_future();
    std::atomic<bool> healthy{true};
    std::atomic<unsigned> notifications{0};
    std::string error;
    injectCommandFailure = failed;
    delayNextCompletion = true;
    {
        MethodReplacement status(command, @selector(status),
                                 reinterpret_cast<IMP>(terminalCommandStatus));
        originalCommandStatus = status.original;
        MethodReplacement completion(command, @selector(addCompletedHandler:),
                                     reinterpret_cast<IMP>(delayCompletionNotification));
        originalCompletedHandler = completion.original;
        auto ticket = backend.submitAsync(dispatch, [&](uint64_t) { ++notifications; });
        require(gpuDone.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "GPU did not reach the delayed completion handler");
        require(!ticket.ready(), "test did not delay the completion notification");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto check = [&] {
            try { backend.checkHealth(); }
            catch (const MetalBackendError &) { healthy = false; }
        };
        auto peer = std::async(std::launch::async, check);
        check();
        peer.get();
        require(ticket.ready(), "terminal command still depends on its completion handler");
        try { (void)ticket.wait(); }
        catch (const MetalBackendError &failure) { error = failure.what(); }
        require(notifications == 1, "host completion did not notify exactly once");
        require(callbackDone.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready,
                "test released the callback before consuming its result");
        require(*static_cast<uint32_t *>(buffer.contents()) == increment,
                "host completion returned the wrong GPU result");
        dispatch.buffers.clear();
        buffer = {};
        require(backend.memoryStats().allocatedBytes == 0,
                "terminal ticket retained allocations until the callback returned");
        splash::metal::CommandTicket next;
        if (!failed) {
            buffer = backend.allocateBuffer(sizeof(uint32_t));
            *static_cast<uint32_t *>(buffer.contents()) = 0;
            dispatch.buffers = {{0, buffer}};
            if (pendingNext) {
                commandWatchdogGate = [MTLCreateSystemDefaultDevice() newSharedEvent];
                MethodReplacement commit(command, @selector(commit),
                                         reinterpret_cast<IMP>(commitBehindWatchdogGate));
                originalCommandCommit = commit.original;
                next = backend.submitAsync(dispatch);
            } else {
                next = backend.submitAsync(dispatch);
            }
        }
        // Metal may serialize later status notifications behind this handler.
        release.set_value();
        require(callbackDone.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "delayed completion handler did not drain");
        if (pendingNext) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::string failure;
            try { backend.checkHealth(); }
            catch (const MetalBackendError &caught) { failure = caught.what(); }
            require(!next.ready() && failure.find("sequence=2") != std::string::npos,
                    "late callback disarmed the next command's watchdog");
            commandWatchdogGate.signaledValue = 1;
            (void)next.wait();
            commandWatchdogGate = nil;
        } else if (next) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!next.ready() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            require(next.ready(), "backend did not resume after the delayed callback");
            (void)next.wait();
        }
    }
    failedCommand.store(nullptr);
    require(healthy == !failed && notifications == 1,
            "completed GPU work timed out while its notification was delayed");
    if (failed) {
        require(!backend.healthy() && error.find("Metal command 1 failed") != std::string::npos,
                "delayed GPU failure was lost or misclassified: " + error);
        requireBackendError([&] { (void)backend.submitAsync(dispatch); },
                            "failed GPU command admitted further work");
        std::cout << "PASS delayed GPU failure preserves its error\n";
        return;
    }
    require(error.empty(), "successful command failed: " + error);
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "backend did not continue after the delayed completion");
    std::cout << (pendingNext ? "PASS late completion preserves the next watchdog\n"
                             : "PASS terminal command completes without its callback\n");
}

void pendingCommandStillTimesOut(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, 0.1);
    auto buffer = backend.allocateBuffer(sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    splash::metal::CommandTicket ticket;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        ticket = backend.submitAsync(dispatch);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool pending = !ticket.ready();
    std::string failure;
    try { backend.checkHealth(); }
    catch (const MetalBackendError &error) { failure = error.what(); }
    commandWatchdogGate.signaledValue = 1;
    (void)ticket.wait();
    commandWatchdogGate = nil;
    require(pending && !backend.healthy(), "pending GPU command escaped the watchdog");
    require(failure.find("sequence=1") != std::string::npos &&
                failure.find("dispatches=1") != std::string::npos &&
                (failure.find("status=committed") != std::string::npos ||
                 failure.find("status=scheduled") != std::string::npos),
            "command timeout lost its submission diagnostics: " + failure);
    requireBackendError([&] { (void)backend.submitAsync(dispatch); },
                        "timed-out backend accepted more work");
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "timed-out command lost resources before GPU completion");
    std::cout << "PASS pending GPU command watchdog and resource lifetime\n";
}

id<MTLSharedEvent> submissionGate = nil;
id<MTLSharedEvent> delayedMappingEvent = nil;
IMP originalSparseSignal = nullptr;
void delayMappingSignal(id queue, SEL selector, id<MTLSharedEvent> event, uint64_t value) {
    if (value == 3) {
        delayedMappingEvent = event;
        // Complete the real mapping on a separate event. The test publishes
        // its dependency only after verifying that backing is ready.
        event = submissionGate;
        value = 1;
    }
    reinterpret_cast<void (*)(id, SEL, id<MTLSharedEvent>, uint64_t)>(
        originalSparseSignal)(queue, selector, event, value);
}

void backendDeferredSubmission(const std::string &metallibPath) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    constexpr uint64_t tile = MetalBackend::kPlacementSparsePageBytes;
    // Exercise the real submission/ticket path, not just the event helper.
    for (const std::string mode : {"resume", "stop", "timeout", "teardown-unmap"}) {
        auto backend = std::make_unique<MetalBackend>(metallibPath);
        auto sparse = backend->allocatePlacementSparseBuffer(tile, tile, "gated-map");
        auto heap = backend->allocatePlacementHeap(tile, tile, "gated-heap");
        SparseMapping mapping{sparse, 0, tile, 0};
        auto readback = backend->allocateBuffer(4 * sizeof(uint32_t));
        std::fill_n(static_cast<uint32_t *>(readback.contents()), 4, 0);
        const uint32_t count = 4, seed = 73;
        ComputeDispatch dispatch;
        dispatch.pipelineName = "sparse_fill_copy_u32";
        dispatch.buffers = {{0, sparse}, {1, readback}};
        dispatch.bytes = {{2, &count, sizeof(count)}, {3, &seed, sizeof(seed)}};
        dispatch.threadgroups = {1, 1, 1};
        dispatch.threadsPerThreadgroup = {4, 1, 1};
        submissionGate = [device newSharedEvent];
        {
            MethodReplacement replacement(queue, @selector(signalEvent:value:),
                reinterpret_cast<IMP>(delayMappingSignal));
            originalSparseSignal = replacement.original;
            backend->mapSparse(heap, {&mapping, 1});
        }
        require(delayedMappingEvent && delayedMappingEvent.signaledValue < 3,
                "mapping test gate was not installed");
        if (mode == "teardown-unmap") {
            backend->unmapSparse({&mapping, 1}, std::move(heap));
            const auto start = std::chrono::steady_clock::now();
            backend.reset();
            require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
                    "backend teardown waited for the mapping queue");
            require([submissionGate waitUntilSignaledValue:1 timeoutMS:5000],
                    "test mapping did not complete");
            delayedMappingEvent.signaledValue = 3;
            require([delayedMappingEvent waitUntilSignaledValue:4 timeoutMS:5000],
                    "unmap ownership did not survive backend teardown");
            continue;
        }
        std::atomic<unsigned> callbacks{0};
        std::promise<void> completion;
        auto notified = completion.get_future();
        auto ticket = backend->submitAsync(dispatch, [&](uint64_t) {
            if (++callbacks == 1) completion.set_value();
        });
        auto requireNotifiedOnce = [&] {
            require(notified.wait_for(std::chrono::seconds(5)) == std::future_status::ready &&
                        callbacks == 1, "deferred ticket did not notify exactly once");
        };
        require(!ticket.ready() && backend->memoryStats().sparseMapWaitEvent == 3,
                "backend did not defer submission behind mapping");
        requireBackendError([&] { (void)backend->submitAsync(dispatch); },
                            "deferred command did not hold the submission gate");
        if (mode == "resume") {
            std::this_thread::sleep_for(std::chrono::seconds(6));
            require(!ticket.ready() && callbacks == 0,
                    "backend completed a command before its mapping");
            require([submissionGate waitUntilSignaledValue:1 timeoutMS:5000],
                    "test mapping did not complete");
            delayedMappingEvent.signaledValue = 3;
            (void)ticket.wait();
            requireNotifiedOnce();
            require(backend->healthy(), "resumed ticket poisoned the backend");
            auto *words = static_cast<uint32_t *>(readback.contents());
            for (uint32_t i = 0; i < count; ++i)
                require(words[i] == seed + i, "deferred command produced wrong output");
            backend->unmapSparse({&mapping, 1}, std::move(heap));
            backend->drainSparseUnmaps();
            awaitSparseRelease(*backend, 0);
        } else {
            const auto start = std::chrono::steady_clock::now();
            if (mode == "stop") backend->stop();
            const auto deadline = start + std::chrono::seconds(mode == "stop" ? 2 : 35);
            while (!ticket.ready() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            require(ticket.ready(), "stopped/timed-out mapping did not finish its ticket");
            try {
                (void)ticket.wait();
                fail("stopped/timed-out mapping completed successfully");
            } catch (const MetalBackendError &error) {
                const std::string message = error.what();
                require(message.find(mode == "stop" ? "stopped before" : "wait exceeded") !=
                            std::string::npos, "deferred failure lost its cause: " + message);
            }
            requireNotifiedOnce();
            require(!backend->healthy(), "failed ticket did not poison the backend");
            requireBackendError([&] { (void)backend->submitAsync(dispatch); },
                                "stopped backend accepted new work");
            backend.reset();
            require([submissionGate waitUntilSignaledValue:1 timeoutMS:5000],
                    "test mapping did not complete");
            delayedMappingEvent.signaledValue = 3;
            require([delayedMappingEvent waitUntilSignaledValue:3 timeoutMS:5000],
                    "map ownership did not survive backend teardown");
            require(static_cast<uint32_t *>(readback.contents())[0] == 0,
                    "cancelled deferred command ran on the GPU");
        }
        std::cout << "PASS backend deferred submission " << mode << '\n';
    }
    submissionGate = nil;
    delayedMappingEvent = nil;
}

BOOL noPlacementSupport(id, SEL) { return NO; }
BOOL failPlacementQuery(id, SEL) {
    id<SplashPlacementSparseDevice> backing = (id<SplashPlacementSparseDevice>)[NSObject new];
    return backing.supportsPlacementSparse;
}
id<MTLHeap> refuseProbeHeap(id, SEL, MTLHeapDescriptor *) { return nil; }
uint64_t failedProbeWaitValue = 0;
IMP originalProbeWait = nullptr;
BOOL failProbeWait(id event, SEL selector, uint64_t value, uint64_t timeout) {
    if (value == failedProbeWaitValue)
        return NO;
    return reinterpret_cast<BOOL (*)(id, SEL, uint64_t, uint64_t)>(
        originalProbeWait)(event, selector, value, timeout);
}

void placementProbeFailures(const std::string &metallibPath) {
    // Faults apply only to this serial test process and are restored per case.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLSharedEvent> event = [device newSharedEvent];
    require(device && event, "probe fault fixtures are unavailable");
    {
        MethodReplacement replacement(device, @selector(supportsPlacementSparse),
                                      reinterpret_cast<IMP>(noPlacementSupport));
        MetalBackend backend(metallibPath);
        require(!backend.capabilities().supportsPlacementSparse,
                "unsupported device was not preserved as a capability result");
    }
    {
        MethodReplacement replacement(device, @selector(supportsPlacementSparse),
                                      reinterpret_cast<IMP>(failPlacementQuery));
        try {
            MetalBackend backend(metallibPath);
            fail("probe query failure was reported as unsupported");
        } catch (const MetalBackendError &error) {
            const std::string message = error.what();
            require(message.find("supportsPlacementSparse query failed") !=
                        std::string::npos &&
                    message.find("unrecognized selector") != std::string::npos,
                    "probe query failure lost its cause");
        }
    }
    {
        MethodReplacement replacement(device, @selector(newHeapWithDescriptor:),
                                      reinterpret_cast<IMP>(refuseProbeHeap));
        try {
            MetalBackend backend(metallibPath);
            fail("probe allocation failure was reported as unsupported");
        } catch (const MetalAllocationError &error) {
            require(error.failure() == splash::metal::AllocationFailure::DriverRejected &&
                        std::string(error.what()).find("probe could not allocate") !=
                            std::string::npos,
                    "probe allocation failure lost its cause");
        }
    }
    for (uint64_t failedValue : {1ULL, 2ULL}) {
        failedProbeWaitValue = failedValue;
        MethodReplacement replacement(event,
            @selector(waitUntilSignaledValue:timeoutMS:),
            reinterpret_cast<IMP>(failProbeWait));
        originalProbeWait = replacement.original;
        try {
            MetalBackend backend(metallibPath);
            fail("probe timeout was reported as unsupported");
        } catch (const splash::metal::MetalAllocationError &) {
            fail("probe timeout was misclassified as an allocation failure");
        } catch (const MetalBackendError &error) {
            const std::string message = error.what();
            require(message.find("timed out after 5000 ms") != std::string::npos &&
                        message.find(failedValue == 1 ? "for mapping" : "for unmapping") !=
                            std::string::npos,
                    "probe timeout lost its phase or deadline");
        }
    }
    std::cout << "placement_probe_failures=PASS\n";
}

void sharedMemoryCompletionLifetime(MetalBackend &backend) {
    struct Gate {
        std::mutex mutex;
        std::condition_variable condition;
        bool entered = false;
        bool release = false;
        std::atomic<bool> ownerReleased{false};
    };
    auto gate = std::make_shared<Gate>();
    const size_t bytes = static_cast<size_t>(getpagesize());
    void *address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    require(address != MAP_FAILED, "unable to allocate lifetime witness");
    auto owner = std::shared_ptr<void>(address, [gate, bytes](void *memory) {
        munmap(memory, bytes);
        gate->ownerReleased.store(true);
    });
    auto buffer = backend.wrapSharedMemory(address, bytes, owner);
    owner.reset();
    const uint32_t count = 1, increment = 1;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    auto ticket = backend.submitAsync(dispatch, [gate](uint64_t) {
        std::unique_lock lock(gate->mutex);
        gate->entered = true;
        gate->condition.notify_all();
        gate->condition.wait_for(lock, std::chrono::seconds(5),
                                [&] { return gate->release; });
    });
    dispatch.buffers.clear();
    buffer = {};
    {
        std::unique_lock lock(gate->mutex);
        require(gate->condition.wait_for(lock, std::chrono::seconds(5),
                                         [&] { return gate->entered; }),
                "lifetime witness completion did not arrive");
    }
    require(!gate->ownerReleased.load(),
            "external memory released while the command ticket still owns it");
    // Applying the ticket drops its C++ allocations. Metal may release the
    // underlying buffer before, during or after the completion callback;
    // only the ticket-owned lifetime above and eventual release are required.
    (void)ticket.wait();
    {
        std::lock_guard lock(gate->mutex);
        gate->release = true;
    }
    gate->condition.notify_all();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    while (!gate->ownerReleased.load() &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(gate->ownerReleased.load(),
            "external memory leaked after Metal released its buffer");
}

void sparseExtentChurn(MetalBackend &backend) {
    // Match the production per-layer data/scale mapping sizes. Rotate three
    // virtual extents, retaining one as a witness while its replacement is
    // mapped and used: at most two 130-MiB heaps plus one extent of readback.
    constexpr splash::kv::Layout layout{16, 4, 256};
    constexpr uint32_t kSegments = layout.attentionLayers * 4;
    constexpr uint32_t kExtentCount = 3;
    constexpr uint32_t kRepetitions = 256;
    constexpr uint32_t kRollbackPeriod = 4;
    constexpr uint32_t kGuardWords = 64;
    constexpr uint32_t kSampleWords = 64;
    constexpr uint32_t kCanary = 0xd15ca11u;
    constexpr uint64_t kGuardBytes = kGuardWords * sizeof(uint32_t);
    constexpr uint64_t kExtentBytes =
        layout.backingExtentPages() * layout.bytesPerModelPage();
    constexpr uint64_t kWitnessBytes =
        kSegments * 2 * kSampleWords * sizeof(uint32_t);
    const auto before = backend.memoryStats();
    const uint64_t submissionsBefore = backend.submissionCount();
    {
        std::array<MetalBuffer, kSegments> buffers;
        std::array<uint64_t, kSegments> segmentBytes{};
        std::array<uint64_t, kSegments> offsets{};
        std::array<uint32_t, kSegments> counts{};
        uint64_t offset = 0;
        for (uint32_t segment = 0; segment < kSegments; ++segment) {
            segmentBytes[segment] = layout.backingExtentPages() *
                (segment % 2 ? layout.scaleBytesPerLayerPage()
                             : layout.dataBytesPerLayerPage());
            offsets[segment] = offset;
            counts[segment] = segmentBytes[segment] / sizeof(uint32_t);
            offset += segmentBytes[segment];
            buffers[segment] = backend.allocatePlacementSparseBuffer(
                kExtentCount * segmentBytes[segment],
                splash::kv::kSparseMappingAlignmentBytes,
                "sparse-churn-layer-buffer-" + std::to_string(segment));
        }
        require(offset == kExtentBytes, "sparse churn geometry disagrees");
        require(backend.memoryStats().sparseVirtualBytes ==
                    before.sparseVirtualBytes + kExtentCount * kExtentBytes,
                "sparse churn virtual accounting disagrees");
        std::array<std::vector<SparseMapping>, kExtentCount> mappings;
        for (uint32_t extent = 0; extent < kExtentCount; ++extent) {
            for (uint32_t segment = 0; segment < kSegments; ++segment) {
                mappings[extent].push_back({buffers[segment],
                    extent * segmentBytes[segment], segmentBytes[segment],
                    offsets[segment]});
            }
        }
        MetalBuffer readback = backend.allocateBuffer(
            kExtentBytes + kWitnessBytes + 3 * kGuardBytes,
            BufferStorage::Shared, "sparse-churn-readback");
        auto *words = static_cast<uint32_t *>(readback.contents());
        const uint64_t middleGuard = kGuardWords + kExtentBytes / sizeof(uint32_t);
        const uint64_t witnessStart = middleGuard + kGuardWords;
        const uint64_t finalGuard = witnessStart + kWitnessBytes / sizeof(uint32_t);
        std::array<std::optional<splash::metal::SparseHeap>, kExtentCount> heaps;
        int32_t witness = -1;
        std::array<uint32_t, kSegments> seeds{};
        std::array<uint32_t, kSegments> witnessSeeds{};
        for (uint32_t repetition = 0; repetition < kRepetitions; ++repetition) {
            const uint32_t extent = repetition % kExtentCount;
            try {
                auto mapExtent = [&] {
                    require(!heaps[extent], "sparse churn reused a resident extent");
                    heaps[extent].emplace(backend.allocatePlacementHeap(
                        kExtentBytes, splash::kv::kSparseMappingAlignmentBytes,
                        "sparse-churn-extent-" + std::to_string(extent)));
                    require(heaps[extent]->sizeBytes() == kExtentBytes,
                            "sparse churn heap size changed");
                    backend.mapSparse(*heaps[extent], mappings[extent]);
                    require(backend.memoryStats().sparseResidentBytes ==
                                before.sparseResidentBytes +
                                    (witness < 0 ? 1 : 2) * kExtentBytes,
                            "sparse churn resident accounting disagrees");
                };
                mapExtent();
                if (repetition % kRollbackPeriod == 0) {
                    // Allocation rollback may unmap while the asynchronous
                    // map is pending, without an intervening compute ticket.
                    backend.unmapSparse(mappings[extent], std::move(*heaps[extent]));
                    heaps[extent].reset();
                    backend.drainSparseUnmaps();
                    awaitSparseRelease(backend, before.sparseResidentBytes +
                        (witness < 0 ? 0 : 1) * kExtentBytes);
                    require(backend.memoryStats().sparseResidentBytes ==
                                before.sparseResidentBytes +
                                    (witness < 0 ? 0 : 1) * kExtentBytes,
                            "rolled-back churn heap remains resident");
                    require(backend.submissionCount() ==
                                submissionsBefore + 2 * repetition,
                            "sparse rollback submitted a compute command");
                    mapExtent();
                }
                for (uint64_t guard : {uint64_t{0}, middleGuard, finalGuard})
                    std::fill_n(words + guard, kGuardWords, kCanary);

                std::vector<ComputeDispatch> fill;
                std::vector<ComputeDispatch> copy;
                auto append = [&](std::vector<ComputeDispatch> &dispatches,
                                  const char *pipeline, const MetalBuffer &source,
                                  uint64_t destinationOffset, const uint32_t &count,
                                  const uint32_t *seed = nullptr) {
                    ComputeDispatch dispatch;
                    dispatch.pipelineName = pipeline;
                    dispatch.buffers = {{0, source}, {1, backend.view(readback,
                        destinationOffset, uint64_t{count} * sizeof(uint32_t))}};
                    dispatch.bytes = {{2, &count, sizeof(count)}};
                    if (seed) dispatch.bytes.push_back({3, seed, sizeof(*seed)});
                    dispatch.threadgroups = {(uint64_t{count} + 255) / 256, 1, 1};
                    dispatch.threadsPerThreadgroup = {256, 1, 1};
                    dispatches.push_back(std::move(dispatch));
                };
                for (uint32_t segment = 0; segment < kSegments; ++segment) {
                    seeds[segment] = (repetition + 1) * 1048576u + segment * 8192u;
                    const auto source = backend.view(buffers[segment],
                        extent * segmentBytes[segment], segmentBytes[segment]);
                    append(fill, "sparse_fill_copy_u32", source,
                        kGuardBytes + offsets[segment], counts[segment], &seeds[segment]);
                    // A separate command reads the sparse resource, rather
                    // than relying on the fill kernel's own write-through copy.
                    append(copy, "test_copy_u32", source,
                        kGuardBytes + offsets[segment], counts[segment]);
                    if (witness < 0) continue;
                    for (uint32_t end = 0; end < 2; ++end) {
                        const uint64_t sampleOffset = end
                            ? segmentBytes[segment] - kSampleWords * sizeof(uint32_t) : 0;
                        append(copy, "test_copy_u32", backend.view(buffers[segment],
                            static_cast<uint64_t>(witness) * segmentBytes[segment] + sampleOffset,
                            kSampleWords * sizeof(uint32_t)),
                            witnessStart * sizeof(uint32_t) +
                                (segment * 2 + end) * kSampleWords * sizeof(uint32_t),
                            kSampleWords);
                    }
                }
                auto fillTicket = backend.submitCommandAsync(fill);
                if (!repetition) {
                    requireBackendError([&] {
                        backend.unmapSparse(mappings[extent], std::move(*heaps[extent]));
                    }, "sparse unmap accepted an undrained compute ticket");
                    require(heaps[extent] && *heaps[extent],
                            "rejected unmap consumed the caller's heap");
                }
                (void)fillTicket.wait();
                std::fill_n(words + kGuardWords, kExtentBytes / sizeof(uint32_t),
                            std::numeric_limits<uint32_t>::max());
                std::fill_n(words + witnessStart, kWitnessBytes / sizeof(uint32_t),
                            std::numeric_limits<uint32_t>::max());
                auto copyTicket = backend.submitCommandAsync(copy);
                (void)copyTicket.wait();
                for (uint32_t segment = 0; segment < kSegments; ++segment) {
                    const uint64_t begin = kGuardWords + offsets[segment] / sizeof(uint32_t);
                    for (uint32_t index = 0; index < counts[segment]; ++index) {
                        if (words[begin + index] != seeds[segment] + index)
                            throw std::runtime_error("sparse full-extent readback mismatch at segment " +
                                std::to_string(segment) + " word " + std::to_string(index));
                    }
                    if (witness < 0) continue;
                    for (uint32_t end = 0; end < 2; ++end) {
                        const uint32_t sampleOffset = end ? counts[segment] - kSampleWords : 0;
                        const uint64_t sample = witnessStart + (segment * 2 + end) * kSampleWords;
                        for (uint32_t index = 0; index < kSampleWords; ++index) {
                            if (words[sample + index] != witnessSeeds[segment] + sampleOffset + index)
                                throw std::runtime_error("resident witness changed at segment " +
                                    std::to_string(segment));
                        }
                    }
                }
                for (uint64_t guard : {uint64_t{0}, middleGuard, finalGuard}) {
                    for (uint32_t index = 0; index < kGuardWords; ++index)
                        require(words[guard + index] == kCanary, "sparse churn readback canary changed");
                }
                // Both real GPU tickets are drained before any mapping or
                // heap is released, matching the engine's cancellation drain.
                if (witness >= 0) {
                    backend.unmapSparse(mappings[witness], std::move(*heaps[witness]));
                    heaps[witness].reset();
                    // The heap stays resident until the queue reports the
                    // unmap complete; the next repetition's map is ordered
                    // behind it on the same queue.
                    backend.drainSparseUnmaps();
                }
                witness = extent;
                witnessSeeds = seeds;
                awaitSparseRelease(backend, before.sparseResidentBytes + kExtentBytes);
                require(backend.memoryStats().sparseResidentBytes ==
                            before.sparseResidentBytes + kExtentBytes,
                        "released churn heap remains resident");
            } catch (const std::exception &error) {
                throw std::runtime_error("sparse churn repetition " +
                    std::to_string(repetition) + " extent " +
                    std::to_string(extent) + ": " + error.what());
            }
        }
        backend.unmapSparse(mappings[witness], std::move(*heaps[witness]));
        heaps[witness].reset();
        backend.drainSparseUnmaps();
        awaitSparseRelease(backend, before.sparseResidentBytes);
        require(backend.memoryStats().sparseResidentBytes == before.sparseResidentBytes,
                "final churn heap remains resident");
    }
    awaitSparseRelease(backend, before.sparseResidentBytes, before.sparseVirtualBytes);
    const auto after = backend.memoryStats();
    require(after.allocatedBytes == before.allocatedBytes &&
                after.sparseVirtualBytes == before.sparseVirtualBytes &&
                after.sparseResidentBytes == before.sparseResidentBytes,
            "sparse churn leaked buffer or heap accounting");
    require(backend.submissionCount() == submissionsBefore + 2 * kRepetitions,
            "sparse churn command count changed");
    require(backend.healthy(), "sparse churn poisoned the backend");
    std::cout << "PASS sparse extent churn repetitions=" << kRepetitions
              << " pending_map_rollbacks=" << (kRepetitions + kRollbackPeriod - 1) / kRollbackPeriod
              << " layers=" << layout.attentionLayers
              << " mappings_per_extent=" << kSegments
              << " extent_bytes=" << kExtentBytes << '\n';
}

// Releases are paced: a second unmap waits for the first, heaps stay resident
// until completion, and a remap of the same range orders behind the unmap.
void sparsePacedRelease(MetalBackend &backend) {
    constexpr uint64_t kTile = splash::kv::kSparseMappingAlignmentBytes;
    constexpr uint32_t kExtents = 3;
    constexpr uint32_t kTilesPerExtent = 4;
    constexpr uint64_t kExtentBytes = kTilesPerExtent * kTile;
    const auto before = backend.memoryStats();
    MetalBuffer buffer = backend.allocatePlacementSparseBuffer(
        kExtents * kExtentBytes, kTile, "sparse-paced");
    std::array<std::optional<splash::metal::SparseHeap>, kExtents> heaps;
    std::array<SparseMapping, kExtents> mappings;
    auto mapExtent = [&](uint32_t extent) {
        heaps[extent].emplace(backend.allocatePlacementHeap(
            kExtentBytes, kTile, "sparse-paced-heap"));
        mappings[extent] = {buffer, extent * kExtentBytes, kExtentBytes, 0};
        backend.mapSparse(*heaps[extent], {&mappings[extent], 1});
    };
    for (uint32_t extent = 0; extent < kExtents; ++extent) mapExtent(extent);
    MetalBuffer readback = backend.allocateBuffer(
        kExtentBytes, BufferStorage::Shared, "sparse-paced-readback");
    const uint32_t count = kExtentBytes / sizeof(uint32_t);
    auto fillAndCheck = [&](uint32_t extent, uint32_t seed) {
        ComputeDispatch fill;
        fill.pipelineName = "sparse_fill_copy_u32";
        fill.buffers = {{0, backend.view(buffer, extent * kExtentBytes, kExtentBytes)},
                        {1, readback}};
        fill.bytes = {{2, &count, sizeof(count)}, {3, &seed, sizeof(seed)}};
        fill.threadgroups = {(uint64_t{count} + 255) / 256, 1, 1};
        fill.threadsPerThreadgroup = {256, 1, 1};
        (void)backend.submit(fill);
        auto *words = static_cast<uint32_t *>(readback.contents());
        for (uint32_t index = 0; index < count; ++index)
            require(words[index] == seed + index, "paced-release extent readback mismatch");
    };
    for (uint32_t extent = 0; extent < kExtents; ++extent) fillAndCheck(extent, 17 + extent);
    require(backend.memoryStats().sparseResidentBytes ==
                before.sparseResidentBytes + kExtents * kExtentBytes,
            "paced-release setup accounting disagrees");

    // Two back-to-back releases: the second waits for the first internally.
    backend.unmapSparse({&mappings[2], 1}, std::move(*heaps[2]));
    heaps[2].reset();
    backend.unmapSparse({&mappings[1], 1}, std::move(*heaps[1]));
    heaps[1].reset();
    require(backend.memoryStats().pendingSparseUnmaps <= 1,
            "two sparse unmaps were outstanding at once");
    backend.drainSparseUnmaps();
    awaitSparseRelease(backend, before.sparseResidentBytes + kExtentBytes);
    auto stats = backend.memoryStats();
    require(stats.sparseResidentBytes == before.sparseResidentBytes + kExtentBytes &&
                stats.pendingSparseUnmaps == 0 &&
                stats.completedSparseUnmaps >= before.completedSparseUnmaps + 2,
            "paced releases did not free both heaps after draining");

    // Remap a released range while its unmap may still be in flight: the
    // queue orders the new map behind the unmap, and the data is fresh.
    mapExtent(1);
    fillAndCheck(1, 4001);
    fillAndCheck(0, 4000);
    backend.unmapSparse({&mappings[1], 1}, std::move(*heaps[1]));
    heaps[1].reset();
    mapExtent(1);
    fillAndCheck(1, 5001);
    for (uint32_t extent = 0; extent < 2; ++extent) {
        backend.unmapSparse({&mappings[extent], 1}, std::move(*heaps[extent]));
        heaps[extent].reset();
    }
    backend.drainSparseUnmaps();
    awaitSparseRelease(backend, before.sparseResidentBytes);
    require(backend.memoryStats().sparseResidentBytes == before.sparseResidentBytes,
            "paced-release cleanup leaked a heap");
    readback = {};
    for (SparseMapping &mapping : mappings) mapping.buffer = {};
    buffer = {};
    awaitSparseRelease(backend, before.sparseResidentBytes, before.sparseVirtualBytes);
    require(backend.memoryStats().sparseVirtualBytes == before.sparseVirtualBytes &&
                backend.healthy(),
            "paced-release cleanup leaked address space or poisoned the backend");
    std::cout << "PASS sparse paced release extents=" << kExtents
              << " tile_bytes=" << kTile << '\n';
}

void run(const std::string &metallibPath) {
    placementProbeFailures(metallibPath);
    backendDeferredSubmission(metallibPath);
    NSData *libraryData = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:metallibPath.c_str()]];
    const auto expectedDigest = libraryDigest(libraryData);
    TemporaryMetallib temporary;
    NSString *temporaryPath = [NSString stringWithUTF8String:temporary.path.c_str()];
    require([libraryData writeToFile:temporaryPath options:0 error:nullptr],
            "could not copy test metallib");
    MetalBackend backend(temporary.path);
    const auto guardedBytes = backend.memoryStats().allocatedBytes;
    const auto denyOperation = [] {
        throw MetalAllocationError("test host pressure", AllocationFailure::HostPressure);
    };
    backend.setOperationGuard(denyOperation);
    try {
        (void)backend.allocateBuffer(16384, BufferStorage::Shared);
        fail("operation guard admitted an allocation");
    } catch (const MetalAllocationError &error) {
        require(error.failure() == AllocationFailure::HostPressure,
                "operation guard lost its failure classification");
    }
    require(backend.healthy() && backend.memoryStats().allocatedBytes == guardedBytes,
            "operation guard leaked memory or poisoned the backend");
    backend.setOperationGuard({});

    require(backend.metallibSha256() == expectedDigest,
            "backend digest does not match the loaded library bytes");
    NSData *replacement = [@"replaced after library loading"
        dataUsingEncoding:NSUTF8StringEncoding];
    require([replacement writeToFile:temporaryPath
                            options:NSDataWritingAtomic error:nullptr],
            "could not replace temporary metallib");
    require(libraryDigest([NSData dataWithContentsOfFile:temporaryPath]) !=
                expectedDigest,
            "temporary metallib replacement did not change its bytes");
    require(backend.metallibSha256() == expectedDigest,
            "backend digest followed the replaced library path");
    // All existing pipeline/dispatch checks below run from the original
    // loaded library even though its former path now contains invalid bytes.
    const auto &capabilities = backend.capabilities();
    require(!capabilities.deviceName.empty(), "device name is empty");
    require(capabilities.physicalMemoryBytes > 0,
            "physical memory capability is missing");
    require(capabilities.recommendedMaxWorkingSetBytes > 0,
            "recommended working set capability is missing");
    require(capabilities.maxBufferLengthBytes > 0,
            "maximum buffer length capability is missing");
    require(capabilities.appleGpuFamily >= 9,
            "Apple GPU family capability is missing");
    require(capabilities.maxThreadgroupMemoryBytes >= 32 * 1024,
            "threadgroup memory capability is insufficient");
    require(capabilities.maxThreadgroupWidth >= 256,
            "threadgroup thread capability is insufficient");
    require(capabilities.supportsPlacementSparse,
            "placement-sparse capability is missing");
    require(backend.healthy(), "new backend is unhealthy");
    require(backend.submissionCount() == 0, "new backend has submissions");
    require(backend.pipelineCount() == 0, "pipeline cache is not empty");

    constexpr uint32_t kElementCount = 64;
    constexpr uint32_t kViewElementCount = kElementCount / 2;
    constexpr uint32_t kIncrement = 7;
    constexpr uint64_t kAllocationBytes =
        sizeof(uint32_t) * kElementCount;

    MetalBuffer base = backend.allocateBuffer(
        kAllocationBytes, BufferStorage::Shared, "metal-backend-test");
    require(base && base.contents(), "shared allocation is not CPU-visible");
    require(base.sizeBytes() == kAllocationBytes,
            "allocation length is unexpected");

    auto stats = backend.memoryStats();
    const uint64_t actualAllocationBytes = stats.allocatedBytes;
    require(actualAllocationBytes >= base.sizeBytes(),
            "actual live allocation bytes were not tracked");
    require(stats.peakAllocatedBytes == actualAllocationBytes,
            "peak allocation bytes were not tracked");
    require(stats.peakResidentBytes == actualAllocationBytes,
            "dense-only physical peak was not tracked");
    require(stats.deviceCurrentAllocatedBytes >= actualAllocationBytes,
            "device allocation counter is smaller than backend allocations");
    require(stats.devicePeakAllocatedBytes >=
                stats.deviceCurrentAllocatedBytes,
            "device peak counter is smaller than current allocations");

    auto *values = static_cast<uint32_t *>(base.contents());
    for (uint32_t i = 0; i < kElementCount; ++i) values[i] = i;

    MetalBuffer view = backend.view(
        base, sizeof(uint32_t) * kViewElementCount,
        sizeof(uint32_t) * kViewElementCount);
    require(view.contents() == values + kViewElementCount,
            "view contents pointer has the wrong offset");
    require(view.sameView(backend.view(base, sizeof(uint32_t) * kViewElementCount,
                                      sizeof(uint32_t) * kViewElementCount)) &&
                view.sameView(backend.view(view, 0, view.sizeBytes())) &&
                !view.sameView(base) && !view.sameView(MetalBuffer{}) &&
                MetalBuffer{}.sameView(MetalBuffer{}),
            "buffer view identity does not compare allocation and exact range");
    require(backend.memoryStats().allocatedBytes == actualAllocationBytes,
            "view was counted as a new allocation");

    double lastWallSeconds = 0.0;
    {
        ComputeDispatch dispatch;
        dispatch.pipelineName = "test_add_u32";
        dispatch.buffers.push_back(BufferBinding{0, view});
        dispatch.bytes.push_back(BytesBinding{
            1, &kViewElementCount, sizeof(kViewElementCount)});
        dispatch.bytes.push_back(
            BytesBinding{2, &kIncrement, sizeof(kIncrement)});
        dispatch.threadgroups = {1, 1, 1};
        dispatch.threadsPerThreadgroup = {kViewElementCount, 1, 1};

        backend.setOperationGuard(denyOperation);
        try {
            (void)backend.submit(dispatch);
            fail("operation guard admitted a GPU submission");
        } catch (const MetalAllocationError &error) {
            require(error.failure() == AllocationFailure::HostPressure &&
                        backend.healthy() && backend.submissionCount() == 0,
                    "guarded submission lost its cause or altered the backend");
        }
        backend.setOperationGuard({});

        for (int runIndex = 0; runIndex < 2; ++runIndex) {
            splash::metal::CommandTiming timing;
            if (!runIndex) {
                timing = backend.submit(dispatch);
            } else {
                std::promise<uint64_t> completedSequence;
                auto notified = completedSequence.get_future();
                auto ticket = backend.submitAsync(
                    dispatch, [&](uint64_t sequence) {
                        completedSequence.set_value(sequence);
                    });
                require(ticket && ticket.sequence() > 0,
                        "async submission returned an empty ticket");
                requireBackendError(
                    [&] { (void)backend.submit(dispatch); },
                    "a second in-flight command was accepted");
                timing = ticket.wait();
                require(ticket.ready(),
                        "completed async ticket is not ready");
                // Completion is published before the callback runs, so
                // wait() may return first; only the callback's own signal
                // shows the notification was delivered.
                require(notified.wait_for(std::chrono::seconds(5)) ==
                                std::future_status::ready &&
                            notified.get() == ticket.sequence(),
                        "async completion notification was not delivered");
            }
            require(std::isfinite(timing.gpuSeconds) &&
                        timing.gpuSeconds >= 0.0,
                    "GPU timing is invalid");
            require(std::isfinite(timing.wallSeconds) &&
                        timing.wallSeconds > 0.0,
                    "wall timing is invalid");
            lastWallSeconds = timing.wallSeconds;
        }
    }

    require(backend.submissionCount() == 2,
            "successful submissions were not counted");
    require(backend.pipelineCount() == 1,
            "pipeline cache did not reuse the pipeline");
    for (uint32_t i = 0; i < kViewElementCount; ++i) {
        require(values[i] == i, "dispatch wrote before the buffer view");
    }
    for (uint32_t i = kViewElementCount; i < kElementCount; ++i) {
        require(values[i] == i + 2 * kIncrement,
                "dispatch produced an incorrect result");
    }

    {
        ComputeDispatch first;
        first.pipelineName = "test_add_u32";
        first.buffers.push_back(BufferBinding{0, view});
        first.bytes.push_back(BytesBinding{
            1, &kViewElementCount, sizeof(kViewElementCount)});
        first.bytes.push_back(
            BytesBinding{2, &kIncrement, sizeof(kIncrement)});
        first.threadgroups = {1, 1, 1};
        first.threadsPerThreadgroup = {kViewElementCount, 1, 1};
        std::vector<ComputeDispatch> command{first, first};
        (void)backend.submitCommand(command);
    }
    require(backend.submissionCount() == 3,
            "explicit operation list did not use one command buffer");
    for (uint32_t i = kViewElementCount; i < kElementCount; ++i) {
        require(values[i] == i + 4 * kIncrement,
                "multi-dispatch command produced an incorrect result");
    }

    requireBackendError(
        [&] { (void)backend.view(base, base.sizeBytes(), 1); },
        "out-of-range view was accepted");

    ComputeDispatch missingPipeline;
    missingPipeline.pipelineName = "does_not_exist";
    requireBackendError(
        [&] { (void)backend.submit(missingPipeline); },
        "missing pipeline was accepted");
    require(backend.healthy(),
            "a descriptor error incorrectly poisoned the backend");
    require(backend.unhealthyReason().empty(),
            "healthy backend has an unhealthy reason");
    require(backend.submissionCount() == 3,
            "failed pre-commit dispatch was counted as submitted");
    require(backend.pipelineCount() == 1,
            "failed pipeline lookup polluted the cache");

    base = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == actualAllocationBytes,
            "a live view did not retain its base allocation");
    view = MetalBuffer{};
    stats = backend.memoryStats();
    require(stats.allocatedBytes == 0,
            "released allocation remains in live byte accounting");
    require(stats.peakAllocatedBytes == actualAllocationBytes,
            "peak allocation accounting changed after release");
    require(stats.devicePeakAllocatedBytes >=
                stats.deviceCurrentAllocatedBytes,
            "device peak allocation accounting regressed");

    MetalBuffer privateBuffer = backend.allocateBuffer(
        16, BufferStorage::Private, "private-test");
    require(privateBuffer.contents() == nullptr,
            "private allocation unexpectedly exposed CPU contents");
    require(privateBuffer.sameView(backend.view(privateBuffer, 0, 16)) &&
                !privateBuffer.sameView(backend.view(privateBuffer, 0, 8)),
            "private buffer view identity depended on CPU visibility");
    privateBuffer = MetalBuffer{};
    require(backend.memoryStats().allocatedBytes == 0,
            "private allocation release was not tracked");

    constexpr uint64_t kSparseTileBytes =
        splash::kv::kSparseMappingAlignmentBytes;
    static_assert(kSparseTileBytes == 64 * 1024,
                  "KV mapping alignment must match the 64 KiB sparse tile");
    requireBackendError(
        [&] { (void)backend.allocatePlacementSparseBuffer(16 * 1024, 16 * 1024,
                                                          "sparse-16k"); },
        "a 16 KiB sparse tile was accepted");
    MetalBuffer sparse = backend.allocatePlacementSparseBuffer(
        kSparseTileBytes, kSparseTileBytes, "sparse-test");
    require(sparse && sparse.contents() == nullptr,
            "placement-sparse allocation is not private");
    stats = backend.memoryStats();
    require(stats.sparseVirtualBytes == kSparseTileBytes,
            "sparse virtual bytes were not tracked");
    require(stats.sparseTileBytes == kSparseTileBytes &&
                stats.pendingSparseUnmaps == 0 &&
                stats.completedSparseUnmaps == 0,
            "sparse tile and unmap telemetry were not reported");
    require(stats.sparseResidentBytes == 0,
            "sparse virtual allocation committed physical memory");

    auto heap = backend.allocatePlacementHeap(
        kSparseTileBytes, kSparseTileBytes, "sparse-test-heap");
    require(heap && heap.sizeBytes() >= kSparseTileBytes,
            "placement heap allocation failed");
    stats = backend.memoryStats();
    require(stats.sparseResidentBytes == heap.sizeBytes(),
            "sparse resident bytes were not tracked");
    require(stats.peakSparseResidentBytes == heap.sizeBytes(),
            "sparse resident peak was not tracked");
    require(stats.peakResidentBytes ==
                std::max(stats.peakAllocatedBytes, heap.sizeBytes()),
            "disjoint dense/sparse peaks were added together");

    SparseMapping sparseMapping{sparse, 0, kSparseTileBytes, 0};
    backend.mapSparse(heap, {&sparseMapping, 1});
    MetalBuffer readback = backend.allocateBuffer(
        sizeof(uint32_t) * kElementCount, BufferStorage::Shared,
        "sparse-readback");
    stats = backend.memoryStats();
    require(stats.peakResidentBytes ==
                stats.allocatedBytes + stats.sparseResidentBytes,
            "simultaneous dense/sparse physical peak was not tracked");
    constexpr uint32_t kSparseValue = 91;
    ComputeDispatch sparseDispatch;
    sparseDispatch.pipelineName = "sparse_fill_copy_u32";
    sparseDispatch.buffers = {{0, sparse}, {1, readback}};
    sparseDispatch.bytes = {
        {2, &kElementCount, sizeof(kElementCount)},
        {3, &kSparseValue, sizeof(kSparseValue)}};
    sparseDispatch.threadgroups = {1, 1, 1};
    sparseDispatch.threadsPerThreadgroup = {kElementCount, 1, 1};
    (void)backend.submit(sparseDispatch);
    auto *readbackValues = static_cast<uint32_t *>(readback.contents());
    for (uint32_t index = 0; index < kElementCount; ++index) {
        require(readbackValues[index] == kSparseValue + index,
                "sparse mapping was not visible to the compute queue");
    }

    // Unmapping is asynchronous: the backend owns the heap until the sparse
    // queue reports completion, and the caller's handle is left empty.
    requireBackendError(
        [&] { backend.unmapSparse({&sparseMapping, 1}, splash::metal::SparseHeap{}); },
        "unmap without the mapped heap was accepted");
    require(heap && backend.memoryStats().sparseResidentBytes == heap.sizeBytes(),
            "rejected unmap changed heap ownership or accounting");
    backend.unmapSparse({&sparseMapping, 1}, std::move(heap));
    require(!heap, "asynchronous unmap left the caller a heap handle");
    require(backend.memoryStats().pendingSparseUnmaps <= 1,
            "more than one sparse unmap was outstanding");
    backend.drainSparseUnmaps();
    awaitSparseRelease(backend, 0);
    require(!backend.sparseUnmapPending(), "drained unmap remains pending");
    stats = backend.memoryStats();
    require(stats.sparseResidentBytes == 0 && stats.pendingSparseUnmaps == 0 &&
                stats.completedSparseUnmaps == 1 &&
                stats.pendingSparseUnmapSeconds == 0.0 &&
                std::isfinite(stats.lastSparseUnmapSeconds) &&
                stats.lastSparseUnmapSeconds >= 0.0 &&
                stats.maxSparseUnmapSeconds >= stats.lastSparseUnmapSeconds,
            "completed unmap did not release its heap or record its timing");
    sparseDispatch = {};
    sparseMapping.buffer = {};
    sparse = {};
    awaitSparseRelease(backend, 0, 0);
    require(backend.memoryStats().sparseVirtualBytes == 0,
            "released sparse address space remains tracked");

    sharedMemoryCompletionLifetime(backend);
    sparsePacedRelease(backend);
    sparseExtentChurn(backend);
    require(capabilities.gpuCoreCount >= 1 && capabilities.gpuCoreCount <= 4096,
            "GPU core count was not read from the IORegistry");
    require(capabilities.meetsMinimumMacos(),
            "the running macOS version was not recorded");

    std::cout << "PASS MetalBackend device=\"" << capabilities.deviceName
              << "\" apple_gpu_family=" << capabilities.appleGpuFamily
              << " gpu_core_count=" << capabilities.gpuCoreCount
              << " macos=" << capabilities.macosVersion()
              << " recommended_working_set="
              << capabilities.recommendedMaxWorkingSetBytes
              << " last_wall_seconds=" << lastWallSeconds << '\n';
}

}  // namespace

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            std::cerr << "usage: metal_backend_test <test.metallib>\n";
            return 2;
        }
        try {
            completionDoesNotWaitForMemoryTelemetry(argv[1]);
            terminalCommandRecovers(argv[1], false);
            terminalCommandRecovers(argv[1], false, true);
            terminalCommandRecovers(argv[1], true);
            pendingCommandStillTimesOut(argv[1]);
            run(argv[1]);
        } catch (const std::exception &error) {
            std::cerr << "FAIL: unexpected exception: " << error.what()
                      << '\n';
            return 1;
        }
    }
    return 0;
}
