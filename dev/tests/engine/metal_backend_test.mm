#include "../../../runtime/metal/BackendInstrumentation.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestStderr.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using splash::metal::AllocationFailure;
using splash::metal::BackendInstrumentation;
using splash::metal::MetalAllocationError;
using splash::metal::BufferBinding;
using splash::metal::BufferReach;
using splash::metal::BufferStorage;
using splash::metal::BytesBinding;
using splash::metal::Command;
using splash::metal::CommandTicket;
using splash::metal::CommandTiming;
using splash::metal::ComputeDispatch;
using splash::metal::EventStep;
using splash::metal::HostStepGpu;
using splash::metal::HostStepTiming;
using splash::metal::IndirectGrid;
using splash::metal::MetalBackend;
using splash::metal::MetalBackendError;
using splash::metal::SharedEvent;
using splash::test::TemporaryDirectory;
using splash::test::capturedStderr;
using splash::test::rejects;
using splash::test::sharedBuffer;
using splash::test::writeFile;
using splash::metal::MetalBuffer;
using splash::metal::kResidencyKeepAliveSeconds;

// The watchdog limit of the tests whose commands the GPU holds for 200 ms.
constexpr double kShortCommandTimeoutSeconds = 0.1;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

// A dispatch of test_add_u32 that adds `increment` to the first `count`
// words of `buffer`.
ComputeDispatch addition(const MetalBuffer &buffer, const uint32_t &count,
                         const uint32_t &increment) {
    return {"test_add_u32", {{0, buffer}},
            {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
            {1, 1, 1}, {1, 1, 1}};
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

class MethodReplacement final {
public:
    MethodReplacement(id object, SEL selector, IMP replacement) {
        method_ = class_getInstanceMethod(object_getClass(object), selector);
        require(method_ != nullptr, "replacement method is missing");
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
    MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    *static_cast<uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch;
    dispatch.pipelineName = "test_add_u32";
    dispatch.buffers = {{0, buffer}};
    dispatch.bytes = {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}};
    dispatch.threadgroups = {1, 1, 1};
    dispatch.threadsPerThreadgroup = {1, 1, 1};
    // Creates the pipeline, whose creation samples memory, before counting.
    (void)backend.submit(dispatch);
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
    auto ticket = backend.submitCommandAsync({&dispatch, 1});
    const bool sampledOnSubmission = memoryQueries != 0;
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
    require(!sampledOnSubmission,
            "submission sampled device memory while the GPU idled");
    require(ready && healthy && completionMemoryQueries == 0,
            "completed GPU work depends on memory telemetry and can trip the watchdog");
    require(memoryQueries > queriesBeforeConsumption,
            "consuming a completed command did not refresh admission telemetry");
    const unsigned queriesAfterConsumption = memoryQueries;
    (void)ticket.wait();
    require(memoryQueries == queriesAfterConsumption,
            "an already-released ticket queried device memory again");
    require(*static_cast<uint32_t *>(buffer.contents()) == 2 * increment,
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

std::atomic<unsigned> commits{0};
IMP originalCountedCommit = nullptr;
void countCommit(id command, SEL selector) {
    ++commits;
    reinterpret_cast<void (*)(id, SEL)>(originalCountedCommit)(command, selector);
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
    MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
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
        auto ticket = backend.submitCommandAsync({&dispatch, 1}, [&] { ++notifications; });
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
            buffer = sharedBuffer(backend, sizeof(uint32_t));
            *static_cast<uint32_t *>(buffer.contents()) = 0;
            dispatch.buffers = {{0, buffer}};
            if (pendingNext) {
                commandWatchdogGate = [MTLCreateSystemDefaultDevice() newSharedEvent];
                MethodReplacement commit(command, @selector(commit),
                                         reinterpret_cast<IMP>(commitBehindWatchdogGate));
                originalCommandCommit = commit.original;
                next = backend.submitCommandAsync({&dispatch, 1});
            } else {
                next = backend.submitCommandAsync({&dispatch, 1});
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
        } else if (!failed) {
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
        rejects([&] { (void)backend.submitCommandAsync({&dispatch, 1}); },
                "Metal backend is unhealthy", "failed GPU command admitted further work");
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
    MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
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
        ticket = backend.submitCommandAsync({&dispatch, 1});
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
    rejects([&] { (void)backend.submitCommandAsync({&dispatch, 1}); },
            "Metal backend is unhealthy", "timed-out backend accepted more work");
    require(*static_cast<uint32_t *>(buffer.contents()) == increment,
            "timed-out command lost resources before GPU completion");

    // A command split at a signal, whose wait the other agent never meets,
    // times out the same way.
    MetalBackend split(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
    auto splitBuffer = sharedBuffer(split, sizeof(uint32_t));
    const ComputeDispatch splitAddition = addition(splitBuffer, count, increment);
    const SharedEvent event = split.newSharedEvent();
    const std::vector<EventStep> events{{1, event, 1, EventStep::Kind::Signal},
                                        {1, event, 2, EventStep::Kind::Wait}};
    CommandTicket splitTicket =
        split.submitCommandAsync(Command{{&splitAddition, 1}, events});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::string splitFailure;
    try { split.checkHealth(); }
    catch (const MetalBackendError &error) { splitFailure = error.what(); }
    event.signal(2);
    (void)splitTicket.wait();
    require(!split.healthy() &&
                splitFailure.find("completion timed out") != std::string::npos &&
                splitFailure.find("sequence=1") != std::string::npos &&
                splitFailure.find("dispatches=1") != std::string::npos,
            "a split command's timeout lost its submission diagnostics: " + splitFailure);
    std::cout << "PASS pending GPU command watchdog and resource lifetime\n";
}

// Polls until the backend holds no allocations.
bool awaitAllocationsReleased(const MetalBackend &backend) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (backend.memoryStats().allocatedBytes != 0 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return backend.memoryStats().allocatedBytes == 0;
}

// A synchronous submission throws once the watchdog gives up on its command,
// which keeps its allocations until the GPU ends it.
void synchronousWaitObeysTheWatchdog(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    std::future<std::string> submitted;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        submitted = std::async(std::launch::async, [&]() -> std::string {
            try {
                (void)backend.submitCommandAsync({&dispatch, 1}).wait();
            } catch (const MetalBackendError &error) {
                return error.what();
            }
            return "the command completed";
        });
        require(submitted.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "a synchronous wait outlasted the command watchdog");
    }
    const std::string failure = submitted.get();
    require(failure.find("completion timed out") != std::string::npos && !backend.healthy(),
            "a synchronous wait did not fail with the command timeout: " + failure);
    dispatch.buffers.clear();
    buffer = {};
    require(backend.memoryStats().allocatedBytes != 0,
            "an abandoned wait released the allocations of a pending command");
    commandWatchdogGate.signaledValue = 1;
    const bool released = awaitAllocationsReleased(backend);
    commandWatchdogGate = nil;
    require(released, "an abandoned command kept its allocations after the GPU ended it");
    std::cout << "PASS synchronous wait obeys the command watchdog\n";
}

// Destroying a ticket whose command the watchdog gave up on returns while the
// command is still pending, and the command completes once the GPU ends it.
void abandonedTicketReturnsAfterTheWatchdog(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    std::atomic<bool> completed{false};
    splash::metal::CommandTicket ticket;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        ticket = backend.submitCommandAsync({&dispatch, 1}, [&] { completed = true; });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // A failure exits here: unwinding would destroy the ticket, which waits for
    // the watchdog this checks.
    try {
        rejects([&] { backend.checkHealth(); }, "Metal command completion timed out",
                "the watchdog did not give up on a pending command");
    } catch (const std::exception &error) {
        fail(error.what());
    }
    auto destroyed = std::async(std::launch::async, [&ticket] {
        splash::metal::CommandTicket dropped = std::move(ticket);
    });
    require(destroyed.wait_for(std::chrono::seconds(3)) == std::future_status::ready,
            "destroying an abandoned ticket waited for its command");
    dispatch.buffers.clear();
    buffer = {};
    require(!completed && backend.memoryStats().allocatedBytes != 0,
            "an abandoned ticket let go of a pending command's allocations");
    commandWatchdogGate.signaledValue = 1;
    const bool released = awaitAllocationsReleased(backend);
    commandWatchdogGate = nil;
    require(completed && released,
            "an abandoned command did not complete once the GPU ended it");
    std::cout << "PASS abandoned ticket returns after the command watchdog\n";
}

// A stopped backend refuses the next submission before encoding it, and
// stopping is not a failure: the backend stays healthy with nothing in flight.
void stopRefusesSubmission(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    const uint32_t count = 1, increment = 7;
    const ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    backend.stop();
    rejects([&] { (void)backend.submitCommandAsync({&dispatch, 1}); },
            "Metal backend is stopping", "a stopped backend accepted a submission");
    require(backend.healthy() && !backend.commandInFlight(),
            "refusing a submission while stopping marked the backend unhealthy "
            "or left a command in flight");
    std::cout << "PASS stopped backend refuses submission\n";
}

// A shutdown gives up a synchronous wait as the watchdog does: the wait
// throws and the backend is unhealthy, and the command keeps its allocations
// until the GPU ends it.
void shutdownInterruptsACommandWait(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    std::atomic<bool> stop{false};
    backend.setWaitInterrupt([&] { return stop.load(); });
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    std::future<std::string> submitted;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        submitted = std::async(std::launch::async, [&]() -> std::string {
            try {
                (void)backend.submitCommandAsync({&dispatch, 1}).wait();
            } catch (const MetalBackendError &error) {
                return error.what();
            }
            return "the command completed";
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        stop = true;
        require(submitted.wait_for(std::chrono::seconds(3)) == std::future_status::ready,
                "a shutdown did not end a synchronous wait");
    }
    const std::string failure = submitted.get();
    require(failure.find("shutdown") != std::string::npos && !backend.healthy(),
            "a shutdown did not fail the synchronous wait: " + failure);
    dispatch.buffers.clear();
    buffer = {};
    require(backend.memoryStats().allocatedBytes != 0,
            "an interrupted wait released the allocations of a pending command");
    commandWatchdogGate.signaledValue = 1;
    const bool released = awaitAllocationsReleased(backend);
    commandWatchdogGate = nil;
    require(released, "an interrupted command kept its allocations after the GPU ended it");
    std::cout << "PASS shutdown interrupts a command wait\n";
}

// Destroying a ticket during a shutdown still waits for its command: only
// wait() gives a command up for a shutdown, and only the watchdog judges the
// backend's health meanwhile.
void shutdownLeavesTicketTeardownToTheCommand(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    backend.setWaitInterrupt([] { return true; });
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    splash::metal::CommandTicket ticket;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        ticket = backend.submitCommandAsync({&dispatch, 1});
    }
    auto destroyed = std::async(std::launch::async, [&ticket] {
        splash::metal::CommandTicket dropped = std::move(ticket);
    });
    // Longer than a wait slice, after which a wait for wait() would give up.
    const bool waited = destroyed.wait_for(std::chrono::milliseconds(1500)) ==
                        std::future_status::timeout;
    backend.checkHealth();
    commandWatchdogGate.signaledValue = 1;
    const bool returned = destroyed.wait_for(std::chrono::seconds(5)) ==
                          std::future_status::ready;
    commandWatchdogGate = nil;
    require(waited && returned && backend.healthy(),
            "a shutdown gave up a command its ticket's teardown waits for");
    std::cout << "PASS shutdown leaves ticket teardown to the command\n";
}

// A command that completes while its wait asks about a shutdown stays a
// success: the wait gives up only a command still unfinished, so the backend
// stays healthy.
void shutdownSparesACommandThatCompletes(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    const uint32_t count = 1, increment = 7;
    ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    commandWatchdogGate = [device newSharedEvent];
    std::atomic<bool> completed{false};
    // The command completes after the wait's slice ended, before the wait
    // decides.
    backend.setWaitInterrupt([&] {
        commandWatchdogGate.signaledValue = 1;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!completed && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    });
    splash::metal::CommandTicket ticket;
    {
        MethodReplacement commit(command, @selector(commit),
                                 reinterpret_cast<IMP>(commitBehindWatchdogGate));
        originalCommandCommit = commit.original;
        ticket = backend.submitCommandAsync({&dispatch, 1}, [&] { completed = true; });
    }
    std::string failure;
    try {
        (void)ticket.wait();
    } catch (const MetalBackendError &error) {
        failure = error.what();
    }
    commandWatchdogGate = nil;
    require(completed && failure.empty() && backend.healthy(),
            "a shutdown gave up a command that completed: " + failure);
    std::cout << "PASS shutdown spares a command that completes\n";
}

// Profiling times every dispatch of every command, a command of one dispatch
// included, and hands back a completed ticket with their summed time.
void dispatchProfilingCoversEveryCommand(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, increment = 1;
    const ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    BackendInstrumentation::setDispatchProfiling(backend, true);
    for (const size_t dispatches : {1, 2}) {
        const std::vector<ComputeDispatch> command(dispatches, dispatch);
        bool notified = false;
        auto ticket =
            backend.submitCommandAsync(command, [&] { notified = true; });
        require(notified && ticket.ready(),
                "a profiled command was not complete on return");
        const auto timing = ticket.wait();
        const auto profile = BackendInstrumentation::takeDispatchProfile(backend);
        double total = 0.0;
        for (const auto &entry : profile) {
            require(entry.pipelineName == dispatch.pipelineName,
                    "a profiled dispatch lost its pipeline name");
            total += entry.gpuSeconds;
        }
        require(profile.size() == dispatches && timing.gpuSeconds == total,
                "profiling skipped a dispatch or misreported the command");
    }
    // A command without event steps profiles as its dispatches do; one with
    // them is refused.
    const std::vector<ComputeDispatch> pair(2, dispatch);
    auto ticket = backend.submitCommandAsync(Command{pair, {}});
    require(ticket.ready() &&
                BackendInstrumentation::takeDispatchProfile(backend).size() == 2,
            "a command without event steps was not profiled");
    (void)ticket.wait();
    const EventStep step{1, backend.newSharedEvent(), 1, EventStep::Kind::Signal};
    rejects([&] { (void)backend.submitCommandAsync(Command{pair, {&step, 1}}); },
            "does not replay event steps",
            "profiling replayed a command without its event steps");
    BackendInstrumentation::setDispatchProfiling(backend, false);
    require(*value == 5, "profiling did not run each dispatch exactly once");
    std::cout << "PASS dispatch profiling covers every command\n";
}

// Preparing a dispatch compiles its pipeline without submitting anything, so
// its submission compiles nothing, and rejects what submission would.
void preparedPipelinesCompileAhead(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    auto buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, increment = 1;
    const ComputeDispatch dispatch{"test_add_u32", {{0, buffer}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    backend.preparePipelines({&dispatch, 1});
    require(BackendInstrumentation::cachedPipelines(backend) == 1 &&
                BackendInstrumentation::submittedCommands(backend) == 0 &&
                *value == 0,
            "preparing a dispatch did more than compile its pipeline");
    (void)backend.submit(dispatch);
    require(BackendInstrumentation::cachedPipelines(backend) == 1 && *value == 1,
            "submitting a prepared dispatch compiled its pipeline again");
    ComputeDispatch oversized = dispatch;
    oversized.threadsPerThreadgroup = {4096, 1, 1};
    rejects([&] { backend.preparePipelines({&oversized, 1}); },
            "threadsPerThreadgroup exceeds pipeline capability",
            "a dispatch past its pipeline's thread limit was prepared");
    ComputeDispatch missing = dispatch;
    missing.pipelineName = "does_not_exist";
    rejects([&] { backend.preparePipelines({&missing, 1}); },
            "missing Metal function: does_not_exist",
            "a dispatch of a missing function was prepared");
    require(backend.healthy() &&
                BackendInstrumentation::submittedCommands(backend) == 1,
            "a rejected preparation poisoned the backend or submitted work");
    std::cout << "PASS prepared pipelines compile ahead\n";
}

std::atomic<unsigned> blitEncoders{0};
std::atomic<unsigned> computeEncoders{0};
IMP originalBlitEncoder = nullptr;
IMP originalComputeEncoder = nullptr;
id countBlitEncoder(id command, SEL selector) {
    ++blitEncoders;
    return reinterpret_cast<id (*)(id, SEL)>(originalBlitEncoder)(command, selector);
}
id countComputeEncoder(id command, SEL selector) {
    ++computeEncoders;
    return reinterpret_cast<id (*)(id, SEL)>(originalComputeEncoder)(command, selector);
}

// Counts what residency sets ask of Metal while it lives, passing every call
// on: requests and ends of residency, and removals of a member. All sets
// share the class of the one it creates. Arm it before the backend it
// observes, so that no call races the swap.
class ResidencyCalls final {
public:
    ResidencyCalls() {
        originalRequest = request_.original;
        originalEnd = end_.original;
        originalRemoval = removal_.original;
        requests = ends = removals = 0;
    }

    static inline std::atomic<unsigned> requests{0};
    static inline std::atomic<unsigned> ends{0};
    static inline std::atomic<unsigned> removals{0};

private:
    static inline IMP originalRequest = nullptr;
    static inline IMP originalEnd = nullptr;
    static inline IMP originalRemoval = nullptr;

    static void countRequest(id set, SEL selector) {
        ++requests;
        reinterpret_cast<void (*)(id, SEL)>(originalRequest)(set, selector);
    }
    static void countEnd(id set, SEL selector) {
        ++ends;
        reinterpret_cast<void (*)(id, SEL)>(originalEnd)(set, selector);
    }
    static void countRemoval(id set, SEL selector, id allocation) {
        ++removals;
        reinterpret_cast<void (*)(id, SEL, id)>(originalRemoval)(set, selector, allocation);
    }

    id<MTLResidencySet> set_ = [MTLCreateSystemDefaultDevice()
        newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:nil];
    MethodReplacement request_{set_, @selector(requestResidency),
                               reinterpret_cast<IMP>(countRequest)};
    MethodReplacement end_{set_, @selector(endResidency), reinterpret_cast<IMP>(countEnd)};
    MethodReplacement removal_{set_, @selector(removeAllocation:),
                               reinterpret_cast<IMP>(countRemoval)};
};

// Polls until `done` holds or `limit` passes, and returns whether it held.
template <typename Predicate>
bool waitFor(Predicate done, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!done() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return done();
}

// A lapsed keep-alive ends residency with one dispatch of a kernel built with
// the library, never a blit whose driver program compiles at that moment, and
// destroying a backend that holds its set submits no GPU work at all.
void residencyEndsWithoutBlits(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 0.2;
    id<MTLCommandBuffer> command =
        [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    commits = 0;
    blitEncoders = 0;
    computeEncoders = 0;
    MethodReplacement committing(command, @selector(commit),
                                 reinterpret_cast<IMP>(countCommit));
    originalCountedCommit = committing.original;
    MethodReplacement blits(command, @selector(blitCommandEncoder),
                            reinterpret_cast<IMP>(countBlitEncoder));
    originalBlitEncoder = blits.original;
    MethodReplacement computes(command, @selector(computeCommandEncoder),
                               reinterpret_cast<IMP>(countComputeEncoder));
    originalComputeEncoder = computes.original;
    ResidencyCalls calls;
    unsigned lapseCommits = 0, lapseComputes = 0;
    {
        MetalBackend backend(metallibPath, kKeepAliveSeconds);
        const uint64_t page = static_cast<uint64_t>(getpagesize());
        MetalBuffer lapsing = sharedBuffer(backend, page);
        (void)waitFor([] { return commits != 0; }, std::chrono::seconds(5));
        lapseCommits = commits.exchange(0);
        lapseComputes = computeEncoders.exchange(0);
        // Allocating another buffer holds the set again for the teardown.
        const unsigned requested = calls.requests;
        MetalBuffer held = sharedBuffer(backend, page);
        require(waitFor([&] { return calls.requests > requested; }, std::chrono::seconds(1)),
                "an allocation did not hold the set");
    }
    require(lapseCommits == 1 && lapseComputes == 1,
            "a lapsed keep-alive did not end residency with one compute dispatch");
    require(commits == 0 && computeEncoders == 0 && calls.ends == 2,
            "backend teardown submitted GPU work or did not end residency");
    require(blitEncoders == 0, "residency encoded a blit");
    std::cout << "PASS residency ends without blits\n";
}

// A buffer's memory returns once its last view is gone, and a set the
// backend still holds lets its buffers go with the backend, though the
// serving thread's autorelease pool, like this one, never drains.
void residencyReturnsRemovedBuffers(const std::string &metallibPath) {
    constexpr uint64_t kBytes = 64ull << 20;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    {
        MetalBackend backend(metallibPath);
        const uint64_t before = device.currentAllocatedSize;
        MetalBuffer buffer = sharedBuffer(backend, kBytes);
        buffer = {};
        require(device.currentAllocatedSize <= before + (1ull << 20),
                "a buffer taken out of the residency set kept its memory");
    }
    const uint64_t before = device.currentAllocatedSize;
    {
        MetalBuffer buffer;
        {
            MetalBackend backend(metallibPath);
            buffer = sharedBuffer(backend, kBytes);
        }
    }
    require(device.currentAllocatedSize <= before + (32ull << 20),
            "a destroyed backend's residency set kept its buffers");
    std::cout << "PASS residency returns removed buffers\n";
}

// Every buffer is held from its allocation until the keep-alive passes
// without a command, the next command holds it again at once, and its last
// view takes it out of the set.
void buffersStayResident(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 1.0;
    ResidencyCalls calls;
    MetalBackend backend(metallibPath, kKeepAliveSeconds);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    const auto start = std::chrono::steady_clock::now();
    MetalBuffer dropped = backend.view(sharedBuffer(backend, page), 0, 64);
    MetalBuffer used = sharedBuffer(backend, page);
    const uint64_t each = backend.memoryStats().allocatedBytes / 2;
    require(waitFor([&] { return calls.requests != 0; }, std::chrono::seconds(1)),
            "buffers were not held at once");
    (void)waitFor([&] { return calls.ends != 0; }, std::chrono::seconds(5));
    const std::chrono::duration<double> lapsedAfter = std::chrono::steady_clock::now() - start;
    require(calls.ends == 1 && lapsedAfter.count() >= kKeepAliveSeconds,
            "buffers did not lapse once the keep-alive passed without a command");
    dropped = {};
    require(calls.removals == 1 && backend.memoryStats().allocatedBytes == each,
            "a buffer whose last view is gone is still a member");
    const uint32_t count = 1, increment = 7;
    *static_cast<uint32_t *>(used.contents()) = 0;
    ComputeDispatch dispatch{"test_add_u32", {{0, used}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    const unsigned requested = calls.requests;
    auto ticket = backend.submitCommandAsync({&dispatch, 1});
    require(waitFor([&] { return calls.requests > requested; }, std::chrono::seconds(1)),
            "a command did not hold the buffers again");
    (void)ticket.wait();
    require(*static_cast<uint32_t *>(used.contents()) == increment,
            "a command on a resident buffer produced the wrong result");
    std::cout << "PASS buffers stay resident keep_alive_seconds=" << kKeepAliveSeconds
              << " lapsed_after_seconds=" << lapsedAfter.count() << '\n';
}

// An infinite keep-alive (--idle-release off) holds the set while the backend
// lives: the heartbeat goes on requesting residency and never ends it. A
// keep-alive that is not positive is refused.
void infiniteKeepAliveHoldsResidency(const std::string &metallibPath) {
    ResidencyCalls calls;
    {
        MetalBackend backend(metallibPath, std::numeric_limits<double>::infinity());
        MetalBuffer held = sharedBuffer(backend, static_cast<uint64_t>(getpagesize()));
        require(waitFor([&] { return calls.requests >= 3; }, std::chrono::seconds(5)),
                "the heartbeat stopped requesting an infinitely kept set");
        require(calls.ends == 0, "an infinite keep-alive ended residency");
    }
    for (double keepAlive : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()})
        rejects([&] { MetalBackend backend(metallibPath, keepAlive); },
                "residency keep-alive must be positive",
                "a keep-alive that is not positive was accepted");
    std::cout << "PASS infinite keep-alive holds residency\n";
}

// Allocating into a held set wires the buffer at the set's commit: the
// serving thread asks Metal for no residency of its own, and only the
// heartbeat requests the set while allocations follow one another.
void allocationDoesNotRequestResidency(const std::string &metallibPath) {
    constexpr uint32_t kAllocations = 20;
    constexpr double kBeatSeconds = 0.5;
    ResidencyCalls calls;
    MetalBackend backend(metallibPath);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    MetalBuffer used = sharedBuffer(backend, page);
    const uint32_t count = 1, increment = 7;
    const ComputeDispatch dispatch{"test_add_u32", {{0, used}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    (void)backend.submitCommandAsync({&dispatch, 1}).wait();
    require(waitFor([&] { return calls.requests != 0; }, std::chrono::seconds(1)),
            "the set was not held before the allocations");
    const unsigned held = calls.requests;
    const auto start = std::chrono::steady_clock::now();
    std::vector<MetalBuffer> buffers;
    for (uint32_t index = 0; index < kAllocations; ++index)
        buffers.push_back(sharedBuffer(backend, page));
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    const unsigned requests = calls.requests - held;
    require(requests <= 1 + elapsed.count() / kBeatSeconds,
            "allocations requested residency: " + std::to_string(requests) +
                " requests in " + std::to_string(elapsed.count()) + " s");
    std::cout << "PASS allocation does not request residency allocations=" << kAllocations
              << " requests=" << requests << '\n';
}

// Allocating, lapsing and holding again race the heartbeat while another
// thread drops buffers, as command completion can, and the backend is then
// destroyed with its heartbeat live and a buffer outliving it. Nothing may
// block, and every command must see its buffer.
void residencyRacesTheHeartbeat(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 0.05;
    constexpr int kRounds = 24;
    ResidencyCalls calls;
    auto backend = std::make_unique<MetalBackend>(metallibPath, kKeepAliveSeconds);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    MetalBuffer used = sharedBuffer(*backend, page);
    *static_cast<uint32_t *>(used.contents()) = 0;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<MetalBuffer> handed;
    bool finished = false;
    std::thread dropper([&] {
        std::unique_lock lock(mutex);
        while (!finished || !handed.empty()) {
            ready.wait(lock, [&] { return finished || !handed.empty(); });
            std::vector<MetalBuffer> drop = std::move(handed);
            handed.clear();
            lock.unlock();
            drop.clear();
            lock.lock();
        }
    });
    const uint32_t count = 1, increment = 1;
    ComputeDispatch dispatch{"test_add_u32", {{0, used}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    int lapses = 0;
    for (int round = 0; round < kRounds; ++round) {
        MetalBuffer member = sharedBuffer(*backend, page);
        {
            std::lock_guard lock(mutex);
            handed.push_back(std::move(member));
        }
        ready.notify_one();
        // Every third round lets the heartbeat end residency, so that its
        // command holds the set again.
        if (round % 3 == 2) {
            const unsigned ended = calls.ends;
            lapses += waitFor([&] { return calls.ends > ended; }, std::chrono::seconds(2));
        }
        (void)backend->submitCommandAsync({&dispatch, 1}).wait();
    }
    {
        std::lock_guard lock(mutex);
        finished = true;
    }
    ready.notify_one();
    dropper.join();
    require(*static_cast<uint32_t *>(used.contents()) == kRounds,
            "a command racing the residency heartbeat produced the wrong result");
    require(lapses == kRounds / 3, "residency did not lapse between the racing commands");
    (void)backend->submitCommandAsync({&dispatch, 1}).wait();
    backend.reset();
    used = {};
    std::cout << "PASS residency races the heartbeat rounds=" << kRounds
              << " lapses=" << lapses << '\n';
}

// Records the residency sets added to command queues.
class QueueSets final {
public:
    QueueSets() {
        original = replacement_.original;
        added.clear();
    }

    static inline std::vector<id<MTLResidencySet>> added;

private:
    static inline IMP original = nullptr;

    static void recordAdd(id queue, SEL selector, id<MTLResidencySet> set) {
        added.push_back(set);
        reinterpret_cast<void (*)(id, SEL, id)>(original)(queue, selector, set);
    }

    id<MTLCommandQueue> queue_ = [MTLCreateSystemDefaultDevice() newCommandQueue];
    MethodReplacement replacement_{queue_, @selector(addResidencySet:),
                                   reinterpret_cast<IMP>(recordAdd)};
};

// The command queue holds one residency set, of the buffers kernels reach
// only through their addresses, and none other: a command buffer waits for
// its queue's set to be resident before it starts (Residency). Only a Shared
// buffer is reached by its address, and only such a buffer has one, zero
// while its memory is released.
void queueHoldsAddressedBuffers(const std::string &metallibPath) {
    QueueSets sets;
    MetalBackend backend(metallibPath);
    require(sets.added.size() == 1, "the command queue does not hold exactly one residency set");
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    MetalBuffer bound = backend.allocateBuffer(page, BufferStorage::Shared, "bound");
    MetalBuffer addressed = backend.allocateBuffer(page, BufferStorage::Shared, "addressed",
                                                   BufferReach::Addressed);
    NSArray<id<MTLAllocation>> *held = sets.added.front().allAllocations;
    require(held.count == 1 && [((id<MTLBuffer>)held.firstObject).label isEqualToString:@"addressed"],
            "the command queue's set does not hold exactly the addressed buffer");
    rejects([&] {
                (void)backend.allocateBuffer(page, BufferStorage::Private, "private", BufferReach::Addressed);
            },
            "only a Shared Metal buffer is reached by its address",
            "a Private buffer was reached by its address");
    rejects([&] { (void)bound.gpuAddress(); }, "allocated BufferReach::Addressed has a GPU address",
            "a buffer kernels reach bound had a GPU address");
    require(addressed.gpuAddress(), "an addressed buffer has no GPU address");
    backend.releaseMemory(addressed);
    require(!addressed.gpuAddress(), "released memory kept its GPU address");
    std::cout << "PASS the command queue holds the addressed buffers\n";
}

// Kernels reach shared buffers only through GPU addresses in a table, as
// they reach KV extents (BufferReach::Addressed). The command queue's
// residency set makes the buffers resident for every command, also once the
// backend's set has lapsed; one dispatch reads what the previous one wrote
// through them; the CPU reads what kernels wrote and kernels read what the CPU
// wrote between commands; and buffers released and allocated again between
// commands work at once.
void buffersReachedThroughTables(const std::string &metallibPath) {
    constexpr double kKeepAliveSeconds = 0.2;
    constexpr uint32_t kBuffers = 6, kWords = 16384, kRounds = 60;
    constexpr uint64_t kBytes = uint64_t{kWords} * sizeof(uint32_t);
    ResidencyCalls calls;
    MetalBackend backend(metallibPath, kKeepAliveSeconds);
    MetalBuffer table = sharedBuffer(backend, kBuffers * sizeof(uint64_t));
    MetalBuffer mismatches = sharedBuffer(backend, sizeof(uint32_t));
    const uint64_t before = backend.memoryStats().allocatedBytes;
    const auto addressedBuffer = [&] {
        return backend.allocateBuffer(kBytes, BufferStorage::Shared, "addressed", BufferReach::Addressed);
    };
    std::vector<MetalBuffer> buffers(kBuffers);
    for (MetalBuffer &buffer : buffers) buffer = addressedBuffer();
    require(buffers[0].storage() == BufferStorage::Shared && buffers[0].contents() &&
                buffers[0].sizeBytes() == kBytes &&
                backend.memoryStats().allocatedBytes == before + kBuffers * kBytes,
            "buffers were not allocated or counted at their size");
    require(buffers[0].gpuAddress() &&
                backend.view(buffers[0], 4096, 4096).gpuAddress() ==
                    buffers[0].gpuAddress() + 4096,
            "a view's GPU address does not start at its offset");

    auto *entries = static_cast<uint64_t *>(table.contents());
    for (uint32_t index = 0; index < kBuffers; ++index)
        entries[index] = buffers[index].gpuAddress();
    const uint32_t words = kWords;
    uint32_t seed = 0;
    const std::array<ComputeDispatch, 2> command{
        ComputeDispatch{"addressed_write_u32", {{0, table}},
            {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
            {kWords / 256, kBuffers, 1}, {256, 1, 1}},
        ComputeDispatch{"addressed_check_u32", {{0, table}, {3, mismatches}},
            {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
            {kWords / 256, kBuffers, 1}, {256, 1, 1}}};
    const std::array<ComputeDispatch, 1> check{command[1]};
    const auto expected = [&](uint32_t index, uint32_t word) {
        return seed ^ (index * 131071u + word);
    };
    uint32_t regrown = 0;
    bool lapsedRound = false;
    for (uint32_t round = 0; round < kRounds; ++round) {
        if (round % 5 == 4) {
            const uint32_t index = round % kBuffers;
            buffers[index] = {};
            require(backend.memoryStats().allocatedBytes ==
                        before + (kBuffers - 1) * kBytes,
                    "a released buffer is still counted");
            buffers[index] = addressedBuffer();
            entries[index] = buffers[index].gpuAddress();
            ++regrown;
        }
        if (round == kRounds / 2) {
            const unsigned ended = calls.ends;
            require(waitFor([&] { return calls.ends > ended; }, std::chrono::seconds(5)),
                    "the buffers did not lapse with the residency set");
            lapsedRound = true;
        }
        seed = 0x9e3779b9u * (round + 1);
        *static_cast<uint32_t *>(mismatches.contents()) = 0;
        // Every third round the CPU writes what the check expects, and the
        // command only checks it.
        const bool hostWrites = round % 3 == 2;
        if (hostWrites) {
            for (uint32_t index = 0; index < kBuffers; ++index) {
                auto *contents = static_cast<uint32_t *>(buffers[index].contents());
                for (uint32_t word = 0; word < kWords; ++word)
                    contents[word] = expected(index, word);
            }
        }
        auto ticket = hostWrites ? backend.submitCommandAsync(check)
                                 : backend.submitCommandAsync(command);
        require(backend.commandInFlight(), "a submitted command is not in flight");
        (void)ticket.wait();
        require(!backend.commandInFlight(), "a consumed command is still in flight");
        require(*static_cast<uint32_t *>(mismatches.contents()) == 0,
                "round " + std::to_string(round) +
                    " read wrong data through the addresses of its buffers");
        for (uint32_t index = 0; index < kBuffers; ++index) {
            const auto *contents = static_cast<const uint32_t *>(buffers[index].contents());
            for (uint32_t word = 0; word < kWords; ++word) {
                if (contents[word] != expected(index, word))
                    fail("round " + std::to_string(round) +
                         ": the CPU read other data than the kernels wrote");
            }
        }
    }
    buffers.clear();
    // Each released buffer leaves both the backend's set and the queue's.
    require(backend.memoryStats().allocatedBytes == before &&
                calls.removals == 2 * (kBuffers + regrown),
            "released buffers stayed counted or in the residency sets");
    std::cout << "PASS buffers reached through tables rounds=" << kRounds
              << " regrown=" << regrown << " lapsed_round=" << lapsedRound << '\n';
}

// Bindings of consecutive entries reach Metal as one run, each buffer at its
// own offset, in whatever order the dispatch lists them.
void bindingRunsKeepOffsets(MetalBackend &backend) {
    constexpr uint32_t kWords = 8, count = 4;
    MetalBuffer source = sharedBuffer(backend, kWords * sizeof(uint32_t));
    MetalBuffer destination = sharedBuffer(backend, kWords * sizeof(uint32_t));
    auto *in = static_cast<uint32_t *>(source.contents());
    auto *out = static_cast<uint32_t *>(destination.contents());
    for (uint32_t word = 0; word < kWords; ++word) {
        in[word] = 100 + word;
        out[word] = 0;
    }
    const ComputeDispatch dispatch{"test_copy_u32",
        {{1, backend.view(destination, 4 * sizeof(uint32_t), 4 * sizeof(uint32_t))},
         {0, backend.view(source, 2 * sizeof(uint32_t), 4 * sizeof(uint32_t))}},
        {{2, &count, sizeof(count)}}, {1, 1, 1}, {count, 1, 1}};
    (void)backend.submit(dispatch);
    for (uint32_t word = 0; word < kWords; ++word) {
        require(out[word] == (word < 4 ? 0 : 98 + word),
                "a run of bindings lost a buffer's entry or offset");
    }
}

// An indirect dispatch runs the grid that an earlier dispatch of its command
// wrote, at the grid's offset into a view; a grid unaligned in its allocation
// or past its buffer is refused.
void indirectGridsRunTheWrittenGrid(MetalBackend &backend) {
    constexpr uint32_t kWords = 8, gridWords = 3, increment = 1;
    MetalBuffer grid = sharedBuffer(backend, 8 * sizeof(uint32_t));
    MetalBuffer written = sharedBuffer(backend, gridWords * sizeof(uint32_t));
    MetalBuffer values = sharedBuffer(backend, kWords * sizeof(uint32_t));
    auto *source = static_cast<uint32_t *>(written.contents());
    source[0] = 3;
    source[1] = source[2] = 1;
    std::fill_n(static_cast<uint32_t *>(grid.contents()), 8, 0u);
    std::fill_n(static_cast<uint32_t *>(values.contents()), kWords, 0u);
    // The grid {3, 1, 1} lands at byte 8 of the view from byte 4.
    const MetalBuffer view = backend.view(grid, 4, 7 * sizeof(uint32_t));
    const ComputeDispatch write{"test_copy_u32",
        {{0, written}, {1, backend.view(grid, 12, gridWords * sizeof(uint32_t))}},
        {{2, &gridWords, sizeof(gridWords)}}, {1, 1, 1}, {gridWords, 1, 1}};
    ComputeDispatch add{"test_add_u32", {{0, values}},
        {{1, &kWords, sizeof(kWords)}, {2, &increment, sizeof(increment)}},
        {kWords, 1, 1}, {1, 1, 1}, {view, 8}};
    const std::array<ComputeDispatch, 2> command{write, add};
    (void)backend.submitCommandAsync(command).wait();
    const auto *out = static_cast<const uint32_t *>(values.contents());
    for (uint32_t word = 0; word < kWords; ++word) {
        require(out[word] == (word < 3 ? increment : 0),
                "an indirect dispatch ran another grid than the one written");
    }
    // Byte 8 of a view from byte 2 lies at byte 10 of the buffer.
    const MetalBuffer unaligned = backend.view(grid, 2, 6 * sizeof(uint32_t));
    for (const IndirectGrid &outside : {IndirectGrid{view, 6},
                                        IndirectGrid{view, 20},
                                        IndirectGrid{unaligned, 8}}) {
        add.indirectThreadgroups = outside;
        rejects([&] { (void)backend.submit(add); }, "reads its grid outside",
                "an unaligned or out-of-range grid was accepted");
    }
}

// Released memory leaves the residency set and the accounting while the
// buffer's views stay handles that no command may bind; a restore gives the
// buffer memory again, which commands use as before. Only whole buffers with
// no command in flight are released, each once until restored.
void releasedMemory(const std::string &metallibPath) {
    ResidencyCalls calls;
    MetalBackend backend(metallibPath);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    const uint64_t before = backend.memoryStats().allocatedBytes;
    MetalBuffer buffer = sharedBuffer(backend, page);
    const MetalBuffer view = backend.view(buffer, 0, sizeof(uint32_t));
    const uint64_t bytes = backend.memoryStats().allocatedBytes - before;
    const uint32_t count = 1, increment = 7;
    const ComputeDispatch dispatch{"test_add_u32", {{0, view}},
        {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
        {1, 1, 1}, {1, 1, 1}};
    {
        auto ticket = backend.submitCommandAsync({&dispatch, 1});
        rejects([&] { backend.releaseMemory(buffer); }, "command is in flight",
                "memory was released under a command in flight");
        (void)ticket.wait();
    }
    rejects([&] { backend.releaseMemory(view); }, "base buffer", "a view's memory was released");
    rejects([&] { backend.restoreMemory(buffer); }, "not released", "allocated memory was restored");
    const unsigned removals = calls.removals;
    backend.releaseMemory(buffer);
    require(!buffer.contents() && !view.contents() &&
                backend.memoryStats().allocatedBytes == before && calls.removals == removals + 1,
            "released memory kept its contents, its accounting or its residency");
    rejects([&] { backend.releaseMemory(buffer); }, "already released", "memory was released twice");
    rejects([&] { (void)backend.submit(dispatch); }, "binds released memory",
            "a command bound released memory");
    backend.restoreMemory(buffer);
    require(view.contents() && view.sizeBytes() == sizeof(uint32_t) &&
                backend.memoryStats().allocatedBytes == before + bytes,
            "restored memory lacks its contents or its accounting");
    *static_cast<uint32_t *>(view.contents()) = 0;
    (void)backend.submit(dispatch);
    require(*static_cast<uint32_t *>(view.contents()) == increment,
            "a command on restored memory produced the wrong result");
    buffer = {};
    std::cout << "PASS released memory\n";
}

// Memory another agent shares, such as the Neural Engine's surfaces, wrapped
// without a copy: kernels and the CPU see the same bytes, the accounting
// counts them, and only their owner frees them.
void wrappedMemory(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    void *memory = nullptr;
    require(posix_memalign(&memory, page, page) == 0, "could not allocate a page");
    std::shared_ptr<void> owner(memory, std::free);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    MetalBuffer buffer = backend.wrapSharedMemory(memory, page, owner, "wrapped page");
    require(buffer.contents() == memory && buffer.allocatedBytes() == page &&
                backend.memoryStats().allocatedBytes == before + page,
            "a wrapped page is not the memory it wraps or is not counted");
    *static_cast<uint32_t *>(memory) = 5;
    const uint32_t count = 1, increment = 7;
    (void)backend.submit({"test_add_u32", {{0, backend.view(buffer, 0, sizeof(uint32_t))}},
                          {{1, &count, sizeof(count)}, {2, &increment, sizeof(increment)}},
                          {1, 1, 1}, {1, 1, 1}});
    require(*static_cast<uint32_t *>(memory) == 12, "a kernel did not write the wrapped page");
    rejects([&] { backend.releaseMemory(buffer); }, "owner's to release",
            "wrapped memory was released");
    rejects([&] {
                (void)backend.wrapSharedMemory(static_cast<uint8_t *>(memory) + 64, page,
                                               owner, "");
            },
            "whole pages", "misaligned memory was wrapped");
    std::cout << "PASS wrapped memory\n";
}

// notify() calls back once, when the event reaches its value, and promptly
// for a value already reached, never on the thread that asked; a signal below
// the event's value leaves it as it is.
void sharedEventNotifies(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    const SharedEvent event = backend.newSharedEvent();
    id<MTLSharedEvent> native = (__bridge id<MTLSharedEvent>)event.nativeHandle();
    const std::thread::id caller = std::this_thread::get_id();
    std::atomic<unsigned> calls{0};
    std::atomic<bool> onCaller{false};
    event.notify(3, [&] {
        if (std::this_thread::get_id() == caller) onCaller = true;
        ++calls;
    });
    event.signal(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(calls == 0, "notify called back before the event reached its value");
    event.signal(3);
    require(waitFor([&] { return calls != 0; }, std::chrono::seconds(5)),
            "notify did not call back when the event reached its value");
    event.signal(4);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(calls == 1 && !onCaller,
            "notify called back more than once or on the thread that asked");
    event.signal(1);
    require(native.signaledValue == 4, "a signal below the event's value lowered it");
    std::promise<std::thread::id> reached;
    auto called = reached.get_future();
    event.notify(2, [&] { reached.set_value(std::this_thread::get_id()); });
    require(called.wait_for(std::chrono::seconds(1)) == std::future_status::ready,
            "notify of a value already reached did not call back promptly");
    require(called.get() != caller, "notify called back on the thread that asked");
    rejects([] { SharedEvent{}.notify(1, [] {}); }, "empty shared event",
            "an empty shared event accepted a notification");
    std::cout << "PASS shared event notifies\n";
}

// Event steps order a command against another agent, the CPU here: the work
// before a signal completes before the event rises, and the work after a wait
// runs only once the agent raises the event to its value.
void eventSteps(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const SharedEvent event = backend.newSharedEvent();
    id<MTLSharedEvent> native = (__bridge id<MTLSharedEvent>)event.nativeHandle();
    const uint32_t count = 1, one = 1, ten = 10;
    const std::vector<ComputeDispatch> dispatches{addition(buffer, count, one),
                                                  addition(buffer, count, ten)};
    const std::vector<EventStep> events{{1, event, 1, EventStep::Kind::Signal},
                                        {1, event, 2, EventStep::Kind::Wait}};
    CommandTicket ticket = backend.submitCommandAsync(Command{dispatches, events});
    require(waitFor([&] { return native.signaledValue >= 1; }, std::chrono::seconds(10)) &&
                native.signaledValue == 1 && *value == 1,
            "the event rose before the work ahead of its signal, or never");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(*value == 1, "the work after a wait ran before the event reached it");
    event.signal(2);
    (void)ticket.wait();
    require(*value == 11, "the work after a wait did not run once the event reached it");
    std::cout << "PASS event steps\n";
}

// Submission refuses event steps it cannot encode before it submits anything:
// a step without an event or a value, one not a host step that would keep the
// GPU awake, out of dispatch order or past the dispatches, and a command of
// more signals (one command buffer each) or host steps (one each, two when
// awake) than its queue holds command buffers for besides its last. The backend stays healthy, and a command of the most
// signals it takes runs.
void malformedEventStepsRefused(const std::string &metallibPath) {
    using enum EventStep::Kind;
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    const SharedEvent event = backend.newSharedEvent();
    const auto submit = [&](const std::vector<EventStep> &events) {
        return backend.submitCommandAsync(Command{dispatches, events}).wait();
    };
    rejects([&] { (void)submit({{1, SharedEvent{}, 1, Signal}}); }, "needs an event",
            "an event step without an event was submitted");
    rejects([&] { (void)submit({{1, event, 0, Wait}}); }, "nonzero value",
            "an event step of value 0 was submitted");
    rejects([&] { (void)submit({{1, event, 1, Signal, {}, {}, HostStepGpu::Awake}}); },
            "only a host step keeps the GPU awake", "a signal that keeps the GPU awake was submitted");
    rejects([&] { (void)submit({{2, event, 1, Signal}, {1, event, 2, Wait}}); },
            "out of dispatch order", "event steps out of dispatch order were submitted");
    rejects([&] { (void)submit({{3, event, 1, Signal}}); },
            "more dispatches than its command has",
            "an event step past the command's dispatches was submitted");
    std::vector<EventStep> signals;
    for (uint64_t signal = 1; signal <= 512; ++signal)
        signals.push_back({2, event, signal, Signal});
    rejects([&] { (void)submit(signals); }, "more command buffers than its queue holds",
            "a command of 512 signals was submitted");
    // A host step takes a command buffer, and one more when it keeps the GPU
    // awake.
    const std::vector<EventStep> hostSteps(512, EventStep{2, {}, 0, Host, [] {}});
    rejects([&] { (void)submit(hostSteps); }, "more command buffers than its queue holds",
            "a command of 512 host steps was submitted");
    const std::vector<EventStep> awakeSteps(256, EventStep{2, {}, 0, Host, [] {}, {}, HostStepGpu::Awake});
    rejects([&] { (void)submit(awakeSteps); }, "more command buffers than its queue holds",
            "a command of 256 awake host steps was submitted");
    require(backend.healthy() && BackendInstrumentation::submittedCommands(backend) == 0,
            "a refused event step marked the backend unhealthy or submitted work");
    // 511 signals take 512 command buffers, as many as the queue holds.
    signals.pop_back();
    auto submitted = std::async(std::launch::async, [&] { return submit(signals); });
    require(submitted.wait_for(std::chrono::seconds(30)) == std::future_status::ready,
            "a command of as many command buffers as its queue holds did not finish");
    (void)submitted.get();
    id<MTLSharedEvent> native = (__bridge id<MTLSharedEvent>)event.nativeHandle();
    require(*value == 2 && native.signaledValue == 511,
            "a command of 511 signals did not run");
    std::cout << "PASS malformed event steps are refused\n";
}

// A command of 100 signal and wait pairs, more command buffers than a queue
// holds by default, runs in order against an agent that answers each signal:
// a signal follows the work before it, and the work after its wait waits for
// the answer.
void eventPairsRunInOrder(const std::string &metallibPath) {
    constexpr uint64_t kPairs = 100;
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(kPairs + 1, addition(buffer, count, one));
    const SharedEvent event = backend.newSharedEvent();
    std::vector<EventStep> events;
    for (uint64_t pair = 0; pair < kPairs; ++pair) {
        events.push_back({pair + 1, event, 2 * pair + 1, EventStep::Kind::Signal});
        events.push_back({pair + 1, event, 2 * pair + 2, EventStep::Kind::Wait});
    }
    // Answers signal 2p + 1, which follows p + 1 additions, with 2p + 2.
    std::atomic<uint64_t> answered{0}, disordered{0};
    std::thread agent([&] {
        for (uint64_t pair = 0; pair < kPairs; ++pair) {
            auto reached = std::make_shared<std::promise<void>>();
            event.notify(2 * pair + 1, [reached] { reached->set_value(); });
            if (reached->get_future().wait_for(std::chrono::seconds(10)) !=
                std::future_status::ready)
                return;
            if (*value != pair + 1) ++disordered;
            event.signal(2 * pair + 2);
            ++answered;
        }
    });
    auto submitted = std::async(std::launch::async, [&] {
        return backend.submitCommandAsync(Command{dispatches, events}).wait();
    });
    const bool finished =
        submitted.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
    agent.join();
    require(finished, "a command of 101 command buffers did not finish");
    (void)submitted.get();
    require(answered == kPairs && disordered == 0 && *value == kPairs + 1,
            "event pairs ran out of order");
    std::cout << "PASS 100 event pairs run in order\n";
}

id<MTLSharedEvent> faultedEvent = nil;
IMP originalSignalEncoding = nullptr;
IMP originalCommandError = nullptr;

// The command buffer that signals faultedEvent to 1 fails instead: it reports
// a failure once it ends (terminalCommandStatus, failedCommandError) and never
// signals.
void failBeforeSignal(id command, SEL selector, id<MTLSharedEvent> event, uint64_t value) {
    if (event != faultedEvent || value != 1) {
        reinterpret_cast<void (*)(id, SEL, id<MTLSharedEvent>, uint64_t)>(
            originalSignalEncoding)(command, selector, event, value);
        return;
    }
    [command addCompletedHandler:^(id<MTLCommandBuffer> ended) {
        failedCommand.store((__bridge void *)ended);
    }];
}

NSError *failedCommandError(id command, SEL selector) {
    if ((__bridge void *)command == failedCommand.load())
        return [NSError
            errorWithDomain:MTLCommandBufferErrorDomain
                       code:MTLCommandBufferErrorPageFault
                   userInfo:@{NSLocalizedDescriptionKey : @"test fault before a signal"}];
    return reinterpret_cast<NSError *(*)(id, SEL)>(originalCommandError)(command, selector);
}

// A command buffer that fails before its signal, which the GPU then never
// delivers, still lets its command drain: the CPU delivers the signal, the
// agent waiting on it answers, and the command fails at once with that
// buffer's error. Without the signal, the buffer after it would wait until
// Metal fails it, after 5 seconds on macOS 27.0 ("Caused GPU Timeout
// Error"), and the command would fail that late; the watchdog here waits
// longer.
void failedBufferStillSignals(const std::string &metallibPath) {
    MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, 10.0);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    const SharedEvent event = backend.newSharedEvent();
    const std::vector<EventStep> events{{1, event, 1, EventStep::Kind::Signal},
                                        {1, event, 2, EventStep::Kind::Wait}};
    event.notify(1, [event] { event.signal(2); });
    faultedEvent = (__bridge id<MTLSharedEvent>)event.nativeHandle();
    id<MTLCommandBuffer> command =
        [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    {
        MethodReplacement signal(command, @selector(encodeSignalEvent:value:),
                                 reinterpret_cast<IMP>(failBeforeSignal));
        originalSignalEncoding = signal.original;
        MethodReplacement status(command, @selector(status),
                                 reinterpret_cast<IMP>(terminalCommandStatus));
        originalCommandStatus = status.original;
        MethodReplacement failure(command, @selector(error),
                                  reinterpret_cast<IMP>(failedCommandError));
        originalCommandError = failure.original;
        try {
            (void)backend.submitCommandAsync(Command{dispatches, events}).wait();
        } catch (const MetalBackendError &caught) {
            error = caught.what();
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    failedCommand.store(nullptr);
    faultedEvent = nil;
    require(error.find("Metal command 1 failed: test fault before a signal") !=
                std::string::npos,
            "a command did not fail with its failed buffer's error: " + error);
    const double seconds = std::chrono::duration<double>(elapsed).count();
    require(seconds < 2.0 && *value == 2 && !backend.healthy(),
            "a command did not drain promptly past its failed buffer: " +
                std::to_string(seconds) + " seconds, value " + std::to_string(*value));
    std::cout << "PASS a failed command buffer still delivers its signal\n";
}

// The backend's check that a GPU spin sees the CPU end it, run again while
// it decides nothing (the GPU busy with other work), for up to ten checks;
// nothing then, with a SKIP line naming what `test` could not check.
std::optional<bool> spinCheck(MetalBackend &backend, std::string_view test) {
    for (int check = 0; check < 10; ++check)
        if (const std::optional<bool> spins = BackendInstrumentation::awakeHostStepsSpin(backend)) return spins;
    std::cout << "SKIP " << test << ": the GPU stayed too busy for the spin check to decide\n";
    return std::nullopt;
}

// How long the GPU work after a host step may take to start: the spin's bound
// (60 ms or more) is far above both, and a GPU idle through the step only wakes
// up, at times slowly under the validation layer or other work.
constexpr double kSpinningResumeSeconds = 0.01;
constexpr double kIdleResumeSeconds = 0.05;

// Host steps run between a command's work in order: each sees what the GPU
// wrote before it, and the GPU reads what it wrote after it; a command of
// fifty of them (more command buffers than a queue holds by default) runs
// them in order, keeping the GPU awake or not, and on the backend's runner,
// never on the submitting thread; a GPU spin through an awake step sees the
// CPU end it on this system (host_step.metal).
// Once the command completed, before its wait returns, each step's round trip
// reached its `timed`: its work as long as the step took, and pickup and
// resume within a second. A step without work is refused.
void hostSteps(const std::string &metallibPath) {
    constexpr uint32_t kSteps = 50;
    // How long the step that sleeps works.
    constexpr double kSleepSeconds = 0.002;
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(kSteps + 1, addition(buffer, count, one));
    const std::thread::id submitter = std::this_thread::get_id();
    const std::optional<bool> spins = spinCheck(backend, "the awake host steps' resume");
    require(spins.value_or(true), "a GPU spin through an awake host step did not see the CPU end it");
    for (const bool awake : {false, true}) {
        const double resumeBound = awake && spins ? kSpinningResumeSeconds : kIdleResumeSeconds;
        const std::string kind = awake ? "awake" : "plain";
        *value = 0;
        std::vector<uint32_t> seen;
        std::atomic<bool> onSubmitter{false};
        std::vector<EventStep> steps;
        std::mutex timedMutex;
        std::vector<std::pair<uint32_t, HostStepTiming>> timings;
        for (uint32_t step = 0; step < kSteps; ++step)
            steps.push_back({step + 1, {}, 0, EventStep::Kind::Host,
                             [&, step] {
                                 if (std::this_thread::get_id() == submitter) onSubmitter = true;
                                 seen.push_back(uint32_t{*value});
                                 if (step == 1)
                                   std::this_thread::sleep_for(std::chrono::duration<double>(kSleepSeconds));
                                 // The GPU adds one to what the step left.
                                 if (step == kSteps - 1) *value += 1000;
                             },
                             [&, step](const HostStepTiming &timing) {
                                 std::lock_guard lock(timedMutex);
                                 timings.push_back({step, timing});
                             },
                             awake ? HostStepGpu::Awake : HostStepGpu::Idle});
        const Command command{dispatches, steps};
        const auto started = std::chrono::steady_clock::now();
        const CommandTiming commandTiming = backend.submitCommandAsync(command).wait();
        const double microseconds =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        std::cout << "  " << kSteps << " host steps of a " << kind << " command: " << microseconds / kSteps
                  << " us each\n";
        bool ordered = seen.size() == kSteps;
        for (uint32_t step = 0; ordered && step < kSteps; ++step) ordered = seen[step] == step + 1;
        require(ordered && !onSubmitter && *value == kSteps + 1 + 1000,
                "host steps of a " + kind +
                    " command ran out of order, on the submitting thread, or unseen by the GPU");
        std::lock_guard lock(timedMutex);
        std::vector<bool> reported(kSteps);
        double pickup = 0.0, resume = 0.0;
        for (const auto &[step, timing] : timings) {
            require(step < kSteps && !reported[step], "a host step's round trip was reported twice");
            reported[step] = true;
            require(timing.pickupSeconds >= 0.0 && timing.pickupSeconds < 1.0 && timing.resumeSeconds >= 0.0 &&
                        timing.resumeSeconds < 1.0 && timing.workSeconds >= (step == 1 ? kSleepSeconds : 0.0) &&
                        timing.workSeconds < 1.0,
                    "a host step's round trip is implausible: pickup " + std::to_string(timing.pickupSeconds) +
                        " s, work " + std::to_string(timing.workSeconds) + " s, resume " +
                        std::to_string(timing.resumeSeconds) + " s");
            // An awake step's GPU spin (host_step_wait) ends at the runner's
            // release, not at its bound of 60 ms or more; without it the GPU
            // only wakes up.
            require(timing.resumeSeconds < resumeBound, "a host step's GPU work after it waited for the spin's bound: " +
                                                            std::to_string(timing.resumeSeconds) + " s");
            pickup += timing.pickupSeconds;
            resume += timing.resumeSeconds;
        }
        require(timings.size() == kSteps, "a host step's round trip was not reported before the command's wait");
        // The command's wait on its steps spans the sleeping step's work and
        // lies within its GPU time.
        require(commandTiming.hostSeconds >= kSleepSeconds && commandTiming.hostSeconds <= commandTiming.gpuSeconds,
                "a command's wait on its host steps is implausible: " + std::to_string(commandTiming.hostSeconds) +
                    " s of " + std::to_string(commandTiming.gpuSeconds) + " s");
        std::cout << "  pickup " << pickup / kSteps * 1e6 << " us, resume " << resume / kSteps * 1e6
                  << " us per step, " << commandTiming.hostSeconds * 1e3 << " ms waited on the steps\n";
    }
    rejects([&] {
        const EventStep empty{1, {}, 0, EventStep::Kind::Host, {}};
        (void)backend.submitCommandAsync(Command{dispatches, {&empty, 1}});
    }, "host step needs work", "a host step without work was submitted");
    // Dispatch profiling runs the host steps between the dispatches it
    // replays, and reports no round trip.
    *value = 0;
    std::vector<uint32_t> profiled;
    bool profiledTimed = false;
    const std::vector<EventStep> one_step{{1, {}, 0, EventStep::Kind::Host,
                                           [&] { profiled.push_back(uint32_t{*value}); },
                                           [&](const HostStepTiming &) { profiledTimed = true; }}};
    BackendInstrumentation::setDispatchProfiling(backend, true);
    (void)backend.submitCommandAsync(Command{std::span(dispatches).first(2), one_step}).wait();
    (void)BackendInstrumentation::takeDispatchProfile(backend);
    BackendInstrumentation::setDispatchProfiling(backend, false);
    require(profiled == std::vector<uint32_t>{1} && *value == 2 && !profiledTimed,
            "dispatch profiling did not run a host step between its dispatches, or timed it");
    require(backend.healthy(), "host steps marked the backend unhealthy");
    std::cout << "PASS host steps run in order between a command's work\n";
}

// The first command of a fresh backend may hold an awake host step: the
// backend builds and checks the spin before it encodes the command, which
// runs the step between its work and spins through it on this system.
void firstCommandWithAnAwakeHostStep(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    uint32_t seen = 0;
    HostStepTiming timing;
    const std::vector<EventStep> steps{{1, {}, 0, EventStep::Kind::Host, [&] { seen = *value; },
                                        [&](const HostStepTiming &step) { timing = step; }, HostStepGpu::Awake}};
    (void)backend.submitCommandAsync(Command{dispatches, steps}).wait();
    require(seen == 1 && *value == 2 && backend.healthy(),
            "an awake host step in a fresh backend's first command did not run between its work");
    // A first check that decided nothing ran the step without the spin.
    const std::optional<bool> spins = spinCheck(backend, "a fresh backend's first awake host step");
    require(spins.value_or(true) && timing.resumeSeconds < kIdleResumeSeconds,
            "the GPU did not spin through an awake host step of a fresh backend's first command: resume " +
                std::to_string(timing.resumeSeconds) + " s");
    std::cout << "PASS a fresh backend's first command spins through its awake host step\n";
}

id noCommandQueue(id, SEL) { return nil; }
CFTimeInterval startedLate(id, SEL) { return std::numeric_limits<CFTimeInterval>::max(); }

// Awake host steps run as plain ones, with one warning, when the backend
// cannot build or check their spin or a spin ran past its release: without
// host-step.metallib beside the backend's library, with one Metal cannot
// load, with one whose host_step_wait never sees its release (the test
// library's, metal_backend_test.metal), and when Metal creates no queue for
// the check. A check whose every spin started after its release, as when the
// GPU is busy with other work, decides nothing and warns of nothing: a later
// check spins.
void hostStepSpinFallsBack(const std::string &metallibPath) {
    namespace fs = std::filesystem;
    constexpr std::string_view kWarning = "Metal host steps run without keeping the GPU awake";
    const auto runAwakeSteps = [](MetalBackend &backend) {
        MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
        auto *value = static_cast<volatile uint32_t *>(buffer.contents());
        const uint32_t count = 1, one = 1;
        const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
        for (int command = 0; command < 2; ++command) {
            *value = 0;
            uint32_t seen = 0;
            const std::vector<EventStep> steps{{1, {}, 0, EventStep::Kind::Host, [&] { seen = *value; }, {}, HostStepGpu::Awake}};
            (void)backend.submitCommandAsync(Command{dispatches, steps}).wait();
            require(seen == 1 && *value == 2, "an awake host step without its spin did not run between its work");
        }
    };
    const auto fallsBack = [&](MetalBackend &backend, std::string_view cause) {
        const std::string warnings = capturedStderr([&] { runAwakeSteps(backend); });
        const size_t warning = warnings.find(kWarning);
        require(warning != std::string::npos && warnings.find(cause) != std::string::npos &&
                    warnings.find(kWarning, warning + 1) == std::string::npos,
                "awake host steps without their spin did not warn once of \"" + std::string(cause) + "\": " +
                    warnings);
        require(backend.healthy() && BackendInstrumentation::awakeHostStepsSpin(backend) == false,
                "awake host steps without their spin marked the backend unhealthy or spun");
    };
    TemporaryDirectory directory("splash-host-step");
    const fs::path library = directory.path() / "test.metallib";
    const fs::path spinLibrary = directory.path() / "host-step.metallib";
    fs::copy_file(metallibPath, library);
    {
        MetalBackend backend(library.string());
        fallsBack(backend, spinLibrary.string());
    }
    writeFile(spinLibrary, "not a metallib");
    {
        MetalBackend backend(library.string());
        fallsBack(backend, "unable to load metallib");
    }
    fs::copy_file(library, spinLibrary, fs::copy_options::overwrite_existing);
    {
        MetalBackend backend(library.string());
        fallsBack(backend, "did not see the CPU's release");
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    {
        MetalBackend backend(metallibPath);
        MethodReplacement noQueue(device, @selector(newCommandQueue), reinterpret_cast<IMP>(noCommandQueue));
        fallsBack(backend, "unable to create Metal command queue");
    }
    MetalBackend backend(metallibPath);
    std::string warnings;
    {
        id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
        MethodReplacement late(command, @selector(GPUStartTime), reinterpret_cast<IMP>(startedLate));
        warnings = capturedStderr([&] {
            require(!BackendInstrumentation::awakeHostStepsSpin(backend).has_value(),
                    "a check whose spins all started late decided");
        });
    }
    runAwakeSteps(backend);
    require(warnings.find(kWarning) == std::string::npos &&
                spinCheck(backend, "a check after one that decided nothing").value_or(true),
            "a check whose spins all started late warned, or kept a later check from spinning: " + warnings);
    std::cout << "PASS awake host steps run as plain ones without a spin that sees its release\n";
}

// The buffers of a command before its last are committed as their encoding
// ends: a host step after the first dispatch runs while the backend still
// encodes the many dispatches after it, before submission returns.
void earlyBuffersRunWhileTheRestEncodes(const std::string &metallibPath) {
    constexpr uint32_t kDispatchesAfter = 20000;
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(kDispatchesAfter + 1, addition(buffer, count, one));
    std::atomic<bool> submitted{false}, ranBeforeSubmitted{false};
    const std::vector<EventStep> steps{{1, {}, 0, EventStep::Kind::Host, [&] { ranBeforeSubmitted = !submitted; }}};
    CommandTicket ticket = backend.submitCommandAsync(Command{dispatches, steps});
    submitted = true;
    (void)ticket.wait();
    require(*value == kDispatchesAfter + 1 && ranBeforeSubmitted,
            "a host step after a command's first buffer did not run before its submission returned");
    std::cout << "PASS a command's early buffers run while the rest encodes\n";
}

std::atomic<unsigned> encodersLeft{0};
IMP originalEncoderAfterCommit = nullptr;
// Hands out encoders until encodersLeft runs out, then none.
id encoderUntilExhausted(id command, SEL selector) {
    if (encodersLeft.fetch_sub(1) == 0) {
        encodersLeft = 0;
        return nil;
    }
    return reinterpret_cast<id (*)(id, SEL)>(originalEncoderAfterCommit)(command, selector);
}

// A command whose encoding fails after its first buffer was committed (Metal
// creates no encoder for the work after a host step) throws, releases the
// submission and marks the backend unhealthy with the reason; the committed
// buffer still runs, and so does the host step after it.
void encodingFailureAfterACommit(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    std::atomic<bool> ran{false};
    const std::vector<EventStep> steps{{1, {}, 0, EventStep::Kind::Host, [&] { ran = true; }}};
    id<MTLCommandBuffer> command = [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
    {
        encodersLeft = 1;
        MethodReplacement encoders(command, @selector(computeCommandEncoder),
                                   reinterpret_cast<IMP>(encoderUntilExhausted));
        originalEncoderAfterCommit = encoders.original;
        rejects([&] { (void)backend.submitCommandAsync(Command{dispatches, steps}); },
                "unable to create Metal compute encoder", "a command Metal gave no encoder was submitted");
    }
    require(!backend.healthy() &&
                backend.unhealthyReason().find("Metal command encoding failed: unable to create Metal compute encoder") !=
                    std::string::npos &&
                !backend.commandInFlight(),
            "a failed encoding left the backend healthy or its submission held: " + backend.unhealthyReason());
    require(waitFor([&] { return ran.load(); }, std::chrono::seconds(5)) && waitFor([&] { return *value == 1; },
                                                                                       std::chrono::seconds(5)),
            "the buffer committed before a failed encoding or the host step after it did not run");
    std::cout << "PASS an encoding failure after a commit marks the backend unhealthy\n";
}

// A host step that throws fails its command with its message once the
// command ends; the work after it still runs, the backend is unhealthy, and
// the failed command reports no round trip. A `timed` that throws leaves its
// command's result alone and marks the backend unhealthy.
void throwingHostStepFailsItsCommand(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    bool timed = false;
    const std::vector<EventStep> steps{{1, {}, 0, EventStep::Kind::Host,
                                        [] { throw std::runtime_error("test host failure"); },
                                        [&](const HostStepTiming &) { timed = true; }}};
    std::string error;
    try {
        (void)backend.submitCommandAsync(Command{dispatches, steps}).wait();
    } catch (const MetalBackendError &caught) {
        error = caught.what();
    }
    require(error.find("Metal command 1 failed: host step failed: test host failure") != std::string::npos,
            "a command did not fail with its host step's error: " + error);
    require(*value == 2 && !backend.healthy(),
            "a failed host step stopped the work after it or left the backend healthy");
    require(!timed, "a failed command reported a host step's round trip");
    MetalBackend timedBackend(metallibPath);
    MetalBuffer timedBuffer = sharedBuffer(timedBackend, sizeof(uint32_t));
    const std::vector<ComputeDispatch> timedDispatches(2, addition(timedBuffer, count, one));
    const std::vector<EventStep> throwingTimed{{1, {}, 0, EventStep::Kind::Host, [] {},
                                                [](const HostStepTiming &) { throw std::runtime_error("timed"); }}};
    (void)timedBackend.submitCommandAsync(Command{timedDispatches, throwingTimed}).wait();
    require(!timedBackend.healthy() && timedBackend.unhealthyReason().find("timing callback") != std::string::npos,
            "a throwing timing callback left the backend healthy");
    std::cout << "PASS a throwing host step fails its command\n";
}

std::atomic<bool> failFirstHostStep{false};

// Once armed, the command buffer that raises a command's first host step's
// event (its value 1) fails instead, as failBeforeSignal's does.
void failBeforeFirstHostStep(id command, SEL selector, id<MTLSharedEvent> event, uint64_t value) {
    if (value != 1 || !failFirstHostStep.exchange(false)) {
        reinterpret_cast<void (*)(id, SEL, id<MTLSharedEvent>, uint64_t)>(
            originalSignalEncoding)(command, selector, event, value);
        return;
    }
    [command addCompletedHandler:^(id<MTLCommandBuffer> ended) {
        failedCommand.store((__bridge void *)ended);
    }];
}

// A command runs none of its host steps after one that threw or did not run
// (the buffer before it failed): a later step may build on an earlier one, as
// an expert load's later rounds build on its first. The command fails with
// the first failure, which names the steps that ran and the longest's work,
// and the work after the steps still runs.
void failedCommandRunsNoLaterHostStep(const std::string &metallibPath) {
    for (const bool fault : {false, true}) {
        MetalBackend backend(metallibPath);
        MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
        auto *value = static_cast<volatile uint32_t *>(buffer.contents());
        *value = 0;
        const uint32_t count = 1, one = 1;
        const std::vector<ComputeDispatch> dispatches(3, addition(buffer, count, one));
        bool first = false, second = false;
        const std::vector<EventStep> steps{
            {1, {}, 0, EventStep::Kind::Host,
             [&] {
                 first = true;
                 std::this_thread::sleep_for(std::chrono::milliseconds(50));
                 throw std::runtime_error("test host failure");
             }},
            {2, {}, 0, EventStep::Kind::Host, [&] { second = true; }}};
        id<MTLCommandBuffer> command = [[MTLCreateSystemDefaultDevice() newCommandQueue] commandBuffer];
        std::string error;
        {
            MethodReplacement signal(command, @selector(encodeSignalEvent:value:),
                                     reinterpret_cast<IMP>(failBeforeFirstHostStep));
            originalSignalEncoding = signal.original;
            MethodReplacement status(command, @selector(status), reinterpret_cast<IMP>(terminalCommandStatus));
            originalCommandStatus = status.original;
            MethodReplacement failure(command, @selector(error), reinterpret_cast<IMP>(failedCommandError));
            originalCommandError = failure.original;
            failFirstHostStep = fault;
            try {
                (void)backend.submitCommandAsync(Command{dispatches, steps}).wait();
            } catch (const MetalBackendError &caught) {
                error = caught.what();
            }
        }
        failFirstHostStep = false;
        failedCommand.store(nullptr);
        const std::string expected = fault ? "Metal command 1 failed: test fault before a signal"
                                           : "Metal command 1 failed: host step failed: test host failure";
        // The step that ran is named with its work; none ran past the fault.
        const std::string named = " (host steps: 1 of 2 ran; the longest worked ";
        const size_t at = error.find(named);
        const bool accounted =
            fault ? error.find("host steps:") == std::string::npos
                  : at != std::string::npos && std::stod(error.substr(at + named.size())) >= 0.05 &&
                        error.find(" s, step 1", at) != std::string::npos;
        require(error.find(expected) != std::string::npos && accounted && first != fault && !second && *value == 3,
                std::string(fault ? "a command whose buffer failed before its first host step"
                                  : "a command whose first host step threw") +
                    " ran its second step (" + std::to_string(second) + "), or failed with \"" + error +
                    "\", or its work stopped at " + std::to_string(*value));
    }
    std::cout << "PASS a failed command runs no later host step\n";
}

// A host step's `then` runs after its work, once the work after the step
// may go: it sees that work's result; it does not run after work that threw,
// and one that throws marks the backend unhealthy (the command may have
// completed by then). Dispatch profiling runs it right after its work.
void hostStepThen(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    uint32_t seen = 0;
    std::atomic<bool> after{false};
    const std::vector<EventStep> steps{{1, {}, 0, EventStep::Kind::Host, [&] { seen = *value; }, {}, HostStepGpu::Idle, [&] {
                                            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                                            while (*value != 2 && std::chrono::steady_clock::now() < end) {}
                                            after = *value == 2;
                                        }}};
    (void)backend.submitCommandAsync(Command{dispatches, steps}).wait();
    require(waitFor([&] { return after.load(); }, std::chrono::seconds(5)) && seen == 1,
            "a host step's then did not run once the work after the step went");
    *value = 0;
    std::vector<uint32_t> order;
    const std::vector<EventStep> profiled{{1, {}, 0, EventStep::Kind::Host, [&] { order.push_back(uint32_t{*value}); }, {}, HostStepGpu::Idle,
                                           [&] { order.push_back(uint32_t{*value} + 10); }}};
    BackendInstrumentation::setDispatchProfiling(backend, true);
    (void)backend.submitCommandAsync(Command{dispatches, profiled}).wait();
    (void)BackendInstrumentation::takeDispatchProfile(backend);
    BackendInstrumentation::setDispatchProfiling(backend, false);
    require(order == std::vector<uint32_t>{1, 11}, "dispatch profiling did not run a host step's then after its work");
    std::atomic<bool> ran{false};
    const std::vector<EventStep> throwing{{1, {}, 0, EventStep::Kind::Host,
                                           [] { throw std::runtime_error("test host failure"); }, {}, HostStepGpu::Idle,
                                           [&] { ran = true; }}};
    MetalBackend failing(metallibPath);
    MetalBuffer failingBuffer = sharedBuffer(failing, sizeof(uint32_t));
    const std::vector<ComputeDispatch> failingDispatches(2, addition(failingBuffer, count, one));
    rejects([&] { (void)failing.submitCommandAsync(Command{failingDispatches, throwing}).wait(); },
            "test host failure", "a command whose host step threw did not fail");
    require(!ran, "a host step's then ran after its work threw");
    MetalBackend thenFails(metallibPath);
    MetalBuffer thenBuffer = sharedBuffer(thenFails, sizeof(uint32_t));
    const std::vector<ComputeDispatch> thenDispatches(2, addition(thenBuffer, count, one));
    const std::vector<EventStep> throwingThen{{1, {}, 0, EventStep::Kind::Host, [] {}, {}, HostStepGpu::Idle,
                                               [] { throw std::runtime_error("test then failure"); }}};
    (void)thenFails.submitCommandAsync(Command{thenDispatches, throwingThen}).wait();
    require(waitFor([&] { return !thenFails.healthy(); }, std::chrono::seconds(5)) &&
                thenFails.unhealthyReason().find("then threw") != std::string::npos,
            "a host step's then that threw left the backend healthy");
    std::cout << "PASS a host step's then runs once the work after it went\n";
}

// A host step whose work before it ends long after the runner began to wait
// for it (the command held 50 ms on an event, past the runner's spin, so the
// runner blocks on the step's event) still runs once that work ended, taken
// up within 25 ms of it: the runner looks at the event at least every 10 ms.
void hostStepAfterALongWait(const std::string &metallibPath) {
    MetalBackend backend(metallibPath);
    MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(3, addition(buffer, count, one));
    const SharedEvent hold = backend.newSharedEvent();
    uint32_t seen = 0;
    HostStepTiming timing;
    bool timed = false;
    // The step's buffer ends with work after the wait: the GPU end time of a
    // buffer is that of its last work.
    const std::vector<EventStep> steps{{1, hold, 1, EventStep::Kind::Wait},
                                       {2, {}, 0, EventStep::Kind::Host, [&] { seen = *value; },
                                        [&](const HostStepTiming &step) {
                                            timing = step;
                                            timed = true;
                                        }}};
    CommandTicket ticket = backend.submitCommandAsync(Command{dispatches, steps});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    hold.signal(1);
    (void)ticket.wait();
    require(seen == 2 && *value == 3, "a host step after a long wait did not run between the command's work");
    require(timed && timing.pickupSeconds < 0.025,
            "a host step after a long wait was not taken up promptly: " + std::to_string(timing.pickupSeconds) + " s");
    require(backend.healthy(), "a host step after a long wait marked the backend unhealthy");
    std::cout << "PASS a host step after a long wait runs (pickup " << timing.pickupSeconds * 1e6 << " us)\n";
}

// The watchdog bounds each stretch of a command between its progress, its
// host steps as the work before each ends and as each returns, not the whole
// command: checked as often as the engine's health checks may, a command of
// eight host steps of 50 ms each outlasts the 100 ms timeout and completes
// with the backend healthy, and one host step of 300 ms is given up on.
void watchdogFollowsHostSteps(const std::string &metallibPath) {
    // The reason the backend gave up a command of `steps` host steps of
    // `each`, empty while it stays healthy.
    const auto givenUp = [&](uint32_t steps, std::chrono::milliseconds each) {
        MetalBackend backend(metallibPath, kResidencyKeepAliveSeconds, kShortCommandTimeoutSeconds);
        MetalBuffer buffer = sharedBuffer(backend, sizeof(uint32_t));
        auto *value = static_cast<volatile uint32_t *>(buffer.contents());
        *value = 0;
        const uint32_t count = 1, one = 1;
        const std::vector<ComputeDispatch> dispatches(steps + 1, addition(buffer, count, one));
        std::vector<EventStep> events;
        for (uint32_t step = 1; step <= steps; ++step)
            events.push_back({step, {}, 0, EventStep::Kind::Host, [each] { std::this_thread::sleep_for(each); }});
        std::atomic<bool> done{false};
        std::thread checks([&] {
            while (!done) {
                try {
                    backend.checkHealth();
                } catch (const MetalBackendError &) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
        try {
            (void)backend.submitCommandAsync(Command{dispatches, events}).wait();
        } catch (const MetalBackendError &) {
        }
        done = true;
        checks.join();
        require(!backend.healthy() || *value == steps + 1, "a command completed without its work");
        return backend.healthy() ? std::string() : backend.unhealthyReason();
    };
    const std::string progressing = givenUp(8, std::chrono::milliseconds(50));
    require(progressing.empty(),
            "a command whose host steps each ended within the timeout was given up: " + progressing);
    const std::string stalled = givenUp(1, std::chrono::milliseconds(300));
    require(stalled.find("completion timed out") != std::string::npos,
            "a host step longer than the timeout was not given up: " + stalled);
    std::cout << "PASS the watchdog follows a command's host steps\n";
}

// A backend destroyed while its runner is blocked on a step (the command
// held on an event past the runner's spin) stops the runner within its stop
// check: the runner raises the step without running it, so the command
// drains once the event lets its work go, and fails, as its step never ran.
void blockedHostStepRunnerStops(const std::string &metallibPath) {
    auto backend = std::make_unique<MetalBackend>(metallibPath);
    MetalBuffer buffer = sharedBuffer(*backend, sizeof(uint32_t));
    *static_cast<volatile uint32_t *>(buffer.contents()) = 0;
    const uint32_t count = 1, one = 1;
    const std::vector<ComputeDispatch> dispatches(2, addition(buffer, count, one));
    const SharedEvent hold = backend->newSharedEvent();
    bool ran = false;
    const std::vector<EventStep> steps{{0, hold, 1, EventStep::Kind::Wait},
                                       {1, {}, 0, EventStep::Kind::Host, [&] { ran = true; }}};
    CommandTicket ticket = backend->submitCommandAsync(Command{dispatches, steps});
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const auto destroying = std::chrono::steady_clock::now();
    backend.reset();
    const double teardownSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - destroying).count();
    hold.signal(1);
    std::string error;
    try {
        (void)ticket.wait();
    } catch (const MetalBackendError &caught) {
        error = caught.what();
    }
    require(!ran && *static_cast<volatile uint32_t *>(buffer.contents()) == 2,
            "a stopped runner ran a step, or its command did not drain");
    require(error.find("host step not run: the backend stopped") != std::string::npos,
            "a command whose step a stopped runner skipped did not fail: " + error);
    require(teardownSeconds < 0.5,
            "destroying a backend whose runner blocked on a step took " + std::to_string(teardownSeconds) + " s");
    std::cout << "PASS a blocked host-step runner stops (" << teardownSeconds * 1e3 << " ms)\n";
}

void run(const std::string &metallibPath) {
    NSData *libraryData = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:metallibPath.c_str()]];
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
        (void)sharedBuffer(backend, 16384);
        fail("operation guard admitted an allocation");
    } catch (const MetalAllocationError &error) {
        require(error.failure() == AllocationFailure::HostPressure,
                "operation guard lost its failure classification");
    }
    require(backend.healthy() && backend.memoryStats().allocatedBytes == guardedBytes,
            "operation guard leaked memory or poisoned the backend");
    backend.setOperationGuard({});

    NSData *replacement = [@"replaced after library loading"
        dataUsingEncoding:NSUTF8StringEncoding];
    require([replacement writeToFile:temporaryPath
                            options:NSDataWritingAtomic error:nullptr],
            "could not replace temporary metallib");
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
    require(capabilities.maxThreadgroupWidth >= 1024,
            "threadgroup thread capability is insufficient");
    const auto probed = splash::metal::probeDeviceCapabilities();
    require(probed.deviceName == capabilities.deviceName &&
                probed.appleGpuFamily == capabilities.appleGpuFamily &&
                probed.macosVersion() == capabilities.macosVersion() &&
                !probed.validationMessage(),
            "the device check read the device differently from the backend");
    require(backend.healthy(), "new backend is unhealthy");
    require(BackendInstrumentation::submittedCommands(backend) == 0,
            "new backend has submissions");
    require(BackendInstrumentation::cachedPipelines(backend) == 0,
            "pipeline cache is not empty");

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
                        backend.healthy() &&
                        BackendInstrumentation::submittedCommands(backend) == 0,
                    "guarded submission lost its cause or altered the backend");
        }
        backend.setOperationGuard({});

        for (int runIndex = 0; runIndex < 2; ++runIndex) {
            splash::metal::CommandTiming timing;
            if (!runIndex) {
                timing = backend.submit(dispatch);
            } else {
                std::promise<void> completion;
                auto notified = completion.get_future();
                auto ticket = backend.submitCommandAsync(
                    {&dispatch, 1}, [&] { completion.set_value(); });
                rejects(
                    [&] { (void)backend.submit(dispatch); },
                    "Metal backend already has an in-flight command",
                    "a second in-flight command was accepted");
                timing = ticket.wait();
                require(ticket.ready(),
                        "completed async ticket is not ready");
                // Completion is published before the callback runs, so
                // wait() may return first; only the callback's own signal
                // shows the notification was delivered.
                require(notified.wait_for(std::chrono::seconds(5)) ==
                            std::future_status::ready,
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

    require(BackendInstrumentation::submittedCommands(backend) == 2,
            "successful submissions were not counted");
    require(BackendInstrumentation::cachedPipelines(backend) == 1,
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
        (void)backend.submitCommandAsync(command).wait();
    }
    require(BackendInstrumentation::submittedCommands(backend) == 3,
            "explicit operation list did not use one command buffer");
    for (uint32_t i = kViewElementCount; i < kElementCount; ++i) {
        require(values[i] == i + 4 * kIncrement,
                "multi-dispatch command produced an incorrect result");
    }

    rejects(
        [&] { (void)backend.view(base, base.sizeBytes(), 1); },
        "Metal buffer view is out of range", "out-of-range view was accepted");

    ComputeDispatch missingPipeline;
    missingPipeline.pipelineName = "does_not_exist";
    rejects(
        [&] { (void)backend.submit(missingPipeline); },
        "missing Metal function: does_not_exist", "missing pipeline was accepted");
    {
        // A binding takes one of the argument table's 31 entries of its own.
        ComputeDispatch rebound;
        rebound.pipelineName = "test_add_u32";
        rebound.buffers = {{0, view}};
        rebound.bytes = {{0, &kIncrement, sizeof(kIncrement)}};
        rejects(
            [&] { (void)backend.submit(rebound); },
            "duplicate compute binding index", "a binding index bound twice was accepted");
        rebound.bytes = {{31, &kIncrement, sizeof(kIncrement)}};
        rejects(
            [&] { (void)backend.submit(rebound); },
            "compute binding index exceeds the argument table",
            "a binding index past the argument table was accepted");
    }
    require(backend.healthy(),
            "a descriptor error incorrectly poisoned the backend");
    require(backend.unhealthyReason().empty(),
            "healthy backend has an unhealthy reason");
    require(BackendInstrumentation::submittedCommands(backend) == 3,
            "failed pre-commit dispatch was counted as submitted");
    require(BackendInstrumentation::cachedPipelines(backend) == 1,
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
    // Restarted, both peaks are what is allocated now.
    backend.restartPeaks();
    stats = backend.memoryStats();
    require(stats.peakAllocatedBytes == 0 &&
                stats.devicePeakAllocatedBytes == stats.deviceCurrentAllocatedBytes,
            "restarted peaks kept a released allocation");

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

    bindingRunsKeepOffsets(backend);
    indirectGridsRunTheWrittenGrid(backend);
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
            synchronousWaitObeysTheWatchdog(argv[1]);
            abandonedTicketReturnsAfterTheWatchdog(argv[1]);
            stopRefusesSubmission(argv[1]);
            shutdownInterruptsACommandWait(argv[1]);
            shutdownLeavesTicketTeardownToTheCommand(argv[1]);
            shutdownSparesACommandThatCompletes(argv[1]);
            dispatchProfilingCoversEveryCommand(argv[1]);
            preparedPipelinesCompileAhead(argv[1]);
            buffersStayResident(argv[1]);
            infiniteKeepAliveHoldsResidency(argv[1]);
            releasedMemory(argv[1]);
            wrappedMemory(argv[1]);
            sharedEventNotifies(argv[1]);
            eventSteps(argv[1]);
            eventPairsRunInOrder(argv[1]);
            malformedEventStepsRefused(argv[1]);
            failedBufferStillSignals(argv[1]);
            firstCommandWithAnAwakeHostStep(argv[1]);
            hostSteps(argv[1]);
            hostStepSpinFallsBack(argv[1]);
            earlyBuffersRunWhileTheRestEncodes(argv[1]);
            encodingFailureAfterACommit(argv[1]);
            throwingHostStepFailsItsCommand(argv[1]);
            failedCommandRunsNoLaterHostStep(argv[1]);
            hostStepThen(argv[1]);
            hostStepAfterALongWait(argv[1]);
            watchdogFollowsHostSteps(argv[1]);
            blockedHostStepRunnerStops(argv[1]);
            allocationDoesNotRequestResidency(argv[1]);
            residencyRacesTheHeartbeat(argv[1]);
            residencyEndsWithoutBlits(argv[1]);
            residencyReturnsRemovedBuffers(argv[1]);
            queueHoldsAddressedBuffers(argv[1]);
            buffersReachedThroughTables(argv[1]);
            run(argv[1]);
        } catch (const std::exception &error) {
            std::cerr << "FAIL: unexpected exception: " << error.what()
                      << '\n';
            return 1;
        }
    }
    return 0;
}
