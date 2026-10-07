#pragma once

#include "AwakeClock.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Status.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace splash::engine {

enum class RuntimeBootstrapStage {
    ResourceAssembly,
    ModelCreation,
    MaximumPrefill,
    DecodeWarmup,
    CompositeStateRestore,
    MemoryAudit,
    AnnounceReady,
    Ready,
};

[[nodiscard]] std::string_view runtimeBootstrapStageName(
    RuntimeBootstrapStage stage);

// A report of a successful bootstrap is at stage Ready.
struct RuntimeBootstrapReport {
    RuntimeBootstrapStage stage = RuntimeBootstrapStage::ResourceAssembly;
    // The step of resource assembly that failed; empty past resource assembly.
    std::optional<RuntimeResourceStage> resourceStage;
    RuntimeResourceFailure resourceFailure = RuntimeResourceFailure::Other;
    std::string message;
    WarmupReport warmup;
    MemoryAuditResult memoryAudit;
    std::string budgetDescription;

    [[nodiscard]] std::string describe() const;
};

class RuntimeBootstrapError final : public std::runtime_error {
public:
    explicit RuntimeBootstrapError(RuntimeBootstrapReport report);
    explicit RuntimeBootstrapError(const RuntimeResourcesError &error);

    [[nodiscard]] const RuntimeBootstrapReport &report() const noexcept {
        return report_;
    }

private:
    RuntimeBootstrapReport report_;
};

// Startup retries a temporary host or driver allocation failure for a
// bounded time. The window opens at the first such failure, not at process
// start, since a start loads the weights and warms up for a while before
// one; a failure at a later stage, or a later step of resource assembly,
// than the failure that opened the window follows progress and opens a new
// one.
class StartupRetryWindow final {
public:
    using Clock = AwakeClock;

    explicit StartupRetryWindow(Clock::duration length) noexcept
        : length_(length) {}

    // Until when startup may retry after this failure; nothing when the
    // failure is not temporary or its window has closed.
    [[nodiscard]] std::optional<Clock::time_point>
    retryUntil(const RuntimeBootstrapReport &failure, Clock::time_point now);

private:
    Clock::duration length_;
    std::optional<Clock::time_point> deadline_;
    // How far the failure that opened the window got.
    std::pair<RuntimeBootstrapStage, std::optional<RuntimeResourceStage>>
        reached_;
};

// Whether memory may not hold a request of contextTokens: the plan within
// what the host had available at startup, beyond its reserve and the warning
// margin, holds less. The estimate is conservative, since macOS compresses
// other applications further once the engine loads.
[[nodiscard]] bool memoryMayNotHold(const EngineMemoryPlan &plan,
                                    uint64_t hostAvailableBytes,
                                    uint32_t contextTokens);

// The wire limits of a model served with maxContext tokens: prompts and
// outputs up to the context, the engine's step and draft query rows, and a
// mask row per draft query and the anchor.
[[nodiscard]] protocol::ProtocolLimits
protocolLimitsFor(const model::ModelCapabilities &capabilities,
                  uint32_t maxContext) noexcept;

struct RuntimeBootstrapConfig {
    RuntimeResourcesConfig resources;
    // A zero engine maxContext is what the memory plan holds, as serve's
    // default; a larger one than that fails the bootstrap.
    NativeLoopConfig nativeLoop;
};

using ActualMemoryReporter = std::function<ActualMemoryReport()>;

// Complete owner returned only after the native loop has sent its
// ReadyEvent. No partially warmed instance escapes start().
class RuntimeBootstrap final {
public:
    [[nodiscard]] static std::unique_ptr<RuntimeBootstrap> start(
        RuntimeBootstrapConfig config,
        NativeRuntime::ByteSink output,
        NativeRuntime::StatusProvider statusProvider);

    // Completes warmup and validation before announcing readiness.
    [[nodiscard]] static RuntimeBootstrapReport requireWarmupAndAnnounce(
        const EngineMemoryPlan &memoryPlan,
        model::RuntimeModel &modelRuntime,
        ActualMemoryReporter memoryReporter,
        NativeRuntime &nativeLoop);

    RuntimeBootstrap(const RuntimeBootstrap &) = delete;
    RuntimeBootstrap &operator=(const RuntimeBootstrap &) = delete;
    ~RuntimeBootstrap();

    [[nodiscard]] RuntimeResources &resources() noexcept {
        return *resources_;
    }
    [[nodiscard]] model::RuntimeModel &modelRuntime() noexcept { return *model_; }
    [[nodiscard]] NativeRuntime &nativeLoop() noexcept {
        return *nativeLoop_;
    }

    // The memory control pass the transport runs at a command-free point
    // (MemoryControl::run).
    [[nodiscard]] bool controlPass(MemoryPressure pressure) {
        reportKvZipOverflow();
        return memoryControl_.run(pressure);
    }
    // The status document, with the metrics and the loop timing the
    // transport keeps.
    [[nodiscard]] std::string statusJson(const RuntimeMetricsSnapshot &metrics,
                                         const NativeLoopTiming &loop);

private:
    RuntimeBootstrap(std::unique_ptr<RuntimeResources> resources,
                     std::unique_ptr<model::RuntimeModel> modelRuntime,
                     std::unique_ptr<NativeRuntime> nativeLoop,
                     RuntimeBootstrapReport report);

    // Reverse destruction order is loop -> modelRuntime -> resources.
    std::unique_ptr<RuntimeResources> resources_;
    std::unique_ptr<model::RuntimeModel> model_;
    std::unique_ptr<NativeRuntime> nativeLoop_;
    MemoryControl memoryControl_;
    RuntimeBootstrapReport report_;
    // ZBF16 overflowing slabs already logged (reportKvZipOverflow).
    uint32_t kvZipOverflowLogged_ = 0;

    // Logs ZBF16 slabs that dropped escapes since the last pass.
    void reportKvZipOverflow();
};

}  // namespace splash::engine
