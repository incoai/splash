#pragma once

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

namespace splash::engine {

enum class RuntimeBootstrapStage {
    ResourceAssembly,
    ModelCreation,
    MaximumPrefill,
    DecodeWarmup,
    DraftVerifyCommit,
    CompositeStateRestore,
    MemoryAudit,
    AnnounceReady,
    Ready,
};

[[nodiscard]] std::string_view runtimeBootstrapStageName(
    RuntimeBootstrapStage stage);

struct RuntimeBootstrapReport {
    bool ready = false;
    RuntimeBootstrapStage stage = RuntimeBootstrapStage::ResourceAssembly;
    RuntimeResourceFailure resourceFailure = RuntimeResourceFailure::Other;
    std::string message;
    WarmupReport warmup;
    MemoryAuditResult memoryAudit;
    // Always present once an immutable plan exists. On planning failure this
    // is the full BudgetValidationStatus JSON instead.
    std::string memoryPlanJson;
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
// start, since a cold start can prepare weights for minutes before one; a
// failure at a later stage than the last one follows progress and opens a
// new window.
class StartupRetryWindow final {
public:
    using Clock = std::chrono::steady_clock;

    explicit StartupRetryWindow(Clock::duration length) noexcept
        : length_(length) {}

    // Until when startup may retry after this failure; nothing when the
    // failure is not temporary or its window has closed.
    [[nodiscard]] std::optional<Clock::time_point>
    retryUntil(const RuntimeBootstrapReport &failure, Clock::time_point now);

private:
    Clock::duration length_;
    std::optional<Clock::time_point> deadline_;
    RuntimeBootstrapStage stage_ = RuntimeBootstrapStage::ResourceAssembly;
};

// Whether memory may not hold a request of contextTokens: the plan within
// what the host had available at startup, beyond its reserve and the warning
// margin, holds less. The estimate is conservative, since macOS compresses
// other applications further once the engine loads.
[[nodiscard]] bool memoryMayNotHold(const EngineMemoryPlan &plan,
                                    uint64_t hostAvailableBytes,
                                    uint32_t contextTokens);

struct RuntimeBootstrapConfig {
    RuntimeResourcesConfig resources;
    NativeLoopConfig nativeLoop;
    protocol::ProtocolLimits protocolLimits;
};

using ActualMemoryReporter =
    std::function<ActualMemoryReport(uint64_t estimatedWarmupPeakBytes)>;

// Complete owner returned only after the real loop has emitted its binary
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
    [[nodiscard]] const RuntimeBootstrapReport &report() const noexcept {
        return report_;
    }

private:
    RuntimeBootstrap(std::unique_ptr<RuntimeResources> resources,
                     std::unique_ptr<model::RuntimeModel> modelRuntime,
                     std::unique_ptr<NativeRuntime> nativeLoop,
                     RuntimeBootstrapReport report);

    // Reverse destruction order is loop -> modelRuntime -> resources.
    std::unique_ptr<RuntimeResources> resources_;
    std::unique_ptr<model::RuntimeModel> model_;
    std::unique_ptr<NativeRuntime> nativeLoop_;
    RuntimeBootstrapReport report_;
};

}  // namespace splash::engine
