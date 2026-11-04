#pragma once

#include "AwakeClock.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Status.hpp"
#include "engine/ThermalState.hpp"

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
    // What a start refused for want of memory or context needed against what
    // it had: RuntimeResourcesError::shortfall, or the --max-context check.
    std::optional<protocol::StartShortfall> shortfall;
    // A command of the start failed for want of GPU memory
    // (metal::MetalBackendError::outOfMemory): a warmup's, or one of
    // resource assembly (RuntimeResourcesError::commandOutOfMemory).
    bool commandOutOfMemory = false;

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

// The failure a start reports in place of Ready (main.mm), classed for the
// server's restart policy and the app's words.
[[nodiscard]] protocol::StartFailedEvent startFailedEvent(
    const RuntimeBootstrapReport &report);

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

// Temporary host/driver allocation failures can recover during startup,
// within supervision::kStartRetryWindow, checked this often. Configuration
// and compute failures, a command's want of GPU memory among them, remain
// immediate and fail-closed.
inline constexpr auto kStartupMemoryRecoveryPoll = std::chrono::seconds(1);

// The wire limits of a model served with maxContext tokens: prompts and
// outputs up to the context, the engine's step and draft query rows, and a
// mask row per draft query and the anchor.
[[nodiscard]] protocol::ProtocolLimits
protocolLimitsFor(const model::ModelCapabilities &capabilities,
                  uint32_t maxContext) noexcept;

// What a model served under `engine` and `limits` (protocolLimitsFor)
// announces in Ready.
[[nodiscard]] protocol::ReadyEvent readyEventFor(
    const model::ModelDescriptor &model, const EngineConfig &engine,
    const protocol::ProtocolLimits &limits);

struct RuntimeBootstrapConfig {
    RuntimeResourcesConfig resources;
    // A relaunch serves its context (RuntimeResourcesConfig::relaunchContext)
    // as far as the memory plan holds it. Otherwise a zero engine maxContext
    // is what the plan holds, as serve's default, and another one larger
    // than the plan holds fails the bootstrap.
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

    // Completes warmup and validation before announcing `ready`. Warmup
    // runs on kvPool's runway (model::ExecutionLimits::warmupKvPages) and
    // holds its pages while it runs (KvPool::RunwayHold).
    [[nodiscard]] static RuntimeBootstrapReport requireWarmupAndAnnounce(
        const EngineMemoryPlan &memoryPlan,
        model::RuntimeModel &modelRuntime,
        KvPool &kvPool,
        ActualMemoryReporter memoryReporter,
        NativeRuntime &nativeLoop,
        const protocol::ReadyEvent &ready);

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

    // The control pass the transport runs at a command-free point: it
    // samples the thermal state, keeps the disk tier within what its volume
    // leaves it (Cache::keepDiskLimit), and runs the memory control pass
    // (MemoryControl::run).
    [[nodiscard]] bool controlPass(MemoryPressure pressure) {
        sampleThermalState();
        resources_->cache().keepDiskLimit();
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

    // Records the thermal state macOS reports for the status document, and
    // logs a change: nothing else tells why serving slowed down. The control
    // pass is the loop's regular tick, so a polled state needs no observer
    // of its own.
    void sampleThermalState();

    // Reverse destruction order is loop -> modelRuntime -> resources.
    std::unique_ptr<RuntimeResources> resources_;
    std::unique_ptr<model::RuntimeModel> model_;
    std::unique_ptr<NativeRuntime> nativeLoop_;
    MemoryControl memoryControl_;
    RuntimeBootstrapReport report_;
    ThermalStateReporter thermalState_;
};

}  // namespace splash::engine
