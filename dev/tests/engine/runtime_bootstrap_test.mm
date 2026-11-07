#include "TestChecks.hpp"
#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "Supervision.hpp"
#include "engine/Bootstrap.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelDescriptor.hpp"
#include "TestModel.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::rejects;
using splash::test::require;

// The identity reports the loaded model's digests in lowercase hex and the
// KV layout as loaded; a malformed digest or a missing build id fails before
// anything is served.
void testRuntimeCacheIdentityReportsTheLoadedModel() {
  constexpr kv::Layout int8Layout{16, 4, 256};
  constexpr kv::Layout bf16Layout{16, 4, 256, kv::Format::BFloat16};
  const std::string combined(64, 'A');
  const std::string target(64, 'c');
  const engine::RuntimeCacheIdentity int8 =
      engine::makeRuntimeCacheIdentity(combined, target, "build", int8Layout, {});
  require(int8.modelLayoutSha256 == std::string(64, 'a') &&
              int8.targetModelSha256 == target && int8.buildId == "build" &&
              int8.kvLayout == int8Layout,
          "the cache identity did not report the loaded model");
  const engine::RuntimeCacheIdentity bf16 =
      engine::makeRuntimeCacheIdentity(combined, target, "build", bf16Layout, {});
  require(bf16.kvLayout == bf16Layout && bf16.kvLayout.format != int8.kvLayout.format,
          "the cache identity did not report the KV format");
  const auto identity = [&](std::string_view combinedDigest, std::string_view targetDigest,
                            std::string_view build) {
    static_cast<void>(
        engine::makeRuntimeCacheIdentity(combinedDigest, targetDigest, build, int8Layout, {}));
  };
  rejects([&] { identity(std::string(63, 'a'), target, "build"); },
          "manifest SHA-256 must contain exactly 64 hex characters",
          "a digest of 63 characters was accepted");
  rejects([&] { identity(combined, std::string(63, 'c') + 'g', "build"); },
          "manifest SHA-256 is not hexadecimal", "a digest that is not hexadecimal was accepted");
  rejects([&] { identity(combined, target, ""); }, "runtime build id is required",
          "an empty build id was accepted");
}

DeviceCapabilities device() {
  DeviceCapabilities result;
  result.deviceName = "bootstrap-test";
  result.appleGpuFamily = 9;
  result.macosMajor = 26;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 32 * kGiB;
  result.recommendedMaxWorkingSetBytes = 24 * kGiB;
  result.maxBufferLengthBytes = 16 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  return result;
}

EngineMemoryPlan memoryPlan() {
  return test::requireMemoryPlan(
      device(), test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB));
}

ActualMemoryReport validActual(const EngineMemoryPlan &plan) {
  const auto &budget = plan.breakdown();
  ActualMemoryReport actual;
  actual.targetWeightsBytes = budget.targetWeightsBytes;
  actual.draftWeightsBytes = budget.draftWeightsBytes;
  actual.visionWeightsBytes = budget.visionWeightsBytes;
  actual.stateAllocatedBytes = budget.laneStateBytes;
  actual.sharedPrefillBytes = budget.sharedPrefillBytes;
  actual.sharedDecodeBytes = budget.sharedDecodeBytes;
  actual.kvAllocatedBytes = budget.kvExtentBytes;
  actual.backendAllocatedBytes =
      actual.targetWeightsBytes + actual.draftWeightsBytes +
      actual.visionWeightsBytes +
      actual.stateAllocatedBytes + actual.sharedPrefillBytes +
      actual.sharedDecodeBytes + actual.kvAllocatedBytes;
  actual.deviceCurrentAllocatedBytes = actual.backendAllocatedBytes;
  actual.devicePeakAllocatedBytes = actual.backendAllocatedBytes;
  actual.backendPeakAllocatedBytes = actual.backendAllocatedBytes;
  return actual;
}

class Executor final : public model::RuntimeModel {
public:
  explicit Executor(int throwingStep = -1) : throwingStep_(throwingStep) {}

  StateAdmission begin(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  std::unique_ptr<StateRestore> beginRestore(uint64_t, uint32_t,
                                             std::shared_ptr<const CompositeState>, bool,
                                             std::function<void()>) override {
    return {};
  }
  void setDraftContextPlan(uint64_t, DraftContextPlan) override {}
  std::vector<ModelStepResult> prefill(const BatchPlan &,
                                          std::span<const ModelBatchItem>) {
    return {};
  }
  std::vector<ModelStepResult> decode(const BatchPlan &,
                                         std::span<const ModelBatchItem>) {
    return {};
  }
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override {
    return test::immediateTicket(plan.kind == WorkKind::Prefill
                                     ? prefill(plan, items)
                                     : decode(plan, items),
                                 completion);
  }
  std::shared_ptr<StateReservation> reserveState(uint64_t, uint32_t,
                                                 metal::AllocationFailure &refusal) override {
    refusal = metal::AllocationFailure::EngineBudget;
    return {};
  }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override { return {}; }
  uint64_t releaseIdleBuffer(bool) noexcept override { return 0; }
  std::optional<std::string> provideMask(uint64_t,
                                         std::span<const uint32_t>) override {
    return std::nullopt;
  }
  void end(uint64_t) override {}

  model::WarmupStepResult warmupPrefill(uint32_t rows) override {
    if (!rows || rows > model::ExecutionLimits::prefillTokenBudget)
      throw std::invalid_argument("invalid prefill warmup rows");
    prefillRows.push_back(rows);
    return warmup(0);
  }
  model::WarmupStepResult warmupDecodeBatch(uint32_t width) override {
    if (width < 1 || width > model::ExecutionLimits::maximumBatchWidth) {
      throw std::invalid_argument("invalid decode width");
    }
    return warmup(static_cast<int>(width));
  }
  model::WarmupStepResult warmupCompositeStateRestore() override {
    return warmup(5);
  }
  model::ModelMemoryActual actualRuntimeMemory() const override {
    return {1, 1};
  }
  model::ModelTelemetry telemetry() const noexcept override {
    return {};
  }

  std::vector<int> calls;
  std::vector<uint32_t> prefillRows;
  std::function<void(int, model::WarmupStepResult &)> warmupHook;

private:
  model::WarmupStepResult warmup(int step) {
    calls.push_back(step);
    if (step == throwingStep_) {
      throw std::runtime_error("injected warmup exception");
    }
    model::WarmupStepResult result{"measured", 0.001, {}};
    if (warmupHook)
      warmupHook(step, result);
    return result;
  }

  int throwingStep_ = -1;
};

class Harness final {
public:
  explicit Harness(int throwingStep = -1)
      : backing_(model::ExecutionLimits::warmupKvPages, 4096, 4),
        pool_(backing_, model::ExecutionLimits::warmupKvPages),
        executor_(throwingStep) {}

  Executor &executor() noexcept { return executor_; }
  KvPool &pool() noexcept { return pool_; }

private:
  test::TestKvStorage backing_;
  KvPool pool_;
  Executor executor_;
};

RuntimeBootstrapReport warmup(Harness &harness, const EngineMemoryPlan &plan) {
  return ServingRuntime::requireWarmup(plan, harness.executor(), harness.pool(),
                                         [&] { return validActual(plan); });
}

void requireReady(const RuntimeBootstrapReport &report) {
  require(report.stage == RuntimeBootstrapStage::Ready && report.memoryAudit.valid,
          "memory-limited warmup did not become ready");
}

void testAllNativeWarmupsPrecedeReady() {
  const EngineMemoryPlan plan = memoryPlan();
  Harness harness;
  auto report = warmup(harness, plan);
  require(report.stage == RuntimeBootstrapStage::Ready && report.memoryAudit.valid,
          "successful native bootstrap was incomplete");
  require(harness.executor().calls == std::vector<int>({0, 1, 2, 3, 4, 5}),
          "bootstrap did not warm a prefill and B1/B2/B3/B4 in order");
  require(harness.executor().prefillRows == std::vector<uint32_t>({model::ExecutionLimits::prefillTokenBudget}),
          "bootstrap did not warm a prefill of maximum rows");
  require(report.warmup.decodeBatches[2] == WarmupStepStatus::Complete,
          "bootstrap report omitted the real B3 graph");
}

// Warmup runs on the KV runway and holds its pages as a request holds its
// own (KvPool::RunwayHold); none is held once the engine is Ready.
void testWarmupHoldsTheRunway() {
  const EngineMemoryPlan plan = memoryPlan();
  Harness harness;
  int warmups = 0;
  harness.executor().warmupHook = [&](int, model::WarmupStepResult &) {
    ++warmups;
    for (uint32_t page = 0; page < model::ExecutionLimits::warmupKvPages; ++page)
      require(harness.pool().pageActive(page), "a warmup ran on a runway page no request held");
  };
  const auto report = warmup(harness, plan);
  require(report.stage == RuntimeBootstrapStage::Ready && warmups == 6 &&
              harness.pool().snapshot().pagesActive == 0,
          "warmup held the runway past Ready");
}

void testBudgetLimitedWarmupTriesOnlyWidthsThatFit() {
  const auto complete = memoryPlan().breakdown();
  for (uint32_t width : {1U, 2U, 3U}) {
    // Enough for the requested lanes' state and the KV runway, with less
    // than one more lane's state to spare. This is a valid single-lane plan.
    const uint64_t ceiling = complete.minimumRequiredBytes +
                            (width - 1) * complete.laneStateBytes +
                            complete.laneStateBytes / 2;
    const EngineMemoryPlan plan = test::requireMemoryPlan(
        device(), test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB),
        ceiling);
    Harness harness;
    const auto report = warmup(harness, plan);
    requireReady(report);
    std::vector<int> expected{0};
    for (uint32_t lane = 1; lane <= width; ++lane)
      expected.push_back(static_cast<int>(lane));
    expected.push_back(5);
    require(harness.executor().calls == expected,
            "warmup attempted a decode width that cannot fit the plan");
    for (uint32_t lane = 0; lane < report.warmup.decodeBatches.size(); ++lane) {
      require(report.warmup.decodeBatches[lane] ==
                  (lane < width ? WarmupStepStatus::Complete
                                : WarmupStepStatus::MemoryLimited),
              "budget-skipped decode was reported as measured or pending");
    }
  }
}

void testOptionalAllocationFailuresAreMemoryLimited() {
  const EngineMemoryPlan plan = memoryPlan();
  for (int deniedWidth : {-1, 2, 3, 4}) {
    for (bool denyRestore : {false, true}) {
      Harness harness;
      harness.executor().warmupHook = [&](int step, model::WarmupStepResult &) {
        if (step == deniedWidth || (step == 5 && denyRestore))
          throw metal::MetalAllocationError("injected allocation denial");
      };
      const auto report = warmup(harness, plan);
      requireReady(report);
      const uint32_t completedWidth = deniedWidth < 0 ? 4 : deniedWidth - 1;
      std::vector<int> expected{0};
      for (uint32_t width = 1; width <= completedWidth; ++width)
        expected.push_back(static_cast<int>(width));
      if (deniedWidth >= 0)
        expected.push_back(deniedWidth);
      expected.push_back(5);
      require(harness.executor().calls == expected,
              "bootstrap retried wider batches after allocation denial");
      for (uint32_t lane = 0; lane < report.warmup.decodeBatches.size(); ++lane) {
        require(report.warmup.decodeBatches[lane] ==
                    (lane < completedWidth ? WarmupStepStatus::Complete
                                           : WarmupStepStatus::MemoryLimited),
                "allocation-limited decode status is incorrect");
      }
      require(report.warmup.compositeStateRestore ==
                  (denyRestore ? WarmupStepStatus::MemoryLimited
                               : WarmupStepStatus::Complete),
              "restore allocation denial was reported as a measured success");
    }
  }
}

void testResourceFailureClassificationSurvivesBootstrap() {
  for (RuntimeResourceStage stage : {RuntimeResourceStage::ModelLoading,
                                    RuntimeResourceStage::MemoryPlanning}) {
    for (RuntimeResourceFailure failure : {RuntimeResourceFailure::Other,
                                          RuntimeResourceFailure::HostCapacity,
                                          RuntimeResourceFailure::EngineCapacity,
                                          RuntimeResourceFailure::DriverAllocation}) {
      // Text must neither opt a generic error into retries nor opt a real
      // capacity shortage out. Preserve the diagnostics through wrapping.
      for (const char *message : {
               "currently available; close memory-heavy applications and retry",
               "different diagnostic wording"}) {
        RuntimeResourcesError resourceError(stage, message, "budget details",
                                             failure);
        RuntimeBootstrapError error(resourceError);
        const auto &report = error.report();
        require(report.stage == RuntimeBootstrapStage::ResourceAssembly &&
                    report.resourceStage == stage &&
                    report.resourceFailure == failure &&
                    report.message == message &&
                    report.budgetDescription == "budget details",
                "bootstrap lost resource failure classification or diagnostics");
        require(report.describe().starts_with(
                    "runtime bootstrap failed [resource_assembly/" +
                    std::string(runtimeResourceStageName(stage)) + "]: "),
                "the failure did not name its step of resource assembly");
      }
    }
  }
  RuntimeResourcesError unclassified(RuntimeResourceStage::BackendCreation,
                                      "currently available; close memory-heavy ");
  require(RuntimeBootstrapError(unclassified).report().resourceFailure ==
              RuntimeResourceFailure::Other,
          "resource error became retryable without explicit classification");
}

void testRequiredWarmupPreservesAllocationFailure() {
  const EngineMemoryPlan plan = memoryPlan();
  for (auto failure : {metal::AllocationFailure::HostPressure,
                       metal::AllocationFailure::EngineBudget,
                       metal::AllocationFailure::DriverRejected}) {
    Harness harness;
    harness.executor().warmupHook =
        [failure](int step, model::WarmupStepResult &) {
          if (step == 0)
            throw metal::MetalAllocationError("required allocation", failure);
        };
    try {
      static_cast<void>(warmup(harness, plan));
      throw std::runtime_error("required allocation refusal passed");
    } catch (const RuntimeBootstrapError &error) {
      require(error.report().resourceFailure == resourceAllocationFailure(failure),
              "required warmup erased allocation refusal classification");
    }
  }
}

void testFinalHostPressurePreventsReady() {
  const EngineMemoryPlan plan = memoryPlan();
  Harness harness;
  try {
    static_cast<void>(ServingRuntime::requireWarmup(
        plan, harness.executor(), harness.pool(), []() -> ActualMemoryReport {
          throw metal::MetalAllocationError("pressure after warmup",
                                             metal::AllocationFailure::HostPressure);
        }));
    throw std::runtime_error("final pressure check passed");
  } catch (const RuntimeBootstrapError &error) {
    require(error.report().resourceFailure == RuntimeResourceFailure::HostCapacity,
            "final host pressure lost retryability");
  }
}

void testWarmupErrorsCannotMasqueradeAsMemoryLimits() {
  enum class Failure {
    Allocation, Backend, General, ZeroTime, InfiniteTime, NanTime
  };
  constexpr RuntimeBootstrapStage stages[] = {
      RuntimeBootstrapStage::MaximumPrefill,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::CompositeStateRestore,
  };
  const EngineMemoryPlan plan = memoryPlan();
  for (int step = 0; step < 6; ++step) {
    for (Failure failure : {Failure::Allocation, Failure::Backend,
                            Failure::General, Failure::ZeroTime,
                            Failure::InfiniteTime, Failure::NanTime}) {
      if (failure == Failure::Allocation &&
          (step == 2 || step == 3 || step == 4 || step == 5))
        continue; // Only these paths may skip a real allocation refusal.
      Harness harness;
      harness.executor().warmupHook =
          [&](int current, model::WarmupStepResult &result) {
            if (current != step)
              return;
            switch (failure) {
            case Failure::Allocation:
              throw metal::MetalAllocationError("injected required allocation");
            case Failure::Backend:
              throw metal::MetalBackendError("injected GPU command failure");
            case Failure::General:
              throw std::runtime_error("injected general warmup failure");
            case Failure::ZeroTime:
              result.wallSeconds = 0.0;
              break;
            case Failure::InfiniteTime:
              result.wallSeconds = std::numeric_limits<double>::infinity();
              break;
            case Failure::NanTime:
              result.wallSeconds = std::numeric_limits<double>::quiet_NaN();
              break;
            }
          };
      try {
        static_cast<void>(warmup(harness, plan));
        throw std::runtime_error("invalid warmup passed");
      } catch (const RuntimeBootstrapError &error) {
        require(error.report().stage == stages[step] &&
                    (failure == Failure::Allocation ||
                     error.report().resourceFailure == RuntimeResourceFailure::Other) &&
                    harness.executor().calls.back() == step,
                "warmup failure was swallowed as a memory-limited success");
      }
    }
  }
}

void testExceptionsAndMemoryAreFailClosed() {
  const EngineMemoryPlan plan = memoryPlan();
  {
    Harness harness(3);
    try {
      static_cast<void>(engine::ServingRuntime::requireWarmup(
          plan, harness.executor(), harness.pool(), [] { return ActualMemoryReport{}; }));
      throw std::runtime_error("warmup exception passed");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::DecodeWarmup &&
                  harness.pool().snapshot().pagesActive == 0,
              "warmup exception was not contained or kept the runway held");
    }
  }
  {
    Harness harness;
    try {
      static_cast<void>(engine::ServingRuntime::requireWarmup(
          plan, harness.executor(), harness.pool(), [] { return ActualMemoryReport{}; }));
      throw std::runtime_error("invalid memory report passed");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::MemoryAudit,
              "invalid memory report escaped the audit");
    }
  }
}

void testStartupRetryWindowOpensAtFirstFailure() {
  using namespace std::chrono_literals;
  // A refusal of host memory before the weights load.
  RuntimeBootstrapReport failure;
  failure.resourceStage = RuntimeResourceStage::ModelLoading;
  failure.resourceFailure = RuntimeResourceFailure::HostCapacity;
  StartupRetryWindow window(30s);
  // A slow start fails for the first time minutes in.
  const auto first = StartupRetryWindow::Clock::time_point{} + 5min;
  require(window.retryUntil(failure, first) == first + 30s &&
              window.retryUntil(failure, first + 29s) == first + 30s &&
              !window.retryUntil(failure, first + 30s),
          "the startup retry window did not open at the first failure");
  // A retry that fails later in startup made progress: a new window opens,
  // at a later step of resource assembly, as after the weights loaded, and at
  // a later stage. Failing again there or before does not extend it.
  failure.resourceStage = RuntimeResourceStage::StorageAllocation;
  require(window.retryUntil(failure, first + 40s) == first + 70s,
          "progress through resource assembly did not open a new window");
  failure.stage = RuntimeBootstrapStage::MaximumPrefill;
  failure.resourceStage.reset();
  require(window.retryUntil(failure, first + 80s) == first + 110s,
          "progress to a later startup stage did not open a new window");
  for (const auto &[stage, resourceStage] :
       {std::pair{RuntimeBootstrapStage::MaximumPrefill,
                  std::optional<RuntimeResourceStage>()},
        std::pair{RuntimeBootstrapStage::ResourceAssembly,
                  std::optional(RuntimeResourceStage::StorageAllocation)},
        std::pair{RuntimeBootstrapStage::ResourceAssembly,
                  std::optional(RuntimeResourceStage::ModelLoading)}}) {
    failure.stage = stage;
    failure.resourceStage = resourceStage;
    require(window.retryUntil(failure, first + 90s) == first + 110s,
            "a failure without progress extended the retry window");
  }
  failure.resourceFailure = RuntimeResourceFailure::DriverAllocation;
  require(StartupRetryWindow(30s).retryUntil(failure, first) == first + 30s,
          "a driver allocation failure was not retried");
  for (auto other : {RuntimeResourceFailure::Other,
                     RuntimeResourceFailure::EngineCapacity}) {
    failure.resourceFailure = other;
    require(!StartupRetryWindow(30s).retryUntil(failure, first),
            "a failure that cannot recover was retried");
  }
}

// The parser's limits follow the model: prompts and outputs up to the served
// context, the engine's step and draft query rows, and a mask row of the
// vocabulary for each draft query and the anchor.
void testProtocolLimitsFollowTheModel() {
  model::ModelCapabilities capabilities;
  capabilities.vocabularySize = 248320;
  const protocol::ProtocolLimits limits = protocolLimitsFor(capabilities, 4096);
  require(limits.maxMaskWords == 7760 * 9 && limits.maxSimulationTokens == 8 &&
              limits.maxTokenBatch == 9 && limits.maxPromptTokens == 4096 &&
              limits.maxLogicalOutputTokens == 4096,
          "the protocol limits do not follow the model and the served context");
}

// Ready says what the start serves: the engine's lanes, the context it
// resolved, vision iff it admits images, and the target's vocabulary, stop
// tokens and mask rows, which the server sizes its constraints by.
void testReadyEventForTheServedModel() {
  model::ModelDescriptor flint;
  flint.target = model::FlintLayout{};
  flint.capabilities.vocabularySize = 150144;
  EngineConfig engine;
  engine.maxContext = 73721;
  engine.maxImagePatches = 16384;
  const protocol::ReadyEvent vision =
      readyEventFor(flint, engine, protocolLimitsFor(flint.capabilities, engine.maxContext));
  require(vision == protocol::ReadyEvent{4, 73721, true, 150144, {150001, 150003, 150005, 150018}, 9},
          "Ready did not announce what a start serving Flint with images serves");
  model::ModelDescriptor dense;
  dense.target = model::Qwen3_8Layout{};
  dense.capabilities.vocabularySize = 248320;
  engine.maxContext = 262144;
  engine.maxImagePatches = 0;
  const protocol::ReadyEvent text =
      readyEventFor(dense, engine, protocolLimitsFor(dense.capabilities, engine.maxContext));
  require(text == protocol::ReadyEvent{4, 262144, false, 248320, {248044, 248046}, 9},
          "Ready did not announce what a start serving a model without images serves");
}

// The failure a start reports in place of Ready names its class, which the
// server's restart policy and the app's words read, with what a memory or
// context refusal needed against what it had.
void testStartFailuresAreClassed() {
  using Kind = protocol::StartFailureKind;
  using Bound = protocol::ShortfallBound;
  require(memoryBound(kGiB, kGiB) == Bound::MaxMemory && memoryBound(0, 24 * kGiB) == Bound::DeviceMemory &&
              memoryBound(64 * kGiB, 24 * kGiB) == Bound::DeviceMemory,
          "a budget --max-memory set was not told from this Mac's own");
  const protocol::StartShortfall memory{Bound::MaxMemory, 10'858'659'840ULL, kGiB, 0};
  const RuntimeBootstrapError refused(RuntimeResourcesError(
      RuntimeResourceStage::MemoryPlanning, "the plan's floors require more", "budget",
      RuntimeResourceFailure::EngineCapacity, memory));
  const protocol::StartFailedEvent plan = startFailedEvent(refused.report());
  require(plan.kind == Kind::MemoryBudget && plan.shortfall == memory &&
              plan.message == "the plan's floors require more",
          "a plan the budget cannot hold was not a memory budget failure with its numbers");
  RuntimeBootstrapReport context;
  context.stage = RuntimeBootstrapStage::ModelCreation;
  context.shortfall = protocol::StartShortfall{.bound = Bound::Disk, .planContextTokens = 73721};
  require(startFailedEvent(context).kind == Kind::ContextBudget &&
              startFailedEvent(context).shortfall == *context.shortfall,
          "a context the plan cannot hold was not a context budget failure");
  const auto kindOf = [](std::optional<RuntimeResourceStage> stage, RuntimeResourceFailure failure) {
    RuntimeBootstrapReport report;
    report.resourceStage = stage;
    report.resourceFailure = failure;
    const protocol::StartFailedEvent event = startFailedEvent(report);
    require(event.shortfall == protocol::StartShortfall{}, "a failure without numbers reported a shortfall");
    return event.kind;
  };
  using Stage = RuntimeResourceStage;
  using Failure = RuntimeResourceFailure;
  require(kindOf(Stage::ModelLoading, Failure::HostCapacity) == Kind::HostMemory &&
              kindOf(std::nullopt, Failure::HostCapacity) == Kind::HostMemory &&
              kindOf(Stage::StorageAllocation, Failure::DriverAllocation) == Kind::DriverAllocation &&
              kindOf(std::nullopt, Failure::EngineCapacity) == Kind::MemoryBudget &&
              kindOf(Stage::Configuration, Failure::Other) == Kind::Configuration &&
              kindOf(Stage::CapabilityValidation, Failure::Other) == Kind::Device &&
              kindOf(Stage::ModelLoading, Failure::Other) == Kind::Model &&
              kindOf(Stage::MemoryPlanning, Failure::Other) == Kind::MemoryBudget &&
              kindOf(Stage::BackendCreation, Failure::Other) == Kind::Internal &&
              kindOf(std::nullopt, Failure::Other) == Kind::Internal,
          "a start failure was not classed by its resource failure and stage");
}

// A warmup command that ran out of GPU memory fails the start as
// CommandMemory, which the start never retries: another process holds what
// the model's commands need, and a retry would take it from that process. A
// command failure of another kind is Internal.
void testAStartCommandOutOfMemoryIsCommandMemoryAndNotRetried() {
  const EngineMemoryPlan plan = memoryPlan();
  for (const bool outOfMemory : {true, false}) {
    Harness harness;
    harness.executor().warmupHook = [&](int step, model::WarmupStepResult &) {
      if (step == 0)
        throw metal::MetalBackendError("Metal command 3 failed", outOfMemory);
    };
    try {
      static_cast<void>(warmup(harness, plan));
      throw std::runtime_error("a failed warmup command passed");
    } catch (const RuntimeBootstrapError &error) {
      const RuntimeBootstrapReport &report = error.report();
      require(report.commandOutOfMemory == outOfMemory &&
                  startFailedEvent(report).kind ==
                      (outOfMemory ? protocol::StartFailureKind::CommandMemory
                                   : protocol::StartFailureKind::Internal),
              "a warmup command's want of GPU memory was not classed as CommandMemory");
      StartupRetryWindow window(supervision::kStartRetryWindow);
      require(!window.retryUntil(report, StartupRetryWindow::Clock::now()),
              "a start whose command failed was retried in-process");
    }
  }
}

} // namespace

int main() {
  try {
    testRuntimeCacheIdentityReportsTheLoadedModel();
    testAllNativeWarmupsPrecedeReady();
    testWarmupHoldsTheRunway();
    testBudgetLimitedWarmupTriesOnlyWidthsThatFit();
    testOptionalAllocationFailuresAreMemoryLimited();
    testResourceFailureClassificationSurvivesBootstrap();
    testRequiredWarmupPreservesAllocationFailure();
    testFinalHostPressurePreventsReady();
    testWarmupErrorsCannotMasqueradeAsMemoryLimits();
    testExceptionsAndMemoryAreFailClosed();
    testStartupRetryWindowOpensAtFirstFailure();
    testProtocolLimitsFollowTheModel();
    testReadyEventForTheServedModel();
    testStartFailuresAreClassed();
    testAStartCommandOutOfMemoryIsCommandMemoryAndNotRetried();
    std::cout << "native bootstrap tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native bootstrap tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
