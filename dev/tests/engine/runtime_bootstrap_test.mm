#include "ProtocolPeer.hpp"
#include "Q8PageFormatReference.hpp"
#include "TestChecks.hpp"
#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "TestStatus.hpp"
#include "engine/Cache.hpp"
#include "engine/Bootstrap.hpp"
#include "TestModel.hpp"

#import <Foundation/Foundation.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using namespace splash;
using namespace splash::engine;
namespace runtime = splash::engine;

using splash::test::require;

class TemporaryModelRoot final {
public:
  TemporaryModelRoot() {
    path_ = std::filesystem::temp_directory_path() /
            ("splash-geometry-" +
             std::string([NSUUID UUID].UUIDString.UTF8String));
    if (!std::filesystem::create_directory(path_))
      throw std::runtime_error("unable to create temporary model root");
    std::filesystem::create_directories(path_ / "tokenizer");
    std::ofstream config(path_ / "tokenizer" / "config.json");
    config << R"({"text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,"hidden_size":5120,"vocab_size":248320}})";
    if (!config)
      throw std::runtime_error("unable to write tokenizer config");
  }
  ~TemporaryModelRoot() { std::filesystem::remove_all(path_); }

  const std::filesystem::path &path() const noexcept { return path_; }
  void write(std::string_view document) const {
    std::ofstream output(path_ / "manifest.json");
    output << document;
    if (!output)
      throw std::runtime_error("unable to write temporary model manifest");
  }

private:
  std::filesystem::path path_;
};

std::string executionManifest(uint32_t draftRows = 8,
                              std::string_view extraGeometry = {}) {
  std::ostringstream out;
  out << R"({"schema_version":3,"model":"Qwen3.8-27B-DFlash2","format":{"name":"splash-packed-q4","q4_bits":4,"q4_group_size":64,"q4_storage_n":256,"section_alignment_bytes":16384,"target_layer_magic":"MDFL0006","draft_layer_magic":"MDFD0004","vision_magic":"MDFV0001"},"execution_geometry":{)"
      << R"("draft_proposal_tokens":7,)"
      << "\"draft_query_rows\":" << draftRows << ','
      << R"("draft_sliding_window":2048,)"
      << R"("maximum_batch_width":4,)"
      << R"("prefill_token_budget":2048,)"
      << R"("target_kv_block_tokens":32,)"
      << R"("target_verify_rows":8)" << extraGeometry << "}}";
  return out.str();
}

void testInstalledManifestBindsExecutionGeometry() {
  TemporaryModelRoot root;
  root.write(executionManifest());
  // The sources' identity is the digest of the record naming them.
  const std::string sourceIdentity = model::inspectModelPackage(root.path()).sourceIdentity;
  require(sourceIdentity == model::weightDigest(executionManifest()),
          "the sources' identity is not the manifest's digest");

  root.write(executionManifest(7));
  try {
    static_cast<void>(model::inspectModelPackage(root.path()));
    throw std::runtime_error("geometry mismatch was accepted");
  } catch (const std::invalid_argument &error) {
    require(std::string_view(error.what()).find("draft_query_rows") !=
                std::string_view::npos,
            "geometry mismatch did not identify its field");
  }

  root.write(executionManifest(8, R"(,"description":"package metadata")"));
  require(model::inspectModelPackage(root.path()).sourceIdentity != sourceIdentity,
          "another manifest named the same sources");

  // A whole number written as a float, as Python writes 1e7, is that
  // integer; a fraction is not one.
  std::string floatRows = executionManifest();
  const std::string rows = "\"draft_query_rows\":8,";
  const size_t rowsAt = floatRows.find(rows);
  require(rowsAt != std::string::npos, "test manifest lost draft rows");
  root.write(std::string(floatRows).replace(rowsAt, rows.size(),
                                            "\"draft_query_rows\":8.0,"));
  static_cast<void>(model::inspectModelPackage(root.path()));
  root.write(floatRows.replace(rowsAt, rows.size(),
                               "\"draft_query_rows\":8.5,"));
  try {
    static_cast<void>(model::inspectModelPackage(root.path()));
    throw std::runtime_error("fractional geometry was accepted");
  } catch (const std::invalid_argument &error) {
    require(std::string_view(error.what()).find("draft_query_rows") !=
                std::string_view::npos,
            "fractional geometry did not identify its field");
  }

  std::string missingGeometry = executionManifest();
  const std::string requiredField = "\"draft_sliding_window\":2048,";
  const size_t field = missingGeometry.find(requiredField);
  require(field != std::string::npos, "test manifest lost required geometry");
  missingGeometry.erase(field, requiredField.size());
  root.write(missingGeometry);
  try {
    static_cast<void>(model::inspectModelPackage(root.path()));
    throw std::runtime_error("missing geometry field was accepted");
  } catch (const std::invalid_argument &error) {
    require(std::string_view(error.what()).find("draft_sliding_window") !=
                std::string_view::npos,
            "missing geometry field did not identify its name");
  }

  std::string wrongStorage = executionManifest();
  const size_t storage = wrongStorage.find("\"q4_storage_n\":256");
  require(storage != std::string::npos, "test manifest lost Q4 storage");
  wrongStorage.replace(storage, std::string("\"q4_storage_n\":256").size(),
                       "\"q4_storage_n\":128");
  root.write(wrongStorage);
  try {
    static_cast<void>(model::inspectModelPackage(root.path()));
    throw std::runtime_error("wrong Q4 storage was accepted");
  } catch (const std::invalid_argument &error) {
    require(std::string_view(error.what()).find("q4_storage_n") !=
                std::string_view::npos,
            "Q4 storage mismatch did not identify the weight format");
  }
}

// The identity reports the loaded model's digests in lowercase hex and the
// KV layout as loaded; a malformed digest or a missing build id fails before
// anything is served.
void testRuntimeCacheIdentityReportsTheLoadedModel() {
  constexpr kv::Layout int8Layout{16, 4, 256};
  constexpr kv::Layout bf16Layout{16, 4, 256, kv::Format::BFloat16};
  const std::string combined(64, 'A');
  const std::string target(64, 'c');
  const engine::RuntimeCacheIdentity int8 =
      engine::makeRuntimeCacheIdentity(combined, target, "build", int8Layout);
  require(int8.modelLayoutSha256 == std::string(64, 'a') &&
              int8.targetModelSha256 == target && int8.buildId == "build" &&
              int8.kvLayout == int8Layout,
          "the cache identity did not report the loaded model");
  const engine::RuntimeCacheIdentity bf16 =
      engine::makeRuntimeCacheIdentity(combined, target, "build", bf16Layout);
  require(bf16.kvLayout == bf16Layout && bf16.kvLayout.format != int8.kvLayout.format,
          "the cache identity did not report the KV format");
  const auto rejected = [&](std::string_view combinedDigest,
                            std::string_view targetDigest, std::string_view build) {
    try {
      static_cast<void>(engine::makeRuntimeCacheIdentity(combinedDigest, targetDigest,
                                                         build, int8Layout));
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  require(rejected(std::string(63, 'a'), target, "build") &&
              rejected(combined, std::string(63, 'c') + 'g', "build") &&
              rejected(combined, target, ""),
          "a malformed digest or an empty build id was accepted");
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

class State final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
};

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
  uint64_t snapshotBytes() const noexcept override { return 64; }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override {
    return std::make_shared<State>();
  }
  uint64_t reclaimIdleState(bool, model::IdleMemory) noexcept override { return 0; }
  std::optional<std::string> provideMask(uint64_t,
                                         std::span<const uint32_t>) override {
    return std::nullopt;
  }
  void end(uint64_t) override {}

  model::WarmupStepResult warmupPrefill(uint32_t rows) override {
    if (!rows || rows > model::ExecutionLimits::prefillTokenBudget)
      throw std::invalid_argument("invalid prefill warmup rows");
    lastPrefillRows = rows;
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
  uint32_t lastPrefillRows = 0;
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
  explicit Harness(int throwingStep = -1, bool failReadyWrite = false)
      : backing_(16, 4096, 4), pool_(backing_, 16),
        resources_(pool_, nullptr, nullptr),
        executor_(throwingStep),
        loop_(
            loopConfig(), resources_, executor_,
            [this, failReadyWrite](std::span<const uint8_t> bytes) {
              if (failReadyWrite) {
                throw std::runtime_error("injected output failure");
              }
              output_.insert(output_.end(), bytes.begin(), bytes.end());
            },
            test::readyStatusJson, protocol::ProtocolLimits{}) {
    backing_.commandInFlight = [this] { return loop_.commandInFlight(); };
  }

  Executor &executor() noexcept { return executor_; }
  engine::NativeRuntime &loop() noexcept { return loop_; }
  const std::vector<uint8_t> &output() const noexcept { return output_; }

private:
  static engine::NativeLoopConfig loopConfig() {
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    return config;
  }

  test::TestKvStorage backing_;
  KvPool pool_;
  engine::Cache resources_;
  Executor executor_;
  std::vector<uint8_t> output_;
  engine::NativeRuntime loop_;
};

void testAllNativeWarmupsPrecedeReady() {
  const EngineMemoryPlan plan = memoryPlan();
  Harness harness;
  require(!harness.loop().ready() && harness.output().empty(),
          "runtime became visible before warmup");
  auto report = engine::RuntimeBootstrap::requireWarmupAndAnnounce(
      plan, harness.executor(), [&] { return validActual(plan); },
      harness.loop());
  require(report.stage == RuntimeBootstrapStage::Ready && report.memoryAudit.valid &&
              harness.loop().ready() && !harness.output().empty(),
          "successful native bootstrap was incomplete");
  require(harness.executor().calls == std::vector<int>({0, 1, 2, 3, 4, 5}),
          "bootstrap did not warm fixed prefill and B1/B2/B3/B4 in order");
  require(harness.executor().lastPrefillRows == model::ExecutionLimits::prefillTokenBudget,
          "bootstrap memory warmup did not explicitly use maximum prefill rows");
  require(report.warmup.decodeBatches[2] == WarmupStepStatus::Complete,
          "bootstrap report omitted the real B3 graph");
}

RuntimeBootstrapReport warmup(Harness &harness, const EngineMemoryPlan &plan) {
  return RuntimeBootstrap::requireWarmupAndAnnounce(
      plan, harness.executor(), [&] { return validActual(plan); },
      harness.loop());
}

void requireReadyWithoutReducingConcurrency(
    Harness &harness, const RuntimeBootstrapReport &report) {
  require(report.stage == RuntimeBootstrapStage::Ready && report.memoryAudit.valid &&
              harness.loop().ready(),
          "memory-limited warmup did not become ready");
  const auto events = protocol::peer::decodeEvents(harness.output());
  require(events.size() == 1, "bootstrap did not emit one complete Ready frame");
  const auto *ready = std::get_if<protocol::ReadyEvent>(&events.front());
  require(ready && ready->maxConcurrentRequests == 4,
          "startup budget permanently reduced the advertised concurrency");
}

void testBudgetLimitedWarmupKeepsRuntimeConcurrency() {
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
    requireReadyWithoutReducingConcurrency(harness, report);
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
      requireReadyWithoutReducingConcurrency(harness, report);
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
        RuntimeResourcesError resourceError(stage, message, "{\"budget\":1}",
                                             "budget details", failure);
        RuntimeBootstrapError error(resourceError);
        const auto &report = error.report();
        require(report.stage == RuntimeBootstrapStage::ResourceAssembly &&
                    report.resourceFailure == failure &&
                    report.message == message &&
                    report.memoryPlanJson == "{\"budget\":1}" &&
                    report.budgetDescription == "budget details",
                "bootstrap lost resource failure classification or diagnostics");
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
      throw std::runtime_error("required allocation refusal announced ready");
    } catch (const RuntimeBootstrapError &error) {
      require(error.report().resourceFailure == resourceAllocationFailure(failure) &&
                  !harness.loop().ready() && harness.output().empty(),
              "required warmup erased allocation refusal classification");
    }
  }
}

void testFinalHostPressurePreventsReady() {
  const EngineMemoryPlan plan = memoryPlan();
  Harness harness;
  try {
    static_cast<void>(RuntimeBootstrap::requireWarmupAndAnnounce(
        plan, harness.executor(),
        []() -> ActualMemoryReport {
          throw metal::MetalAllocationError("pressure after warmup",
                                             metal::AllocationFailure::HostPressure);
        }, harness.loop()));
    throw std::runtime_error("final pressure check announced ready");
  } catch (const RuntimeBootstrapError &error) {
    require(error.report().resourceFailure == RuntimeResourceFailure::HostCapacity &&
                !harness.loop().ready() && harness.output().empty(),
            "final host pressure lost retryability or announced ready");
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
        throw std::runtime_error("invalid warmup announced ready");
      } catch (const RuntimeBootstrapError &error) {
        require(error.report().stage == stages[step] &&
                    (failure == Failure::Allocation ||
                     error.report().resourceFailure == RuntimeResourceFailure::Other) &&
                    !harness.loop().ready() &&
                    harness.output().empty() &&
                    harness.executor().calls.back() == step,
                "warmup failure was swallowed as a memory-limited success");
      }
    }
  }
}

void testExceptionsMemoryAndReadyWriteAreFailClosed() {
  const EngineMemoryPlan plan = memoryPlan();
  {
    Harness harness(3);
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(), [] { return ActualMemoryReport{}; },
          harness.loop()));
      throw std::runtime_error("warmup exception announced ready");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::DecodeWarmup &&
                  !harness.loop().ready(),
              "warmup exception was not contained");
    }
  }
  {
    Harness harness;
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(), [] { return ActualMemoryReport{}; },
          harness.loop()));
      throw std::runtime_error("invalid memory report announced ready");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::MemoryAudit &&
                  !harness.loop().ready(),
              "invalid memory report escaped the audit");
    }
  }
  {
    Harness harness(-1, true);
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(), [&] { return validActual(plan); },
          harness.loop()));
      throw std::runtime_error("failed Ready write left runtime ready");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::AnnounceReady &&
                  !harness.loop().ready() && harness.output().empty(),
              "Ready write failure left a visible runtime");
    }
  }
}

void testStartupRetryWindowOpensAtFirstFailure() {
  using namespace std::chrono_literals;
  RuntimeBootstrapReport failure;
  failure.resourceFailure = RuntimeResourceFailure::HostCapacity;
  StartupRetryWindow window(30s);
  // A cold start fails for the first time after minutes of preparation.
  const auto first = StartupRetryWindow::Clock::time_point{} + 5min;
  require(window.retryUntil(failure, first) == first + 30s &&
              window.retryUntil(failure, first + 29s) == first + 30s &&
              !window.retryUntil(failure, first + 30s),
          "the startup retry window did not open at the first failure");
  // A retry that fails later in startup made progress: a new window opens.
  // Failing again at that stage or before does not extend it.
  failure.stage = RuntimeBootstrapStage::MaximumPrefill;
  require(window.retryUntil(failure, first + 40s) == first + 70s,
          "progress to a later startup stage did not open a new window");
  for (auto stage : {RuntimeBootstrapStage::MaximumPrefill,
                     RuntimeBootstrapStage::ResourceAssembly}) {
    failure.stage = stage;
    require(window.retryUntil(failure, first + 50s) == first + 70s,
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

// The disk tier suggestion follows the plan within the host's headroom
// beyond its reserve and the warning margin; a host with no more than those
// holds nothing.
void testMemoryMayNotHoldBeyondHostHeadroom() {
  const EngineMemoryPlan plan = memoryPlan();
  const auto &budget = plan.breakdown();
  const uint64_t held = EngineMemoryPolicy::hostAvailableReserveBytes(
                            budget.physicalMemoryBytes) +
                        kHostWarningMarginBytes;
  const uint64_t available = held + budget.minimumRequiredBytes + 64 * kMiB;
  const uint32_t fits = plan.contextTokensWithin(available - held);
  require(fits && fits < plan.maximumContextTokens() &&
              !memoryMayNotHold(plan, available, fits) &&
              memoryMayNotHold(plan, available, fits + 1) &&
              !memoryMayNotHold(plan, 64 * kGiB, plan.maximumContextTokens()) &&
              memoryMayNotHold(plan, held, 1),
          "the disk tier suggestion does not follow the host's headroom");
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

} // namespace

int main() {
  try {
    testInstalledManifestBindsExecutionGeometry();
    testRuntimeCacheIdentityReportsTheLoadedModel();
    testAllNativeWarmupsPrecedeReady();
    testBudgetLimitedWarmupKeepsRuntimeConcurrency();
    testOptionalAllocationFailuresAreMemoryLimited();
    testResourceFailureClassificationSurvivesBootstrap();
    testRequiredWarmupPreservesAllocationFailure();
    testFinalHostPressurePreventsReady();
    testWarmupErrorsCannotMasqueradeAsMemoryLimits();
    testExceptionsMemoryAndReadyWriteAreFailClosed();
    testStartupRetryWindowOpensAtFirstFailure();
    testMemoryMayNotHoldBeyondHostHeadroom();
    testProtocolLimitsFollowTheModel();
    std::cout << "native bootstrap tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native bootstrap tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
