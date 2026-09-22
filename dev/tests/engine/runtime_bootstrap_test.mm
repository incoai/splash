#include "Q8PageFormatReference.hpp"
#include "TestImmediateTicket.hpp"
#include "engine/Cache.hpp"
#include "engine/Bootstrap.hpp"
#include "TestModel.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
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

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

void testWarmupLaneComparisons() {
  const model::WarmupLaneResult baseline{
      {17, 32, {101, 102}, false, DecodeStage::Regular, 7, 2, 0}, 103, 34};
  require(baseline == baseline, "warmup result equality is not reflexive");
  require(baseline.sameWorkAs(baseline), "warmup work equality is not reflexive");
  auto requireDifferent = [&](auto change, bool changesWork = true) {
    auto candidate = baseline;
    change(candidate);
    require(candidate != baseline, "warmup comparison ignored an observable result");
    require(candidate.sameWorkAs(baseline) != changesWork,
            "warmup work comparison confused token identity and work counts");
  };
  requireDifferent([](auto &value) { ++value.step.requestId; });
  requireDifferent([](auto &value) { ++value.step.consumedPromptTokens; });
  requireDifferent([](auto &value) { ++value.step.outputTokens[0]; }, false);
  requireDifferent([](auto &value) { value.step.outputTokens.pop_back(); });
  requireDifferent([](auto &value) { value.step.finished = true; });
  requireDifferent([](auto &value) {
    value.step.nextDecodeStage = DecodeStage::RequestInitialMask;
  });
  requireDifferent([](auto &value) { ++value.step.draftedTokens; });
  requireDifferent([](auto &value) { ++value.step.acceptedDraftTokens; });
  requireDifferent([](auto &value) { ++value.step.outputTokensWithoutKv; });
  requireDifferent([](auto &value) { ++*value.pendingToken; }, false);
  requireDifferent([](auto &value) { value.pendingToken.reset(); });
  requireDifferent([](auto &value) { ++value.committedTokens; });
  std::vector lanes{baseline, baseline};
  ++lanes[1].step.requestId;
  auto reordered = lanes;
  std::swap(reordered[0], reordered[1]);
  require(lanes != reordered, "warmup comparison ignored batch-plan lane order");
  require(lanes != std::vector{baseline}, "warmup comparison ignored missing lanes");
}

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
      << R"("allocation_extent_target_bytes":134217728,)"
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
  static_cast<void>(model::inspectModelPackage(root.path()));

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
  static_cast<void>(model::inspectModelPackage(root.path()));

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

void testDescriptorRetainsInspectedManifestDigest() {
  TemporaryModelRoot root, identicalRoot;
  std::string manifest = executionManifest();
  manifest.pop_back();
  manifest += ",\"artifacts\":[{\"path\":\"target/shared.bin\",\"sha256\":\"" +
              std::string(64, 'a') + "\"}]}";
  root.write(manifest);
  identicalRoot.write(manifest);
  const auto inspected = model::inspectModelPackage(root.path());
  const auto originalDigest = inspected.packageManifestSha256;
  require(std::any_of(originalDigest.begin(), originalDigest.end(),
                      [](uint8_t byte) { return byte != 0; }),
          "inspected descriptor omitted its package manifest digest");
  require(model::inspectModelPackage(root.path()).packageManifestSha256 == originalDigest &&
              model::inspectModelPackage(identicalRoot.path()).packageManifestSha256 == originalDigest,
          "identical manifest bytes produced different package digests");

  std::string changedArtifact = manifest;
  const size_t artifactDigest = changedArtifact.find(std::string(64, 'a'));
  require(artifactDigest != std::string::npos, "test manifest lost its artifact digest");
  changedArtifact[artifactDigest] = 'b';
  root.write(changedArtifact);
  require(model::inspectModelPackage(root.path()).packageManifestSha256 != originalDigest,
          "package manifest digest omitted an artifact SHA-256 change");
  require(inspected.packageManifestSha256 == originalDigest,
          "rewriting a manifest changed an already inspected descriptor");

  root.write(manifest + "\n");
  require(model::inspectModelPackage(root.path()).packageManifestSha256 != originalDigest,
          "package manifest digest did not identify the exact parsed bytes");
  require(inspected.packageManifestSha256 == originalDigest,
          "later raw manifest edits changed the loaded descriptor digest");
}

void testRuntimeCacheNamespaceBindsIdentityOnce() {
  constexpr kv::Layout kvLayout{16, 4, 256};
  const std::string combinedA(64, 'a');
  const std::string combinedB(64, 'b');
  const std::string targetA(64, 'c');
  const std::string targetB(64, 'd');
  const engine::RuntimeCacheIdentity first =
      engine::makeRuntimeCacheIdentity(combinedA, targetA, "build-a",
                                       kvLayout);
  const engine::RuntimeCacheIdentity same =
      engine::makeRuntimeCacheIdentity(combinedA, targetA, "build-a",
                                       kvLayout);
  const engine::RuntimeCacheIdentity modelChanged =
      engine::makeRuntimeCacheIdentity(combinedB, targetA, "build-a",
                                       kvLayout);
  const engine::RuntimeCacheIdentity targetChanged =
      engine::makeRuntimeCacheIdentity(combinedA, targetB, "build-a",
                                       kvLayout);
  const engine::RuntimeCacheIdentity buildChanged =
      engine::makeRuntimeCacheIdentity(combinedA, targetA, "build-b",
                                       kvLayout);
  auto bf16Layout = kvLayout;
  bf16Layout.format = kv::Format::BFloat16;
  const auto formatChanged = engine::makeRuntimeCacheIdentity(
      combinedA, targetA, "build-a", bf16Layout);
  require(first.cacheNamespace != formatChanged.cacheNamespace &&
              first.namespaceSha256 != formatChanged.namespaceSha256,
          "INT8 and BF16 aliased the same prefix-cache namespace");
  require(first.cacheNamespace == same.cacheNamespace &&
              first.namespaceSha256 == same.namespaceSha256,
          "runtime cache namespace is not deterministic");
  require(first.cacheNamespace != modelChanged.cacheNamespace &&
              first.cacheNamespace != targetChanged.cacheNamespace &&
              first.cacheNamespace != buildChanged.cacheNamespace,
          "runtime cache namespace omitted model, layout, or build identity");
  require(kv::matchesLayout(first.kvLayout, kvLayout) &&
              first.kvLayout.modelArtifactSha256 !=
                  targetChanged.kvLayout.modelArtifactSha256,
          "runtime Q8 layout guard omitted the target artifact");
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
  result.supportsPlacementSparse = true;
  return result;
}

EngineMemoryPlan memoryPlan() {
  return requireEngineMemoryPlan(
      device(), test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB));
}

ActualMemoryReport validActual(const EngineMemoryPlan &plan) {
  const auto &budget = plan.breakdown();
  ActualMemoryReport actual;
  actual.targetWeightsBytes = budget.targetWeightsBytes;
  actual.draftWeightsBytes = budget.draftWeightsBytes;
  actual.visionWeightsBytes = budget.visionWeightsBytes;
  actual.stateResidentBytes = budget.activeStateCellBytes;
  actual.sharedPrefillBytes = budget.sharedPrefillBytes;
  actual.sharedDecodeBytes = budget.sharedDecodeBytes;
  actual.kvResidentBytes = budget.kvExtentBytes;
  actual.backendAllocatedBytes =
      actual.targetWeightsBytes + actual.draftWeightsBytes +
      actual.visionWeightsBytes +
      actual.stateResidentBytes + actual.sharedPrefillBytes +
      actual.sharedDecodeBytes + actual.kvResidentBytes;
  actual.deviceCurrentAllocatedBytes = actual.backendAllocatedBytes;
  actual.devicePeakAllocatedBytes = actual.backendAllocatedBytes;
  actual.estimatedWarmupPeakBytes = actual.backendAllocatedBytes;
  return actual;
}

class Backing final : public KvBacking {
public:
  explicit Backing(uint32_t pages) : resident_(pages, true) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    resident_.at(page) = true;
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    const bool resident = resident_.at(page);
    resident_.at(page) = false;
    return resident;
  }
  uint32_t extentFirstPage(uint32_t page) const override {
    return page - page % 4;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    return std::min<uint32_t>(4, resident_.size() - extentFirstPage(page));
  }
private:
  std::vector<bool> resident_;
};

class State final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
};

class Executor final : public model::RuntimeModel {
public:
  explicit Executor(uint64_t estimatedPeak, int failingStep = -1,
                    int throwingStep = -1)
      : estimatedPeak_(estimatedPeak), failingStep_(failingStep),
        throwingStep_(throwingStep) {}

  StateAdmission begin(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override {
    return {0, StateFailure::None};
  }
  void restore(uint64_t, uint32_t, std::shared_ptr<const CompositeState>,
                     bool) override {}
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
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override {
    return std::make_shared<State>();
  }
  uint64_t reclaimIdleState() noexcept override { return 0; }
  void provideMask(uint64_t, std::span<const uint32_t>) override {}
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
  model::WarmupStepResult warmupDraftVerifyCommit() override {
    return warmup(5);
  }
  model::WarmupStepResult warmupCompositeStateRestore() override {
    return warmup(6);
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
    model::WarmupStepResult result{
        step != failingStep_, estimatedPeak_, "measured", 0.001, {}};
    if (warmupHook)
      warmupHook(step, result);
    return result;
  }

  uint64_t estimatedPeak_ = 0;
  int failingStep_ = -1;
  int throwingStep_ = -1;
};

class Harness final {
public:
  Harness(const EngineMemoryPlan &plan, int failingStep = -1,
          int throwingStep = -1, bool failReadyWrite = false)
      : backing_(16), pool_(backing_),
        resources_(pool_, CacheNamespace{}),
        executor_(validActual(plan).devicePeakAllocatedBytes, failingStep,
                  throwingStep),
        loop_(
            loopConfig(), resources_, executor_,
            [this, failReadyWrite](std::span<const uint8_t> bytes) {
              if (failReadyWrite) {
                throw std::runtime_error("injected output failure");
              }
              output_.insert(output_.end(), bytes.begin(), bytes.end());
            },
            [] { return std::string("{\"schema_version\":5}"); }) {}

  Executor &executor() noexcept { return executor_; }
  engine::NativeRuntime &loop() noexcept { return loop_; }
  const std::vector<uint8_t> &output() const noexcept { return output_; }

private:
  static engine::NativeLoopConfig loopConfig() {
    engine::NativeLoopConfig config;
    config.engine.maxContext = 1024;
    return config;
  }

  Backing backing_;
  KvPool pool_;
  engine::Cache resources_;
  Executor executor_;
  std::vector<uint8_t> output_;
  engine::NativeRuntime loop_;
};

void testAllNativeWarmupsPrecedeReady() {
  const EngineMemoryPlan plan = memoryPlan();
  Harness harness(plan);
  require(!harness.loop().ready() && harness.output().empty(),
          "runtime became visible before warmup");
  ActualMemoryReport actual = validActual(plan);
  auto report = engine::RuntimeBootstrap::requireWarmupAndAnnounce(
      plan, harness.executor(),
      [&](uint64_t estimate) {
        require(estimate == actual.devicePeakAllocatedBytes,
                "bootstrap lost the maximum measured peak");
        actual.estimatedWarmupPeakBytes = estimate;
        return actual;
      },
      harness.loop());
  require(report.ready && report.warmup.ready() && report.memoryAudit.valid &&
              harness.loop().ready() && !harness.output().empty(),
          "successful native bootstrap was incomplete");
  require(harness.executor().calls == std::vector<int>({0, 1, 2, 3, 4, 5, 6}),
          "bootstrap did not warm fixed prefill and B1/B2/B3/B4 in order");
  require(harness.executor().lastPrefillRows == model::ExecutionLimits::prefillTokenBudget,
          "bootstrap memory warmup did not explicitly use maximum prefill rows");
  require(report.warmup.decodeBatches[2] == WarmupStepStatus::Complete,
          "bootstrap report omitted the real B3 graph");
}

RuntimeBootstrapReport warmup(Harness &harness, const EngineMemoryPlan &plan) {
  return RuntimeBootstrap::requireWarmupAndAnnounce(
      plan, harness.executor(),
      [&](uint64_t estimate) {
        auto actual = validActual(plan);
        actual.estimatedWarmupPeakBytes = estimate;
        return actual;
      },
      harness.loop());
}

void requireReadyWithoutReducingConcurrency(
    Harness &harness, const RuntimeBootstrapReport &report) {
  require(report.ready && report.warmup.ready() && report.memoryAudit.valid &&
              harness.loop().ready(),
          "memory-limited warmup did not become ready");
  protocol::FrameParser parser;
  const auto parsed = parser.consume(harness.output());
  require(!parsed.issue && parsed.frame &&
              parsed.consumedBytes == harness.output().size() && !parser.finish(),
          "bootstrap did not emit one complete Ready frame");
  const auto message = protocol::decodeFrame(*parsed.frame);
  const auto *ready = message ? std::get_if<protocol::ReadyEvent>(&*message.value)
                              : nullptr;
  require(ready && ready->maxConcurrentRequests == 4,
          "startup budget permanently reduced the advertised concurrency");
}

void testBudgetLimitedWarmupKeepsRuntimeConcurrency() {
  const auto complete = memoryPlan().breakdown();
  for (uint32_t width : {1U, 2U, 3U}) {
    // Enough for the requested resident cells and one KV extent, with less
    // than one extra cell of headroom. This is a valid single-lane plan.
    const uint64_t ceiling = complete.minimumRequiredBytes +
                            (width - 1) * complete.activeStateCellBytes +
                            complete.activeStateCellBytes / 2;
    const EngineMemoryPlan plan = requireEngineMemoryPlan(
        device(), test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB),
        ceiling);
    Harness harness(plan);
    const auto report = warmup(harness, plan);
    requireReadyWithoutReducingConcurrency(harness, report);
    std::vector<int> expected{0};
    for (uint32_t lane = 1; lane <= width; ++lane)
      expected.push_back(static_cast<int>(lane));
    expected.insert(expected.end(), {5, 6});
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
      Harness harness(plan);
      harness.executor().warmupHook = [&](int step, model::WarmupStepResult &) {
        if (step == deniedWidth || (step == 6 && denyRestore))
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
      expected.insert(expected.end(), {5, 6});
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
        require(!report.ready &&
                    report.stage == RuntimeBootstrapStage::ResourceAssembly &&
                    report.resourceFailure == failure &&
                    report.message == message && report.warmup.error == message &&
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
    Harness harness(plan);
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
  Harness harness(plan);
  try {
    static_cast<void>(RuntimeBootstrap::requireWarmupAndAnnounce(
        plan, harness.executor(),
        [](uint64_t) -> ActualMemoryReport {
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

void testEveryWarmupFailureIsFailClosed() {
  const EngineMemoryPlan plan = memoryPlan();
  constexpr engine::RuntimeBootstrapStage expected[] = {
      engine::RuntimeBootstrapStage::MaximumPrefill,
      engine::RuntimeBootstrapStage::DecodeWarmup,
      engine::RuntimeBootstrapStage::DecodeWarmup,
      engine::RuntimeBootstrapStage::DecodeWarmup,
      engine::RuntimeBootstrapStage::DecodeWarmup,
      engine::RuntimeBootstrapStage::DraftVerifyCommit,
      engine::RuntimeBootstrapStage::CompositeStateRestore,
  };
  for (int step = 0; step < 7; ++step) {
    Harness harness(plan, step);
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(),
          [&](uint64_t estimate) {
            auto actual = validActual(plan);
            actual.estimatedWarmupPeakBytes = estimate;
            return actual;
          },
          harness.loop()));
      throw std::runtime_error("failed warmup announced ready");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage == expected[step] &&
                  error.report().resourceFailure == RuntimeResourceFailure::Other &&
                  !harness.loop().ready() && harness.output().empty(),
              "failed warmup escaped the native bootstrap gate");
    }
  }
}

void testWarmupErrorsCannotMasqueradeAsMemoryLimits() {
  enum class Failure {
    Allocation, Backend, General, MissingPeak, ZeroTime, InfiniteTime, NanTime
  };
  constexpr RuntimeBootstrapStage stages[] = {
      RuntimeBootstrapStage::MaximumPrefill,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DecodeWarmup,
      RuntimeBootstrapStage::DraftVerifyCommit,
      RuntimeBootstrapStage::CompositeStateRestore,
  };
  const EngineMemoryPlan plan = memoryPlan();
  for (int step = 0; step < 7; ++step) {
    for (Failure failure : {Failure::Allocation, Failure::Backend,
                            Failure::General, Failure::MissingPeak,
                            Failure::ZeroTime, Failure::InfiniteTime,
                            Failure::NanTime}) {
      if (failure == Failure::Allocation &&
          (step == 2 || step == 3 || step == 4 || step == 6))
        continue; // Only these paths may skip a real allocation refusal.
      Harness harness(plan);
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
            case Failure::MissingPeak:
              result.estimatedPeakBytes = 0;
              break;
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
                    !error.report().warmup.ready() && !harness.loop().ready() &&
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
    Harness harness(plan, -1, 3);
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(),
          [](uint64_t) { return ActualMemoryReport{}; }, harness.loop()));
      throw std::runtime_error("warmup exception announced ready");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::DecodeWarmup &&
                  !harness.loop().ready(),
              "warmup exception was not contained");
    }
  }
  {
    Harness harness(plan);
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(),
          [](uint64_t) { return ActualMemoryReport{}; }, harness.loop()));
      throw std::runtime_error("invalid memory report announced ready");
    } catch (const engine::RuntimeBootstrapError &error) {
      require(error.report().stage ==
                      engine::RuntimeBootstrapStage::MemoryAudit &&
                  !harness.loop().ready(),
              "invalid memory report escaped the audit");
    }
  }
  {
    Harness harness(plan, -1, -1, true);
    try {
      static_cast<void>(engine::RuntimeBootstrap::requireWarmupAndAnnounce(
          plan, harness.executor(),
          [&](uint64_t estimate) {
            auto actual = validActual(plan);
            actual.estimatedWarmupPeakBytes = estimate;
            return actual;
          },
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

} // namespace

int main() {
  try {
    testWarmupLaneComparisons();
    testInstalledManifestBindsExecutionGeometry();
    testDescriptorRetainsInspectedManifestDigest();
    testRuntimeCacheNamespaceBindsIdentityOnce();
    testAllNativeWarmupsPrecedeReady();
    testBudgetLimitedWarmupKeepsRuntimeConcurrency();
    testOptionalAllocationFailuresAreMemoryLimited();
    testResourceFailureClassificationSurvivesBootstrap();
    testRequiredWarmupPreservesAllocationFailure();
    testFinalHostPressurePreventsReady();
    testEveryWarmupFailureIsFailClosed();
    testWarmupErrorsCannotMasqueradeAsMemoryLimits();
    testExceptionsMemoryAndReadyWriteAreFailClosed();
    std::cout << "native bootstrap tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "native bootstrap tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
