#include "ClaimScope.hpp"
#include "ProtocolPeer.hpp"
#include "SyntheticModels.hpp"
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestCheckpoint.hpp"
#include "TestDrafts.hpp"
#include "TestStderr.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"
#include "model/ModelSizing.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/SparseRowStore.hpp"
#include "model/SsdReaders.hpp"
#include "model/WorkFormulas.hpp"
#include "tools/LoadOptions.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::flashNextConfig;
using splash::test::kMoeExperts;
using splash::test::kMoeLayers;
using splash::test::loadableConfig;
using splash::test::localVolume;
using splash::test::moeConfig;
using splash::test::PlannedModel;
using splash::test::plannedModel;
using splash::test::reading;
using splash::test::rejects;
using splash::test::require;

// The target and draft checkpoints of `model` under root, which weight
// planning reads; its vision tower stays absent, so loading stops there.
void writeCheckpoints(const std::filesystem::path &root, const model::ModelDescriptor &model) {
  std::visit([&](const auto &target) { test::writeSyntheticCheckpoints(root, target, model.draft); },
             model.target);
}

RuntimeResourcesConfig budgetConfig(const char *metallibPath, const std::filesystem::path &root) {
  const PlannedModel planned = plannedModel();
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("budget-test", planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  config.buildId = "budget-test";
  writeCheckpoints(root, config.model);
  return config;
}

// What a start sized: its context, the least memory its plan needs, its
// arenas, the budget of states and KV, its streamed table's row cache and
// what the host held back of the streamed stores, its routed experts' store,
// whether its SSD cache started, its backend's memory, its log and how many
// times it loaded the weights.
struct StreamedStart final {
  uint32_t context = 0;
  uint64_t minimumRequired = 0, arenaBytes = 0, dynamicBudget = 0, table = 0, heldBack = 0;
  std::optional<model::ExpertStoreStats> experts;
  bool tier = false;
  metal::MetalMemoryStats memory;
  std::string log;
  size_t loads = 0;
};

// How many times `part` is in `text`.
size_t occurrences(std::string_view text, std::string_view part) {
  size_t count = 0;
  for (size_t at = text.find(part); at != std::string_view::npos; at = text.find(part, at + 1)) ++count;
  return count;
}

// A start of `config` asked for `requestedContextTokens` (--max-context),
// zero for the automatic context.
StreamedStart startStreamed(const RuntimeResourcesConfig &config, uint32_t requestedContextTokens = 0) {
  StreamedStart started;
  started.log = test::capturedStderr([&] {
    auto resources = RuntimeResources::create(config, requestedContextTokens);
    const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
    started.context = resources->memoryPlan().maximumContextTokens();
    started.minimumRequired = budget.minimumRequiredBytes;
    started.arenaBytes = budget.sharedPrefillBytes + budget.sharedDecodeBytes;
    started.dynamicBudget = budget.dynamicBudgetBytes;
    started.table = budget.streamedTableBytes;
    started.heldBack = budget.hostHeldBackBytes;
    started.experts = resources->streamedExpertStats();
    started.tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
    started.memory = resources->backend().refreshMemoryStats();
  });
  started.loads = occurrences(started.log, "Weights loaded");
  return started;
}

// The KV allowance an automatic start of `config` keeps beside the floors
// before a streamable part loads whole: the first 64K tokens (the window
// where that is less) and their speculative rows, in the fewest pages whose
// extents hold them as one request's KV capacity rounds them
// (kvCapacityPagesFor), less the runway among the floors. A part loads whole
// only where the budget holds its floors, all of it and this allowance. It is
// not the allowance a store's growth above its smallest is priced with, which
// in a windowed layout also keeps a streamed command's pages Full.
uint64_t residentAllowanceBytes(const RuntimeResourcesConfig &config) {
  kv::Layout layout = config.model.targetKvLayout;
  layout.format = config.kvFormat;
  const uint64_t tokens = std::min<uint64_t>(config.model.capabilities.maximumContextTokens, 65'536) +
                          model::ExecutionLimits::speculativeScratchTokens;
  const uint64_t pages = (tokens + kv::kPageTokens - 1) / kv::kPageTokens;
  uint64_t budget = pages;
  while (kvCapacityPagesFor(layout, budget) < pages) ++budget;
  return kvPagesBytes(layout, budget, kvWindowFloorPages(layout, 0), layout.extentPagesFor(budget)) -
         kvRunwayPages(layout.minimumExtentPages()) * layout.bytesPerModelPage();
}

// The sizes a model's weights take, planned from its files before it loads,
// are what it loads: the images without its routed experts and those with
// all of them, and the smallest store, which holds a layer's experts and
// refuses a byte less. One load plans the arenas of either: those of streamed
// experts planned from a model that loads them whole are a streaming model's,
// and the reverse.
void testStreamableWeightsAreWhatLoads(const RuntimeResourcesConfig &config) {
  const std::filesystem::path &root = config.modelRoot;
  const model::StreamableWeights weights =
      model::streamableWeights(root, config.model, {.streamExperts = true, .streamTable = true});
  require(weights.expertBytes > weights.expertMinimumBytes && weights.expertMinimumBytes && !weights.tableBytes &&
              !weights.tableMinimumBytes && !weights.tableRowBytes && weights.tableRefusal.empty(),
          "the streamable weights of a MoE without a table are not its experts'");
  const auto allocated = [](const model::LoadedModel &loaded) {
    return loaded.targetActualAllocatedBytes() + loaded.draft.actualAllocatedBytes +
           loaded.vision.actualAllocatedBytes;
  };
  metal::MetalBackend backend(config.metallibPath.string());
  const ops::ExecutionPlans operators(backend.capabilities());
  const auto arenas = [&](const model::LoadedModel &loaded, model::StreamedParts parts) {
    const model::ModelMemoryPlan plan =
        model::plannedRuntimeMemory(loaded, operators, kv::Format::Int8, ops::GdnStateFormat::Float32, parts);
    return plan.sharedPrefillPlannedAllocatedBytes + plan.sharedDecodePlannedAllocatedBytes;
  };
  const model::LoadedModel whole = model::loadModel(backend, root, config.model);
  require(allocated(whole) == weights.loadedBytes + weights.expertBytes,
          "the experts loaded with their layers did not take what was planned");
  // Streamed experts read on a pool of SSD readers, as the engine's do.
  const auto readers = std::make_shared<model::SsdReaders>(model::kSsdReaders);
  const model::LoadedModel streamed = model::loadModel(
      backend, root, config.model, {.expertMemoryBytes = weights.expertMinimumBytes, .readers = readers});
  require(streamed.expertStore() && streamed.expertStore()->stats().slots == kMoeExperts &&
              allocated(streamed) <= weights.loadedBytes + weights.expertMinimumBytes,
          "the smallest store did not hold a layer's experts within its bytes");
  require(arenas(whole, {.experts = true}) == arenas(streamed, model::StreamedParts::loaded(streamed)) &&
              arenas(streamed, {}) == arenas(whole, model::StreamedParts::loaded(whole)) &&
              arenas(whole, {.experts = true}) > arenas(whole, {}),
          "the arenas of what streams are not those of a model loaded so, or prefill groups took no more");
  rejects([&] {
    static_cast<void>(
        model::loadModel(backend, root, config.model,
                         {.expertMemoryBytes = weights.expertMinimumBytes - 1, .readers = readers}));
  }, "holds fewer slots", "a store a byte below the smallest loaded");
}

// A tool's --expert-memory SIZE (tools::takeLoadOption) is the store, as
// splash serve's is: SIZE in whole slots, all of it its floor, which it
// starts at and keeps. A SIZE between the smallest store and every expert;
// the synthetic target's experts are of one size, so the store is short of
// SIZE by less than two slots (one, and its buffers' page rounding).
void testAToolsStoreIsItsSize(const RuntimeResourcesConfig &config) {
  const model::StreamableWeights weights =
      model::streamableWeights(config.modelRoot, config.model, {.streamExperts = true});
  const uint64_t size = (weights.expertMinimumBytes + weights.expertBytes) / 2;
  model::ModelLoadOptions options{.readers = std::make_shared<model::SsdReaders>(model::kSsdReaders)};
  require(tools::takeLoadOption("--expert-memory", std::to_string(size), options),
          "--expert-memory is not a tool's load option");
  metal::MetalBackend backend(config.metallibPath.string());
  const model::LoadedModel loaded = model::loadModel(backend, config.modelRoot, config.model, options);
  require(loaded.expertStore() != nullptr, "a tool's --expert-memory below every expert did not stream them");
  const model::ExpertStoreStats store = loaded.expertStore()->stats();
  require(store.floorBytes == store.capBytes && store.liveBytes == store.capBytes && store.capBytes <= size &&
              size - store.capBytes < 2 * store.arenaBytes / store.slots,
          "a tool's store of --expert-memory " + std::to_string(size) + " bytes has a floor of " +
              std::to_string(store.floorBytes) + " bytes and holds " + std::to_string(store.liveBytes) +
              " of its " + std::to_string(store.capBytes));
}

// --expert-memory auto loads a MoE target's routed experts where the plan
// holds them, in one load as main does. Where it does not, they stream
// through a store of all that the first 64K tokens of KV leave, in two
// loads: the first plans the arenas of either, and the second loads that
// store; the backend's peaks, which the memory audit holds to the plan,
// count only the second. The automatic SSD cache gives its state staging up
// where it alone would stream them. Off loads them whatever the budget, a
// budget without room for their arenas stops the start with its breakdown,
// a size streams them through a store of exactly that size in whole slots,
// its floor and its cap, in one load, whether or not memory holds every
// expert, and a size below the smallest store is refused by name before
// anything loads.
void testAutomaticExpertMemory(RuntimeResourcesConfig config) {
  const model::StreamableWeights weights =
      model::streamableWeights(config.modelRoot, config.model, {.streamExperts = true});
  const StreamedStart whole = startStreamed(config);
  require(!whole.experts && whole.loads == 1 &&
              whole.log.find("Routed experts load with their layers") != std::string::npos,
          "routed experts memory holds did not load with their layers in one load");
  const auto streams = [&](const StreamedStart &start, const char *message) {
    require(start.experts && start.experts->slots > kMoeExperts &&
                start.experts->slots < kMoeLayers * kMoeExperts && start.loads == 2 &&
                start.log.find("--expert-memory auto leaves it beside the floors") != std::string::npos &&
                start.context >= 65'536,
            message);
  };
  // The budget that holds every expert: the floors, all of the experts and
  // the KV kept resident (residentAllowanceBytes).
  const uint64_t holdsEvery = whole.minimumRequired + residentAllowanceBytes(config);
  // A byte short of it, which the plan without the arenas holds: whole, then
  // through the store. The first load's experts do not count toward the
  // peaks.
  config.maximumMemoryBytes = holdsEvery - 1;
  const StreamedStart reloaded = startStreamed(config);
  streams(reloaded,
          "experts that fit beside everything but the arenas did not stream through the store the plan sizes");
  // The store starts at its floor, the chunks above it reserved and held in
  // the plan (testTwoLaunchesGetOneFloor).
  require(reloaded.experts->chunks < reloaded.experts->capChunks &&
              reloaded.experts->liveBytes == reloaded.experts->floorBytes &&
              reloaded.heldBack == reloaded.experts->capBytes - reloaded.experts->liveBytes,
          "a streamed store did not start at its floor, the chunks above it held in the plan");
  const uint64_t firstLoad = weights.loadedBytes + weights.expertBytes;
  require(reloaded.memory.peakAllocatedBytes < firstLoad && reloaded.memory.devicePeakAllocatedBytes < firstLoad,
          "the backend's peaks counted the load the start discarded");
  // Short of that too: at the floor, then through the store.
  config.maximumMemoryBytes = holdsEvery - whole.arenaBytes - 1;
  streams(startStreamed(config),
          "routed experts memory cannot hold did not stream through the store the plan sizes");

  // A byte short of every expert beside the automatic SSD cache's staging:
  // the cache gives it up, and they load whole in one load.
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  config.maximumMemoryBytes = holdsEvery + config.model.stateLayout.cachedBytes() - 1;
  const StreamedStart yielded = startStreamed(config);
  require(!yielded.experts && !yielded.tier && yielded.loads == 1 &&
              yielded.log.find("SSD cache off: its " + std::to_string(config.model.stateLayout.cachedBytes() >> 20) +
                               " MiB of state staging would stream the routed experts") != std::string::npos,
          "the automatic SSD cache's staging streamed experts that load whole without it");
  config.maximumMemoryBytes += 1;
  const StreamedStart kept = startStreamed(config);
  require(!kept.experts && kept.tier && kept.loads == 1, "the automatic SSD cache yielded room it had");
  config.cacheDisk = {};
  config.diskVolume = RuntimeResourcesConfig{}.diskVolume;

  config.expertMemory = {StreamedSetting::Mode::Off, 0};
  config.maximumMemoryBytes = 0;
  require(!startStreamed(config).experts, "--expert-memory off streamed the experts");
  // A budget that holds the weights but not their arenas: the start loads
  // them, and stops with the engine plan's breakdown.
  config.maximumMemoryBytes = whole.minimumRequired - whole.arenaBytes / 2;
  try {
    static_cast<void>(RuntimeResources::create(config, 0));
    throw std::runtime_error("a plan without room for its arenas started");
  } catch (const RuntimeResourcesError &error) {
    require(error.stage() == RuntimeResourceStage::MemoryPlanning &&
                error.budgetDescription().find("deficit: " + std::to_string(whole.arenaBytes / 2) + " bytes") !=
                    std::string::npos,
            "a start without room for its arenas was refused without its breakdown");
  }
  config.maximumMemoryBytes = 0;
  const uint64_t size = (weights.expertMinimumBytes + weights.expertBytes) / 2;
  config.expertMemory = {StreamedSetting::Mode::Size, size};
  for (const uint64_t limit : {uint64_t{0}, whole.minimumRequired - 1}) {
    config.maximumMemoryBytes = limit;
    const StreamedStart sized = startStreamed(config);
    require(sized.experts && sized.loads == 1 && sized.experts->floorBytes == sized.experts->capBytes &&
                sized.experts->liveBytes == sized.experts->capBytes && sized.experts->capBytes <= size &&
                size - sized.experts->capBytes < 2 * sized.experts->arenaBytes / sized.experts->slots &&
                !sized.heldBack &&
                sized.log.find("the " + std::to_string(size >> 20) + " MiB --expert-memory sets") !=
                    std::string::npos &&
                sized.log.find("it gives none back") != std::string::npos,
            "an explicit size did not stream the experts through a store of that size, its floor and its cap, "
            "in one load");
  }
  config.maximumMemoryBytes = 0;
  config.expertMemory = {StreamedSetting::Mode::Size, weights.expertMinimumBytes - 1};
  try {
    static_cast<void>(RuntimeResources::create(config, 0));
    throw std::runtime_error("a size below the smallest store was accepted");
  } catch (const RuntimeResourcesError &error) {
    require(error.stage() == RuntimeResourceStage::Configuration &&
                std::string_view(error.what()).find("is less than its smallest store") != std::string_view::npos,
            "a size below the smallest store was not refused by name before loading");
  }
}

// --verify-rows: by default a MoE target's decode cycles verify the rows the
// verify budget chooses, whether its routed experts load with their layers or
// stream from the SSD, and a dense target's every row; `all` verifies every
// row. The start says which, and the runtime built on what it assembled runs
// it.
void testVerifyRows(const char *metallibPath, RuntimeResourcesConfig moe) {
  struct Started final {
    std::string log;
    bool budget = false;
  };
  const auto start = [](const RuntimeResourcesConfig &config) {
    Started started;
    started.log = test::capturedStderr([&] {
      const auto resources = RuntimeResources::create(config, 0);
      const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
      std::unique_ptr<model::RuntimeModel> runtime;
      require(static_cast<bool>(resources->memoryGovernor().allocationAdmission()(
                  budget.sharedPrefillBytes + budget.sharedDecodeBytes,
                  [&] { runtime = model::createRuntime(resources->modelContext()); })),
              "a runtime's arenas were refused");
      started.budget = runtime->telemetry().verifyBudget;
    });
    return started;
  };
  const auto requireStart = [&](const RuntimeResourcesConfig &config, std::string_view line, bool budget,
                                const char *message) {
    const Started started = start(config);
    require(started.log.find(line) != std::string::npos && started.budget == budget, message);
  };
  const std::string_view budgeted = "Decode cycles verify the rows the verify budget chooses (--verify-rows auto, as "
                                    "moe-test is a MoE target): greedy and seeded outputs can differ from run to run.";
  requireStart(moe, budgeted, true, "a MoE target with resident experts did not budget its rows");
  // A budget a byte short of the whole model's minimum, below the one that
  // holds every expert and the KV allowance, streams them.
  moe.maximumMemoryBytes = startStreamed(moe).minimumRequired - 1;
  requireStart(moe, budgeted, true, "a MoE target whose experts stream did not budget its rows");
  moe.verifyRows = model::VerifyRows::All;
  requireStart(moe, "Decode cycles verify every row of the draft's block (--verify-rows all).", false,
               "--verify-rows all did not verify every row of a MoE target");
  const test::TemporaryDirectory root("splash-verify-rows");
  requireStart(loadableConfig(metallibPath, root.path(), "verify-rows-test"),
               "Decode cycles verify every row of the draft's block (--verify-rows auto, as verify-rows-test is "
               "dense).",
               false, "a dense target did not verify every row");
}

// Startup fails at the loader, whose vision weight files are deliberately
// absent. Reaching it is the assertion: everything the engine checks before
// loading the weights let this configuration through.
void requireReachesModelLoader(RuntimeResourcesConfig config,
                               const std::filesystem::path &root,
                               const char *message) {
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model without weights unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    const std::string_view text = error.what();
    require(error.failure() == RuntimeResourceFailure::Other &&
                error.stage() == RuntimeResourceStage::ModelLoading &&
                text.find(root.string()) != std::string_view::npos,
            message);
  }
}

// Beside its weights a model needs at least the runtime reserves, one lane's
// state and the KV runway.
uint64_t minimumBytes(const RuntimeResourcesConfig &config) {
  return minimumRequiredBytes(model::streamableWeights(config.modelRoot, config.model, {}).loadedBytes +
                                  model::kPipelineReserveBytes + model::kRuntimeOverheadReserveBytes,
                              config.model.stateLayout.laneBytes(),
                              config.model.targetKvLayout)
      .value();
}

// A persistent cache's directory is named for what its copies hold: the same
// models and layouts name the same one under any build, and a change to any
// of them, or to the RoPE scaling its keys went through, names another.
void testPersistentCacheNamespace() {
  const model::ModelDescriptor model = model::makeModelDescriptor(
      "namespace-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
      test::qwen3_8Vision(), model::TargetSource::Safetensors,
      model::VisionSource::Safetensors);
  const auto identity = [&](char models, std::string_view build, kv::Format format) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = format;
    return makeRuntimeCacheIdentity(std::string(64, models), std::string(64, 'b'), build,
                                    layout, {});
  };
  const std::string name =
      persistentCacheNamespace(identity('a', "build", kv::Format::Int8), model.stateLayout);
  require(name.size() == 32 && name.find_first_not_of("0123456789abcdef") == std::string::npos,
          "a cache namespace is not 32 lowercase hex digits");
  require(persistentCacheNamespace(identity('a', "another build", kv::Format::Int8),
                                   model.stateLayout) == name,
          "the build changed the cache namespace");
  model::CompositeStateLayout states = model.stateLayout, bfloatStates = model.stateLayout;
  ++states.draft.layers;
  bfloatStates.target.format = ops::GdnStateFormat::BFloat16;
  require(persistentCacheNamespace(identity('c', "build", kv::Format::Int8), model.stateLayout) !=
                  name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::BFloat16),
                                       model.stateLayout) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), states) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), bfloatStates) != name,
          "other models, layouts or state formats shared a cache namespace");
  // A dense layout's namespace is the one main (be3b083c) names, with FP32
  // and BF16 states: a change of its lines would leave every 27B's and 35B's
  // persistent cache cold.
  require(name == "6365d687c8be103a3bbcc6d2a83b1543" &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), bfloatStates) ==
                  "172879058b5e2bcc713e129a2b62393a",
          "a dense layout's cache namespace is not the one main names");
  // A QSA layout's pages hold token-major values and the index's pooled
  // keys, and its states the PLE history and the index's pending keys.
  RuntimeCacheIdentity tokenMajor = identity('a', "build", kv::Format::Int8), indexed = tokenMajor;
  tokenMajor.kvLayout.valuesTokenMajor = true;
  indexed.kvLayout.indexDimension = 128;
  model::CompositeStateLayout auxiliary = model.stateLayout;
  auxiliary.target.auxiliaryBytes = 9216;
  require(persistentCacheNamespace(tokenMajor, model.stateLayout) != name &&
              persistentCacheNamespace(indexed, model.stateLayout) != name &&
              persistentCacheNamespace(identity('a', "build", kv::Format::Int8), auxiliary) != name,
          "token-major values, an index or auxiliary state shared a dense layout's cache namespace");
  const auto scaled = [&](double factor) {
    kv::Layout layout = model.targetKvLayout;
    layout.format = kv::Format::Int8;
    return persistentCacheNamespace(makeRuntimeCacheIdentity(std::string(64, 'a'), std::string(64, 'b'), "build",
                                                             layout, {factor, 262144}),
                                    model.stateLayout);
  };
  require(scaled(1.0) == name && scaled(4.0) != name && scaled(2.0) != name && scaled(2.0) != scaled(4.0),
          "YaRN's keys shared a cache namespace with another scaling's");
  // Factors that print alike to six digits still rotate keys differently.
  require(scaled(3.9999999) != scaled(4.0) && scaled(1.0000001) != scaled(1.0000004),
          "factors alike to six digits shared a cache namespace");
  // A target without GDN layers stores nothing in the state format.
  const RuntimeCacheIdentity dense = identity('a', "build", kv::Format::Int8);
  model::CompositeStateLayout attentionOnly{{}, model.stateLayout.draft}, bfloatAttentionOnly = attentionOnly;
  bfloatAttentionOnly.target.format = ops::GdnStateFormat::BFloat16;
  require(persistentCacheNamespace(dense, attentionOnly) != name &&
              persistentCacheNamespace(dense, attentionOnly) == persistentCacheNamespace(dense, bfloatAttentionOnly),
          "a target without GDN layers shared a hybrid's namespace, or took another for its state format");
  // A sliding window orders a page's regions and labels its slots with their
  // page's form: each window and each set of windowed layers names a cache of
  // its own.
  RuntimeCacheIdentity windowed = dense, otherWindow = dense, otherLayers = dense;
  windowed.kvLayout.windowTokens = otherLayers.kvLayout.windowTokens = 512;
  otherWindow.kvLayout.windowTokens = 1024;
  windowed.kvLayout.windowedLayers = otherWindow.kvLayout.windowedLayers = 0b1100;
  otherLayers.kvLayout.windowedLayers = 0b0110;
  const std::string windowName = persistentCacheNamespace(windowed, model.stateLayout);
  require(windowName != name && persistentCacheNamespace(otherWindow, model.stateLayout) != windowName &&
              persistentCacheNamespace(otherLayers, model.stateLayout) != windowName,
          "a window, or another window or windowed layers, shared a cache namespace");
}

void testWeightBudgetBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());

  {
    config.memoryPressure = [] { return MemoryPressure::Critical; };
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("model load ignored system pressure");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "startup pressure did not remain retryable");
      require(std::string_view(error.what()).find("not enough free memory") !=
                  std::string_view::npos,
              "startup pressure reached the weight loader");
    }
  }
  config.memoryPressure = [] { return MemoryPressure::Warning; };

  // The low ceiling is one byte short of the minimum, so admission must
  // charge the weights and all a model needs beside them. The other ceilings
  // must reach the real loader, whose expected weight files are deliberately
  // absent.
  const uint64_t minimum = minimumBytes(config);
  for (uint64_t ceiling : {minimum - 1, minimum, uint64_t{0}}) {
    config.maximumMemoryBytes = ceiling;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("a model without weights unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      if (ceiling == minimum - 1) {
        require(error.failure() == RuntimeResourceFailure::EngineCapacity,
                "hard weight budget lost its engine-capacity classification");
        const std::string_view text = error.what();
        require(error.stage() == RuntimeResourceStage::MemoryPlanning &&
                    text.find("require " + std::to_string(minimum) + " bytes") !=
                        std::string_view::npos &&
                    text.find("budget is " + std::to_string(minimum - 1) +
                              " bytes") != std::string_view::npos,
                "weight loading began before checking the memory ceiling");
        // What the start reports in place of Ready (engine::startFailedEvent).
        require(error.shortfall() ==
                    protocol::StartShortfall{.bound = protocol::ShortfallBound::MaxMemory,
                                             .requiredBytes = minimum,
                                             .budgetBytes = minimum - 1},
                "the refusal lost what the plan's floors require against the ceiling");
      } else {
        require(error.failure() == RuntimeResourceFailure::Other,
                "missing model file was misclassified as allocation pressure");
      }
    }
  }
  // Without a ceiling, or with one above this Mac's budget, which only a
  // notice names, the start reaches the loader.
  for (const uint64_t ceiling : {uint64_t{0}, 1024 * kGiB}) {
    config.maximumMemoryBytes = ceiling;
    const std::string log = test::capturedStderr([&] {
      requireReachesModelLoader(config, root.path(),
                                "a sufficient weight budget did not reach the "
                                "model loader");
    });
    require((log.find("--max-memory ignored: its 1024.0 GiB exceed this Mac's Metal budget of ") !=
             std::string::npos) == (ceiling != 0),
            "a ceiling above the budget was ignored without a notice, or no ceiling was named");
  }
}

// A state's write to the disk tier stages through a state-sized buffer the
// plan sets aside only when the tier starts: a quota that holds no state
// leaves the tier off and the budget to KV. An explicit quota's staging is
// planned whatever it costs, and the automatic one never costs a start. The
// notices and the refusal name the tier as users know it, the SSD cache.
void testStateStagingNeedsAStartedTier(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  const uint64_t minimum = minimumBytes(config);
  const uint64_t stateBytes = config.model.stateLayout.cachedBytes();
  config.maximumMemoryBytes = minimum;
  config.cacheDisk = {.bytes = stateBytes - 1};
  const std::string notice = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "a quota that holds no state set staging aside");
  });
  require(notice.find("SSD cache off: its quota holds no state of " +
                      std::to_string(stateBytes >> 20) + " MiB") != std::string::npos,
          "the notice of a quota that holds no state did not name the SSD cache");

  // The automatic tier lets a budget with no room for its staging start,
  // and says why it is off.
  config.cacheDisk = {.automatic = true};
  const std::string automatic = test::capturedStderr([&] {
    requireReachesModelLoader(config, root.path(),
                              "the automatic tier's staging refused a start");
  });
  require(automatic.find("SSD cache off: the memory budget has no room for its " +
                         std::to_string(stateBytes >> 20) +
                         " MiB of state staging beside the model") != std::string::npos,
          "the automatic tier did not say the memory budget turned it off");

  config.cacheDisk = {.bytes = stateBytes};
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model without weights unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                error.stage() == RuntimeResourceStage::MemoryPlanning &&
                std::string_view(error.what())
                        .find("SSD cache state staging require " +
                              std::to_string(minimum + stateBytes) +
                              " bytes") != std::string_view::npos,
            "a started tier's staging was not counted by name before loading");
  }
  config.maximumMemoryBytes = minimum + stateBytes;
  requireReachesModelLoader(config, root.path(),
                            "a budget with room for the staging did not "
                            "reach the model loader");
}

// The image patch limit is a positive multiple of four up to the
// protocol's per-image ceiling; anything else is refused before the backend
// is created.
void testImagePatchCapIsBounded(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  for (const uint32_t patches : {ops::kMaximumImagePatches + 4, 6U}) {
    config.maximumImagePatches = patches;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("an image patch cap outside the protocol's was accepted");
    } catch (const RuntimeResourcesError &error) {
      require(error.stage() == RuntimeResourceStage::Configuration,
              "an invalid image patch cap was not a configuration error");
    }
  }
  config.maximumImagePatches = 4096;
  requireReachesModelLoader(config, root.path(),
                            "a smaller image patch cap did not reach the model loader");
}

// Startup and the governor sample host memory through the config's probe,
// and the automatic SSD cache its volume through another: without either,
// the config is refused before the backend is created.
void testHostMemoryProbeIsRequired(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  for (const bool memory : {true, false}) {
    RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
    if (memory)
      config.hostMemory = {};
    else
      config.diskVolume = {};
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("resources were assembled without a host memory or volume probe");
    } catch (const RuntimeResourcesError &error) {
      require(error.stage() == RuntimeResourceStage::Configuration,
              memory ? "a missing host memory probe was not a configuration error"
                     : "a missing volume probe was not a configuration error");
    }
  }
}

// The 27B under a budget half a GiB above its weights: the weights alone fit,
// but not with what the runtime needs beside them. Startup refuses it before
// any weight is loaded, against the bound that binds: --max-memory, or this
// Mac's Metal budget where that is lower (a 16 GB Mac's 10.8 GiB), which
// --max-memory can only lower. There, a --max-memory half a GiB below the
// Metal budget binds instead.
void testModelBeyondBudgetIsRefusedBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  config.model = model::makeModelDescriptor("budget-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
                                            test::qwen3_8Vision(), model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  const uint64_t device =
      EngineMemoryPolicy::hardBudgetBytes(metal::probeDeviceCapabilities().recommendedMaxWorkingSetBytes, 0);
  const uint64_t above = model::streamableWeights(root.path(), config.model, {}).loadedBytes + kGiB / 2;
  std::vector<uint64_t> limits{above};
  if (above >= device)
    limits.push_back(device - kGiB / 2);
  for (const uint64_t limit : limits) {
    config.maximumMemoryBytes = limit;
    const bool maxMemory = limit <= device;
    const uint64_t budget = maxMemory ? limit : device;
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("a model that cannot fit unexpectedly loaded");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                  error.stage() == RuntimeResourceStage::MemoryPlanning,
              "a model that cannot fit reached the weight loader");
      require(error.shortfall() &&
                  error.shortfall()->bound ==
                      (maxMemory ? protocol::ShortfallBound::MaxMemory : protocol::ShortfallBound::DeviceMemory) &&
                  error.shortfall()->budgetBytes == budget && error.shortfall()->requiredBytes > budget,
              maxMemory ? "the refusal of a model beyond --max-memory lost what it needed against it"
                        : "the refusal of a model beyond this Mac's Metal budget lost what it needed against it");
    }
  }
}

// The rule that keeps users off the startup floor: admission weighs
// reclaimable memory against the macOS reserve, never against the model.
// A model far larger than everything reclaimable still starts, because the
// operation guard checks each image's allocation as loading builds the
// residency up. A quarter of the 27B's layers keeps it within the automatic
// budget of a 24 GB Mac.
void testStartupAdmissionIgnoresModelSize(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  model::Qwen3_8Layout target;
  target.layers = 16;
  model::DFlashDraftLayout draft = test::qwen3_8Draft();
  draft.targetLayers = {1, 4, 7, 10, 13};
  config.model = model::makeModelDescriptor("budget-test", target, draft, test::qwen3_8Vision(),
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  require(model::streamableWeights(root.path(), config.model, {}).loadedBytes > 3 * kGiB,
          "the model must exceed the sample");
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{3 * kGiB, 3 * kGiB}); };
  requireReachesModelLoader(config, root.path(),
                            "a model larger than reclaimable host memory "
                            "refused to start");

  // Below the reserve macOS is the one at risk, so startup waits instead.
  // Unmeasurable telemetry waits the same way.
  for (std::optional<uint64_t> available :
       {std::optional<uint64_t>(64 * kMiB), std::optional<uint64_t>()}) {
    config.hostMemory = [available] {
      return available ? std::optional<HostMemory>(HostMemory{*available, *available}) : std::nullopt;
    };
    try {
      auto resources = RuntimeResources::create(config, 0);
      throw std::runtime_error("model load ignored the macOS reserve");
    } catch (const RuntimeResourcesError &error) {
      require(error.failure() == RuntimeResourceFailure::HostCapacity,
              "exhausted host memory did not remain retryable");
      require(std::string_view(error.what()).find("not enough free memory") !=
                  std::string_view::npos,
              "exhausted host memory reached the weight loader");
    }
  }
}

// The memory plan takes the vision category from what loaded, so a model
// with vision whose loader produced no vision bytes must stop here.
void testLoadedVisionIsRequiredOnlyWithVision() {
  model::LoadedModel loaded;
  loaded.descriptor = model::makeModelDescriptor(
      "loaded-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
      test::qwen3_8Vision(), model::TargetSource::Safetensors,
      model::VisionSource::Safetensors);
  model::Qwen3_8Weights target;
  target.actualAllocatedBytes = 1;
  target.manifestFingerprintSha256 = "target";
  loaded.target = std::move(target);
  loaded.draft.actualAllocatedBytes = 1;
  loaded.manifestFingerprintSha256 = "model";
  require(loaded.descriptor.hasVision(), "the test model has no vision");
  rejects([&] { requireLoadedModel(loaded); },
          "loaded model has incomplete allocation accounting",
          "a multimodal model without loaded vision weights was accepted");
  loaded.vision.actualAllocatedBytes = 1;
  requireLoadedModel(loaded);
  loaded.vision.actualAllocatedBytes = 0;
  loaded.descriptor.visionSource = model::VisionSource::None;
  requireLoadedModel(loaded);
}

// Resource assembly names the step it fails at, at its real throw site: a
// host with less than its reserve refuses the start before the weights are
// opened, and the startup retry window retries it. Past the weights the host
// refuses only what loading would (testStartGrowsThroughHostPressure).
void testFailedStepIsNamed(const char *metallibPath) {
  using namespace std::chrono_literals;
  const test::TemporaryDirectory root("splash-assembly");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "assembly-test");
  const auto host = [&](uint64_t available) {
    config.hostMemory = [available] { return std::optional<HostMemory>(HostMemory{available, available}); };
  };
  // Nothing but the host refuses this model. The suite's one complete start
  // also reports the SSD cache a disk quota gives it, by that name: a size of
  // its temporary files' whole volume, which the room the volume has beyond
  // its floor always limits, as that room.
  host(64 * kGiB);
  const std::optional<DiskVolume> temporary = volumeOf(std::filesystem::temp_directory_path());
  require(temporary && temporary->local, "the temporary files' volume is not a local volume measured");
  config.cacheDisk = {.bytes = temporary->sizeBytes};
  const std::string started =
      test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); });
  const size_t at = started.find("SSD cache: up to ");
  const std::string line =
      at == std::string::npos ? std::string() : started.substr(at, started.find('\n', at) - at);
  require(line.find(" of temporary files in ") != std::string::npos &&
              line.find(" (the ") != std::string::npos &&
              line.find(" its volume has beyond the ") != std::string::npos &&
              line.find("; KV pages of ") != std::string::npos &&
              line.find("--max-cache-disk 0 turns it off.") != std::string::npos,
          "a start did not report its SSD cache by that name, or a size its volume limits as the "
          "volume's room");
  config.cacheDisk = {};
  const auto failure = [&](uint64_t available) {
    host(available);
    try {
      static_cast<void>(RuntimeResources::create(config, 0));
    } catch (const RuntimeResourcesError &error) {
      return RuntimeBootstrapError(error).report();
    }
    throw std::runtime_error("resources were assembled on a host short of memory");
  };
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  const RuntimeBootstrapReport before = failure(reserve / 2);
  require(before.resourceStage == RuntimeResourceStage::ModelLoading &&
              before.resourceFailure == RuntimeResourceFailure::HostCapacity &&
              before.message.find("not enough free memory to start") != std::string::npos,
          "a start the host refused before the weights loaded named another step");
  require(before.message.find("--max-memory") != std::string::npos,
          "a start the host refused did not name --max-memory as a remedy");
  const auto first = StartupRetryWindow::Clock::time_point{};
  require(StartupRetryWindow(30s).retryUntil(before, first) == first + 30s,
          "a start the host refused was not retried");
}

// A Metal command of resource assembly that ran out of GPU memory fails the
// start as CommandMemory, which no start retries, as a warmup's does: the
// Neural Engine split's setup and calibration run theirs before any warmup.
// Another process holds what the commands need, and a retry, in the process
// or the server's relaunch, would take it from that process. A command that
// failed otherwise fails its step as before. The automatic SSD cache's
// volume, read as its files open, stands in for the split's command here.
void testACommandOutOfMemoryInAssemblyIsCommandMemory(const char *metallibPath) {
  using namespace std::chrono_literals;
  using Kind = protocol::StartFailureKind;
  const test::TemporaryDirectory root("splash-command-memory");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "command-memory-test");
  config.cacheDisk = {.automatic = true};
  for (const bool outOfMemory : {true, false}) {
    config.diskVolume = [outOfMemory](const std::filesystem::path &) -> std::optional<DiskVolume> {
      throw metal::MetalBackendError("Metal command 2 failed: Insufficient Memory", outOfMemory);
    };
    const std::string what = outOfMemory ? "for want of GPU memory" : "otherwise";
    try {
      static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
      throw std::runtime_error("resources were assembled past a command that failed " + what);
    } catch (const RuntimeResourcesError &error) {
      const RuntimeBootstrapReport report = RuntimeBootstrapError(error).report();
      const Kind kind = startFailedEvent(report).kind;
      require(report.resourceStage == RuntimeResourceStage::StorageAllocation &&
                  kind == (outOfMemory ? Kind::CommandMemory : Kind::Internal) &&
                  report.message == "Metal command 2 failed: Insufficient Memory",
              "a start whose command failed " + what + " in resource assembly failed as kind " +
                  std::to_string(static_cast<int>(kind)));
      require(!StartupRetryWindow(30s).retryUntil(report, StartupRetryWindow::Clock::time_point{}),
              "a start whose command failed " + what + " in resource assembly was retried");
    }
  }
}

// What a start allocates, the KV runway and then the arenas and the warmup as
// ServingRuntime::start allocates them, is memory a request in service
// needs: neither the warning margin above the spare reserve nor macOS's
// warning refuses it, only what refuses loading, and a claim the engine
// opens meanwhile leaves the start's admission in place. Once serving
// begins, growth that no request in service needs keeps to both again.
void testStartGrowsThroughHostPressure(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-start-pressure");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "start-pressure-test");
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  // A host with its reserve but not the warning margin beyond it, then one
  // with ample room while macOS warns.
  for (const bool warning : {false, true}) {
    const uint64_t available = warning ? 64 * kGiB : reserve + kHostMarginBytes / 2;
    config.hostMemory = [available] {
      return std::optional<HostMemory>(HostMemory{available, available});
    };
    config.memoryPressure = [warning] {
      return warning ? MemoryPressure::Warning : MemoryPressure::Normal;
    };
    const std::string host = warning ? "under macOS's warning" : "short of the warning margin";
    std::unique_ptr<RuntimeResources> resources;
    static_cast<void>(
        test::capturedStderr([&] { resources = RuntimeResources::create(config, 0); }));
    const metal::AllocationAdmission admit = resources->memoryGovernor().allocationAdmission();
    const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
    std::unique_ptr<model::RuntimeModel> runtime;
    require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
              runtime = model::createRuntime(resources->modelContext());
            })),
            "the start's arenas waited " + host);
    {
      const test::ClaimScope request(resources->memoryGovernor(), ClaimClass::Start);
    }
    // The warmup's lane state; it throws if refused.
    {
      const KvPool::RunwayHold runway(resources->kvPool());
      static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
      static_cast<void>(runtime->warmupDecodeBatch(1));
    }
    resources->beginServing();
    const test::ClaimScope growth(resources->memoryGovernor(), ClaimClass::Growth);
    const metal::AllocationResult grown = admit(kMiB, [] {});
    require(!grown && grown.failure == metal::AllocationFailure::HostPressure,
            "growth that no request in service needs went on " + host + " once serving began");
  }
}

// A start checks the host's reserve one last time before Ready, with what
// the engine serves with. Its decode warmups past one lane leave their
// lanes' state idle, which goes back to macOS before Ready, and macOS counts
// it as available only a moment later: a check that still counted it, or
// that did not wait for that moment, would refuse a start the host has room
// for (#572).
void testStartChecksTheReserveWithWhatItServes(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-start-reserve");
  RuntimeBootstrapConfig config;
  config.resources = loadableConfig(metallibPath, root.path(), "start-reserve-test");
  // The process's Metal memory at the last check of the host. Once the start
  // has given back a lane's state, the host refuses its next two checks, as
  // macOS would before it counts what Metal freed.
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  const uint64_t lane = config.resources.model.stateLayout.laneBytes();
  uint64_t checked = 0;
  uint64_t peak = 0;
  int refused = 0;
  config.resources.hostMemory = [&, host = config.resources.hostMemory] {
    checked = device.currentAllocatedSize;
    peak = std::max(peak, checked);
    if (checked + lane <= peak && refused < 2) {
      ++refused;
      return std::optional<HostMemory>(HostMemory{0, 0});
    }
    return host();
  };
  std::unique_ptr<ServingRuntime> bootstrap;
  std::vector<uint8_t> output;
  static_cast<void>(test::capturedStderr([&] {
    bootstrap = ServingRuntime::start(
        config, [&](std::span<const uint8_t> bytes) { output.insert(output.end(), bytes.begin(), bytes.end()); },
        [] { return 0.0; });
    bootstrap->announceReady();
  }));
  require(refused == 2, "the start checked the host before it gave back its warmups' idle state");
  // The warmups leave three lanes' state idle beside the lane the engine
  // keeps. Under shader validation (MTL_SHADER_VALIDATION), less than a lane
  // of it leaves Metal's count only a moment later, which the last check may
  // still see.
  require(checked < device.currentAllocatedSize + 2 * lane,
          "the start's last check of the host counted the lanes' state its warmups left idle");
  // The runtime answers its loop's status requests itself.
  require(bootstrap->nativeLoop().receive(protocol::peer::serialize(protocol::StatusRequestFrame{1})),
          "the loop refused a status request");
  std::string status;
  for (const protocol::EngineEvent &event : protocol::peer::decodeEvents(output))
    if (const auto *answer = std::get_if<protocol::StatusJsonEvent>(&event); answer && answer->correlationId == 1)
      status = answer->json;
  require(status.find("\"memory_limited_steps\":[]") != std::string::npos &&
              status.find("\"memory_audit\":{\"scope\":\"startup_warmup\",\"valid\":true,") != std::string::npos,
          "the start did not warm every decode width, or its loop did not answer with its status: " + status);
}

// A model served with YaRN starts and runs as any other, with YaRN all the
// way into its runtime: the start names its factor and the context it
// serves, the runtime's geometry serves that window, its arenas hold YaRN's
// rotary frequencies, and its prefill and decode build the target's RoPE
// tables with the attention factor.
void testYarnRuntime(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-yarn");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "yarn-test");
  config.model = model::servedFor(std::move(config.model), 1'010'000);
  std::unique_ptr<RuntimeResources> resources;
  const std::string started =
      test::capturedStderr([&] { resources = RuntimeResources::create(config, 0); });
  require(started.find("YaRN is on: the target's RoPE is scaled by 3.85284 for 1010000 tokens of context, past the "
                       "native window of 262144. Static YaRN changes answers at every length, so pass a "
                       "--max-context above 262144 only when a request needs it.") != std::string::npos,
          "a start with YaRN did not name its factor and the context it serves");
  const model::RuntimeContext context = resources->modelContext();
  const auto geometry = model::RuntimeGeometry::from(context.model, kv::Format::Int8, ops::GdnStateFormat::Float32);
  require(geometry.target.ropeScaling == config.model.ropeScaling && geometry.target.contextTokens() == 1'010'000,
          "the runtime's geometry lost YaRN or its window");
  std::array<float, SPLASH_TARGET_ROPE_PAIRS> expected{};
  model::rotaryInverseFrequencies(geometry.target.rotaryTheta, geometry.target.rotaryPairs,
                                  geometry.target.ropeScaling, expected);
  const metal::AllocationAdmission admit = resources->memoryGovernor().allocationAdmission();
  const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
  std::unique_ptr<model::RuntimeModel> runtime;
  require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes, [&] {
            const model::PrefillArena arena(context.backend, geometry, context.operators);
            require(std::memcmp(arena.get(model::PrefillTensor::TargetInverseFrequencies).contents(),
                                expected.data(), sizeof(expected)) == 0,
                    "the prefill arena does not hold YaRN's frequencies");
          })),
          "the YaRN prefill arena was refused");
  require(static_cast<bool>(admit(budget.sharedPrefillBytes + budget.sharedDecodeBytes,
                                  [&] { runtime = model::createRuntime(context); })),
          "the YaRN runtime's arenas were refused");
  const KvPool::RunwayHold runway(resources->kvPool());
  static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
  static_cast<void>(runtime->warmupDecodeBatch(1));
}

// The SSD cache holds no write back unless a write budget is set; one that
// is set starts with a quarter of a day's.
void testWriteBudgetOnlyWhenSet(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-write-budget");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "write-budget-test");
  config.cacheDisk = {.bytes = kGiB};
  const auto tier = [&] {
    KvTierSnapshot snapshot;
    static_cast<void>(test::capturedStderr([&] {
      snapshot = RuntimeResources::create(config, 0)->cache().snapshot().kvTier;
    }));
    return snapshot;
  };
  const KvTierSnapshot unset = tier();
  require(unset.capacityBytes == kGiB && !unset.writeBudgetBytes &&
              unset.writesGated == DiskWritesGate::No,
          "the SSD cache had a write budget nobody set");
  config.cacheDiskWritesPerDay = 8 * kGiB;
  const KvTierSnapshot set = tier();
  require(set.writeBudgetBytes == 2 * kGiB && set.writesGated == DiskWritesGate::No,
          "a write budget that was set did not start with a quarter of a day's");
}

// Every start removes the caches under its root that nobody used for two
// weeks and keeps the others, one without --persistent-cache or the SSD
// cache too.
void testStaleCachesGoAtEveryStart(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-stale-model"), caches("splash-stale");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "stale-test");
  config.cacheRoot = caches.path() / "prefix-cache";
  const std::string stale(CacheDirectory::kNamespaceDigits, 'a'),
      fresh(CacheDirectory::kNamespaceDigits, 'b');
  for (const std::string &name : {stale, fresh}) {
    auto directory = CacheDirectory::open(config.cacheRoot, name, std::chrono::milliseconds(0));
    std::ofstream(directory->kvSlots()) << "cache";
  }
  const timeval old[2] = {{0, 0}, {1000000000, 0}};
  for (const char *file : {"lock", "kv.slots"})
    require(::utimes((config.cacheRoot / stale / file).c_str(), old) == 0,
            "a cache file could not be aged");
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(!std::filesystem::exists(config.cacheRoot / stale) &&
              std::filesystem::exists(config.cacheRoot / fresh / "kv.slots"),
          "a start without a persistent cache kept a stale cache under its root, or removed another");
}

// A cache root the engine makes for a persistent cache stays out of Time
// Machine backups; one that was there keeps its own setting.
void testMadeCacheRootStaysOutOfBackups(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-backups-model"), caches("splash-backups");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "backups-test");
  config.cacheDisk = {.bytes = kGiB};
  config.persistentCache = true;
  const auto excluded = [](const std::filesystem::path &path) {
    NSURL *url = [NSURL fileURLWithPath:@(path.c_str()) isDirectory:YES];
    NSNumber *value = nil;
    return [url getResourceValue:&value forKey:NSURLIsExcludedFromBackupKey error:nil] &&
           value.boolValue;
  };
  config.cacheRoot = caches.path() / "made" / "prefix-cache";
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(std::filesystem::is_directory(config.cacheRoot) && excluded(config.cacheRoot) &&
              !excluded(caches.path() / "made"),
          "the cache root the engine made was not excluded from backups, or its parent was");
  config.cacheRoot = caches.path() / "there";
  std::filesystem::create_directory(config.cacheRoot);
  static_cast<void>(test::capturedStderr([&] { static_cast<void>(RuntimeResources::create(config, 0)); }));
  require(!excluded(config.cacheRoot), "a cache root that was there was excluded from backups");
}

// The automatic tier keeps its staging only where the plan holds the
// requested context beside it. Asked for the context the plan holds without
// the staging but not with it, a start turns the tier off with one line and
// plans as without it.
void testAutomaticTierYieldsToTheContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-automatic");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "automatic-test");
  config.diskVolume = reading(localVolume(500 * kGiB));
  uint64_t minimumRequired = 0;
  const auto start = [&](uint32_t requestedContextTokens, uint32_t &contextTokens,
                         bool &tier) {
    return test::capturedStderr([&] {
      auto resources = RuntimeResources::create(config, requestedContextTokens);
      contextTokens = resources->memoryPlan().maximumContextTokens();
      minimumRequired = resources->memoryPlan().breakdown().minimumRequiredBytes;
      tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
    });
  };
  uint32_t without = 0, with = 0, yielded = 0;
  bool tier = false;
  static_cast<void>(start(0, without, tier));
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV.
  config.maximumMemoryBytes =
      minimumRequired + config.model.stateLayout.cachedBytes() + 64 * kMiB;
  static_cast<void>(start(0, without, tier));
  require(!tier && without, "a start without the SSD cache had one, or no context");
  config.cacheDisk = {.automatic = true};
  static_cast<void>(start(0, with, tier));
  require(tier && with < without,
          "the automatic tier did not start, or its staging cost the plan no context");
  const std::string off = start(without, yielded, tier);
  require(!tier && yielded == without &&
              off.find("SSD cache off: the memory plan cannot hold a " + std::to_string(without) +
                       "-token context beside its state staging") != std::string::npos,
          "the automatic tier kept staging the requested context needs");
}

// The automatic tier's staging is decided by memory alone, so a start serves
// the same context whatever room the volume its files would go to has: a
// relaunch after something else filled the disk announces the context the
// server serves. The volume decides only whether the files open. A volume
// without room, one that cannot be measured or one that is not local leaves
// the tier off with one line and the staging the plan set aside unused; a
// persistent cache whose root has no room keeps temporary files.
void testAutomaticContextIgnoresTheVolume(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-volume"), caches("splash-volume-root");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "volume-test");
  struct Started {
    uint32_t context = 0, planContext = 0;
    uint64_t staging = 0, minimumRequired = 0;
    bool tier = false, persistent = false;
    std::string log;
  };
  const auto start = [&] {
    Started started;
    started.log = test::capturedStderr([&] {
      auto resources = RuntimeResources::create(config, 0);
      started.context = resources->aneFfnOutcome().context;
      started.planContext = resources->memoryPlan().maximumContextTokens();
      started.staging = resources->memoryPlan().breakdown().stateStagingBytes;
      started.minimumRequired = resources->memoryPlan().breakdown().minimumRequiredBytes;
      const CacheSnapshot cache = resources->cache().snapshot();
      started.tier = cache.kvTier.capacityBytes != 0;
      started.persistent = cache.persistent;
    });
    return started;
  };
  const auto offOnce = [](const Started &started, std::string_view reason) {
    const size_t at = started.log.find("SSD cache off: ");
    return at != std::string::npos && started.log.find("SSD cache off", at + 1) == std::string::npos &&
           started.log.find(reason, at) != std::string::npos &&
           started.log.find("; the state staging the plan set aside stays unused.", at) !=
               std::string::npos;
  };
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV.
  config.maximumMemoryBytes =
      start().minimumRequired + config.model.stateLayout.cachedBytes() + 64 * kMiB;
  const Started off = start();
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  const Started room = start();
  require(!off.tier && room.tier && room.staging == config.model.stateLayout.cachedBytes() &&
              room.context == room.planContext && room.context < off.context,
          "the automatic tier did not start beside its staging, or the staging cost no context");
  const std::array<std::pair<std::optional<DiskVolume>, std::string>, 3> refusals{{
      {localVolume(11 * kGiB), "11.0 GiB available on /Volumes/Test is too little beyond the 10.0 GiB"},
      {std::nullopt, "its volume cannot be measured"},
      {DiskVolume{"/Volumes/Share", 500 * kGiB, 1000 * kGiB, false}, "/Volumes/Share is not a local volume"},
  }};
  for (const auto &[volume, reason] : refusals) {
    config.diskVolume = reading(volume);
    const Started refused = start();
    require(!refused.tier && refused.staging == room.staging && refused.context == room.context &&
                refused.planContext == room.planContext && offOnce(refused, reason),
            "a volume without room changed the automatic context, or did not leave the tier off "
            "once (" + reason + ")");
  }
  // A persistent cache's root without room leaves the tier temporary files;
  // with no room for those either, it is off.
  config.persistentCache = true;
  config.cacheRoot = caches.path() / "prefix-cache";
  config.diskVolume = [&](const std::filesystem::path &directory) {
    return std::optional(localVolume(directory == config.cacheRoot ? 11 * kGiB : 500 * kGiB));
  };
  const Started temporary = start();
  require(temporary.tier && !temporary.persistent && temporary.context == room.context &&
              temporary.log.find("Persistent cache off: 11.0 GiB available on /Volumes/Test") !=
                  std::string::npos &&
              temporary.log.find("SSD cache off") == std::string::npos,
          "a persistent cache whose root had no room did not keep temporary files");
  config.diskVolume = reading(localVolume(11 * kGiB));
  const Started neither = start();
  require(!neither.tier && neither.context == room.context &&
              offOnce(neither, "for its root, 11.0 GiB available on /Volumes/Test"),
          "a persistent cache with no room anywhere did not leave the tier off once");
}

// A block-sparse target's context past what memory holds spills to the SSD
// cache, so where the automatic cache's volume has no room for it, a first
// start serves only what memory holds. A relaunch keeps its first engine's
// context whatever room the volume has left since, that of a server started
// with --max-context, which asks for it again, as well: its spilling plan
// opens the cache's temporary files however little room their volume has.
void testRelaunchKeepsTheSpilledContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-relaunch");
  RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path());
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  // A budget whose memory holds less of the window than spilling does.
  config.maximumMemoryBytes = startStreamed(config).minimumRequired + 512 * kMiB;
  const StreamedStart first = startStreamed(config);
  require(startStreamed(config, first.context).context >= first.context,
          "a --max-context of the context a start spills for was not held");
  config.diskVolume = reading(localVolume(11 * kGiB));
  const StreamedStart refused = startStreamed(config);
  require(first.tier && !refused.tier && refused.context < first.context &&
              refused.log.find("SSD cache off: 11.0 GiB available on /Volumes/Test") != std::string::npos,
          "a block-sparse target did not spill past what its memory holds, or spilled to an SSD cache "
          "whose volume has no room for it");
  config.relaunchContext = first.context;
  const StreamedStart relaunch = startStreamed(config);
  require(relaunch.tier && relaunch.context >= first.context,
          "a relaunch served less than its first engine's context for want of SSD room");
  const StreamedStart explicitRelaunch = startStreamed(config, first.context);
  require(explicitRelaunch.tier && explicitRelaunch.context >= first.context,
          "a relaunch for --max-context served less than its first engine's context for want of SSD room");
}

// An automatic expert store keeps every slot as its floor only where the run
// spills, except at its smallest size, where every slot is its floor at any
// share. A byte short of the least budget that loads every expert,
// Flash-Next's experts stream through a store the plan sizes beside a spill,
// all of it its floor. Where the SSD cache's files then cannot be made, the
// run serves without spilling: the weights load once more, and the store
// keeps kExpertFloorShare of its slots, as in a run that cannot spill. A
// budget lower by the experts' bytes above their smallest store gives that
// store, beside a spill too: without the SSD cache's files it is all its
// floor still, and the weights load as often as where the run spills. Each
// start reports the store once, as the run serves it.
void testAStoreKeepsTheFloorOfTheRunsSpill(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-store-floor");
  RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path());
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  // Halving from the least memory a start needs, where the experts stream
  // through their smallest store, to 2 GiB above it, where they load whole.
  uint64_t streams = startStreamed(config).minimumRequired, whole = streams + 2 * kGiB;
  while (streams + 1 < whole) {
    config.maximumMemoryBytes = streams + (whole - streams) / 2;
    (startStreamed(config).experts ? streams : whole) = config.maximumMemoryBytes;
  }
  config.maximumMemoryBytes = streams;
  const StreamedStart spilling = startStreamed(config);
  require(spilling.tier && spilling.experts && spilling.experts->floorBytes == spilling.experts->capBytes &&
              spilling.loads == 2,
          "a store sized beside a spill did not keep all of it as its floor");
  // A start of `at` with the temporary directory a regular file while the
  // runtime starts, where no file of the SSD cache can be made.
  const auto startClosed = [&](const RuntimeResourcesConfig &at) {
    const char *const saved = std::getenv("TMPDIR");
    const std::string temporary = saved ? saved : "";
    const auto restore = [&] { saved ? ::setenv("TMPDIR", temporary.c_str(), 1) : ::unsetenv("TMPDIR"); };
    StreamedStart closed;
    closed.log = test::capturedStderr([&] {
      ::setenv("TMPDIR", (root.path() / "target" / "model.gguf").c_str(), 1);
      try {
        const auto resources = RuntimeResources::create(at, 0);
        closed.experts = resources->streamedExpertStats();
        closed.tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
      } catch (...) {
        restore();
        throw;
      }
      restore();
    });
    closed.loads = occurrences(closed.log, "Weights loaded");
    return closed;
  };
  const StreamedStart closed = startClosed(config);
  require(!closed.tier && closed.experts && closed.experts->capBytes == spilling.experts->capBytes &&
              closed.experts->floorBytes < closed.experts->capBytes && closed.loads == 3 &&
              occurrences(closed.log, "Routed experts stream") == 1 &&
              closed.log.find(std::to_string(static_cast<int>(model::kExpertFloorShare * 100)) +
                              "% of its slots") != std::string::npos,
          "a store sized beside a spill the run could not make kept all of it as its floor");
  const model::StreamableWeights weights =
      model::streamableWeights(config.modelRoot, config.model, {.streamExperts = true});
  config.maximumMemoryBytes = streams - (weights.expertBytes - weights.expertMinimumBytes);
  const StreamedStart least = startStreamed(config), smallest = startClosed(config);
  require(least.tier && least.experts && least.experts->capBytes == weights.expertMinimumBytes,
          "a budget lower by the experts' bytes above their smallest store gave another store");
  require(!smallest.tier && smallest.experts && smallest.experts->capBytes == weights.expertMinimumBytes &&
              smallest.experts->floorBytes == smallest.experts->capBytes && smallest.loads == least.loads &&
              occurrences(smallest.log, "Routed experts stream") == 1 &&
              smallest.log.find("100% of its slots") != std::string::npos,
          "a store at its smallest size, sized beside a spill the run could not make, loaded again");
}

// A relaunch is planned as its first engine was. That of a server started
// with --max-context asks for it again, beside the context the first engine
// served (server/server.py `_native_command`), and on the budget that engine
// had announces that context: the automatic SSD cache turns its staging off
// for it, as it did for the first engine, where a relaunch planned as an
// automatic start keeps the staging and holds less. On a budget that holds
// less, as once this Mac's memory shrank, it serves what its plan holds.
void testAnExplicitRelaunchServesItsFirstContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-explicit-relaunch");
  RuntimeBootstrapConfig config;
  config.resources = loadableConfig(metallibPath, root.path(), "explicit-relaunch-test");
  config.resources.diskVolume = reading(localVolume(500 * kGiB));
  // The context the plan of a start asked for `requestedContextTokens`
  // holds, and the least memory it needs.
  const auto plan = [&](uint32_t requestedContextTokens) {
    std::pair<uint32_t, uint64_t> planned;
    static_cast<void>(test::capturedStderr([&] {
      const auto resources = RuntimeResources::create(config.resources, requestedContextTokens);
      planned = {resources->memoryPlan().maximumContextTokens(),
                 resources->memoryPlan().breakdown().minimumRequiredBytes};
    }));
    return planned;
  };
  // A budget whose KV the staging shortens: the plan's least, the staging
  // and a little KV. Without the SSD cache it holds the context a first
  // engine started with --max-context serves.
  config.resources.maximumMemoryBytes =
      plan(0).second + config.resources.model.stateLayout.cachedBytes() + 64 * kMiB;
  const uint32_t first = plan(0).first;
  config.resources.cacheDisk = {.automatic = true};
  config.resources.relaunchContext = first;
  require(plan(0).first < first && plan(first).first == first,
          "the automatic SSD cache's staging cost a relaunch planned as an automatic start no context, or "
          "one planned for --max-context did not hold it");
  // What the relaunch announces (Ready's maxContextTokens, which the status
  // reports), and its log.
  config.nativeLoop.engine.maxContext = first;
  const auto relaunch = [&] {
    std::unique_ptr<ServingRuntime> bootstrap;
    const std::string log = test::capturedStderr([&] {
      bootstrap = ServingRuntime::start(config, [](std::span<const uint8_t>) {}, [] { return 0.0; });
      bootstrap->announceReady();
    });
    return std::pair{bootstrap->nativeLoop().statusSnapshot().maximumContextTokens, log};
  };
  require(relaunch().first == first, "a relaunch on its first engine's budget served less than its --max-context");
  config.resources.maximumMemoryBytes -= 32 * kMiB;
  const uint32_t held = plan(first).first;
  const auto [served, log] = relaunch();
  require(held < first && served == held &&
              log.find("The memory plan holds " + std::to_string(held) + " tokens of context, less than the " +
                       std::to_string(first) + " the server's first engine served.") != std::string::npos,
          "a relaunch whose plan holds less than its --max-context did not serve what the plan holds");
}

// --ple-memory auto takes at most the rows of every id a 128K-token prompt
// stages beside the row cache's working memory: on the Flash-Next fixture,
// 128K tokens of its PLE heads' ids.
void testTableShareHoldsAPromptsIds(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-table-share");
  const RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path());
  const model::StreamableWeights weights =
      model::streamableWeights(config.modelRoot, config.model, {.streamExperts = true, .streamTable = true});
  const auto &layout = std::get<model::Qwen4ExpLayout>(config.model.target);
  const model::SparseRowStoreOptions rows = model::streamedTableRows(layout);
  require(weights.tableRowBytes &&
              weights.tableShareBytes ==
                  model::SparseRowStore::workingBytes(weights.tableRowBytes, rows.prefetchRows, rows.stageRows) +
                      model::SparseRowStore::cacheBytesFor(uint64_t{128 * 1024} * layout.pleHeads(),
                                                           weights.tableRowBytes) &&
              weights.tableShareBytes > weights.tableMinimumBytes,
          "--ple-memory auto's share is not the rows of every id a 128K-token prompt stages");
}

// The PLE table's row cache is what the engine's limit gives it, whatever
// the host spares as the engine starts: Flash-Next's fixture of a table three
// times its row cache's working memory, under a limit that leaves half of
// the table above its smallest beside the floors and the KV kept resident
// (residentAllowanceBytes), which streams it through a row cache above its
// smallest, starts on a host with no spare room with the roomy host's row
// cache, context and budget of states and KV, nothing held back.
void testTheTableIgnoresTheHostsSpareMemory(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-table-host");
  RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path(), 491'520);
  config.expertMemory = {StreamedSetting::Mode::Off, 0};
  const model::StreamableWeights weights =
      model::streamableWeights(config.modelRoot, config.model, {.streamTable = true});
  const StreamedStart whole = startStreamed(config);
  require(weights.tableMinimumBytes < weights.tableBytes && !whole.table,
          "the fixture's table does not stream above its smallest, or loads whole on a roomy limit");
  config.maximumMemoryBytes = whole.minimumRequired + residentAllowanceBytes(config) -
                              (weights.tableBytes - weights.tableMinimumBytes) / 2;
  const StreamedStart roomy = startStreamed(config);
  require(roomy.table > weights.tableMinimumBytes,
          "a limit that leaves the table half of it above its smallest kept its row cache at its smallest");
  RuntimeResourcesConfig bounded = config;
  const uint64_t spareReserve =
      EngineMemoryPolicy::hostSpareReserveBytes(metal::probeDeviceCapabilities().physicalMemoryBytes);
  bounded.hostMemory = [spareReserve] {
    return std::optional<HostMemory>(HostMemory{64 * kGiB, spareReserve + kHostMarginBytes});
  };
  const StreamedStart held = startStreamed(bounded);
  require(held.table == roomy.table && !held.heldBack &&
              held.context == roomy.context && held.dynamicBudget == roomy.dynamicBudget,
          "a host with no spare room started the table below the row cache the limit gives it");
}

// Two launches of one model on hosts that spare different memory get one
// floor: the store starts at its floor either way, the share of its slots
// the model's plan sets, its cap the limit's, the chunks above its floor
// held in the plan, so the floor, the store's start, the context and the
// budget of states and KV are the roomy host's on a host whose spare memory
// holds nothing past the governor's reserve and margin.
void testTwoLaunchesGetOneFloor(const RuntimeResourcesConfig &config) {
  const StreamedStart roomy = startStreamed(config);
  RuntimeResourcesConfig bounded = config;
  const uint64_t spareReserve =
      EngineMemoryPolicy::hostSpareReserveBytes(metal::probeDeviceCapabilities().physicalMemoryBytes);
  bounded.hostMemory = [spareReserve] {
    return std::optional<HostMemory>(HostMemory{64 * kGiB, spareReserve + kHostMarginBytes});
  };
  const StreamedStart held = startStreamed(bounded);
  require(roomy.experts && held.experts && held.experts->floorBytes == roomy.experts->floorBytes &&
              held.experts->capBytes == roomy.experts->capBytes && held.experts->chunks == roomy.experts->chunks &&
              held.experts->liveBytes == held.experts->floorBytes && held.heldBack == roomy.heldBack &&
              held.context == roomy.context && held.dynamicBudget == roomy.dynamicBudget,
          "two launches on hosts that spare different memory got floors of " +
              std::to_string(roomy.experts ? roomy.experts->floorBytes : 0) + " and " +
              std::to_string(held.experts ? held.experts->floorBytes : 0) + " bytes, or did not start at them");
  require(held.log.find("it starts at its floor and grows toward its cap as memory allows") != std::string::npos &&
              held.log.find(std::to_string(static_cast<int>(model::kExpertFloorShare * 100)) + "% of its slots") !=
                  std::string::npos,
          "the start did not say the store starts at the floor share of its slots");
}

// A start of `resources` as the server makes one, asked for `maxContext`
// tokens (--max-context, zero for the automatic context), whose log it keeps
// in `log`.
std::unique_ptr<ServingRuntime> bootstrapOf(const RuntimeResourcesConfig &resources, std::string &log,
                                            uint32_t maxContext = 0) {
  RuntimeBootstrapConfig config;
  config.resources = resources;
  config.nativeLoop.engine.maxContext = maxContext;
  std::unique_ptr<ServingRuntime> bootstrap;
  log = test::capturedStderr([&] {
    bootstrap = ServingRuntime::start(config, [](std::span<const uint8_t>) {}, [] { return 0.0; });
    bootstrap->announceReady();
  });
  return bootstrap;
}

// The ledger's account named `name` in `memory`, or nullptr.
const MemoryAccountSnapshot *accountOf(const MemoryGovernorSnapshot &memory, std::string_view name) {
  const auto found = std::ranges::find(memory.accounts, name, &MemoryAccountSnapshot::name);
  return found == memory.accounts.end() ? nullptr : &*found;
}

// The ledger's accounts sum to what the governor charges: at Ready a start of
// streamed experts books the KV pages and the lane states as their storage
// counts them and the expert store as its pool reports it, and the fixed
// account holds the rest, so that the accounts' current bytes add up to the
// charged bytes exactly.
void testLedgerAccountsSumToTheBackend(const RuntimeResourcesConfig &config) {
  std::string log;
  const std::unique_ptr<ServingRuntime> bootstrap = bootstrapOf(config, log);
  RuntimeResources &resources = bootstrap->resources();
  const MemoryGovernorSnapshot memory = resources.memoryGovernor().snapshot();
  const MemoryAccountSnapshot *kv = accountOf(memory, "kv"), *states = accountOf(memory, "states"),
                              *experts = accountOf(memory, "experts");
  require(!memory.accounts.empty() && memory.accounts.front().name == "fixed" && kv && states && experts,
          "the ledger has no account of the KV pages, the lane states or the expert store");
  uint64_t booked = 0;
  for (const MemoryAccountSnapshot &account : memory.accounts) booked += account.currentBytes;
  require(booked == memory.chargedBytes && memory.accounts.front().currentBytes > 0,
          "the ledger's accounts did not sum to the charged bytes, " + std::to_string(booked) + " of " +
              std::to_string(memory.chargedBytes));
  const ActualMemoryReport actual = resources.actualMemoryReport(bootstrap->modelRuntime().actualRuntimeMemory());
  require(kv->currentBytes == resources.kvPool().allocatedBytes() && states->currentBytes == actual.stateAllocatedBytes &&
              experts->currentBytes == resources.streamedExpertStats()->liveBytes,
          "an account did not book what its owner holds");
}

// Before Ready the ledger grows the store from its floor into the room the
// host's rule for growth leaves: on a roomy host to its cap, one logged
// decision a chunk, and on a host whose spare memory keeps less than the
// margin beyond the spare reserve not at all. The first control pass after
// Ready gives nothing back. The starts ask for 98,304 tokens of context, KV
// of two extents, of which the engine at Ready holds the runway's one: the
// store's growth does not rest on the memory Metal holds beside the
// backend's buffers.
void testStartupGrowsByTheHostRule(const RuntimeResourcesConfig &config) {
  constexpr uint32_t kContext = 98'304;
  std::string log;
  model::ExpertStoreStats grown;
  {
    const std::unique_ptr<ServingRuntime> roomy = bootstrapOf(config, log, kContext);
    grown = *roomy->resources().streamedExpertStats();
    const MemoryGovernorSnapshot memory = roomy->resources().memoryGovernor().snapshot();
    const MemoryAccountSnapshot *experts = accountOf(memory, "experts");
    size_t logged = 0;
    for (size_t at = log.find("Memory: experts grew"); at != std::string::npos;
         at = log.find("Memory: experts grew", at + 1))
      ++logged;
    require(grown.chunks == grown.capChunks && grown.liveBytes == grown.capBytes &&
                grown.floorBytes < grown.capBytes && experts && experts->grows == grown.regrows && grown.regrows &&
                logged == grown.regrows,
            "a roomy host's start did not grow the store from its floor to its cap by the ledger, one logged "
            "decision a chunk: " + std::to_string(grown.chunks) + " of " + std::to_string(grown.capChunks) +
                " chunks, " + std::to_string(grown.liveBytes) + " of " + std::to_string(grown.capBytes) +
                " bytes, " + std::to_string(experts ? experts->grows : 0) + " grows, " + std::to_string(logged) +
                " logged, " + std::to_string(memory.headroomBytes) + " bytes of headroom");
    static_cast<void>(roomy->controlPass(MemoryPressure::Normal));
    require(roomy->resources().streamedExpertStats()->liveBytes == grown.liveBytes,
            "the first control pass after Ready gave back what the start grew");
  }
  RuntimeResourcesConfig bounded = config;
  const uint64_t spareReserve =
      EngineMemoryPolicy::hostSpareReserveBytes(metal::probeDeviceCapabilities().physicalMemoryBytes);
  bounded.hostMemory = [spareReserve] {
    return std::optional<HostMemory>(HostMemory{64 * kGiB, spareReserve + kHostMarginBytes / 2});
  };
  const std::unique_ptr<ServingRuntime> held = bootstrapOf(bounded, log, kContext);
  const model::ExpertStoreStats floor = *held->resources().streamedExpertStats();
  require(floor.liveBytes == floor.floorBytes && floor.floorBytes == grown.floorBytes && !floor.regrows &&
              log.find("Memory: experts grew") == std::string::npos,
          "a start on a host inside the margin grew the store past its floor");
}

// The runtime of a start, built under the governor's admission of its
// arenas, as Bootstrap builds it.
std::unique_ptr<model::Runtime> runtimeOf(RuntimeResources &resources) {
  const EngineMemoryBreakdown &budget = resources.memoryPlan().breakdown();
  std::unique_ptr<model::Runtime> runtime;
  require(static_cast<bool>(resources.memoryGovernor().allocationAdmission()(
              budget.sharedPrefillBytes + budget.sharedDecodeBytes,
              [&] { runtime = std::make_unique<model::Runtime>(resources.modelContext()); })),
          "a runtime's arenas were refused");
  return runtime;
}

// The bound of one attention layer's pairs a command may compute: a full
// chunk's at the end of the native window of 262,144 tokens of every family
// (Bootstrap.mm's former EngineConfig::maximumPrefillAttentionPairs).
constexpr uint64_t kLayerPairs = 534'774'784;
constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;

// The names of a runtime's pools, in order.
std::vector<std::string> poolNames(model::Runtime &runtime) {
  std::vector<std::string> names;
  for (const model::ElasticPool *pool : runtime.pools()) names.emplace_back(pool->state().name);
  return names;
}

// A prompt of `tokens` tokens prefilled on the runway's pages (KvPool::RunwayHold) as one command in `passes`
// passes of the target's layers, each its own command; its last pass's results.
std::vector<ModelStepResult> prefillInPasses(model::Runtime &runtime, uint64_t id, uint32_t tokens, uint32_t passes) {
  std::vector<uint32_t> prompt(tokens);
  for (uint32_t index = 0; index < tokens; ++index) prompt[index] = 1 + index * 37 % 997;
  ModelRequest request;
  request.id = id;
  request.prompt = prompt;
  request.maxNewTokens = 4;
  runtime.beginColdRequest(request, 0);
  std::vector<uint32_t> pages((tokens + kv::kPageTokens - 1) / kv::kPageTokens);
  std::iota(pages.begin(), pages.end(), 0u);
  ModelBatchItem item{.requestId = id, .logicalPosition = 0, .tokenCount = tokens, .pageTable = pages,
                      .pageTableRevision = 1};
  item.inputTokens = prompt;
  std::vector<ModelStepResult> results;
  try {
    BatchPlan plan{
        .kind = WorkKind::Prefill, .items = {{id, tokens}}, .decodeStage = DecodeStage::Regular, .passes = passes};
    for (; plan.pass < passes; plan = followingPass(plan, runtime.commandLimits().layers)) {
      const uint32_t pass = plan.pass;
      results = runtime.submit(plan, std::span(&item, 1), {})->wait();
      require(results.size() == 1 && results[0].failure.empty() &&
                  (pass + 1 == passes || !results[0].consumedPromptTokens),
              "a pass before a command's last committed its rows, or a pass failed");
    }
  } catch (...) {
    runtime.end(id);
    throw;
  }
  runtime.end(id);
  return results;
}

// A resident dense start's runtime: a command in two passes, which the
// runtime refused while only streamed targets ran passes, leaves what it
// leaves whole, and one in more passes than its layers is refused before
// anything of it runs; its command limits are one chunk's rows in up to a
// pass per layer, one lane past a chunk, no streamed history and the
// per-layer pair bound. What a command computes counts its attention
// layer's pairs over the layers its pass runs, the heaviest layer's whatever
// the pass.
// Loading read its weights from the SSD, and its pools are the vision arena
// and the embedding cache of a model that serves vision, units that cost
// only their rebuild, which a start may take (model::PoolState::rebuildOnly,
// amend/M.md B2 as the owner amended it 11-03; their value once an image's
// start built the arena is the runtime oracle's, whose model's tower the
// vision kernels run).
void testDenseRuntimeBoundary(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-boundary-dense");
  const RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "boundary-test");
  const std::unique_ptr<RuntimeResources> resources = RuntimeResources::create(config, 0);
  require(!resources->kvPool().freePageCount(kv::PageForm::Index), "a start that cannot spill holds an Index runway");
  const std::unique_ptr<model::Runtime> runtime = runtimeOf(*resources);
  const uint32_t layers = test::plannedModel().target.layers;
  const KvPool::RunwayHold runway(resources->kvPool());
  const std::vector<ModelStepResult> once = prefillInPasses(*runtime, 902, 64, 1);
  const std::vector<ModelStepResult> twice = prefillInPasses(*runtime, 903, 64, 2);
  require(once[0].consumedPromptTokens == 64 && twice[0].consumedPromptTokens == 64 && once[0].firstToken &&
              once[0].firstToken == twice[0].firstToken,
          "a resident command in two passes did not leave what it leaves whole");
  rejects([&] { static_cast<void>(prefillInPasses(*runtime, 901, 64, layers + 1)); }, "passes are not ones",
          "a command in more passes than the target's layers ran");
  require(runtime->commandLimits() == model::CommandLimits{.commandRows = kChunk,
                                                           .layers = layers,
                                                           .maximumPasses = layers,
                                                           .lanesPastChunk = 1,
                                                           .indexPageLanes = 0,
                                                           .attentionPairs = kLayerPairs},
          "a resident dense runtime's command limits are not a chunk in a pass a layer, with the per-layer bound");
  const std::array<model::PrefillLane, 1> chunk{{{.position = 0, .rows = kChunk}}};
  const model::CommandWork whole = runtime->prefillWork(chunk, 0, 1);
  const uint64_t pairs = model::denseAttentionPairs(0, kChunk);
  require(whole == model::CommandWork{.attentionPairs = pairs, .layerAttentionPairs = pairs},
          "a chunk's work is not its attention layer's pairs over the target's layers");
  // The planned model's one attention layer is its last.
  require(runtime->prefillWork(chunk, 0, 2) == model::CommandWork{.layerAttentionPairs = pairs} &&
              runtime->prefillWork(chunk, layers / 2, 1).attentionPairs == pairs,
          "a pass's work does not count its own layers' pairs and the heaviest layer's of the target");
  require(poolNames(*runtime) == std::vector<std::string>{"vision_arena", "embeddings"},
          "a model that serves vision does not have its vision arena and embeddings as pools");
  require(runtime->pools()[0]->state().rebuildOnly && runtime->pools()[1]->state().rebuildOnly,
          "the vision arena or the embeddings is not a pool whose units cost only their rebuild");
}

// A streamed MoE start's runtime, its store sized automatically: a prefill
// group's rows in a pass per layer and no streamed history. Its pool is the
// expert store's, whose bytes are Metal's: a unit taken above its floor and
// given back again moves the backend's allocations by its bytes, rounded to
// whole pages.
void testStreamedRuntimeBoundary(RuntimeResourcesConfig moe) {
  moe.maximumMemoryBytes = startStreamed(moe).minimumRequired - 1;
  const std::unique_ptr<RuntimeResources> resources = RuntimeResources::create(moe, 0);
  const std::unique_ptr<model::Runtime> runtime = runtimeOf(*resources);
  require(resources->streamedExpertStats().has_value(), "the MoE start did not stream its experts");
  require(runtime->commandLimits() == model::CommandLimits{.commandRows = 4 * kChunk,
                                                           .layers = kMoeLayers,
                                                           .maximumPasses = kMoeLayers,
                                                           .lanesPastChunk = 1,
                                                           .indexPageLanes = 0,
                                                           .streamsWeights = true,
                                                           .attentionPairs = kLayerPairs},
          "a streamed MoE runtime's command limits are not a group's rows in a pass a layer");
  require(poolNames(*runtime) == std::vector<std::string>{"experts", "vision_arena", "embeddings"},
          "a streamed MoE runtime's pools are not its store's and its vision's");
  require(!runtime->pools()[0]->state().rebuildOnly && runtime->pools()[1]->state().rebuildOnly &&
              runtime->pools()[2]->state().rebuildOnly,
          "the store, which the lanes in service read through, is a pool a start may take, or its vision's are not");
  // The store starts at its floor: a unit it takes, then gives back.
  model::ElasticPool &experts = *runtime->pools()[0];
  metal::MetalBackend &backend = resources->backend();
  const auto allocated = [&] { return backend.refreshMemoryStats().allocatedBytes; };
  uint64_t before = allocated();
  const model::PoolState floor = experts.state();
  require(floor.currentBytes == floor.floorBytes && floor.growUnitBytes &&
              static_cast<bool>(experts.grow(resources->memoryGovernor().allocationAdmission())) &&
              allocated() - before >= floor.growUnitBytes && allocated() - before - floor.growUnitBytes < 16384 &&
              experts.state().currentBytes == floor.currentBytes + floor.growUnitBytes,
          "a unit the store took did not take the backend's allocations");
  before = allocated();
  const model::PoolState state = experts.state();
  const uint64_t released = experts.release(1);
  const uint64_t freed = before - allocated();
  require(released && released == state.releaseUnitBytes && freed >= released && freed - released < 16384 &&
              experts.state().currentBytes == state.currentBytes - released,
          "a unit the store gave back did not leave the backend's allocations");
}

// A Flash-Next start whose K/V spills: its commands hold a prefill group's
// rows, stream one lane's history, and run a pass per layer. Its pools are
// its store's where the plan streams its experts, its hot block cache's and
// its history's staging; of them only the staging, which no lane in service
// reads through, costs only its rebuild (model::PoolState::rebuildOnly), and
// its value is its reservations and the time allocating it took, which the
// next reservation pays once it went. A lane's decode reads and a prompt's staging lack
// what a reservation of them then allocates, and nothing once they hold it;
// before the runtime begins the request, which the engine places first,
// they lack the same. The staging stays a prompt's while it goes on,
// whatever other prompts reserved since. (The fixture runs no command: no
// hyper-connection kernel takes its 256-wide hidden state.)
void testSpilledRuntimeBoundary(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-boundary-spill");
  RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path());
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  // A budget whose memory holds less of the window than spilling does
  // (testRelaunchKeepsTheSpilledContext).
  config.maximumMemoryBytes = startStreamed(config).minimumRequired + 512 * kMiB;
  const std::unique_ptr<RuntimeResources> resources = RuntimeResources::create(config, 0);
  const model::RuntimeContext context = resources->modelContext();
  require(context.blockCache && context.historyStream, "the Flash-Next start did not spill its K/V");
  // An empty Index extent from the start, where a lane's first conversion
  // lands its pages at the memory wall.
  require(resources->kvPool().freePageCount(kv::PageForm::Index) != 0,
          "a start that can spill holds no Index runway");
  const bool streamsExperts = resources->streamedExpertStats().has_value();
  const std::unique_ptr<model::Runtime> runtime = runtimeOf(*resources);
  const uint32_t layers = std::get<model::Qwen4ExpLayout>(config.model.target).layers;
  require(runtime->commandLimits() == model::CommandLimits{.commandRows = 4 * kChunk,
                                                           .layers = layers,
                                                           .maximumPasses = layers,
                                                           .lanesPastChunk = 1,
                                                           .indexPageLanes = 1,
                                                           .streamsWeights = streamsExperts,
                                                           .attentionPairs = kLayerPairs},
          "a spilling Flash-Next runtime's command limits do not stream one lane's history");
  std::vector<std::string> pools{"hot_kv", "kv_staging"};
  if (streamsExperts) pools.insert(pools.begin(), "experts");
  require(poolNames(*runtime) == pools,
          "a spilling Flash-Next runtime's pools are not its store's, hot block cache's and staging's");
  ModelRequest request;
  const std::vector<uint32_t> prompt(2 * kChunk, 7);
  request.id = 911;
  request.prompt = prompt;
  request.maxNewTokens = 4;
  model::ElasticPool &hot = *runtime->pools()[streamsExperts], &staging = *runtime->pools()[streamsExperts + 1];
  require(staging.state().rebuildOnly && hot.state().rebuildOnly &&
              !(streamsExperts && runtime->pools()[0]->state().rebuildOnly),
          "the staging or the hot block cache no lane reads is not a pool whose units cost only their rebuild, "
          "or the store, which the lanes in service read through, is");
  const struct {
    model::SpilledReadNeed need;
    model::ElasticPool &pool;
  } needs[] = {{{.kind = WorkKind::Decode}, hot}, {{.kind = WorkKind::Prefill, .pages = 128}, staging}};
  // Through the model's interface, as the engine places and reserves them.
  model::Model &reads = *runtime;
  const auto shortfallOf = [&](const model::SpilledReadNeed &need) {
    return need.kind == WorkKind::Decode ? reads.hotFloorShortfall(std::span(&request.id, 1))
                                         : reads.stagingShortfall(request.id, need.pages);
  };
  std::array<uint64_t, 2> placed{};
  for (size_t index = 0; index < placed.size(); ++index)
    placed[index] = shortfallOf(needs[index].need);
  runtime->beginColdRequest(request, 0);
  for (size_t index = 0; index < placed.size(); ++index) {
    const auto &[need, pool] = needs[index];
    const uint64_t shortfall = shortfallOf(need);
    const uint64_t before = pool.state().currentBytes;
    require(shortfall && shortfall == placed[index] && reads.reserveSpilledReads(request.id, need).granted() &&
                pool.state().currentBytes - before == shortfall && !shortfallOf(need),
            std::string(need.kind == WorkKind::Decode ? "a lane's decode reads" : "a prompt's staging") +
                " did not allocate the shortfall it named before and after the request began");
  }
  // The staging stays the prompt's while it goes on, as its command in
  // passes reserved it once, before its first pass, whatever other prompts
  // reserved meanwhile: one that reserved it and was suspended, and one that
  // reserved it and ended, leave it, and so do an idle reclaim, a release of
  // the pool and a decode's floor; it goes once none of the requests it was
  // reserved for holds a lane.
  const uint64_t held = staging.state().currentBytes;
  for (const bool suspends : {true, false}) {
    ModelRequest other = request;
    other.id = suspends ? 912 : 913;
    runtime->beginColdRequest(other, 1);
    require(reads.reserveSpilledReads(other.id, {.kind = WorkKind::Prefill, .pages = 64}).granted(),
            "a prompt's staging within what the staging holds was refused");
    if (suspends) runtime->suspend(other.id);
    else runtime->end(other.id);
    require(staging.state().currentBytes == held &&
                !reads.stagingShortfall(request.id, 128),
            std::string("a prompt's staging went with another prompt that reserved it and ") +
                (suspends ? "was suspended" : "ended"));
  }
  require(!staging.release(UINT64_MAX) && reads.reserveSpilledReads(request.id, {.kind = WorkKind::Decode}).granted() &&
              staging.state().currentBytes == held,
          "a release of the pool or a decode's floor took the staging of a prompt that goes on");
  runtime->end(request.id);
  require(!staging.state().currentBytes, "the staging stayed once every request it was reserved for left");
  runtime->end(912);
}

// The engine allocates for its requests' claims in the governor's ledger
// (EngineConfig::governor): inside the warning margin the host refuses a
// start, and a request in service until nothing in place meets its need.
void testEngineFollowsTheGovernor(const char *metallibPath) {
  metal::MetalBackend backend(metallibPath);
  constexpr uint64_t kGiB = 1ULL << 30;
  constexpr uint64_t hostReserve = 2 * kGiB;
  std::optional<uint64_t> available = hostReserve + 8 * kGiB;
  const metal::MetalMemoryStats memory = backend.memoryStats();
  MemoryGovernor governor(
      backend, std::max(memory.allocatedBytes, memory.deviceCurrentAllocatedBytes) + kGiB,
      hostReserve, hostReserve,
      [&available] { return std::optional<HostMemory>(HostMemory{*available, *available}); }, 0);
  EngineConfig config;
  connectToGovernor(config, governor);
  require(config.governor == &governor, "the engine was not connected to the governor");
  const auto admits = [admit = governor.allocationAdmission()] {
    return static_cast<bool>(admit(1024, [] {}));
  };
  available = hostReserve + kGiB / 2;
  {
    const test::ClaimScope start(governor, ClaimClass::Start);
    require(!admits(), "the host's margin admitted a start");
  }
  const test::ClaimScope inService(governor, ClaimClass::InService);
  require(!admits(), "the host's margin admitted a request in service before what it holds");
  require(governor.free(inService.id(), governor.need(inService.id())).outcome == FreeOutcome::Met && admits(),
          "a request in service did not pass the host's margin once nothing in place met its need");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "expected metallib path");
      testLoadedVisionIsRequiredOnlyWithVision();
      testPersistentCacheNamespace();
      testWeightBudgetBeforeLoading(argv[1]);
      testStateStagingNeedsAStartedTier(argv[1]);
      testImagePatchCapIsBounded(argv[1]);
      testHostMemoryProbeIsRequired(argv[1]);
      testModelBeyondBudgetIsRefusedBeforeLoading(argv[1]);
      testStartupAdmissionIgnoresModelSize(argv[1]);
      testFailedStepIsNamed(argv[1]);
      testACommandOutOfMemoryInAssemblyIsCommandMemory(argv[1]);
      testStartGrowsThroughHostPressure(argv[1]);
      testStartChecksTheReserveWithWhatItServes(argv[1]);
      testYarnRuntime(argv[1]);
      testWriteBudgetOnlyWhenSet(argv[1]);
      testStaleCachesGoAtEveryStart(argv[1]);
      testMadeCacheRootStaysOutOfBackups(argv[1]);
      testAutomaticTierYieldsToTheContext(argv[1]);
      testAutomaticContextIgnoresTheVolume(argv[1]);
      testRelaunchKeepsTheSpilledContext(argv[1]);
      testAStoreKeepsTheFloorOfTheRunsSpill(argv[1]);
      testAnExplicitRelaunchServesItsFirstContext(argv[1]);
      testTableShareHoldsAPromptsIds(argv[1]);
      testTheTableIgnoresTheHostsSpareMemory(argv[1]);
      testEngineFollowsTheGovernor(argv[1]);
      testDenseRuntimeBoundary(argv[1]);
      testSpilledRuntimeBoundary(argv[1]);
      {
        const test::TemporaryDirectory root("splash-moe");
        const RuntimeResourcesConfig moe = moeConfig(argv[1], root.path());
        testStreamableWeightsAreWhatLoads(moe);
        testAToolsStoreIsItsSize(moe);
        testAutomaticExpertMemory(moe);
        testVerifyRows(argv[1], moe);
        testStreamedRuntimeBoundary(moe);
        RuntimeResourcesConfig streamed = moe;
        streamed.maximumMemoryBytes = startStreamed(moe).minimumRequired - 1;
        testLedgerAccountsSumToTheBackend(streamed);
        testTwoLaunchesGetOneFloor(streamed);
        testStartupGrowsByTheHostRule(streamed);
      }
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
