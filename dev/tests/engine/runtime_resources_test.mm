#include "GgufFixtures.hpp"
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
#include "model/RuntimeArenas.hpp"
#include "model/SsdReaders.hpp"

#import <Foundation/Foundation.h>

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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace splash;
using namespace splash::engine;

using splash::test::rejects;
using splash::test::require;

// A small model whose weights load and whose geometry the runtime plans:
// four target and two draft layers of the 2048-wide hidden state a compiled
// draft attention takes, the draft reading a GDN layer and the attention
// layer, with Qwen3.8-27B's attention and GDN heads and a two-block vision
// tower.
struct PlannedModel final {
  model::Qwen3_8Layout target;
  model::DFlashDraftLayout draft;
  ops::VisionLayout vision;
};

PlannedModel plannedModel() {
  PlannedModel result;
  model::Qwen3_8Layout &target = result.target;
  target.layers = 4;
  target.hiddenSize = 2048;
  target.vocabularySize = 1024;
  target.intermediateSize = 1024;
  constexpr std::array<uint32_t, 2> draftReads{1, 3};
  result.draft = model::draftLayout(target.hiddenSize, target.vocabularySize, 2, 1024, 10'000'000.0F, draftReads);
  ops::VisionLayout &vision = result.vision;
  vision.depth = 2;
  vision.hiddenSize = 128;
  vision.patchDimension = 1536;
  vision.intermediateSize = 200;
  vision.paddedIntermediateSize = 256;
  vision.mergedHiddenSize = 512;
  vision.outputHiddenSize = target.hiddenSize;
  vision.heads = 2;
  vision.headDimension = 64;
  vision.positionGridSide = 4;
  return result;
}

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

// A model under root whose weights load, which nothing but the host
// refuses: `name` names its descriptor and build.
RuntimeResourcesConfig loadableConfig(const char *metallibPath, const std::filesystem::path &root,
                                      const std::string &name) {
  const PlannedModel planned = plannedModel();
  static_cast<void>(test::writeSyntheticModel(root, planned.target, planned.draft, planned.vision));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor(name, planned.target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = name;
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// A local volume of 1000 GiB with `availableBytes` free, which leaves the
// automatic SSD cache room from 12 GiB on.
DiskVolume localVolume(uint64_t availableBytes) {
  return {"/Volumes/Test", availableBytes, 1000 * kGiB, true};
}

// A volume probe that reads `volume` for every directory.
std::function<std::optional<DiskVolume>(const std::filesystem::path &)>
reading(std::optional<DiskVolume> volume) {
  return [volume](const std::filesystem::path &) { return volume; };
}

// The routed experts of each layer of the MoE model below.
constexpr uint32_t kMoeExperts = 128;
constexpr uint32_t kMoeLayers = 8;

// A MoE model under root whose weights load: the planned model's draft and
// vision tower beside kMoeLayers layers of Qwen3.6-35B-A3B's geometry, each
// with kMoeExperts routed experts, eight a token. Its experts above their
// smallest store, a layer's, take more than the arenas of its prefill groups
// and the first 64K tokens of its KV.
RuntimeResourcesConfig moeConfig(const char *metallibPath, const std::filesystem::path &root) {
  const PlannedModel planned = plannedModel();
  model::Qwen3_6MoeLayout target;
  target.layers = kMoeLayers;
  target.vocabularySize = planned.target.vocabularySize;
  target.experts = kMoeExperts;
  static_cast<void>(test::writeSyntheticModel(root, target, planned.draft, planned.vision));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("moe-test", target, planned.draft, planned.vision,
                                            model::TargetSource::Safetensors, model::VisionSource::Safetensors);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "moe-test";
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// A Qwen3.8-Flash-Next model under root whose weights load, a block-sparse
// target whose KV spills: a GGUF of its family's tensors
// (gguf_fixtures::targetTensors) with its attention, GDN, indexer and PLE
// heads but a hidden state of 256 values, a vocabulary of 256 tokens, eight
// layers, every second one full attention, and four routed experts a layer,
// and a draft of one layer that reads four of them; no vision tower.
RuntimeResourcesConfig flashNextConfig(const char *metallibPath, const std::filesystem::path &root) {
  model::Qwen4ExpLayout target;
  target.layers = 8;
  target.hiddenSize = 256;
  target.vocabularySize = 256;
  target.fullAttentionPeriod = 2;
  target.maskToken = 252;
  target.stopTokens = {250, 251};
  target.pleEosToken = 250;
  target.pleImageToken = 253;
  target.experts = 4;
  target.expertsPerToken = 2;
  target.expertIntermediateSize = 256;
  target.pleLayer = 0;
  gguf_fixtures::TensorTypes types = gguf_fixtures::qwen4ExpTypes();
  for (const char *name : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                           "output_hc_down.weight", "output_hc_up.weight", "ffn_gate_exps.weight",
                           "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight",
                           "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
    types[name] = gguf_fixtures::kQ8_0;
  // The PLE hash, its heads an equal share of the table's rows each.
  std::vector<test::gguf::Key> keys = gguf_fixtures::metadata(target);
  const std::string ple = std::string(model::Qwen4ExpLayout::ggufArchitecture) + ".ple.";
  const uint32_t heads = target.pleHeads(), headRows = gguf_fixtures::kPleTableRows / heads;
  std::vector<int32_t> offsets, rows(heads, int32_t(headRows));
  for (uint32_t head = 0; head < heads; ++head)
    offsets.push_back(int32_t(head * headRows));
  keys.push_back(test::gguf::int32ArrayKey(ple + "layer_multipliers", {1, 31, 961}));
  keys.push_back(test::gguf::int32ArrayKey(ple + "head_offsets", offsets));
  keys.push_back(test::gguf::int32ArrayKey(ple + "head_vocab_sizes", rows));
  keys.push_back(test::gguf::uint32Key(ple + "eos_token_id", target.pleEosToken));
  std::filesystem::create_directories(root / "target");
  test::writeFile(root / "target" / "model.gguf",
                  test::gguf::file(keys, gguf_fixtures::targetTensors(target, types)));
  constexpr std::array<uint32_t, 4> draftReads{1, 3, 5, 7};
  const model::DFlashDraftLayout draft =
      model::draftLayout(target.hiddenSize, target.vocabularySize, 1, 512, 1'000'000.0F, draftReads);
  test::writeSyntheticShard(root / "draft" / "model.safetensors", test::draftTensors(draft));
  RuntimeResourcesConfig config;
  config.metallibPath = metallibPath;
  config.modelRoot = root;
  config.model = model::makeModelDescriptor("flash-next-test", target, draft,
                                            {.outputHiddenSize = target.hiddenSize}, model::TargetSource::Gguf,
                                            model::VisionSource::None);
  config.model.sourceIdentity = "synthetic";
  config.buildId = "flash-next-test";
  config.hostMemory = [] { return std::optional<HostMemory>(HostMemory{64 * kGiB, 64 * kGiB}); };
  return config;
}

// What a start sized: its context, the least memory its plan needs and its
// arenas, its routed experts' store, whether its SSD cache started, its
// backend's memory, its log and how many times it loaded the weights.
struct StreamedStart final {
  uint32_t context = 0;
  uint64_t minimumRequired = 0, arenaBytes = 0;
  std::optional<model::ExpertStoreStats> experts;
  bool tier = false;
  metal::MetalMemoryStats memory;
  std::string log;
  size_t loads = 0;
};

StreamedStart startStreamed(const RuntimeResourcesConfig &config) {
  StreamedStart started;
  started.log = test::capturedStderr([&] {
    auto resources = RuntimeResources::create(config, 0);
    const EngineMemoryBreakdown &budget = resources->memoryPlan().breakdown();
    started.context = resources->memoryPlan().maximumContextTokens();
    started.minimumRequired = budget.minimumRequiredBytes;
    started.arenaBytes = budget.sharedPrefillBytes + budget.sharedDecodeBytes;
    started.experts = resources->streamedExpertStats();
    started.tier = resources->cache().snapshot().kvTier.capacityBytes != 0;
    started.memory = resources->backend().refreshMemoryStats();
  });
  for (size_t at = started.log.find("Weights loaded"); at != std::string::npos;
       at = started.log.find("Weights loaded", at + 1))
    ++started.loads;
  return started;
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
  require(weights.expertBytes > weights.expertFloorBytes && weights.expertFloorBytes && !weights.tableBytes &&
              !weights.tableFloorBytes && !weights.tableRowBytes && weights.tableRefusal.empty(),
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
      backend, root, config.model, {.expertMemoryBytes = weights.expertFloorBytes, .readers = readers});
  require(streamed.expertStore() && streamed.expertStore()->stats().slots == kMoeExperts &&
              allocated(streamed) <= weights.loadedBytes + weights.expertFloorBytes,
          "the smallest store did not hold a layer's experts within its bytes");
  require(arenas(whole, {.experts = true}) == arenas(streamed, model::StreamedParts::loaded(streamed)) &&
              arenas(streamed, {}) == arenas(whole, model::StreamedParts::loaded(whole)) &&
              arenas(whole, {.experts = true}) > arenas(whole, {}),
          "the arenas of what streams are not those of a model loaded so, or prefill groups took no more");
  rejects([&] {
    static_cast<void>(
        model::loadModel(backend, root, config.model,
                         {.expertMemoryBytes = weights.expertFloorBytes - 1, .readers = readers}));
  }, "holds fewer slots", "a store a byte below the smallest loaded");
}

// --expert-memory auto loads a MoE target's routed experts where the plan
// holds them, in one load as main does. Where it does not, they stream
// through a store of all that the first 64K tokens of KV leave, in two
// loads: the first plans the arenas of either, and the second loads that
// store; the backend's peaks, which the memory audit holds to the plan,
// count only the second. The automatic SSD cache gives its state staging up
// where it alone would stream them. Off loads them whatever the budget, a
// budget without room for their arenas stops the start with its breakdown,
// and a size below the smallest store is refused by name before anything
// loads.
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
  // A byte short of the budget that holds every expert, which the plan
  // without the arenas holds: whole, then through the store. The first
  // load's experts do not count toward the peaks.
  config.maximumMemoryBytes = whole.minimumRequired - 1;
  const StreamedStart reloaded = startStreamed(config);
  streams(reloaded,
          "experts that fit beside everything but the arenas did not stream through the store the plan sizes");
  const uint64_t firstLoad = weights.loadedBytes + weights.expertBytes;
  require(reloaded.memory.peakAllocatedBytes < firstLoad && reloaded.memory.devicePeakAllocatedBytes < firstLoad,
          "the backend's peaks counted the load the start discarded");
  // Short of that too: at the floor, then through the store.
  config.maximumMemoryBytes = whole.minimumRequired - whole.arenaBytes - 1;
  streams(startStreamed(config),
          "routed experts memory cannot hold did not stream through the store the plan sizes");

  // A byte short of every expert beside the automatic SSD cache's staging:
  // the cache gives it up, and they load whole in one load.
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  config.maximumMemoryBytes = whole.minimumRequired + config.model.stateLayout.cachedBytes() - 1;
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
  config.expertMemory = {StreamedSetting::Mode::Size, weights.expertFloorBytes};
  const StreamedStart floor = startStreamed(config);
  require(floor.experts && floor.experts->slots == kMoeExperts && floor.loads == 1,
          "an explicit store did not load once at its size");
  config.expertMemory = {StreamedSetting::Mode::Size, weights.expertFloorBytes - 1};
  try {
    static_cast<void>(RuntimeResources::create(config, 0));
    throw std::runtime_error("a store below its floor was accepted");
  } catch (const RuntimeResourcesError &error) {
    require(error.stage() == RuntimeResourceStage::Configuration &&
                std::string_view(error.what()).find("is less than its smallest store") != std::string_view::npos,
            "a store below its floor was not refused by name before loading");
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
  moe.expertMemory = {StreamedSetting::Mode::Size,
                      model::streamableWeights(moe.modelRoot, moe.model, {.streamExperts = true}).expertFloorBytes};
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
      } else {
        require(error.failure() == RuntimeResourceFailure::Other,
                "missing model file was misclassified as allocation pressure");
      }
    }
  }
  config.maximumMemoryBytes = 0;
  requireReachesModelLoader(config, root.path(),
                            "a sufficient weight budget did not reach the "
                            "model loader");
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
// any weight is loaded.
void testModelBeyondBudgetIsRefusedBeforeLoading(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-budget");
  RuntimeResourcesConfig config = budgetConfig(metallibPath, root.path());
  config.model = model::makeModelDescriptor("budget-test", model::Qwen3_8Layout{}, test::qwen3_8Draft(),
                                            test::qwen3_8Vision(), model::TargetSource::Safetensors,
                                            model::VisionSource::Safetensors);
  writeCheckpoints(root.path(), config.model);
  config.maximumMemoryBytes = model::streamableWeights(root.path(), config.model, {}).loadedBytes + kGiB / 2;
  try {
    auto resources = RuntimeResources::create(config, 0);
    throw std::runtime_error("a model that cannot fit unexpectedly loaded");
  } catch (const RuntimeResourcesError &error) {
    require(error.failure() == RuntimeResourceFailure::EngineCapacity &&
                error.stage() == RuntimeResourceStage::MemoryPlanning,
            "a model that cannot fit reached the weight loader");
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

// What a start allocates, the KV runway and then the arenas and the warmup as
// RuntimeBootstrap::start allocates them, is memory a request in service
// needs: neither the warning margin above the spare reserve nor macOS's
// warning refuses it, only what refuses loading, and an allocation the
// engine marks meanwhile leaves the start's mark in place. Once serving
// begins, growth that no request in service needs keeps to both again.
void testStartGrowsThroughHostPressure(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-start-pressure");
  RuntimeResourcesConfig config = loadableConfig(metallibPath, root.path(), "start-pressure-test");
  const uint64_t reserve = EngineMemoryPolicy::hostAvailableReserveBytes(
      metal::probeDeviceCapabilities().physicalMemoryBytes);
  // A host with its reserve but not the warning margin beyond it, then one
  // with ample room while macOS warns.
  for (const bool warning : {false, true}) {
    const uint64_t available = warning ? 64 * kGiB : reserve + kHostWarningMarginBytes / 2;
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
    EngineConfig engine;
    connectToGovernor(engine, resources->memoryGovernor());
    {
      const Serving allocation(engine.serving, true);
    }
    // The warmup's lane state; it throws if refused.
    static_cast<void>(runtime->warmupPrefill(model::ExecutionLimits::prefillTokenBudget));
    static_cast<void>(runtime->warmupDecodeBatch(1));
    resources->beginServing();
    const metal::AllocationResult grown = admit(kMiB, [] {});
    require(!grown && grown.failure == metal::AllocationFailure::HostPressure,
            "growth that no request in service needs went on " + host + " once serving began");
  }
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
// context whatever room the volume has left since: its spilling plan opens
// the cache's temporary files however little room their volume has.
void testRelaunchKeepsTheSpilledContext(const char *metallibPath) {
  const test::TemporaryDirectory root("splash-relaunch");
  RuntimeResourcesConfig config = flashNextConfig(metallibPath, root.path());
  config.cacheDisk = {.automatic = true};
  config.diskVolume = reading(localVolume(500 * kGiB));
  // A budget whose memory holds less of the window than spilling does.
  config.maximumMemoryBytes = startStreamed(config).minimumRequired + 512 * kMiB;
  const StreamedStart first = startStreamed(config);
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
}

// The engine asks the governor whether the host pauses growth, and its
// serving mark lets a request in service grow through that pause.
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
  require(config.growthPaused && config.serving && !config.growthPaused(),
          "the engine was not connected to the governor");
  const auto admits = [admit = governor.allocationAdmission()] {
    return static_cast<bool>(admit(1024, [] {}));
  };
  // Inside the warning margin the host pauses growth that no request in
  // service needs.
  available = hostReserve + kGiB / 2;
  require(config.growthPaused() && !admits(), "the engine did not see the host's pause");
  config.serving(true);
  require(admits(), "the serving mark did not reach the governor");
  config.serving(false);
  require(!admits(), "the serving mark was not cleared");
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
      testStartGrowsThroughHostPressure(argv[1]);
      testYarnRuntime(argv[1]);
      testWriteBudgetOnlyWhenSet(argv[1]);
      testStaleCachesGoAtEveryStart(argv[1]);
      testMadeCacheRootStaysOutOfBackups(argv[1]);
      testAutomaticTierYieldsToTheContext(argv[1]);
      testAutomaticContextIgnoresTheVolume(argv[1]);
      testRelaunchKeepsTheSpilledContext(argv[1]);
      testEngineFollowsTheGovernor(argv[1]);
      {
        const test::TemporaryDirectory root("splash-moe");
        const RuntimeResourcesConfig moe = moeConfig(argv[1], root.path());
        testStreamableWeightsAreWhatLoads(moe);
        testAutomaticExpertMemory(moe);
        testVerifyRows(argv[1], moe);
      }
      std::cout << "runtime resources tests passed\n";
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "runtime resources tests failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
