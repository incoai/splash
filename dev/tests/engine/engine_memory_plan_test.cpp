#include "TestChecks.hpp"
#include "engine/MemoryPlan.hpp"
#include "TestModel.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

DeviceCapabilities device(uint64_t workingSet = 12 * kGiB) {
  DeviceCapabilities result;
  result.deviceName = "test";
  result.appleGpuFamily = 9;
  result.macosMajor = 26;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 16 * kGiB;
  result.recommendedMaxWorkingSetBytes = workingSet;
  result.maxBufferLengthBytes = 8 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  return result;
}

ModelMemoryProfile model() {
  return test::modelMemoryProfile(2 * kGiB, 1 * kGiB, 1 * kGiB);
}

// The pages the budget holds for one request's KV beside one lane's state.
uint64_t budgetPages(const EngineMemoryBreakdown &budget) {
  return (budget.dynamicBudgetBytes - budget.laneStateBytes) / budget.kvPageBytes;
}

void testUnifiedElasticBudget() {
  EngineMemoryPlan plan = test::requireMemoryPlan(device(), model());
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.maximumBatchWidth == 4 &&
              budget.kvExtentPages == 128,
          "execution geometry did not reach memory planning");
  require(budget.fixedRuntimeBytes == model().fixedRuntimeBytes(),
          "active state was incorrectly precharged as fixed memory");
  require(budget.dynamicBudgetBytes ==
              budget.hardBudgetBytes - budget.fixedRuntimeBytes,
          "state and KV do not share one dynamic budget");
  require(budget.minimumDynamicBytes ==
                  budget.laneStateBytes + budget.kvExtentBytes &&
              budget.minimumRequiredBytes ==
                  budget.fixedRuntimeBytes + budget.minimumDynamicBytes,
          "minimum B1 plus the KV runway is incorrect");
  require(budget.kvCapacityPages % 128 == 0 && budget.kvCapacityPages >= 128 &&
              budget.kvCapacityPages == budgetPages(budget) - budgetPages(budget) % 128 &&
              budget.kvCapacityBytes ==
                  uint64_t{budget.kvCapacityPages} * budget.kvPageBytes &&
              budget.kvCapacityTokens == uint64_t{budget.kvCapacityPages} * 32,
          "one request's KV capacity is not the budget's whole extents");
  require(plan.maximumContextTokens() ==
              std::min<uint64_t>(model().maximumContextTokens,
                                 budget.kvCapacityTokens -
                                     model::ExecutionLimits::speculativeScratchTokens),
          "advertised context exceeds elastic KV capacity");
  const std::string json = plan.toStatusJson();
  require(json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
              json.find("\"kv_extent_pages\":128") != std::string::npos,
          "elastic state/KV budget is missing from memory status");
  require(json.find("\"working_set_margin_bytes\":" +
                    std::to_string(budget.workingSetMarginBytes)) !=
                  std::string::npos &&
              json.find("\"headroom_bytes\"") == std::string::npos,
          "memory status did not name the working-set margin");
}

void testBf16BudgetAndStatus() {
  auto profile = model();
  profile.targetKvLayout.format = kv::Format::BFloat16;
  const auto bf16 = test::requireMemoryPlan(device(), profile);
  const auto int8 = test::requireMemoryPlan(device(), model());
  const auto &budget = bf16.breakdown();
  require(budget.kvPageBytes == profile.targetKvLayout.bytesPerModelPage() &&
              budget.kvPageBytes > int8.breakdown().kvPageBytes &&
              budget.kvExtentPages ==
                  profile.targetKvLayout.extentPagesFor(budgetPages(budget)) &&
              budget.kvCapacityPages % budget.kvExtentPages == 0,
          "BF16 planning did not use its payload size and its pool's extent size");
  require(budget.kvCapacityBytes <= budget.dynamicBudgetBytes &&
              budget.kvCapacityPages < int8.breakdown().kvCapacityPages,
          "BF16 KV capacity exceeded the shared budget");
  const auto json = bf16.toStatusJson();
  require(json.find("\"kv_format\":\"bf16\"") != std::string::npos &&
              json.find("\"kv_scale_value_bytes\"") == std::string::npos &&
              json.find("\"kv_quantization_bits\"") == std::string::npos,
          "BF16 memory status restated its format");
  const uint64_t minimum = budget.minimumRequiredBytes;
  require(!evaluateEngineMemoryPlan(device(), profile, minimum - 1).plan,
          "BF16 startup admitted less than its minimum footprint");
}

// The memory status names the recurrent state's format, and a profile without
// one is refused.
void testStateFormatStatus() {
  auto profile = model();
  profile.stateFormat = ops::GdnStateFormat::BFloat16;
  require(test::requireMemoryPlan(device(), profile).toStatusJson().find("\"state_format\":\"bf16\"") !=
                  std::string::npos &&
              test::requireMemoryPlan(device(), model()).toStatusJson().find("\"state_format\":\"fp32\"") !=
                  std::string::npos,
          "the memory status did not name the state format");
  profile.stateFormat = {};
  require(!evaluateEngineMemoryPlan(device(), profile, 0).plan, "a profile without a state format was planned");
}

// The minimum holds the KV runway, the whole smallest extents that hold the
// pages startup warmup runs on: two of the 27B's 32-page BF16 extents.
void testMinimumHoldsTheWarmupRunway() {
  auto profile = model();
  profile.targetKvLayout.format = kv::Format::BFloat16;
  require(profile.targetKvLayout.minimumExtentPages() == 32,
          "the BF16 fixture's smallest extent changed");
  const EngineMemoryPlan plan = test::requireMemoryPlan(device(), profile);
  const auto &budget = plan.breakdown();
  require(budget.minimumDynamicBytes ==
              budget.laneStateBytes + 64 * budget.kvPageBytes,
          "the minimum does not hold the warmup runway");
  const auto refused =
      evaluateEngineMemoryPlan(device(), profile, budget.minimumRequiredBytes - 1);
  require(!refused.plan && refused.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "a budget short of the warmup runway was accepted");
}

// The pre-load fit check and the plan share one minimum: the fixed bytes,
// one lane's state and the KV runway.
void testMinimumRequiredBytesIsThePlans() {
  for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
    ModelMemoryProfile profile = model();
    profile.targetKvLayout.format = format;
    const EngineMemoryPlan plan = test::requireMemoryPlan(device(), profile);
    const EngineMemoryBreakdown &budget = plan.breakdown();
    require(minimumRequiredBytes(budget.fixedRuntimeBytes, budget.laneStateBytes,
                                 profile.targetKvLayout) == budget.minimumRequiredBytes,
            "the minimum differs from the plan's");
  }
  require(!minimumRequiredBytes(std::numeric_limits<uint64_t>::max(), 1,
                                model().targetKvLayout),
          "an overflowing minimum was not refused");
}

void testUserCeilingAndFailure() {
  EngineMemoryPlan automatic = test::requireMemoryPlan(device(), model());
  const uint64_t ceiling =
      automatic.breakdown().minimumRequiredBytes + 64 * kMiB;
  EngineMemoryPlan limited =
      test::requireMemoryPlan(device(), model(), ceiling);
  require(limited.breakdown().hardBudgetBytes == ceiling,
          "explicit memory ceiling was ignored");
  require(test::requireMemoryPlan(device(), model(), 16 * kGiB)
                  .breakdown().hardBudgetBytes ==
              automatic.breakdown().hardBudgetBytes,
          "explicit memory ceiling overrode the safe working set");

  auto failed = evaluateEngineMemoryPlan(
      device(), model(), automatic.breakdown().minimumRequiredBytes - 1);
  require(!failed.plan &&
              failed.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "budget smaller than B1 plus one extent was accepted");
}

// A state's write to the disk tier stages through one state-sized buffer the
// plan sets aside beside the weights, so one lone request still reaches the
// advertised context.
void testDiskTierStateStagingIsBudgeted() {
  const EngineMemoryPlan without = test::requireMemoryPlan(device(), model());
  ModelMemoryProfile tiered = model();
  // One Qwen3.8-27B state (DEVELOPMENT.md, SSD cache).
  const uint64_t staging = 187 * kMiB;
  tiered.footprint.stateStagingBytes = staging;
  const EngineMemoryPlan with = test::requireMemoryPlan(device(), tiered);
  const auto &budget = with.breakdown();
  require(without.breakdown().fixedRuntimeBytes + staging +
                  budget.laneStateBytes + budget.kvCapacityBytes <=
              budget.hardBudgetBytes,
          "the advertised context cannot be allocated beside the state staging buffer");
  require(with.maximumContextTokens() < without.maximumContextTokens(),
          "a budget-limited context did not shrink by the state staging buffer");
  require(budget.stateStagingBytes == staging &&
              budget.fixedRuntimeBytes ==
                  without.breakdown().fixedRuntimeBytes + staging,
          "state staging was not planned as fixed runtime memory");
  // A budget that fits everything but the staging buffer is refused by the
  // plan, not by a warmup allocation.
  const auto tight = evaluateEngineMemoryPlan(
      device(), tiered, without.breakdown().minimumRequiredBytes);
  require(!tight.plan &&
              tight.status.code == BudgetErrorCode::KvPoolDoesNotFit,
          "a budget without room for the state staging buffer was accepted");
  const std::string field = "\"state_staging_bytes\":" + std::to_string(staging);
  require(with.toStatusJson().find(field + ",\"fixed_runtime_bytes\"") !=
                  std::string::npos &&
              with.toStatusJson().find(field + "}}") != std::string::npos &&
              budget.describe().find("SSD cache state staging: " +
                                     std::to_string(staging)) != std::string::npos,
          "state staging is missing from the memory plan status");
  require(without.breakdown().stateStagingBytes == 0 &&
              without.toStatusJson().find("\"state_staging_bytes\":0,") !=
                  std::string::npos,
          "a plan without the disk tier reported state staging");
}

// The prefill FFN's Neural Engine split is fixed runtime memory of its own
// category, which the KV cache gives up, so the context it leaves is the
// plan's to state; the status and description report it.
void testNeuralEngineSplitIsBudgeted() {
  const EngineMemoryPlan without = test::requireMemoryPlan(device(), model());
  ModelMemoryProfile split = model();
  // Qwen3.8-27B's split at the M6's share (DEVELOPMENT.md, Neural Engine prefill).
  const uint64_t surfaces = 508 * kMiB;
  split.footprint.aneFfnBytes = surfaces;
  const EngineMemoryPlan with = test::requireMemoryPlan(device(), split);
  const auto &budget = with.breakdown();
  require(budget.aneFfnBytes == surfaces &&
              budget.fixedRuntimeBytes == without.breakdown().fixedRuntimeBytes + surfaces &&
              with.maximumContextTokens() < without.maximumContextTokens(),
          "the split was not planned as fixed runtime memory out of the KV cache");
  require(with.toStatusJson().find("\"ane_ffn_bytes\":" + std::to_string(surfaces) +
                                   ",\"state_staging_bytes\"") != std::string::npos &&
              budget.describe().find("Neural Engine split: " + std::to_string(surfaces)) !=
                  std::string::npos &&
              without.toStatusJson().find("\"ane_ffn_bytes\":0,") != std::string::npos,
          "the split is missing from the memory plan status");
}

// A streamed table's host memory (--ple-memory) is fixed runtime memory
// of its own category, which the KV cache gives up; the status and the
// description report it.
void testStreamedTableIsBudgeted() {
  const EngineMemoryPlan without = test::requireMemoryPlan(device(), model());
  ModelMemoryProfile streamed = model();
  const uint64_t cache = 2 * kGiB;
  streamed.footprint.streamedTableBytes = cache;
  const EngineMemoryPlan with = test::requireMemoryPlan(device(), streamed);
  const auto &budget = with.breakdown();
  require(budget.streamedTableBytes == cache &&
              budget.fixedRuntimeBytes == without.breakdown().fixedRuntimeBytes + cache &&
              with.maximumContextTokens() <= without.maximumContextTokens(),
          "a streamed table's cache was not planned as fixed runtime memory");
  require(with.toStatusJson().find("\"streamed_table_bytes\":" + std::to_string(cache) + ",\"ane_ffn_bytes\"") !=
                  std::string::npos &&
              budget.describe().find("streamed table cache: " + std::to_string(cache)) != std::string::npos,
          "a streamed table's cache is missing from the memory plan status");
}

// The pipeline and runtime reserves are the model constants rather than part
// of a model's plan, and a model's plan without one of its arenas is refused.
void testReservesAreTheModelConstants() {
  const EngineMemoryPlan plan = test::requireMemoryPlan(device(), model());
  require(plan.breakdown().pipelineReserveBytes == model::kPipelineReserveBytes &&
              plan.breakdown().runtimeOverheadReserveBytes ==
                  model::kRuntimeOverheadReserveBytes,
          "the plan's reserves are not the model constants");
  ModelMemoryProfile withoutDecode = model();
  withoutDecode.footprint.runtime.sharedDecodePlannedAllocatedBytes = 0;
  const EngineMemoryPlanResult refused =
      evaluateEngineMemoryPlan(device(), withoutDecode, 0);
  require(!refused.plan &&
              refused.status.code == BudgetErrorCode::InvalidModelSpec &&
              refused.status.message == "shared_decode_bytes_required",
          "a model plan without its decode arena was accepted");
}

void testHardBudgetBoundaries() {
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 0) == 11 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 8 * kGiB) ==
                  8 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 16 * kGiB) ==
                  11 * kGiB,
          "preflight ceiling disagrees with automatic or explicit policy");
  require(EngineMemoryPolicy::hardBudgetBytes(0, 0) == 0 &&
              EngineMemoryPolicy::hardBudgetBytes(kGiB, 1) == 0,
          "insufficient working set underflowed the preflight ceiling");
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(EngineMemoryPolicy::hardBudgetBytes(maximum, maximum) ==
              maximum - EngineMemoryPolicy::workingSetMarginBytes(maximum),
          "maximum working set overflowed the preflight ceiling");
}

void testModelProvidedKvGeometry() {
  ModelMemoryProfile compact = model();
  compact.name = "compact-test-model";
  compact.targetKvLayout = {10, 2, 256};
  EngineMemoryPlan plan = test::requireMemoryPlan(device(), compact);
  const auto &budget = plan.breakdown();
  require(budget.kvPageTokens == 32 && budget.kvPageBytes == 332'800 &&
              budget.kvExtentPages ==
                  compact.targetKvLayout.extentPagesFor(budgetPages(budget)) &&
              budget.kvExtentPages % 256 == 0,
          "memory plan ignored model-provided Q8 geometry");
  require(plan.toStatusJson().find("\"attention_layers\":10") !=
              std::string::npos &&
              plan.toStatusJson().find("\"kv_heads\":2") !=
                  std::string::npos,
          "model-provided Q8 geometry is missing from status");
}

// The pool's extents leave the fewest of the budget's pages unused, the size
// nearest the 128 MiB target on a tie, and a budget below the smallest
// extent holds no pool.
void testExtentSizeFollowsThePool() {
  ModelMemoryProfile compact = model();
  compact.targetKvLayout = {10, 2, 256};
  const EngineMemoryBreakdown reference =
      test::requireMemoryPlan(device(), compact).breakdown();
  const auto planFor = [&](uint64_t pages) {
    return evaluateEngineMemoryPlan(
        device(), compact,
        reference.fixedRuntimeBytes + reference.laneStateBytes +
            pages * reference.kvPageBytes);
  };
  for (const auto [pages, extent] :
       std::array<std::array<uint64_t, 2>, 4>{{{10'240, 512}, {10'496, 256},
                                              {10'751, 256}, {511, 256}}}) {
    const auto result = planFor(pages);
    require(result.plan && result.plan->breakdown().kvExtentPages == extent &&
                result.plan->breakdown().kvCapacityPages == pages - pages % extent &&
                result.plan->breakdown().kvExtentBytes == extent * 332'800,
            "the extent size does not leave the fewest pool pages over");
  }
  const auto tooSmall = planFor(255);
  require(!tooSmall.plan && tooSmall.status.code == BudgetErrorCode::KvPoolDoesNotFit &&
              tooSmall.status.breakdown.minimumDynamicBytes ==
                  reference.laneStateBytes + 256 * 332'800,
          "a budget below the smallest extent was accepted");
}

// The Macs the planner is pinned on: their memory and working sets as Metal
// reports them, and the room for active KV of an SSD cache on volumes with
// 16 GiB free (the M6's) and with plenty (the automatic quota's 32 GiB).
struct Mac final {
  const char *name;
  uint64_t memory;
  uint64_t workingSet;
  uint64_t activeLimit;
};
constexpr Mac kM6{"M6 24 GB", 24 * kGiB, 19'069'665'280, 6 * kGiB};
constexpr Mac kStudio{"M5 Max 36 GB", 36 * kGiB, 30'150'672'384, 32 * kGiB};
constexpr Mac kM5Pro{"M5 Pro 64 GB", 64 * kGiB, 55'662'788'608, 32 * kGiB};
constexpr Mac kMacs[] = {kM6, kStudio, kM5Pro};

uint64_t hardBudget(const Mac &mac) { return EngineMemoryPolicy::hardBudgetBytes(mac.workingSet, 0); }

DeviceCapabilities deviceOf(const Mac &mac) {
  DeviceCapabilities result = device();
  result.physicalMemoryBytes = mac.memory;
  result.recommendedMaxWorkingSetBytes = mac.workingSet;
  return result;
}

uint64_t pagesOf(uint64_t tokens) { return (tokens + 31) / 32; }

uint64_t arenaBytes(const model::ModelMemoryPlan &plan) {
  return plan.sharedPrefillPlannedAllocatedBytes + plan.sharedDecodePlannedAllocatedBytes;
}

// The fewest pages of KV budget whose capacity holds `pages` pages
// (kvCapacityPagesFor).
uint64_t budgetPagesHolding(const kv::Layout &layout, uint64_t pages) {
  uint64_t budget = pages;
  while (kvCapacityPagesFor(layout, budget) < pages) ++budget;
  return budget;
}

// A target as it loads: its images with the streamable parts out, its
// draft and vision, its arenas with one prefill chunk and a group's, its
// lane state, a state's staging, its KV and the parts that can stream.
struct Target final {
  const char *name;
  uint64_t loadedBytes, draftBytes, visionBytes;
  model::ModelMemoryPlan chunk, group;
  uint64_t stagingBytes;
  kv::Layout layout;
  uint32_t window;
  StreamedPart experts, table;
  std::optional<model::SparseKvShape> sparseKv;

  // What main loads: everything, its arenas of one chunk; or the parts
  // through stores of expertBytes and tableBytes, prefill groups where the
  // experts stream.
  [[nodiscard]] ModelMemoryProfile profile(uint64_t expertBytes = 0, uint64_t tableBytes = 0,
                                           const KvSpillPlan &spill = {}) const {
    return {name,
            window,
            layout,
            {loadedBytes + (expertBytes ? expertBytes : experts.allBytes) + (tableBytes ? 0 : table.allBytes),
             draftBytes, visionBytes, expertBytes ? group : chunk, stagingBytes, 0, tableBytes},
            ops::GdnStateFormat::Float32,
            spill};
  }
  // The planner's inputs on `mac`: the arenas of one chunk, and of a group
  // where the experts stream, whether the table streams or not.
  [[nodiscard]] StreamedMemoryInputs inputs(const Mac &mac) const {
    return {.hardBudgetBytes = hardBudget(mac),
            .fixedBytes = loadedBytes + draftBytes + visionBytes + model::kPipelineReserveBytes +
                          model::kRuntimeOverheadReserveBytes + stagingBytes,
            .arenaBytes = {{{arenaBytes(chunk), arenaBytes(chunk)}, {arenaBytes(group), arenaBytes(group)}}},
            .laneStateBytes = chunk.laneStatePlannedAllocatedBytes,
            .kvLayout = layout,
            .windowTokens = window,
            .experts = experts,
            .table = table,
            .sparseKv = sparseKv,
            .tierWritable = true,
            .kvSlotBytes = (layout.bytesPerModelPage() + 16'383) / 16'384 * 16'384,
            .stateSlotBytes = (stagingBytes + 16'383) / 16'384 * 16'384,
            .activeLimitBytes = mac.activeLimit,
            .groupPages = 4 * 2048 / kv::kPageTokens};
  }
};

// Qwen3.6-35B-A3B UD-Q2_K_XL as the M6 loads it (/status): its routed
// experts are 10.5 GB of its 12.3 GB, which load with their layers.
const Target k35BQ2{"Qwen3.6-35B-A3B",
                    1'761'600'784,
                    480'165'888,
                    901'939'200,
                    {181'469'184, 629'981'184, 194'560'128},
                    {181'469'184, 1'027'620'864, 194'715'776},
                    64 * kMiB,
                    kv::Layout{10, 2, 256},
                    262'144,
                    {.allBytes = 10'530'069'232, .floorBytes = 1'280 * kMiB},
                    {},
                    std::nullopt};
// UD-Q4_K_M: a 22.1 GB GGUF the Studio and the M5 Pro serve.
const Target k35BQ4 = [] {
  Target target = k35BQ2;
  target.loadedBytes = 2'960'000'000;
  target.experts.allBytes = 19'176'000'000;
  target.experts.floorBytes = 2 * kGiB;
  return target;
}();
// Qwen3.8-Flash-Next as /status reports it with --expert-memory 12G and
// --ple-memory 2G: 3.1 GB of images beside 34 GB of routed experts and a PLE
// table of 28.8 GB, the YaRN window, int8 KV that can spill.
const Target kFlashNext{"Qwen3.8-Flash-Next",
                        3'076'735'760,
                        523'436'032,
                        0,
                        {276'496'384, 1'349'959'680, 287'859'072},
                        {276'496'384, 1'984'724'992, 287'760'768},
                        276'496'384,
                        test::sparseKvLayout(),
                        1'010'000,
                        {.allBytes = 33'976'681'712, .floorBytes = 2'560 * kMiB},
                        {.allBytes = 28'799'139'840, .floorBytes = 16 * kMiB},
                        test::sparseKvShape()};

StreamedMemoryPlan requirePlan(const StreamedMemoryInputs &inputs) {
  StreamedMemoryResult result = planStreamedMemory(inputs);
  if (!result.plan) throw std::runtime_error("the planner refused: " + result.error);
  return *result.plan;
}

// One definition of the hot block cache: its floor holds requiredSlots(1)
// slots a layer, (1 + 1) x 8 rows x 512 blocks, in whole extents: four of
// Flash-Next's 256-page extents of K/V without the index, 390 MiB.
void testHotBlockCacheIsDefinedOnce() {
  const model::SparseKvShape shape = test::sparseKvShape();
  require(shape.laneSlots() == 4'096 && shape.requiredSlots(0) == 0 && shape.requiredSlots(1) == 8'192 &&
              shape.requiredSlots(4) == 20'480,
          "the hot block cache's floor is not (lanes + 1) x laneSlots");
  const KvSpillPlan spill = requirePlan(kFlashNext.inputs(kM5Pro)).spill;
  const kv::Layout layout = test::sparseKvLayout();
  require(spill.enabled && spill.laneSlots == 4'096 && spill.hotFloorBytes == 4 * 256 * layout.kvBytesPerModelPage() &&
              spill.hotFloorBytes == 408'944'640 && spill.hotTargetBytes == spill.hotFloorBytes,
          "the hot floor is not requiredSlots(1) in whole extents");
  require(spill.indexRunwayBytes == 256 * layout.indexBytesPerModelPage() && spill.densePages == 64 &&
              spill.windowPages == 64 && spill.groupPages == 256,
          "the spill floors' pages are not the shape's");
}

// The priorities on the three Macs: the floors, then Flash-Next's experts
// take all that the KV allowance (the first 64K tokens, resident) and the
// PLE table's 512 MiB leave, all of them on the M5 Pro; the table its 512 MiB;
// the hot cache stays at its floor. The context spilling serves is the
// window where memory and the SSD cache's room hold it, the room's on the M6.
void testFlashNextOnTheThreeMacs() {
  const uint64_t page = test::sparseKvLayout().bytesPerModelPage();
  for (const Mac &mac : kMacs) {
    const StreamedMemoryInputs inputs = kFlashNext.inputs(mac);
    const StreamedMemoryPlan plan = requirePlan(inputs);
    const KvSpillPlan &spill = plan.spill;
    require(spill.enabled && plan.tableBytes == 512 * kMiB, std::string("Flash-Next's table on the ") + mac.name +
                                                                " does not stream through its 512 MiB");
    // The floors, the arenas, the stores and the KV of the first 64K tokens
    // and their speculative rows (2,049 pages), in the fewest pages whose
    // extents hold them as one request's capacity, the runway among them,
    // fill the budget, or leave the KV more where the experts load whole.
    const uint64_t charged = inputs.fixedBytes + inputs.arenaBytes[plan.expertBytes != 0][1] + inputs.laneStateBytes +
                             spill.floorBytes +
                             (plan.expertBytes ? plan.expertBytes : kFlashNext.experts.allBytes) + plan.tableBytes +
                             budgetPagesHolding(test::sparseKvLayout(), 2'049) * page;
    if (mac.workingSet == kM5Pro.workingSet)
      require(!plan.expertBytes && charged < hardBudget(mac), "the M5 Pro does not load Flash-Next's experts whole");
    else
      require(plan.expertBytes > kFlashNext.experts.floorBytes && charged == hardBudget(mac),
              std::string("Flash-Next's experts on the ") + mac.name + " do not take what the floors leave");
    const uint32_t disk = static_cast<uint32_t>(
        std::min<uint64_t>(1'010'000, (mac.activeLimit - 4 * inputs.stateSlotBytes) / inputs.kvSlotBytes * 32));
    require(spill.memoryTokens == 1'010'000 && spill.diskTokens == disk &&
                spill.floorTokens == std::min<uint32_t>(1'010'000, disk),
            std::string("Flash-Next's context bounds on the ") + mac.name + " are wrong");
    const uint64_t stagedPages = (pagesOf(spill.floorTokens + 7) + 255) / 256 * 256;
    require(spill.stagingBuffers == 2 &&
                spill.stagingBytes == 2 * stagedPages * test::sparseKvLayout().bytesPerLayerPage(),
            std::string("Flash-Next on the ") + mac.name + " does not stage in two buffers");
  }
  require(requirePlan(kFlashNext.inputs(kM6)).expertBytes <
              requirePlan(kFlashNext.inputs(kStudio)).expertBytes,
          "a larger Mac did not stream more experts");
}

// Main's plan for the 35B on the three Macs: every routed expert loads with
// its layer, and the KV holds main's context, pinned: the M6 229,369 tokens
// beside the whole Q2_K_XL, the others the 256K window of UD-Q4_K_M.
void testDense35BKeepsMainsPlan() {
  const std::tuple<Mac, Target, uint32_t, uint64_t> pinned[] = {
      {kM6, k35BQ2, 229'369, 229'376}, {kStudio, k35BQ4, 262'144, 352'256}, {kM5Pro, k35BQ4, 262'144, 2'801'664}};
  for (const auto &[mac, target, context, capacityTokens] : pinned) {
    const StreamedMemoryPlan plan = requirePlan(target.inputs(mac));
    require(!plan.expertBytes && !plan.tableBytes && !plan.spill.enabled,
            std::string("the 35B streams its experts on the ") + mac.name + " where main loads them");
    const EngineMemoryPlan mains = test::requireMemoryPlan(deviceOf(mac), target.profile());
    require(mains.maximumContextTokens() == context && mains.breakdown().kvCapacityTokens == capacityTokens &&
                mains.breakdown().contextBounds.bound() != std::string_view("disk") &&
                mains.toStatusJson().find("\"kv_spill\":null") != std::string::npos,
            std::string("the 35B's plan on the ") + mac.name + " is not main's");
  }
}

// An explicit context loads every expert of the 35B where main holds it so
// on the M6, beside the SSD cache's state staging up to main's 229,369
// tokens, and without it up to main's context then (the automatic SSD cache
// gives its staging up for them, RuntimeResources::create); a token more
// streams them. 225,000 tokens load them beside the staging, and 233,000
// without it.
void testDenseExplicitContextKeepsMainsExperts() {
  ModelMemoryProfile withoutStaging = k35BQ2.profile();
  withoutStaging.footprint.stateStagingBytes = 0;
  const uint32_t beside = test::requireMemoryPlan(deviceOf(kM6), k35BQ2.profile()).maximumContextTokens(),
                 alone = test::requireMemoryPlan(deviceOf(kM6), withoutStaging).maximumContextTokens();
  require(beside == 229'369 && beside < 233'000 && alone >= 233'000 && alone < 262'144,
          "main's contexts of the 35B on the M6 moved");
  StreamedMemoryInputs inputs = k35BQ2.inputs(kM6);
  const auto streams = [&](uint32_t context) {
    inputs.requestedContextTokens = context;
    return requirePlan(inputs).expertBytes != 0;
  };
  require(!streams(225'000) && !streams(beside) && streams(beside + 1) && streams(233'000),
          "the 35B beside the SSD cache's staging did not stream its experts exactly past main's context");
  inputs.fixedBytes -= k35BQ2.stagingBytes;
  require(!streams(233'000) && !streams(alone) && streams(alone + 1),
          "the 35B without the SSD cache's staging did not stream its experts exactly past main's context");
}

// A dense target's explicit context is its KV allowance: where the experts
// loaded whole leave less, they stream, and the plan then holds the context.
void testDenseExplicitContextStreamsExperts() {
  StreamedMemoryInputs inputs = k35BQ2.inputs(kM6);
  inputs.requestedContextTokens = 262'144;
  const StreamedMemoryPlan plan = requirePlan(inputs);
  require(plan.expertBytes && plan.expertBytes < k35BQ2.experts.allBytes,
          "the 35B did not stream experts for a 256K context on the M6");
  // The experts take all the context leaves: a page more of them and the
  // plan would not hold it.
  const uint64_t page = k35BQ2.layout.bytesPerModelPage();
  const auto context = [&](uint64_t expertBytes, uint64_t maximumMemoryBytes) {
    return test::requireMemoryPlan(deviceOf(kM6), k35BQ2.profile(expertBytes), maximumMemoryBytes)
        .maximumContextTokens();
  };
  require(context(plan.expertBytes, 0) == 262'144 && context(plan.expertBytes + page, 0) < 262'144,
          "the streamed plan does not hold the context it was sized for, or holds more");
  // Without one, the experts that do not fit stream and the KV keeps the
  // first 64K tokens: the M6 with --max-memory 12G.
  StreamedMemoryInputs tight = k35BQ2.inputs(kM6);
  tight.hardBudgetBytes = 12 * kGiB;
  const StreamedMemoryPlan streamed = requirePlan(tight);
  require(streamed.expertBytes, "the 35B loaded whole beyond its budget");
  require(context(streamed.expertBytes, 12 * kGiB) >= 65'536 &&
              context(streamed.expertBytes + page, 12 * kGiB) < 65'536,
          "streaming experts did not keep exactly the first 64K tokens resident");
}

// The arenas are those of the parts that stream: each plan is the one made
// with the arenas of what it streams, the routed experts' prefill groups and
// the table's staging. Flash-Next with --expert-memory 12G from a budget
// that loads its table whole to one that streams it, with a staging of
// 512 MiB, and with experts the plan sizes from one that loads them whole
// to one that streams them.
void testArenasFollowWhatStreams() {
  const auto same = [](const StreamedMemoryPlan &plan, const StreamedMemoryPlan &other) {
    return plan.expertBytes == other.expertBytes && plan.tableBytes == other.tableBytes &&
           plan.spill.floorBytes == other.spill.floorBytes;
  };
  const auto planned = [](StreamedMemoryInputs inputs, uint64_t arenaBytes) {
    inputs.arenaBytes = {{{arenaBytes, arenaBytes}, {arenaBytes, arenaBytes}}};
    return requirePlan(inputs);
  };
  StreamedMemoryInputs inputs = kFlashNext.inputs(kM5Pro);
  inputs.experts.setting = {StreamedSetting::Mode::Size, 12 * kGiB};
  const uint64_t group = arenaBytes(kFlashNext.group), staged = group + 512 * kMiB;
  inputs.arenaBytes[1] = {group, staged};
  bool whole = false, streams = false;
  for (inputs.hardBudgetBytes = 40 * kGiB; inputs.hardBudgetBytes < 56 * kGiB; inputs.hardBudgetBytes += kGiB / 4) {
    const StreamedMemoryPlan plan = requirePlan(inputs);
    require(same(plan, planned(inputs, plan.tableBytes ? staged : group)),
            "a plan's table was sized beside the arenas of another");
    (plan.tableBytes ? streams : whole) = true;
  }
  inputs = kFlashNext.inputs(kM5Pro);
  bool expertsWhole = false, expertsStream = false;
  for (inputs.hardBudgetBytes = 40 * kGiB; inputs.hardBudgetBytes < 72 * kGiB; inputs.hardBudgetBytes += kGiB / 4) {
    const StreamedMemoryPlan plan = requirePlan(inputs);
    require(same(plan, planned(inputs, arenaBytes(plan.expertBytes ? kFlashNext.group : kFlashNext.chunk))),
            "a plan's experts were sized beside the arenas of another");
    (plan.expertBytes ? expertsStream : expertsWhole) = true;
  }
  require(whole && streams && expertsWhole && expertsStream, "the budgets did not span both plans of each part");
}

// An explicit setting replaces its part: a size streams through that store,
// one that holds the whole part loads it, off loads it, and a size below the
// part's floor is refused; --sparse-kv-memory raises the hot cache's target
// and is refused below its floor. An automatic part whose smallest store
// takes all it would loads whole.
void testOverridesAndRefusals() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  inputs.experts.setting = {StreamedSetting::Mode::Size, 12 * kGiB};
  const StreamedMemoryPlan explicitExperts = requirePlan(inputs);
  require(explicitExperts.expertBytes == 12 * kGiB && explicitExperts.tableBytes == 512 * kMiB,
          "an explicit --expert-memory was not its store");
  inputs.experts.setting = {StreamedSetting::Mode::Size, kFlashNext.experts.allBytes};
  require(!planStreamedMemory(inputs).plan, "experts that do not fit loaded whole");
  inputs.experts.setting = {StreamedSetting::Mode::Size, kGiB};
  const StreamedMemoryResult below = planStreamedMemory(inputs);
  require(!below.plan && below.error.find("--expert-memory of 1024 MiB is less than its smallest store, 2560 MiB") !=
                             std::string::npos,
          "an --expert-memory below its floor was not refused by name");

  StreamedMemoryInputs dense = k35BQ2.inputs(kM5Pro);
  for (const StreamedSetting setting :
       {StreamedSetting{StreamedSetting::Mode::Off, 0}, StreamedSetting{StreamedSetting::Mode::Size, 20 * kGiB}}) {
    dense.experts.setting = setting;
    require(!requirePlan(dense).expertBytes, "off, or a store that holds every expert, did not load them whole");
  }
  StreamedMemoryInputs tight = k35BQ2.inputs(kM6);
  tight.hardBudgetBytes = 12 * kGiB;
  tight.experts.floorBytes = tight.experts.allBytes;
  const StreamedMemoryResult whole = planStreamedMemory(tight);
  require(!whole.plan && whole.requiredBytes > 12 * kGiB,
          "experts whose smallest store takes all they would streamed instead of loading whole");

  inputs = kFlashNext.inputs(kStudio);
  const StreamedMemoryPlan automatic = requirePlan(inputs);
  inputs.hotCacheBytes = 1'200 * kMiB;
  const StreamedMemoryPlan raised = requirePlan(inputs);
  require(raised.spill.hotTargetBytes == 1'200 * kMiB && raised.spill.hotFloorBytes == automatic.spill.hotFloorBytes &&
              automatic.expertBytes - raised.expertBytes == 1'200 * kMiB - automatic.spill.hotFloorBytes,
          "--sparse-kv-memory did not take its target from the experts");
  // Below its floor, or beyond what the floors leave, it is refused by name
  // as a setting, never as floors the budget cannot hold.
  for (const auto &[bytes, refusal] : {std::pair<uint64_t, std::string>{300 * kMiB, "is less than"},
                                       std::pair<uint64_t, std::string>{16 * kGiB, "MiB the floors leave"}}) {
    inputs.hotCacheBytes = bytes;
    const StreamedMemoryResult refused = planStreamedMemory(inputs);
    require(!refused.plan && !refused.requiredBytes &&
                refused.error.find("--sparse-kv-memory of " + std::to_string(bytes / kMiB) + " MiB") !=
                    std::string::npos &&
                refused.error.find(refusal) != std::string::npos,
            "a hot cache target the plan cannot hold was not refused by name");
  }

  // Floors beyond the budget: Flash-Next with --max-memory 8G.
  inputs = kFlashNext.inputs(kM6);
  inputs.hardBudgetBytes = 8 * kGiB;
  const StreamedMemoryResult refused = planStreamedMemory(inputs);
  require(!refused.plan && refused.requiredBytes > 8 * kGiB &&
              refused.error.find("bytes but the Metal memory budget is " + std::to_string(8 * kGiB)) !=
                  std::string::npos,
          "floors beyond the budget were planned");
}

// A target spills only with a spillable layout, its shape without a local
// window its rows read whole, and an SSD cache that takes writes, and only
// where its floors hold an empty Spill request; otherwise it plans as a dense
// target.
void testSpillNeedsATierAndRoom() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  inputs.tierWritable = false;
  require(!requirePlan(inputs).spill.enabled, "a target spilled without an SSD cache that takes writes");
  inputs = kFlashNext.inputs(kStudio);
  inputs.sparseKv.reset();
  require(!requirePlan(inputs).spill.enabled, "a target spilled without declaring its sparse attention");
  inputs = kFlashNext.inputs(kStudio);
  inputs.sparseKv->localTokens = 128;
  require(!requirePlan(inputs).spill.enabled, "a target whose rows read a local window spilled");
  inputs = k35BQ2.inputs(kStudio);
  inputs.sparseKv = test::sparseKvShape();
  require(!requirePlan(inputs).spill.enabled, "a layout that cannot spill spilled");
  // A budget whose floors hold the model but not the hot cache's floor.
  inputs = kFlashNext.inputs(kM6);
  inputs.experts.setting = inputs.table.setting = {StreamedSetting::Mode::Off, 0};
  inputs.hardBudgetBytes = inputs.fixedBytes + inputs.arenaBytes[0][0] + inputs.laneStateBytes +
                           256 * test::sparseKvLayout().bytesPerModelPage() +
                           kFlashNext.experts.allBytes + kFlashNext.table.allBytes + 200 * kMiB;
  require(!requirePlan(inputs).spill.enabled, "spilling was planned without room for its floors");
}

// A plan that spills holds the arenas of a runtime whose K/V spills, which
// runs prefill groups and holds where its verifies' blocks are: beside
// streamed experts what they add comes out of the experts' store, and where
// the floors hold an empty Spill request only beside the arenas of a runtime
// whose K/V stays in memory, the plan does not spill.
void testSpillHoldsTheArenasOfASpillingRuntime() {
  constexpr uint64_t kExtra = 256 * kMiB;
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  const StreamedMemoryPlan resident = requirePlan(inputs);
  const uint64_t group = arenaBytes(kFlashNext.group);
  inputs.spillArenaBytes = {group + kExtra, group + kExtra};
  const StreamedMemoryPlan spilling = requirePlan(inputs);
  require(resident.spill.enabled && spilling.spill.enabled && resident.expertBytes - spilling.expertBytes == kExtra,
          "a plan that spills did not hold the arenas of a runtime whose K/V spills");

  inputs = kFlashNext.inputs(kM6);
  inputs.experts.setting = inputs.table.setting = {StreamedSetting::Mode::Off, 0};
  const uint64_t chunk = arenaBytes(kFlashNext.chunk);
  inputs.spillArenaBytes = {chunk, chunk};
  const uint64_t base = inputs.fixedBytes + chunk + inputs.laneStateBytes +
                        256 * test::sparseKvLayout().bytesPerModelPage() + kFlashNext.experts.allBytes +
                        kFlashNext.table.allBytes;
  for (inputs.hardBudgetBytes = base; !requirePlan(inputs).spill.enabled; inputs.hardBudgetBytes += 16 * kMiB)
    require(inputs.hardBudgetBytes < base + 4 * kGiB, "no budget let the target spill");
  inputs.spillArenaBytes = {chunk + kExtra, chunk + kExtra};
  require(!requirePlan(inputs).spill.enabled, "spilling was planned beside arenas its runtime does not run");
  inputs.hardBudgetBytes += kExtra;
  require(requirePlan(inputs).spill.enabled, "the spilling runtime's arenas took more than they add");
}

// The SSD cache's room bounds the context spilling serves at a first start,
// beside every lane's state: none leaves the resident context. A relaunch
// keeps its first engine's context.
void testDiskBoundsTheSpilledContext() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  inputs.activeLimitBytes = 4 * inputs.stateSlotBytes + 1'000 * inputs.kvSlotBytes;
  const StreamedMemoryPlan plan = requirePlan(inputs);
  require(plan.spill.diskTokens == 32'000 && plan.spill.floorTokens == 32'000,
          "the disk bound is not the room's slots beside every lane's state");
  inputs.activeLimitBytes = 4 * inputs.stateSlotBytes;
  require(requirePlan(inputs).spill.diskTokens == 0, "a full SSD cache bounded nothing");
  // An explicit context is held at its own length.
  inputs = kFlashNext.inputs(kStudio);
  inputs.requestedContextTokens = 100'000;
  require(requirePlan(inputs).spill.floorTokens == 100'000, "the floors were not held at --max-context");
  // A relaunch keeps its first engine's context, and its floors, whatever
  // room the volume has left since; more room serves more.
  inputs = kFlashNext.inputs(kStudio);
  inputs.relaunchContextTokens = 300'000;
  inputs.activeLimitBytes = 4 * inputs.stateSlotBytes + 1'000 * inputs.kvSlotBytes;
  const StreamedMemoryPlan relaunched = requirePlan(inputs);
  require(relaunched.spill.diskTokens == 300'000 && relaunched.spill.floorTokens == 300'000,
          "a relaunch served less than its first engine's context for want of SSD room");
  inputs.activeLimitBytes = 4 * inputs.stateSlotBytes + 20'000 * inputs.kvSlotBytes;
  require(requirePlan(inputs).spill.diskTokens == 640'000, "a relaunch did not take the SSD's room past its context");
}

// One staging buffer where the floors leave less than 4 GiB.
void testOneStagingBufferWhenTight() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kM6);
  inputs.hardBudgetBytes = 14 * kGiB;
  const StreamedMemoryPlan plan = requirePlan(inputs);
  require(plan.spill.stagingBuffers == 1 && plan.expertBytes >= kFlashNext.experts.floorBytes,
          "a tight budget staged in two buffers");
}

// The plan serves what spilling serves, the window where memory and the SSD
// cache's room hold it, and admission's geometry follows the plan.
void testEnginePlanServesTheSpilledContext() {
  for (const Mac &mac : kMacs) {
    const StreamedMemoryPlan streamed = requirePlan(kFlashNext.inputs(mac));
    const EngineMemoryPlan plan = test::requireMemoryPlan(
        deviceOf(mac), kFlashNext.profile(streamed.expertBytes, streamed.tableBytes, streamed.spill));
    const ContextBounds &bounds = plan.breakdown().contextBounds;
    require(bounds.window == 1'010'000 && bounds.spilled == 1'010'000 && bounds.disk == streamed.spill.diskTokens &&
                bounds.resident >= 65'536 && bounds.resident < bounds.disk &&
                plan.maximumContextTokens() == std::min(bounds.spilled, bounds.disk),
            std::string("Flash-Next's context on the ") + mac.name + " is not what spilling serves");
    require(plan.toStatusJson().find("\"context_bounds\":{\"window\":1010000") != std::string::npos &&
                plan.toStatusJson().find("\"kv_spill\":{\"dense_pages\":64") != std::string::npos,
            "the memory status does not report the spill plan");
    const SpillGeometry geometry = plan.spillGeometry();
    require(geometry.fullPageBytes == 423'936 && geometry.indexPageBytes == 24'576 &&
                geometry.slotBytes == 425'984 && geometry.densePages == 64 && geometry.windowPages == 64 &&
                geometry.groupPages == 256 &&
                geometry.residentKvBytes == plan.breakdown().dynamicBudgetBytes -
                                                plan.breakdown().laneStateBytes - streamed.spill.indexRunwayBytes -
                                                std::max(streamed.spill.stagingBytes, streamed.spill.hotTargetBytes) &&
                geometry.residentTokens == bounds.resident,
            "admission's geometry is not the plan's");
  }
  const SpillGeometry geometry{.fullPageBytes = 423'936, .indexPageBytes = 24'576, .densePages = 64,
                               .windowPages = 64, .groupPages = 256};
  require(geometry.residentBytes(1) == 423'936 && geometry.residentBytes(32 * 1'000) == 1'000 * 423'936 &&
              geometry.spillFloorBytes(32 * 100) == 100 * 423'936 &&
              geometry.spillFloorBytes(32 * 1'000) == 640 * 423'936 + 360 * 24'576,
          "a request's KV is not its pages, Full or Index");
}

// A disk bound below the resident context leaves the resident context, and
// names what bounds the maximum.
void testContextBoundNames() {
  require(ContextBounds{.window = 262'144, .resident = 300'000}.bound() == std::string_view("model") &&
              ContextBounds{.window = 262'144, .resident = 100'000}.bound() == std::string_view("memory") &&
              ContextBounds{.window = 1'010'000, .resident = 60'000, .spilled = 1'010'000, .disk = 400'000}.bound() ==
                  std::string_view("disk") &&
              ContextBounds{.window = 1'010'000, .resident = 60'000, .spilled = 500'000, .disk = 900'000}.bound() ==
                  std::string_view("memory") &&
              ContextBounds{.window = 1'010'000, .resident = 60'000, .spilled = 500'000, .disk = 9'000}
                      .maximum() == 60'000,
          "the context bound is misnamed");
}

} // namespace

void testDeviceValidationNamesTheMacosFloor() {
  require(!device().validationError(),
          "the reference device reported a validation error");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "macOS 26.3 was not refused with the macOS reason");
  // The operating system is checked before the device.
  older.appleGpuFamily = 8;
  require(older.validationError().value_or("") == "macos_26_4_required",
          "an older macOS did not take precedence over the device's reason");
  require(older.macosVersion() == "26.3.0",
          "the macOS version string is not major.minor.patch");
  DeviceCapabilities unknown = device();
  unknown.macosMajor = 0;
  unknown.macosMinor = 0;
  require(unknown.validationError().value_or("") == "macos_26_4_required",
          "an unknown macOS version was accepted");
  DeviceCapabilities newer = device();
  newer.macosMajor = 27;
  newer.macosMinor = 0;
  require(!newer.validationError(), "a newer macOS major was refused");
}

// The widest kernels dispatch threadgroups of 1024 threads.
void testDeviceValidationNamesTheThreadgroupWidth() {
  DeviceCapabilities narrow = device();
  narrow.maxThreadgroupWidth = 512;
  require(narrow.validationError().value_or("") ==
              "threadgroup_width_below_1024",
          "512-thread threadgroups were not refused for their width");
}

void testDeviceValidationMessageNamesWhatTheMacHas() {
  require(!device().validationMessage(),
          "the reference device has a validation message");
  const std::string needs =
      "Splash needs Apple GPU family 9 or newer (M3 or later) on macOS 26.4 "
      "or newer; this Mac has ";
  DeviceCapabilities m2 = device();
  m2.deviceName = "Apple M2 Max";
  m2.appleGpuFamily = 8;
  m2.macosPatch = 1;
  require(m2.validationMessage().value_or("") ==
              needs + "Apple M2 Max (Apple GPU family 8) on macOS 26.4.1 "
                      "(apple_gpu_family_9_required)",
          "a family-8 GPU was not named against the family required");
  DeviceCapabilities older = device();
  older.macosMinor = 3;
  require(older.validationMessage().value_or("") ==
              needs + "test (Apple GPU family 9) on macOS 26.3.0 "
                      "(macos_26_4_required)",
          "an older macOS was not named against the macOS required");
}

int main() {
  try {
    testUnifiedElasticBudget();
    testBf16BudgetAndStatus();
    testStateFormatStatus();
    testMinimumHoldsTheWarmupRunway();
    testMinimumRequiredBytesIsThePlans();
    testUserCeilingAndFailure();
    testDiskTierStateStagingIsBudgeted();
    testNeuralEngineSplitIsBudgeted();
    testStreamedTableIsBudgeted();
    testReservesAreTheModelConstants();
    testHardBudgetBoundaries();
    testModelProvidedKvGeometry();
    testExtentSizeFollowsThePool();
    testHotBlockCacheIsDefinedOnce();
    testFlashNextOnTheThreeMacs();
    testDense35BKeepsMainsPlan();
    testDenseExplicitContextKeepsMainsExperts();
    testDenseExplicitContextStreamsExperts();
    testArenasFollowWhatStreams();
    testOverridesAndRefusals();
    testSpillNeedsATierAndRoom();
    testSpillHoldsTheArenasOfASpillingRuntime();
    testDiskBoundsTheSpilledContext();
    testOneStagingBufferWhenTight();
    testEnginePlanServesTheSpilledContext();
    testContextBoundNames();
    testDeviceValidationNamesTheMacosFloor();
    testDeviceValidationNamesTheThreadgroupWidth();
    testDeviceValidationMessageNamesWhatTheMacHas();
    std::cout << "elastic memory plan tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "elastic memory plan tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
