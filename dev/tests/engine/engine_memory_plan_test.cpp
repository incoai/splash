#include "TestChecks.hpp"
#include "TestJson.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/StreamedMemoryPlan.hpp"
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

// The pages the budget holds for one request's KV beside one lane's state,
// each with its tokens' lookup index on the host.
uint64_t budgetPages(const EngineMemoryBreakdown &budget) {
  return (budget.dynamicBudgetBytes - budget.laneStateBytes) /
         (budget.kvPageBytes + 32 * model::kLookupIndexTokenBytes);
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
                  budget.laneStateBytes + budget.kvExtentBytes + 128 * 32 * model::kLookupIndexTokenBytes &&
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
  const std::string json = test::jsonOf(plan);
  require(json.find("\"dynamic_budget_bytes\"") != std::string::npos &&
              json.find("\"kv_extent_pages\":128") != std::string::npos,
          "elastic state/KV budget is missing from memory status");
  require(json.find("\"working_set_margin_bytes\":" +
                    std::to_string(budget.workingSetMarginBytes)) !=
                  std::string::npos &&
              json.find("\"headroom_bytes\"") == std::string::npos,
          "memory status did not name the working-set margin");
}

// What the budget holds a lone request beside more than its lane's state is
// its KV capacity less the speculative scratch beside nothing, and an extent's
// tokens less beside an extent's bytes, windowed or not.
void testTheContextBesideMoreIsTheCapacityLessWhatItTakes() {
  ModelMemoryProfile windowed = model();
  windowed.targetKvLayout = {49, 4, 128, kv::Format::Int8, 0, false, 512, 0xffffull};
  for (const ModelMemoryProfile &profile : {model(), windowed}) {
    const EngineMemoryPlan plan = test::requireMemoryPlan(device(), profile);
    const EngineMemoryBreakdown &budget = plan.breakdown();
    const uint64_t scratch = model::ExecutionLimits::speculativeScratchTokens;
    require(plan.residentTokensBeside(0) == budget.kvCapacityTokens - scratch &&
                plan.residentTokensBeside(budget.dynamicBudgetBytes) == 0,
            "the context beside nothing is not the KV capacity less the scratch");
    const uint64_t lessExtent = plan.residentTokensBeside(budget.kvExtentBytes + budget.kvExtentPages * 32 *
                                                                                     model::kLookupIndexTokenBytes);
    require(lessExtent < budget.kvCapacityTokens - scratch &&
                lessExtent + uint64_t{budget.kvExtentPages} * 32 >= budget.kvCapacityTokens - scratch,
            "the context beside an extent more is not an extent's tokens less: " + std::to_string(lessExtent) +
                " of " + std::to_string(budget.kvCapacityTokens));
  }
}

// The requests a budget runs at once beside one of its served context are the
// lanes whose states fit beside that request's KV: all of them beside a short
// context; fewer beside a tight budget's whole context, with the most context
// and the fewest bytes more that hold one more, each just enough.
void testTheRequestsRunAtOnceAreTheLanesWhoseStatesFitBesideOne() {
  const EngineMemoryPlan roomy = test::requireMemoryPlan(device(), model());
  const EngineMemoryBreakdown &automatic = roomy.breakdown();
  EngineMemoryPlan tight = test::requireMemoryPlan(
      device(), model(), automatic.minimumRequiredBytes + 3 * automatic.laneStateBytes + automatic.kvExtentBytes / 2);
  for (EngineMemoryPlan *plan : {&tight}) {
    const EngineMemoryBreakdown &budget = plan->breakdown();
    const uint32_t width = budget.maximumBatchWidth;
    require(plan->noteConcurrentRequests(256) == width && budget.moreRequestsBytes == 0 &&
                budget.moreRequestsContextTokens == 0,
            "a short context did not leave every lane its state");
    const uint32_t served = plan->maximumContextTokens();
    const uint32_t requests = plan->noteConcurrentRequests(served);
    const uint64_t beside = uint64_t{requests} * budget.laneStateBytes;
    require(requests >= 1 && requests < width && budget.concurrentRequests == requests &&
                plan->residentTokensBeside(beside) < served &&
                budget.moreRequestsContextTokens == plan->residentTokensBeside(beside) &&
                plan->residentTokensBeside(beside - budget.moreRequestsBytes) >= served &&
                plan->residentTokensBeside(beside - budget.moreRequestsBytes + 1) < served,
            "a tight budget's whole context did not hold fewer requests than lanes, with what one more needs: " +
                std::to_string(requests) + " of " + std::to_string(width) + ", " +
                std::to_string(budget.moreRequestsBytes) + " bytes more, context " +
                std::to_string(budget.moreRequestsContextTokens));
    require(plan->noteConcurrentRequests(budget.moreRequestsContextTokens) > requests,
            "the context named for one more request did not hold one more");
    plan->noteConcurrentRequests(served);
    require(test::jsonOf(*plan).find("\"concurrent_requests\":" + std::to_string(requests)) != std::string::npos,
            "the status does not report the requests the budget runs at once");
  }
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
  const auto json = test::jsonOf(bf16);
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
  require(test::jsonOf(test::requireMemoryPlan(device(), profile)).find("\"state_format\":\"bf16\"") !=
                  std::string::npos &&
              test::jsonOf(test::requireMemoryPlan(device(), model())).find("\"state_format\":\"fp32\"") !=
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
              budget.laneStateBytes + 64 * (budget.kvPageBytes + 32 * model::kLookupIndexTokenBytes),
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
    require(minimumRequiredBytes(budget.fixedRuntimeBytes, budget.laneStateBytes, profile.targetKvLayout,
                                 budget.maximumBatchWidth) == budget.minimumRequiredBytes,
            "the minimum differs from the plan's");
  }
  require(!minimumRequiredBytes(std::numeric_limits<uint64_t>::max(), 1, model().targetKvLayout,
                                model::ExecutionLimits::defaultLanes),
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
  // One Qwen3.8-27B state (docs/server.md, SSD cache).
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
  // The room the host held back of the streamed stores, none here, follows
  // the staging among the fixed parts.
  const std::string field = "\"state_staging_bytes\":" + std::to_string(staging);
  require(test::jsonOf(with).find(field + ",\"host_held_back_bytes\":0,\"fixed_runtime_bytes\"") !=
                  std::string::npos &&
              test::jsonOf(with).find(field + "}}") != std::string::npos &&
              budget.describe().find("SSD cache state staging: " +
                                     std::to_string(staging)) != std::string::npos,
          "state staging is missing from the memory plan status");
  require(without.breakdown().stateStagingBytes == 0 &&
              test::jsonOf(without).find("\"state_staging_bytes\":0,") !=
                  std::string::npos,
          "a plan without the disk tier reported state staging");
}

// The prefill FFN's Neural Engine split is fixed runtime memory of its own
// category, which the KV cache gives up, so the context it leaves is the
// plan's to state; the status and description report it.
void testNeuralEngineSplitIsBudgeted() {
  const EngineMemoryPlan without = test::requireMemoryPlan(device(), model());
  ModelMemoryProfile split = model();
  // Qwen3.8-27B's split at the M6's share (docs/internals.md, Neural Engine prefill).
  const uint64_t surfaces = 508 * kMiB;
  split.footprint.aneFfnBytes = surfaces;
  const EngineMemoryPlan with = test::requireMemoryPlan(device(), split);
  const auto &budget = with.breakdown();
  require(budget.aneFfnBytes == surfaces &&
              budget.fixedRuntimeBytes == without.breakdown().fixedRuntimeBytes + surfaces &&
              with.maximumContextTokens() < without.maximumContextTokens(),
          "the split was not planned as fixed runtime memory out of the KV cache");
  require(test::jsonOf(with).find("\"ane_ffn_bytes\":" + std::to_string(surfaces) +
                                   ",\"state_staging_bytes\"") != std::string::npos &&
              budget.describe().find("Neural Engine split: " + std::to_string(surfaces)) !=
                  std::string::npos &&
              test::jsonOf(without).find("\"ane_ffn_bytes\":0,") != std::string::npos,
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
  require(test::jsonOf(with).find("\"streamed_table_bytes\":" + std::to_string(cache) + ",\"ane_ffn_bytes\"") !=
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
  require(EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 0, 16 * kGiB) == 11 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 8 * kGiB, 16 * kGiB) ==
                  8 * kGiB &&
              EngineMemoryPolicy::hardBudgetBytes(12 * kGiB, 16 * kGiB, 16 * kGiB) ==
                  11 * kGiB,
          "preflight ceiling disagrees with automatic or explicit policy");
  require(EngineMemoryPolicy::hardBudgetBytes(0, 0, 0) == 0 &&
              EngineMemoryPolicy::hardBudgetBytes(kGiB, 1, 16 * kGiB) == 0,
          "insufficient working set underflowed the preflight ceiling");
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  require(EngineMemoryPolicy::hardBudgetBytes(maximum, maximum, maximum) ==
              maximum - EngineMemoryPolicy::workingSetMarginBytes(maximum),
          "maximum working set overflowed the preflight ceiling");
}

// A recommendation raised past 90% of the Mac's memory (iogpu.wired_limit_mb, #735's 124,518 MiB on a 128 GB M5 Max)
// leaves the automatic budget at the Mac's memory less a tenth; an explicit --max-memory is the user's, capped by the
// recommendation alone. On the Macs measured (recommendations as Metal reports them, no raised limit), nothing
// changes: the budget is the recommendation less its margin.
void testTheAutomaticBudgetLeavesATenthUnwired() {
  constexpr uint64_t memory = 128 * kGiB, raised = 124'518ULL << 20;
  const uint64_t recommended = raised - EngineMemoryPolicy::workingSetMarginBytes(raised);
  require(EngineMemoryPolicy::hardBudgetBytes(raised, 0, memory) == memory - memory / 100 * 10,
          "the automatic budget of a Mac with a raised wired limit left less than a tenth of its memory unwired: " +
              std::to_string(EngineMemoryPolicy::hardBudgetBytes(raised, 0, memory)));
  require(EngineMemoryPolicy::hardBudgetBytes(raised, 120 * kGiB, memory) == recommended &&
              EngineMemoryPolicy::hardBudgetBytes(raised, 116 * kGiB, memory) == 116 * kGiB,
          "an explicit --max-memory was held to the unwired floor, or past the recommendation");
  struct Measured {
    const char *mac;
    uint64_t memory, recommendation;
  };
  constexpr Measured measured[] = {{"M4 16 GB", 16 * kGiB, 12'713'115'648},
                                   {"M6 24 GB", 24 * kGiB, 19'069'665'280},
                                   {"M5 Max 36 GB", 36 * kGiB, 30'150'672'384},
                                   {"M5 Pro 48 GB", 48 * kGiB, 40'200'896'512},
                                   {"M5 Pro 64 GB", 64 * kGiB, 55'662'788'608}};
  for (const Measured &each : measured)
    require(EngineMemoryPolicy::hardBudgetBytes(each.recommendation, 0, each.memory) ==
                each.recommendation - EngineMemoryPolicy::workingSetMarginBytes(each.recommendation),
            std::string("the unwired floor changed the budget of the ") + each.mac);
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
  require(test::jsonOf(plan).find("\"attention_layers\":10") !=
              std::string::npos &&
              test::jsonOf(plan).find("\"kv_heads\":2") !=
                  std::string::npos,
          "model-provided Q8 geometry is missing from status");
}

// A windowed layout's request holds whole only its lane's floor of blocks, its window's 16, a prefill chunk's 64
// and one more, 81 blocks of a base page and a window unit, and past them each block's base page: Flint's 49
// layers of 4 heads of 128, 35 of them windowed over 512 keys, take pages of its 14 layers no window bounds, a
// window unit of 3 pages (7 slots unused). One request's KV is priced so in whole extents, with each block's
// tokens' lookup index; it holds more than three times the tokens of pages of every layer, the status names the
// floor in window pages, and a streamed group's 256 pages make the floor's command.
void testWindowedKvCostsItsFloorWhole() {
  constexpr uint64_t windowedLayers = [] {
    uint64_t layers = (uint64_t{1} << 48) - 4;
    for (uint32_t layer = 5; layer < 48; layer += 4)
      layers &= ~(uint64_t{1} << layer);
    return layers;
  }();
  ModelMemoryProfile flint = model();
  flint.targetKvLayout = {49, 4, 128, kv::Format::Int8, 0, false, 512, windowedLayers};
  ModelMemoryProfile unwindowed = flint;
  unwindowed.targetKvLayout.windowTokens = 0;
  unwindowed.targetKvLayout.windowedLayers = 0;
  const EngineMemoryPlan windowedPlan = test::requireMemoryPlan(device(), flint);
  const EngineMemoryBreakdown &budget = windowedPlan.breakdown();
  const EngineMemoryBreakdown full = test::requireMemoryPlan(device(), unwindowed).breakdown();
  const uint64_t available = budget.dynamicBudgetBytes - budget.laneStateBytes;
  const uint64_t index = 32 * model::kLookupIndexTokenBytes, floor = 81, unit = 3;
  const uint64_t blocks = budget.kvCapacityPages - unit * floor;
  const uint64_t held = uint64_t{budget.kvCapacityPages} * budget.kvPageBytes + blocks * index;
  require(budget.kvPageBytes == 473'088 && budget.kvWindowFloorPages == unit * floor &&
              budget.kvCapacityPages % budget.kvExtentPages == 0 &&
              budget.kvCapacityBytes == uint64_t{budget.kvCapacityPages} * budget.kvPageBytes &&
              budget.kvCapacityTokens == blocks * 32 && held <= available &&
              available - held < uint64_t{budget.kvExtentPages} * (budget.kvPageBytes + index) &&
              budget.kvCapacityTokens > 3 * full.kvCapacityTokens,
          "a windowed layout's KV capacity is not its lane's floor of whole blocks and base pages past it, in whole "
          "extents");
  const uint64_t runway = model::ExecutionLimits::runwayPages(flint.targetKvLayout.minimumExtentPages(),
                                                              budget.maximumBatchWidth, 1 + unit);
  require(budget.minimumDynamicBytes ==
                  budget.laneStateBytes + runway * budget.kvPageBytes + runway / (1 + unit) * index &&
              budget.kvSmallestExtentPages == flint.targetKvLayout.minimumExtentPages(),
          "a windowed runway's window pages were charged a lookup index, or its extent went unreported");
  require(full.kvWindowFloorPages == 0 &&
              test::jsonOf(windowedPlan).find("\"kv_window\":{\"floor_pages\":243}") != std::string::npos &&
              test::jsonOf(test::requireMemoryPlan(device(), unwindowed)).find("kv_window") == std::string::npos,
          "the status did not name a windowed layout's floor alone, in window pages");
  flint.commandPages = 256;
  require(test::requireMemoryPlan(device(), flint).breakdown().kvWindowFloorPages == unit * 273,
          "a streamed group's pages did not make the floor's command");
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
            pages * (reference.kvPageBytes + 32 * model::kLookupIndexTokenBytes));
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
                  reference.laneStateBytes + 256 * (332'800 + 32 * model::kLookupIndexTokenBytes),
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

uint64_t hardBudget(const Mac &mac) { return EngineMemoryPolicy::hardBudgetBytes(mac.workingSet, 0, mac.memory); }

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
  // The row cache an automatic table takes at most
  // (model::StreamableWeights::tableShareBytes).
  uint64_t tableShareBytes = 0;

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
            .tableShareBytes = tableShareBytes,
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
                    {.allBytes = 10'530'069'232, .minimumBytes = 1'280 * kMiB},
                    {},
                    std::nullopt};
// UD-Q4_K_M: a 22.1 GB GGUF the Studio and the M5 Pro serve.
const Target k35BQ4 = [] {
  Target target = k35BQ2;
  target.loadedBytes = 2'960'000'000;
  target.experts.allBytes = 19'176'000'000;
  target.experts.minimumBytes = 2 * kGiB;
  return target;
}();
// Qwen3.8-Flash-Next's automatic row cache: the 2M rows of 90 bytes of
// every id a 128K-token prompt stages, with their ids, recency links and
// index, beside the working memory of its stages and prefetches.
constexpr uint64_t kFlashNextTableShare = 248'840'192;
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
                        {.allBytes = 33'976'681'712, .minimumBytes = 2'560 * kMiB},
                        {.allBytes = 28'799'139'840, .minimumBytes = 16 * kMiB},
                        test::sparseKvShape(),
                        kFlashNextTableShare};

StreamedMemoryPlan requirePlan(const StreamedMemoryInputs &inputs) {
  StreamedMemoryResult result = planStreamedMemory(inputs);
  if (!result.plan) throw std::runtime_error("the planner refused: " + result.error);
  return *result.plan;
}

// One definition of the hot block cache: its floor holds requiredSlots(1)
// slots a layer, a verify block's 16 rows x 512 blocks and a spare of 8
// rows', in whole extents: six of Flash-Next's 256-page extents of K/V
// without the index, 585 MiB.
void testHotBlockCacheIsDefinedOnce() {
  const model::SparseKvShape shape = test::sparseKvShape();
  require(shape.laneSlots() == 8'192 && shape.requiredSlots(0) == 0 && shape.requiredSlots(1) == 12'288 &&
              shape.requiredSlots(4) == 36'864,
          "the hot block cache's floor is not lanes x laneSlots and 8 rows' spare");
  const KvSpillPlan spill = requirePlan(kFlashNext.inputs(kM5Pro)).spill;
  const kv::Layout layout = test::sparseKvLayout();
  require(spill.enabled && spill.laneSlots == 8'192 && spill.hotFloorBytes == 6 * 256 * layout.kvBytesPerModelPage() &&
              spill.hotFloorBytes == 613'416'960 && spill.hotTargetBytes == spill.hotFloorBytes,
          "the hot floor is not requiredSlots(1) in whole extents");
  require(spill.indexRunwayBytes == 256 * layout.indexBytesPerModelPage() && spill.densePages == 64 &&
              spill.windowPages == 64 && spill.groupPages == 256,
          "the spill floors' pages are not the shape's");
}

// The priorities on the three Macs: the floors, then Flash-Next's experts
// take all that the KV allowance (the first 64K tokens, resident) and the
// PLE table's share leave, all of them on the M5 Pro; the table its share,
// the rows of every id a 128K-token prompt stages (it took 512 MiB before
// that share, whose rows hit no more often); the hot cache stays at its
// floor. The context spilling serves is the window where memory and the SSD
// cache's room hold it, the room's on the M6. A prefill stages its history in
// two buffers on the M5 Pro alone, where the experts load whole and leave the
// second its room: elsewhere the expert store takes it.
void testFlashNextOnTheThreeMacs() {
  const uint64_t page = test::sparseKvLayout().bytesPerModelPage();
  for (const Mac &mac : kMacs) {
    const StreamedMemoryInputs inputs = kFlashNext.inputs(mac);
    const StreamedMemoryPlan plan = requirePlan(inputs);
    const KvSpillPlan &spill = plan.spill;
    require(spill.enabled && plan.tableBytes == kFlashNextTableShare,
            std::string("Flash-Next's table on the ") + mac.name + " does not stream through its share");
    // The floors, the arenas, the stores and the KV of the first 64K tokens
    // and their speculative rows (2,049 pages) with their lookup index, in the fewest pages whose
    // extents hold them as one request's capacity, the runway among them,
    // fill the budget, or leave the KV more where the experts load whole.
    const uint64_t charged = inputs.fixedBytes + inputs.arenaBytes[plan.expertBytes != 0][1] + inputs.laneStateBytes +
                             spill.floorBytes +
                             (plan.expertBytes ? plan.expertBytes : kFlashNext.experts.allBytes) + plan.tableBytes +
                             budgetPagesHolding(test::sparseKvLayout(), 2'049) *
                                 (page + 32 * model::kLookupIndexTokenBytes);
    if (mac.workingSet == kM5Pro.workingSet)
      require(!plan.expertBytes && charged < hardBudget(mac), "the M5 Pro does not load Flash-Next's experts whole");
    else
      require(plan.expertBytes > kFlashNext.experts.minimumBytes && !plan.expertFloorBytes &&
                  charged == hardBudget(mac),
              std::string("Flash-Next's experts on the ") + mac.name + " do not take what the floors leave");
    const uint32_t disk = static_cast<uint32_t>(
        std::min<uint64_t>(1'010'000, (mac.activeLimit - 4 * inputs.stateSlotBytes) / inputs.kvSlotBytes * 32));
    require(spill.memoryTokens == 1'010'000 && spill.diskTokens == disk &&
                spill.floorTokens == std::min<uint32_t>(1'010'000, disk),
            std::string("Flash-Next's context bounds on the ") + mac.name + " are wrong");
    const uint64_t stagedPages =
        (pagesOf(spill.floorTokens + model::ExecutionLimits::speculativeScratchTokens) + 255) / 256 * 256;
    const uint32_t buffers = plan.expertBytes ? 1 : 2;
    require(spill.stagingBuffers == buffers &&
                spill.stagingBytes == buffers * stagedPages * test::sparseKvLayout().bytesPerLayerPage(),
            std::string("Flash-Next on the ") + mac.name + " does not stage in " + std::to_string(buffers) +
                " buffers");
  }
  require(requirePlan(kFlashNext.inputs(kM6)).expertBytes <
              requirePlan(kFlashNext.inputs(kStudio)).expertBytes,
          "a larger Mac did not stream more experts");
  // The floors hold the larger of the hot cache's floor and the staging,
  // both buffers where there are two, and the readers' tables: beside them
  // the M5 Max's and the M5 Pro's floors of one context are the same.
  const KvSpillPlan max = requirePlan(kFlashNext.inputs(kStudio)).spill, pro = requirePlan(kFlashNext.inputs(kM5Pro)).spill;
  require(max.floorTokens == pro.floorTokens && max.stagingBuffers == 1 && pro.stagingBuffers == 2 &&
              max.floorBytes - max.tableBytes - std::max(max.stagingBytes, max.hotFloorBytes) ==
                  pro.floorBytes - pro.tableBytes - std::max(pro.stagingBytes, pro.hotFloorBytes),
          "the floors did not hold both staging buffers and their tables where there are two");
}

// --ple-memory auto streams Flash-Next's table through its share however
// much the budget holds: a budget of 192 GiB loads every routed expert and
// would hold the table's 28.8 GB too, which off loads. A table no larger
// than its share loads whole where the budget holds it
// (testArenasFollowWhatStreams).
void testAutoStreamsTheTableWhereMemoryHoldsIt() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kM5Pro);
  inputs.hardBudgetBytes = 192 * kGiB;
  const StreamedMemoryPlan automatic = requirePlan(inputs);
  require(!automatic.expertBytes && automatic.tableBytes == kFlashNextTableShare,
          "auto did not stream Flash-Next's table through its share where a budget of 192 GiB holds it whole");
  inputs.table.setting = {StreamedSetting::Mode::Off, 0};
  require(!requirePlan(inputs).tableBytes, "--ple-memory off streamed Flash-Next's table");
}

// A spill-capable target keeps resident the requested context where that is
// less than 64K: Flash-Next with --max-context 16K on the M6 leaves the KV of
// 16K tokens and their speculative rows (513 pages) beside the experts, which
// take the rest.
void testSpillAllowanceFollowsASmallerRequestedContext() {
  const uint64_t page = test::sparseKvLayout().bytesPerModelPage();
  StreamedMemoryInputs inputs = kFlashNext.inputs(kM6);
  inputs.requestedContextTokens = 16'384;
  const StreamedMemoryPlan plan = requirePlan(inputs);
  const uint64_t charged = inputs.fixedBytes + inputs.arenaBytes[plan.expertBytes != 0][1] + inputs.laneStateBytes +
                           plan.spill.floorBytes + plan.expertBytes + plan.tableBytes +
                           budgetPagesHolding(test::sparseKvLayout(), 513) *
                               (page + 32 * model::kLookupIndexTokenBytes);
  require(plan.spill.enabled && plan.expertBytes && charged == hardBudget(kM6),
          "Flash-Next with a 16K context kept more than 16K tokens resident beside its experts");
}

// Main's plan for the 35B on the three Macs: every routed expert loads with
// its layer, and the KV holds main's context, pinned: the M6 229,361 tokens
// beside the whole Q2_K_XL, the others the 256K window of UD-Q4_K_M.
void testDense35BKeepsMainsPlan() {
  const std::tuple<Mac, Target, uint32_t, uint64_t> pinned[] = {
      {kM6, k35BQ2, 229'361, 229'376}, {kStudio, k35BQ4, 262'144, 344'064}, {kM5Pro, k35BQ4, 262'144, 2'777'088}};
  for (const auto &[mac, target, context, capacityTokens] : pinned) {
    const StreamedMemoryPlan plan = requirePlan(target.inputs(mac));
    require(!plan.expertBytes && !plan.tableBytes && !plan.spill.enabled,
            std::string("the 35B streams its experts on the ") + mac.name + " where main loads them");
    const EngineMemoryPlan mains = test::requireMemoryPlan(deviceOf(mac), target.profile());
    require(mains.maximumContextTokens() == context && mains.breakdown().kvCapacityTokens == capacityTokens &&
                mains.breakdown().contextBounds.bound() != std::string_view("disk") &&
                test::jsonOf(mains).find("\"kv_spill\":null") != std::string::npos,
            std::string("the 35B's plan on the ") + mac.name + " is not main's");
  }
}

// A runtime of eight lanes is planned as eight: its arenas, which hold
// 191.6 MB more decode and 1.0 MB more prefill than four lanes' on the 35B
// (model::plannedDecodeBytes, Apple10), and their KV runway are in the plan
// and its minimum. Where memory bounds the context, the 35B's on a 24 GB M6, the
// plan offers less than four lanes' and carries what four would leave
// (EngineMemoryPlan::compareDefaultLanes); where it holds the window beside
// them, a 36 GB M5 Max's, it carries nothing.
void testEightLanesArePlannedAsEight() {
  constexpr uint64_t kDecode = 191'643'648, kPrefill = 1'015'808;
  for (const auto &[mac, target, costs] : {std::tuple{kM6, k35BQ2, true}, {kStudio, k35BQ4, false}}) {
    const EngineMemoryPlan four = test::requireMemoryPlan(deviceOf(mac), target.profile());
    Target wide = target;
    wide.chunk.sharedDecodePlannedAllocatedBytes += kDecode;
    wide.chunk.sharedPrefillPlannedAllocatedBytes += kPrefill;
    wide.chunk.lanes = 8;
    EngineMemoryPlan eight = test::requireMemoryPlan(deviceOf(mac), wide.profile());
    const EngineMemoryBreakdown &a = four.breakdown(), &b = eight.breakdown();
    const uint32_t extent = target.layout.minimumExtentPages();
    const uint64_t runway =
        (model::ExecutionLimits::runwayPages(extent, 8, 1) - model::ExecutionLimits::runwayPages(extent, 4, 1)) *
        (target.layout.bytesPerModelPage() + kv::kPageTokens * model::kLookupIndexTokenBytes);
    require(a.maximumBatchWidth == 4 && b.maximumBatchWidth == 8 &&
                b.fixedRuntimeBytes == a.fixedRuntimeBytes + kDecode + kPrefill &&
                b.minimumRequiredBytes == a.minimumRequiredBytes + kDecode + kPrefill + runway,
            std::string("eight lanes of the 35B on the ") + mac.name + " were not planned as eight");
    require((eight.maximumContextTokens() < four.maximumContextTokens()) == costs,
            std::string("eight lanes of the 35B on the ") + mac.name + (costs ? " cost no" : " cost") + " context");
    eight.compareDefaultLanes(four, eight.maximumContextTokens(), 0);
    require(b.defaultLanesContextTokens == (costs ? four.maximumContextTokens() : 0),
            std::string("eight lanes of the 35B on the ") + mac.name + " did not carry four lanes' context");
    // A --max-context both hold is served alike: nothing to say.
    eight.compareDefaultLanes(four, 65'536, 65'536);
    require(b.defaultLanesContextTokens == 0,
            std::string("eight lanes of the 35B on the ") + mac.name + " said a cost a 64K context does not pay");
  }
}

// An explicit context loads every expert of the 35B where main holds it so
// on the M6, beside the SSD cache's state staging up to main's 229,361
// tokens, and without it up to main's context then (the automatic SSD cache
// gives its staging up for them, RuntimeResources::create); a token more
// streams them. 225,000 tokens load them beside the staging, and 233,000
// without it.
void testDenseExplicitContextKeepsMainsExperts() {
  ModelMemoryProfile withoutStaging = k35BQ2.profile();
  withoutStaging.footprint.stateStagingBytes = 0;
  const uint32_t beside = test::requireMemoryPlan(deviceOf(kM6), k35BQ2.profile()).maximumContextTokens(),
                 alone = test::requireMemoryPlan(deviceOf(kM6), withoutStaging).maximumContextTokens();
  require(beside == 229'361 && beside < 233'000 && alone >= 233'000 && alone < 262'144,
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

// A windowed target whose experts stream gets the KV its request asked for, in the layout's own extents, and the
// experts take the rest: its capacity holds the requested context with its speculative rows, and less than one
// extent of pages more.
void testAWindowedRequestGetsItsKvAndTheExpertsTheRest() {
  constexpr uint64_t windowedLayers = [] {
    uint64_t layers = (uint64_t{1} << 48) - 4;
    for (uint32_t layer = 5; layer < 48; layer += 4)
      layers &= ~(uint64_t{1} << layer);
    return layers;
  }();
  Target windowed = k35BQ2;
  windowed.layout = {49, 4, 128, kv::Format::Int8, 0, false, 512, windowedLayers};
  // Where the experts stream, a lane's floor holds a streamed group's command (RuntimeResources).
  const uint32_t group = 4 * 2048 / kv::kPageTokens, floor = kvWindowFloorBlocks(windowed.layout, group);
  for (const uint32_t context : {4'096u, 8'192u, 16'384u}) {
    StreamedMemoryInputs tight = windowed.inputs(kM6);
    tight.hardBudgetBytes = 12 * kGiB;
    tight.requestedContextTokens = context;
    const StreamedMemoryPlan plan = requirePlan(tight);
    require(plan.expertBytes && plan.expertBytes < windowed.experts.allBytes,
            "the windowed target's experts did not stream");
    ModelMemoryProfile profile = windowed.profile(plan.expertBytes);
    profile.commandPages = group;
    const EngineMemoryBreakdown budget = test::requireMemoryPlan(deviceOf(kM6), profile, 12 * kGiB).breakdown();
    const uint64_t blocks = pagesOf(context + SPLASH_SPECULATIVE_SCRATCH_TOKENS);
    const uint64_t pages = kvBlockPages(windowed.layout, blocks, floor);
    require(budget.kvCapacityTokens >= blocks * kv::kPageTokens && budget.kvCapacityPages >= pages &&
                budget.kvCapacityPages < pages + 2 * uint64_t{budget.kvExtentPages},
            "a windowed request did not get the KV it asked for in the layout's extents, or the experts left it more");
  }
}

// The 48 GB Mac the harness checks Flint and Inkling-2-Mini on, capped.
constexpr Mac k48g{"M5 Pro 48 GB", 48 * kGiB, 40'200'896'512, 32 * kGiB};

// A windowed target at the cap and context its capped harness checks run with, one lane, as /status reports it
// at 85015a6d: its images without the routed experts, its arenas, lane state and KV layout, its experts in their
// slots, a cached state's bytes and the expert store's slot.
struct CappedTarget final {
  Target target;
  uint64_t capMiB;
  uint32_t context;
  uint64_t stateBytes, slotBytes;

  // The stores' plan at `capBytes` with cached states of `stateBytes`, and the engine's beside its store of
  // exactly the bytes planned, no slot's rounding left over: nullopt where either refuses, with why.
  [[nodiscard]] std::pair<StreamedMemoryResult, std::optional<EngineMemoryBreakdown>>
  planned(uint64_t capBytes, uint64_t states, std::string &refusal) const {
    StreamedMemoryInputs inputs = target.inputs(k48g);
    inputs.hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(k48g.workingSet, capBytes, k48g.memory);
    inputs.requestedContextTokens = context;
    inputs.lanes = 1;
    inputs.cachedStateBytes = states;
    StreamedMemoryResult result = planStreamedMemory(inputs);
    if (!result.plan) {
      refusal = result.error;
      return {std::move(result), std::nullopt};
    }
    ModelMemoryProfile profile = target.profile(result.plan->expertBytes);
    profile.commandPages = 4 * 2048 / kv::kPageTokens;
    profile.footprint.cachedStatesBytes = result.plan->cachedStatesBytes;
    const EngineMemoryPlanResult engine = evaluateEngineMemoryPlan(deviceOf(k48g), profile, capBytes);
    if (!engine.plan) {
      refusal = engine.status.message;
      return {std::move(result), std::nullopt};
    }
    return {std::move(result), engine.plan->breakdown()};
  }
};

constexpr uint64_t kFlintWindowed = [] {
  uint64_t layers = (uint64_t{1} << 48) - 4;
  for (uint32_t layer = 5; layer < 48; layer += 4)
    layers &= ~(uint64_t{1} << layer);
  return layers;
}();
const CappedTarget kFlint{
    {"flint-mixed-3-4bit", 1'290'059'776, 338'116'608, 0, {44'302'336, 643'776'512, 47'972'352, 1},
     {44'302'336, 643'776'512, 47'972'352, 1}, 0,
     kv::Layout{49, 4, 128, kv::Format::Int8, 0, false, 512, kFlintWindowed}, 262'144,
     {.allBytes = 12'032 * 1'376'256ull, .minimumBytes = 2 * kGiB}, {}, std::nullopt},
    18'674, 4'096, 2'359'296, 1'376'256};
const CappedTarget kInkling2Mini{
    {"Inkling-2-Mini", 1'829'076'992, 428'015'616, 639'844'352, {48'431'104, 1'154'072'576, 66'699'264, 1},
     {48'431'104, 1'154'072'576, 66'699'264, 1}, 0,
     kv::Layout{48, 8, 128, kv::Format::Int8, 0, false, 1'023, 0x7DF7DF7DF7DFull}, 1'010'000,
     {.allBytes = 3'072 * 5'016'907ull, .minimumBytes = 2 * kGiB}, {}, std::nullopt},
    19'222, 8'192, 4'571'136, 5'016'907};

// A windowed target whose experts stream keeps room for cached states beside its exact K/V: as many as its
// capacity holds the tails of (a tail is the window units of a state's window of blocks), taken from the
// experts, so that the pool's last extent lands with them held. Without the room the plan left 1,537,008 bytes
// past Flint's lane state and K/V at 18674M, less than one state of 2,359,296: the K/V reached its capacity
// only with no cached state held, and a held state kept its third extent of 121,110,528 bytes out
// (conv:left+nossd@on). The K/V capacity, the context and the floors do not move; the experts give the room.
// Flint's 768 pages hold 16 tails of 16 blocks' 3 window pages, Inkling's 1,600 pages 10 of 32 blocks' 5.
void testCachedStatesKeepTheirRoomBesideAnExactKv() {
  for (const auto &[capped, states, capacityPages] : {std::tuple{kFlint, 16u, 768u}, {kInkling2Mini, 10u, 1'600u}}) {
    const Target &target = capped.target;
    std::string refusal;
    const auto [result, budget] = capped.planned(capped.capMiB * kMiB, capped.stateBytes, refusal);
    const auto [without, today] = capped.planned(capped.capMiB * kMiB, 0, refusal);
    require(result.plan && budget && without.plan && today, std::string(target.name) + " refused: " + refusal);
    const StreamedMemoryPlan &plan = *result.plan;
    const uint64_t room = states * capped.stateBytes;
    require(plan.expertBytes && plan.expertBytes < target.experts.allBytes && budget->kvCapacityPages == capacityPages &&
                capacityPages / (target.layout.windowPages() * target.layout.windowUnitPages()) == states,
            std::string(target.name) + ": the plan is not the one /status reports, or its tails are miscounted");
    require(plan.cachedStatesBytes == room && budget->cachedStatesBytes == room &&
                budget->dynamicBudgetBytes >= budget->laneStateBytes + budget->kvCapacityBytes + room,
            std::string(target.name) + ": the K/V reaches its capacity only with fewer cached states held than its "
                                       "capacity holds the tails of");
    require(budget->kvCapacityPages == today->kvCapacityPages && budget->kvCapacityTokens == today->kvCapacityTokens &&
                budget->contextBounds.resident == today->contextBounds.resident &&
                without.plan->expertBytes - plan.expertBytes == room && !without.plan->cachedStatesBytes,
            std::string(target.name) + ": the states' room moved the K/V or the context, or took other than itself "
                                       "from the experts");
  }
  // Without a window a state pins no K/V: no room.
  require(!kvCachedStates(k35BQ2.layout, 4'096), "a layout without a window kept room for cached states");
}

// A streamed request's allowance buys the capacity its requested context needs as the engine's plan finds it,
// its lookup index priced as that plan prices it (kvPagesCost): the context served is at least the one
// requested by construction, with no slack from the store's rounding to its slots. Priced for the blocks of
// its budget pages instead, Inkling-2-Mini's allowance at 19222M fell a share of a block's index short of the
// 1,600 pages it was to buy: the engine's plan found 1,599, whose largest whole extents are 1,536 pages, 8,192
// tokens, a context of 8,177 against the 8,192 requested. At every cap a store's slot apart from 300 MiB below
// to 300 MiB above each target's capped setting, the plan either refuses, saying what fits, or serves at least
// the requested context.
void testAStreamedRequestIsServedTheContextItAsked() {
  for (const CappedTarget &capped : {kFlint, kInkling2Mini}) {
    const uint64_t setting = capped.capMiB * kMiB;
    uint32_t served = 0, refused = 0;
    for (uint64_t cap = setting - 300 * kMiB; cap <= setting + 300 * kMiB; cap += capped.slotBytes) {
      std::string refusal;
      const auto [result, budget] = capped.planned(cap, capped.stateBytes, refusal);
      if (!budget) {
        require(!refusal.empty(), std::string(capped.target.name) + " refused a cap without saying what fits");
        ++refused;
        continue;
      }
      require(budget->contextBounds.maximum() >= capped.context,
              std::string(capped.target.name) + " at " + std::to_string(cap) + " bytes serves " +
                  std::to_string(budget->contextBounds.maximum()) + " tokens of the " + std::to_string(capped.context) +
                  " requested");
      ++served;
    }
    require(served > 100, std::string(capped.target.name) + " served too few caps of the sweep to judge it");
    (void)refused;
  }
}

// The 35B's context never shrinks as the budget grows: its experts load
// whole only where that leaves the first 64K tokens, and otherwise stream
// and keep them. Budgets from 12 GiB to the M6's with UD-Q2_K_XL, and from
// 20 GiB to the Studio's with UD-Q4_K_M, in steps of 16 MiB, cross from
// streamed experts to whole ones.
void testDense35BContextGrowsWithTheBudget() {
  const std::tuple<Mac, Target, uint64_t> sweeps[] = {{kM6, k35BQ2, 12 * kGiB}, {kStudio, k35BQ4, 20 * kGiB}};
  for (const auto &[mac, target, lowest] : sweeps) {
    uint32_t previous = 0;
    bool streams = false, whole = false;
    for (uint64_t budget = lowest; budget <= hardBudget(mac); budget += 16 * kMiB) {
      StreamedMemoryInputs inputs = target.inputs(mac);
      inputs.hardBudgetBytes = budget;
      const StreamedMemoryPlan plan = requirePlan(inputs);
      const uint32_t context =
          test::requireMemoryPlan(deviceOf(mac), target.profile(plan.expertBytes), budget).maximumContextTokens();
      require(context >= 65'536 && context >= previous,
              std::string("the 35B's context on the ") + mac.name + " fell to " + std::to_string(context) +
                  " tokens at a budget of " + std::to_string(budget / kMiB) + " MiB");
      previous = context;
      (plan.expertBytes ? streams : whole) = true;
    }
    require(streams && whole,
            std::string("the budgets on the ") + mac.name + " did not span both plans of the 35B's experts");
  }
}

// The arenas are those of the parts that stream: each plan is the one made
// with the arenas of what it streams, the routed experts' prefill groups and
// the table's staging. Beside Flash-Next's experts loaded through a store of
// 12 GiB, a table no larger than its share, 200 MiB, from a budget that
// streams it to one that loads it whole, with a staging of 512 MiB; and
// Flash-Next with experts the plan sizes from one that loads them whole to
// one that streams them.
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
  inputs.experts.setting = {StreamedSetting::Mode::Loaded, 12 * kGiB};
  inputs.table.allBytes = 200 * kMiB;
  const uint64_t group = arenaBytes(kFlashNext.group), staged = group + 512 * kMiB;
  inputs.arenaBytes[1] = {group, staged};
  bool whole = false, streams = false;
  for (inputs.hardBudgetBytes = 20 * kGiB; inputs.hardBudgetBytes < 24 * kGiB; inputs.hardBudgetBytes += kGiB / 16) {
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

// An explicit size of the routed experts is their store, exactly, which the
// floors charge: its cap and its floor both, whether or not memory holds
// every expert (12 GiB of Flash-Next's on the Studio), and a size beyond
// what the floors leave is refused as floors the budget cannot hold. A size
// that holds every expert loads them, as off does, and one below their
// smallest store is refused by name. A loaded store is exactly its size.
// --sparse-kv-memory raises the hot cache's target, and takes from the
// experts only what it adds to the readers, which hold the larger of it and
// the history staging (a buffer of 0.98 GiB on the Studio), and to their
// tables: the hot cache's, an entry for each page of its target, and that of
// the second staging buffer a target past two buffers holds; it is refused
// below its floor and past what the floors leave. An automatic part whose
// smallest store takes all it would loads whole.
void testOverridesAndRefusals() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  const StreamedMemoryPlan automaticExperts = requirePlan(inputs);
  require(automaticExperts.expertBytes > 12 * kGiB && !automaticExperts.expertFloorBytes,
          "Flash-Next's automatic store on the Studio is not above 12 GiB with the store's own floor");
  inputs.experts.setting = {StreamedSetting::Mode::Size, 12 * kGiB};
  const StreamedMemoryPlan explicitExperts = requirePlan(inputs);
  require(explicitExperts.expertBytes == 12 * kGiB && explicitExperts.expertFloorBytes == 12 * kGiB &&
              explicitExperts.tableBytes == kFlashNextTableShare,
          "an explicit --expert-memory below the automatic store was not the store's cap and floor");
  inputs.experts.setting = {StreamedSetting::Mode::Size, 24 * kGiB};
  const StreamedMemoryResult beyond = planStreamedMemory(inputs);
  require(!beyond.plan && beyond.requiredBytes > hardBudget(kStudio),
          "an --expert-memory beyond what the floors leave was planned");
  inputs.experts.setting = {StreamedSetting::Mode::Loaded, 12 * kGiB};
  const StreamedMemoryPlan loaded = requirePlan(inputs);
  require(loaded.expertBytes == 12 * kGiB && !loaded.expertFloorBytes, "a loaded store was not exactly its size");
  inputs.experts.setting = {StreamedSetting::Mode::Size, kFlashNext.experts.allBytes};
  require(!planStreamedMemory(inputs).plan, "experts that do not fit loaded whole");
  inputs.experts.setting = {StreamedSetting::Mode::Size, kGiB};
  const StreamedMemoryResult below = planStreamedMemory(inputs);
  require(!below.plan && below.error.find("--expert-memory of 1024 MiB is less than its smallest store, 2560 MiB") !=
                             std::string::npos,
          "an --expert-memory below its smallest store was not refused by name");

  StreamedMemoryInputs dense = k35BQ2.inputs(kM5Pro);
  for (const StreamedSetting setting :
       {StreamedSetting{StreamedSetting::Mode::Off, 0}, StreamedSetting{StreamedSetting::Mode::Size, 20 * kGiB}}) {
    dense.experts.setting = setting;
    require(!requirePlan(dense).expertBytes, "off, or a store that holds every expert, did not load them whole");
  }
  dense.experts.setting = {StreamedSetting::Mode::Size, 4 * kGiB};
  const StreamedMemoryPlan sized = requirePlan(dense);
  require(sized.expertBytes == 4 * kGiB && sized.expertFloorBytes == 4 * kGiB,
          "a size did not stream the experts memory holds every one of through a store of that size");
  dense.hardBudgetBytes = 10 * kGiB;
  const StreamedMemoryPlan smaller = requirePlan(dense);
  require(smaller.expertBytes == 4 * kGiB && smaller.expertFloorBytes == 4 * kGiB,
          "a size did not stream the experts a smaller budget cannot hold through a store of that size");
  StreamedMemoryInputs tight = k35BQ2.inputs(kM6);
  tight.hardBudgetBytes = 12 * kGiB;
  tight.experts.minimumBytes = tight.experts.allBytes;
  const StreamedMemoryResult whole = planStreamedMemory(tight);
  require(!whole.plan && whole.requiredBytes > 12 * kGiB,
          "experts whose smallest store takes all they would streamed instead of loading whole");

  inputs = kFlashNext.inputs(kStudio);
  const StreamedMemoryPlan automatic = requirePlan(inputs);
  const uint64_t staging = automatic.spill.stagingBytes;
  require(staging > 900 * kMiB && automatic.spill.readerBytes() == staging,
          "the Studio's readers were not its history staging");
  inputs.hotCacheBytes = 900 * kMiB;
  const StreamedMemoryPlan within = requirePlan(inputs);
  inputs.hotCacheBytes = staging + kGiB;
  const StreamedMemoryPlan raised = requirePlan(inputs);
  require(within.spill.hotTargetBytes == 900 * kMiB && within.expertBytes == automatic.expertBytes &&
              within.spill.readerBytes() == staging && raised.spill.hotTargetBytes == staging + kGiB &&
              raised.spill.hotFloorBytes == automatic.spill.hotFloorBytes &&
              raised.spill.tableBytes > automatic.spill.tableBytes &&
              automatic.expertBytes - raised.expertBytes ==
                  kGiB + raised.spill.tableBytes - automatic.spill.tableBytes &&
              raised.spill.readerBytes() == staging + kGiB,
          "--sparse-kv-memory did not take from the experts exactly what it adds to the readers and their tables");
  // Below its floor, or beyond what the floors leave, it is refused by name
  // as a setting, never as floors the budget cannot hold.
  for (const auto &[bytes, refusal] : {std::pair<uint64_t, std::string>{300 * kMiB, "is less than"},
                                       std::pair<uint64_t, std::string>{24 * kGiB, "MiB the floors leave"}}) {
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

// A target spills only with a spillable layout that declares its sparse
// attention and an SSD cache that takes writes, and only where its floors
// hold an empty Spill request; otherwise it plans as a dense target.
void testSpillNeedsATierAndRoom() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  inputs.tierWritable = false;
  require(!requirePlan(inputs).spill.enabled, "a target spilled without an SSD cache that takes writes");
  inputs = kFlashNext.inputs(kStudio);
  inputs.sparseKv.reset();
  require(!requirePlan(inputs).spill.enabled, "a target spilled without declaring its sparse attention");
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
// runs prefill groups and holds where its verifies' blocks are, those of
// what the plan streams: beside streamed experts what they add comes out of
// the experts' store, and where the floors hold an empty Spill request only
// beside the arenas of a runtime whose K/V stays in memory, the plan does
// not spill.
void testSpillHoldsTheArenasOfASpillingRuntime() {
  constexpr uint64_t kExtra = 256 * kMiB;
  StreamedMemoryInputs inputs = kFlashNext.inputs(kStudio);
  const StreamedMemoryPlan resident = requirePlan(inputs);
  const uint64_t group = arenaBytes(kFlashNext.group);
  // The arenas of a spilling runtime whose experts stream, which Flash-Next's
  // plan streams on the Studio, hold more; those of one whose experts load
  // whole do not count.
  inputs.spillArenaBytes = {{{group, group}, {group + kExtra, group + kExtra}}};
  const StreamedMemoryPlan spilling = requirePlan(inputs);
  require(resident.spill.enabled && spilling.spill.enabled && resident.expertBytes - spilling.expertBytes == kExtra,
          "a plan that spills did not hold the arenas of a runtime whose K/V spills and whose experts stream");
  inputs.spillArenaBytes = {{{group + kExtra, group + kExtra}, {group, group}}};
  require(requirePlan(inputs).expertBytes == resident.expertBytes,
          "a plan whose experts stream held the arenas of a spilling runtime whose experts load whole");

  inputs = kFlashNext.inputs(kM6);
  inputs.experts.setting = inputs.table.setting = {StreamedSetting::Mode::Off, 0};
  const uint64_t chunk = arenaBytes(kFlashNext.chunk);
  inputs.spillArenaBytes.fill({chunk, chunk});
  const uint64_t base = inputs.fixedBytes + chunk + inputs.laneStateBytes +
                        256 * (test::sparseKvLayout().bytesPerModelPage() + 32 * model::kLookupIndexTokenBytes) +
                        kFlashNext.experts.allBytes + kFlashNext.table.allBytes;
  for (inputs.hardBudgetBytes = base; !requirePlan(inputs).spill.enabled; inputs.hardBudgetBytes += 16 * kMiB)
    require(inputs.hardBudgetBytes < base + 4 * kGiB, "no budget let the target spill");
  inputs.spillArenaBytes.fill({chunk + kExtra, chunk + kExtra});
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

// A second staging buffer goes where the readers' memory holds it already,
// and costs the plan only its page table then: on a tight M6 with a
// 128K-token context, whose two buffers the hot cache's floor holds, also
// where the limit holds the stores at their smallest, and beside a hot cache
// target that holds them. Otherwise one buffer where a store binds, which
// keeps the second's room, also where the table's share above its smallest
// would hold it (the M6 at a 384K-token context, whose two buffers exceed the
// hot block cache's floor), and two where none does, the second taking its
// room from the KV kept resident beyond the first 64K tokens: on the M5 Pro
// under --max-memory 44G, which loads the experts whole.
void testSecondStagingBufferOnlyWhereFree() {
  StreamedMemoryInputs inputs = kFlashNext.inputs(kM6);
  inputs.hardBudgetBytes = 14 * kGiB;
  const StreamedMemoryPlan tight = requirePlan(inputs);
  require(tight.spill.stagingBuffers == 1 && tight.expertBytes >= kFlashNext.experts.minimumBytes,
          "a tight budget staged in two buffers");
  const auto free = [&](const StreamedMemoryInputs &bounded) {
    const KvSpillPlan spill = requirePlan(bounded).spill;
    return spill.stagingBuffers == 2 && spill.readerBytes() == spill.hotTargetBytes;
  };
  StreamedMemoryInputs context = inputs;
  context.requestedContextTokens = 131'072;
  StreamedMemoryInputs smallest = context;
  smallest.hardBudgetBytes = 10'752 * kMiB;
  const StreamedMemoryPlan least = requirePlan(smallest);
  StreamedMemoryInputs target = inputs;
  target.hotCacheBytes = 1'200 * kMiB;
  require(least.expertBytes == kFlashNext.experts.minimumBytes && least.tableBytes == kFlashNext.table.minimumBytes &&
              free(context) && free(smallest) && free(target),
          "a second staging buffer the readers' memory holds was not planned");
  StreamedMemoryInputs longer = kFlashNext.inputs(kM6);
  longer.requestedContextTokens = 393'216;
  const KvSpillPlan bound = requirePlan(longer).spill;
  require(bound.stagingBuffers == 1 && bound.readerBytes() == bound.hotTargetBytes,
          "the M6 at a 384K-token context staged a second buffer in the table's share");
  StreamedMemoryInputs capped = kFlashNext.inputs(kM5Pro);
  capped.hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(kM5Pro.workingSet, 44 * kGiB, kM5Pro.memory);
  const StreamedMemoryPlan whole = requirePlan(capped);
  require(!whole.expertBytes && whole.tableBytes == kFlashNextTableShare && whole.spill.stagingBuffers == 2 &&
              whole.spill.readerBytes() > whole.spill.hotTargetBytes,
          "Flash-Next on the M5 Pro under --max-memory 44G, its experts whole, did not stage in two buffers");
}

// The plan that serves charges the stores at what they hold: deciding its
// staging beside them, it decides as the plan that sized them did on each
// Mac.
void testServedPlanStagesAsTheSizedPlan() {
  const auto servedOf = [](const StreamedMemoryInputs &inputs, const StreamedMemoryPlan &sized) {
    StreamedMemoryInputs served = inputs;
    served.experts.setting = sized.expertBytes ? StreamedSetting{StreamedSetting::Mode::Loaded, sized.expertBytes}
                                               : StreamedSetting{StreamedSetting::Mode::Off, 0};
    served.table.setting = {StreamedSetting::Mode::Loaded, sized.tableBytes};
    return served;
  };
  for (const Mac &mac : kMacs) {
    const StreamedMemoryInputs inputs = kFlashNext.inputs(mac);
    const StreamedMemoryPlan sized = requirePlan(inputs);
    require(requirePlan(servedOf(inputs, sized)).spill.stagingBuffers == sized.spill.stagingBuffers,
            std::string("the plan that serves Flash-Next on the ") + mac.name +
                " staged otherwise than the plan that sized its stores");
  }
}

// The floor of the routed experts' store: every slot when the run can
// spill, else kExpertFloorShare of its slots, and all of a store
// --expert-memory sizes, which is its cap too, spilling or not. The engine's
// plan holds the store at its cap, the chunks above its floor it starts
// without among the fixed bytes, so that the context, one request's KV
// capacity and the KV admission keeps resident are the limit's. Flash-Next
// on the M6, with and without an SSD cache that takes writes
// (--max-cache-disk 0).
void testPlanSetsTheStoresFloor() {
  const StreamedMemoryInputs inputs = kFlashNext.inputs(kM6);
  const StreamedMemoryPlan spilling = requirePlan(inputs);
  require(spilling.spill.enabled && spilling.expertBytes && !spilling.expertFloorBytes &&
              spilling.expertFloorShare == 1.0,
          "a store that can spill does not keep every slot as its floor");
  StreamedMemoryInputs unwritable = inputs;
  unwritable.tierWritable = false;
  const StreamedMemoryPlan limit = requirePlan(unwritable);
  require(!limit.spill.enabled && limit.expertBytes && !limit.expertFloorBytes &&
              limit.expertFloorShare == model::kExpertFloorShare,
          "a store that cannot spill does not keep kExpertFloorShare of its slots");
  for (StreamedMemoryInputs sized : {inputs, unwritable}) {
    sized.experts.setting = {StreamedSetting::Mode::Size, 8 * kGiB};
    const StreamedMemoryPlan exact = requirePlan(sized);
    require(exact.expertBytes == 8 * kGiB && exact.expertFloorBytes == 8 * kGiB && exact.expertFloorShare == 1.0,
            std::string("--expert-memory did not set the store's cap and floor ") +
                (sized.tierWritable ? "where the run can spill" : "without an SSD cache that takes writes"));
  }

  const EngineMemoryPlan mains = test::requireMemoryPlan(
      deviceOf(kM6), kFlashNext.profile(limit.expertBytes, limit.tableBytes, limit.spill));
  const uint64_t floor = static_cast<uint64_t>(model::kExpertFloorShare * double(limit.expertBytes));
  const uint64_t heldBack = limit.expertBytes - floor;
  ModelMemoryProfile profile = kFlashNext.profile(floor, limit.tableBytes, limit.spill);
  profile.footprint.hostHeldBackBytes = heldBack;
  const EngineMemoryPlan held = test::requireMemoryPlan(deviceOf(kM6), profile);
  const EngineMemoryBreakdown &budget = held.breakdown();
  require(held.maximumContextTokens() == mains.maximumContextTokens() &&
              budget.dynamicBudgetBytes == mains.breakdown().dynamicBudgetBytes &&
              budget.kvCapacityTokens == mains.breakdown().kvCapacityTokens &&
              budget.hostHeldBackBytes == heldBack &&
              test::jsonOf(held).find("\"host_held_back_bytes\":" + std::to_string(heldBack) +
                                       ",\"fixed_runtime_bytes\"") != std::string::npos &&
              budget.describe().find("expert store above its floor at the start: " + std::to_string(heldBack)) !=
                  std::string::npos,
          "the store's chunks above its floor moved the context or the KV the plan holds");
}

// The plan serves what spilling serves, the window where memory and the SSD
// cache's room hold it, and admission's geometry follows the plan. The M5
// Pro, whose experts load whole, holds one request's KV of the whole window
// in memory too, since the table takes its share rather than 512 MiB.
void testEnginePlanServesTheSpilledContext() {
  for (const Mac &mac : kMacs) {
    const StreamedMemoryPlan streamed = requirePlan(kFlashNext.inputs(mac));
    const EngineMemoryPlan plan = test::requireMemoryPlan(
        deviceOf(mac), kFlashNext.profile(streamed.expertBytes, streamed.tableBytes, streamed.spill));
    const ContextBounds &bounds = plan.breakdown().contextBounds;
    require(bounds.window == 1'010'000 && bounds.spilled == 1'010'000 && bounds.disk == streamed.spill.diskTokens &&
                bounds.resident >= 65'536 &&
                (streamed.expertBytes ? bounds.resident < bounds.disk : bounds.resident == bounds.window) &&
                plan.maximumContextTokens() == std::min(bounds.spilled, bounds.disk),
            std::string("Flash-Next's context on the ") + mac.name + " is not what spilling serves");
    require(test::jsonOf(plan).find("\"context_bounds\":{\"window\":1010000") != std::string::npos &&
                test::jsonOf(plan).find("\"kv_spill\":{\"dense_pages\":64") != std::string::npos,
            "the memory status does not report the spill plan");
    const SpillGeometry geometry = plan.spillGeometry();
    require(geometry.fullPageBytes == 423'936 && geometry.indexPageBytes == 24'576 &&
                geometry.slotBytes == 425'984 && geometry.densePages == 64 && geometry.windowPages == 64 &&
                geometry.groupPages == 256 && geometry.residentTokens == bounds.resident,
            "admission's geometry is not the plan's");
  }
  const SpillGeometry geometry{.fullPageBytes = 423'936, .indexPageBytes = 24'576, .densePages = 64,
                               .windowPages = 64, .groupPages = 256};
  require(geometry.residentBytes(1) == 423'936 && geometry.residentBytes(32 * 1'000) == 1'000 * 423'936 &&
              geometry.spillFloorBytes(32 * 100) == 100 * 423'936 &&
              geometry.spillFloorBytes(32 * 1'000) == 640 * 423'936 + 360 * 24'576,
          "a request's KV is not its pages, Full or Index");
}

// The resident context is the one a lone start takes: one request's KV in
// whole extents beside its lane's state and what a target that spills holds
// from its start whatever its requests, the Index runway and the readers'
// page tables (KvSpillPlan::heldBytes), which runtime-resources finds as the
// start allocates them. Across the budgets that leave the room past eight
// extents beside the state anything from nothing to an extent less a byte,
// Flash-Next keeps its eighth extent exactly where that room holds what it
// holds: beside the state alone, the plan advertised the extent there too,
// which no lone start could take, wherever the room was short of the
// 6,881,280 bytes it holds with two staging buffers.
void testResidentContextIsWhatALoneStartTakes() {
  const kv::Layout layout = test::sparseKvLayout();
  for (const Mac &mac : {kStudio, kM5Pro}) {
    const StreamedMemoryPlan streamed = requirePlan(kFlashNext.inputs(mac));
    const KvSpillPlan &spill = streamed.spill;
    const uint64_t held = spill.heldBytes();
    const ModelMemoryProfile profile = kFlashNext.profile(streamed.expertBytes, streamed.tableBytes, spill);
    // An extent's pages with their tokens' lookup index.
    const uint64_t extent = 256 * (layout.bytesPerModelPage() + 32 * model::kLookupIndexTokenBytes);
    const uint64_t eight = profile.fixedRuntimeBytes() + kFlashNext.chunk.laneStatePlannedAllocatedBytes + 8 * extent;
    for (const uint64_t room : {uint64_t{0}, held - 1, held, extent / 2, extent - 1}) {
      const EngineMemoryBreakdown budget =
          test::requireMemoryPlan(deviceOf(mac), profile, eight + room).breakdown();
      const uint64_t extents = room < held ? 7 : 8;
      require(budget.hardBudgetBytes == eight + room &&
                  budget.contextBounds.resident == extents * 8'192 - model::ExecutionLimits::speculativeScratchTokens,
              std::string("Flash-Next on the ") + mac.name + " with " + std::to_string(room) +
                  " bytes past eight extents beside its state advertised " +
                  std::to_string(budget.contextBounds.resident) + " resident tokens, not what a lone start takes");
    }
  }
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
    testTheContextBesideMoreIsTheCapacityLessWhatItTakes();
    testTheRequestsRunAtOnceAreTheLanesWhoseStatesFitBesideOne();
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
    testTheAutomaticBudgetLeavesATenthUnwired();
    testModelProvidedKvGeometry();
    testWindowedKvCostsItsFloorWhole();
    testExtentSizeFollowsThePool();
    testHotBlockCacheIsDefinedOnce();
    testFlashNextOnTheThreeMacs();
    testAutoStreamsTheTableWhereMemoryHoldsIt();
    testSpillAllowanceFollowsASmallerRequestedContext();
    testDense35BKeepsMainsPlan();
    testEightLanesArePlannedAsEight();
    testDenseExplicitContextKeepsMainsExperts();
    testDenseExplicitContextStreamsExperts();
    testAWindowedRequestGetsItsKvAndTheExpertsTheRest();
    testCachedStatesKeepTheirRoomBesideAnExactKv();
    testAStreamedRequestIsServedTheContextItAsked();
    testDense35BContextGrowsWithTheBudget();
    testArenasFollowWhatStreams();
    testOverridesAndRefusals();
    testSpillNeedsATierAndRoom();
    testSpillHoldsTheArenasOfASpillingRuntime();
    testDiskBoundsTheSpilledContext();
    testSecondStagingBufferOnlyWhereFree();
    testServedPlanStagesAsTheSizedPlan();
    testPlanSetsTheStoresFloor();
    testEnginePlanServesTheSpilledContext();
    testResidentContextIsWhatALoneStartTakes();
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
