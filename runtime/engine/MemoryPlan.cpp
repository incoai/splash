#include "engine/MemoryPlan.hpp"
#include "Checked.hpp"
#include "engine/Json.hpp"
#include "metal/abi/PagedAttention.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::engine {
namespace {

// The context a spill-capable target keeps resident before the streamed
// stores take the rest of the budget, and a dense target too once its
// experts stream: requests up to it never spill while they decode alone.
constexpr uint64_t kResidentAllowanceTokens = 65'536;
// The streamed PLE table's row cache before the experts take the rest. It
// hits only rows read before, and a request reads few distinct rows: a
// 128K-token prompt about 1.1M, a quarter of what 512 MiB holds. A larger
// cache hits no more often and leaves streamed experts less.
constexpr uint64_t kTableShareBytes = 512 * kMiB;
// A Spill request's tail window: the pages before its open page that it
// keeps Full. A decode selects about a tenth of its blocks in its last 64
// pages.
constexpr uint32_t kSpillWindowPages = 64;
// A prefill's history stages in two buffers, so the next layer's reads
// overlap this layer's attention, where the floors leave this much.
constexpr uint64_t kDoubleStagingBytes = 4 * kGiB;

uint64_t pagesOf(uint64_t tokens) noexcept { return (tokens + kv::kPageTokens - 1) / kv::kPageTokens; }

uint64_t remaining(uint64_t bytes, uint64_t taken) noexcept { return bytes > taken ? bytes - taken : 0; }

// The KV of a context of `tokens` tokens holds its speculative rows too.
uint64_t kvTokens(uint64_t tokens) noexcept { return tokens + model::ExecutionLimits::speculativeScratchTokens; }

// `pages` pages of `pageBytes` bytes in whole extents of `extentPages`.
uint64_t extentsBytes(uint64_t pages, uint64_t extentPages, uint64_t pageBytes) noexcept {
  return (pages + extentPages - 1) / extentPages * extentPages * pageBytes;
}

// A streamed part as its setting leaves it to the planner: the bytes the
// floors charge it, and all of it; an Auto part streams through more than
// its floor or loads whole as the budget allows, any other only as set. An
// Auto part whose smallest store takes all it would loads whole.
struct Part final {
  uint64_t floor = 0;
  uint64_t all = 0;
  bool automatic = false;
  bool streams = false;
};

std::optional<Part> resolve(const StreamedPart &part, std::string_view option, std::string &error) {
  if (!part.allBytes) return Part{};
  switch (part.setting.mode) {
  case StreamedSetting::Mode::Auto:
    if (part.floorBytes < part.allBytes) return Part{part.floorBytes, part.allBytes, true, false};
    [[fallthrough]];
  case StreamedSetting::Mode::Off:
    return Part{part.allBytes, part.allBytes, false, false};
  case StreamedSetting::Mode::Size:
    if (part.setting.bytes >= part.allBytes) return Part{part.allBytes, part.allBytes, false, false};
    if (part.setting.bytes < part.floorBytes) {
      error = std::string(option) + " of " + std::to_string(part.setting.bytes / kMiB) +
              " MiB is less than its smallest store, " + std::to_string((part.floorBytes + kMiB - 1) / kMiB) +
              " MiB";
      return std::nullopt;
    }
    return Part{part.setting.bytes, part.setting.bytes, false, true};
  }
  return Part{};
}

// What a spill-capable target's floors take, by the context they hold a
// Spill request of: the request's pages, the Index runway, and the larger of
// the hot block cache's floor and `buffers` buffers of staged history.
struct SpillFloors final {
  kv::Layout layout;
  SpillGeometry geometry;
  uint64_t extentPages = 0;
  uint64_t hotFloorBytes = 0;
  uint64_t indexRunwayBytes = 0;

  [[nodiscard]] uint64_t stagingBytes(uint64_t tokens, uint32_t buffers) const noexcept {
    return buffers * extentsBytes(pagesOf(kvTokens(tokens)), extentPages, layout.bytesPerLayerPage());
  }
  [[nodiscard]] uint64_t bytes(uint64_t tokens, uint32_t buffers) const noexcept {
    return geometry.spillFloorBytes(kvTokens(tokens)) + indexRunwayBytes +
           std::max(stagingBytes(tokens, buffers), hotFloorBytes);
  }
};

// A spill-capable target's spill plan beside floors of `floors` bytes; a
// disabled one for a target that cannot spill, or whose floors cannot hold
// even an empty Spill request, which then serves what stays resident. A
// target whose rows read their last tokens through their page tables
// (SparseKvShape::localTokens) cannot spill: the cache keeps a page Full for
// rows that read it whole only within the dense history, and a restore or a
// writer's move onto a spilled block may leave such a page in a holder's
// window holding only its index.
KvSpillPlan planSpill(const StreamedMemoryInputs &in, uint64_t floors) {
  KvSpillPlan spill;
  const kv::Layout &layout = in.kvLayout;
  if (!in.sparseKv || in.sparseKv->localTokens || !layout.spillable() || !in.tierWritable) return spill;
  const model::SparseKvShape &shape = *in.sparseKv;
  const uint32_t extentPages = layout.maximumExtentPages();
  const SpillFloors spillFloors{
      .layout = layout,
      .geometry = {.fullPageBytes = layout.bytesPerModelPage(),
                   .indexPageBytes = layout.indexBytesPerModelPage(),
                   .densePages = static_cast<uint32_t>(pagesOf(shape.denseTokens())),
                   .windowPages = kSpillWindowPages,
                   .groupPages = in.groupPages},
      .extentPages = extentPages,
      .hotFloorBytes =
          extentsBytes(shape.requiredSlots(1) >> SPLASH_ATTENTION_LOCATION_BLOCK_BITS, extentPages,
                       layout.kvBytesPerModelPage()),
      .indexRunwayBytes = uint64_t{extentPages} * layout.indexBytesPerModelPage()};
  const uint64_t room = remaining(in.hardBudgetBytes, floors);
  const auto fits = [&](uint32_t tokens) { return spillFloors.bytes(tokens, 1) <= room; };
  if (!fits(0)) return spill;
  // The longest context the floors hold, and the longest the SSD cache's
  // room for active KV holds beside every lane's state.
  uint32_t low = 0, high = in.windowTokens;
  while (low < high) {
    const uint32_t middle = high - (high - low) / 2;
    if (fits(middle))
      low = middle;
    else
      high = middle - 1;
  }
  const uint64_t states = uint64_t{model::ExecutionLimits::maximumBatchWidth} * in.stateSlotBytes;
  spill.enabled = true;
  spill.memoryTokens = low;
  // The room only bounds a first start's context: it follows the volume's
  // free space, and a relaunch keeps its first engine's context, which each
  // request's commitment checks against the room as it starts.
  spill.diskTokens = static_cast<uint32_t>(std::min<uint64_t>(
      in.windowTokens,
      std::max<uint64_t>(in.relaunchContextTokens,
                         in.kvSlotBytes ? remaining(in.activeLimitBytes, states) / in.kvSlotBytes * kv::kPageTokens
                                        : 0)));
  spill.floorTokens = std::min({in.requestedContextTokens ? in.requestedContextTokens : in.windowTokens,
                                spill.memoryTokens, spill.diskTokens});
  spill.stagingBuffers = room >= spillFloors.bytes(spill.floorTokens, 1) + kDoubleStagingBytes ? 2 : 1;
  spill.densePages = spillFloors.geometry.densePages;
  spill.windowPages = spillFloors.geometry.windowPages;
  spill.groupPages = in.groupPages;
  spill.slotBytes = in.kvSlotBytes;
  spill.laneSlots = shape.laneSlots();
  spill.hotFloorBytes = spill.hotTargetBytes = spillFloors.hotFloorBytes;
  spill.hotLimitBytes = extentsBytes(
      shape.requiredSlots(model::ExecutionLimits::maximumBatchWidth) >> SPLASH_ATTENTION_LOCATION_BLOCK_BITS,
      extentPages, layout.kvBytesPerModelPage());
  spill.indexRunwayBytes = spillFloors.indexRunwayBytes;
  spill.stagingBytes = spillFloors.stagingBytes(spill.floorTokens, spill.stagingBuffers);
  spill.floorBytes = spillFloors.bytes(spill.floorTokens, spill.stagingBuffers);
  return spill;
}

// The plan of the parts as resolved beside arenas of `arenaBytes`, or of
// `spillArenaBytes` where the plan spills (planStreamedMemory).
StreamedMemoryResult planParts(const StreamedMemoryInputs &in, const Part &experts, const Part &table,
                               uint64_t arenaBytes, uint64_t spillArenaBytes) {
  const kv::Layout &layout = in.kvLayout;
  const auto overflow = [] { return StreamedMemoryResult{std::nullopt, "the memory plan overflows uint64"}; };
  // The floors: what loads, the arenas and the stores at their floors, a
  // lane's state and the KV runway, and a spill-capable target's spill
  // floors.
  uint64_t loaded = 0;
  for (const uint64_t bytes : {in.fixedBytes, arenaBytes, experts.floor, table.floor})
    if (!checkedAdd(loaded, bytes, loaded)) return overflow();
  const std::optional<uint64_t> minimum = minimumRequiredBytes(loaded, in.laneStateBytes, layout);
  if (!minimum) return overflow();
  const uint64_t runwayBytes = *minimum - loaded - in.laneStateBytes;
  uint64_t floors = *minimum;
  // A plan that spills holds the arenas of a runtime whose K/V spills
  // instead, at least as large, beside which its spill floors must fit.
  uint64_t spillFloors = 0;
  if (!checkedAdd(floors, remaining(spillArenaBytes, arenaBytes), spillFloors)) return overflow();
  StreamedMemoryPlan plan{.spill = planSpill(in, spillFloors)};
  KvSpillPlan &spill = plan.spill;
  if (spill.enabled) floors = spillFloors;
  if (!checkedAdd(floors, spill.floorBytes, floors)) return overflow();
  if (floors > in.hardBudgetBytes)
    return {std::nullopt,
            "the floors of the memory plan require " + std::to_string(floors) +
                " bytes but the Metal memory budget is " + std::to_string(in.hardBudgetBytes) + " bytes",
            floors};
  // An explicit hot cache target takes its part above the floor first, of
  // what the floors leave.
  if (spill.enabled && in.hotCacheBytes) {
    const uint64_t target = *in.hotCacheBytes;
    if (target < spill.hotFloorBytes)
      return {std::nullopt, "--sparse-kv-memory of " + std::to_string(target / kMiB) +
                                " MiB is less than the hot block cache's floor, " +
                                std::to_string(spill.hotFloorBytes / kMiB) + " MiB"};
    if (target - spill.hotFloorBytes > in.hardBudgetBytes - floors)
      return {std::nullopt, "--sparse-kv-memory of " + std::to_string(target / kMiB) +
                                " MiB is more than the hot block cache's floor, " +
                                std::to_string(spill.hotFloorBytes / kMiB) + " MiB, and the " +
                                std::to_string((in.hardBudgetBytes - floors) / kMiB) + " MiB the floors leave"};
    spill.hotTargetBytes = target;
    spill.hotLimitBytes = std::max(spill.hotLimitBytes, target);
    floors += target - spill.hotFloorBytes;
  }

  // Then the priorities, out of what the floors leave: the experts first,
  // then the table. The KV allowance a part leaves holds the context kept
  // resident, with its speculative rows, as one request's KV capacity does:
  // in the fewest pages whose whole extents hold its pages, of which the
  // floors hold the runway.
  int64_t rest = static_cast<int64_t>(in.hardBudgetBytes - floors);
  const auto allowance = [&](bool streaming) -> int64_t {
    const uint64_t tokens = spill.enabled ? std::min<uint64_t>(in.windowTokens, kResidentAllowanceTokens)
                            : in.requestedContextTokens ? in.requestedContextTokens
                            : streaming ? std::min<uint64_t>(in.windowTokens, kResidentAllowanceTokens)
                                        : 0;
    if (!tokens) return 0;
    const uint64_t pages = pagesOf(kvTokens(tokens));
    uint64_t budget = pages;
    while (kvCapacityPagesFor(layout, budget) < pages) ++budget;
    return static_cast<int64_t>(remaining(budget * layout.bytesPerModelPage(), runwayBytes));
  };
  const int64_t tableShare =
      table.automatic ? static_cast<int64_t>(remaining(std::min(table.all, kTableShareBytes), table.floor)) : 0;
  bool expertsStream = experts.streams;
  plan.expertBytes = experts.streams ? experts.floor : 0;
  if (experts.automatic) {
    const auto above = static_cast<int64_t>(experts.all - experts.floor);
    if (rest - allowance(false) - tableShare >= above) {
      rest -= above;
    } else {
      expertsStream = true;
      const int64_t bytes = std::clamp<int64_t>(rest - allowance(true) - tableShare, 0, above);
      plan.expertBytes = experts.floor + static_cast<uint64_t>(bytes);
      rest -= bytes;
    }
  }
  plan.tableBytes = table.streams ? table.floor : 0;
  if (table.automatic) {
    const int64_t kept = rest - allowance(expertsStream);
    if (kept < static_cast<int64_t>(table.all - table.floor))
      plan.tableBytes = table.floor + static_cast<uint64_t>(std::clamp<int64_t>(kept, 0, tableShare));
  }
  return {plan, {}};
}

std::string bytesAndMiB(uint64_t bytes) {
  std::ostringstream out;
  out << bytes << " bytes (" << std::fixed << std::setprecision(2)
      << static_cast<long double>(bytes) / kMiB << " MiB)";
  return out.str();
}

BudgetValidationStatus failure(BudgetErrorCode code, std::string message,
                               EngineMemoryBreakdown breakdown) {
  return {false, code, std::move(message), std::move(breakdown)};
}

std::string_view budgetErrorCodeName(BudgetErrorCode code) {
  switch (code) {
  case BudgetErrorCode::None:
    return "none";
  case BudgetErrorCode::InvalidDeviceCapabilities:
    return "invalid_device_capabilities";
  case BudgetErrorCode::InvalidModelSpec:
    return "invalid_model_spec";
  case BudgetErrorCode::WorkingSetTooSmall:
    return "working_set_too_small";
  case BudgetErrorCode::ArithmeticOverflow:
    return "arithmetic_overflow";
  case BudgetErrorCode::KvPoolDoesNotFit:
    return "kv_pool_does_not_fit";
  }
  return "unknown";
}

// A spill-capable target's plan (KvSpillPlan), as /status reports it; null
// for a target that keeps its KV in memory.
std::string spillStatusJson(const KvSpillPlan &spill) {
  if (!spill.enabled) return "null";
  std::ostringstream out;
  out << "{\"dense_pages\":" << spill.densePages
      << ",\"window_pages\":" << spill.windowPages << ",\"group_pages\":" << spill.groupPages
      << ",\"slot_bytes\":" << spill.slotBytes << ",\"lane_slots\":" << spill.laneSlots
      << ",\"hot_floor_bytes\":" << spill.hotFloorBytes << ",\"hot_target_bytes\":" << spill.hotTargetBytes
      << ",\"hot_limit_bytes\":" << spill.hotLimitBytes
      << ",\"staging_buffers\":" << spill.stagingBuffers << ",\"staging_bytes\":" << spill.stagingBytes
      << ",\"index_runway_bytes\":" << spill.indexRunwayBytes << ",\"floor_tokens\":" << spill.floorTokens
      << ",\"floor_bytes\":" << spill.floorBytes << '}';
  return out.str();
}

std::string deviceStatusJson(const DeviceCapabilities &device) {
  std::ostringstream out;
  out << '{' << "\"device_name\":" << json::quote(device.deviceName) << ','
      << "\"macos_version\":" << json::quote(device.macosVersion()) << ','
      << "\"apple_gpu_family\":" << device.appleGpuFamily << ','
      << "\"gpu_core_count\":" << device.gpuCoreCount << ','
      << "\"physical_memory_bytes\":" << device.physicalMemoryBytes << ','
      << "\"recommended_max_working_set_bytes\":"
      << device.recommendedMaxWorkingSetBytes << ','
      << "\"max_buffer_length_bytes\":" << device.maxBufferLengthBytes << ','
      << "\"max_threadgroup_memory_bytes\":"
      << device.maxThreadgroupMemoryBytes << ','
      << "\"max_threadgroup_width\":" << device.maxThreadgroupWidth << ','
      << "\"has_unified_memory\":"
      << (device.hasUnifiedMemory ? "true" : "false") << '}';
  return out.str();
}

std::string modelStatusJson(const ModelMemoryProfile &model) {
  std::ostringstream out;
  out << '{' << "\"model_name\":" << json::quote(model.name) << ','
      << "\"maximum_context_tokens\":" << model.maximumContextTokens << ','
      << "\"attention_layers\":" << model.targetKvLayout.attentionLayers << ','
      << "\"kv_heads\":" << model.targetKvLayout.kvHeads << ','
      << "\"head_dimension\":" << model.targetKvLayout.headDimension << ','
      << "\"kv_page_tokens\":" << kv::kPageTokens << ','
      << "\"kv_format\":" << json::quote(kv::formatName(model.targetKvLayout.format)) << ','
      << "\"kv_elements_per_scale\":"
      << model.targetKvLayout.elementsPerScale() << ','
      << "\"kv_page_bytes\":" << model.targetKvLayout.bytesPerModelPage() << ','
      << "\"state_format\":" << json::quote(ops::formatName(model.stateFormat)) << ','
      << "\"memory\":{" << "\"target_weights_bytes\":"
      << model.footprint.targetWeightsBytes << ','
      << "\"draft_weights_bytes\":" << model.footprint.draftWeightsBytes << ','
      << "\"vision_weights_bytes\":" << model.footprint.visionWeightsBytes << ','
      << "\"lane_state_bytes\":"
      << model.footprint.runtime.laneStatePlannedAllocatedBytes << ','
      << "\"shared_prefill_bytes\":"
      << model.footprint.runtime.sharedPrefillPlannedAllocatedBytes << ','
      << "\"shared_decode_bytes\":"
      << model.footprint.runtime.sharedDecodePlannedAllocatedBytes << ','
      << "\"pipeline_reserve_bytes\":" << model::kPipelineReserveBytes << ','
      << "\"runtime_overhead_reserve_bytes\":"
      << model::kRuntimeOverheadReserveBytes << ','
      << "\"streamed_table_bytes\":" << model.footprint.streamedTableBytes << ','
      << "\"ane_ffn_bytes\":" << model.footprint.aneFfnBytes << ','
      << "\"state_staging_bytes\":" << model.footprint.stateStagingBytes << "}}";
  return out.str();
}

} // namespace

std::optional<uint64_t> minimumRequiredBytes(uint64_t fixedBytes,
                                             uint64_t laneStateBytes,
                                             const kv::Layout &layout) noexcept {
  uint64_t runwayBytes = 0;
  uint64_t dynamicBytes = 0;
  uint64_t result = 0;
  if (!checkedMultiply(layout.bytesPerModelPage(),
                       kvRunwayPages(layout.minimumExtentPages()), runwayBytes) ||
      !checkedAdd(laneStateBytes, runwayBytes, dynamicBytes) ||
      !checkedAdd(fixedBytes, dynamicBytes, result))
    return std::nullopt;
  return result;
}

uint64_t kvCapacityPagesFor(const kv::Layout &layout, uint64_t pages) noexcept {
  const uint32_t extentPages = layout.extentPagesFor(pages);
  return extentPages ? pages - pages % extentPages : 0;
}

uint32_t ContextBounds::maximum() const noexcept {
  return std::min(window, std::max(resident, std::min(spilled, disk)));
}

std::string_view ContextBounds::bound() const noexcept {
  const uint32_t context = maximum();
  if (context == window) return "model";
  return context == resident || disk >= spilled ? "memory" : "disk";
}

StreamedMemoryResult planStreamedMemory(const StreamedMemoryInputs &in) {
  if (!in.kvLayout.valid() || !in.windowTokens || in.requestedContextTokens > in.windowTokens ||
      (in.sparseKv && !in.sparseKv->laneSlots()))
    return {std::nullopt, "invalid memory planner inputs"};
  std::string error;
  const std::optional<Part> experts = resolve(in.experts, "--expert-memory", error);
  const std::optional<Part> table = experts ? resolve(in.table, "--ple-memory", error) : std::nullopt;
  if (!table) return {std::nullopt, error};
  // The arenas of the parts set to stream, and where that plan streams
  // more, those of the parts it streams: a part that streams with smaller
  // arenas streams with larger ones, so a plan streams no more than the
  // arenas it was made with after three at most.
  bool expertsStream = experts->streams, tableStreams = table->streams;
  for (;;) {
    StreamedMemoryResult result = planParts(in, *experts, *table, in.arenaBytes[expertsStream][tableStreams],
                                            in.spillArenaBytes[tableStreams]);
    if (!result.plan || ((expertsStream || !result.plan->expertBytes) && (tableStreams || !result.plan->tableBytes)))
      return result;
    expertsStream = expertsStream || result.plan->expertBytes;
    tableStreams = tableStreams || result.plan->tableBytes;
  }
}

std::optional<std::string> ModelMemoryProfile::validationError() const {
  if (name.empty()) return "model_name_required";
  if (!maximumContextTokens ||
      maximumContextTokens > kv::kMaximumLogicalTokens) {
    return "invalid_model_context_length";
  }
  if (!targetKvLayout.valid()) return "invalid_target_kv_layout";
  if (spill.enabled && !targetKvLayout.spillable()) return "spill_plan_needs_a_spillable_layout";
  if (!ops::validFormat(stateFormat)) return "invalid_state_format";
  if (!footprint.targetWeightsBytes) return "target_weight_bytes_required";
  if (!footprint.draftWeightsBytes) return "draft_weight_bytes_required";
  if (!footprint.runtime.laneStatePlannedAllocatedBytes) {
    return "lane_state_bytes_required";
  }
  if (!footprint.runtime.sharedPrefillPlannedAllocatedBytes) {
    return "shared_prefill_bytes_required";
  }
  if (!footprint.runtime.sharedDecodePlannedAllocatedBytes) {
    return "shared_decode_bytes_required";
  }
  try {
    static_cast<void>(fixedRuntimeBytes());
  } catch (const std::overflow_error &) {
    return "fixed_runtime_cost_overflow";
  }
  return std::nullopt;
}

uint64_t ModelMemoryProfile::fixedRuntimeBytes() const {
  uint64_t result = 0;
  for (uint64_t value : {
           footprint.targetWeightsBytes, footprint.draftWeightsBytes,
           footprint.visionWeightsBytes,
           footprint.runtime.sharedPrefillPlannedAllocatedBytes,
           footprint.runtime.sharedDecodePlannedAllocatedBytes,
           model::kPipelineReserveBytes, model::kRuntimeOverheadReserveBytes,
           footprint.streamedTableBytes, footprint.aneFfnBytes, footprint.stateStagingBytes}) {
    if (!checkedAdd(result, value, result)) {
      throw std::overflow_error("fixed runtime cost overflow");
    }
  }
  return result;
}

std::string EngineMemoryBreakdown::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"physical_memory_bytes\":" << physicalMemoryBytes << ','
      << "\"recommended_working_set_bytes\":" << recommendedWorkingSetBytes
      << ','
      << "\"configured_memory_limit_bytes\":" << configuredMemoryLimitBytes
      << ',' << "\"working_set_margin_bytes\":" << workingSetMarginBytes << ','
      << "\"hard_budget_bytes\":" << hardBudgetBytes << ','
      << "\"target_weights_bytes\":" << targetWeightsBytes << ','
      << "\"draft_weights_bytes\":" << draftWeightsBytes << ','
      << "\"vision_weights_bytes\":" << visionWeightsBytes << ','
      << "\"maximum_batch_width\":" << maximumBatchWidth << ','
      << "\"lane_state_bytes\":" << laneStateBytes << ','
      << "\"shared_prefill_bytes\":" << sharedPrefillBytes << ','
      << "\"shared_decode_bytes\":" << sharedDecodeBytes << ','
      << "\"pipeline_reserve_bytes\":" << pipelineReserveBytes << ','
      << "\"runtime_overhead_reserve_bytes\":" << runtimeOverheadReserveBytes
      << ',' << "\"streamed_table_bytes\":" << streamedTableBytes << ','
      << "\"ane_ffn_bytes\":" << aneFfnBytes << ','
      << "\"state_staging_bytes\":" << stateStagingBytes << ','
      << "\"fixed_runtime_bytes\":" << fixedRuntimeBytes << ','
      << "\"dynamic_budget_bytes\":" << dynamicBudgetBytes << ','
      << "\"kv_page_tokens\":" << kvPageTokens << ','
      << "\"kv_page_bytes\":" << kvPageBytes << ','
      << "\"kv_extent_pages\":" << kvExtentPages << ','
      << "\"kv_extent_bytes\":" << kvExtentBytes << ','
      << "\"kv_capacity_pages\":" << kvCapacityPages << ','
      << "\"kv_capacity_bytes\":" << kvCapacityBytes << ','
      << "\"kv_capacity_tokens\":" << kvCapacityTokens << ','
      << "\"kv_index_page_bytes\":" << kvIndexPageBytes << ','
      << "\"kv_spill\":" << spillStatusJson(kvSpill) << ','
      << "\"resident_kv_bytes\":" << residentKvBytes << ','
      << "\"context_bounds\":{\"window\":" << contextBounds.window
      << ",\"resident\":" << contextBounds.resident << ",\"spilled\":" << contextBounds.spilled
      << ",\"disk\":" << contextBounds.disk << ",\"bound\":" << json::quote(contextBounds.bound()) << "},"
      << "\"minimum_dynamic_bytes\":" << minimumDynamicBytes << ','
      << "\"minimum_required_bytes\":" << minimumRequiredBytes << ','
      << "\"deficit_bytes\":" << deficitBytes << '}';
  return out.str();
}

std::string EngineMemoryBreakdown::describe() const {
  std::ostringstream out;
  out << "physical memory: " << bytesAndMiB(physicalMemoryBytes) << '\n'
      << "recommended working set: " << bytesAndMiB(recommendedWorkingSetBytes)
      << '\n'
      << "configured memory limit: "
      << (configuredMemoryLimitBytes ? bytesAndMiB(configuredMemoryLimitBytes)
                                     : "automatic")
      << '\n'
      << "working-set margin (max of 1 GiB or 2%): "
      << bytesAndMiB(workingSetMarginBytes) << '\n'
      << "hard budget: " << bytesAndMiB(hardBudgetBytes) << '\n'
      << "target weights: " << bytesAndMiB(targetWeightsBytes) << '\n'
      << "draft weights: " << bytesAndMiB(draftWeightsBytes) << '\n'
      << "vision weights: " << bytesAndMiB(visionWeightsBytes) << '\n'
      << "maximum DFlash batch width: " << maximumBatchWidth << '\n'
      << "lane state: " << bytesAndMiB(laneStateBytes) << '\n'
      << "shared prefill: " << bytesAndMiB(sharedPrefillBytes) << '\n'
      << "shared decode: " << bytesAndMiB(sharedDecodeBytes) << '\n'
      << "pipeline reserve: " << bytesAndMiB(pipelineReserveBytes) << '\n'
      << "allocator/runtime reserve: "
      << bytesAndMiB(runtimeOverheadReserveBytes) << '\n'
      << "streamed table cache: " << bytesAndMiB(streamedTableBytes) << '\n'
      << "Neural Engine split: " << bytesAndMiB(aneFfnBytes) << '\n'
      << "SSD cache state staging: " << bytesAndMiB(stateStagingBytes) << '\n'
      << "fixed runtime: " << bytesAndMiB(fixedRuntimeBytes) << '\n'
      << "elastic state/KV budget: " << bytesAndMiB(dynamicBudgetBytes) << '\n'
      << "KV page: " << kvPageTokens << " tokens, "
      << bytesAndMiB(kvPageBytes) << '\n'
      << "KV extent: " << kvExtentPages << " pages, "
      << bytesAndMiB(kvExtentBytes) << '\n'
      << "KV capacity of one request: " << kvCapacityPages << " pages / "
      << kvCapacityTokens << " tokens\n";
  if (kvSpill.enabled)
    out << "KV on the SSD cache: Index pages of " << bytesAndMiB(kvIndexPageBytes) << "; a Spill request keeps "
        << kvSpill.densePages << " dense, " << kvSpill.windowPages << " window and 2 x " << kvSpill.groupPages
        << " group pages Full; floors of " << bytesAndMiB(kvSpill.floorBytes) << " at "
        << kvSpill.floorTokens << " tokens: hot block cache " << bytesAndMiB(kvSpill.hotFloorBytes)
        << " (target " << bytesAndMiB(kvSpill.hotTargetBytes) << "), history staging "
        << bytesAndMiB(kvSpill.stagingBytes) << " in " << kvSpill.stagingBuffers << ", Index runway "
        << bytesAndMiB(kvSpill.indexRunwayBytes) << "; resident KV " << bytesAndMiB(residentKvBytes) << '\n';
  out << "context: " << contextBounds.maximum() << " tokens (" << contextBounds.bound() << "; window "
      << contextBounds.window << ", resident " << contextBounds.resident << ", spilled " << contextBounds.spilled
      << ", disk " << contextBounds.disk << ")\n"
      << "minimum dynamic runtime: " << bytesAndMiB(minimumDynamicBytes) << '\n'
      << "minimum required: " << bytesAndMiB(minimumRequiredBytes) << '\n'
      << "deficit: " << bytesAndMiB(deficitBytes);
  return out.str();
}

std::string BudgetValidationStatus::describe() const {
  std::ostringstream out;
  out << (valid ? "engine memory plan valid"
                : "engine memory budget validation failed")
      << " [" << budgetErrorCodeName(code) << "]: " << message << '\n'
      << breakdown.describe();
  return out.str();
}

EngineMemoryPlan::EngineMemoryPlan(DeviceCapabilities device,
                                   ModelMemoryProfile model,
                                   EngineMemoryBreakdown breakdown)
    : device_(std::move(device)), model_(std::move(model)),
      breakdown_(std::move(breakdown)) {}

uint32_t EngineMemoryPlan::maximumContextTokens() const noexcept { return breakdown_.contextBounds.maximum(); }

SpillGeometry EngineMemoryPlan::spillGeometry() const noexcept {
  const KvSpillPlan &spill = breakdown_.kvSpill;
  return {.fullPageBytes = breakdown_.kvPageBytes,
          .indexPageBytes = breakdown_.kvIndexPageBytes,
          .slotBytes = spill.slotBytes,
          .residentKvBytes = breakdown_.residentKvBytes,
          .residentTokens = breakdown_.contextBounds.resident,
          .densePages = spill.densePages,
          .windowPages = spill.windowPages,
          .groupPages = spill.groupPages};
}

std::string EngineMemoryPlan::toStatusJson() const {
  std::ostringstream out;
  out << '{' << "\"valid\":true,"
      << "\"maximum_context_tokens\":" << maximumContextTokens() << ','
      << "\"device\":" << deviceStatusJson(device_) << ','
      << "\"model\":" << modelStatusJson(model_) << ','
      << "\"budget\":" << breakdown_.toStatusJson() << '}';
  return out.str();
}

EngineMemoryPlanResult
evaluateEngineMemoryPlan(const DeviceCapabilities &device,
                         const ModelMemoryProfile &model,
                         uint64_t maximumMemoryBytes) {
  EngineMemoryBreakdown breakdown;
  breakdown.physicalMemoryBytes = device.physicalMemoryBytes;
  breakdown.recommendedWorkingSetBytes = device.recommendedMaxWorkingSetBytes;
  breakdown.configuredMemoryLimitBytes = maximumMemoryBytes;
  breakdown.targetWeightsBytes = model.footprint.targetWeightsBytes;
  breakdown.draftWeightsBytes = model.footprint.draftWeightsBytes;
  breakdown.visionWeightsBytes = model.footprint.visionWeightsBytes;
  breakdown.laneStateBytes =
      model.footprint.runtime.laneStatePlannedAllocatedBytes;
  breakdown.sharedPrefillBytes =
      model.footprint.runtime.sharedPrefillPlannedAllocatedBytes;
  breakdown.sharedDecodeBytes =
      model.footprint.runtime.sharedDecodePlannedAllocatedBytes;
  breakdown.pipelineReserveBytes = model::kPipelineReserveBytes;
  breakdown.runtimeOverheadReserveBytes = model::kRuntimeOverheadReserveBytes;
  breakdown.stateStagingBytes = model.footprint.stateStagingBytes;
  breakdown.aneFfnBytes = model.footprint.aneFfnBytes;
  breakdown.streamedTableBytes = model.footprint.streamedTableBytes;
  breakdown.kvPageTokens = kv::kPageTokens;

  if (auto error = device.validationError()) {
    return {std::nullopt, failure(BudgetErrorCode::InvalidDeviceCapabilities,
                                  *error, std::move(breakdown))};
  }
  if (auto error = model.validationError()) {
    return {std::nullopt, failure(BudgetErrorCode::InvalidModelSpec, *error,
                                  std::move(breakdown))};
  }

  breakdown.workingSetMarginBytes = EngineMemoryPolicy::workingSetMarginBytes(
      breakdown.recommendedWorkingSetBytes);
  if (breakdown.recommendedWorkingSetBytes <=
      breakdown.workingSetMarginBytes) {
    return {std::nullopt,
            failure(BudgetErrorCode::WorkingSetTooSmall,
                    "recommended working set does not exceed the working-set "
                    "margin",
                    std::move(breakdown))};
  }
  breakdown.hardBudgetBytes = EngineMemoryPolicy::hardBudgetBytes(
      breakdown.recommendedWorkingSetBytes, maximumMemoryBytes);

  breakdown.kvPageBytes = model.targetKvLayout.bytesPerModelPage();
  breakdown.fixedRuntimeBytes = model.fixedRuntimeBytes();
  breakdown.dynamicBudgetBytes =
      breakdown.hardBudgetBytes > breakdown.fixedRuntimeBytes
          ? breakdown.hardBudgetBytes - breakdown.fixedRuntimeBytes
          : 0;
  const std::optional<uint64_t> required =
      minimumRequiredBytes(breakdown.fixedRuntimeBytes,
                           breakdown.laneStateBytes, model.targetKvLayout);
  if (!required) {
    return {std::nullopt,
            failure(BudgetErrorCode::ArithmeticOverflow,
                    "minimum elastic runtime footprint overflows uint64",
                    std::move(breakdown))};
  }
  breakdown.minimumRequiredBytes = *required;
  breakdown.minimumDynamicBytes = *required - breakdown.fixedRuntimeBytes;
  breakdown.kvIndexPageBytes = model.targetKvLayout.indexBytesPerModelPage();
  breakdown.kvSpill = model.spill;
  // Page ids stay 32-bit. One request's KV capacity is the whole extents of
  // the size that leaves the fewest of the budget's pages unused.
  const uint64_t availableForOneRequestKv =
      breakdown.dynamicBudgetBytes > breakdown.laneStateBytes
          ? breakdown.dynamicBudgetBytes - breakdown.laneStateBytes
          : 0;
  const uint64_t budgetPages =
      std::min<uint64_t>(availableForOneRequestKv / breakdown.kvPageBytes,
                         std::numeric_limits<uint32_t>::max());
  breakdown.kvExtentPages = model.targetKvLayout.extentPagesFor(budgetPages);
  breakdown.kvCapacityPages = static_cast<uint32_t>(kvCapacityPagesFor(model.targetKvLayout, budgetPages));

  if (!checkedMultiply(breakdown.kvPageBytes, breakdown.kvExtentPages,
                       breakdown.kvExtentBytes) ||
      !checkedMultiply(breakdown.kvPageBytes, breakdown.kvCapacityPages,
                       breakdown.kvCapacityBytes) ||
      !checkedMultiply(breakdown.kvPageTokens, breakdown.kvCapacityPages,
                       breakdown.kvCapacityTokens)) {
    return {std::nullopt, failure(BudgetErrorCode::ArithmeticOverflow,
                                  "KV capacity overflows uint64",
                                  std::move(breakdown))};
  }
  const KvSpillPlan &spill = model.spill;
  breakdown.contextBounds = {
      .window = model.maximumContextTokens,
      .resident = static_cast<uint32_t>(std::min<uint64_t>(
          model.maximumContextTokens,
          remaining(breakdown.kvCapacityTokens, model::ExecutionLimits::speculativeScratchTokens))),
      .spilled = spill.memoryTokens,
      .disk = spill.diskTokens};
  if (spill.enabled)
    breakdown.residentKvBytes =
        remaining(breakdown.dynamicBudgetBytes, breakdown.laneStateBytes + spill.indexRunwayBytes +
                                                    std::max(spill.stagingBytes, spill.hotTargetBytes));
  if (!breakdown.kvExtentPages ||
      breakdown.dynamicBudgetBytes < breakdown.minimumDynamicBytes) {
    if (breakdown.minimumRequiredBytes > breakdown.hardBudgetBytes) {
      breakdown.deficitBytes =
          breakdown.minimumRequiredBytes - breakdown.hardBudgetBytes;
    }
    return {
        std::nullopt,
        failure(
            BudgetErrorCode::KvPoolDoesNotFit,
            "hard budget cannot fit one lane's state and the KV runway",
            std::move(breakdown))};
  }

  BudgetValidationStatus status{true, BudgetErrorCode::None,
                                "memory plan fits hard budget", breakdown};
  EngineMemoryPlan plan(device, model, breakdown);
  return {std::move(plan), std::move(status)};
}

} // namespace splash::engine
