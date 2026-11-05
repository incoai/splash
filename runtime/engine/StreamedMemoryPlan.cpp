#include "engine/StreamedMemoryPlan.hpp"
#include "Checked.hpp"
#include "metal/abi/PagedAttention.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace splash::engine {
namespace {

// The context every target keeps resident: the streamed stores take the rest
// of the budget, and load whole only where they leave it. Requests up to it
// never spill while they decode alone.
constexpr uint64_t kResidentAllowanceTokens = 65'536;
// A Spill request's tail window: the pages before its open page that it
// keeps Full. A decode selects about a tenth of its blocks in its last 64
// pages.
constexpr uint32_t kSpillWindowPages = 64;

uint64_t pagesOf(uint64_t tokens) noexcept { return (tokens + kv::kPageTokens - 1) / kv::kPageTokens; }

// The KV of a context of `tokens` tokens holds its speculative rows too.
uint64_t kvTokens(uint64_t tokens) noexcept { return tokens + model::ExecutionLimits::speculativeScratchTokens; }

// `pages` pages of `pageBytes` bytes in whole extents of `extentPages`.
uint64_t extentsBytes(uint64_t pages, uint64_t extentPages, uint64_t pageBytes) noexcept {
  return (pages + extentPages - 1) / extentPages * extentPages * pageBytes;
}

// A streamed part as its setting leaves it to the planner: the bytes the
// floors charge it, and all of it. An automatic part loads whole where the
// budget holds it beside the KV allowance, and otherwise streams through
// what the floors charge it and as much more as the budget allows. A part
// given a size (`sized`) streams through a store of exactly that size, which
// the floors charge and which is all its floor, so it neither grows nor gives
// memory back. Any other part loads whole. An Auto part whose smallest store
// takes all it would loads whole.
struct Part final {
  uint64_t charged = 0;
  uint64_t all = 0;
  bool automatic = false;
  bool streams = false;
  bool sized = false;
};

// The part as `option` sets it: a size is its store, exactly.
std::optional<Part> resolve(const StreamedPart &part, std::string_view option, std::string &error) {
  if (!part.allBytes) return Part{};
  const uint64_t bytes = part.setting.bytes;
  switch (part.setting.mode) {
  case StreamedSetting::Mode::Auto:
    if (part.minimumBytes < part.allBytes) return Part{part.minimumBytes, part.allBytes, true, false};
    [[fallthrough]];
  case StreamedSetting::Mode::Off:
    return Part{part.allBytes, part.allBytes, false, false};
  case StreamedSetting::Mode::Size:
    if (bytes >= part.allBytes) return Part{part.allBytes, part.allBytes, false, false};
    if (bytes < part.minimumBytes) {
      error = std::string(option) + " of " + std::to_string(bytes / kMiB) + " MiB is less than its smallest store, " +
              std::to_string((part.minimumBytes + kMiB - 1) / kMiB) + " MiB";
      return std::nullopt;
    }
    return Part{bytes, bytes, false, true, true};
  case StreamedSetting::Mode::Loaded:
    return Part{bytes, bytes, false, true};
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
// even an empty Spill request, which then serves what stays resident.
KvSpillPlan planSpill(const StreamedMemoryInputs &in, uint64_t floors) {
  KvSpillPlan spill;
  const kv::Layout &layout = in.kvLayout;
  if (!in.sparseKv || !layout.spillable() || !in.tierWritable) return spill;
  const model::SparseKvShape &shape = *in.sparseKv;
  const uint32_t extentPages = layout.maximumExtentPages();
  const SpillFloors spillFloors{
      .layout = layout,
      .geometry = {.fullPageBytes = layout.bytesPerModelPage(),
                   .indexPageBytes = layout.indexBytesPerModelPage(),
                   .densePages = static_cast<uint32_t>(pagesOf(shape.denseHistoryTokens)),
                   .windowPages = kSpillWindowPages,
                   .groupPages = in.groupPages},
      .extentPages = extentPages,
      .hotFloorBytes =
          extentsBytes(shape.requiredSlots(1) >> SPLASH_ATTENTION_LOCATION_BLOCK_BITS, extentPages,
                       layout.kvBytesPerModelPage()),
      .indexRunwayBytes = uint64_t{extentPages} * layout.indexBytesPerModelPage()};
  const uint64_t room = saturatingSubtract(in.hardBudgetBytes, floors);
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
                         in.kvSlotBytes
                             ? saturatingSubtract(in.activeLimitBytes, states) / in.kvSlotBytes * kv::kPageTokens
                             : 0)));
  spill.floorTokens = std::min({in.requestedContextTokens ? in.requestedContextTokens : in.windowTokens,
                                spill.memoryTokens, spill.diskTokens});
  // One buffer until the stores are placed (planParts).
  spill.stagingBuffers = 1;
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
  // The floors: what loads, the arenas and the stores at what the floors
  // charge them (their smallest, or the size --expert-memory and
  // --ple-memory set), a lane's state and the KV runway, and a
  // spill-capable target's spill floors.
  uint64_t loaded = 0;
  for (const uint64_t bytes : {in.fixedBytes, arenaBytes, experts.charged, table.charged})
    if (!checkedAdd(loaded, bytes, loaded)) return overflow();
  const std::optional<uint64_t> minimum = minimumRequiredBytes(loaded, in.laneStateBytes, layout);
  if (!minimum) return overflow();
  const uint64_t runwayBytes = *minimum - loaded - in.laneStateBytes;
  uint64_t floors = *minimum;
  // A plan that spills holds the arenas of a runtime whose K/V spills
  // instead, at least as large, beside which its spill floors must fit.
  uint64_t spillFloors = 0;
  if (!checkedAdd(floors, saturatingSubtract(spillArenaBytes, arenaBytes), spillFloors)) return overflow();
  StreamedMemoryPlan plan{.spill = planSpill(in, spillFloors)};
  KvSpillPlan &spill = plan.spill;
  if (spill.enabled) floors = spillFloors;
  if (!checkedAdd(floors, spill.floorBytes, floors)) return overflow();
  if (floors > in.hardBudgetBytes)
    return {std::nullopt,
            "the floors of the memory plan require " + std::to_string(floors) +
                " bytes but the Metal memory budget is " + std::to_string(in.hardBudgetBytes) + " bytes",
            floors};
  // An explicit hot cache target takes what it adds to the readers first,
  // of what the floors leave: the floors hold the larger of the hot cache's
  // floor and the history staging, and the readers the larger of its target
  // and the staging.
  if (spill.enabled && in.hotCacheBytes) {
    const uint64_t target = *in.hotCacheBytes;
    if (target < spill.hotFloorBytes)
      return {std::nullopt, "--sparse-kv-memory of " + std::to_string(target / kMiB) +
                                " MiB is less than the hot block cache's floor, " +
                                std::to_string(spill.hotFloorBytes / kMiB) + " MiB"};
    const uint64_t readers = spill.readerBytes(), added = saturatingSubtract(target, readers);
    if (added > in.hardBudgetBytes - floors)
      return {std::nullopt, "--sparse-kv-memory of " + std::to_string(target / kMiB) +
                                " MiB is more than the readers' floor, the larger of the hot block cache's floor "
                                "and the history staging, " +
                                std::to_string(readers / kMiB) + " MiB, and the " +
                                std::to_string((in.hardBudgetBytes - floors) / kMiB) + " MiB the floors leave"};
    spill.hotTargetBytes = target;
    spill.hotLimitBytes = std::max(spill.hotLimitBytes, target);
    floors += added;
  }

  // Then the priorities, out of what the floors leave: the experts first,
  // then the table. The KV allowance a part leaves holds the context kept
  // resident, with its speculative rows, as one request's KV capacity does:
  // in the fewest pages whose whole extents hold its pages, of which the
  // floors hold the runway.
  int64_t rest = static_cast<int64_t>(in.hardBudgetBytes - floors);
  // Every target keeps 64K tokens, or the requested context: a target that
  // cannot spill holds all of it, and a spill-capable one no more than 64K,
  // as more is KV no request can use.
  const auto allowance = [&](bool streaming) -> int64_t {
    const uint64_t cap = std::min<uint64_t>(in.windowTokens, kResidentAllowanceTokens);
    const uint64_t requested = in.requestedContextTokens;
    const uint64_t tokens = !requested ? cap : spill.enabled ? std::min(cap, requested) : requested;
    const uint64_t pages = pagesOf(kvTokens(tokens));
    uint64_t budget = pages;
    while (kvCapacityPagesFor(layout, budget) < pages) ++budget;
    // A windowed layout's KV beyond its lane's Full pages is Slim, in
    // extents of its own.
    const uint32_t floorPages = kvWindowFloorPages(layout, streaming ? in.groupPages : 0);
    return static_cast<int64_t>(
        saturatingSubtract(kvPagesBytes(layout, budget, floorPages, layout.extentPagesFor(budget)), runwayBytes));
  };
  const int64_t tableShare =
      table.automatic
          ? static_cast<int64_t>(saturatingSubtract(std::min(table.all, in.tableShareBytes), table.charged))
          : 0;
  bool expertsStream = experts.streams;
  plan.expertBytes = experts.streams ? experts.charged : 0;
  if (experts.automatic) {
    const auto above = static_cast<int64_t>(experts.all - experts.charged);
    if (rest - allowance(false) - tableShare >= above) {
      rest -= above;
    } else {
      expertsStream = true;
      const int64_t bytes = std::clamp<int64_t>(rest - allowance(true) - tableShare, 0, above);
      plan.expertBytes = experts.charged + static_cast<uint64_t>(bytes);
      rest -= bytes;
    }
  }
  // The store's floor: every slot of a store a size sets, which is all of
  // it, and of an automatic store the share this plan's spill gives
  // (expertFloorShare; model::ExpertStoreOptions).
  plan.expertFloorBytes = experts.sized ? experts.charged : 0;
  plan.expertFloorShare = experts.sized ? 1.0 : expertFloorShare(spill.enabled);
  plan.tableBytes = table.streams ? table.charged : 0;
  if (table.automatic) {
    const int64_t kept = rest - allowance(expertsStream);
    if (kept < static_cast<int64_t>(table.all - table.charged))
      plan.tableBytes = table.charged + static_cast<uint64_t>(std::clamp<int64_t>(kept, 0, tableShare));
  }
  // A second buffer of staged history lets the next layer's reads overlap
  // this layer's attention. It is planned where the readers' memory holds it
  // already (readerBytes), or where what the stores and the KV allowance
  // leave holds it: that is nothing where a store binds, which keeps the
  // room, and otherwise KV kept resident beyond the allowance.
  if (spill.enabled) {
    const uint64_t one = spill.stagingBytes;
    const uint64_t added = std::max(spill.hotTargetBytes, 2 * one) - spill.readerBytes();
    const uint64_t tableAbove =
        table.automatic ? (plan.tableBytes ? plan.tableBytes : table.all) - table.charged : 0;
    const int64_t left = rest - allowance(expertsStream) - static_cast<int64_t>(tableAbove);
    if (!added || left >= static_cast<int64_t>(added)) {
      spill.stagingBuffers = 2;
      spill.stagingBytes = 2 * one;
      spill.floorBytes += std::max(2 * one, spill.hotFloorBytes) - std::max(one, spill.hotFloorBytes);
    }
  }
  return {plan, {}};
}

} // namespace

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

} // namespace splash::engine
