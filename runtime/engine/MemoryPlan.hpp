#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "model/Model.hpp"
#include "ops/GDN.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace splash::json {
class Writer;
} // namespace splash::json

namespace splash::engine {

inline constexpr uint64_t kMiB = 1024ULL * 1024;
inline constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

// The pages of the KV runway in extents of extentPages pages: the whole
// extents that hold the pages startup warmup runs on, which the KV pool
// allocates when it is built.
[[nodiscard]] constexpr uint64_t kvRunwayPages(uint32_t extentPages) noexcept {
  return (uint64_t{model::ExecutionLimits::warmupKvPages} + extentPages - 1) /
         extentPages * extentPages;
}

// What one request needs at the least: the fixed bytes, one lane's state and
// the KV runway in the layout's smallest extents; nullopt on overflow.
[[nodiscard]] std::optional<uint64_t>
minimumRequiredBytes(uint64_t fixedBytes, uint64_t laneStateBytes,
                     const kv::Layout &layout) noexcept;

// One request's KV capacity, in pages, of a budget of `pages` pages: the
// whole extents of the size a pool of them takes (kv::Layout::extentPagesFor).
[[nodiscard]] uint64_t kvCapacityPagesFor(const kv::Layout &layout, uint64_t pages) noexcept;

// The pages a lane of a layout whose pages give up a window's layers
// (kv::Layout::slimmable) keeps Full at the most: its window's, the pages
// its command's rows write (`commandPages`, zero for a prefill chunk's) and
// one more, where its window starts inside a page. Zero for another layout,
// whose pages are all Full.
[[nodiscard]] uint32_t kvWindowFloorPages(const kv::Layout &layout, uint32_t commandPages) noexcept;
// The bytes of the extents of `extentPages` pages that one request's `pages`
// KV pages take: Full ones, but for a layout whose pages give up a window's
// layers, the `floorPages` its lane keeps Full (kvWindowFloorPages) and Slim
// ones past them, each form in whole extents of its own. kvPagesWithin is
// the inverse: the most pages whose extents `bytes` hold. Extents of one
// page price the pages one by one.
[[nodiscard]] uint64_t kvPagesBytes(const kv::Layout &layout, uint64_t pages, uint32_t floorPages,
                                    uint32_t extentPages) noexcept;
[[nodiscard]] uint64_t kvPagesWithin(const kv::Layout &layout, uint64_t bytes, uint32_t floorPages,
                                     uint32_t extentPages) noexcept;

// How a part of the weights that can stream from the SSD is held
// (--expert-memory, --ple-memory): as the planner decides (Auto), loaded
// whole (Off), or streamed through a store of exactly `bytes` (Size),
// charged among the floors, which loads the part whole when `bytes` would
// hold all of it: the PLE table's row cache, or the routed experts' store,
// all of it floor, so it neither grows nor gives memory back. Loaded, set by
// no option, is the store of exactly `bytes` a loaded model streams through,
// which the engine's plan uses (RuntimeResources::create).
struct StreamedSetting final {
  enum class Mode : uint8_t { Auto, Off, Size, Loaded };
  Mode mode = Mode::Auto;
  uint64_t bytes = 0;

  bool operator==(const StreamedSetting &) const = default;
};

// A part of the weights that can stream from the SSD instead of loading: a
// MoE target's routed experts (model::ExpertStore) or the table its GGUF
// holds (model::SparseRowStore). allBytes is what it takes loaded whole, zero
// for a model without it, and minimumBytes the smallest store that streams
// it.
struct StreamedPart final {
  uint64_t allBytes = 0;
  uint64_t minimumBytes = 0;
  StreamedSetting setting;
};

// Inputs that the memory planner needs from a loaded model. Model tensor and
// KV geometry stay with their owners; the planner receives only identity,
// capacity, and measured allocation sizes.
struct ModelMemoryFootprint final {
  uint64_t targetWeightsBytes = 0;
  uint64_t draftWeightsBytes = 0;
  uint64_t visionWeightsBytes = 0;
  model::ModelMemoryPlan runtime;
  // The buffer a state's write to the disk tier stages through, set aside
  // when the tier's state file opened (a quota that holds one state); zero
  // otherwise.
  uint64_t stateStagingBytes = 0;
  // The prefill FFN's Neural Engine split (ops::AneFfn::plannedBytes); zero
  // without one.
  uint64_t aneFfnBytes = 0;
  // A streamed table's host memory, its row cache above all
  // (SparseRowStore::committedBytes): not Metal memory, but the engine's for
  // as long as it runs, so the plan and the governor charge it like the
  // weights it stands in for. Zero without one.
  uint64_t streamedTableBytes = 0;
  // The expert store's chunks above its floor, which it starts without and
  // the ledger grows into as the host allows (StreamedMemoryPlan). No memory
  // at the start, but fixed in the plan, so that the context and the KV it
  // holds rest on the engine's limit alone.
  uint64_t hostHeldBackBytes = 0;
};

// What a spill-capable target plans for long requests' K/V on the SSD cache
// (docs/server.md, KV on the SSD): a target whose layout can spill
// (kv::Layout::spillable), whose model declares its block-sparse attention
// (model::SparseKvShape) and whose SSD cache takes writes. A Spill request's
// cache keeps Full the pages of its dense history (densePages), its tail
// window (windowPages) and up to two prefill groups awaiting their copies
// (groupPages each), and of every other page the index alone; its tier
// copies take slots of slotBytes. The readers hold the larger of the hot
// block cache of decoding lanes' selected blocks (laneSlots a lane, floor
// requiredSlots(1), target at least that) and a prefill's history staged a
// layer at a time in stagingBuffers buffers; the pool keeps one empty Index
// extent to spill through at the memory limit. Reader sizes are in whole
// extents of the largest size a pool of the layout takes, so the pool's own
// extents fit.
struct KvSpillPlan final {
  bool enabled = false;
  uint32_t densePages = 0;
  uint32_t windowPages = 0;
  uint32_t groupPages = 0;
  uint64_t slotBytes = 0;
  uint64_t laneSlots = 0;
  uint64_t hotFloorBytes = 0;
  uint64_t hotTargetBytes = 0;
  // The most the hot cache holds: its target, or the floor of every lane
  // decoding over Index pages (requiredSlots(maximumBatchWidth)).
  uint64_t hotLimitBytes = 0;
  uint32_t stagingBuffers = 0;
  uint64_t stagingBytes = 0;
  uint64_t indexRunwayBytes = 0;
  // The context the floors hold one Spill request of, and the floors: its
  // pages, the readers' and the runway.
  uint32_t floorTokens = 0;
  uint64_t floorBytes = 0;
  // What the plan sets aside for one Spill request's readers: the larger of
  // the hot cache's target and the history staging. Not the most they hold:
  // once their floors pass its target, each further lane decoding over Index
  // pages adds a lane's floor up to hotLimitBytes, and a prompt's staging may
  // sit beside other lanes' floors; KV's memory pays for both as they grow.
  [[nodiscard]] uint64_t readerBytes() const noexcept { return std::max(hotTargetBytes, stagingBytes); }
  // The longest context spilling serves: what memory holds beside the
  // streamed stores at their smallest, and what the SSD cache's room for
  // active KV holds (DiskSpace) beside every lane's state.
  uint32_t memoryTokens = 0;
  uint32_t diskTokens = 0;
};

struct ModelMemoryProfile final {
  std::string name;
  uint32_t maximumContextTokens = 0;
  kv::Layout targetKvLayout;
  ModelMemoryFootprint footprint;
  ops::GdnStateFormat stateFormat = ops::GdnStateFormat::Float32;
  // A spill-capable target's (planStreamedMemory); disabled for others.
  KvSpillPlan spill{};
  // The pages one lane's prefill command writes at most: a group's where
  // its routed experts stream (model::kPrefillGroupChunks chunks), zero for
  // a chunk's. A windowed layout's lane keeps them Full (kvWindowFloorPages).
  uint32_t commandPages = 0;

  [[nodiscard]] std::optional<std::string> validationError() const;
  [[nodiscard]] uint64_t fixedRuntimeBytes() const;
};

struct EngineMemoryPolicy {
  // recommendedMaxWorkingSetSize already describes Metal's performance-safe
  // working set. Keep only a small runtime/measurement margin below it; the
  // separate host reserve below protects the rest of unified memory.
  static constexpr uint64_t minimumWorkingSetMarginBytes = 1 * kGiB;
  static constexpr uint32_t workingSetMarginPercent = 2;

  [[nodiscard]] static constexpr uint64_t
  workingSetMarginBytes(uint64_t recommendedWorkingSetBytes) noexcept {
    uint64_t proportional =
        (recommendedWorkingSetBytes / 100) * workingSetMarginPercent +
        ((recommendedWorkingSetBytes % 100) * workingSetMarginPercent) / 100;
    return proportional > minimumWorkingSetMarginBytes
               ? proportional
               : minimumWorkingSetMarginBytes;
  }

  [[nodiscard]] static constexpr uint64_t
  hardBudgetBytes(uint64_t recommendedWorkingSetBytes,
                  uint64_t maximumMemoryBytes) noexcept {
    const uint64_t margin = workingSetMarginBytes(recommendedWorkingSetBytes);
    const uint64_t automatic = recommendedWorkingSetBytes > margin
                                   ? recommendedWorkingSetBytes - margin
                                   : 0;
    return maximumMemoryBytes && maximumMemoryBytes < automatic
               ? maximumMemoryBytes
               : automatic;
  }

  // A bounded host cushion, independent of the engine's Metal capacity.
  [[nodiscard]] static constexpr uint64_t
  hostAvailableReserveBytes(uint64_t physicalMemoryBytes) noexcept {
    return std::min<uint64_t>(physicalMemoryBytes / 10, 2 * kGiB);
  }
  // What growth no request in service needs, the cache, leaves of the host's
  // spare memory to other applications: 7% of the Mac's memory, from 2 to
  // 7 GiB. With kHostMarginBytes above it, such growth stops 3 GiB short of
  // a 24 GB Mac's spare memory.
  [[nodiscard]] static constexpr uint64_t
  hostSpareReserveBytes(uint64_t physicalMemoryBytes) noexcept {
    return std::clamp<uint64_t>(physicalMemoryBytes / 100 * 7, 2 * kGiB, 7 * kGiB);
  }
};

enum class BudgetErrorCode {
  None,
  InvalidDeviceCapabilities,
  InvalidModelSpec,
  WorkingSetTooSmall,
  ArithmeticOverflow,
  KvPoolDoesNotFit,
};

// The longest context a request may take, and what bounds it: the model's
// window (model); the context one request's KV holds resident (memory); and
// for a spill-capable target the longest one spilling serves, which memory
// or the SSD cache's room bounds (KvSpillPlan::memoryTokens, diskTokens).
struct ContextBounds final {
  uint32_t window = 0;
  uint32_t resident = 0;
  uint32_t spilled = 0;
  uint32_t disk = 0;

  // The window, unless resident KV or spilling holds less of it.
  [[nodiscard]] uint32_t maximum() const noexcept;
  // What the maximum is: model, memory or disk.
  [[nodiscard]] std::string_view bound() const noexcept;
};

struct EngineMemoryBreakdown {
  uint64_t physicalMemoryBytes = 0;
  uint64_t recommendedWorkingSetBytes = 0;
  // Optional user ceiling. Zero means the automatic safe working-set
  // ceiling. A higher value never overrides the OS-safe ceiling.
  uint64_t configuredMemoryLimitBytes = 0;
  uint64_t workingSetMarginBytes = 0;
  uint64_t hardBudgetBytes = 0;

  uint64_t targetWeightsBytes = 0;
  uint64_t draftWeightsBytes = 0;
  uint64_t visionWeightsBytes = 0;
  uint32_t maximumBatchWidth = model::ExecutionLimits::maximumBatchWidth;
  uint64_t laneStateBytes = 0;
  uint64_t sharedPrefillBytes = 0;
  uint64_t sharedDecodeBytes = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;
  uint64_t aneFfnBytes = 0;
  uint64_t stateStagingBytes = 0;
  uint64_t streamedTableBytes = 0;
  uint64_t hostHeldBackBytes = 0;
  uint64_t fixedRuntimeBytes = 0;

  // All lanes' state, cached composite states, and KV extents grow from this
  // one governor-controlled byte budget; none is preallocated.
  uint64_t dynamicBudgetBytes = 0;

  uint32_t kvPageTokens = 0;
  uint64_t kvPageBytes = 0;
  // The pool grows and shrinks in extents of kvExtentPages pages, a size
  // chosen for this pool (kv::Layout::extentPagesFor). This is allocation
  // geometry, never the cache block size.
  uint32_t kvExtentPages = 0;
  uint64_t kvExtentBytes = 0;
  // One request's KV capacity: the whole extents of the budget's pages left
  // after one lane's state. The pool's page ids cover more (RuntimeResources
  // sizes them by the hard budget). Request context and four-lane execution
  // are independent policy limits.
  uint32_t kvCapacityPages = 0;
  uint64_t kvCapacityBytes = 0;
  uint64_t kvCapacityTokens = 0;
  // An Index page's bytes (kv::PageForm::Index): its blocks' index alone.
  uint64_t kvIndexPageBytes = 0;
  // A Slim page's bytes and the Full pages a lane keeps beside them
  // (kvWindowFloorPages), of a layout whose pages give up a window's layers
  // past every window; zero for another layout. One request's capacity is
  // then those Full pages, in whole Full extents, and the pages of as many
  // whole Slim extents as the rest holds. The tails of states the SSD keeps
  // on a request's chain stay Full beside these and are not counted.
  uint64_t kvSlimPageBytes = 0;
  uint32_t kvWindowFloorPages = 0;
  // A spill-capable target's plan.
  KvSpillPlan kvSpill;
  ContextBounds contextBounds;

  uint64_t minimumDynamicBytes = 0;
  uint64_t minimumRequiredBytes = 0;
  uint64_t deficitBytes = 0;

  [[nodiscard]] std::string describe() const;
};

struct BudgetValidationStatus {
  bool valid = false;
  BudgetErrorCode code = BudgetErrorCode::None;
  std::string message;
  EngineMemoryBreakdown breakdown;

  [[nodiscard]] std::string describe() const;
};

// What the engine's admission needs of the memory plan to decide which
// requests spill (Engine::placement), as data: the bytes of a Full page, of
// an Index page and of a tier slot; the tokens one request's KV may take in
// memory alone, as without the SSD cache (ContextBounds::resident), for a
// request that cannot spill; and, per Spill request, the pages its cache
// keeps Full (its dense history, its tail window and up to two prefill
// groups awaiting copies).
struct SpillGeometry final {
  uint64_t fullPageBytes = 0;
  uint64_t indexPageBytes = 0;
  uint64_t slotBytes = 0;
  uint32_t residentTokens = 0;
  uint32_t densePages = 0;
  uint32_t windowPages = 0;
  uint32_t groupPages = 0;

  // A Resident request's KV at `tokens` tokens: its pages, Full.
  [[nodiscard]] uint64_t residentBytes(uint64_t tokens) const noexcept { return pages(tokens) * fullPageBytes; }
  // A Spill request's at `tokens` tokens: every page's index, and the pages
  // it keeps Full.
  [[nodiscard]] uint64_t spillFloorBytes(uint64_t tokens) const noexcept {
    const uint64_t all = pages(tokens);
    const uint64_t full = std::min<uint64_t>(all, uint64_t{densePages} + windowPages + 2ULL * groupPages);
    return full * fullPageBytes + (all - full) * indexPageBytes;
  }

private:
  [[nodiscard]] static constexpr uint64_t pages(uint64_t tokens) noexcept {
    return (tokens + kv::kPageTokens - 1) / kv::kPageTokens;
  }
};

// Whether the target can spill (a spillable layout that declares its
// SparseKvShape, with a writable tier), whether every request spills
// (--kv-spill always, for validation), the history a row may read whole
// through its page table, the pages the cache keeps Full for it
// (SparseKvShape::denseHistoryTokens, KvSpillPlan::densePages), and the
// geometry.
struct EngineSpillConfig final {
  bool capable = false;
  bool always = false;
  uint32_t denseTokens = 0;
  SpillGeometry geometry;
};

struct EngineMemoryPlanResult;

class EngineMemoryPlan {
public:
  [[nodiscard]] const EngineMemoryBreakdown &breakdown() const noexcept {
    return breakdown_;
  }
  // Stable per-request ceiling advertised by the runtime: never more than the
  // model supports, and no more than one request's KV capacity holds (less
  // the speculative scratch) unless spilling serves more
  // (ContextBounds::maximum); requests share the budget dynamically.
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept;
  // What admission decides which requests spill by (EngineSpillConfig).
  [[nodiscard]] SpillGeometry spillGeometry() const noexcept;

  // The plan as /status reports it (memory_plan): the device, the model and
  // the budget.
  friend void writeJson(json::Writer &w, const EngineMemoryPlan &plan);

private:
  EngineMemoryPlan(DeviceCapabilities device, ModelMemoryProfile model,
                   EngineMemoryBreakdown breakdown);

  DeviceCapabilities device_;
  ModelMemoryProfile model_;
  EngineMemoryBreakdown breakdown_;

  friend EngineMemoryPlanResult
  evaluateEngineMemoryPlan(const DeviceCapabilities &,
                           const ModelMemoryProfile &,
                           uint64_t);
};

struct EngineMemoryPlanResult {
  std::optional<EngineMemoryPlan> plan;
  BudgetValidationStatus status;
};

// Pure planning without resource allocation.
[[nodiscard]] EngineMemoryPlanResult
evaluateEngineMemoryPlan(const DeviceCapabilities &device,
                         const ModelMemoryProfile &model,
                         uint64_t maximumMemoryBytes);

} // namespace splash::engine
