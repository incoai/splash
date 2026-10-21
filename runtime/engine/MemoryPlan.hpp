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

// How a part of the weights that can stream from the SSD is held
// (--expert-memory, --ple-memory): as the planner decides (Auto), loaded
// whole (Off), or streamed through a store of `bytes` (Size), which loads
// the part whole instead when it would hold all of it.
struct StreamedSetting final {
  enum class Mode : uint8_t { Auto, Off, Size };
  Mode mode = Mode::Auto;
  uint64_t bytes = 0;

  bool operator==(const StreamedSetting &) const = default;
};

// A part of the weights that can stream from the SSD instead of loading: a
// MoE target's routed experts (model::ExpertStore) or the table its GGUF
// holds (model::SparseRowStore). allBytes is what it takes loaded whole, zero
// for a model without it, and floorBytes the smallest store that streams it.
struct StreamedPart final {
  uint64_t allBytes = 0;
  uint64_t floorBytes = 0;
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
};

// What a spill-capable target plans for the K/V of long requests that lives
// on the SSD cache (DEVELOPMENT.md, KV on the SSD): a target whose layout can
// spill (kv::Layout::spillable), whose model declares its block-sparse
// attention (model::SparseKvShape) and whose SSD cache takes writes. A Spill
// request's cache keeps Full the pages of its dense history (densePages), of
// its tail window (windowPages) and of up to two prefill groups awaiting
// their copies (groupPages each), and of every other page the index alone;
// its tier copies take slots of slotBytes. Beside the requests' pages the
// readers hold the larger of the hot block cache of the decoding lanes'
// selected blocks (laneSlots a lane, its floor requiredSlots(1), its target
// at least that) and a prefill's history staged a layer at a time in
// stagingBuffers buffers; and the pool keeps one empty Index extent to spill
// through at the memory limit. The readers' sizes are in whole extents of
// the largest a pool of the layout takes, which the pool's own fit.
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
  // The longest context spilling serves: what memory holds beside the
  // streamed stores at their floors, and what the SSD cache's room for
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

  [[nodiscard]] std::optional<std::string> validationError() const;
  [[nodiscard]] uint64_t fixedRuntimeBytes() const;
};

// What the planner sizes the streamed stores and a spill-capable target's
// floors by, before the model loads (RuntimeResources::create), and once it
// loaded with the arenas its geometry takes.
struct StreamedMemoryInputs final {
  uint64_t hardBudgetBytes = 0;
  // Everything the plan holds besides the streamed parts, the arenas, the
  // lanes' state and KV: the weights that load, the reserves, the state
  // staging and the Neural Engine split.
  uint64_t fixedBytes = 0;
  // The arenas of the model's geometry (model::plannedRuntimeMemory) by what
  // streams, [experts][table]: a target whose routed experts stream runs its
  // prefill commands as groups, and one whose table streams stages its rows,
  // so streaming only grows them. None before the model loads.
  std::array<std::array<uint64_t, 2>, 2> arenaBytes{};
  // The arenas of the geometry of a runtime whose K/V spills, by whether the
  // table streams (model::StreamedParts::kv): it runs prefill groups whether
  // the experts stream or not, and holds where its verifies' listed blocks
  // are, so they are at least arenaBytes'. A plan that spills takes them.
  std::array<uint64_t, 2> spillArenaBytes{};
  uint64_t laneStateBytes = 0;
  kv::Layout kvLayout;
  // The model's window (servedContextTokens), the context --max-context
  // asks for, zero for auto, and the one a relaunch keeps
  // (--relaunch-context), zero for a first start.
  uint32_t windowTokens = 0;
  uint32_t requestedContextTokens = 0;
  uint32_t relaunchContextTokens = 0;
  StreamedPart experts;
  StreamedPart table;
  // --sparse-kv-memory: the hot block cache's target, at least its floor;
  // none keeps it at its floor.
  std::optional<uint64_t> hotCacheBytes;
  // The block-sparse attention a target declares, none for a dense one.
  std::optional<model::SparseKvShape> sparseKv;
  // The SSD cache: whether it takes writes, its slots and the room active
  // KV may take of its volume (model::DiskBudget::activeLimitBytes).
  bool tierWritable = false;
  uint64_t kvSlotBytes = 0;
  uint64_t stateSlotBytes = 0;
  uint64_t activeLimitBytes = 0;
  // The pages of a prefill group's rows (model::kPrefillGroupChunks chunks).
  uint32_t groupPages = 0;
};

// The stores the plan streams the parts through, zero for a part that loads
// whole or a model without it, and a spill-capable target's plan.
struct StreamedMemoryPlan final {
  uint64_t expertBytes = 0;
  uint64_t tableBytes = 0;
  KvSpillPlan spill{};
};

struct StreamedMemoryResult final {
  std::optional<StreamedMemoryPlan> plan;
  // Why nothing was planned, and when the floors exceed the budget, their
  // bytes.
  std::string error;
  uint64_t requiredBytes = 0;
};

// Plans the stores of the parts that stream and a spill-capable target's
// floors from one budget (DEVELOPMENT.md, Memory and context). The floors
// come first: the fixed bytes and the arenas, a lane's state and the KV
// runway (minimumRequiredBytes), the stores at their floors and the spill
// floors at the context spilling serves. Then the routed experts load whole
// where that leaves KV its allowance, and otherwise stream through all that
// the allowance and the PLE table's share leave; the table loads whole where
// the rest holds it, and otherwise streams through its share, 512 MiB at most.
// The allowance is the KV of a context in the fewest pages whose whole
// extents hold it, as one request's capacity rounds them (kvCapacityPagesFor):
// a spill-capable target's first 64K tokens; a dense target's --max-context,
// or none, so that its experts load wherever main loads them, and the first
// 64K tokens once they stream. The arenas are those of the parts that
// stream: where the plan with the arenas of fewer parts streams more, it
// plans again with those. An explicit setting replaces its part; one below
// the part's floor, a hot cache target beyond what the floors leave, and
// floors beyond the budget are refused.
[[nodiscard]] StreamedMemoryResult planStreamedMemory(const StreamedMemoryInputs &inputs);

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
  // 7 GiB. With the warning margin above it, such growth stops 3 GiB short of
  // a 24 GB Mac's spare memory, about 5.5 GiB short of 64 GB and 8 GiB short
  // of 128 GB or more.
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
  // A spill-capable target's plan, and what the dynamic budget leaves the
  // requests' pages beside one lane's state, its readers and its runway
  // (SpillGeometry::residentKvBytes).
  KvSpillPlan kvSpill;
  uint64_t residentKvBytes = 0;
  ContextBounds contextBounds;

  uint64_t minimumDynamicBytes = 0;
  uint64_t minimumRequiredBytes = 0;
  uint64_t deficitBytes = 0;

  [[nodiscard]] std::string toStatusJson() const;
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
// requests spill (Engine::kvModeFor), as data: the bytes of a Full page, of
// an Index page and of a tier slot; the memory left to resident KV after the
// floors; the tokens one request's KV may take in memory alone, as without
// the SSD cache (ContextBounds::resident), for a request that cannot spill;
// and, per Spill request, the pages its cache keeps Full (its dense history,
// its tail window and up to two prefill groups awaiting copies).
struct SpillGeometry final {
  uint64_t fullPageBytes = 0;
  uint64_t indexPageBytes = 0;
  uint64_t slotBytes = 0;
  uint64_t residentKvBytes = 0;
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
// (SparseKvShape::denseTokens, KvSpillPlan::densePages), and the geometry.
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

  [[nodiscard]] std::string toStatusJson() const;

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
