#pragma once

// The startup planner of the parts of the weights that can stream from the
// SSD and of a spill-capable target's floors (RuntimeResources::create): it
// sizes the stores the model loads with, before the engine's plan
// (EngineMemoryPlan) is made from what loaded.

#include "engine/MemoryPlan.hpp"
#include "model/ExpertChunkLayout.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace splash::engine {

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
  // The row cache an automatic table takes at most
  // (model::StreamableWeights::tableShareBytes).
  uint64_t tableShareBytes = 0;
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
// whole or a model without it, and a spill-capable target's plan. The
// routed experts' store takes expertBytes, its cap, starts at its floor and
// gives back what memory needs down to it: expertFloorBytes, all of the
// store where --expert-memory sizes it, or where that is zero
// expertFloorShare of its slots (model::ExpertStoreOptions::floorBytes,
// floorShare). The ledger grows it toward its cap
// (MemoryGovernor::growToTargets).
struct StreamedMemoryPlan final {
  uint64_t expertBytes = 0;
  uint64_t expertFloorBytes = 0;
  double expertFloorShare = model::kExpertFloorShare;
  uint64_t tableBytes = 0;
  KvSpillPlan spill{};
};

// The share of its slots an automatic routed experts' store keeps as its
// floor: all of them where the run's KV can spill, which gives memory back
// instead while every chunk the store gives up costs expert reads from the
// SSD until it regrows; otherwise model::kExpertFloorShare (0.85).
[[nodiscard]] constexpr double expertFloorShare(bool spills) noexcept {
  return spills ? 1.0 : model::kExpertFloorShare;
}

struct StreamedMemoryResult final {
  std::optional<StreamedMemoryPlan> plan;
  // Why nothing was planned, and when the floors exceed the budget, their
  // bytes.
  std::string error;
  uint64_t requiredBytes = 0;
};

// Plans the stores of the parts that stream and a spill-capable target's
// floors from one budget (docs/server.md, Memory and context). The floors
// come first: the fixed bytes and the arenas, a lane's state and the KV
// runway (minimumRequiredBytes), the stores at their smallest or at the
// size --expert-memory or --ple-memory sets, and the spill floors
// at the context spilling serves. Then the routed experts load whole where
// that leaves KV its allowance, and otherwise stream through all that the
// allowance and the PLE table's share leave; the table too keeps the
// allowance first: it loads whole where the rest holds it beside the
// allowance, and otherwise streams through what the allowance leaves, up to
// its share, tableShareBytes. The allowance is the KV of a context in the
// fewest pages whose whole extents hold it, as one request's capacity rounds
// them (kvCapacityPagesFor): a spill-capable target's first 64K tokens, or
// --max-context where that is less; a dense target's --max-context, or else
// its first 64K tokens too, so that its context never shrinks as the budget
// grows. Nothing rests on the host: the stores, the floors, what loads whole
// and the context follow the engine's limit alone.
// The arenas are those of the parts that stream: where the plan with the
// arenas of fewer parts streams more, it plans again with those. An
// explicit size below all of a part replaces it: the part streams through a
// store of exactly that size, charged among the floors, and the routed
// experts' store keeps all of it as its floor. A size below the part's
// smallest store, a hot cache target beyond what the floors leave, and
// floors beyond the budget are refused.
[[nodiscard]] StreamedMemoryResult planStreamedMemory(const StreamedMemoryInputs &inputs);

} // namespace splash::engine
