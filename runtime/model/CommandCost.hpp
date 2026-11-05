#pragma once

// What a prefill command may hold (CommandLimits) and what a command or pass does in the model's units
// (CommandWork: expected from PrefillWorkSource::prefillWork, realized in CommandSample). Data only: the
// formulas behind the counts are the model's own (model/WorkFormulas.hpp), and no value is a time the model
// guesses.

#include <cstdint>
#include <limits>
#include <span>

namespace splash::model {

struct LayerRange final {
  uint32_t first = 0, end = 0;
  bool operator==(const LayerRange &) const = default;
};
// The layers a pass from layer `first` runs with `passesLeft` passes left, itself included: an even share of the
// layers left, as Runtime::Impl::passLayers splits them.
[[nodiscard]] constexpr LayerRange passLayers(uint32_t first, uint32_t passesLeft, uint32_t layers) noexcept {
  return {first, first + (layers - first) / passesLeft};
}

// The capabilities of one prefill command that differ between models, fixed when the model is built.
// The constant ones stay in ExecutionLimits (maximumBatchWidth, prefillTokenBudget).
struct CommandLimits final {
  uint32_t commandRows = 0;      // rows of one command, all lanes: prefillTokenBudget * RuntimeGeometry::prefillChunks
  uint32_t layers = 1;           // the target's layers, over which passes run (BatchPlan::layer, passLayers)
  uint32_t maximumPasses = 1;    // the most passes of one command: a layer each, so `layers`; 1 with the Neural
                                 // Engine FFN split, which takes a chunk's layers whole (QwenTarget::requirePrefill)
  uint32_t minimumUnitRows = 1;  // the fewest rows of a command its faster path takes: with the Neural Engine FFN
                                 // split the least rows of a chunk it takes (ops::AneFfn::minimumRows), as fewer
                                 // run the FFN on the GPU alone at a higher cost a row; else 1. A planning bound
                                 // (Scheduler), which the runtime does not refuse
  uint32_t lanesPastChunk = 1;   // lanes whose rows may run past a chunk as the lanes fill the chunks in order: 1
                                 // with GDN state, whose prefill group has one scratch cell for the state between a
                                 // lane's parts (ExecutionLimits::prefillTokenBudget), else maximumBatchWidth
  uint32_t indexPageLanes = 0;   // lanes of one prefill command whose history may hold Index pages while it runs:
                                 // 1 with the readers (KvHistoryStream), 0 for K/V that does not spill
  bool streamsWeights = false;   // the target's weights stream from the SSD (ExpertStore): its commands run in
                                 // layer passes, as a pass reads only its layers' weights
  // CommandWork::layerAttentionPairs one attention layer of a command, or of one pass of it, may compute:
  // denseAttentionPairs(nativeContextTokens - prefillTokenBudget, prefillTokenBudget), the pairs of a full chunk
  // at the end of the native window, the bound Bootstrap sets in EngineConfig::maximumPrefillAttentionPairs. A layer
  // whose reach is bounded saturates below it. Passes do not lower a layer's pairs, as a pass does not split a layer
  // and the GPU's wait limit binds each layer; fewer rows do. A planning bound; the runtime does not refuse it.
  uint64_t attentionPairs = std::numeric_limits<uint64_t>::max();
  bool operator==(const CommandLimits &) const = default;
};

// One lane of a prefill command as the engine would run it.
struct PrefillLane final {
  uint32_t position = 0;    // rows already in its KV (BatchItem::promptOffset)
  uint32_t rows = 0;        // rows this command adds
  uint32_t indexPages = 0;  // pages before the command's end holding only their index when it runs
  bool operator==(const PrefillLane &) const = default;
};

// What a command or one pass of it does, as planning and the decode gap's attribution read it.
struct CommandWork final {
  uint64_t attentionPairs = 0;      // query-key pairs over the attention layers run, each within its reach
  uint64_t layerAttentionPairs = 0; // the lanes' query-key pairs in the heaviest attention layer of the target, each
                                    // layer at its reach, over all its layers whatever the pass (a pass does not
                                    // split a layer); compared with CommandLimits::attentionPairs
  uint64_t weightReadBytes = 0;     // weights read from the SSD on demand (streamed experts)
  uint64_t spilledKvReadBytes = 0;  // spilled K/V read from the SSD: a streaming lane's history, a decode's hot-block
                                    // misses
  bool operator==(const CommandWork &) const = default;
};

// One completed command or pass (ModelBatchTicket::sample, valid once wait() returned).
struct CommandSample final {
  CommandWork work;                       // realized: bytes as the store's and the readers' counters moved
  double wallMilliseconds = 0.0;
  double imageEncodeMilliseconds = 0.0;   // the encode command ahead of a prefill (Runtime::prefillAsync)
  bool representative = true;             // false when the completion added work (the Neural Engine FFN's rerun)
};

// What planning asks a model. Engine thread, no command in flight; no allocation beyond the result.
class PrefillWorkSource {
public:
  virtual ~PrefillWorkSource() = default;
  // Expected work of a pass from layer `first` with `passesLeft` passes left (passLayers) of a prefill command
  // whose lanes fill its chunks in order, with the residency now; (0, 1) is the whole command.
  [[nodiscard]] virtual CommandWork prefillWork(std::span<const PrefillLane> lanes, uint32_t first,
                                                uint32_t passesLeft) const = 0;
};

} // namespace splash::model
