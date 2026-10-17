#pragma once

// Routed experts that stay in their source files and are read into slots when
// a step routes them (--expert-memory), so a MoE target whose experts do not
// fit in memory runs with part of them resident. The kernels read every
// routed expert through its block's slot table (ops::BlockMoeWeights), so
// resident and streamed experts run the same kernels.
//
// Experts of one size (the formats and shapes of their gate, up and down
// projections) share an arena of slots. A size of several blocks streams:
// its slots are recycled least recently used first across its blocks,
// those a long prefill chunk loaded before the others (Impl::route). A size
// of one block (a layer whose experts take formats no other layer's do, as a
// dynamic quantization leaves some) holds every one of its experts: they are
// read when the store is built or restored, and nothing loads them again.
//
// A streamed block's load is one host step of the step's command after its
// grouping (ops::MoeExpertStream), and one repack: it reads the tiles'
// routed experts, claims a slot for each that is not resident, never one of
// an expert the same load routes, reads their native rows from the sources
// into places of a staging pool from several threads (readRepackRows), each
// run of consecutive experts of a projection in reads of up to
// kLoadStepBytes, and writes the jobs of moe_expert_load, which repacks each
// from its place into its slot before the expert passes run. A pool holds
// every expert a load may route, so no load needs a second step:
// - the store's staging, a decode cycle's most routes (its most rows times
//   top-k, at most a block's), for decode steps and short prefills;
// - for a prefill load that may route more, slots it borrows (Impl::borrow):
//   the last of the arena of the streamed size of the most blocks, whose
//   bytes in every plane region hold two blocks of experts of any streamed
//   size, whatever their formats. The first such load evicts the experts
//   there, and the first decode load after it gives the slots back free, so
//   decode steps keep every slot; such a prefill churns the arena far more
//   than they hold.
//
// Once a block has staged its misses, the next block's experts that are
// likely routed and not resident are read ahead while the GPU runs the
// block, most likely first, into the places of the pool the load left free,
// in pieces of at most kAheadPieceBytes: a long prefill chunk's (rows * topK
// >= 3 * experts: it routes nearly every expert of each block), or a decode
// step's that the next layer's router picks from this block's input (the
// rows change little from layer to layer), by their summed routing weights.
// That block's load takes the read-ahead over: once the readers end the
// pieces they took, it reads the rest of the experts they began where they
// are, and its repack reads them there.
//
// The arena and the staging are memory the engine gives back while idle
// (model::WeightMemory): released, every slot is free; a restore reads the
// experts of the sizes of one block again, and the loads after it every
// expert they route.
//
// A load checks its sources unchanged after it read them (StreamedBlocks).
// A load that fails (a source cannot be read or was written) frees the slots
// it claimed and fails its command; every later load fails too.
//
// Threads: bind() runs while the model loads, before any command; addLoad()
// on the thread that encodes a command, reading only what construction and
// bind() set; the loads on the backend's host-step thread, one at a time
// (metal::EventStep); the readers of a load's read-ahead on the store's own
// thread and libdispatch's, until the next load, a release or the store's
// end waits for them; release() and restore() on the engine thread while no
// command is in flight (model::WeightMemory), so never beside a load;
// released() on the engine thread and stats() on any. Neither of those two
// waits for a load: a status query never waits behind a command's reads.

#include "metal/MetalBackend.hpp"
#include "model/ExpertStoreStats.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightMemory.hpp"
#include "ops/MoE.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::model {

// The blocks of a target whose routed experts stream (gguf::PlanOptions::
// streamExperts): each block's experts as its plan left them out of its
// image, in block order (the target's `layers` layers), and a check of each
// source they read, which holds it open and throws once it was written since
// it was opened.
struct StreamedBlocks final {
  std::vector<gguf::StreamedExperts> blocks;
  std::vector<std::function<void()>> checks;
};

struct ExpertStoreOptions final {
  // Every byte the store allocates (--expert-memory): its staging and tables
  // (ExpertStore::fixedBytes), and in the rest the arenas' slots, which hold
  // at least a block's experts of each size, all of a size of one block, and
  // besides the slots prefill loads borrow (ExpertStore::borrowedSlots).
  uint64_t memoryBytes = 0;
};

class ExpertStore final : public ops::MoeExpertStream, public WeightMemory {
public:
  // Allocates the arenas, the staging and the slot tables, and reads the
  // experts of the sizes of one block. Throws std::invalid_argument when a
  // block's experts differ from `shape`, or when options.memoryBytes, less
  // the staging and tables (fixedBytes), holds fewer slots than a block's
  // experts of each streamed size, every expert of a size of one block and
  // the slots prefill loads borrow.
  ExpertStore(metal::MetalBackend &backend, StreamedBlocks blocks, ops::MoeShape shape, ExpertStoreOptions options);
  ~ExpertStore() override;
  ExpertStore(const ExpertStore &) = delete;
  ExpertStore &operator=(const ExpertStore &) = delete;

  // Block `block`'s routed segments (views of its size's arena) and slot
  // table, which `weights` reads them through, and, when its experts stream,
  // the store, which loads them; the store keeps the block's router and
  // shared-expert gate, set before, which predict its routes.
  void bind(uint32_t block, ops::BlockMoeWeights &weights);
  void addLoad(metal::CommandGraph &graph, std::span<const ops::MoeLoadPart> parts, uint32_t block) const override;
  [[nodiscard]] ExpertStoreStats stats() const;
  // The arena and the staging, while no command is in flight; restore()
  // takes both back in one part and reads the experts of the sizes of one
  // block again.
  [[nodiscard]] bool released() const noexcept override;
  void release() override;
  [[nodiscard]] bool restore(const metal::AllocationAdmission &admit) override;
  // The bytes a store of these blocks allocates besides its arenas: its
  // staging, slot tables, jobs and the decode prediction's buffers.
  [[nodiscard]] static uint64_t fixedBytes(const StreamedBlocks &blocks, ops::MoeShape shape);
  // The slots of the streamed size of the most blocks that its prefill loads
  // borrow to stage in (ExpertStoreStats::borrowed), besides a block's.
  [[nodiscard]] static uint32_t borrowedSlots(const StreamedBlocks &blocks, ops::MoeShape shape);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// The slot table of a block whose routed experts are all resident: expert e
// in slot e.
[[nodiscard]] metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts);

} // namespace splash::model
