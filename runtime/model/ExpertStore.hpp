#pragma once

// Routed experts that stay in their source files and are read into slots when
// a step routes them (--expert-memory), so a MoE target whose experts do not
// fit in memory runs with part of them resident. The kernels read every
// routed expert through its block's slot table (ops::BlockMoeWeights), so
// resident and streamed experts run the same kernels.
//
// Experts of one size (the formats and shapes of their gate, up and down
// projections) share an arena of slots, or several, each of some of its
// blocks, when one would take more than the device's largest buffer. A size
// of several blocks streams: an arena's slots are recycled across its blocks
// by adaptive replacement (ExpertSlots), which keeps the experts decode steps
// route again and again, and recycles a long prefill chunk's loads first. A
// size of one block (a layer whose experts take formats no other layer's do, as a
// dynamic quantization leaves some) holds every one of its experts: they are
// read when the store is built or restored, and nothing loads them again.
//
// A streamed block's load runs after its steps' grouping (ops::
// MoeExpertStream), in rounds: each a host step of the command, which claims
// a slot for each routed expert that is not resident (in the first round),
// never one of an expert the same load routes, and reads the native rows of
// some of them into places of the store's staging, then a repack
// (moe_expert_load) of those places into their slots. A decode step's load
// is one round over every place; any other load stages half the places a
// round, as many rounds as its rows may need, so that the other half is free
// for the read-ahead below while the GPU repacks. A run of consecutive
// experts' rows that a file stores as they are is read straight from it
// (F_NOCACHE without the buffer cache): its bytes widened to whole pages, into
// places that start on a page, and the repack starts from the rows' offset
// in the first page. Rows a loader builds (a quantized safetensors tensor's)
// are built into the places instead.
//
// Once a round has staged its experts, the next experts are read ahead while
// the GPU runs, into the places the round left free, an expert at a time in
// pieces of at most kAheadPieceBytes, most likely first: the load's experts
// of its next round, or after a decode step's round the next block's experts
// that are likely routed and not resident, those the next block's router
// picks from this block's input (the rows change little from block to
// block), by their summed routing weights. A long prefill chunk (rows * topK
// >= 3 * experts) routes nearly every expert of the next block: one more host
// step after its last repack reads them ahead into the whole staging, while
// the GPU runs this block's experts and the next block's mixing. The next
// round stops the read-ahead and takes over the experts it routes where they
// are, reading what the readers had not (or failed to: a read ahead of an
// expert no step routes is no failure).
//
// The arena and the staging are memory the engine gives back while idle
// (model::WeightMemory): released, every slot is free; a restore reads the
// experts of the sizes of one block again, and the loads after it every
// expert they route.
//
// Reads run on the model's weight readers (model::WeightReaders), which try a
// failed read once more. A load checks its sources unchanged after it read
// them. A load that fails (a read of an expert it routes fails twice, or a
// source was written) frees the slots of the experts it had not repacked and
// fails its command; every later load fails too.
//
// Threads: bind() runs while the model loads, before any command; addLoad()
// on the thread that encodes a command, reading only what construction and
// bind() set; the loads on the backend's host-step thread, one at a time
// (metal::EventStep); the reads of a read-ahead on the readers, until the
// next load, a release or the store's end stops them; release() and
// restore() on the engine thread while no command is in flight
// (model::WeightMemory), so never beside a load; released() on the engine
// thread and stats() on any. Neither of those two waits for a load: a status
// query never waits behind a command's reads.

#include "metal/MetalBackend.hpp"
#include "model/ExpertStoreStats.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightMemory.hpp"
#include "model/WeightReaders.hpp"
#include "ops/MoE.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::model {

// The blocks of a target whose routed experts stream (gguf::PlanOptions::
// streamExperts): each block's experts as its plan left them out of its
// image, in block order (the target's `layers` layers), what holds the
// sources their rows name open (the store keeps it), and a check of each
// source, which throws once it was written since it was opened.
struct StreamedBlocks final {
  std::vector<gguf::StreamedExperts> blocks;
  std::shared_ptr<const void> sources;
  std::vector<std::function<void()>> checks;
};

struct ExpertStoreOptions final {
  // Every byte the store allocates (--expert-memory): its staging and tables
  // (ExpertStore::fixedBytes), and in the rest the arenas' slots, which hold
  // at least a block's experts of each size and all of a size of one block.
  uint64_t memoryBytes = 0;
  // The readers its reads run on, which a model's streamed weights share.
  std::shared_ptr<WeightReaders> readers;
  // The most bytes one arena takes: the device's largest buffer
  // (maxBufferLength) when zero or more.
  uint64_t bufferBytes = 0;
};

class ExpertStore final : public ops::MoeExpertStream, public WeightMemory {
public:
  // Allocates the arenas, the staging and the slot tables, and reads the
  // experts of the sizes of one block. Throws std::invalid_argument when a
  // block's experts differ from `shape`, or when options.memoryBytes, less
  // the staging and tables (fixedBytes), holds fewer slots than a block's
  // experts of each streamed size and every expert of a size of one block.
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

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// The slot table of a block whose routed experts are all resident: expert e
// in slot e.
[[nodiscard]] metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts);

} // namespace splash::model
