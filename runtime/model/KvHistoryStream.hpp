#pragma once

// The history of a prefill whose pages hold only their index
// (kv::PageForm::Index), streamed into staging one attention layer at a time
// so that the chunk's attention runs the same kernels through another table.
// Staging is `buffers` kv::PageStorage of the pool's layout with one layer
// and no index, logical page j at staging page j: a layer's Index pages are
// read whole from their tier copies (KvCopy::readAt of kv::SlotFormat::
// layerRange) on background SSD reader batches and scattered in; its Full
// pages, and each chunk's own pages once stored, are copied on the GPU
// (ops::PagedAttention::addStageLayer). With one buffer, a layer's reads
// start once the previous layer's attention ended; with two, while it runs.
// A page that cannot be read (twice) leaves zeros in its staging page and
// fails the sequence (KvStreamResult::failures), not the command.
//
// One stream serves one command and one sequence at a time: a command, or a
// pass of one over some of the target's layers (BatchPlan::passes), streams
// the attention layers it runs, each pass its own command that reads and
// stages what a command run whole does of those layers. end() cancels and
// drains every batch the command started, also one a host step waits on, on
// completion, on a GPU fault and when the command is abandoned, before
// staging may change.
//
// Threads: the host steps of the command in flight own its batches; the
// engine thread owns the stream between commands. A step owns what it
// reads, so that one its command gave up may still run, and does nothing
// once its command's stream ended.

#include "Checked.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "model/KvReadStats.hpp"
#include "model/Model.hpp"
#include "model/SsdReaders.hpp"
#include "ops/KvSlotFormat.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace splash::model {

// The K/V a prefill chunk's attention reads: a page table and the layer's
// place in the extents it reaches, the pool's or staging's.
struct PrefillKv final {
  metal::MetalBuffer table;
  SplashKvLayer kv{};
};

// The attention layers a command streams, [first, end) of the pool's: all of
// them for a command run whole, those of its layers for a pass.
struct AttentionLayers final {
  uint32_t first = 0, end = 0;
};

class KvHistoryStream final {
public:
  // `layout` and `extentPages` are the pool's, `format` its layout's; the
  // staging pages come from `admission`, as KV memory. Throws
  // std::invalid_argument for a layout that cannot spill
  // (kv::Layout::spillable), a format of another layout, no readers, or
  // buffers other than 1 or 2.
  KvHistoryStream(metal::MetalBackend &backend, metal::AllocationAdmission admission, const kv::Layout &layout,
                  uint32_t extentPages, std::shared_ptr<SsdReaders> readers, kv::SlotFormat format,
                  uint32_t buffers);
  // Drains what the last command started.
  ~KvHistoryStream();
  KvHistoryStream(const KvHistoryStream &) = delete;
  KvHistoryStream &operator=(const KvHistoryStream &) = delete;
  // The page table of one buffer, which a stream of a pool's `layout`
  // allocates for each as it is built and holds until it goes: an entry for
  // each staging page, in whole host pages, as Metal allocates them, in
  // whichever extent size the pool takes (kv::Layout::mostOverExtents), so
  // that the memory plan, made before the pool's, counts it
  // (KvSpillPlan::tableBytes); 278,528 bytes for 256-page extents.
  [[nodiscard]] static constexpr uint64_t tableBytes(const kv::Layout &layout) {
    return layout.mostOverExtents(
        [](uint32_t extentPages) { return alignUp(stagedPages(extentPages) * sizeof(SplashKvPage)); });
  }

  // Between commands: staging for a sequence of `pages` pages, its history
  // and the command's own, or a refusal with nothing changed. Staging stays
  // as an idle buffer after the prompt until releaseIdle() gives it back,
  // returning its bytes.
  [[nodiscard]] ModelMemoryAdmission reserve(uint32_t pages);
  // Whether reserve(pages) would allocate nothing: the staging, which the
  // prompts that stream share, holds those pages already.
  [[nodiscard]] bool holds(uint32_t pages) const noexcept;
  // The bytes reserve(pages) would allocate now: zero when it holds them.
  [[nodiscard]] uint64_t shortfallBytes(uint32_t pages) const noexcept;
  uint64_t releaseIdle();

  // Encode of a command whose one streamed sequence's history holds Index
  // pages, over the attention layers `layers` it runs: starts the reads of
  // the first of them of those among the sources' pages before
  // commandPages, the pages the command reads (its history and its
  // chunks'). poolTable and poolLayers are the sequence's page table and the
  // pool's layers. std::logic_error until the previous command's stream
  // ended, when staging holds fewer pages, or for layers that are none or
  // past the pool's.
  void begin(const KvPageSources &sources, uint32_t commandPages, const metal::MetalBuffer &poolTable,
             std::span<const SplashKvLayer> poolLayers, AttentionLayers layers);
  // A layer's first chunk: a host step that finishes the layer's reads (and
  // with two buffers starts the next layer's, but past the command's last);
  // returns the staged table and layer its chunks attend. std::logic_error
  // for a layer the command does not run.
  [[nodiscard]] PrefillKv addLayer(metal::CommandGraph &graph, uint32_t layer);
  // After a chunk's store: the copy of its pages, [firstPage, firstPage +
  // pages), into staging, and with the layer's first chunk of every Full
  // page before them, its history's. std::logic_error for a layer the
  // command does not run, pages past the command's or one that holds only
  // its index.
  void addChunk(metal::CommandGraph &graph, uint32_t layer, uint32_t firstPage, uint32_t pages);
  // After a layer's attention, with one buffer: a host step whose `then`
  // starts the next layer's reads, unless the layer is the command's last.
  void addLayerEnd(metal::CommandGraph &graph, uint32_t layer);
  // Once the command completed, failed or was abandoned: cancels and drains
  // every batch it started, and returns the blocks it could not read, each
  // for lane 0 (the stream's one sequence, which the caller names), and the
  // bytes they brought.
  // Nothing more for a command it already ended.
  [[nodiscard]] KvStreamResult end();

  // Engine thread.
  [[nodiscard]] KvHistoryStreamStats stats() const;

private:
  // A buffer's staging pages: one for each page of the longest sequence, in
  // whole extents.
  [[nodiscard]] static constexpr uint64_t stagedPages(uint32_t extentPages) noexcept {
    constexpr uint64_t most = (uint64_t{kv::kMaximumPhysicalTokens} + kv::kPageTokens - 1) / kv::kPageTokens;
    return (most + extentPages - 1) / extentPages * extentPages;
  }

  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace splash::model
