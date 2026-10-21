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
// One stream serves one command and one sequence at a time. end() cancels
// and drains every batch the command started, also one a host step waits
// on, on completion, on a GPU fault and when the command is abandoned,
// before staging may change.
//
// Threads: the host steps of the command in flight own its batches; the
// engine thread owns the stream between commands. A step owns what it
// reads, so that one its command gave up may still run, and does nothing
// once its command's stream ended.

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

  // Between commands: staging for a sequence of `pages` pages, its history
  // and the command's own, or a refusal with nothing changed. Staging stays
  // as an idle buffer after the prompt until releaseIdle() gives it back,
  // returning its bytes.
  [[nodiscard]] ModelMemoryAdmission reserve(uint32_t pages);
  uint64_t releaseIdle();

  // Encode of a command whose one streamed sequence's history holds Index
  // pages: starts the first layer's reads of those among the sources' pages
  // before commandPages, the pages the command reads (its history and its
  // chunks'). poolTable and poolLayers are the sequence's page table and the
  // pool's layers. std::logic_error until the previous command's stream
  // ended, or when staging holds fewer pages.
  void begin(const KvPageSources &sources, uint32_t commandPages, const metal::MetalBuffer &poolTable,
             std::span<const SplashKvLayer> poolLayers);
  // A layer's first chunk: a host step that finishes the layer's reads (and
  // with two buffers starts the next layer's); returns the staged table and
  // layer its chunks attend.
  [[nodiscard]] PrefillKv addLayer(metal::CommandGraph &graph, uint32_t layer);
  // After a chunk's store: the copy of its pages, [firstPage, firstPage +
  // pages), into staging, and with the layer's first chunk of every Full
  // page before them, its history's. std::logic_error for pages past the
  // command's or one that holds only its index.
  void addChunk(metal::CommandGraph &graph, uint32_t layer, uint32_t firstPage, uint32_t pages);
  // After a layer's attention, with one buffer: a host step whose `then`
  // starts the next layer's reads.
  void addLayerEnd(metal::CommandGraph &graph, uint32_t layer);
  // Once the command completed, failed or was abandoned: cancels and drains
  // every batch it started, and returns the blocks it could not read, each
  // for lane 0 (the stream's one sequence, which the caller names), and the
  // time the command waited on its reads. Nothing more for a command it
  // already ended.
  [[nodiscard]] KvStreamResult end();

  // Engine thread.
  [[nodiscard]] KvHistoryStreamStats stats() const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace splash::model
