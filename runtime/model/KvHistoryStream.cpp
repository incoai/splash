#include "model/KvHistoryStream.hpp"

#include "AwakeClock.hpp"
#include "model/KvCopyRead.hpp"
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

// Staging's pages: one layer of the pool's K/V, at the pool's places in an
// extent.
kv::Layout oneLayer(kv::Layout layout) {
  layout.attentionLayers = 1;
  layout.indexDimension = 0;
  return layout;
}

const kv::SlotFormat &checked(const kv::Layout &layout, const kv::SlotFormat &format, const SsdReaders *readers,
                              uint32_t buffers) {
  if (!layout.spillable()) throw std::invalid_argument("a history stream needs a layout that can spill");
  if (format.layout() != layout) throw std::invalid_argument("a history stream's slot format is another layout's");
  if (!readers || buffers < 1 || buffers > 2)
    throw std::invalid_argument("a history stream needs readers and one or two buffers");
  return format;
}

} // namespace

struct KvHistoryStream::Impl {
  // A page of the history that holds only its index: its place in the
  // sequence, its block and the tier copy its K/V is read from.
  struct Page final {
    uint32_t page = 0;
    uint64_t block = 0;
    std::shared_ptr<const KvCopy> copy;
  };
  // One layer's reads, once started, whose failures are the pages read twice
  // in vain, and whether a chunk staged the layer's history.
  struct LayerReads final {
    std::shared_ptr<SsdReaders::Batch> batch;
    bool staged = false;
  };
  // A command's stream, from begin() to end(): the attention layers it runs,
  // the pages the command reads, before commandPages, of which `pages` hold
  // only their index (in order), the pool's table and layers, and each
  // layer's reads.
  struct Run final {
    uint64_t serial = 0;
    bool active = false;
    AttentionLayers range;
    uint32_t commandPages = 0;
    std::shared_ptr<const std::vector<Page>> pages;
    metal::MetalBuffer poolTable;
    std::vector<SplashKvLayer> poolLayers;
    std::vector<LayerReads> layers;
  };

  Impl(metal::MetalBackend &backend, metal::AllocationAdmission admission, const kv::Layout &layout,
       uint32_t extentPages, std::shared_ptr<SsdReaders> readers, kv::SlotFormat format, uint32_t buffers);

  [[nodiscard]] uint32_t stagingPages() const noexcept { return extents * staging.front()->extentPages(); }
  // The extents each buffer holds for `pages` pages.
  [[nodiscard]] uint32_t extentsFor(uint32_t pages) const noexcept {
    const uint32_t extentPages = staging.front()->extentPages();
    return (pages + extentPages - 1) / extentPages;
  }
  // Whether the run's command runs `layer`; the engine thread's.
  [[nodiscard]] bool runs(uint32_t layer) const noexcept {
    return run.active && layer >= run.range.first && layer < run.range.end;
  }
  // Starts `layer`'s reads into its buffer; under the lock.
  void startLocked(uint32_t layer);
  // A host step's `then`: starts `layer`'s reads unless run `serial` ended;
  // a start that fails is the layer's first chunk's.
  void start(uint64_t serial, uint32_t layer) noexcept;
  // A layer's first chunk's host step: its reads end before the GPU goes on,
  // or once end() stopped them.
  void finish(uint64_t serial, uint32_t layer);
  // Stops and drains every batch of the run, which ends; under the lock.
  void drainLocked() noexcept;
  // A reader task: throws when the page cannot be read.
  void readPage(uint32_t layer, const Page &page) const;

  // The pool layout's.
  const kv::SlotFormat format;
  const std::shared_ptr<SsdReaders> readers;
  const uint32_t buffers;
  // The staging's admission, which reserve() asks once for every extent.
  const metal::AllocationAdmission admission;
  // The bytes of a page's records of one layer (kv::SlotFormat::layerRange).
  const uint64_t layerBytes;
  std::vector<std::unique_ptr<kv::PageStorage>> staging;
  // Each buffer's page table: logical page j at staging page j (Shared).
  std::vector<metal::MetalBuffer> tables;
  // The engine thread's: the extents each buffer holds, and what every
  // command streamed.
  uint32_t extents = 0;
  KvHistoryStreamStats counts;
  // The engine thread writes the run under `mutex`, which host steps and
  // their `then` read it under; it reads it without.
  std::mutex mutex;
  Run run;
  // Updated from any thread.
  LatencyHistogram readTimes, waits;
};

KvHistoryStream::Impl::Impl(metal::MetalBackend &backend, metal::AllocationAdmission admission,
                            const kv::Layout &layout, uint32_t extentPages, std::shared_ptr<SsdReaders> readers,
                            kv::SlotFormat format, uint32_t buffers)
    : format(checked(layout, format, readers.get(), buffers)), readers(std::move(readers)), buffers(buffers),
      admission(admission), layerBytes(this->format.layerRange(0).bytes) {
  const uint64_t pages = KvHistoryStream::stagedPages(extentPages);
  for (uint32_t buffer = 0; buffer < buffers; ++buffer) {
    staging.push_back(std::make_unique<kv::PageStorage>(backend, admission, oneLayer(layout),
                                                        static_cast<uint32_t>(pages), extentPages));
    tables.push_back(backend.allocateBuffer(tableBytes(layout), metal::BufferStorage::Shared,
                                            "kv-history/table-" + std::to_string(buffer)));
  }
  counts.buffers = buffers;
}

void KvHistoryStream::Impl::readPage(uint32_t layer, const Page &page) const {
  const std::array<std::span<std::byte>, 4> staged = staging[layer % buffers]->layerSpans(page.page, 0);
  if (const std::optional<std::span<const std::byte>> records = readKvCopy(*page.copy, format.layerRange(layer))) {
    format.scatterLayer(*records, staged);
    return;
  }
  // A page that cannot be read leaves zeros, which the sequence's attention
  // reads before it fails.
  for (const std::span<std::byte> tensor : staged) std::memset(tensor.data(), 0, tensor.size());
  throw std::runtime_error("a spilled K/V page could not be read");
}

void KvHistoryStream::Impl::startLocked(uint32_t layer) {
  LayerReads &reads = run.layers[layer];
  // A layer's reads take from their start to their last page's end, which a
  // reader reaches; reads stopped early take no time.
  std::function<void(const SsdReaders::Batch &)> ended;
  if (!run.pages->empty())
    ended = [this, started = AwakeClock::now()](const SsdReaders::Batch &) {
      readTimes.add(static_cast<uint64_t>(millisecondsSince(started) * 1e6));
    };
  reads.batch = readers->start(
      run.pages->size(), [this, layer, pages = run.pages](size_t index) { readPage(layer, (*pages)[index]); },
      readers->readers(), std::move(ended));
}

void KvHistoryStream::Impl::start(uint64_t serial, uint32_t layer) noexcept {
  const std::lock_guard lock(mutex);
  if (!run.active || run.serial != serial || run.layers[layer].batch) return;
  try {
    startLocked(layer);
  } catch (...) {
    // finish() starts them.
  }
}

void KvHistoryStream::Impl::finish(uint64_t serial, uint32_t layer) {
  std::shared_ptr<SsdReaders::Batch> batch;
  {
    const std::lock_guard lock(mutex);
    if (!run.active || run.serial != serial) return;
    LayerReads &reads = run.layers[layer];
    if (!reads.batch) startLocked(layer);
    batch = reads.batch;
  }
  // Without the lock: end() stops the reads of a command given up, and they
  // end here.
  const AwakeClock::time_point began = AwakeClock::now();
  readers->finish(*batch);
  const double waited = millisecondsSince(began);
  const std::lock_guard lock(mutex);
  if (!run.active || run.serial != serial) return;
  waits.add(static_cast<uint64_t>(waited * 1e6));
}

void KvHistoryStream::Impl::drainLocked() noexcept {
  for (LayerReads &reads : run.layers)
    if (reads.batch) readers->stop(*reads.batch);
  run.active = false;
}

KvHistoryStream::KvHistoryStream(metal::MetalBackend &backend, metal::AllocationAdmission admission,
                                 const kv::Layout &layout, uint32_t extentPages, std::shared_ptr<SsdReaders> readers,
                                 kv::SlotFormat format, uint32_t buffers)
    : impl_(std::make_shared<Impl>(backend, std::move(admission), layout, extentPages, std::move(readers),
                                   std::move(format), buffers)) {}

KvHistoryStream::~KvHistoryStream() {
  const std::lock_guard lock(impl_->mutex);
  impl_->drainLocked();
}

bool KvHistoryStream::holds(uint32_t pages) const noexcept { return impl_->extentsFor(pages) <= impl_->extents; }

uint64_t KvHistoryStream::shortfallBytes(uint32_t pages) const noexcept {
  const Impl &impl = *impl_;
  const uint32_t wanted = impl.extentsFor(pages);
  return wanted > impl.extents ? uint64_t{wanted - impl.extents} * impl.buffers * impl.staging.front()->extentBytes()
                               : 0;
}

ModelMemoryAdmission KvHistoryStream::reserve(uint32_t pages) {
  Impl &impl = *impl_;
  if (holds(pages)) return {.ok = true};
  const uint32_t extentPages = impl.staging.front()->extentPages();
  const uint32_t wanted = impl.extentsFor(pages);
  if (uint64_t{wanted} * extentPages > impl.staging.front()->pageCount())
    throw std::invalid_argument("history staging past the longest sequence");
  if (impl.run.active) throw std::logic_error("history staging grows while a command streams");
  // One admission of every buffer's extents, so that a refusal asks for all
  // of them; a driver's refusal of one changes nothing: what this call
  // allocated goes back.
  uint32_t buffer = 0, extent = impl.extents;
  const metal::AllocationResult result = impl.admission(shortfallBytes(pages), [&] {
    try {
      for (buffer = 0; buffer < impl.buffers; ++buffer)
        for (extent = impl.extents; extent < wanted; ++extent)
          static_cast<void>(impl.staging[buffer]->allocateExtent(extent, kv::PageForm::Full, metal::admittedWithin));
    } catch (const metal::MetalAllocationError &) {
      for (uint32_t back = 0; back <= buffer; ++back)
        for (uint32_t allocated = impl.extents; allocated < (back == buffer ? extent : wanted); ++allocated)
          impl.staging[back]->releaseExtent(allocated);
      throw;
    }
  });
  if (!result) return {.ok = false, .allocationFailure = result.failure};
  impl.extents = wanted;
  std::vector<uint32_t> staged(impl.stagingPages());
  std::iota(staged.begin(), staged.end(), 0u);
  for (uint32_t each = 0; each < impl.buffers; ++each)
    impl.staging[each]->writeEntries(staged, 0, impl.tables[each]);
  return {.ok = true};
}

uint64_t KvHistoryStream::releaseIdle() {
  Impl &impl = *impl_;
  if (impl.run.active) throw std::logic_error("history staging released while a command streams");
  uint64_t bytes = 0;
  for (const std::unique_ptr<kv::PageStorage> &staging : impl.staging) {
    for (uint32_t extent = 0; extent < impl.extents; ++extent) staging->releaseExtent(extent);
    bytes += impl.extents * staging->extentBytes();
  }
  impl.extents = 0;
  return bytes;
}

void KvHistoryStream::begin(const KvPageSources &sources, uint32_t commandPages, const metal::MetalBuffer &poolTable,
                            std::span<const SplashKvLayer> poolLayers, AttentionLayers layers) {
  Impl &impl = *impl_;
  if (impl.run.active) throw std::logic_error("a history stream begins before the last command's ended");
  if (poolLayers.size() != impl.format.layout().attentionLayers || commandPages > impl.stagingPages())
    throw std::logic_error("a history stream's layers or staging do not hold the command's pages");
  if (layers.first >= layers.end || layers.end > poolLayers.size())
    throw std::logic_error("a history stream runs no layer or layers past the pool's");
  auto pages = std::make_shared<std::vector<Impl::Page>>();
  for (uint32_t page = 0; page < std::min(commandPages, sources.size()); ++page) {
    const KvPageSource &source = sources[page];
    if (!source.indexed()) continue;
    if (!source.copy) throw std::logic_error("a page that holds only its index has no tier copy");
    pages->push_back({page, source.block, source.copy});
  }
  const std::lock_guard lock(impl.mutex);
  impl.run = {.serial = impl.run.serial + 1,
              .active = true,
              .range = layers,
              .commandPages = commandPages,
              .pages = std::move(pages),
              .poolTable = poolTable,
              .poolLayers = {poolLayers.begin(), poolLayers.end()},
              .layers = std::vector<Impl::LayerReads>(poolLayers.size())};
  impl.startLocked(layers.first);
}

PrefillKv KvHistoryStream::addLayer(metal::CommandGraph &graph, uint32_t layer) {
  const Impl &impl = *impl_;
  const Impl::Run &run = impl.run;
  if (!impl.runs(layer)) throw std::logic_error("a history layer outside its stream");
  const uint32_t buffer = layer % impl.buffers;
  // With two buffers the next layer's reads go on while this layer attends.
  std::function<void()> then;
  if (impl.buffers == 2 && layer + 1 < run.range.end)
    then = [impl = impl_, serial = run.serial, layer] { impl->start(serial, layer + 1); };
  graph.host([impl = impl_, serial = run.serial, layer] { impl->finish(serial, layer); }, {},
             metal::HostStepGpu::Idle, std::move(then));
  return {impl.tables[buffer], impl.staging[buffer]->layers().front()};
}

void KvHistoryStream::addChunk(metal::CommandGraph &graph, uint32_t layer, uint32_t firstPage, uint32_t pages) {
  Impl &impl = *impl_;
  Impl::Run &run = impl.run;
  if (!impl.runs(layer)) throw std::logic_error("a history chunk outside its stream");
  if (!pages || uint64_t{firstPage} + pages > run.commandPages)
    throw std::logic_error("a history chunk past the command's pages");
  const auto indexed = std::ranges::lower_bound(*run.pages, firstPage, {}, &Impl::Page::page);
  if (indexed != run.pages->end() && indexed->page < firstPage + pages)
    throw std::logic_error("a prefill writes a page that holds only its index");
  // The layer's first chunk, whose first page ends the history, stages the
  // history's Full pages with its own.
  bool history = false;
  {
    const std::lock_guard lock(impl.mutex);
    history = !std::exchange(run.layers[layer].staged, true);
  }
  const uint32_t first = history ? 0 : firstPage, buffer = layer % impl.buffers;
  ops::PagedAttention::addStageLayer(graph, run.poolLayers[layer], run.poolTable,
                                     impl.staging[buffer]->layers().front(), impl.tables[buffer], first,
                                     firstPage + pages - first, impl.format.layout());
}

void KvHistoryStream::addLayerEnd(metal::CommandGraph &graph, uint32_t layer) {
  const Impl &impl = *impl_;
  const Impl::Run &run = impl.run;
  if (!impl.runs(layer)) throw std::logic_error("a history layer outside its stream");
  // With one buffer the next layer's reads wait for this layer's attention.
  if (impl.buffers != 1 || layer + 1 == run.range.end) return;
  graph.host([] {}, {}, metal::HostStepGpu::Idle,
             [impl = impl_, serial = run.serial, layer] { impl->start(serial, layer + 1); });
}

KvStreamResult KvHistoryStream::end() {
  Impl &impl = *impl_;
  const std::lock_guard lock(impl.mutex);
  Impl::Run &run = impl.run;
  if (!run.active) return {};
  impl.drainLocked();
  KvStreamResult result;
  std::vector<uint64_t> unreadable;
  for (const Impl::LayerReads &reads : run.layers) {
    if (!reads.batch) continue;
    const SsdReaders::Batch &batch = *reads.batch;
    const size_t taken = batch.taken(), failed = batch.tally().failures;
    ++impl.counts.layers;
    impl.counts.pages += taken - failed;
    result.readBytes += (taken - failed) * impl.layerBytes;
    for (size_t index = 0; failed && index < taken; ++index)
      if (batch.failed(index)) unreadable.push_back((*run.pages)[index].block);
  }
  impl.counts.bytes += result.readBytes;
  ++impl.counts.commands;
  // A page that failed in several layers is one block.
  std::ranges::sort(unreadable);
  unreadable.erase(std::unique(unreadable.begin(), unreadable.end()), unreadable.end());
  for (const uint64_t block : unreadable) result.failures.push_back({0, block});
  run = {.serial = run.serial};
  return result;
}

KvHistoryStreamStats KvHistoryStream::stats() const {
  const Impl &impl = *impl_;
  KvHistoryStreamStats result = impl.counts;
  result.stagingBytes = impl.extents * impl.buffers * impl.staging.front()->extentBytes();
  for (const metal::MetalBuffer &table : impl.tables) result.tableBytes += table.sizeBytes();
  result.read = impl.readTimes.snapshot();
  result.wait = impl.waits.snapshot();
  return result;
}

} // namespace splash::model
