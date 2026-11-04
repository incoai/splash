// The history stream of a prefill whose pages hold only their index
// (model::KvHistoryStream), on Qwen3.8-Flash-Next's attention at three
// layers, int8 and bf16, with one staging buffer and with two: each layer's
// staged pages hold the pool's bytes, its Index pages read from their tier
// copies (the pool's K/V of them overwritten and their entries absent), its
// Full pages and the command's chunks copied on the GPU, every layer checked
// before a later one reuses its buffer. A page that cannot be read leaves
// zeros and fails once, whatever layers; staging the budget refuses changes
// nothing; a stream begins only once the last ended, ends a command given up
// before it ran, and one given up while a host step waits on its reads
// without the reads not yet taken, drains when it goes with reads in flight,
// and refuses chunks past the command's pages and a prefill that writes an
// Index page. Passes over some of the layers each stage and read what one
// command over all of them does of their layers, and none starts reads past
// its last layer; reserve() takes what shortfallBytes() says.
#include "TestAdmission.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestKvTier.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/KvHistoryStream.hpp"
#include "ops/PageStorage.hpp"
#include "ops/PagedAttention.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace splash;
using metal::MetalBackend;
using metal::MetalBuffer;
using model::KvHistoryStream;
using model::KvPageSource;
using model::KvPageSources;
using model::KvStreamResult;
using test::rejects;
using test::require;
using test::sourcesOf;
using test::TestKvCopy;

constexpr uint32_t kExtentPages = 256, kLayers = 3;
// A sequence of 100 pages: 80 of history, then a command of two chunks of
// 10 pages each.
constexpr uint32_t kHistoryPages = 80, kPages = 100, kChunkPages = 10;

constexpr kv::Layout layoutOf(kv::Format format) {
  return {kLayers, 2, 256, format, 128, true, 0, 0, kv::Reads::SelectedBlocks};
}

std::vector<std::byte> layerBytes(const kv::PageStorage &storage, uint32_t page, uint32_t layer) {
  std::vector<std::byte> bytes;
  for (const std::span<std::byte> tensor : storage.layerSpans(page, layer))
    bytes.insert(bytes.end(), tensor.begin(), tensor.end());
  return bytes;
}

// Holds the reads of the copies that share it until it opens.
struct Gate final {
  // A read waits here.
  void pass() {
    std::unique_lock lock(mutex);
    ++waiting;
    changed.notify_all();
    changed.wait(lock, [this] { return open; });
  }
  // Whether `reads` reads wait, within ten seconds.
  bool hold(uint32_t reads) {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, std::chrono::seconds(10), [&] { return waiting >= reads; });
  }
  void release() {
    {
      const std::lock_guard lock(mutex);
      open = true;
    }
    changed.notify_all();
  }

  std::mutex mutex;
  std::condition_variable changed;
  uint32_t waiting = 0;
  bool open = false;
};

class GatedKvCopy final : public TestKvCopy {
public:
  GatedKvCopy(std::vector<std::byte> content, std::shared_ptr<Gate> gate)
      : TestKvCopy(std::move(content)), gate_(std::move(gate)) {}

  std::optional<std::span<const std::byte>> readAt(uint64_t offset, uint64_t count,
                                                   std::span<std::byte> buffer) const noexcept override {
    gate_->pass();
    return TestKvCopy::readAt(offset, count, buffer);
  }

private:
  std::shared_ptr<Gate> gate_;
};

// A sequence of the pool's pages, mixed over its two extents, its history's
// pages from the fourth on but each third holding only their index: their
// tier copies hold the pool's bytes, which are then overwritten, and the
// sequence's table names none of them.
struct Sequence final {
  Sequence(MetalBackend &backend, kv::Layout layout)
      : layout(layout), pool(backend, test::admitAll, layout, 2 * kExtentPages, kExtentPages),
        table(test::sharedBuffer(backend, kPages * sizeof(SplashKvPage))) {
    std::mt19937 random(11);
    for (uint32_t extent = 0; extent < 2; ++extent)
      require(static_cast<bool>(pool.allocateExtent(extent)), "the pool's extents were refused");
    std::vector<KvPageSource> sources(kPages);
    for (uint32_t page = 0; page < kPages; ++page) {
      pages.push_back(page * 5 % (2 * kExtentPages));
      for (const std::span<std::byte> tensor : pool.spans(pages.back()))
        for (std::byte &byte : tensor) byte = std::byte(random());
      expected.emplace_back();
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        expected.back().push_back(layerBytes(pool, pages.back(), layer));
    }
    pool.writeEntries(pages, 0, table);
    const kv::SlotFormat format(layout);
    for (uint32_t page = 4; page < kHistoryPages; ++page) {
      if (page % 3 == 0) continue;
      std::vector<std::byte> payload;
      for (const auto &span : format.transferSpans(pool, pages[page]))
        payload.insert(payload.end(), span.begin(), span.end());
      copies.push_back(std::make_shared<TestKvCopy>(std::move(payload)));
      sources[page] = {1000 + page, copies.back()};
      for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer)
        for (const std::span<std::byte> tensor : pool.layerSpans(pages[page], layer))
          std::ranges::fill(tensor, std::byte{0x5a});
      static_cast<SplashKvPage *>(table.contents())[page] = SPLASH_KV_PAGE_ABSENT;
    }
    this->sources = sourcesOf(sources);
  }

  kv::Layout layout;
  kv::PageStorage pool;
  MetalBuffer table;
  std::vector<uint32_t> pages;
  // Each page's bytes of each layer, before the pool's K/V of the Index
  // pages was overwritten.
  std::vector<std::vector<std::vector<std::byte>>> expected;
  std::vector<std::shared_ptr<TestKvCopy>> copies;
  std::shared_ptr<const KvPageSources> sources;
};

KvHistoryStream makeStream(MetalBackend &backend, kv::Layout layout, uint32_t buffers,
                           metal::AllocationAdmission admission = test::admitAll) {
  return KvHistoryStream(backend, std::move(admission), layout, kExtentPages,
                         std::make_shared<model::SsdReaders>(8), kv::SlotFormat(layout), buffers);
}

// A prefill command over the sequence, or a pass of it over `layers`, as a
// target's layers encode it, each layer's staged pages copied to `readback`,
// if any (one layer a buffer's), before its attention would read them.
metal::CommandGraph prefillGraph(KvHistoryStream &stream, const Sequence &sequence,
                                 const kv::PageStorage *readback = nullptr, const MetalBuffer &readbackTable = {},
                                 model::AttentionLayers layers = {0, kLayers}) {
  metal::CommandGraph graph;
  for (uint32_t layer = layers.first; layer < layers.end; ++layer) {
    const model::PrefillKv staged = stream.addLayer(graph, layer);
    for (uint32_t first = kHistoryPages; first < kPages; first += kChunkPages)
      stream.addChunk(graph, layer, first, kChunkPages);
    if (readback)
      ops::PagedAttention::addStageLayer(graph, staged.kv, staged.table, readback->layers()[layer], readbackTable,
                                         0, kPages, sequence.layout);
    stream.addLayerEnd(graph, layer);
  }
  return graph;
}

// One prefill command, or a pass of it over `layers`; returns what the
// stream's end says. A host step and the readback's copy of a page after its
// last layer hold the command's end until the steps before them ran their
// `then` (the backend's runner runs each step's `then` before the next
// step): what the command's steps started has started when it ends.
KvStreamResult prefill(MetalBackend &backend, KvHistoryStream &stream, const Sequence &sequence,
                       const kv::PageStorage &readback, const MetalBuffer &readbackTable,
                       model::AttentionLayers layers = {0, kLayers}) {
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), layers);
  metal::CommandGraph graph = prefillGraph(stream, sequence, &readback, readbackTable, layers);
  graph.host([] {});
  ops::PagedAttention::addStageLayer(graph, sequence.pool.layers()[0], sequence.table, readback.layers()[0],
                                     readbackTable, 0, 1, sequence.layout);
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  return stream.end();
}

// The staging that reads the staged pages back, one page per page of the
// sequence, at page j.
struct Readback final {
  Readback(MetalBackend &backend, kv::Layout layout)
      : storage(backend, test::admitAll, withoutIndex(layout), kExtentPages, kExtentPages),
        table(test::sharedBuffer(backend, kPages * sizeof(SplashKvPage))) {
    require(static_cast<bool>(storage.allocateExtent(0)), "the readback extent was refused");
    std::vector<uint32_t> pages(kPages);
    std::iota(pages.begin(), pages.end(), 0u);
    storage.writeEntries(pages, 0, table);
  }
  static kv::Layout withoutIndex(kv::Layout layout) {
    layout.indexDimension = 0;
    return layout;
  }

  kv::PageStorage storage;
  MetalBuffer table;
};

void streams(MetalBackend &backend, kv::Format format, uint32_t buffers) {
  const kv::Layout layout = layoutOf(format);
  Sequence sequence(backend, layout);
  KvHistoryStream stream = makeStream(backend, layout, buffers);
  // holds() tells reserve() it would take nothing, by whole extents.
  require(!stream.holds(1), "staging nothing reserved held a page");
  require(stream.reserve(kPages).granted() && stream.holds(kPages) && stream.holds(kExtentPages) &&
              !stream.holds(kExtentPages + 1) &&
              stream.stats().stagingBytes == uint64_t{buffers} * kExtentPages * layout.bytesPerLayerPage() &&
              stream.stats().buffers == buffers,
          "staging does not hold a sequence's pages in each buffer");
  kv::Layout readbackLayout = layout;
  readbackLayout.indexDimension = 0;
  kv::PageStorage readback(backend, test::admitAll, readbackLayout, kExtentPages, kExtentPages);
  require(static_cast<bool>(readback.allocateExtent(0)), "the readback extent was refused");
  const MetalBuffer readbackTable = test::sharedBuffer(backend, kPages * sizeof(SplashKvPage));
  std::vector<uint32_t> readbackPages(kPages);
  std::iota(readbackPages.begin(), readbackPages.end(), 0u);
  readback.writeEntries(readbackPages, 0, readbackTable);
  // A page that cannot be read in either layer fails once.
  constexpr uint32_t kUnreadable = 11;
  const auto unreadable = std::ranges::find_if(
      sequence.copies, [&](const auto &copy) { return copy == (*sequence.sources)[kUnreadable].copy; });
  require(unreadable != sequence.copies.end(), "the unreadable page has no copy");
  const uint64_t unreadableBlock = (*sequence.sources)[kUnreadable].block;
  (*unreadable)->failAll = true;
  const KvStreamResult result = prefill(backend, stream, sequence, readback, readbackTable);
  for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer)
    for (uint32_t page = 0; page < kPages; ++page) {
      const std::vector<std::byte> staged = layerBytes(readback, page, layer);
      if (page == kUnreadable)
        require(std::ranges::all_of(staged, [](std::byte byte) { return byte == std::byte{0}; }),
                "a page that could not be read left bytes in staging");
      else
        require(staged == sequence.expected[page][layer],
                "a staged page differs from its pool page: layer " + std::to_string(layer) + ", page " +
                    std::to_string(page));
    }
  require(result.failures == std::vector<model::KvReadFailure>{{0, unreadableBlock}},
          "a page that could not be read did not fail once");
  const uint64_t index = sequence.copies.size();
  const model::KvHistoryStreamStats stats = stream.stats();
  require(stats.commands == 1 && stats.layers == kLayers && stats.pages == (index - 1) * kLayers &&
              stats.bytes == stats.pages * kv::SlotFormat(layout).layerRange(0).bytes && stats.wait.count == kLayers &&
              stats.read.count == kLayers && stats.read.totalMilliseconds > 0,
          "the stream's counts are off");
  for (const auto &copy : sequence.copies)
    require(copy->reads() == (copy == *unreadable ? 2 : 1) * kLayers,
            "a page was not read once a layer, twice on failure");
  // The next turn reuses staging.
  (*unreadable)->failAll = false;
  const KvStreamResult next = prefill(backend, stream, sequence, readback, readbackTable);
  require(next.failures.empty() && layerBytes(readback, kUnreadable, 2) == sequence.expected[kUnreadable][2] &&
              stream.stats().commands == 2,
          "a second command did not stream the history again");
  require(stream.releaseIdle() == stats.stagingBytes && stream.stats().stagingBytes == 0 &&
              !stream.holds(1),
          "an idle reclaim did not give staging back");
  std::cout << "kv_history_stream_test: " << kv::formatName(format) << ", " << buffers << " buffer"
            << (buffers == 1 ? "" : "s") << ", " << index << " pages read a layer PASS\n";
}

// A command run in passes over the layers [0, h) and [h, kLayers), each its
// own command, stages what a command over all of them stages, byte for byte,
// and reads each page that holds only its index once a layer, as that
// command does: each pass starts its first layer's reads and none past its
// last layer's, with one buffer or two; each says the bytes it read, its
// pages times a page's of a layer. A layer outside the pass is none of its
// stream's.
void passesStageWhatOneCommandStages(MetalBackend &backend, uint32_t buffers) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  Sequence sequence(backend, layout);
  KvHistoryStream stream = makeStream(backend, layout, buffers);
  require(stream.reserve(kPages).granted(), "staging was refused");
  const uint64_t pageBytes = kv::SlotFormat(layout).layerRange(0).bytes;
  const uint64_t index = sequence.copies.size();
  for (uint32_t split = 1; split < kLayers; ++split) {
    Readback readback(backend, layout);
    const model::KvHistoryStreamStats before = stream.stats();
    std::vector<uint64_t> reads;
    for (const auto &copy : sequence.copies) reads.push_back(copy->reads());
    for (const model::AttentionLayers pass :
         {model::AttentionLayers{0, split}, model::AttentionLayers{split, kLayers}}) {
      const model::KvHistoryStreamStats started = stream.stats();
      const KvStreamResult result = prefill(backend, stream, sequence, readback.storage, readback.table, pass);
      const model::KvHistoryStreamStats ended = stream.stats();
      require(result.failures.empty() && ended.layers - started.layers == pass.end - pass.first &&
                  ended.pages - started.pages == index * (pass.end - pass.first),
              "a pass over layers [" + std::to_string(pass.first) + ", " + std::to_string(pass.end) + ") with " +
                  std::to_string(buffers) + " buffers read other layers than its own");
      require(result.readBytes == (ended.pages - started.pages) * pageBytes &&
                  ended.bytes - started.bytes == result.readBytes,
              "a pass's read bytes are not its pages' records of a layer");
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      for (uint32_t page = 0; page < kPages; ++page)
        require(layerBytes(readback.storage, page, layer) == sequence.expected[page][layer],
                "two passes staged other bytes than one command: layer " + std::to_string(layer) + ", page " +
                    std::to_string(page));
    for (size_t copy = 0; copy < sequence.copies.size(); ++copy)
      require(sequence.copies[copy]->reads() - reads[copy] == kLayers,
              "two passes read a page other than once a layer");
    require(stream.stats().commands == before.commands + 2, "two passes were not two commands");
  }
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), {1, 2});
  metal::CommandGraph outside;
  rejects([&] { static_cast<void>(stream.addLayer(outside, 0)); }, "outside its stream",
          "a pass staged a layer before its own");
  rejects([&] { static_cast<void>(stream.addLayer(outside, 2)); }, "outside its stream",
          "a pass staged a layer past its own");
  rejects([&] { stream.addChunk(outside, 2, kHistoryPages, kChunkPages); }, "outside its stream",
          "a pass staged a chunk of a layer past its own");
  static_cast<void>(stream.end());
  for (const model::AttentionLayers none : {model::AttentionLayers{1, 1}, model::AttentionLayers{2, kLayers + 1}})
    rejects([&] { stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), none); },
            "no layer or layers past", "a stream began over no layer or past the pool's");
  std::cout << "kv_history_stream_test: passes with " << buffers << " buffer" << (buffers == 1 ? "" : "s")
            << " PASS\n";
}

// reserve() allocates in each buffer the extents shortfallBytes() says it
// lacks, under one admission of them all, and then lacks none.
void reservesItsShortfall(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  uint64_t allocated = 0;
  uint32_t admissions = 0;
  const metal::AllocationAdmission counting = [&](uint64_t bytes, const std::function<void()> &allocate) {
    allocated += bytes;
    ++admissions;
    return test::admitAll(bytes, allocate);
  };
  KvHistoryStream stream = makeStream(backend, layout, 2, counting);
  for (const uint32_t pages : {1u, kExtentPages, kExtentPages + 1, 3 * kExtentPages}) {
    const uint64_t shortfall = stream.shortfallBytes(pages);
    allocated = 0;
    admissions = 0;
    require((shortfall == 0) == stream.holds(pages) && stream.reserve(pages).granted() && allocated == shortfall &&
                !stream.shortfallBytes(pages) && stream.holds(pages),
            "staging for " + std::to_string(pages) + " pages did not allocate its shortfall");
    require(admissions == (shortfall ? 1u : 0u), "staging for " + std::to_string(pages) + " pages took " +
                                                     std::to_string(admissions) + " admissions, not one of its shortfall");
  }
  require(stream.shortfallBytes(1) == 0 && stream.releaseIdle() &&
              stream.shortfallBytes(1) == 2 * kExtentPages * layout.bytesPerLayerPage(),
          "staging given back lacked other than an extent in each buffer");
  std::cout << "kv_history_stream_test: shortfall PASS\n";
}

void lifecycle(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  Sequence sequence(backend, layout);
  // Staging the budget refuses changes nothing, with room for one buffer's
  // extent of two: refused, it asked for both, which is what the ledger
  // frees for.
  uint64_t room = kExtentPages * layout.bytesPerLayerPage();
  std::vector<uint64_t> asked;
  KvHistoryStream refused = makeStream(backend, layout, 2, [&](uint64_t bytes, const std::function<void()> &allocate) {
    asked.push_back(bytes);
    if (bytes > room) return metal::AllocationResult(metal::AllocationFailure::EngineBudget);
    room -= bytes;
    return test::admitAll(bytes, allocate);
  });
  const uint64_t shortfall = refused.shortfallBytes(kPages);
  const model::ModelMemoryAdmission admission = refused.reserve(kPages);
  require(!admission.granted() && admission.allocationFailure == metal::AllocationFailure::EngineBudget &&
              refused.stats().stagingBytes == 0,
          "refused staging kept an extent");
  require(!asked.empty() && asked.back() == shortfall,
          "refused staging asked its admission for " + std::to_string(asked.empty() ? 0 : asked.back()) +
              " bytes at once, not the " + std::to_string(shortfall) + " it lacks");
  rejects([&] { refused.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), {0, kLayers}); },
          "staging do not hold", "a stream began without staging for its pages");
  KvHistoryStream stream = makeStream(backend, layout, 1);
  require(stream.reserve(kPages).granted(), "staging was refused");
  // A command given up before it ran ends with its reads stopped, and the
  // next begins.
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), {0, kLayers});
  rejects([&] { stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), {0, kLayers}); },
          "before the last command's ended", "a stream began twice");
  rejects([&] { static_cast<void>(stream.releaseIdle()); }, "while a command streams",
          "staging was released while a command streamed");
  // A chunk lies within the command's pages, and a prefill does not write a
  // page that holds only its index.
  metal::CommandGraph refusedChunks;
  rejects([&] { stream.addChunk(refusedChunks, 0, kPages - 5, kChunkPages); }, "past the command's pages",
          "a chunk past the command's pages was staged");
  rejects([&] { stream.addChunk(refusedChunks, 0, kHistoryPages - kChunkPages, kChunkPages); },
          "writes a page that holds only its index", "a prefill wrote an Index page");
  require(stream.end().failures.empty() && stream.end().failures.empty(), "a command given up failed a page");
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), {0, kLayers});
  require(stream.end().failures.empty(), "a stream did not begin again once the last ended");
  // A stream that goes with reads in flight drains them first.
  {
    KvHistoryStream going = makeStream(backend, layout, 2);
    require(going.reserve(kPages).granted(), "staging was refused");
    going.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers(), {0, kLayers});
  }
  for (const auto &copy : sequence.copies)
    require(copy->reads() <= 3, "a page was read past the commands that read it");
  std::cout << "kv_history_stream_test: lifecycle PASS\n";
}

// A command given up while its first layer's host step finishes the layer's
// reads, on the eight readers and its own thread, nine reads at the gate:
// end() leaves the others unread and returns once those nine ended.
void endsTheReadsOfACommandGivenUp(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  Sequence sequence(backend, layout);
  const auto gate = std::make_shared<Gate>();
  std::vector<KvPageSource> pages(kPages);
  std::vector<std::shared_ptr<GatedKvCopy>> copies;
  for (uint32_t page = 0; page < kPages; ++page) {
    const KvPageSource &source = (*sequence.sources)[page];
    if (!source.indexed()) continue;
    copies.push_back(
        std::make_shared<GatedKvCopy>(static_cast<const TestKvCopy &>(*source.copy).bytes, gate));
    pages[page] = {source.block, copies.back()};
  }
  const std::shared_ptr<const KvPageSources> sources = sourcesOf(pages);
  KvHistoryStream stream = makeStream(backend, layout, 1);
  require(stream.reserve(kPages).granted(), "staging was refused");
  stream.begin(*sources, kPages, sequence.table, sequence.pool.layers(), {0, kLayers});
  const metal::CommandGraph graph = prefillGraph(stream, sequence);
  metal::CommandTicket command = backend.submitCommandAsync(graph.command());
  constexpr uint32_t kTaken = 9;
  const bool held = gate->hold(kTaken);
  std::thread opener([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    gate->release();
  });
  const KvStreamResult result = held ? stream.end() : KvStreamResult{};
  opener.join();
  static_cast<void>(command.wait());
  require(held, "the first layer's host step did not read with the readers");
  uint64_t reads = 0;
  for (const auto &copy : copies) reads += copy->reads();
  require(result.failures.empty() && reads == kTaken && stream.stats().pages == kTaken &&
              stream.stats().read.count == 0,
          "a command given up read past the reads its step had taken, or timed them");
  std::cout << "kv_history_stream_test: a command given up PASS\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: kv_history_stream_test METALLIB\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16})
        for (const uint32_t buffers : {1u, 2u}) streams(backend, format, buffers);
      for (const uint32_t buffers : {1u, 2u}) passesStageWhatOneCommandStages(backend, buffers);
      reservesItsShortfall(backend);
      lifecycle(backend);
      endsTheReadsOfACommandGivenUp(backend);
      std::cout << "kv_history_stream_test: PASS\n";
    } catch (const std::exception &error) {
      std::cerr << "kv_history_stream_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
