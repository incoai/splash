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
// Index page.
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

constexpr kv::Layout layoutOf(kv::Format format) { return {kLayers, 2, 256, format, 128, true}; }

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

// A prefill command over the sequence, as a target's layers encode it, each
// layer's staged pages copied to `readback`, if any (one layer a buffer's),
// before its attention would read them.
metal::CommandGraph prefillGraph(KvHistoryStream &stream, const Sequence &sequence,
                                 const kv::PageStorage *readback = nullptr, const MetalBuffer &readbackTable = {}) {
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < sequence.layout.attentionLayers; ++layer) {
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

// One prefill command; returns what the stream's end says.
KvStreamResult prefill(MetalBackend &backend, KvHistoryStream &stream, const Sequence &sequence,
                       const kv::PageStorage &readback, const MetalBuffer &readbackTable) {
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers());
  const metal::CommandGraph graph = prefillGraph(stream, sequence, &readback, readbackTable);
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  return stream.end();
}

void streams(MetalBackend &backend, kv::Format format, uint32_t buffers) {
  const kv::Layout layout = layoutOf(format);
  Sequence sequence(backend, layout);
  KvHistoryStream stream = makeStream(backend, layout, buffers);
  require(stream.reserve(kPages).granted() &&
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
  require(stream.releaseIdle() == stats.stagingBytes && stream.stats().stagingBytes == 0,
          "an idle reclaim did not give staging back");
  std::cout << "kv_history_stream_test: " << kv::formatName(format) << ", " << buffers << " buffer"
            << (buffers == 1 ? "" : "s") << ", " << index << " pages read a layer PASS\n";
}

void lifecycle(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  Sequence sequence(backend, layout);
  // Staging the budget refuses changes nothing, the second buffer's extent
  // refused too.
  uint32_t grants = 1;
  KvHistoryStream refused = makeStream(backend, layout, 2, [&](uint64_t bytes, const std::function<void()> &allocate) {
    if (!grants) return metal::AllocationResult(metal::AllocationFailure::EngineBudget);
    --grants;
    return test::admitAll(bytes, allocate);
  });
  const model::ModelMemoryAdmission admission = refused.reserve(kPages);
  require(!admission.granted() && admission.allocationFailure == metal::AllocationFailure::EngineBudget &&
              refused.stats().stagingBytes == 0,
          "refused staging kept an extent");
  rejects([&] { refused.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers()); },
          "staging do not hold", "a stream began without staging for its pages");
  KvHistoryStream stream = makeStream(backend, layout, 1);
  require(stream.reserve(kPages).granted(), "staging was refused");
  // A command given up before it ran ends with its reads stopped, and the
  // next begins.
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers());
  rejects([&] { stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers()); },
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
  stream.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers());
  require(stream.end().failures.empty(), "a stream did not begin again once the last ended");
  // A stream that goes with reads in flight drains them first.
  {
    KvHistoryStream going = makeStream(backend, layout, 2);
    require(going.reserve(kPages).granted(), "staging was refused");
    going.begin(*sequence.sources, kPages, sequence.table, sequence.pool.layers());
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
  stream.begin(*sources, kPages, sequence.table, sequence.pool.layers());
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
