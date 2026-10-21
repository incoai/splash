// The hot block cache of spilled K/V (model::KvBlockCache) on Qwen3.8-Flash-
// Next's attention at two layers: a resolve leaves a block of a Full page to
// the page table (SPLASH_ATTENTION_PAGED), as it does every block of a lane
// none of whose pages holds only its index, and places every other in a slot
// of its layer's: a block it holds stays where it is, and a block it misses is
// read once, whatever lanes list it, a page's misses in one read from its
// first missed block to its last; it keys a block by its tier copy and
// refuses an Index page without one. It refuses a resolve that misses more
// blocks than it has slots it does not use, before it claims one. It grows to
// the floors of the lanes that hold one, then toward its target an extent at
// a time while any does, keeps its blocks when it grows, gives back what is
// above the floors, and a refusal changes nothing. A block read twice in vain
// fails every lane that lists it, once, and the next resolve reads it again;
// one read once more succeeds. And, int8 and bf16, a verify's attention
// through the resolve's locations, with the pool's K/V of its Index pages
// overwritten, gives the bits it gives through the page tables alone, also
// once its blocks' copies changed after the cache read them (an adoption,
// which keeps a block's id) and from cached blocks.
#include "AttentionFixture.hpp"
#include "TestAdmission.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "TestKvTier.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/KvBlockCache.hpp"
#include "ops/PagedAttention.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using namespace splash;
using metal::MetalBackend;
using metal::MetalBuffer;
using model::KvBlockCache;
using model::KvPageSource;
using model::KvPageSources;
using model::KvReadFailure;
using model::KvReadStats;
using test::rejects;
using test::require;
using test::sourcesOf;
using test::TestKvCopy;

constexpr uint32_t kExtentPages = 256, kPlaces = 8, kLaneSlots = 1024;
// An extent's slots per layer.
constexpr uint32_t kExtentSlots = kExtentPages * kPlaces;

constexpr kv::Layout layoutOf(kv::Format format) { return {2, 2, 256, format, 128, true}; }

uint64_t extentBytes(kv::Layout layout) { return uint64_t{kExtentPages} * layout.kvBytesPerModelPage(); }

KvBlockCache makeCache(MetalBackend &backend, kv::Layout layout, uint32_t targetExtents,
                       metal::AllocationAdmission admission = test::admitAll, uint32_t laneSlots = kLaneSlots) {
  return KvBlockCache(backend, std::move(admission), layout, kExtentPages,
                      {.laneSlots = laneSlots,
                       .targetBytes = targetExtents * extentBytes(layout),
                       .readers = std::make_shared<model::SsdReaders>(8),
                       .format = kv::SlotFormat(layout)});
}

// A page's tier copy, whose bytes a resolve reads into its slots.
std::shared_ptr<TestKvCopy> copyOf(kv::Layout layout) {
  return std::make_shared<TestKvCopy>(std::vector<std::byte>(kv::SlotFormat(layout).payloadBytes(), std::byte{7}));
}

// Every block of pages [first, end).
std::vector<uint32_t> blocksOf(uint32_t first, uint32_t end) {
  std::vector<uint32_t> blocks;
  for (uint32_t block = first * kPlaces; block < end * kPlaces; ++block) blocks.push_back(block);
  return blocks;
}

using Lanes = std::vector<std::shared_ptr<const KvPageSources>>;

// Resolves each lane's list of `blocks`, ascending, which every verify row
// selects, in a command of its own that makes the lists first, as QSA's
// selection does (PagedAttention::addBlockLists of the rows' masks, the rows
// past every block); returns each lane's locations.
std::vector<std::vector<uint32_t>> resolve(MetalBackend &backend, KvBlockCache &cache, uint32_t layer,
                                           const Lanes &lanes, const std::vector<std::vector<uint32_t>> &blocks,
                                           const std::shared_ptr<KvReadStats> &stats) {
  constexpr uint32_t kRows = kv::kVerifyRows;
  uint32_t history = 0;
  for (const std::vector<uint32_t> &list : blocks)
    if (!list.empty()) history = std::max(history, (list.back() + 1) * SPLASH_ATTENTION_BLOCK_TOKENS);
  const uint32_t stride = history / SPLASH_ATTENTION_BLOCK_TOKENS + kRows,
                 words = (history + kRows) / SPLASH_ATTENTION_BLOCK_TOKENS / 32 + 1;
  const uint64_t entries = uint64_t{stride} * lanes.size();
  const MetalBuffer masks = test::sharedBuffer(backend, lanes.size() * kRows * words * sizeof(uint32_t)),
                    lists = test::sharedBuffer(backend, entries * sizeof(uint32_t)),
                    counts = test::sharedBuffer(backend, lanes.size() * sizeof(uint32_t)),
                    locations = test::sharedBuffer(backend, entries * sizeof(uint32_t));
  std::memset(masks.contents(), 0, masks.sizeBytes());
  std::memset(locations.contents(), 0xfe, locations.sizeBytes());
  std::vector<ops::BlockListRows> sequences;
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    for (uint32_t row = 0; row < kRows; ++row)
      for (const uint32_t block : blocks[lane])
        static_cast<uint32_t *>(masks.contents())[(lane * kRows + row) * words + block / 32] |= 1u << (block % 32);
    sequences.push_back({lane * kRows, kRows, history, lane});
  }
  metal::CommandGraph graph;
  const ops::AttentionBlocks lanesBlocks{lists, counts, stride, 0};
  ops::PagedAttention::addBlockLists(graph, masks, words, sequences, lanesBlocks);
  const ops::AttentionBlocks located = cache.addResolve(graph, layer, lanes, lanesBlocks, locations, stats);
  require(located.locations.sameView(locations) && located.locationPages && located.lists.sameView(lists) &&
              located.stride == stride,
          "a resolve did not hand its lists on with their locations");
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  std::vector<std::vector<uint32_t>> result(lanes.size());
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    const auto *at = static_cast<const uint32_t *>(locations.contents()) + lane * stride;
    result[lane].assign(at, at + blocks[lane].size());
  }
  return result;
}

void resolvesEntries(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  KvBlockCache cache = makeCache(backend, layout, 2);
  require(cache.reserveLane(0).granted() && cache.reserveLane(1).granted() &&
              cache.stats().slotsPerLayer == 2 * kExtentSlots && cache.stats().lanes == 2,
          "two lanes' floors are not two extents");
  // Lane 0's pages 2-37 hold only their index; lane 1 shares its pages 2-19
  // and holds its own pages 20-29.
  std::vector<std::shared_ptr<TestKvCopy>> copies;
  std::vector<KvPageSource> first(40), second(40);
  for (uint32_t page = 2; page < 38; ++page) {
    copies.push_back(copyOf(layout));
    first[page] = {1000 + page, copies.back()};
    if (page < 20) second[page] = first[page];
    else if (page < 30) {
      copies.push_back(copyOf(layout));
      second[page] = {2000 + page, copies.back()};
    }
  }
  const Lanes lanes{sourcesOf(first), sourcesOf(second)};
  const std::vector<std::vector<uint32_t>> blocks{blocksOf(0, 40), blocksOf(10, 30)};
  auto stats = std::make_shared<KvReadStats>();
  const auto located = resolve(backend, cache, 0, lanes, blocks, stats);
  std::set<uint32_t> slots;
  for (uint32_t entry = 0; entry < blocks[0].size(); ++entry) {
    const uint32_t page = blocks[0][entry] / kPlaces;
    if (page < 2 || page >= 38) {
      require(located[0][entry] == SPLASH_ATTENTION_PAGED, "a block of a Full page was given a slot");
      continue;
    }
    require(located[0][entry] < 2 * kExtentSlots && slots.insert(located[0][entry]).second,
            "a lane's blocks share a slot or one lies past the cache");
  }
  for (uint32_t entry = 0; entry < blocks[1].size(); ++entry) {
    const uint32_t block = blocks[1][entry];
    if (block < 20 * kPlaces)
      require(located[1][entry] == located[0][block], "two lanes listing one block were given two slots");
    else
      require(slots.insert(located[1][entry]).second, "a lane's own block shares another's slot");
  }
  require(stats->resolves == 1 && stats->entries == 480 && stats->paged == 32 && stats->misses == 368 &&
              stats->hits == 80 && stats->readBytes == 46 * kv::SlotFormat(layout).layerRange(0).bytes &&
              !stats->readFailures,
          "a resolve's counts are off");
  for (const auto &copy : copies) require(copy->reads() == 1, "a page's missed blocks were not one read");
  // The next step finds every block where it was.
  auto again = std::make_shared<KvReadStats>();
  require(resolve(backend, cache, 0, lanes, blocks, again) == located && !again->misses && again->hits == 448,
          "a block the cache holds moved or was read again");
  // Each layer has slots of its own.
  auto other = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 1, lanes, blocks, other));
  require(other->misses == 368, "a layer found another layer's blocks");
  cache.mergeStats(*stats);
  cache.mergeStats(*again);
  const model::KvBlockCacheStats totals = cache.stats();
  require(totals.reads.resolves == 2 && totals.reads.misses == 368 && totals.reads.hits == 528 &&
              totals.resolve.count == 3 && totals.read.count == 2 && totals.work.count == 3 &&
              totals.bytes == 2 * extentBytes(layout),
          "the cache's totals are not its commands' counts");
  require(cache.takeFailures().empty(), "a resolve that read every block failed one");
  // A lane none of whose pages holds only its index reads every block
  // through its table, and so does a page whose source names no block.
  std::vector<KvPageSource> unnamed(4);
  unnamed[1] = {0, copies.front()};
  const auto paged = resolve(backend, cache, 0, {nullptr, sourcesOf(unnamed)}, {blocksOf(0, 4), blocksOf(0, 4)},
                             std::make_shared<KvReadStats>());
  for (const auto &lane : paged)
    require(std::ranges::all_of(lane, [](uint32_t location) { return location == SPLASH_ATTENTION_PAGED; }),
            "a block of a lane without Index pages was given a slot");
  std::cout << "kv_block_cache_test: resolves PASS\n";
}

// A resolve it refuses fails its command, after which the backend serves no
// other: each runs on a backend of its own. It refuses an Index page without
// its copy, and more misses than its slots: without a floor it has none;
// with one, a lane's floor and its margin, 2048.
void refusesWhatItCannotPlace(const char *metallib) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  std::vector<KvPageSource> pages(300);
  for (uint32_t page = 0; page < pages.size(); ++page) pages[page] = {1 + page, copyOf(layout)};
  const Lanes lanes{sourcesOf(pages)};
  {
    MetalBackend backend(metallib);
    KvBlockCache cache = makeCache(backend, layout, 1);
    require(cache.reserveLane(0).granted(), "a lane's floor was refused");
    std::vector<KvPageSource> copyless(4);
    copyless[2] = {3000, nullptr};
    rejects([&] { static_cast<void>(resolve(backend, cache, 0, {sourcesOf(copyless)}, {blocksOf(0, 4)},
                                            std::make_shared<KvReadStats>())); },
            "has no tier copy", "a resolve read a page that holds only its index without its copy");
  }
  {
    MetalBackend backend(metallib);
    KvBlockCache cache = makeCache(backend, layout, 1);
    rejects([&] { static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 1)},
                                            std::make_shared<KvReadStats>())); },
            "misses more blocks", "a resolve placed a block in a cache without a floor");
  }
  MetalBackend backend(metallib);
  KvBlockCache cache = makeCache(backend, layout, 1);
  require(cache.reserveLane(0).granted() && cache.stats().slotsPerLayer == kExtentSlots,
          "a lane's floor is not one extent");
  auto stats = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 256)}, stats));
  require(stats->misses == kExtentSlots, "a resolve did not fill the cache");
  std::vector<uint32_t> past = blocksOf(0, 256);
  past.push_back(256 * kPlaces);
  rejects([&] { static_cast<void>(resolve(backend, cache, 0, lanes, {past}, std::make_shared<KvReadStats>())); },
          "misses more blocks", "a resolve placed more blocks than its cache's slots");
  std::cout << "kv_block_cache_test: refusals PASS\n";
}

// A page's blocks a resolve misses in a layer are one read, from the first
// to the last, the records between included; its hits and the blocks of
// other layers stay out of it.
void readsAPagesMissesAtOnce(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const kv::SlotFormat format(layout);
  KvBlockCache cache = makeCache(backend, layout, 1);
  require(cache.reserveLane(0).granted(), "the floor was refused");
  std::vector<std::shared_ptr<TestKvCopy>> copies;
  std::vector<KvPageSource> pages(4);
  for (uint32_t page = 0; page < pages.size(); ++page) {
    copies.push_back(copyOf(layout));
    pages[page] = {1 + page, copies.back()};
  }
  const Lanes lanes{sourcesOf(pages)};
  // Page 0's blocks 1 and 4, page 1's block 7, every block of page 2.
  std::vector<uint32_t> blocks{1, 4, kPlaces + 7};
  for (uint32_t place = 0; place < kPlaces; ++place) blocks.push_back(2 * kPlaces + place);
  const auto span = [&](uint32_t first, uint32_t last) {
    return format.recordRange(1, last).offset + format.recordBytes() - format.recordRange(1, first).offset;
  };
  auto stats = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 1, lanes, {blocks}, stats));
  require(stats->misses == 11 && stats->readBytes == span(1, 4) + span(7, 7) + span(0, 7) &&
              copies[0]->reads() == 1 && copies[1]->reads() == 1 && copies[2]->reads() == 1 && !copies[3]->reads(),
          "a page's missed blocks were not one read across them");
  // Page 0's blocks 2 and 6 are its misses now, around a block it holds.
  auto next = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 1, lanes, {{2, 4, 6}}, next));
  require(next->hits == 1 && next->misses == 2 && next->readBytes == span(2, 6) && copies[0]->reads() == 2,
          "a page's misses around a block the cache holds were not one read");
  std::cout << "kv_block_cache_test: a page's misses in one read PASS\n";
}

void growsAndReleases(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  const uint64_t extent = extentBytes(layout);
  bool refuse = false;
  KvBlockCache cache = makeCache(backend, layout, 4, [&](uint64_t bytes, const std::function<void()> &allocate) {
    if (refuse) return metal::AllocationResult(metal::AllocationFailure::EngineBudget);
    return test::admitAll(bytes, allocate);
  });
  require(cache.stats().bytes == 0 && cache.growTowardTarget().granted() && cache.stats().bytes == 0,
          "a cache without floors took memory");
  refuse = true;
  const model::ModelMemoryAdmission refused = cache.reserveLane(0);
  require(!refused.granted() && refused.allocationFailure == metal::AllocationFailure::EngineBudget &&
              cache.stats().bytes == 0 && cache.stats().lanes == 0,
          "a refused floor changed the cache");
  refuse = false;
  require(cache.reserveLane(0).granted() && cache.stats().bytes == extent &&
              cache.stats().floorBytes == extent && cache.stats().targetBytes == 4 * extent,
          "one lane's floor is not one extent");
  std::vector<KvPageSource> pages(256);
  for (uint32_t page = 0; page < pages.size(); ++page) pages[page] = {1 + page, copyOf(layout)};
  const Lanes lanes{sourcesOf(pages)};
  static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 200)}, std::make_shared<KvReadStats>()));
  // Toward the target an extent at a time; the blocks stay.
  for (uint32_t extents = 2; extents <= 4; ++extents)
    require(cache.growTowardTarget().granted() && cache.stats().bytes == extents * extent,
            "growth toward the target is not an extent at a time");
  require(cache.growTowardTarget().granted() && cache.stats().bytes == 4 * extent, "the cache grew past its target");
  auto grown = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 256)}, grown));
  require(grown->hits == 200 * kPlaces && grown->misses == 56 * kPlaces, "growing lost the cache's blocks");
  // A second lane's floor fits; giving back what is above the floors, an
  // extent a call, keeps two lanes' (3,072 slots, two extents), then one
  // lane's.
  require(cache.reserveLane(1).granted() && cache.stats().bytes == 4 * extent &&
              cache.stats().floorBytes == 2 * extent,
          "a second lane's floor is not two lanes' extents");
  // An idle reclaim gives back one extent above the floors a call.
  require(cache.releaseAboveFloor() == extent && cache.stats().slotsPerLayer == 3 * kExtentSlots &&
              cache.releaseAboveFloor() == extent && cache.stats().slotsPerLayer == 2 * kExtentSlots,
          "an idle reclaim did not give back an extent above the floors at a time");
  cache.releaseLane(1);
  require(cache.releaseAboveFloor() == extent && cache.stats().bytes == extent && cache.stats().lanes == 1,
          "an idle reclaim kept a released lane's floor");
  auto shrunk = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 256)}, shrunk));
  require(shrunk->misses + shrunk->hits == kExtentSlots && shrunk->hits < kExtentSlots && shrunk->misses,
          "the slots an idle reclaim gave back kept their blocks");
  // All or nothing: two lanes need a second extent, which is refused.
  refuse = true;
  require(!cache.reserveLane(1).granted() && cache.stats().bytes == extent && cache.stats().lanes == 1,
          "a refused floor kept its extent");
  require(!cache.growTowardTarget().granted() && cache.stats().bytes == extent, "a refused growth changed the cache");
  refuse = false;
  cache.releaseLane(0);
  require(cache.growTowardTarget().granted() && cache.stats().bytes == extent,
          "a cache whose lanes released their floors grew");
  require(cache.releaseAboveFloor() == extent && cache.stats().bytes == 0,
          "an idle reclaim kept a cache no lane needs");
  rejects([&] { static_cast<void>(cache.reserveLane(model::ExecutionLimits::maximumBatchWidth)); }, "no such lane",
          "a floor past the lanes was reserved");
  // A floor of two extents, the second refused, leaves neither.
  uint32_t grants = 1;
  KvBlockCache wide = makeCache(
      backend, layout, 2,
      [&](uint64_t bytes, const std::function<void()> &allocate) {
        if (!grants) return metal::AllocationResult(metal::AllocationFailure::HostPressure);
        --grants;
        return test::admitAll(bytes, allocate);
      },
      2 * kLaneSlots);
  require(!wide.reserveLane(0).granted() && wide.stats().bytes == 0 && wide.stats().lanes == 0,
          "a floor refused its second extent kept its first");
  std::cout << "kv_block_cache_test: growth and release PASS\n";
}

void failsTheLanesOfUnreadBlocks(MetalBackend &backend) {
  const kv::Layout layout = layoutOf(kv::Format::Int8);
  KvBlockCache cache = makeCache(backend, layout, 2);
  require(cache.reserveLane(0).granted() && cache.reserveLane(1).granted(), "the floors were refused");
  std::vector<KvPageSource> pages(8);
  std::vector<std::shared_ptr<TestKvCopy>> copies;
  for (uint32_t page = 0; page < pages.size(); ++page) {
    copies.push_back(copyOf(layout));
    pages[page] = {1 + page, copies.back()};
  }
  const Lanes lanes{sourcesOf(pages), sourcesOf(pages)};
  copies[5]->failAll = true;
  // A transient error: the read is tried once more.
  copies[3]->failNext = 1;
  auto stats = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 8), blocksOf(4, 6)}, stats));
  require(stats->misses == 64 && stats->readFailures == kPlaces, "a page's failed reads are not counted");
  require(cache.takeFailures() == std::vector<KvReadFailure>{{0, 6}, {1, 6}},
          "a block that could not be read did not fail each lane that listed it, once");
  require(cache.takeFailures().empty(), "failures were taken twice");
  // Their slots were freed: the next resolve reads them again.
  copies[5]->failAll = false;
  auto again = std::make_shared<KvReadStats>();
  static_cast<void>(resolve(backend, cache, 0, lanes, {blocksOf(0, 8), {}}, again));
  require(again->misses == kPlaces && again->hits == 56 && cache.takeFailures().empty(),
          "a block that could not be read was not read again");
  std::cout << "kv_block_cache_test: read failures PASS\n";
}

// Block `block` of a lane's history as its record holds it (kv::SlotFormat):
// its four tokens of each tensor, head by head: keys, key scales, values,
// value scales (int8 only).
template <typename T>
std::vector<std::byte> recordOf(const test::AttentionFixture &fixture, uint32_t lane, uint32_t block) {
  constexpr bool scaled = std::is_same_v<T, int8_t>;
  constexpr uint32_t tokens = SPLASH_ATTENTION_BLOCK_TOKENS, dimensions = SPLASH_KV_HEAD_DIMENSION;
  const uint32_t heads = fixture.plan().shape.kvHeads, first = block * tokens;
  std::vector<std::byte> record;
  const auto append = [&](const void *data, size_t bytes) {
    const auto *from = static_cast<const std::byte *>(data);
    record.insert(record.end(), from, from + bytes);
  };
  for (uint32_t head = 0; head < heads; ++head)
    append(fixture.keyRow<T>(lane, head, first), tokens * dimensions * sizeof(T));
  if (scaled)
    for (uint32_t head = 0; head < heads; ++head)
      append(fixture.scale(lane, SPLASH_KV_KEY_SCALES, head, first), tokens * sizeof(float));
  for (uint32_t head = 0; head < heads; ++head)
    append(fixture.valueOf<T>(lane, head, first, 0), tokens * dimensions * sizeof(T));
  if (scaled)
    for (uint32_t head = 0; head < heads; ++head)
      append(fixture.scale(lane, SPLASH_KV_VALUE_SCALES, head, first), tokens * sizeof(float));
  return record;
}

// Overwrites the K/V of a lane's page in the pool.
template <typename T> void overwrite(const test::AttentionFixture &fixture, uint32_t lane, uint32_t page) {
  for (uint32_t token = page * kv::kPageTokens; token < (page + 1) * kv::kPageTokens; ++token)
    for (uint32_t head = 0; head < fixture.plan().shape.kvHeads; ++head) {
      std::memset(fixture.keyRow<T>(lane, head, token), 0x5a, SPLASH_KV_HEAD_DIMENSION * sizeof(T));
      std::memset(fixture.valueOf<T>(lane, head, token, 0), 0x5a, SPLASH_KV_HEAD_DIMENSION * sizeof(T));
      if constexpr (std::is_same_v<T, int8_t>)
        for (const uint32_t tensor : {SPLASH_KV_KEY_SCALES, SPLASH_KV_VALUE_SCALES})
          *fixture.scale(lane, tensor, head, token) = 1e6f;
    }
}

template <typename T> void attendsThroughLocations(MetalBackend &backend, kv::Format format) {
  using Tensor = test::AttentionFixturePlan::Tensor;
  constexpr uint32_t kQueryHeads = 24, kLayer = 1, kRows = kv::kVerifyRows, kTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  const kv::Layout layout = layoutOf(format);
  const test::AttentionShape shape{kQueryHeads, layout.kvHeads, layout.headDimension, format, true};
  const test::AttentionFixturePlan::Histories histories{1503, 1219};
  constexpr uint32_t lanes = 2;
  kv::Layout attended = layout;
  attended.attentionLayers = 1;
  attended.indexDimension = 0;
  const test::AttentionFixturePlan plan = test::AttentionFixturePlan::make(
      shape, lanes, kRows, histories, ops::PagedAttention::verifyWorkspace(lanes, kQueryHeads, attended),
      {.poolLayers = layout.attentionLayers, .layer = kLayer, .extentPages = kExtentPages});
  test::AttentionFixture fixture(backend, plan, "kv-block-cache-test");
  fixture.fill();
  const ops::VerifyAttentionPlan verify = ops::PagedAttention::verifyPlan(
      lanes, kQueryHeads, plan.layout(), std::span(histories).first(lanes));
  // Each row of lane 0 selects every other block of its history besides its
  // own, as QSA's lists do (masks of each lane's rows, a bit a block); lane
  // 1's rows select blocks 1, 2 and 5 of each page, so that a page's read
  // starts past its first record and passes over records it does not place.
  const uint32_t visible = std::max(histories[0], histories[1]) + kRows, words = visible / kTokens / 32 + 1;
  const MetalBuffer masks = test::sharedBuffer(backend, uint64_t{lanes} * kRows * words * sizeof(uint32_t));
  std::memset(masks.contents(), 0, masks.sizeBytes());
  std::vector<ops::BlockListRows> sequences;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (uint32_t row = 0; row < kRows; ++row) {
      uint32_t *bits = static_cast<uint32_t *>(masks.contents()) + (uint64_t{lane} * kRows + row) * words;
      const uint32_t own = (histories[lane] + row) / kTokens;
      for (uint32_t block = 0; block <= own; ++block) {
        const uint32_t place = block % kPlaces;
        if (block == own || (lane ? place == 1 || place == 2 || place == 5 : (block + row) % 2 == 0))
          bits[block / 32] |= 1u << (block % 32);
      }
    }
    sequences.push_back({lane * kRows, kRows, histories[lane], lane});
  }
  const uint32_t stride = visible / kTokens + 1;
  const ops::AttentionBlocks blocks{test::sharedBuffer(backend, uint64_t{lanes} * stride * sizeof(uint32_t)),
                                    test::sharedBuffer(backend, lanes * sizeof(uint32_t)), stride, 0};
  const auto attend = [&](KvBlockCache *cache, const Lanes &sources, const std::shared_ptr<KvReadStats> &stats) {
    metal::CommandGraph graph;
    ops::PagedAttention::addBlockLists(graph, masks, words, sequences, blocks);
    const MetalBuffer locations = test::sharedBuffer(backend, uint64_t{lanes} * stride * sizeof(uint32_t));
    fixture.addGraph(graph, verify,
                     cache ? cache->addResolve(graph, kLayer, sources, blocks, locations, stats) : blocks);
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    const auto *output = static_cast<const uint16_t *>(fixture.buffer(Tensor::Output).contents());
    return std::vector<uint16_t>(output, output + fixture.buffer(Tensor::Output).sizeBytes() / sizeof(uint16_t));
  };
  const std::vector<uint16_t> resident = attend(nullptr, {}, nullptr);
  // Every page of committed history but each third holds only its index;
  // its copy holds the pool's records, and the pool's K/V is overwritten.
  // The cache first read the blocks of other copies of those pages, which
  // hold other bytes: the blocks keep their ids, their copies changed.
  const kv::SlotFormat slots(layout);
  KvBlockCache cache = makeCache(backend, layout, 2);
  require(cache.reserveLane(0).granted() && cache.reserveLane(1).granted(), "the floors were refused");
  Lanes sources, earlier;
  uint32_t indexed = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    std::vector<KvPageSource> pages(plan.pages[lane]), stale(plan.pages[lane]);
    for (uint32_t page = 0; page < histories[lane] / kv::kPageTokens; ++page) {
      if (page % 3 == 0) continue;
      std::vector<std::byte> payload(slots.payloadBytes());
      for (uint32_t place = 0; place < kPlaces; ++place) {
        const std::vector<std::byte> record = recordOf<T>(fixture, lane, page * kPlaces + place);
        const kv::SlotRange range = slots.recordRange(kLayer, place);
        require(record.size() == range.bytes, "a block's record has another size than its format's");
        std::ranges::copy(record, payload.begin() + range.offset);
      }
      pages[page] = {1 + lane * 1000 + page, std::make_shared<TestKvCopy>(std::move(payload))};
      stale[page] = {pages[page].block, copyOf(layout)};
      overwrite<T>(fixture, lane, page);
      ++indexed;
    }
    sources.push_back(sourcesOf(pages));
    earlier.push_back(sourcesOf(stale));
  }
  auto read = std::make_shared<KvReadStats>();
  require(attend(&cache, earlier, read) != resident, "attention through other copies' bytes did not differ");
  auto stats = std::make_shared<KvReadStats>();
  require(attend(&cache, sources, stats) == resident && stats->misses == read->misses && !stats->hits,
          "attention through the resolve's locations differs, or found blocks of the copies before");
  auto again = std::make_shared<KvReadStats>();
  require(attend(&cache, sources, again) == resident && again->hits == stats->misses && !again->misses,
          "attention through cached blocks differs");
  require(stats->misses > indexed && stats->paged && cache.takeFailures().empty(),
          "the located attention read no block or left none to the page table");
  std::cout << "kv_block_cache_test: located attention " << kv::formatName(format) << ", " << stats->misses
            << " blocks read, " << stats->paged << " paged PASS\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: kv_block_cache_test METALLIB\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      resolvesEntries(backend);
      readsAPagesMissesAtOnce(backend);
      growsAndReleases(backend);
      failsTheLanesOfUnreadBlocks(backend);
      attendsThroughLocations<int8_t>(backend, kv::Format::Int8);
      attendsThroughLocations<uint16_t>(backend, kv::Format::BFloat16);
      refusesWhatItCannotPlace(argv[1]);
      std::cout << "kv_block_cache_test: PASS\n";
    } catch (const std::exception &error) {
      std::cerr << "kv_block_cache_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
