// Times Splash's production attention kernels on one layer of Page32 KV
// across cache lengths, for the prefill chunk (2048 rows), and the DFlash
// verify batch (8 rows per lane, one and four lanes). Each case builds the same
// store + attention graph the executor encodes, reports the fused GPU time of
// the whole graph and, with each dispatch submitted as its own command, the GPU
// time of each pipeline over the deterministic synthetic history of
// dev/tests/engine/AttentionFixture.hpp. These are kernel timings, not a
// correctness oracle (the attention kernel tests are). The KV sits in extents
// of the size the memory plan picks for the model, or of --extent-pages pages,
// which must hold whole alignment units of every swept shape; the swept layer
// is the second of two so that its region starts past the first one's, and each
// case prints a digest of its output: two builds that fill the same pages must
// print the same digests, whatever their storage.
//
// usage: attention-sweep METALLIB [--histories 0,2048,...] [--shapes 27b,35b,fn]
//                        [--lanes 1,4] [--repeat N] [--phases both|verify|prefill]
//                        [--compare-metallib PATH] [--kv-format int8|bf16]
//                        [--extent-pages N] [--qsa shared|independent|overlapping|pages|indexer]
//                        [--located]
//
// fn is Qwen3.8-Flash-Next's GQA-12 attention, whose rows attend past 512
// complete blocks of four tokens only to the 512 blocks their QSA bitmap
// names and their own (kernels/shared/block_indexer.metal); its tiles attend the
// union of their rows' blocks, which the graph lists from the bitmaps first
// (ops::PagedAttention::addBlockLists). --qsa makes such bitmaps: blocks
// scattered alike for every row (shared), independently per row
// (independent), independently per row of a set three times a row's alike
// for every row (overlapping, the unions real selections make), or whole
// pages alike for every row (pages). --qsa indexer
// runs the QSA indexer itself before the attention, as a Flash-Next attention
// layer does (model/Qwen4ExpTarget.cpp): its rows' query heads and keys, the
// pooled keys of the blocks they complete, every row's scores of every complete
// block and its bitmap, over random pooled keys of the history, a lane's rows'
// indexer queries sharing most of their direction as consecutive tokens' do.
// Every case with lists reports their mean entries (the union of a tile's
// rows' blocks, which its attention reads). Without --qsa the fn tiles attend
// every block. --located moves each fn verify case's listed blocks of its
// lanes' complete pages to a block cache of their own, at shuffled places,
// and reads them there through their locations (the located split,
// ops::AttentionBlocks::locations), as a request whose K/V spilled reads
// them; the case checks its output bits against the lists alone before the
// move, so its digest is the one a run without --located prints, and a split
// that read a moved block from its page would print another.
//
// The comparison library loads into a MetalBackend of its own, which needs
// residency_kick (kernels/shared/residency.metal) in every library it loads:
// build baselines from a tree that has that kernel.
#include "../tests/engine/AttentionFixture.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/BlockIndexer.h"
#include "metal/abi/RoPE.h"
#include "ops/BlockIndexer.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

using test::AttentionFixture;
using test::AttentionFixturePlan;
using test::AttentionShape;
using test::HostKvExtents;
using test::moveBlock;

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPrefillRows = SPLASH_PREFILL_TOKEN_BUDGET;

// The swept attention layer is the second of the extents' two.
constexpr uint32_t kLayer = 1;
// The memory plan's extents of the full models: 16, 10 and 12 attention
// layers.
constexpr uint32_t kModelLayers27b = 16, kModelLayers35b = 10, kModelLayersFn = 12;
// Qwen3.8-Flash-Next's QSA: the indexer's top_k of 2048 tokens in blocks of 4.
constexpr uint32_t kQsaTopBlocks = 512, kQsaBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
// --qsa overlapping: the blocks a tile's rows choose theirs from, three times
// a row's, which makes unions of about 1,500 blocks, those Flash-Next's
// verify steps make at 126K-252K tokens of real text and code.
constexpr uint32_t kQsaOverlapBlocks = 3 * kQsaTopBlocks;

AttentionShape shapeOf(const std::string &shape, kv::Format format) {
  if (shape == "fn") return AttentionShape{24, 2, 256, format, true};
  return shape == "27b" ? AttentionShape{24, 4, 256, format}
                        : AttentionShape{16, 2, 256, format};
}

uint32_t modelLayers(const std::string &shape) {
  return shape == "27b" ? kModelLayers27b : shape == "35b" ? kModelLayers35b : kModelLayersFn;
}

std::string modelName(const std::string &shape) {
  return shape == "27b" ? "qwen3.8-27b" : shape == "35b" ? "qwen3.6-35b-a3b" : "qwen3.8-flash-next";
}

// A 32-bit mix of a block (and a row), uniform enough to scatter selections.
uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

// The QSA bitmaps of a case's rows, words per row: lane l's verify rows from
// row 8 l, a prefill chunk's from row 0. A row with at most kQsaTopBlocks
// complete blocks attends causally; past that to the kQsaTopBlocks of them
// whose mix under `pattern` is least, and to its own block.
// The words of a case's QSA bitmaps per row: a bit per block of its longest
// lane and its tail block.
uint32_t qsaWords(const AttentionFixturePlan &plan) {
  uint32_t visible = 0;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    visible = std::max(visible, plan.histories[lane] + plan.rows);
  return visible / kQsaBlockTokens / 32 + 2;
}

std::vector<uint32_t> qsaBitmaps(const AttentionFixturePlan &plan, const std::string &pattern) {
  const uint32_t words = qsaWords(plan);
  std::vector<uint32_t> bits(uint64_t{plan.lanes} * plan.rows * words, 0u);
  std::vector<std::pair<uint32_t, uint32_t>> keyed;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    for (uint32_t row = 0; row < plan.rows; ++row) {
      uint32_t *out = bits.data() + (uint64_t{lane} * plan.rows + row) * words;
      const auto take = [&](uint32_t block) { out[block / 32] |= 1u << (block % 32); };
      const uint32_t complete = (plan.histories[lane] + row + 1) / kQsaBlockTokens;
      take(complete);
      if (complete <= kQsaTopBlocks) {
        for (uint32_t block = 0; block < complete; ++block) take(block);
        continue;
      }
      // Pages: whole pages alike for every row; shared: blocks alike for
      // every row; independent: blocks of each row's own; overlapping: each
      // row's own of the kQsaOverlapBlocks blocks alike for every row.
      keyed.clear();
      for (uint32_t block = 0; block < complete; ++block)
        keyed.emplace_back(pattern == "pages"         ? mix(block / 8)
                           : pattern == "independent" ? mix(block ^ mix(row + lane * 65536))
                                                      : mix(block),
                           block);
      if (pattern == "overlapping" && keyed.size() > kQsaOverlapBlocks) {
        std::nth_element(keyed.begin(), keyed.begin() + kQsaOverlapBlocks, keyed.end());
        keyed.resize(kQsaOverlapBlocks);
        for (auto &[key, block] : keyed) key = mix(block ^ mix(row + lane * 65536));
      }
      std::nth_element(keyed.begin(), keyed.begin() + kQsaTopBlocks, keyed.end());
      for (uint32_t index = 0; index < kQsaTopBlocks; ++index) take(keyed[index].second);
    }
  return bits;
}

// The prefill chunk on one lane, or the verify rows of `lanes` lanes, all
// after `history` tokens, with their plan's scratch.
AttentionFixturePlan casePlan(AttentionShape shape, bool prefill, uint32_t lanes,
                              uint32_t history, uint32_t extentPages) {
  const kv::Layout layout{.attentionLayers = 1, .kvHeads = shape.kvHeads, .headDimension = shape.headDimension,
                          .format = shape.format, .valuesTokenMajor = shape.valuesTokenMajor};
  const uint32_t caseLanes = prefill ? 1 : lanes;
  AttentionFixturePlan::Histories histories{};
  std::fill_n(histories.begin(), caseLanes, history);
  const AttentionWorkspace scratch =
      prefill ? PagedAttention::prefillPlan(kPrefillRows, shape.queryHeads, layout)
                    .workspace
              : PagedAttention::verifyPlan(caseLanes, shape.queryHeads, layout,
                                           std::span(histories).first(caseLanes))
                    .workspace;
  return AttentionFixturePlan::make(shape, caseLanes, prefill ? kPrefillRows : kVerifyRows,
                                    histories, scratch, {kLayer + 1, kLayer, extentPages});
}

// The QSA indexer of a case's rows (--qsa indexer): its inputs, as the
// indexer's projections write them, and its scratch, as model/Qwen4Steps
// sizes it. The index of every lane's pages is one extent of random pooled
// keys, lane after lane.
struct QsaIndexer final {
  metal::MetalBuffer index;
  std::vector<metal::MetalBuffer> tables;
  std::vector<metal::MetalBuffer> pending;
  std::vector<uint32_t> firstPositions;
  std::vector<uint32_t> counts;
  uint32_t laneRecords = 0;
  // A prefill chunk's rows score in batches of model/Qwen4Steps.cpp's
  // kPrefillScoreRows, a verify's all at once.
  uint32_t scoreRows = 0;
  metal::MetalBuffer records, query, key, norm, ropeCos, ropeSin, inverseFrequencies, queries, keys, scores;
};

// A value in [-1, 1) of `index` under `seed`.
float noise(uint64_t index, uint32_t seed) {
  return float(mix(uint32_t(index) ^ mix(uint32_t(index >> 32) + seed))) * 0x1p-31f - 1.0f;
}

uint16_t bfloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

template <class T> metal::MetalBuffer sharedBuffer(metal::MetalBackend &backend, uint64_t count, const char *label) {
  return backend.allocateBuffer(std::max<uint64_t>(count * sizeof(T), 16), metal::BufferStorage::Shared, label);
}

QsaIndexer qsaIndexer(metal::MetalBackend &backend, const AttentionFixturePlan &plan, bool prefill,
                      uint32_t stride) {
  constexpr uint32_t D = BLOCK_INDEXER_DIMENSION, H = BLOCK_INDEXER_HEADS;
  constexpr uint32_t pageBlocks = kv::kPageTokens / kQsaBlockTokens;
  QsaIndexer result;
  const uint32_t lanes = plan.lanes, rows = plan.rows, laneRows = lanes * rows;
  uint64_t pages = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) pages += plan.pages[lane];
  result.index = backend.allocateBuffer(pages * pageBlocks * D * 2, metal::BufferStorage::Shared,
                                        "attention-sweep-index", metal::BufferReach::Addressed);
  auto *pooled = static_cast<uint16_t *>(result.index.contents());
  for (uint64_t value = 0; value < pages * pageBlocks * D; ++value) pooled[value] = bfloat16(noise(value, 1));
  uint64_t page = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    metal::MetalBuffer table = sharedBuffer<uint64_t>(backend, plan.pages[lane], "attention-sweep-index-table");
    auto *entries = static_cast<uint64_t *>(table.contents());
    for (uint32_t index = 0; index < plan.pages[lane]; ++index) entries[index] = result.index.gpuAddress() | page++;
    result.tables.push_back(std::move(table));
    metal::MetalBuffer pending = sharedBuffer<uint16_t>(backend, BLOCK_INDEXER_PENDING * D, "attention-sweep-pending");
    std::memset(pending.contents(), 0, pending.sizeBytes());
    result.pending.push_back(std::move(pending));
    result.firstPositions.push_back(plan.histories[lane]);
  }
  // The blocks each lane's rows complete, each rotated to its first token.
  result.laneRecords = rows / kQsaBlockTokens + 1;
  result.records =
      sharedBuffer<BlockIndexerBlock>(backend, uint64_t{lanes} * result.laneRecords, "attention-sweep-blocks");
  auto *records = static_cast<BlockIndexerBlock *>(result.records.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    uint32_t count = 0;
    for (uint32_t last = plan.histories[lane]; last < plan.histories[lane] + rows; ++last)
      if (last % kQsaBlockTokens == kQsaBlockTokens - 1) {
        const uint32_t first = last + 1 - kQsaBlockTokens;
        records[lane * result.laneRecords + count++] = {first, {first, first, first}};
      }
    result.counts.push_back(count);
  }
  // A lane's rows' query heads: the lane's own direction and a third as much
  // of the row's.
  result.query = sharedBuffer<float>(backend, uint64_t{laneRows} * H * D, "attention-sweep-indexer-query");
  auto *query = static_cast<float *>(result.query.contents());
  for (uint32_t row = 0; row < laneRows; ++row)
    for (uint32_t value = 0; value < H * D; ++value)
      query[uint64_t{row} * H * D + value] =
          noise(uint64_t{row / rows} * H * D + value, 2) + noise(uint64_t{row} * H * D + value, 3) / 3;
  result.key = sharedBuffer<float>(backend, uint64_t{laneRows} * D, "attention-sweep-indexer-key");
  for (uint64_t value = 0; value < uint64_t{laneRows} * D; ++value)
    static_cast<float *>(result.key.contents())[value] = noise(value, 4);
  result.norm = sharedBuffer<float>(backend, D, "attention-sweep-indexer-norm");
  std::fill_n(static_cast<float *>(result.norm.contents()), D, 1.0f);
  result.inverseFrequencies = sharedBuffer<float>(backend, SPLASH_TARGET_ROPE_PAIRS, "attention-sweep-frequencies");
  auto *frequencies = static_cast<float *>(result.inverseFrequencies.contents());
  for (uint32_t pair = 0; pair < SPLASH_TARGET_ROPE_PAIRS; ++pair)
    frequencies[pair] = std::pow(10'000'000.0f, -float(pair) / SPLASH_TARGET_ROPE_PAIRS);
  result.ropeCos = sharedBuffer<float>(backend, uint64_t{laneRows} * SPLASH_TARGET_ROPE_PAIRS, "attention-sweep-cos");
  result.ropeSin = sharedBuffer<float>(backend, uint64_t{laneRows} * SPLASH_TARGET_ROPE_PAIRS, "attention-sweep-sin");
  for (uint32_t row = 0; row < laneRows; ++row)
    for (uint32_t pair = 0; pair < SPLASH_TARGET_ROPE_PAIRS; ++pair) {
      const float angle = float(plan.histories[row / rows] + row % rows) * frequencies[pair];
      static_cast<float *>(result.ropeCos.contents())[row * SPLASH_TARGET_ROPE_PAIRS + pair] = std::cos(angle);
      static_cast<float *>(result.ropeSin.contents())[row * SPLASH_TARGET_ROPE_PAIRS + pair] = std::sin(angle);
    }
  result.queries = sharedBuffer<uint16_t>(backend, uint64_t{laneRows} * H * D, "attention-sweep-indexer-queries");
  result.keys = sharedBuffer<uint16_t>(backend, uint64_t{laneRows} * D, "attention-sweep-indexer-keys");
  result.scoreRows = prefill ? 256 : laneRows;
  result.scores = backend.allocateBuffer(uint64_t{result.scoreRows} * stride * sizeof(float),
                                         metal::BufferStorage::Private, "attention-sweep-scores");
  return result;
}

// A case's QSA bitmaps (qsaBitmaps, or the indexer's) and the block lists its
// tiles attend through: one per verify lane, or per prefill tile; with
// --located, the block cache that its lists' locations name.
struct QsaBlocks final {
  metal::MetalBuffer bitmaps;
  uint32_t words = 0;
  AttentionBlocks blocks;
  uint32_t lists = 0;
  std::optional<QsaIndexer> indexer;
  std::optional<HostKvExtents> cache;
};

// Located lists are read back to place their blocks (locate).
QsaBlocks qsaBlocks(metal::MetalBackend &backend, const AttentionFixturePlan &plan, bool prefill,
                    const std::vector<uint32_t> &bitmaps, uint32_t words, bool indexer, bool located) {
  QsaBlocks result;
  if (!words) return result;
  result.lists = prefill ? kv::prefillAttentionTiles(plan.rows) : plan.lanes;
  const uint32_t stride = kVerifyRows * (kQsaTopBlocks + 1);
  result.bitmaps = sharedBuffer<uint32_t>(backend, uint64_t{plan.lanes} * plan.rows * words, "attention-sweep-qsa");
  std::ranges::copy(bitmaps, static_cast<uint32_t *>(result.bitmaps.contents()));
  result.words = words;
  result.blocks = {backend.allocateBuffer(uint64_t{result.lists} * stride * sizeof(uint32_t),
                                          located ? metal::BufferStorage::Shared : metal::BufferStorage::Private,
                                          "attention-sweep-lists"),
                   sharedBuffer<uint32_t>(backend, result.lists, "attention-sweep-counts"), stride, 0};
  if (indexer) result.indexer = qsaIndexer(backend, plan, prefill, words * 32);
  return result;
}

// --located: each listed block of the lanes' complete pages moves from the
// fixture's pool to a block cache of the pool's layout and extents, at a
// shuffled place (moveBlock), where the verify reads it through its
// location; a block of the page its rows write stays SPLASH_ATTENTION_PAGED.
// The lists are those the case's graph made last, which it makes again alike.
void locate(metal::MetalBackend &backend, AttentionFixture &fixture, QsaBlocks &qsa) {
  constexpr uint32_t pageBlocks = kv::kPageTokens / kQsaBlockTokens;
  const AttentionFixturePlan &plan = fixture.plan();
  const uint32_t stride = qsa.blocks.stride, extentPages = plan.geometry.extentPages;
  const auto *lists = static_cast<const uint32_t *>(qsa.blocks.lists.contents());
  const auto *counts = static_cast<const uint32_t *>(qsa.blocks.counts.contents());
  const uint32_t pages = (std::accumulate(counts, counts + plan.lanes, 0U) + pageBlocks - 1) / pageBlocks;
  qsa.cache.emplace(backend, plan.poolLayout(), extentPages, (pages + extentPages - 1) / extentPages);
  std::vector<uint32_t> cachePages(qsa.cache->pageCount()), slots(qsa.cache->pageCount() * pageBlocks);
  std::iota(cachePages.begin(), cachePages.end(), 0U);
  std::iota(slots.begin(), slots.end(), 0U);
  std::shuffle(slots.begin(), slots.end(), std::mt19937(1));
  qsa.blocks.locationPages = sharedBuffer<SplashKvPage>(backend, cachePages.size(), "attention-sweep-cache-pages");
  qsa.cache->writeTable(cachePages, qsa.blocks.locationPages.contents());
  qsa.blocks.locations = sharedBuffer<uint32_t>(backend, uint64_t{qsa.lists} * stride, "attention-sweep-locations");
  auto *locations = static_cast<uint32_t *>(qsa.blocks.locations.contents());
  uint32_t placed = 0;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    for (uint32_t entry = 0; entry < counts[lane]; ++entry) {
      const uint32_t block = lists[uint64_t{lane} * stride + entry] >> SPLASH_ATTENTION_BLOCK_ROW_BITS;
      uint32_t &location = locations[uint64_t{lane} * stride + entry];
      if ((block / pageBlocks + 1) * kv::kPageTokens > plan.histories[lane]) {
        location = SPLASH_ATTENTION_PAGED;
        continue;
      }
      location = slots[placed++];
      moveBlock(plan.poolLayout(), plan.geometry.layer, fixture.pool(), fixture.page(lane, block * kQsaBlockTokens),
                block % pageBlocks, *qsa.cache, location / pageBlocks, location % pageBlocks);
    }
}

// The mean entries of a case's lists, as its last run left them.
double listEntries(const QsaBlocks &qsa) {
  if (!qsa.words) return 0.0;
  const auto *counts = static_cast<const uint32_t *>(qsa.blocks.counts.contents());
  return std::accumulate(counts, counts + qsa.lists, 0.0) / qsa.lists;
}

// The store and attention graph the runtime encodes for a case, after its
// block lists when it has QSA bitmaps, and the indexer that selects them
// with --qsa indexer.
metal::CommandGraph caseGraph(const AttentionFixture &fixture, bool prefill, const QsaBlocks &qsa) {
  const AttentionFixturePlan &plan = fixture.plan();
  metal::CommandGraph graph;
  if (const QsaIndexer *indexer = qsa.indexer ? &*qsa.indexer : nullptr) {
    const ops::IndexerRows rows{plan.rows, indexer->firstPositions, {indexer->tables, 0}, kQsaTopBlocks, qsa.words,
                                qsa.words * 32};
    BlockIndexer::addProject(graph, indexer->query, indexer->key, {indexer->norm, true}, indexer->ropeCos,
                             indexer->ropeSin, indexer->queries, indexer->keys, rows);
    BlockIndexer::addPool(graph,
                          {indexer->records, indexer->counts, indexer->laneRecords, indexer->keys, indexer->pending},
                          {indexer->norm, true}, indexer->inverseFrequencies, 1.0F, rows);
    BlockIndexer::addSelect(graph, indexer->queries, indexer->scores, indexer->scoreRows, qsa.bitmaps, rows);
  }
  if (qsa.words) {
    std::vector<BlockListRows> sequences;
    for (uint32_t lane = 0; lane < plan.lanes; ++lane)
      sequences.push_back({lane * plan.rows, plan.rows, plan.histories[lane], lane});
    PagedAttention::addBlockLists(graph, qsa.bitmaps, qsa.words, sequences, qsa.blocks);
  }
  if (prefill)
    fixture.addGraph(graph,
                     PagedAttention::prefillPlan(plan.rows, plan.shape.queryHeads, plan.layout()),
                     qsa.blocks);
  else
    fixture.addGraph(graph,
                     PagedAttention::verifyPlan(plan.lanes, plan.shape.queryHeads, plan.layout(),
                                                std::span(plan.histories).first(plan.lanes)),
                     qsa.blocks);
  return graph;
}

// Both cache payloads of every lane's history and rows, including scales
// only for INT8: what the page loop reads, of which a sparse selection reads
// its blocks only.
uint64_t historyBytes(const AttentionFixturePlan &plan) {
  uint64_t tokens = 0;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    tokens += uint64_t{plan.histories[lane]} + plan.rows;
  return tokens * (plan.layout().bytesPerLayerPage() / kv::kPageTokens);
}

std::span<const uint8_t> outputBytes(const AttentionFixture &fixture) {
  const metal::MetalBuffer output = fixture.buffer(AttentionFixture::Tensor::Output);
  return {static_cast<const uint8_t *>(output.contents()), output.sizeBytes()};
}

// FNV-1a of the attention output.
uint64_t outputDigest(const AttentionFixture &fixture) {
  uint64_t digest = 0xcbf29ce484222325ULL;
  for (const uint8_t byte : outputBytes(fixture))
    digest = (digest ^ byte) * 0x100000001b3ULL;
  return digest;
}

struct Case final {
  double fusedMilliseconds = 0.0;
  std::map<std::string, double> pipelineMilliseconds;
  uint64_t kvBytes = 0;
  uint64_t outputDigest = 0;
  double listEntries = 0.0;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// warmupSeconds of GPU time on every variant come first: the first case of
// a run also brings an idle GPU up to its clocks. A located case (a verify
// with lists) runs on the lists alone first, then moves its blocks (locate)
// and must give the same output reading them through their locations.
std::vector<Case> measure(std::span<metal::MetalBackend *> backends,
                          const AttentionFixturePlan &plan, bool prefill, uint32_t repeat,
                          double warmupSeconds, const std::string &qsa, bool located) {
  std::vector<std::unique_ptr<AttentionFixture>> fixtures;
  std::vector<QsaBlocks> blocks;
  std::vector<metal::CommandGraph> graphs;
  std::vector<Case> results(backends.size());
  // The indexer writes its own bitmaps.
  const uint32_t words = qsa.empty() ? 0 : qsaWords(plan);
  const std::vector<uint32_t> bitmaps =
      qsa.empty() || qsa == "indexer" ? std::vector<uint32_t>{} : qsaBitmaps(plan, qsa);
  for (size_t i = 0; i < backends.size(); ++i) {
    fixtures.push_back(
        std::make_unique<AttentionFixture>(*backends[i], plan, "attention-sweep-fixture"));
    fixtures.back()->fill();
    blocks.push_back(qsaBlocks(*backends[i], plan, prefill, bitmaps, words, qsa == "indexer", located));
    graphs.push_back(caseGraph(*fixtures.back(), prefill, blocks.back()));
    if (located) {
      (void)backends[i]->submitCommandAsync(graphs.back().dispatches()).wait();
      const uint64_t paged = outputDigest(*fixtures.back());
      locate(*backends[i], *fixtures.back(), blocks.back());
      graphs.back() = caseGraph(*fixtures.back(), prefill, blocks.back());
      (void)backends[i]->submitCommandAsync(graphs.back().dispatches()).wait();
      if (outputDigest(*fixtures.back()) != paged)
        throw std::runtime_error("the located verify split changed attention output bits");
    }
    results[i].kvBytes = historyBytes(plan);
  }
  // Warm every variant, then alternate order to limit clock/thermal drift.
  double warmup = 0.0;
  while (warmup < warmupSeconds)
    for (size_t i = 0; i < backends.size(); ++i)
      warmup += backends[i]->submitCommandAsync(graphs[i].dispatches()).wait().gpuSeconds;
  for (size_t i = 1; i < fixtures.size(); ++i)
    if (!std::ranges::equal(outputBytes(*fixtures[0]), outputBytes(*fixtures[i])))
      throw std::runtime_error("comparison metallib changed attention output bits");
  for (size_t i = 0; i < fixtures.size(); ++i) {
    results[i].outputDigest = outputDigest(*fixtures[i]);
    results[i].listEntries = listEntries(blocks[i]);
  }
  std::vector<std::vector<double>> fused(backends.size());
  std::vector<std::map<std::string, std::vector<double>>> perPipeline(backends.size());
  for (uint32_t round = 0; round < repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      fused[i].push_back(backends[i]->submitCommandAsync(graphs[i].dispatches()).wait().gpuSeconds * 1000.0);
    }
  for (uint64_t round = 0; round <= repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      const auto run =
          benchmark::replayDispatches(*backends[i], graphs[i].dispatches());
      if (round)
        for (const auto &[name, seconds] : run)
          perPipeline[i][name].push_back(seconds * 1000.0);
    }
  for (size_t i = 0; i < backends.size(); ++i) {
    results[i].fusedMilliseconds = median(fused[i]);
    for (auto &[name, samples] : perPipeline[i])
      results[i].pipelineMilliseconds[name] = median(samples);
  }
  return results;
}

uint32_t parseCount(std::string_view text, uint32_t minimum, uint32_t maximum,
                    std::string_view option) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value < minimum || value > maximum)
    throw std::invalid_argument(std::string(option) + " requires integers from " +
                                std::to_string(minimum) + " to " + std::to_string(maximum));
  return value;
}

std::vector<uint32_t> parseList(const std::string &text, uint32_t minimum,
                                uint32_t maximum, std::string_view option) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const std::string item = text.substr(start, comma == std::string::npos ? std::string::npos
                                                                            : comma - start);
    values.push_back(parseCount(item, minimum, maximum, option));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return values;
}

std::string hex(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::string json(const Case &item, const std::string &shape, uint32_t history,
                 const std::string &kind, uint32_t lanes, size_t variant,
                 const std::string &qsa, bool located) {
  std::string out = "{\"variant\":" + std::to_string(variant) + ",\"shape\":\"" + shape + "\",\"qsa\":\"" + qsa +
                    "\",\"located\":" + (located ? "true" : "false") +
                    ",\"history\":" + std::to_string(history) +
                    ",\"kind\":\"" + kind + "\",\"lanes\":" + std::to_string(lanes) +
                    ",\"fused_ms\":" + std::to_string(item.fusedMilliseconds) +
                    ",\"kv_bytes\":" + std::to_string(item.kvBytes) +
                    ",\"list_entries\":" + std::to_string(item.listEntries) +
                    ",\"output_digest\":\"" + hex(item.outputDigest) + "\",\"pipelines\":{";
  bool first = true;
  for (const auto &[name, milliseconds] : item.pipelineMilliseconds) {
    out += (first ? "" : ",") + std::string("\"") + name + "\":" + std::to_string(milliseconds);
    first = false;
  }
  return out + "}}";
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "usage: attention-sweep METALLIB [--histories LIST] [--shapes 27b,35b,fn] "
                   "[--lanes LIST] [--repeat N] [--phases both|verify|prefill] "
                   "[--compare-metallib PATH] [--kv-format int8|bf16] "
                   "[--extent-pages N] [--qsa shared|independent|overlapping|pages|indexer] "
                   "[--located]\n";
      return 64;
    }
    std::vector<uint32_t> histories{0, 2048, 8192, 16384, 32768, 65536, 131072};
    std::vector<uint32_t> lanes{1, 4};
    std::vector<std::string> shapes{"27b", "35b"};
    uint32_t repeat = 5;
    // Zero: the largest extent the memory plan picks for the model's K/V
    // pages (Flash-Next's without its index regions, which no case reads).
    uint32_t extentPages = 0;
    kv::Format format = kv::Format::Int8;
    std::string comparisonLibrary, phases = "both", qsa;
    bool located = false;
    for (int index = 2; index < argc; ++index) {
      const std::string option(argv[index]);
      if (option == "--located") {
        located = true;
        continue;
      }
      if (index + 1 >= argc)
        throw std::invalid_argument(option + " requires a value");
      const std::string value(argv[++index]);
      if (option == "--histories")
        histories = parseList(value, 0, kv::kMaximumPhysicalTokens - kPrefillRows, option);
      else if (option == "--lanes")
        lanes = parseList(value, 1, kMaximumLanes, option);
      else if (option == "--repeat")
        repeat = parseCount(value, 1, std::numeric_limits<uint32_t>::max(), option);
      else if (option == "--extent-pages")
        extentPages = parseCount(value, 1, SPLASH_KV_PAGE_INDEX_MASK, option);
      else if (option == "--kv-format") {
        if (value != "int8" && value != "bf16")
          throw std::invalid_argument("--kv-format takes int8 or bf16");
        format = value == "int8" ? kv::Format::Int8 : kv::Format::BFloat16;
      }
      else if (option == "--compare-metallib") comparisonLibrary = value;
      else if (option == "--qsa") {
        qsa = value;
        if (qsa != "shared" && qsa != "independent" && qsa != "overlapping" && qsa != "pages" && qsa != "indexer")
          throw std::invalid_argument("--qsa takes shared, independent, overlapping, pages or indexer");
      }
      else if (option == "--phases") {
        phases = value;
        if (phases != "both" && phases != "verify" && phases != "prefill")
          throw std::invalid_argument("--phases takes both, verify or prefill");
      }
      else if (option == "--shapes") {
        shapes.clear();
        size_t start = 0;
        while (start <= value.size()) {
          const size_t comma = value.find(',', start);
          const std::string shape =
              value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
          if (shape != "27b" && shape != "35b" && shape != "fn")
            throw std::invalid_argument("--shapes takes 27b, 35b or fn");
          shapes.push_back(shape);
          if (comma == std::string::npos) break;
          start = comma + 1;
        }
      } else throw std::invalid_argument("unknown option " + option);
    }
    if (!qsa.empty() && std::ranges::find(shapes, std::string("fn")) == shapes.end())
      throw std::invalid_argument("--qsa needs --shapes with fn");
    if (located && qsa.empty())
      throw std::invalid_argument("--located needs --qsa");
    for (const std::string &shape : shapes) {
      const AttentionShape geometry = shapeOf(shape, format);
      if (extentPages % kv::Layout{1, geometry.kvHeads, geometry.headDimension, format}
                            .extentAlignmentPages())
        throw std::invalid_argument("--extent-pages " + std::to_string(extentPages) +
                                    " does not hold whole alignment units of " + shape +
                                    " " + std::string(kv::formatName(format)));
    }
    metal::MetalBackend backend(argv[1]);
    std::unique_ptr<metal::MetalBackend> comparison;
    std::vector<metal::MetalBackend *> backends{&backend};
    if (!comparisonLibrary.empty()) {
      comparison = std::make_unique<metal::MetalBackend>(comparisonLibrary);
      backends.push_back(comparison.get());
    }
    std::cerr << "device " << backend.capabilities().deviceName << ", one attention layer, "
              << "Page32 " << kv::formatName(format) << " KV, median of " << repeat << " fused graphs (ms)\n";
    double warmupSeconds = 1.5;
    std::cout << "{\"device\":\"" << backend.capabilities().deviceName << "\",\"kv_format\":\"" << kv::formatName(format) << "\",\"cases\":[";
    bool firstCase = true;
    for (const std::string &shape : shapes) {
      const AttentionShape geometry = shapeOf(shape, format);
      const std::string name = modelName(shape);
      const uint32_t shapeExtentPages =
          extentPages ? extentPages
                      : kv::Layout{modelLayers(shape), geometry.kvHeads,
                                   geometry.headDimension, format}
                            .maximumExtentPages();
      // Only the GQA-12 kernels attend sparsely.
      const std::string shapeQsa = shape == "fn" ? qsa : std::string();
      std::cerr << "\n" << name << "  (" << geometry.queryHeads << " query heads, "
                << geometry.kvHeads << " KV heads, d=" << geometry.headDimension
                << ", extents of " << shapeExtentPages << " pages"
                << (shapeQsa.empty() ? "" : ", QSA " + shapeQsa) << ")\n";
      for (uint32_t history : histories) {
        auto report = [&](bool prefill, uint32_t lane) {
          // Only a verify with lists reads its blocks through locations.
          const bool caseLocated = located && !prefill && !shapeQsa.empty();
          const auto cases =
              measure(backends, casePlan(geometry, prefill, lane, history, shapeExtentPages),
                      prefill, repeat, warmupSeconds, shapeQsa, caseLocated);
          warmupSeconds = 0.1;
          for (size_t i = 0; i < cases.size(); ++i) {
            std::cout << (firstCase ? "" : ",")
                      << json(cases[i], name, history, prefill ? "prefill" : "verify", lane, i,
                              shapeQsa, caseLocated);
            firstCase = false;
            std::cerr << history << " " << (prefill ? "prefill" : "verify")
                      << " lanes=" << lane << " variant=" << i << " fused="
                      << cases[i].fusedMilliseconds << " ms digest="
                      << hex(cases[i].outputDigest);
            if (cases[i].listEntries) std::cerr << " list_entries=" << cases[i].listEntries;
            if (caseLocated) std::cerr << " located";
            std::cerr << "\n";
          }
        };
        if (phases != "verify") report(true, 1);
        if (phases != "prefill") for (uint32_t lane : lanes) report(false, lane);
      }
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "attention-sweep: " << error.what() << '\n';
    return 70;
  }
}
