#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/PagedAttention.hpp"
#include "model/RopeScaling.hpp"
#include "ops/RoPE.hpp"

#include "AttentionFixture.hpp"
#include "HostKvExtents.hpp"
#include "LinearInputReference.hpp"
#include "LinearNumerics.hpp"
#include "NormReference.hpp"
#include "StagedKvStore.hpp"
#include "TestFamilies.hpp"
#include "ops/KernelCalibration.hpp"
#include "ops/LaneRows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// The attention row these plans take: the Apple10 row (KernelCalibration.hpp),
// with int8 activations on (kInt8), or off, where its plans score bf16
// queries (kBf16).
constexpr splash::ops::AttentionCalibration kApple10Row = splash::ops::kApple10Attention;
constexpr splash::ops::Int8Activations kInt8 = splash::ops::Int8Activations::On;
constexpr splash::ops::Int8Activations kBf16 = splash::ops::Int8Activations::Off;

using namespace splash;
using test::HostKvExtents;
using test::attentionRow;
using test::bf16ToFloat;
using test::floatToBf16;
using test::moveBlock;
using test::ulpBf16;

static_assert(!std::is_aggregate_v<ops::AttentionPlan> && !std::is_default_constructible_v<ops::AttentionPlan> &&
              !std::is_copy_assignable_v<ops::AttentionPlan>);

using splash::test::rejects;
using splash::test::require;

// The production attention geometries, every family's: query heads over the
// KV heads of a layout, whose values a QSA family keeps token-major for its
// sparse kernels.
using Geometry = splash::test::AttentionGeometry;
const std::vector<Geometry> kGeometries = splash::test::familyAttentionGeometries();

// A decode lane's rows, one tile.
constexpr uint32_t kLaneRows = ops::kDecodeLaneRows;
static_assert(kLaneRows == kv::kAttentionTileRows);

// Whether a layout's attention reads the blocks its index selects (the
// sparse kernels).
bool selectsBlocks(kv::Layout layout) { return layout.reads == kv::Reads::SelectedBlocks; }

// The parts of a layout's tiles, each a threadgroup of the split grid's x
// beside its KV head.
uint32_t tileParts(uint32_t queryHeads, kv::Layout layout) {
  return selectsBlocks(layout) ? 1 : SPLASH_ATTENTION_TILE_PARTS(queryHeads / layout.kvHeads);
}

// A sequence of `rows` rows after `history` tokens, staged in whole KV
// blocks per KV head.
kv::ChunkedPrefillParams sequenceOf(uint32_t history, uint32_t rows) {
  return ops::PagedAttention::chunkParams(history, rows, (rows + 31) / 32 * 32, (history + rows + 31) / 32);
}

// The plan of `sequences` in a workspace of `slots` tile splits.
ops::AttentionPlan planOf(std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads, kv::Layout layout,
                          ops::Int8Activations int8 = kBf16, uint32_t slots = 4 * ops::kAttentionSequenceSplits,
                          ops::AttentionCalibration calibration = kApple10Row) {
  return ops::PagedAttention::plan(sequences, queryHeads, layout, calibration, int8,
                                   ops::PagedAttention::workspace(slots, queryHeads, layout));
}
ops::AttentionPlan planOf(kv::ChunkedPrefillParams sequence, uint32_t queryHeads, kv::Layout layout,
                          ops::Int8Activations int8 = kBf16, ops::AttentionCalibration calibration = kApple10Row) {
  return planOf(std::span(&sequence, 1), queryHeads, layout, int8, ops::kAttentionSequenceSplits, calibration);
}

// A layout's split or reduce pipeline.
std::string pipelineName(bool split, uint32_t queryHeads, kv::Layout layout) {
  const bool sparse = selectsBlocks(layout);
  // The kernels of four KV heads, the first compiled, have no suffix.
  const std::string suffix = layout.kvHeads == 4 ? "" : "_kv" + std::to_string(layout.kvHeads) + "_g" +
                                                            std::to_string(queryHeads / layout.kvHeads);
  const std::string format = layout.format == kv::Format::Int8 ? "q8_" : "bf16_";
  return "attention_" + (split ? format + (sparse ? "sparse" : "split") : std::string(sparse ? "sparse_reduce" : "reduce")) +
         suffix;
}

// The split that reads a sparse layout's listed blocks where their locations
// name (ops::AttentionBlocks::locations).
std::string locatedSplitName(kv::Layout layout) {
  return std::string("attention_") + (layout.format == kv::Format::Int8 ? "q8" : "bf16") + "_sparse_located_kv2_g12";
}

// Each sequence's history splits follow its rows and the pages its tiles
// read of its history alone (test::attentionSplits): a sequence of one tile,
// a decode lane's, one split per 16 pages its rows see between 32 and 128; a
// sequence of more tiles, a whole split per 32 pages (the device's), at least
// 32 shared out over its tiles and at most 32, nor 256 over its tiles; a
// sparse sequence of more than one tile the shared splits alone. Its plan's
// dispatch, workspace and grids are its tiles'.
void checkSplitRule(uint32_t queryHeads, kv::Layout layout) {
  const bool sparse = selectsBlocks(layout);
  const std::string split = pipelineName(true, queryHeads, layout), reduce = pipelineName(false, queryHeads, layout);
  for (uint32_t rows = 1; rows <= 2048; ++rows)
    for (const uint32_t history : {0U, 4000U, 8192U, 32768U, 131072U, 1'000'000U}) {
      const auto plan = planOf(sequenceOf(history, rows), queryHeads, layout);
      const uint32_t tiles = kv::attentionTiles(rows), splits = test::attentionSplits(rows, history, sparse);
      require(plan.sequences().size() == 1 && plan.sequences()[0].rows == rows &&
                  plan.sequences()[0].historyTokens == history && plan.sequences()[0].firstTile == 0 &&
                  plan.sequences()[0].splits == splits,
              "a sequence's splits do not follow its rows and history");
      require(plan.splitPipeline() == split && plan.reducePipeline() == reduce, "a plan runs the wrong pipelines");
      require(plan.dispatches().size() == 1, "a sequence alone took more than one dispatch");
      const ops::AttentionPlan::Dispatch &dispatch = plan.dispatches()[0];
      const auto groups = plan.splitGroups(dispatch), reduceGroups = plan.reduceGroups(dispatch);
      require(dispatch.sequences == 1 && dispatch.tiles == tiles && dispatch.splits == splits &&
                  dispatch.slots == tiles * splits &&
                  (sparse ? groups.x == layout.kvHeads * splits && groups.y == tiles && groups.z == 1
                          : groups.x == layout.kvHeads * tileParts(queryHeads, layout) && groups.y == tiles &&
                                groups.z == splits) &&
                  reduceGroups.x == layout.kvHeads && reduceGroups.y == 8 * queryHeads / layout.kvHeads &&
                  reduceGroups.z == tiles,
              "a plan's split/reduce geometry disagrees");
      const uint64_t fused = uint64_t{tiles} * splits * 8 * queryHeads;
      require(plan.workspace().partialsBytes == fused * 256 * 4 && plan.workspace().statisticsBytes == fused * 2 * 4,
              "a plan's workspace is not its tiles' splits");
    }
  // A sequence of one tile splits as a decode lane always has, sparse or
  // dense: one split per 16 pages its rows see, between 32 and 128.
  for (uint32_t rows = 1; rows <= kLaneRows; ++rows)
    for (const uint32_t history : {0U, 31U, 7160U, 7161U, 16'383U, 16'384U, 65'536U, 131'072U, 1'000'000U}) {
      const uint64_t pages = (uint64_t{history} + rows + 31) / 32;
      require(planOf(sequenceOf(history, rows), queryHeads, layout).sequences()[0].splits ==
                  std::clamp<uint64_t>((pages + 15) / 16, 32, 128),
              "a tile's splits are not a decode lane's");
    }
  // Concrete counts of the rule, dense: one tile; and more, as a prompt
  // chunk's have been (a whole split per 32 pages, at most 32 a tile and 256
  // over the tiles).
  if (!sparse)
    for (const auto [rows, history, splits] :
         {std::array{1U, 0U, 32U}, std::array{8U, 16'384U, 33U}, std::array{8U, 131'072U, 128U},
          std::array{64U, 0U, 4U}, std::array{256U, 1500U, 1U}, std::array{64U, 4096U, 4U},
          std::array{256U, 4096U, 4U}, std::array{64U, 8192U, 8U}, std::array{256U, 16'384U, 8U},
          std::array{64U, 65'536U, 32U}, std::array{256U, 131'072U, 8U}, std::array{512U, 65'536U, 4U},
          std::array{300U, 65'536U, 6U}, std::array{2048U, 131'072U, 1U}, std::array{16U, 65'536U, 32U},
          std::array{40U, 131'072U, 32U}, std::array{16U, 1000U, 16U}})
      require(planOf(sequenceOf(history, rows), queryHeads, layout).sequences()[0].splits == splits,
              "a sequence's split count changed: " + std::to_string(rows) + " rows after " + std::to_string(history));
}

// A command's sequences share dispatches while their tiles' splits fit the
// workspace, in order, each sequence in one, and keep the splits each takes
// alone. The plan of their last tiles keeps every sequence's splits and
// kernels. A decode step's four lanes at the most splits fit four lanes'
// workspaces in one dispatch, and a decode lane of up to 16 rows one lane's.
void checkGrouping(uint32_t queryHeads, kv::Layout layout) {
  const std::array sequences{sequenceOf(4000, 13), sequenceOf(0, 1), sequenceOf(65'536, 40), sequenceOf(300, 5)};
  const auto together = planOf(sequences, queryHeads, layout);
  uint32_t tiles = 0, slots = 0, most = 0;
  for (size_t s = 0; s < sequences.size(); ++s) {
    const auto alone = planOf(sequences[s], queryHeads, layout);
    require(together.sequences()[s].splits == alone.sequences()[0].splits,
            "a sequence's splits depend on the sequences beside it");
    tiles += kv::attentionTiles(sequences[s].chunk_tokens);
    slots += kv::attentionTiles(sequences[s].chunk_tokens) * alone.sequences()[0].splits;
    most = std::max(most, alone.sequences()[0].splits);
  }
  require(together.dispatches().size() == 1 && together.dispatches()[0].sequences == 4 &&
              together.dispatches()[0].tiles == tiles && together.dispatches()[0].slots == slots &&
              together.dispatches()[0].splits == most,
          "a command's sequences that fit the workspace do not share one dispatch");
  for (const uint32_t workspace : {ops::kAttentionSequenceSplits, 300U}) {
    const auto apart = planOf(sequences, queryHeads, layout, kBf16, workspace);
    uint32_t next = 0;
    for (const ops::AttentionPlan::Dispatch &dispatch : apart.dispatches()) {
      require(dispatch.slots <= workspace && dispatch.sequences > 0, "a dispatch does not fit its workspace");
      for (uint32_t d = 0; d < dispatch.sequences; ++d)
        require(dispatch.sequence[d] == next++, "a dispatch's sequences are not the next in order");
    }
    require(next == sequences.size() && (workspace < slots) == (apart.dispatches().size() > 1),
            "sequences that do not fit one workspace were not dispatched in turn");
  }
  const auto tails = ops::PagedAttention::lastTiles(together);
  require(tails.dispatches().size() == 1 && tails.dispatches()[0].tiles == sequences.size() &&
              tails.splitPipeline() == together.splitPipeline() && tails.reducePipeline() == together.reducePipeline(),
          "the last tiles' plan departed from its sequences'");
  for (size_t s = 0; s < sequences.size(); ++s)
    require(tails.sequences()[s].firstTile == kv::attentionTiles(sequences[s].chunk_tokens) - 1 &&
                tails.sequences()[s].splits == together.sequences()[s].splits,
            "a sequence's last tile does not keep its splits");
  std::array<kv::ChunkedPrefillParams, 4> lanes{};
  for (uint32_t lane = 0; lane < 4; ++lane) lanes[lane] = sequenceOf(1'000'000 - lane, kLaneRows);
  const auto step = planOf(lanes, queryHeads, layout, kBf16, 4 * kv::kAttentionMaximumSplits);
  require(step.dispatches().size() == 1 && step.dispatches()[0].slots == 4 * kv::kAttentionMaximumSplits,
          "a decode step's lanes at the most splits do not attend in one dispatch");
  // A decode lane of up to SPLASH_TARGET_VERIFY_ROWS rows takes at most a
  // lane's workspace, kAttentionMaximumSplits tile splits, after any history
  // up to the longest context: its one tile up to 128 splits, or a 9-16-row
  // lane's two tiles up to 32 each. Four 16-row lanes at the longest context
  // attend in one dispatch of four lanes' workspaces.
  for (uint32_t rows = 1; rows <= SPLASH_TARGET_VERIFY_ROWS; ++rows)
    for (const uint32_t history : {0U, 65'536U, 131'072U, SPLASH_MAXIMUM_CONTEXT_TOKENS - rows}) {
      const uint32_t tiles = kv::attentionTiles(rows);
      const uint32_t splits = planOf(sequenceOf(history, rows), queryHeads, layout).sequences()[0].splits;
      require(tiles * splits <= kv::kAttentionMaximumSplits && (tiles == 1 || splits <= 32),
              "a decode lane's splits exceed a lane's workspace: " + std::to_string(rows) + " rows after " +
                  std::to_string(history));
    }
  for (uint32_t lane = 0; lane < 4; ++lane)
    lanes[lane] = sequenceOf(SPLASH_MAXIMUM_CONTEXT_TOKENS - SPLASH_TARGET_VERIFY_ROWS - lane,
                             SPLASH_TARGET_VERIFY_ROWS);
  const auto wide = planOf(lanes, queryHeads, layout, kBf16, 4 * kv::kAttentionMaximumSplits);
  require(wide.dispatches().size() == 1 && wide.dispatches()[0].slots <= 4 * kv::kAttentionMaximumSplits,
          "a decode step's 16-row lanes at the longest context do not attend in one dispatch");
  for (const size_t count : {size_t{0}, size_t{5}}) {
    const std::vector<kv::ChunkedPrefillParams> many(count, sequenceOf(0, 8));
    rejects([&] { (void)planOf(many, queryHeads, layout); }, "invalid attention sequences",
            "an attention of no sequences or more than a batch was accepted");
  }
  rejects([&] { (void)planOf(sequences, queryHeads, layout, kBf16, 1); },
          "the attention workspace holds no sequence's splits", "a workspace too small for a sequence was accepted");
  rejects([&] { (void)ops::PagedAttention::chunkParams(kv::kMaximumPhysicalTokens - 7, 8, 32, 1u << 16); },
          "context_out_of_range", "rows past the physical context were accepted");
  rejects([&] { (void)planOf(sequences[0], queryHeads + 1, layout); },
          selectsBlocks(layout) ? "no sparse attention kernel for layout" : "no paged-attention kernel for layout",
          "a query head count without a kernel was accepted");
}

// With int8 activations on, INT8 pages score against INT8 queries (the q8q
// entries) where some tile's splits read at least the threshold's pages
// each, a command's sequences sharing the choice: at 8, a lane's 32 splits
// read 7 pages after 7160 tokens and 8 after 7161, a 64-row chunk's 4 splits
// 7 after 832 tokens and 8 after 833. BF16 pages, selected blocks and int8
// activations off never do.
void checkInt8Queries(uint32_t queryHeads, kv::Layout layout) {
  const std::string split = pipelineName(true, queryHeads, layout);
  std::string int8Split = split;
  if (layout.format == kv::Format::Int8 && !selectsBlocks(layout)) int8Split.replace(int8Split.find("q8_"), 3, "q8q_");
  constexpr ops::AttentionCalibration fromEight = attentionRow(8);
  const auto after = [&](std::vector<kv::ChunkedPrefillParams> sequences, ops::Int8Activations int8) {
    return planOf(sequences, queryHeads, layout, int8, 4 * ops::kAttentionSequenceSplits, fromEight).splitPipeline();
  };
  require(after({sequenceOf(7160, 8)}, kInt8) == split &&
              after({sequenceOf(7160, 8), sequenceOf(7161, 8)}, kInt8) == int8Split &&
              after({sequenceOf(832, 64)}, kInt8) == split && after({sequenceOf(833, 64)}, kInt8) == int8Split &&
              after({sequenceOf(832, 64), sequenceOf(7161, 8)}, kInt8) == int8Split,
          "INT8 queries do not follow the pages the longest split reads");
  require(after({sequenceOf(7160, 8), sequenceOf(7161, 8)}, kBf16) == split && after({sequenceOf(833, 64)}, kBf16) == split,
          "INT8 queries ran with int8 activations off");
}

// The attention layer under test is the second of a pool's two, so its region
// starts past the first one's in every extent.
constexpr uint32_t kLayer = 1;

// A command's lanes, each a sequence of its rows after its history, staged
// after the lanes before it, `stride` rows per KV head; `seeds` set each
// lane's data, so that a lane of another case of the same seed holds the
// same.
struct Case final {
  uint32_t queryHeads;
  kv::Layout layout;
  uint32_t lanes;
  std::array<uint32_t, 4> rows{};
  std::array<uint32_t, 4> seeds{};
  HostKvExtents pool;
  SplashKvLayer layer{};
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer queries;
  std::array<metal::MetalBuffer, 4> tables;
  // Each lane's page ids, which its table holds as entries.
  std::array<std::vector<uint32_t>, 4> pages;
  std::array<kv::ChunkedPrefillParams, 4> stores{};
  // QSA block masks (sparse layouts): maskWords words per row of the
  // command's rows, lane by lane; none when maskWords is 0.
  std::vector<uint32_t> mask;
  uint32_t maskWords = 0;
  // Each lane's rows of the command (SplashLaneRows), which attentionRows
  // writes from `rows`.
  metal::MetalBuffer layoutRows{};
  // The command's packed projection rows, which the reduce reads each row's
  // gates from: +inf (whose sigmoid is 1) unless a check sets them, so a
  // row's output is its attention row.
  metal::MetalBuffer packed{};

  [[nodiscard]] std::span<const kv::ChunkedPrefillParams> sequences() const { return std::span(stores).first(lanes); }
  // The command's rows: each lane's after the lanes before it.
  [[nodiscard]] ops::AttentionRows attentionRows() const {
    auto *entries = static_cast<SplashLaneRows *>(layoutRows.contents());
    for (uint32_t lane = 0; lane < lanes; ++lane) entries[lane] = {rowBegin(lane), rows[lane]};
    return {layoutRows, rowBegin(lanes)};
  }
  // A lane's first row of the command's, its first staged row and its tiles'
  // first list.
  [[nodiscard]] uint32_t rowBegin(uint32_t lane) const {
    return std::accumulate(rows.begin(), rows.begin() + lane, 0U);
  }
  [[nodiscard]] uint64_t stageRow(uint32_t lane) const {
    uint64_t row = 0;
    for (uint32_t before = 0; before < lane; ++before) row += uint64_t{layout.kvHeads} * stores[before].chunk_stride;
    return row;
  }
  [[nodiscard]] uint32_t firstList(uint32_t lane) const {
    uint32_t list = 0;
    for (uint32_t before = 0; before < lane; ++before) list += kv::attentionTiles(stores[before].chunk_tokens);
    return list;
  }
  [[nodiscard]] uint32_t lists() const { return firstList(lanes); }

  bool attends(uint32_t lane, uint32_t row, uint32_t token) const {
    if (!maskWords) return true;
    const uint32_t block = token / SPLASH_ATTENTION_BLOCK_TOKENS;
    return mask[(uint64_t{rowBegin(lane)} + row) * maskWords + block / 32] >> (block % 32) & 1u;
  }
  // An element of the out-projection's input rows the reduce writes: lane's
  // row's (its packed row, each lane's rows after the lanes before it), or
  // with `last` lane's last row's alone, as row `lane`.
  uint64_t outputIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension, bool last = false) const {
    return (uint64_t{last ? lane : rowBegin(lane) + row} * queryHeads + head) * 256 + dimension;
  }
  // A packed projection row's elements: each query head's queries and gate,
  // then the K and V heads.
  [[nodiscard]] uint64_t packedWidth() const { return (uint64_t{queryHeads} * 2 + uint64_t{layout.kvHeads} * 2) * 256; }
  // Queries [KV head][row][query head in group][dimension], keys
  // [KV head][row][dimension] and values [KV head][dimension][row] of each
  // lane's staging.
  uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    const uint32_t group = queryHeads / layout.kvHeads;
    return (((stageRow(lane) + uint64_t{head / group} * stores[lane].chunk_stride + row) * group + head % group) *
                256 +
            dimension);
  }
  uint64_t keyIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return (stageRow(lane) + uint64_t{head} * stores[lane].chunk_stride + row) * 256 + dimension;
  }
  uint64_t valueIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return stageRow(lane) * 256 + (uint64_t{head} * 256 + dimension) * stores[lane].chunk_stride + row;
  }

  uint32_t page(uint32_t lane, uint32_t token) const { return pages[lane][token / 32]; }

  float key(uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index = splash_kv_key_element(head, token % 32, dimension);
    const uint64_t scale = splash_kv_scale_element(head, token % 32);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, id)[index]);
    return pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, id)[index] *
           pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, id)[scale];
  }

  float value(uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) const {
    const uint32_t id = page(lane, token);
    const uint64_t index = pool.valueElement(head, token % 32, dimension);
    const uint64_t scale = splash_kv_scale_element(head, token % 32);
    if (layout.format == kv::Format::BFloat16)
      return bf16ToFloat(pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, id)[index]);
    return pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, id)[index] *
           pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, id)[scale];
  }
};

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// One lane of a case: its rows after its history, of data `seed`, at most
// `bound` (its rows when 0): a decode lane's cap, which sizes what the host
// lays out for it, past its rows.
struct Lane final {
  uint32_t rows;
  uint32_t history;
  uint32_t seed;
  uint32_t bound = 0;
  [[nodiscard]] uint32_t most() const noexcept { return bound ? bound : rows; }
};

// The lanes' pages are mixed over three or more extents of the pool, boundary
// pages first; oneExtent puts the same pages in one extent, which must give
// the same bits.
Case makeCase(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout, std::span<const Lane> lanes,
              bool oneExtent = false) {
  std::array<uint32_t, 4> pageCounts{};
  uint32_t allPages = 0, extent = 0;
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    pageCounts[lane] = (lanes[lane].history + lanes[lane].most() + 31) / 32;
    allPages += pageCounts[lane];
    extent += lanes[lane].history + lanes[lane].most();
  }
  const auto spread = HostKvExtents::spread(allPages + 2);
  const std::vector<uint32_t> ids = HostKvExtents::mixedPages(spread, allPages, extent + uint32_t(lanes.size()));
  kv::Layout poolLayout = layout;
  poolLayout.attentionLayers = kLayer + 1;
  Case data{queryHeads, layout, uint32_t(lanes.size()), {}, {},
            oneExtent ? HostKvExtents(backend, poolLayout, spread.extentPages * spread.extents, 1)
                      : HostKvExtents(backend, poolLayout, spread.extentPages, spread.extents),
            {}, {}, {}, {}, {}, {}, {}, {}, 0};
  data.layer = data.pool.layer(kLayer);
  data.layoutRows = allocate(backend, 4 * sizeof(SplashLaneRows));
  uint32_t firstPage = 0;
  for (size_t lane = 0; lane < lanes.size(); ++lane) {
    const auto [rows, history, seed, bound] = lanes[lane];
    const uint32_t most = lanes[lane].most();
    data.rows[lane] = rows;
    data.seeds[lane] = seed;
    data.pages[lane].assign(ids.begin() + firstPage, ids.begin() + firstPage + pageCounts[lane]);
    firstPage += pageCounts[lane];
    data.tables[lane] = allocate(backend, uint64_t{pageCounts[lane]} * sizeof(SplashKvPage));
    data.pool.writeTable(data.pages[lane], data.tables[lane]);
    data.stores[lane] = ops::PagedAttention::chunkParams(history, most, (most + 31) / 32 * 32, pageCounts[lane]);
  }
  const uint64_t staged = data.stageRow(data.lanes);
  data.keys = allocate(backend, staged * 256 * 2);
  data.values = allocate(backend, data.keys.sizeBytes());
  data.queries = allocate(backend, staged * (queryHeads / layout.kvHeads) * 256 * 2);
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const uint32_t seed = data.seeds[lane];
    for (uint32_t token = 0; token < data.stores[lane].committed_tokens; ++token) {
      const uint32_t id = data.page(lane, token);
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        const uint64_t slot = splash_kv_scale_element(head, token % 32);
        // INT8 scales differ from token to token, so that a scale read at
        // another token's place shows.
        if (layout.format == kv::Format::Int8) {
          data.pool.slab<float>(kLayer, SPLASH_KV_KEY_SCALES, id)[slot] = 0.006f * (1 + float((token + head) % 9) / 16);
          data.pool.slab<float>(kLayer, SPLASH_KV_VALUE_SCALES, id)[slot] =
              0.007f * (1 + float((token * 5 + head) % 11) / 16);
        }
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          const int key = int((token * 37 + head * 101 + dimension * 17 + token * dimension * 3 + seed * 7) % 255) - 127;
          const int value =
              int((token * 53 + head * 79 + dimension * 29 + token * dimension * 5 + seed * 19) % 255) - 127;
          const uint64_t keyIndex = splash_kv_key_element(head, token % 32, dimension);
          const uint64_t valueIndex = data.pool.valueElement(head, token % 32, dimension);
          if (layout.format == kv::Format::Int8) {
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = key;
            data.pool.slab<int8_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = value;
          } else {
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_KEYS, id)[keyIndex] = floatToBf16(key * 0.006f);
            data.pool.slab<uint16_t>(kLayer, SPLASH_KV_VALUES, id)[valueIndex] = floatToBf16(value * 0.007f);
          }
        }
      }
    }
    for (uint32_t row = 0; row < data.rows[lane]; ++row) {
      for (uint32_t head = 0; head < layout.kvHeads; ++head)
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          static_cast<uint16_t *>(data.keys.contents())[data.keyIndex(lane, head, row, dimension)] = floatToBf16(
              float(int((row * 37 + head * 101 + dimension * 17 + seed * 7) % 255) - 127) * 0.006f);
          static_cast<uint16_t *>(data.values.contents())[data.valueIndex(lane, head, row, dimension)] = floatToBf16(
              float(int((row * 53 + head * 79 + dimension * 29 + seed * 19) % 255) - 127) * 0.007f);
        }
      for (uint32_t head = 0; head < queryHeads; ++head)
        for (uint32_t dimension = 0; dimension < 256; ++dimension)
          static_cast<uint16_t *>(data.queries.contents())[data.queryIndex(lane, head, row, dimension)] =
              floatToBf16(float(int((row * 43 + head * 67 + dimension * 11 + head * dimension * 7 + seed * 29) % 1019) -
                                509) /
                          1018.0f);
    }
  }
  for (uint32_t lane = data.lanes; lane < 4; ++lane) {
    data.tables[lane] = data.tables[0];
    data.pages[lane] = data.pages[0];
  }
  data.packed = allocate(backend, uint64_t{data.rowBegin(data.lanes)} * data.packedWidth() * 2);
  auto *packed = static_cast<uint16_t *>(data.packed.contents());
  for (uint32_t row = 0; row < data.rowBegin(data.lanes); ++row)
    for (uint32_t head = 0; head < queryHeads; ++head)
      std::fill_n(packed + row * data.packedWidth() + (uint64_t{head} * 2 + 1) * 256, 256, uint16_t{0x7f80});
  return data;
}

// `count` lanes of `rows` rows after `history` tokens each, of data `seed`
// on, a decode step's (8 rows) histories differing by more than a split's
// worth of pages, so long histories give each lane its own split count.
std::vector<Lane> lanesOf(uint32_t count, uint32_t rows, uint32_t history) {
  std::vector<Lane> lanes;
  for (uint32_t lane = 0; lane < count; ++lane)
    lanes.push_back({rows, history + (rows == kLaneRows ? lane * 649 : 0), lane});
  return lanes;
}

// QSA masks of every lane's rows, in every way a tile's block list must tell
// apart (setQsaMask in q8_flash_attention_metal_test.mm): rows past
// kTopBlocks complete blocks attend whole pages for every row, one block per
// row of a page, blocks only one row attends and pages no row attends, and
// their own block; earlier rows attend causally.
void setQsaMasks(Case &data) {
  constexpr uint32_t kTopBlocks = 40, kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS,
                     kPageBlocks = kv::kPageTokens / kBlockTokens;
  uint32_t visible = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    visible = std::max(visible, data.stores[lane].committed_tokens + data.rows[lane]);
  data.maskWords = visible / kBlockTokens / 32 + 2;
  data.mask.assign(uint64_t{data.rowBegin(data.lanes)} * data.maskWords, 0u);
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t row = 0; row < data.rows[lane]; ++row) {
      uint32_t *bits = data.mask.data() + (uint64_t{data.rowBegin(lane)} + row) * data.maskWords;
      const uint32_t complete = (data.stores[lane].committed_tokens + row + 1) / kBlockTokens;
      for (uint32_t block = 0; block <= complete; ++block) {
        const uint32_t page = block / kPageBlocks, slot = block % kPageBlocks;
        const bool take = complete <= kTopBlocks || block == complete || page % 5 == 0 ||
                          (page % 5 == 2 && slot == row % 8) || (page % 11 == 1 + row % 8 && slot % 2 == 0) ||
                          (page % 13 == 2 + row % 8 && slot == (page + row) % 8);
        if (take) bits[block / 32] |= 1u << (block % 32);
      }
    }
}

// Independent scalar softmax over the exact KV pages produced by the store.
// Sampling query rows bounds the full-2048 oracle cost. Each lane is checked
// independently against its own causal KV history.
void checkReference(const Case &data, const std::vector<uint16_t> &actual) {
  double dot = 0, actualSquared = 0, expectedSquared = 0;
  float maximumError = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const uint32_t rows = data.rows[lane];
    std::vector<uint32_t> selectedRows{0, std::min(7U, rows - 1), rows / 2, rows - 1};
    std::sort(selectedRows.begin(), selectedRows.end());
    selectedRows.erase(std::unique(selectedRows.begin(), selectedRows.end()), selectedRows.end());
    for (uint32_t row : selectedRows) {
      for (uint32_t head : {0U, data.queryHeads - 1}) {
        const uint32_t kvHead = head / (data.queryHeads / data.layout.kvHeads);
        const uint32_t tokens = data.stores[lane].committed_tokens + row + 1;
        std::vector<float> scores(tokens);
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t token = 0; token < tokens; ++token) {
          float score = 0;
          for (uint32_t dimension = 0; dimension < 256; ++dimension)
            score += bf16ToFloat(static_cast<const uint16_t *>(
                         data.queries.contents())[data.queryIndex(lane, head, row, dimension)]) *
                     data.key(lane, kvHead, token, dimension);
          scores[token] =
              data.attends(lane, row, token) ? score * 0.0625f : -std::numeric_limits<float>::infinity();
          maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0;
        for (float &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          float expected = 0;
          for (uint32_t token = 0; token < tokens; ++token)
            expected += scores[token] * data.value(lane, kvHead, token, dimension);
          expected /= denominator;
          const float value = bf16ToFloat(actual[data.outputIndex(lane, head, row, dimension)]);
          require(std::isfinite(value), "attention output is nonfinite");
          maximumError = std::max(maximumError, std::abs(value - expected));
          dot += value * expected;
          actualSquared += value * value;
          expectedSquared += expected * expected;
        }
      }
    }
  }
  const double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  if (!(maximumError < 0.02f && cosine > 0.9995))
    throw std::runtime_error("attention candidate failed scalar KV oracle: history=" +
                             std::to_string(data.stores[0].committed_tokens) + " rows=" +
                             std::to_string(data.rows[0]) + " maximum_absolute_error=" + std::to_string(maximumError) +
                             " cosine=" + std::to_string(cosine));
}

void checkEquivalent(const Case &data, const std::vector<uint16_t> &baseline,
                     const std::vector<uint16_t> &candidate) {
  float maximumError = 0;
  double dot = 0, baselineSquared = 0, candidateSquared = 0;
  for (uint32_t lane = 0; lane < data.lanes; ++lane)
    for (uint32_t head = 0; head < data.queryHeads; ++head)
      for (uint32_t row = 0; row < data.rows[lane]; ++row)
        for (uint32_t dimension = 0; dimension < 256; ++dimension) {
          const uint64_t index = data.outputIndex(lane, head, row, dimension);
          const float left = bf16ToFloat(baseline[index]);
          const float right = bf16ToFloat(candidate[index]);
          require(std::isfinite(right), "attention candidate is nonfinite");
          maximumError = std::max(maximumError, std::abs(left - right));
          dot += left * right;
          baselineSquared += left * left;
          candidateSquared += right * right;
        }
  const double cosine = dot / std::sqrt(baselineSquared * candidateSquared);
  if (!(maximumError < 0.02f && cosine > 0.9995))
    throw std::runtime_error("attention candidate differs from baseline: history=" +
                             std::to_string(data.stores[0].committed_tokens) + " rows=" +
                             std::to_string(data.rows[0]) + " maximum_absolute_error=" + std::to_string(maximumError) +
                             " cosine=" + std::to_string(cosine));
}

// Construct expected extents from the source bits, independently of GPU
// stores: every byte of every extent but the stored rows' slots stays.
std::vector<std::vector<std::byte>> expectedBf16Store(const Case &data) {
  std::vector<std::vector<std::byte>> result;
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    result.emplace_back(bytes.begin(), bytes.end());
  }
  for (unsigned tensor = 0; tensor < 2; ++tensor) {
    const auto *source = static_cast<const uint16_t *>((tensor ? data.values : data.keys).contents());
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      for (uint32_t row = 0; row < data.rows[lane]; ++row) {
        const uint32_t token = data.stores[lane].committed_tokens + row;
        const uint32_t id = data.page(lane, token);
        const auto *page = data.pool.slab<uint16_t>(kLayer, tensor ? SPLASH_KV_VALUES : SPLASH_KV_KEYS, id);
        const uint32_t extent = id / data.pool.extentPages();
        for (uint32_t head = 0; head < data.layout.kvHeads; ++head)
          for (uint32_t d = 0; d < 256; ++d) {
            const uint64_t element = tensor ? data.pool.valueElement(head, token % 32, d)
                                            : splash_kv_key_element(head, token % 32, d);
            const auto offset =
                reinterpret_cast<const std::byte *>(page + element) - data.pool.bytes(extent).data();
            const uint64_t input = tensor ? data.valueIndex(lane, head, row, d) : data.keyIndex(lane, head, row, d);
            std::memcpy(result[extent].data() + offset, source + input, sizeof(uint16_t));
          }
      }
    }
  }
  return result;
}

void checkBf16Store(const Case &data, const std::vector<std::vector<std::byte>> &expected) {
  for (uint32_t extent = 0; extent < data.pool.extentCount(); ++extent) {
    const auto bytes = data.pool.bytes(extent);
    require(std::equal(bytes.begin(), bytes.end(), expected[extent].begin()),
            "BF16 store changed source bits, history, another layer or an unused page slot");
  }
}

// The case's staged K/V into its pages, as the q/k projection stores rows.
void addStore(metal::CommandGraph &graph, const Case &data) {
  test::addStagedStore(graph, data.layer, data.keys, data.values, std::span(data.tables).first(data.lanes),
                       data.sequences(), data.attentionRows(), data.layout);
}

void checkBf16StoreEdges(metal::MetalBackend &backend) {
  for (const Geometry geometry : kGeometries) {
    const std::array lanes{Lane{33, 31, 0}, Lane{8, 100, 1}};
    auto data = makeCase(backend, geometry.queryHeads, geometry.layout(kv::Format::BFloat16), lanes);
    // Signed zero, subnormals, large finite values, infinities and NaN payloads.
    constexpr std::array<uint16_t, 12> bits{0, 0x8000, 1, 0x8001, 0x007f, 0x0080,
                                            0x4960, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc3};
    for (auto buffer : {data.keys, data.values}) {
      auto *source = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i) source[i] = bits[i % bits.size()];
    }
    const auto expected = expectedBf16Store(data);
    metal::CommandGraph graph;
    addStore(graph, data);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    checkBf16Store(data, expected);
  }
}

// The page table and layer a lane's attention reads its K/V through.
using AttendedKv = std::pair<metal::MetalBuffer, SplashKvLayer>;

// How a run attends: the plan's workspace (its tile splits), only each
// lane's last tile (lastTiles), which outputs each lane's last row alone, as
// row `lane`, with `locate`, the lists are made in a command
// of their own, after which locate gives the attention their entries'
// locations (ops::AttentionBlocks::locations), and with `stage`, lane 0
// attends through the table and layer stage gives, which it fills after the
// store (ops::PagedAttention::addStageLayer).
struct Run final {
  uint32_t slots = 4 * ops::kAttentionSequenceSplits;
  ops::Int8Activations int8 = kInt8;
  bool lastTiles = false;
  bool testBounds = false;
  // The out-projection's table the reduce writes beside its rows, and the
  // scratch that takes it (AttentionOutputRows).
  ops::LinearInput input = ops::LinearInput::Plain;
  ops::LinearScratch scratch{};
  std::function<void(ops::AttentionBlocks &)> locate{};
  std::function<AttendedKv(metal::CommandGraph &)> stage{};
};

// The store and attention of a case's lanes as the runtime encodes a
// command's sequences, each dispatch checked against its plan; its output,
// the out-projection's input rows (Case::outputIndex).
std::vector<uint16_t> run(metal::MetalBackend &backend, Case &data, const Run &how = {}) {
  const ops::AttentionPlan whole = planOf(data.sequences(), data.queryHeads, data.layout, how.int8, how.slots);
  const ops::AttentionPlan plan = how.lastTiles ? ops::PagedAttention::lastTiles(whole) : whole;
  constexpr uint64_t guardBytes = 256;
  const uint64_t written = how.lastTiles ? data.lanes : data.rowBegin(data.lanes);
  const std::array sizes{plan.workspace().partialsBytes, plan.workspace().statisticsBytes,
                         written * data.queryHeads * 256 * 2};
  std::array<metal::MetalBuffer, 3> backing, views;
  for (size_t i = 0; i < sizes.size(); ++i) {
    backing[i] = allocate(backend, sizes[i] + 2 * guardBytes);
    std::memset(backing[i].contents(), 0xa5, sizes[i] + 2 * guardBytes);
    views[i] = backend.view(backing[i], guardBytes, sizes[i]);
    std::memset(views[i].contents(), 0, sizes[i]);
  }
  const auto partials = views[0], statistics = views[1], output = views[2];
  const ops::AttentionOutputRows outputRows{data.packed, output,
                                            how.lastTiles ? ops::AttentionOutput::Last : ops::AttentionOutput::Every,
                                            how.scratch, how.input};
  const ops::AttentionRows command = data.attentionRows();
  std::vector<ops::AttentionHistory> histories;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) histories.push_back({data.layer, data.tables[lane]});
  if (how.testBounds) {
    metal::CommandGraph shortGraph;
    std::vector<kv::ChunkedPrefillParams> mismatch(data.sequences().begin(), data.sequences().end());
    mismatch[0].chunk_tokens = mismatch[0].chunk_tokens == 1 ? 2 : mismatch[0].chunk_tokens - 1;
    rejects([&] {
      (void)ops::PagedAttention::addAttention(shortGraph, plan, data.queries, partials, statistics, mismatch, histories,
                                              command, outputRows);
    },
            "attention sequences do not match the plan", "sequences of other rows than their plan's were accepted");
    require(shortGraph.empty(), "a mismatched plan partially encoded a graph");
  }
  // With QSA masks the tiles attend through the block lists the graph makes
  // of them first, lane l's tiles from list firstList(l) on.
  metal::CommandGraph graph;
  ops::AttentionBlocks blocks;
  if (data.maskWords) {
    uint32_t visible = 0;
    for (uint32_t lane = 0; lane < data.lanes; ++lane)
      visible = std::max(visible, data.stores[lane].committed_tokens + data.rows[lane]);
    const uint32_t stride = visible / SPLASH_ATTENTION_BLOCK_TOKENS + 1;
    const metal::MetalBuffer masks = allocate(backend, data.mask.size() * sizeof(uint32_t));
    std::memcpy(masks.contents(), data.mask.data(), masks.sizeBytes());
    blocks = {allocate(backend, uint64_t{data.lists()} * stride * sizeof(uint32_t)),
              allocate(backend, data.lists() * sizeof(uint32_t)), stride, 0};
    std::vector<ops::BlockListRows> sequences;
    for (uint32_t lane = 0; lane < data.lanes; ++lane)
      sequences.push_back(
          {lane, data.stores[lane].chunk_tokens, data.stores[lane].committed_tokens, data.firstList(lane)});
    ops::PagedAttention::addBlockLists(graph, masks, data.maskWords, sequences, command, blocks);
  }
  if (how.locate) {
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    graph = {};
    how.locate(blocks);
  }
  const size_t first = graph.dispatches().size();
  addStore(graph, data);
  if (how.stage) std::tie(histories[0].pageTable, histories[0].kv) = how.stage(graph);
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    histories[lane].blocks = blocks;
    if (blocks.stride) histories[lane].blocks.list += data.firstList(lane);
  }
  const size_t attention = graph.dispatches().size();
  (void)ops::PagedAttention::addAttention(graph, plan, data.queries, partials, statistics, data.sequences(), histories,
                                          command, outputRows);
  require(how.stage || attention == first + 1, "the case's store should encode one dispatch before the attention");
  // Each planned dispatch runs as a dispatch per run of its sequences that
  // read their layer in one placement (SplashAttentionParams::kv): a split and
  // a reduce each, on the planned dispatch's partial slots.
  const auto &dispatches = graph.dispatches();
  size_t encoded = attention;
  for (const ops::AttentionPlan::Dispatch &planned : plan.dispatches()) {
    uint32_t slot = 0;
    for (uint32_t first = 0, end = 0; first < planned.sequences; first = end) {
      const SplashKvLayer layer = histories[planned.sequence[first]].kv;
      ops::AttentionPlan::Dispatch run{};
      for (end = first; end < planned.sequences && histories[planned.sequence[end]].kv.offset == layer.offset &&
                        histories[planned.sequence[end]].kv.extent_pages == layer.extent_pages;
           ++end) {
        const ops::AttentionPlan::Sequence &sequence = plan.sequences()[planned.sequence[end]];
        const uint32_t tiles = kv::attentionTiles(sequence.rows) - sequence.firstTile;
        run.sequence[run.sequences++] = planned.sequence[end];
        run.tiles += tiles;
        run.splits = std::max(run.splits, sequence.splits);
      }
      const auto checkDispatch = [&](const metal::ComputeDispatch &dispatch, metal::DispatchSize groups,
                                     std::string_view pipeline) {
        require(dispatch.pipelineName == pipeline && dispatch.threadgroups.x == groups.x &&
                    dispatch.threadgroups.y == groups.y && dispatch.threadgroups.z == groups.z &&
                    dispatch.bytes.size() == 1 && dispatch.bytes[0].sizeBytes == sizeof(kv::AttentionParams),
                "production attention dispatch departed from its plan");
        kv::AttentionParams params;
        std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
        require(params.sequences == run.sequences && params.first_sequence == run.sequence[0] &&
                    params.rows == command.rows && params.tiles == run.tiles &&
                    params.kv.extent_pages == layer.extent_pages && params.kv.offset == layer.offset,
                "a dispatch's parameters do not hold its sequences");
        uint32_t tile = 0, next = slot;
        for (uint32_t s = 0; s < run.sequences; ++s) {
          const uint32_t lane = run.sequence[s], skipped = plan.sequences()[lane].firstTile * 8;
          const SplashAttentionSequence &sequence = params.sequence[s];
          require(sequence.committed_tokens == data.stores[lane].committed_tokens + skipped &&
                      sequence.row_bound == data.stores[lane].chunk_tokens - skipped &&
                      sequence.layout_row == skipped &&
                      sequence.stage_row == data.stageRow(lane) + skipped &&
                      sequence.chunk_stride == data.stores[lane].chunk_stride &&
                      sequence.page_table_entries == data.stores[lane].page_table_entries &&
                      sequence.split_count == plan.sequences()[lane].splits && sequence.first_tile == tile &&
                      sequence.first_slot == next && sequence.block_stride == blocks.stride &&
                      sequence.block_list ==
                          (blocks.stride ? data.firstList(lane) + plan.sequences()[lane].firstTile : 0),
                  "recorded attention ABI does not describe the actual plan");
          const uint32_t tiles = kv::attentionTiles(sequence.row_bound);
          tile += tiles;
          next += tiles * data.layout.kvHeads * sequence.split_count;
        }
        return next;
      };
      require(dispatches.size() >= encoded + 2, "production attention encoded fewer dispatches than its runs");
      const uint32_t next = checkDispatch(dispatches[encoded], plan.splitGroups(run),
                                          plan.splitPipeline(static_cast<bool>(blocks.locations)));
      checkDispatch(dispatches[encoded + 1], plan.reduceGroups(run),
                    std::string(plan.reducePipeline()) + ops::tableSuffix(how.input));
      slot = next;
      encoded += 2;
    }
  }
  require(dispatches.size() == encoded, "production attention encoded other dispatches than its runs'");
  const auto expected =
      data.layout.format == kv::Format::BFloat16 ? expectedBf16Store(data) : std::vector<std::vector<std::byte>>{};
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  if (data.layout.format == kv::Format::BFloat16) checkBf16Store(data, expected);
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto *bytes = static_cast<const uint8_t *>(backing[i].contents());
    for (uint64_t byte = 0; byte < guardBytes; ++byte)
      require(bytes[byte] == 0xa5 && bytes[guardBytes + sizes[i] + byte] == 0xa5,
              "attention scratch/output write canary changed");
  }
  const auto *values = static_cast<const uint16_t *>(output.contents());
  return {values, values + output.sizeBytes() / 2};
}

void checkPrefill(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout, uint32_t history, uint32_t rows) {
  const std::array lane{Lane{rows, history, 0}};
  auto data = makeCase(backend, heads, layout, lane);
  const auto output = run(backend, data, {.testBounds = true});
  checkReference(data, output);
  checkEquivalent(data, output, run(backend, data));
  auto oneExtent = makeCase(backend, heads, layout, lane, true);
  require(run(backend, oneExtent) == output, "attention over extents differs from one extent of the same pages");
  if (rows == 1057) {
    // Reuse the identical BF16 inputs and KV history across unaligned
    // host chunks, checking each path against its independent causal oracle.
    const auto copy = [](metal::MetalBuffer buffer) {
      const auto *begin = static_cast<const uint16_t *>(buffer.contents());
      return std::vector<uint16_t>(begin, begin + buffer.sizeBytes() / 2);
    };
    const auto keys = copy(data.keys), values = copy(data.values), queries = copy(data.queries);
    const Case full = makeCase(backend, heads, layout, lane);
    uint32_t offset = 0;
    for (uint32_t chunk : {3U, 5U, 31U, 509U, 509U}) {
      data.rows[0] = chunk;
      data.stores[0].committed_tokens = history + offset;
      data.stores[0].chunk_tokens = chunk;
      for (uint32_t head = 0; head < layout.kvHeads; ++head)
        for (uint32_t row = 0; row < chunk; ++row)
          for (uint32_t d = 0; d < 256; ++d) {
            static_cast<uint16_t *>(data.keys.contents())[data.keyIndex(0, head, row, d)] =
                keys[full.keyIndex(0, head, offset + row, d)];
            static_cast<uint16_t *>(data.values.contents())[data.valueIndex(0, head, row, d)] =
                values[full.valueIndex(0, head, offset + row, d)];
          }
      for (uint32_t head = 0; head < heads; ++head)
        for (uint32_t row = 0; row < chunk; ++row)
          for (uint32_t d = 0; d < 256; ++d)
            static_cast<uint16_t *>(data.queries.contents())[data.queryIndex(0, head, row, d)] =
                queries[full.queryIndex(0, head, offset + row, d)];
      const auto chunkOutput = run(backend, data, {.testBounds = true});
      checkReference(data, chunkOutput);
      checkEquivalent(data, chunkOutput, run(backend, data));
      offset += chunk;
    }
    require(offset == rows, "chunk comparison dropped logical query rows");
  }
  std::cout << "paged attention chunk: format=" << kv::formatName(layout.format) << " q=" << heads
            << " history=" << history << " rows=" << rows << " PASS\n";
}

// Each lane's last tile attended alone (ops::PagedAttention::lastTiles) on
// its sequence's splits, from the tile's staged rows, with the rows before
// it committed and, with `masks`, QSA masks of a sparse layout, through its
// sequence's block lists from its tile's on, writes the bits the whole
// command writes for its last row, as row `lane`.
void checkLastTiles(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout, std::span<const Lane> lanes,
                    bool masks = false) {
  auto data = makeCase(backend, heads, layout, lanes);
  if (masks) setQsaMasks(data);
  const std::vector<uint16_t> whole = run(backend, data);
  const std::vector<uint16_t> tails = run(backend, data, {.lastTiles = true});
  const uint64_t rowValues = uint64_t{heads} * layout.headDimension;
  for (uint32_t lane = 0; lane < data.lanes; ++lane) {
    const uint64_t at = data.outputIndex(lane, 0, 0, 0, true), from = data.outputIndex(lane, 0, data.rows[lane] - 1, 0);
    require(std::equal(tails.begin() + std::ptrdiff_t(at), tails.begin() + std::ptrdiff_t(at + rowValues),
                       whole.begin() + std::ptrdiff_t(from)),
            "a last tile's last row differs from its whole sequence's");
  }
  std::cout << "paged attention last tiles: format=" << kv::formatName(layout.format) << " q=" << heads
            << " lanes=" << data.lanes << (masks ? " (block lists)" : "") << " PASS\n";
}

// A prefill that reads its history staged, as one whose history has pages
// that hold only their index does (ops::PagedAttention::addStageLayer): the
// pool table it stages from has every third page of the history
// SPLASH_KV_PAGE_ABSENT, whose bytes the host copies into staging first, as
// the SSD reads fill them; the history's other pages, and after the store the
// chunk's own pages, are copied on the GPU; and the attention through the
// staged table and layer gives the pool's bits. Staging holds one layer in
// extents of the pool's size, its pages in the reverse order of the table's
// over more than the table holds; every staged page ends with the bytes of
// its pool page and every other one as it was. A second sequence reads the
// pool beside it, in a dispatch of its own (its layer placed apart), with
// the bits it has in one dispatch with the first.
void checkStagedPrefill(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout, uint32_t history,
                        uint32_t rows, bool masks) {
  const std::array lanes{Lane{rows, history, 0}, Lane{13, history / 2 + 5, 1}};
  auto data = makeCase(backend, heads, layout, lanes);
  if (masks) setQsaMasks(data);
  const std::vector<uint16_t> baseline = run(backend, data);
  const uint32_t pages = data.stores[0].page_table_entries, chunkFirst = history / kv::kPageTokens;
  kv::Layout stagingLayout = layout;
  stagingLayout.attentionLayers = 1;
  const uint32_t extentPages = data.pool.extentPages();
  const HostKvExtents staging(backend, stagingLayout, extentPages, pages / extentPages + 2);
  for (uint32_t extent = 0; extent < staging.extentCount(); ++extent)
    std::ranges::fill(staging.bytes(extent), std::byte{0xa5});
  std::vector<uint32_t> stagedPages(pages);
  for (uint32_t page = 0; page < pages; ++page) stagedPages[page] = staging.pageCount() - 1 - page;
  const metal::MetalBuffer stagedTable = allocate(backend, pages * sizeof(SplashKvPage)),
                           poolTable = allocate(backend, pages * sizeof(SplashKvPage));
  staging.writeTable(stagedPages, stagedTable);
  std::memcpy(poolTable.contents(), data.tables[0].contents(), poolTable.sizeBytes());
  data.pool.reachThrough(poolTable);
  for (uint32_t page = 0; page < chunkFirst; page += 3) {
    static_cast<SplashKvPage *>(poolTable.contents())[page] = SPLASH_KV_PAGE_ABSENT;
    const auto from = data.pool.layerSpans(data.pages[0][page], kLayer);
    const auto to = staging.layerSpans(stagedPages[page], 0);
    for (size_t tensor = 0; tensor < from.size(); ++tensor) std::ranges::copy(from[tensor], to[tensor].begin());
  }
  const std::vector<uint16_t> output = run(backend, data, {.stage = [&](metal::CommandGraph &graph) {
                                                             const SplashKvLayer layer = staging.layer(0);
                                                             if (chunkFirst)
                                                               ops::PagedAttention::addStageLayer(
                                                                   graph, data.layer, poolTable, layer, stagedTable, 0,
                                                                   chunkFirst, layout);
                                                             ops::PagedAttention::addStageLayer(
                                                                 graph, data.layer, poolTable, layer, stagedTable,
                                                                 chunkFirst, pages - chunkFirst, layout);
                                                             return AttendedKv{stagedTable, layer};
                                                           }});
  require(output == baseline, "attention through staged history differs from the pool's");
  for (uint32_t page = 0; page < staging.pageCount(); ++page) {
    const uint32_t listed = staging.pageCount() - 1 - page;
    const auto staged = staging.layerSpans(page, 0);
    for (size_t tensor = 0; tensor < staged.size(); ++tensor) {
      const auto unstaged = [](std::byte byte) { return byte == std::byte{0xa5}; };
      const bool kept =
          listed < pages ? std::ranges::equal(staged[tensor], data.pool.layerSpans(data.pages[0][listed], kLayer)[tensor])
                         : std::ranges::all_of(staged[tensor], unstaged);
      require(kept, "staging differs from its pool pages or changed a page it does not stage");
    }
  }
  std::cout << "staged prefill: format=" << kv::formatName(layout.format) << " q=" << heads << " history=" << history
            << " rows=" << rows << (masks ? " lists" : "") << " PASS\n";
}

// Lanes through block locations (ops::AttentionBlocks::locations, as the hot
// block cache of a request whose K/V spilled gives them): each lane's listed
// blocks of its committed history moved to places of a cache of at least two
// extents in shuffled order (moveBlock, which leaves other bits in the pool),
// every third entry and the blocks the step writes left to the page table
// (SPLASH_ATTENTION_PAGED), attend as the lists alone did before the move,
// bit for bit. Locations past a list's count name no page of the cache.
void checkLocations(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout) {
  constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS, kPageBlocks = kv::kPageTokens / kBlockTokens;
  auto data = makeCase(backend, heads, layout, lanesOf(3, kLaneRows, 1023));
  setQsaMasks(data);
  const std::vector<uint16_t> baseline = run(backend, data);
  std::optional<HostKvExtents> cache;
  metal::MetalBuffer locations, cacheTable;
  std::vector<uint32_t> slots;
  uint32_t placed = 0, paged = 0;
  const std::vector<uint16_t> output = run(backend, data, {.locate = [&](ops::AttentionBlocks &blocks) {
    const auto *lists = static_cast<const uint32_t *>(blocks.lists.contents());
    const auto *counts = static_cast<const uint32_t *>(blocks.counts.contents());
    // The cache holds the pool's layers where the pool's extents do: the
    // located split reaches its pages through the pool's layer placement.
    kv::Layout cacheLayout = layout;
    cacheLayout.attentionLayers = kLayer + 1;
    const uint32_t extentPages = data.pool.extentPages();
    const uint32_t pages = (std::accumulate(counts, counts + data.lanes, 0U) + kPageBlocks - 1) / kPageBlocks;
    cache.emplace(backend, cacheLayout, extentPages, std::max(2U, (pages + extentPages - 1) / extentPages));
    std::vector<uint32_t> cachePages(cache->pageCount());
    std::iota(cachePages.begin(), cachePages.end(), 0U);
    cacheTable = allocate(backend, cachePages.size() * sizeof(SplashKvPage));
    cache->writeTable(cachePages, cacheTable);
    slots.resize(cachePages.size() * kPageBlocks);
    std::iota(slots.begin(), slots.end(), 0U);
    std::shuffle(slots.begin(), slots.end(), std::mt19937(7));
    locations = allocate(backend, uint64_t{data.lanes} * blocks.stride * sizeof(uint32_t));
    std::memset(locations.contents(), 0xfe, locations.sizeBytes());
    auto *located = static_cast<uint32_t *>(locations.contents());
    for (uint32_t lane = 0; lane < data.lanes; ++lane)
      for (uint32_t entry = 0; entry < counts[lane]; ++entry) {
        const uint32_t block = lists[lane * blocks.stride + entry] >> SPLASH_ATTENTION_BLOCK_ROW_BITS;
        uint32_t &location = located[lane * blocks.stride + entry];
        if (entry % 3 == 0 || (block + 1) * kBlockTokens > data.stores[lane].committed_tokens) {
          location = SPLASH_ATTENTION_PAGED;
          ++paged;
          continue;
        }
        location = slots[placed++];
        moveBlock(cacheLayout, kLayer, data.pool, data.page(lane, block * kBlockTokens), block % kPageBlocks, *cache,
                  location / kPageBlocks, location % kPageBlocks);
      }
    blocks.locations = locations;
    blocks.locationPages = cacheTable;
  }});
  const auto extent = [&](uint32_t index) { return slots[index] / kPageBlocks / cache->extentPages(); };
  bool extents = false;
  for (uint32_t index = 1; index < placed; ++index) extents |= extent(index) != extent(0);
  require(paged && extents, "the location test left no block to the page table or placed blocks in one extent");
  require(output == baseline, "attention through block locations differs from the lists alone");
  std::cout << "paged sparse lanes through locations: format=" << kv::formatName(layout.format)
            << " located=" << placed << " paged=" << paged << " PASS\n";
}

void checkLanes(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout, uint32_t history, uint32_t lanes) {
  auto data = makeCase(backend, heads, layout, lanesOf(lanes, kLaneRows, history));
  const auto output = run(backend, data, {.testBounds = true});
  checkReference(data, output);
  auto oneExtent = makeCase(backend, heads, layout, lanesOf(lanes, kLaneRows, history), true);
  require(run(backend, oneExtent) == output, "lanes' attention over extents differs from one extent of the same pages");
  std::cout << "paged attention lanes: format=" << kv::formatName(layout.format) << " q=" << heads
            << " history=" << history << " lanes=" << lanes << " PASS\n";
}

// Sequences of 1, 5, 8, 13 and 40 rows at histories of their own, in one
// dispatch and in dispatches of a workspace that holds fewer, attend as each
// alone, bit for bit: no sequence's partition follows another's. (Their
// queries score as INT8 or not by the command's longest tile, as a decode
// step's lanes always did, checkInt8Queries: this runs bf16 queries.) Sparse
// layouts attend through their masks' lists.
void checkMixed(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout) {
  const bool masks = selectsBlocks(layout);
  for (const std::vector<Lane> &command :
       {std::vector<Lane>{{13, 4093, 0}, {1, 0, 1}, {40, 1023, 2}, {5, 33, 3}},
        std::vector<Lane>{{8, 16'384, 4}, {40, 6145, 5}, {1, 255, 6}}}) {
    auto data = makeCase(backend, heads, layout, command);
    if (masks) setQsaMasks(data);
    const std::vector<uint16_t> together = run(backend, data, {.int8 = kBf16});
    require(run(backend, data, {.slots = ops::kAttentionSequenceSplits, .int8 = kBf16}) == together,
            "sequences dispatched apart differ from one dispatch");
    checkReference(data, together);
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      const std::array one{command[lane]};
      auto alone = makeCase(backend, heads, layout, one);
      if (masks) setQsaMasks(alone);
      const std::vector<uint16_t> output = run(backend, alone, {.int8 = kBf16});
      for (uint32_t head = 0; head < heads; ++head)
        for (uint32_t row = 0; row < data.rows[lane]; ++row) {
          const uint64_t at = data.outputIndex(lane, head, row, 0), from = alone.outputIndex(0, head, row, 0);
          require(std::equal(output.begin() + std::ptrdiff_t(from), output.begin() + std::ptrdiff_t(from + 256),
                             together.begin() + std::ptrdiff_t(at)),
                  "a sequence beside others attends otherwise than alone");
        }
    }
  }
  std::cout << "mixed sequences: format=" << kv::formatName(layout.format) << " q=" << heads
            << " each as alone, in one dispatch and apart PASS\n";
}

// A decode step's lanes packed by their rows (ops::AttentionRows), each of 1
// to its bound's rows, a bound of 8 or 16 a lane (a cycle's cap), at
// histories of its own: the store and attention of every lane, together and
// each lane alone at its bound (whose split count is the same), bit for bit
// (bf16 queries), and against the scalar oracle, which reads a 9-16-row
// lane's first tile's keys from the pages the store wrote before its second
// tile attends. Past a lane's rows nothing is written: no output row past the
// step's rows (run's canaries), and its staged keys and values past its rows
// (nonzero here) reach none of its bound's slots past its rows, which keep
// theirs. Sparse layouts attend through their masks' lists.
void checkPackedLanes(metal::MetalBackend &backend, uint32_t heads, kv::Layout layout) {
  constexpr uint8_t kSlotFill = 0x5a;
  const bool masks = selectsBlocks(layout);
  const uint64_t element = layout.format == kv::Format::Int8 ? 1 : 2;
  // Each byte of a lane's slots past its rows within its bound.
  const auto spareSlotBytes = [&](Case &data, uint32_t lane, const std::function<void(uint8_t &)> &visit) {
    const uint32_t first = data.stores[lane].committed_tokens + data.rows[lane],
                   end = data.stores[lane].committed_tokens + data.stores[lane].chunk_tokens;
    for (uint32_t token = first; token < end; ++token) {
      const uint32_t id = data.page(lane, token), place = token % kv::kPageTokens;
      for (uint32_t head = 0; head < layout.kvHeads; ++head) {
        for (uint32_t dimension = 0; dimension < 256; ++dimension)
          for (uint64_t byte = 0; byte < element; ++byte) {
            visit(data.pool.slab<uint8_t>(kLayer, SPLASH_KV_KEYS, id)[splash_kv_key_element(head, place, dimension) *
                                                                          element +
                                                                      byte]);
            visit(data.pool.slab<uint8_t>(kLayer, SPLASH_KV_VALUES,
                                          id)[data.pool.valueElement(head, place, dimension) * element + byte]);
          }
        if (layout.format == kv::Format::Int8)
          for (const uint32_t tensor : {SPLASH_KV_KEY_SCALES, SPLASH_KV_VALUE_SCALES})
            for (uint64_t byte = 0; byte < sizeof(float); ++byte)
              visit(data.pool.slab<uint8_t>(kLayer, tensor, id)[splash_kv_scale_element(head, place) * sizeof(float) +
                                                                byte]);
      }
    }
  };
  for (const std::vector<Lane> &command :
       {std::vector<Lane>{{1, 300, 0, 8}, {8, 4093, 1, 8}, {5, 0, 2, 8}, {3, 1023, 3, 8}},
        std::vector<Lane>{{16, 2049, 4, 16}, {9, 31, 5, 16}, {1, 6145, 6, 16}, {12, 700, 7, 16}},
        std::vector<Lane>{{3, 100, 8, 16}, {16, 0, 9, 16}}, std::vector<Lane>{{13, 16'384, 10, 16}}}) {
    auto data = makeCase(backend, heads, layout, command);
    if (masks) setQsaMasks(data);
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      for (uint32_t row = data.rows[lane]; row < data.stores[lane].chunk_tokens; ++row)
        for (uint32_t head = 0; head < layout.kvHeads; ++head)
          for (uint32_t dimension = 0; dimension < 256; ++dimension) {
            static_cast<uint16_t *>(data.keys.contents())[data.keyIndex(lane, head, row, dimension)] = 0x3f80;
            static_cast<uint16_t *>(data.values.contents())[data.valueIndex(lane, head, row, dimension)] = 0x3f80;
          }
      spareSlotBytes(data, lane, [](uint8_t &byte) { byte = kSlotFill; });
    }
    const std::vector<uint16_t> together = run(backend, data, {.int8 = kBf16});
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      bool kept = true;
      spareSlotBytes(data, lane, [&](uint8_t &byte) { kept = kept && byte == kSlotFill; });
      require(kept, "a lane's store wrote a slot past its rows");
    }
    checkReference(data, together);
    for (uint32_t lane = 0; lane < data.lanes; ++lane) {
      const std::array one{command[lane]};
      auto alone = makeCase(backend, heads, layout, one);
      if (masks) setQsaMasks(alone);
      const std::vector<uint16_t> output = run(backend, alone, {.int8 = kBf16});
      for (uint32_t head = 0; head < heads; ++head)
        for (uint32_t row = 0; row < data.rows[lane]; ++row) {
          const uint64_t at = data.outputIndex(lane, head, row, 0), from = alone.outputIndex(0, head, row, 0);
          require(std::equal(output.begin() + std::ptrdiff_t(from), output.begin() + std::ptrdiff_t(from + 256),
                             together.begin() + std::ptrdiff_t(at)),
                  "a packed lane attends otherwise than alone");
        }
    }
  }
  std::cout << "packed lanes: format=" << kv::formatName(layout.format) << " q=" << heads
            << " lanes of 1-16 rows at bounds 8 and 16, each as alone, spare tiles and slots untouched PASS\n";
}

// The reduce's out-projection rows of a decode step's lanes of a cap of 16
// rows holding 1, 16, 9 and 6 (ops::AttentionRows), each with gates of its
// own in its packed row: every row its attention row (the rows of gates of
// +inf, whose sigmoid is 1) times its query head's gate's sigmoid, rounded
// once to bf16 (within its ulp, for the device's exp2), and with Table16 the
// out-projection's table and sums of the rows, bit for bit as the
// projection's own preparation of them makes them.
void checkPackedGate(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout) {
  const std::vector<Lane> lanes{{1, 300, 0, 16}, {16, 31, 1, 16}, {9, 700, 2, 16}, {6, 0, 3, 16}};
  auto data = makeCase(backend, queryHeads, layout, lanes);
  if (selectsBlocks(layout)) setQsaMasks(data);
  const std::vector<uint16_t> attention = run(backend, data, {.int8 = kBf16});
  const uint32_t rows = data.rowBegin(data.lanes), width = queryHeads * 256;
  auto *packed = static_cast<uint16_t *>(data.packed.contents());
  const auto gateOf = [&](uint32_t row, uint32_t head, uint32_t d) -> uint16_t & {
    return packed[row * data.packedWidth() + (uint64_t{head} * 2 + 1) * 256 + d];
  };
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t head = 0; head < queryHeads; ++head)
      for (uint32_t d = 0; d < 256; ++d)
        gateOf(row, head, d) = floatToBf16(float(int((row * 7919 + head * 104729 + d * 31) % 1999) - 999) / 333.0f);
  const auto tables = [&] {
    return ops::LinearScratch{allocate(backend, ops::tableBytes(width, rows)),
                              allocate(backend, ops::tableSumsBytes(ops::LinearInput::Table16, width, rows)), {}, {}};
  };
  const ops::LinearScratch scratch = tables(), expected = tables();
  const std::vector<uint16_t> gated =
      run(backend, data, {.int8 = kBf16, .input = ops::LinearInput::Table16, .scratch = scratch});
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t head = 0; head < queryHeads; ++head)
      for (uint32_t d = 0; d < 256; ++d) {
        const uint64_t index = (uint64_t{row} * queryHeads + head) * 256 + d;
        const double product = double(bf16ToFloat(attention[index])) / (1 + std::exp(-double(bf16ToFloat(gateOf(row, head, d)))));
        require(std::fabs(bf16ToFloat(gated[index]) - product) <= ulpBf16(float(product)),
                "a gated row is not its attention row times its gate's sigmoid");
      }
  const metal::MetalBuffer hidden = allocate(backend, uint64_t{rows} * width * 2);
  std::copy(gated.begin(), gated.end(), static_cast<uint16_t *>(hidden.contents()));
  metal::CommandGraph graph;
  test::addReferencePreparation(graph, ops::LinearInput::Table16, hidden, expected.input, expected.sums, width,
                                rows / kLaneRows);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  require(!std::memcmp(scratch.input.contents(), expected.input.contents(), scratch.input.sizeBytes()) &&
              !std::memcmp(scratch.sums.contents(), expected.sums.contents(), scratch.sums.sizeBytes()),
          "the reduce's table differs from the projection's preparation of its rows");
  std::cout << "gated rows: format=" << kv::formatName(layout.format) << " q=" << queryHeads
            << " lanes of 1, 16, 9 and 6 rows of 16, Table16 PASS\n";
}

// The q/k RMS norms, RoPE and K/V store of the attention prepare against
// fp64 with norm weights in bf16 or F32 (a GGUF's), of a command's sequences:
// three lanes of 8 rows in a KV block each, a chunk of 37 rows in 64, and
// sequences of 5, 13 and 8 rows. Past the rotary pairs a row holds the norm
// rounded once to bf16; each rotated value is within an ulp of the fp64
// rotation of the bf16-rounded norms plus an ulp of the larger input, which
// covers the fp32 kernel rounding a norm to the other bf16 neighbour and a
// rotation that cancels. The queries are staged; a row's keys and values sit
// in its page slot after the sequence's committed tokens: BF16 pages hold
// the keys so and the values' bits, INT8 pages a scale per row and head of
// the largest magnitude over 127 (that of the bf16 keys or values, within an
// ulp of the division) and codes of the values over it rounded to nearest
// (within one of the reference's, for a rounding of the quotient), so a key
// dequantizes within half a scale of the bound above. No slot past a
// sequence's rows, nor its staged queries, is written.
void checkProjection(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout, bool float32,
                     std::span<const kv::ChunkedPrefillParams> sequences, std::span<const uint32_t> laneRows = {}) {
  constexpr uint32_t kDim = 256, kPairs = 32;
  constexpr uint16_t kFill = 0xa5a5;
  constexpr std::byte kPageFill{0xa5};
  const uint32_t kvHeads = layout.kvHeads, group = queryHeads / kvHeads;
  const bool int8 = layout.format == kv::Format::Int8;
  // Each sequence's rows of the command: laneRows', or its chunk's.
  std::vector<uint32_t> held;
  uint32_t rows = 0;
  uint64_t staged = 0;
  for (size_t s = 0; s < sequences.size(); ++s) {
    held.push_back(laneRows.empty() ? sequences[s].chunk_tokens : laneRows[s]);
    rows += held.back();
    staged += uint64_t{kvHeads} * sequences[s].chunk_stride;
  }
  const uint32_t packedWidth = 2 * queryHeads * kDim + 2 * kvHeads * kDim;
  auto packed = allocate(backend, uint64_t{rows} * packedWidth * 2);
  auto ropeCos = allocate(backend, uint64_t{rows} * kPairs * 4);
  auto ropeSin = allocate(backend, ropeCos.sizeBytes());
  auto queries = allocate(backend, staged * group * kDim * 2);
  std::fill_n(static_cast<uint16_t *>(queries.contents()), queries.sizeBytes() / 2, kFill);
  // Each sequence's pages, one after another, of a pool whose every byte
  // starts as kPageFill.
  uint32_t allPages = 0;
  for (const kv::ChunkedPrefillParams &sequence : sequences) allPages += sequence.page_table_entries;
  const auto spread = HostKvExtents::spread(allPages);
  kv::Layout poolLayout = layout;
  poolLayout.attentionLayers = 1;
  HostKvExtents pool(backend, poolLayout, spread.extentPages, spread.extents);
  for (uint32_t extent = 0; extent < pool.extentCount(); ++extent)
    std::fill(pool.bytes(extent).begin(), pool.bytes(extent).end(), kPageFill);
  std::vector<std::vector<uint32_t>> pageIds;
  std::vector<metal::MetalBuffer> tables;
  for (uint32_t s = 0, first = 0; s < sequences.size(); first += sequences[s++].page_table_entries) {
    pageIds.emplace_back(sequences[s].page_table_entries);
    std::iota(pageIds.back().begin(), pageIds.back().end(), first);
    tables.push_back(allocate(backend, uint64_t{sequences[s].page_table_entries} * sizeof(SplashKvPage)));
    pool.writeTable(pageIds.back(), tables.back());
  }
  uint32_t state = 0x2545F491U + queryHeads + rows;
  const auto unit = [&] {
    state = state * 1664525U + 1013904223U;
    return double(state >> 8) / double(1U << 23) - 1.0;
  };
  auto *packedData = static_cast<uint16_t *>(packed.contents());
  for (uint64_t i = 0; i < packed.sizeBytes() / 2; ++i) packedData[i] = floatToBf16(float(2 * unit()));
  for (uint64_t i = 0; i < ropeCos.sizeBytes() / 4; ++i) {
    const double angle = 3.14159265358979 * unit();
    static_cast<float *>(ropeCos.contents())[i] = float(std::cos(angle));
    static_cast<float *>(ropeSin.contents())[i] = float(std::sin(angle));
  }
  const auto weight = [&](uint32_t) { return float(1.0 + 0.3 * unit()); };
  const ops::NormWeights queryNorm = test::makeNormWeights(backend, kDim, float32, weight);
  const ops::NormWeights keyNorm = test::makeNormWeights(backend, kDim, float32, weight);
  // The sequences' rows, each after the ones before it.
  const metal::MetalBuffer layoutRows = allocate(backend, sequences.size() * sizeof(SplashLaneRows));
  for (uint32_t s = 0, first = 0; s < sequences.size(); first += held[s++])
    static_cast<SplashLaneRows *>(layoutRows.contents())[s] = {first, held[s]};
  const auto addProjection = [&](metal::CommandGraph &graph, const ops::NormWeights &keyWeights) {
    ops::PagedAttention::addProjection(graph, packed, queryNorm, keyWeights, ropeCos, ropeSin, queries,
                                       pool.layer(0), tables, sequences, {layoutRows, rows}, queryHeads, layout);
  };
  metal::CommandGraph graph;
  addProjection(graph, keyNorm);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();

  const auto *queryData = static_cast<const uint16_t *>(queries.contents());
  // The bytes of a slot's tensor element of a token's page: keys, values or,
  // INT8, their scales.
  const auto element = [&](size_t s, uint32_t token, uint32_t tensor, uint32_t head, uint32_t d) -> const std::byte * {
    const uint32_t id = pageIds[s][token / 32], slot = token % 32;
    const auto *page = pool.slab<std::byte>(0, tensor, id);
    if (tensor == SPLASH_KV_KEY_SCALES || tensor == SPLASH_KV_VALUE_SCALES)
      return page + splash_kv_scale_element(head, slot) * sizeof(float);
    const uint64_t index = tensor == SPLASH_KV_KEYS ? splash_kv_key_element(head, slot, d) : pool.valueElement(head, slot, d);
    return page + index * (int8 ? 1 : 2);
  };
  const auto stored = [&](size_t s, uint32_t token, uint32_t tensor, uint32_t head, uint32_t d) {
    const std::byte *at = element(s, token, tensor, head, d);
    if (int8) {
      const std::byte *scale = element(s, token, tensor == SPLASH_KV_KEYS ? SPLASH_KV_KEY_SCALES : SPLASH_KV_VALUE_SCALES,
                                       head, 0);
      float factor = 0;
      std::memcpy(&factor, scale, sizeof(factor));
      return double(int8_t(*at)) * factor;
    }
    uint16_t bits = 0;
    std::memcpy(&bits, at, sizeof(bits));
    return double(bf16ToFloat(bits));
  };
  uint32_t rowBegin = 0;
  uint64_t stageRow = 0;
  for (size_t index = 0; index < sequences.size(); ++index) {
    const kv::ChunkedPrefillParams &sequence = sequences[index];
    const uint32_t stride = sequence.chunk_stride;
    // Past the sequence's rows, within its bound, no row is stored or staged.
    for (uint32_t row = held[index]; row < sequence.chunk_tokens; ++row)
      for (uint32_t head = 0; head < kvHeads; ++head) {
        const uint32_t token = sequence.committed_tokens + row;
        for (uint32_t d = 0; d < kDim; ++d)
          require(*element(index, token, SPLASH_KV_KEYS, head, d) == kPageFill &&
                      *element(index, token, SPLASH_KV_VALUES, head, d) == kPageFill,
                  "the attention prepare stored a key or value past its sequence's rows");
        if (int8)
          require(*element(index, token, SPLASH_KV_KEY_SCALES, head, 0) == kPageFill &&
                      *element(index, token, SPLASH_KV_VALUE_SCALES, head, 0) == kPageFill,
                  "the attention prepare stored a scale past its sequence's rows");
        for (uint32_t local = 0; local < group; ++local)
          for (uint32_t d = 0; d < kDim; ++d)
            require(queryData[((stageRow + uint64_t{head} * stride + row) * group + local) * kDim + d] == kFill,
                    "the attention prepare staged a query past its sequence's rows");
      }
    for (uint32_t row = 0; row < held[index]; ++row) {
      const uint32_t token = sequence.committed_tokens + row;
      const uint64_t packedRow = uint64_t{rowBegin + row} * packedWidth;
      const uint64_t rope = uint64_t{rowBegin + row} * kPairs;
      for (uint32_t head = 0; head < queryHeads + kvHeads; ++head) {
        const bool query = head < queryHeads;
        const uint32_t h = query ? head : head - queryHeads;
        const uint16_t *source = packedData + packedRow + (query ? h * 2 * kDim : 2 * queryHeads * kDim + h * kDim);
        const std::vector<double> normalized = test::rmsNorm(source, query ? queryNorm : keyNorm, kDim);
        // The head's bf16 reference: the norm past the pairs, the fp64
        // rotation within them, and the bound each output keeps to it.
        std::vector<double> reference(kDim), bound(kDim);
        for (uint32_t d = 2 * kPairs; d < kDim; ++d) {
          reference[d] = normalized[d];
          bound[d] = ulpBf16(float(normalized[d]));
        }
        for (uint32_t d = 0; d < kPairs; ++d) {
          const double first = bf16ToFloat(floatToBf16(float(normalized[d])));
          const double second = bf16ToFloat(floatToBf16(float(normalized[d + kPairs])));
          const double c = static_cast<const float *>(ropeCos.contents())[rope + d];
          const double sine = static_cast<const float *>(ropeSin.contents())[rope + d];
          reference[d] = first * c - second * sine;
          reference[d + kPairs] = second * c + first * sine;
          const double larger = ulpBf16(float(std::max(std::fabs(first), std::fabs(second))));
          bound[d] = ulpBf16(float(reference[d])) + larger;
          bound[d + kPairs] = ulpBf16(float(reference[d + kPairs])) + larger;
        }
        if (query) {
          const uint16_t *out =
              queryData + (((stageRow + uint64_t{h / group} * stride + row) * group + h % group) * kDim);
          for (uint32_t d = 2 * kPairs; d < kDim; ++d)
            require(test::roundedOnceToBf16(out[d], normalized[d]),
                    "attention prepare norm differs from the fp64 reference");
          for (uint32_t d = 0; d < 2 * kPairs; ++d)
            require(std::fabs(bf16ToFloat(out[d]) - reference[d]) <= bound[d],
                    "attention prepare rotation differs from the fp64 reference");
          continue;
        }
        // A key and a value of the row and head, as the page stores them.
        double keyScale = 0, valueScale = 0, keyMaximum = 0, valueMaximum = 0;
        for (uint32_t d = 0; d < kDim; ++d) {
          keyMaximum = std::max(keyMaximum, std::fabs(reference[d]) + bound[d]);
          valueMaximum = std::max(valueMaximum, double(std::fabs(bf16ToFloat(source[kvHeads * kDim + d]))));
        }
        if (int8) {
          float factor = 0;
          std::memcpy(&factor, element(index, token, SPLASH_KV_KEY_SCALES, h, 0), sizeof(factor));
          keyScale = factor;
          std::memcpy(&factor, element(index, token, SPLASH_KV_VALUE_SCALES, h, 0), sizeof(factor));
          valueScale = factor;
          require(keyScale <= keyMaximum / 127 * (1 + 1e-6) && keyScale > 0,
                  "attention prepare key scale exceeds the largest key over 127");
          require(std::fabs(valueScale - valueMaximum / 127) <= ulpBf16(float(valueMaximum / 127)) / 256,
                  "attention prepare value scale is not the largest value over 127");
        }
        for (uint32_t d = 0; d < kDim; ++d) {
          const double key = stored(index, token, SPLASH_KV_KEYS, h, d);
          if (d >= 2 * kPairs && !int8) {
            uint16_t bits = 0;
            std::memcpy(&bits, element(index, token, SPLASH_KV_KEYS, h, d), sizeof(bits));
            require(test::roundedOnceToBf16(bits, normalized[d]), "attention prepare key norm differs from the fp64 reference");
          } else {
            require(std::fabs(key - reference[d]) <= bound[d] + 0.51 * keyScale,
                    "attention prepare key differs from the fp64 reference");
          }
          const uint16_t valueBits = source[kvHeads * kDim + d];
          if (int8) {
            const double expected = valueScale ? double(bf16ToFloat(valueBits)) / valueScale : 0;
            require(std::fabs(stored(index, token, SPLASH_KV_VALUES, h, d) / (valueScale ? valueScale : 1) -
                              std::nearbyint(expected)) <= 1,
                    "attention prepare value code is not the value's over its scale");
          } else {
            uint16_t bits = 0;
            std::memcpy(&bits, element(index, token, SPLASH_KV_VALUES, h, d), sizeof(bits));
            require(bits == valueBits, "attention prepare value copy differs");
          }
        }
      }
    }
    rowBegin += held[index];
    stageRow += uint64_t{kvHeads} * stride;
  }
  // One dispatch normalizes the queries and the keys, so it takes one norm type.
  if (float32) {
    metal::CommandGraph rejected;
    try {
      addProjection(rejected, {keyNorm.buffer, false});
      throw std::runtime_error("mixed q/k norm types were accepted");
    } catch (const std::invalid_argument &error) {
      require(std::string_view(error.what()) == "query and key norms differ in type",
              "mixed q/k norm types rejected for the wrong reason");
    }
  }
  std::cout << "attention prepare: q=" << queryHeads << " " << kv::formatName(layout.format)
            << " sequences=" << sequences.size() << " rows=" << rows << (float32 ? " f32" : " bf16") << " norms PASS\n";
}

// Each buffer the projection, gate, store and attention reach, at its extent
// and one element short, of a command of two sequences: 37 tokens after 100
// in 64 rows per KV head, and a lane's 8 after 2000 in 32. The packed rows
// (the gates read each row's query heads only), the RoPE rows, the staging up
// to the last sequence's last KV head's rows (its rows, or the attention's
// whole tiles of 8), the page table of each sequence's visible tokens, which
// the projection stores the rows' K/V through and the attention reads, and
// the plan's partials and statistics.
void bufferExtents(metal::MetalBackend &backend, uint32_t queryHeads, kv::Layout layout) {
  constexpr uint32_t dimension = 256;
  const std::array sequences{ops::PagedAttention::chunkParams(100, 37, 64, 5),
                             ops::PagedAttention::chunkParams(2000, 8, 32, 64)};
  constexpr uint32_t rows = 45;
  const uint32_t kvHeads = layout.kvHeads, group = queryHeads / kvHeads;
  const uint64_t packedWidth = uint64_t{2 * queryHeads + 2 * kvHeads} * dimension;
  const uint64_t queryWidth = uint64_t{group} * dimension, lastStage = uint64_t{kvHeads} * 64;
  // The staging up to the last sequence's last KV head's first `reach` rows
  // of `width`.
  const auto staged = [&](uint32_t reach, uint64_t width) { return (lastStage + (kvHeads - 1) * 32 + reach) * width * 2; };
  const auto gateRows = [&](uint64_t count) { return ((count - 1) * packedWidth + uint64_t{queryHeads} * 2 * dimension) * 2; };
  const auto pageTable = [](const kv::ChunkedPrefillParams &chunk) {
    return uint64_t{(chunk.committed_tokens + chunk.chunk_tokens + 31) / 32} * sizeof(SplashKvPage);
  };
  // F32 norms, a GGUF's, which every geometry's projection kernels take.
  const ops::NormWeights queryNorm{allocate(backend, dimension * 4), true}, keyNorm{allocate(backend, dimension * 4), true};
  using Extents = std::initializer_list<test::BufferExtent>;
  using Buffers = std::vector<metal::MetalBuffer>;
  const metal::MetalBuffer packed = allocate(backend, rows * packedWidth * 2), rope = allocate(backend, rows * 32 * 4),
                           staging = allocate(backend, staged(32, queryWidth));
  const std::array tables{allocate(backend, pageTable(sequences[0])), allocate(backend, pageTable(sequences[1]))};
  // The sequences' rows of the command: 37 from row 0, 8 from row 37.
  const auto layoutOf = [&](std::initializer_list<SplashLaneRows> entries) {
    const metal::MetalBuffer buffer = allocate(backend, entries.size() * sizeof(SplashLaneRows));
    std::copy(entries.begin(), entries.end(), static_cast<SplashLaneRows *>(buffer.contents()));
    return buffer;
  };
  const metal::MetalBuffer laneRows = layoutOf({{0, 37}, {37, 8}});
  const ops::AttentionRows command{laneRows, rows};
  constexpr uint64_t layoutBytes = 2 * sizeof(SplashLaneRows);
  for (const bool key : {false, true})
    test::requireExtent(backend, (key ? keyNorm : queryNorm).buffer, dimension * 4, 4, "norm weight",
                        [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                          const ops::NormWeights query{key ? queryNorm.buffer : view, true},
                              keys{key ? view : keyNorm.buffer, true};
                          ops::PagedAttention::addProjection(graph, packed, query, keys, rope, rope, staging,
                                                             {}, tables, sequences, command, queryHeads, layout);
                        });
  test::requireExtents(backend,
                       Extents{{0, rows * packedWidth * 2, 2, "attention q/k/v"},
                               {1, uint64_t{rows} * 32 * 4, 4, "attention RoPE cosine"},
                               {2, uint64_t{rows} * 32 * 4, 4, "attention RoPE sine"},
                               {3, staged(8, queryWidth), 2, "attention query"},
                               {4, pageTable(sequences[0]), sizeof(SplashKvPage), "page table"},
                               {5, pageTable(sequences[1]), sizeof(SplashKvPage), "page table"},
                               {6, layoutBytes, sizeof(SplashLaneRows), "attention layout"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         ops::PagedAttention::addProjection(graph, b[0], queryNorm, keyNorm, b[1], b[2], b[3],
                                                            {}, std::span(b).subspan(4, 2), sequences, {b[6], rows},
                                                            queryHeads, layout);
                       });
  const uint32_t width = queryHeads * dimension;
  // The attention: its staged queries, its workspace, its sequences' page
  // tables and layout, and its output: the packed rows' gates and the
  // out-projection's rows, every row's or each sequence's last's, and table.
  const auto plan = planOf(sequences, queryHeads, layout);
  const auto histories = [](const metal::MetalBuffer &first, const metal::MetalBuffer &second) {
    return std::array{ops::AttentionHistory{{}, first}, ops::AttentionHistory{{}, second}};
  };
  test::requireExtents(backend,
                       Extents{{0, staged(8, queryWidth), 2, "attention query"},
                               {1, plan.workspace().partialsBytes, 4, "attention partials"},
                               {2, plan.workspace().statisticsBytes, 4, "attention statistics"},
                               {3, pageTable(sequences[0]), sizeof(SplashKvPage), "page table"},
                               {4, pageTable(sequences[1]), sizeof(SplashKvPage), "page table"},
                               {5, layoutBytes, sizeof(SplashLaneRows), "attention layout"},
                               {6, gateRows(rows), 2, "attention gate"},
                               {7, uint64_t{rows} * width * 2, 2, "attention hidden"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         (void)ops::PagedAttention::addAttention(graph, plan, b[0], b[1], b[2], sequences,
                                                                 histories(b[3], b[4]), {b[5], rows}, {b[6], b[7]});
                       });
  // Each sequence's last row alone, of a plan of their last tiles, as rows
  // 0 and 1.
  const auto tails = ops::PagedAttention::lastTiles(plan);
  const metal::MetalBuffer partials = allocate(backend, plan.workspace().partialsBytes),
                           statistics = allocate(backend, plan.workspace().statisticsBytes),
                           firstTable = allocate(backend, pageTable(sequences[0])),
                           secondTable = allocate(backend, pageTable(sequences[1]));
  test::requireExtents(backend,
                       Extents{{0, gateRows(rows), 2, "attention gate"}, {1, uint64_t{2} * width * 2, 2, "attention hidden"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         (void)ops::PagedAttention::addAttention(
                             graph, tails, staging, partials, statistics, sequences,
                             histories(firstTable, secondTable), command, {b[0], b[1], ops::AttentionOutput::Last});
                       });
  rejects([&] {
    metal::CommandGraph graph;
    (void)ops::PagedAttention::addAttention(graph, plan, staging, partials, statistics, sequences,
                                            histories(firstTable, secondTable), command,
                                            {packed, staging, ops::AttentionOutput::Last});
  }, "an attention of each sequence's last row takes the plan of its last tiles",
          "every tile's plan wrote each sequence's last row alone");
  // A decode step's three lanes, writing the out-projection's table too.
  const std::array lanes{ops::PagedAttention::chunkParams(0, 8, 32, 1), ops::PagedAttention::chunkParams(9, 8, 32, 1),
                         ops::PagedAttention::chunkParams(31, 8, 32, 2)};
  const auto lanePlan = planOf(lanes, queryHeads, layout);
  const metal::MetalBuffer laneTable = allocate(backend, pageTable(lanes[2])),
                           lanePartials = allocate(backend, lanePlan.workspace().partialsBytes),
                           laneStatistics = allocate(backend, lanePlan.workspace().statisticsBytes);
  test::requireExtents(backend,
                       Extents{{0, gateRows(24), 2, "attention gate"},
                               {1, uint64_t{24} * width * 2, 2, "attention hidden"},
                               {2, ops::tableBytes(width, 24), 2, "linear table"},
                               {3, ops::tableSumsBytes(ops::LinearInput::Table16, width, 24), 4, "linear table sums"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         const std::array laneHistories{ops::AttentionHistory{{}, laneTable},
                                                        ops::AttentionHistory{{}, laneTable},
                                                        ops::AttentionHistory{{}, laneTable}};
                         (void)ops::PagedAttention::addAttention(
                             graph, lanePlan, staging, lanePartials, laneStatistics, lanes, laneHistories,
                             {layoutOf({{0, 8}, {8, 8}, {16, 8}}), 24},
                             {b[0], b[1], ops::AttentionOutput::Every, {b[2], b[3], {}, {}}, ops::LinearInput::Table16});
                       });
  // A last tile of whole rows reaches as far as its sequence's whole tiles.
  const std::array chunk{ops::PagedAttention::chunkParams(100, 37, 64, 5)};
  const auto tail = ops::PagedAttention::lastTiles(planOf(chunk, queryHeads, layout));
  test::requireExtents(backend,
                       Extents{{0, ((kvHeads - 1) * 64 + 40) * queryWidth * 2, 2, "attention query"},
                               {1, tail.workspace().partialsBytes, 4, "attention partials"}},
                       [&](metal::CommandGraph &graph, const Buffers &b) {
                         const std::array chunkHistories{ops::AttentionHistory{{}, allocate(backend, 64)}};
                         (void)ops::PagedAttention::addAttention(
                             graph, tail, b[0], b[1], allocate(backend, tail.workspace().statisticsBytes), chunk,
                             chunkHistories, {layoutOf({{0, 37}}), 37},
                             {allocate(backend, gateRows(37)), allocate(backend, uint64_t{width} * 2),
                              ops::AttentionOutput::Last});
                       });
}

// The stage of a layer's pages (ops::PagedAttention::addStageLayer): a
// threadgroup a page, the pool's table and the staging's bound before the
// parameters, which place both layers and hold the first page and a page's
// bytes of each tensor of the layer; both tables hold the staged pages'
// entries, and a stage of no pages is refused.
void stageEncoding(metal::MetalBackend &backend) {
  constexpr uint32_t firstPage = 5, pages = 7;
  constexpr SplashKvLayer pool{256, 1u << 20}, staged{256, 0};
  for (const kv::Format format : {kv::Format::Int8, kv::Format::BFloat16}) {
    const kv::Layout layout{.attentionLayers = 12, .kvHeads = 2, .headDimension = 256, .format = format,
                            .valuesTokenMajor = true};
    const metal::MetalBuffer poolTable = test::sharedBuffer(backend, (firstPage + pages) * sizeof(SplashKvPage)),
                             stagedTable = test::sharedBuffer(backend, (firstPage + pages) * sizeof(SplashKvPage));
    metal::CommandGraph graph;
    ops::PagedAttention::addStageLayer(graph, pool, poolTable, staged, stagedTable, firstPage, pages, layout);
    require(graph.dispatches().size() == 1, "a stage encoded other than one dispatch");
    const metal::ComputeDispatch &stage = graph.dispatches()[0];
    require(stage.pipelineName == "paged_kv_stage_layer" && stage.threadgroups.x == pages &&
                stage.threadgroups.y == 1 && stage.threadgroups.z == 1 && stage.buffers.size() == 2 &&
                stage.buffers[0].buffer.sameView(poolTable) && stage.buffers[1].buffer.sameView(stagedTable) &&
                stage.bytes.size() == 1 && stage.bytes[0].index == 2 &&
                stage.bytes[0].sizeBytes == sizeof(SplashKvStageParams),
            "a stage's dispatch does not take a threadgroup a page and both tables before its parameters");
    SplashKvStageParams params;
    std::memcpy(&params, stage.bytes[0].data, sizeof(params));
    require(params.pool.extent_pages == pool.extent_pages && params.pool.offset == pool.offset &&
                params.staged.extent_pages == staged.extent_pages && params.staged.offset == staged.offset &&
                params.first_page == firstPage && params.data_bytes == layout.dataBytesPerLayerPage() &&
                params.scale_bytes == layout.scaleBytesPerLayerPage(),
            "a stage's parameters do not place its layers, its first page and its tensors' bytes");
    const uint64_t tableBytes = (firstPage + pages) * sizeof(SplashKvPage);
    test::requireExtents(backend,
                         std::initializer_list<test::BufferExtent>{
                             {0, tableBytes, sizeof(SplashKvPage), "KV stage pool table"},
                             {1, tableBytes, sizeof(SplashKvPage), "KV stage staged table"}},
                         [&](metal::CommandGraph &encoded, const std::vector<metal::MetalBuffer> &b) {
                           ops::PagedAttention::addStageLayer(encoded, pool, b[0], staged, b[1], firstPage, pages,
                                                              layout);
                         });
  }
  metal::CommandGraph refused;
  const metal::MetalBuffer table = test::sharedBuffer(backend, 64);
  rejects([&] { ops::PagedAttention::addStageLayer(refused, pool, table, staged, table, 0, 0, {}); },
          "invalid KV stage geometry", "a stage of no layout was accepted");
  rejects(
      [&] {
        ops::PagedAttention::addStageLayer(refused, pool, table, staged, table, 0, 0,
                                           {.attentionLayers = 1, .kvHeads = 2, .headDimension = 256});
      },
      "invalid KV stage geometry", "a stage of no pages was accepted");
  require(refused.empty(), "a refused stage encoded a dispatch");
}

// Block lists (ops::AttentionBlocks): a layout of token-major values runs
// the sparse kernels, compiled for the group of twelve over two KV heads
// alone, and one of dimension-major values the page loop, which has no group
// of twelve. A sparse split binds the lists and their counts after its
// page tables and its reduce the counts, before their parameters, which hold
// each sequence's lists' stride and its first list; without lists both bind
// another buffer of the dispatch there, and a sequence without lists beside
// one with them takes none. The lists and counts hold every list the
// sequences' tiles read, and a plan whose kernels take no lists refuses
// them. With locations the located split runs, which binds them and their
// pages after the row masks; they hold a location for every entry of the
// sequences' lists, and need their pages. A decode step's tree rows bind
// their row masks after the counts, a mask per row, which lists need.
void attentionBlocks(metal::MetalBackend &backend) {
  constexpr uint32_t entries = 40, first = 2;
  const std::array sequences{ops::PagedAttention::chunkParams(100, 37, 64, 5),
                             ops::PagedAttention::chunkParams(2000, 8, 32, 64),
                             ops::PagedAttention::chunkParams(0, 8, 32, 1)};
  constexpr uint32_t tiles = 5 + 1 + 1;
  const kv::Layout dense{1, 2, 256, kv::Format::Int8};
  kv::Layout tokenMajor = dense;
  tokenMajor.valuesTokenMajor = true;
  tokenMajor.reads = kv::Reads::SelectedBlocks;
  rejects([&] { (void)planOf(sequences, 24, dense); }, "no paged-attention kernel for layout",
          "the sparse kernels took dimension-major values");
  rejects([&] { (void)planOf(sequences, 16, tokenMajor); }, "no sparse attention kernel for layout",
          "the page loop took token-major values");
  const auto buffer = [&](uint64_t bytes) { return test::sharedBuffer(backend, bytes); };
  // Staging for every sequence's rows of every head of the widest group.
  const metal::MetalBuffer staging = buffer(uint64_t{dense.kvHeads} * (64 + 32 + 32) * 12 * 256 * 2),
                           table = buffer(64 * sizeof(SplashKvPage)), laneRows = buffer(3 * sizeof(SplashLaneRows));
  // The out-projection's rows and the packed rows' gates of the widest
  // geometry's 53 rows.
  const metal::MetalBuffer hidden = buffer(uint64_t{53} * 24 * 256 * 2),
                           packed = buffer(uint64_t{53} * (2 * 24 + 2 * 2) * 256 * 2);
  for (uint32_t s = 0, row = 0; s < sequences.size(); row += sequences[s++].chunk_tokens)
    static_cast<SplashLaneRows *>(laneRows.contents())[s] = {row, sequences[s].chunk_tokens};
  const uint32_t lists = first + tiles;
  const metal::MetalBuffer listBuffer = buffer(uint64_t{lists} * entries * 4), counts = buffer(lists * 4);
  // Locations of every list the dispatches read, and a page table for them.
  const metal::MetalBuffer locations = buffer(uint64_t{lists} * entries * 4), locationPages = buffer(64);
  for (const uint32_t queryHeads : {16U, 24U}) {
    const bool sparse = queryHeads == 24;
    const kv::Layout layout = sparse ? tokenMajor : dense;
    const auto plan = planOf(sequences, queryHeads, layout);
    require(plan.sparse == sparse, "a plan's kernels do not take block lists as its group size states");
    const metal::MetalBuffer partials = buffer(plan.workspace().partialsBytes),
                             statistics = buffer(plan.workspace().statisticsBytes);
    // Sequence s's lists from blocks.list + the tiles of those before it, or
    // with `alone` none for the second.
    const auto attend = [&](metal::CommandGraph &graph, ops::AttentionBlocks blocks, bool alone = false) {
      std::array<ops::AttentionHistory, 3> histories{};
      uint32_t list = blocks.list;
      for (size_t s = 0; s < sequences.size(); ++s) {
        histories[s] = {{1, 0}, table, blocks};
        histories[s].blocks.list = list;
        if (alone && s == 1) histories[s].blocks = {};
        list += kv::attentionTiles(sequences[s].chunk_tokens);
      }
      (void)ops::PagedAttention::addAttention(graph, plan, staging, partials, statistics, sequences, histories,
                                              {laneRows, 37 + 8 + 8}, {packed, hidden});
    };
    if (!sparse) {
      metal::CommandGraph refused;
      rejects([&] { attend(refused, {listBuffer, counts, entries, first}); },
              "the attention kernels take no block lists", "a plan of the page loop took block lists");
      require(refused.empty(), "refused block lists encoded a dispatch");
      continue;
    }
    for (const uint32_t listStride : {entries, 0U}) {
      const metal::MetalBuffer &boundLists = listStride ? listBuffer : partials;
      const metal::MetalBuffer &boundCounts = listStride ? counts : partials;
      metal::CommandGraph graph, partly;
      attend(graph, {listBuffer, counts, listStride, first});
      attend(partly, {listBuffer, counts, listStride, first}, true);
      // `after` buffers follow the lists, their counts and the row masks'
      // place, which follow the tree rows' place (another buffer for each of
      // the two without tree rows) and the layout; the reduce binds the
      // counts after its output's buffers (the packed rows, the rows, and the
      // rows again for the table and its sums it does not write) and the
      // layout.
      const auto bindsLists = [&](const metal::ComputeDispatch &split, const metal::ComputeDispatch &reduce,
                                  uint32_t after) {
        return split.buffers.size() == 9 + 3 + after && split.buffers[7].buffer.sameView(partials) &&
               split.buffers[8].buffer.sameView(laneRows) && split.buffers[9].buffer.sameView(boundLists) &&
               split.buffers[10].buffer.sameView(boundCounts) && split.buffers[11].buffer.sameView(partials) &&
               split.bytes.size() == 1 && split.bytes[0].index == 12 + after && reduce.buffers.size() == 8 &&
               reduce.buffers[2].buffer.sameView(packed) && reduce.buffers[3].buffer.sameView(hidden) &&
               reduce.buffers[4].buffer.sameView(hidden) && reduce.buffers[5].buffer.sameView(hidden) &&
               reduce.buffers[6].buffer.sameView(laneRows) && reduce.buffers[7].buffer.sameView(boundCounts) &&
               reduce.bytes.size() == 1 && reduce.bytes[0].index == 8;
      };
      require(bindsLists(graph.dispatches()[0], graph.dispatches()[1], 0) &&
                  bindsLists(partly.dispatches()[0], partly.dispatches()[1], 0) &&
                  graph.dispatches()[0].pipelineName == plan.splitPipeline(),
              "a sparse kernel's block lists are not bound before its parameters");
      kv::AttentionParams params, partlyParams;
      std::memcpy(&params, graph.dispatches()[0].bytes[0].data, sizeof(params));
      std::memcpy(&partlyParams, partly.dispatches()[0].bytes[0].data, sizeof(partlyParams));
      bool listed = true;
      for (uint32_t s = 0, list = first; s < sequences.size(); list += kv::attentionTiles(sequences[s++].chunk_tokens)) {
        listed &= params.sequence[s].block_stride == listStride &&
                  params.sequence[s].block_list == (listStride ? list : 0);
        listed &= s == 1 ? partlyParams.sequence[s].block_stride == 0
                         : partlyParams.sequence[s].block_stride == listStride;
      }
      require(listed, "a split's parameters do not hold its sequences' block lists");
      metal::CommandGraph refused;
      if (listStride) {
        metal::CommandGraph located;
        attend(located, {listBuffer, counts, listStride, first, locations, locationPages});
        const metal::ComputeDispatch &split = located.dispatches()[0];
        require(bindsLists(split, located.dispatches()[1], 2) && split.buffers[12].buffer.sameView(locations) &&
                    split.buffers[13].buffer.sameView(locationPages) && split.pipelineName == locatedSplitName(layout),
                "a located split does not bind its lists' locations and their pages before its parameters");
        rejects([&] { attend(refused, {listBuffer, counts, listStride, first, locations, {}}); },
                "attention block locations need their pages", "block locations without pages were taken");
        rejects([&] { attend(refused, {listBuffer, counts, listStride, first, locations, locationPages}, true); },
                "a located attention dispatch's sequences all read block lists",
                "a located dispatch took a sequence without lists");
      }
      require(refused.empty(), "refused block locations encoded a dispatch");
    }
    // A decode step's tree rows (lanes of 8 rows) bind their row masks after
    // the lists' counts, which hold a mask per row of the command, or
    // without lists another buffer of the dispatch.
    const std::array treeLanes{sequences[1], sequences[2]};
    const auto treePlan = planOf(treeLanes, queryHeads, layout);
    const metal::MetalBuffer treeRows = buffer(16 * sizeof(SplashTreeRow)), rowMasks = buffer(16 * 4),
                             treeLayout = buffer(2 * sizeof(SplashLaneRows));
    static_cast<SplashLaneRows *>(treeLayout.contents())[0] = {0, 8};
    static_cast<SplashLaneRows *>(treeLayout.contents())[1] = {8, 8};
    const auto attendTree = [&](metal::CommandGraph &graph, ops::AttentionBlocks blocks) {
      std::array<ops::AttentionHistory, 2> histories{};
      for (uint32_t s = 0; s < 2; ++s) {
        histories[s] = {{1, 0}, table, blocks};
        histories[s].blocks.list += s;
      }
      (void)ops::PagedAttention::addAttention(graph, treePlan, staging, partials, statistics, treeLanes, histories,
                                              {treeLayout, 16, treeRows}, {packed, hidden});
    };
    metal::CommandGraph masked, listless, unmasked;
    attendTree(masked, {listBuffer, counts, entries, first, {}, {}, rowMasks});
    attendTree(listless, {listBuffer, counts, 0, 0, {}, {}, rowMasks});
    require(masked.dispatches()[0].buffers[7].buffer.sameView(treeRows) &&
                masked.dispatches()[0].buffers[8].buffer.sameView(treeLayout) &&
                masked.dispatches()[0].buffers[11].buffer.sameView(rowMasks) &&
                listless.dispatches()[0].buffers[11].buffer.sameView(partials),
            "a sparse split of tree rows does not bind their row masks after its lists' counts");
    rejects([&] { attendTree(unmasked, {listBuffer, counts, entries, first}); },
            "attention row masks buffer holds 0 bytes, needs 64", "tree rows took lists without row masks");
    require(unmasked.empty(), "tree rows' lists without row masks encoded a dispatch");
    test::requireExtent(backend, listBuffer, uint64_t{first + tiles} * entries * 4, 4, "attention block lists",
                        [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                          attend(graph, {view, counts, entries, first});
                        });
    test::requireExtent(backend, counts, uint64_t{first + tiles} * 4, 4, "attention block list counts",
                        [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                          attend(graph, {listBuffer, view, entries, first});
                        });
    test::requireExtent(backend, locations, uint64_t{first + tiles} * entries * 4, 4, "attention block locations",
                        [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                          attend(graph, {listBuffer, counts, entries, first, view, locationPages});
                        });
  }
}

// A tile's block list (addBlockLists) is, in block order, every block of the
// tile's history that one of its rows' masks names, with the bits of those
// rows; mask bits past the tile's history and words past its last block stay
// unlisted. One dispatch lists the tiles of several sequences, each from its
// own mask row, history and list, in tiles of eight rows, the last partial,
// up to the end of the 1,010,000 tokens YaRN serves; a list keeps to its
// stride, and the dispatch refuses a mask shorter than its sequences' rows.
// A sequence of fewer rows than its bound, a decode lane's under the step's
// lane cap, lists its rows' tile and gives its bound's tiles past them empty
// lists.
void blockListsMatchMasks(metal::MetalBackend &backend) {
  constexpr uint32_t rowBits = SPLASH_ATTENTION_BLOCK_ROW_BITS, blockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  constexpr uint32_t tileRows = kv::kAttentionTileRows;
  // The dispatch's lists start past list `first`.
  constexpr uint32_t first = 2;
  // A sequence's first row of the command (its mask row), rows and first
  // position.
  struct Rows final {
    uint32_t maskRow;
    uint32_t rows;
    uint32_t firstPosition;
  };
  struct Case final {
    uint32_t stride;
    std::vector<Rows> rows;
  };
  const auto buffer = [&](uint64_t bytes) { return test::sharedBuffer(backend, bytes); };
  for (Case c : {Case{4096, {{3, 8, 4095}, {11, 37, 100}, {48, 8, 4096}}}, Case{4096, {{0, 70, 9000}, {70, 5, 0}}},
                 Case{5, {{0, 8, 4096}}}, Case{4104, {{0, 8, 1'009'992}}}}) {
    uint32_t maskRows = 0, words = 0, lists = 0;
    std::vector<ops::BlockListRows> sequences;
    const metal::MetalBuffer laneRows = buffer(c.rows.size() * sizeof(SplashLaneRows));
    for (uint32_t s = 0; s < c.rows.size(); ++s) {
      const Rows &rows = c.rows[s];
      maskRows = std::max(maskRows, rows.maskRow + rows.rows);
      words = std::max(words, (rows.firstPosition + rows.rows) / blockTokens / 32 + 3);
      static_cast<SplashLaneRows *>(laneRows.contents())[s] = {rows.maskRow, rows.rows};
      sequences.push_back({s, rows.rows, rows.firstPosition, lists});
      lists += (rows.rows + tileRows - 1) / tileRows;
    }
    const ops::AttentionRows command{laneRows, maskRows};
    // Each row names about one block in three of all the words' blocks, its
    // own and the next included.
    const metal::MetalBuffer mask = buffer(uint64_t{maskRows} * words * 4);
    auto *bits = static_cast<uint32_t *>(mask.contents());
    for (uint64_t word = 0; word < uint64_t{maskRows} * words; ++word)
      bits[word] = uint32_t((word * 0x9e3779b97f4a7c15ULL) >> 32) & uint32_t(word * 0x2545f491ULL >> 3);
    for (const Rows &rows : c.rows)
      for (uint32_t row = 0; row < rows.rows; ++row)
        for (const uint32_t block : {(rows.firstPosition + row) / blockTokens, (rows.firstPosition + row + 1) / blockTokens})
          bits[uint64_t{rows.maskRow + row} * words + block / 32] |= 1U << (block % 32);
    const metal::MetalBuffer listBuffer = buffer(uint64_t{first + lists + 1} * c.stride * 4),
                             counts = buffer(uint64_t{first + lists + 1} * 4);
    std::memset(listBuffer.contents(), 0xff, listBuffer.sizeBytes());
    std::memset(counts.contents(), 0xff, counts.sizeBytes());
    const ops::AttentionBlocks blocks{listBuffer, counts, c.stride, first};
    metal::CommandGraph graph;
    ops::PagedAttention::addBlockLists(graph, mask, words, sequences, command, blocks);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *listed = static_cast<const uint32_t *>(listBuffer.contents());
    const auto *listCounts = static_cast<const uint32_t *>(counts.contents());
    for (uint32_t s = 0; s < c.rows.size(); ++s) {
      const ops::BlockListRows &sequence = sequences[s];
      for (uint32_t row0 = 0; row0 < sequence.rows; row0 += tileRows) {
        const uint32_t rows = std::min(tileRows, sequence.rows - row0);
        const uint32_t history = (sequence.firstPosition + row0 + rows + blockTokens - 1) / blockTokens;
        std::vector<uint32_t> expected;
        for (uint32_t block = 0; block < history; ++block) {
          uint32_t entry = block << rowBits;
          for (uint32_t row = 0; row < rows; ++row)
            if (bits[uint64_t{c.rows[s].maskRow + row0 + row} * words + block / 32] >> (block % 32) & 1U)
              entry |= 1U << row;
          if (entry & ((1U << rowBits) - 1U)) expected.push_back(entry);
        }
        const uint32_t index = first + sequence.list + row0 / tileRows;
        const uint32_t kept = std::min<uint32_t>(expected.size(), c.stride);
        const uint32_t *list = listed + uint64_t{index} * c.stride;
        require(listCounts[index] == kept && std::equal(expected.begin(), expected.begin() + kept, list) &&
                    std::all_of(list + kept, list + c.stride, [](uint32_t entry) { return entry == ~0U; }),
                "a tile's block list differs from its rows' masks");
      }
    }
    require(listCounts[first - 1] == ~0U && listCounts[first + lists] == ~0U &&
                std::all_of(listed + uint64_t{first + lists} * c.stride,
                            listed + uint64_t{first + lists + 1} * c.stride,
                            [](uint32_t entry) { return entry == ~0U; }),
            "block lists were written past the dispatch's lists");
    test::requireExtent(backend, mask, uint64_t{maskRows} * words * 4, 4, "attention block mask",
                        [&](metal::CommandGraph &encoded, const metal::MetalBuffer &view) {
                          ops::PagedAttention::addBlockLists(encoded, view, words, sequences, command, blocks);
                        });
    test::requireExtent(backend, laneRows, c.rows.size() * sizeof(SplashLaneRows), sizeof(SplashLaneRows),
                        "attention layout", [&](metal::CommandGraph &encoded, const metal::MetalBuffer &view) {
                          ops::PagedAttention::addBlockLists(encoded, mask, words, sequences,
                                                             {view, command.rows}, blocks);
                        });
  }
  {
    // Five rows from 40 of a bound of sixteen, row r naming block r: the
    // second tile's count goes from no list's to zero, and its entries stay
    // unwritten.
    constexpr uint32_t stride = 64, words = 1;
    const metal::MetalBuffer laneRows = buffer(sizeof(SplashLaneRows)), mask = buffer(uint64_t{5} * words * 4),
                             listBuffer = buffer(uint64_t{2} * stride * 4), counts = buffer(2 * 4);
    *static_cast<SplashLaneRows *>(laneRows.contents()) = {0, 5};
    std::memset(mask.contents(), 0, mask.sizeBytes());
    for (uint32_t row = 0; row < 5; ++row) static_cast<uint32_t *>(mask.contents())[row * words] = 1U << row;
    std::memset(listBuffer.contents(), 0xff, listBuffer.sizeBytes());
    std::memset(counts.contents(), 0xff, counts.sizeBytes());
    const std::vector<ops::BlockListRows> shorter{{0, 16, 40, 0}};
    metal::CommandGraph graph;
    ops::PagedAttention::addBlockLists(graph, mask, words, shorter, {laneRows, 5}, {listBuffer, counts, stride, 0});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *listed = static_cast<const uint32_t *>(listBuffer.contents());
    const auto *listCounts = static_cast<const uint32_t *>(counts.contents());
    require(listCounts[0] == 5 && listCounts[1] == 0 &&
                std::all_of(listed + stride, listed + 2 * stride, [](uint32_t entry) { return entry == ~0U; }),
            "a sequence's tile of its bound past its rows did not list nothing");
  }
  const metal::MetalBuffer any = buffer(4096);
  const ops::BlockListRows rows{0, 8, 0, 0};
  const std::vector<ops::BlockListRows> one{rows}, none, empty{{0, 0, 0, 0}},
      tooMany(SPLASH_MAXIMUM_BATCH_WIDTH + 1, rows);
  metal::CommandGraph invalid;
  for (const auto &[sequences, words, stride] :
       {std::tuple{&none, 1U, 8U}, std::tuple{&tooMany, 1U, 8U}, std::tuple{&empty, 1U, 8U}, std::tuple{&one, 0U, 8U},
        std::tuple{&one, 1U, 0U}})
    rejects([&] { ops::PagedAttention::addBlockLists(invalid, any, words, *sequences, {any, 8}, {any, any, stride, 0}); },
            "invalid attention block lists", "invalid block lists were accepted");
  // Rows at 252 on reach 65 blocks, past one word's 32.
  const std::vector<ops::BlockListRows> wide{{0, 8, 252, 0}};
  rejects([&] { ops::PagedAttention::addBlockLists(invalid, any, 1, wide, {any, 8}, {any, any, 8, 0}); },
          "attention block masks narrower than their rows' history",
          "block masks narrower than their rows' history were accepted");
  require(invalid.empty(), "invalid block lists encoded a dispatch");
}

// Each buffer the RoPE tables reach, at its extent and one element short: the
// target rows' three positions and the draft rows' one, each kind's inverse
// frequencies, and its rows of cosines and sines.
void ropeExtents(metal::MetalBackend &backend) {
  constexpr RopeTableParams rows{37, 24, 1.0F};
  test::requireExtents(backend,
                       std::initializer_list<test::BufferExtent>{
                           {0, rows.target_rows * 3 * 4, 4, "target RoPE position"},
                           {1, rows.draft_rows * 4, 4, "draft RoPE position"},
                           {2, SPLASH_TARGET_ROPE_PAIRS * 4, 4, "target inverse frequency"},
                           {3, SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft inverse frequency"},
                           {4, rows.target_rows * SPLASH_TARGET_ROPE_PAIRS * 4, 4, "target RoPE cosine"},
                           {5, rows.target_rows * SPLASH_TARGET_ROPE_PAIRS * 4, 4, "target RoPE sine"},
                           {6, rows.draft_rows * SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft RoPE cosine"},
                           {7, rows.draft_rows * SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft RoPE sine"}},
                       [&](metal::CommandGraph &graph, const std::vector<metal::MetalBuffer> &b) {
                         ops::RoPE::addTables(graph, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], rows, 2048);
                       });
  test::requireExtents(backend,
                       std::initializer_list<test::BufferExtent>{
                           {0, SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft inverse frequency"},
                           {1, rows.draft_rows * SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft RoPE cosine"},
                           {2, rows.draft_rows * SPLASH_DRAFT_ROPE_PAIRS * 4, 4, "draft RoPE sine"}},
                       [&](metal::CommandGraph &graph, const std::vector<metal::MetalBuffer> &b) {
                         ops::RoPE::addDraftRangeTables(graph, b[0], b[1], b[2], rows.draft_rows, 6130, 2048);
                       });
}

// The draft rows of a range of consecutive positions come out bit for bit as
// the tables of those positions listed one by one do: a restore rotates the
// keys it computes again exactly as their capture did.
void ropeRangeMatchesPositions(metal::MetalBackend &backend) {
  const auto buffer = [&](uint64_t bytes) { return test::sharedBuffer(backend, bytes); };
  constexpr uint32_t kMaximumRows = 2048;
  const metal::MetalBuffer inverse = buffer(SPLASH_DRAFT_ROPE_PAIRS * 4);
  auto *frequencies = static_cast<float *>(inverse.contents());
  for (uint32_t pair = 0; pair < SPLASH_DRAFT_ROPE_PAIRS; ++pair)
    frequencies[pair] = std::pow(1.0e6F, -float(pair) / SPLASH_DRAFT_ROPE_PAIRS);
  for (const auto [rows, start] : {std::pair{37U, 6130U}, std::pair{kMaximumRows, 260000U}}) {
    const metal::MetalBuffer targetPositions = buffer(3 * 4), draftPositions = buffer(rows * 4);
    std::memset(targetPositions.contents(), 0, 3 * 4);
    for (uint32_t row = 0; row < rows; ++row)
      static_cast<uint32_t *>(draftPositions.contents())[row] = start + row;
    const metal::MetalBuffer targetTable = buffer(SPLASH_TARGET_ROPE_PAIRS * 4);
    const uint64_t tableBytes = uint64_t{rows} * SPLASH_DRAFT_ROPE_PAIRS * 4;
    const metal::MetalBuffer cosine = buffer(tableBytes), sine = buffer(tableBytes);
    const metal::MetalBuffer rangeCosine = buffer(tableBytes), rangeSine = buffer(tableBytes);
    metal::CommandGraph graph;
    ops::RoPE::addTables(graph, targetPositions, draftPositions,
                         buffer(SPLASH_TARGET_ROPE_PAIRS * 4), inverse, targetTable,
                         buffer(SPLASH_TARGET_ROPE_PAIRS * 4), cosine, sine, {1, rows, 1.0F}, kMaximumRows);
    ops::RoPE::addDraftRangeTables(graph, inverse, rangeCosine, rangeSine, rows, start, kMaximumRows);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    require(std::memcmp(cosine.contents(), rangeCosine.contents(), tableBytes) == 0 &&
                std::memcmp(sine.contents(), rangeSine.contents(), tableBytes) == 0,
            "draft RoPE tables of a position range differ from those of its positions");
  }
  metal::CommandGraph invalid;
  rejects([&] { ops::RoPE::addDraftRangeTables(invalid, inverse, inverse, inverse, 0, 0, kMaximumRows); },
          "invalid RoPE table row count", "an empty draft RoPE range was accepted");
  rejects([&] {
            ops::RoPE::addDraftRangeTables(invalid, inverse, inverse, inverse, kMaximumRows + 1, 0, kMaximumRows);
          },
          "invalid RoPE table row count", "a draft RoPE range past its rows was accepted");
  require(invalid.empty(), "an invalid draft RoPE range encoded a dispatch");
}

// The target's tables hold the cosine and sine of each row's position on the
// axis its pair takes (interleaved M-RoPE: pair i reads axis i % 3) times the
// pair's inverse frequency, scaled by the factor given: YaRN's attention
// factor, or 1, which leaves them as they were. Positions run up to the KV
// ceiling, where the angles are largest, with and without YaRN's frequencies.
void ropeTargetScaling(metal::MetalBackend &backend) {
  const auto buffer = [&](uint64_t bytes) { return test::sharedBuffer(backend, bytes); };
  constexpr uint32_t kPairs = SPLASH_TARGET_ROPE_PAIRS, kMaximumRows = 2048;
  constexpr uint32_t kLast = kv::kMaximumLogicalTokens - 1;
  const std::vector<std::array<uint32_t, 3>> positions{
      {0, 0, 0}, {1, 1, 1}, {4097, 4097, 4097}, {262143, 262143, 262143}, {262144, 262144, 262144},
      {700001, 699990, 699995}, {kLast - 7, kLast - 7, kLast - 7}, {kLast, kLast, kLast}};
  const auto rows = static_cast<uint32_t>(positions.size());
  const metal::MetalBuffer targetPositions = buffer(rows * 3 * 4);
  std::memcpy(targetPositions.contents(), positions.data(), rows * 3 * 4);
  const metal::MetalBuffer inverse = buffer(kPairs * 4), draft = buffer(SPLASH_DRAFT_ROPE_PAIRS * 4);
  const uint64_t tableBytes = uint64_t{rows} * kPairs * 4;
  double largestError = 0.0;
  for (const model::RopeScaling scaling : {model::RopeScaling{}, model::RopeScaling{4.0, 262144}}) {
    auto *frequencies = static_cast<float *>(inverse.contents());
    model::rotaryInverseFrequencies(1.0e7F, kPairs, scaling, {frequencies, kPairs});
    const float factor = scaling.attentionFactor();
    std::array<metal::MetalBuffer, 4> tables{buffer(tableBytes), buffer(tableBytes), buffer(tableBytes),
                                             buffer(tableBytes)};
    metal::CommandGraph graph;
    ops::RoPE::addTables(graph, targetPositions, draft, inverse, draft, tables[0], tables[1], draft, draft,
                         {rows, 0, 1.0F}, kMaximumRows);
    ops::RoPE::addTables(graph, targetPositions, draft, inverse, draft, tables[2], tables[3], draft, draft,
                         {rows, 0, factor}, kMaximumRows);
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto value = [&](uint32_t table, uint32_t element) {
      return static_cast<const float *>(tables[table].contents())[element];
    };
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t pair = 0; pair < kPairs; ++pair) {
        const uint32_t element = row * kPairs + pair;
        const float angle = static_cast<float>(positions[row][pair % 3]) * frequencies[pair];
        largestError = std::max({largestError, std::abs(value(0, element) - std::cos(double{angle})),
                                 std::abs(value(1, element) - std::sin(double{angle}))});
        require(value(2, element) == value(0, element) * factor && value(3, element) == value(1, element) * factor,
                "the target RoPE tables are not the unscaled ones times the scale");
      }
    }
  }
  std::cout << "target RoPE tables to position " << kLast << ": largest error " << largestError << '\n';
  require(largestError < 1e-6, "the target RoPE tables lost precision at large positions");
  metal::CommandGraph invalid;
  rejects([&] {
            ops::RoPE::addTables(invalid, targetPositions, draft, inverse, draft, inverse, inverse, draft, draft,
                                 {rows, 0, 0.0F}, kMaximumRows);
          },
          "invalid target RoPE scale", "a target RoPE scale of zero was accepted");
  require(invalid.empty(), "an invalid RoPE table encoded a dispatch");
}

} // namespace

// A decode step's commit of each lane's accepted path
// (PagedAttention::addPathCommit): in every layer of a pool of three, the
// path's row at depth k moves from its slot, committed plus its row, to
// committed plus k: its keys, values and, INT8, both scales, each from the
// bytes before the commit (lane 1's rows 2, 3 and 7 move to 1, 2 and 3,
// where row 3's target is row 2's source). Every other byte of the pool stays: the
// rows the path leaves, lanes whose paths are chains (lane 0 the anchor
// alone, lane 3 every row) and the layers and heads around the moved ones.
// Lanes' blocks straddle pages of three extents, through tables of
// scattered pages.
void checkPathCommit(metal::MetalBackend &backend, kv::Layout layout) {
  layout.attentionLayers = 3;
  const std::string what = std::string("KV path commit ") + std::string(kv::formatName(layout.format)) + " kv" +
                           std::to_string(layout.kvHeads) + " d" + std::to_string(layout.headDimension) +
                           (layout.valuesTokenMajor ? " token-major values" : "");
  constexpr uint32_t kLanes = 4, kRows = SPLASH_TARGET_VERIFY_ROWS;
  const std::array<uint32_t, kLanes> committed{0, 25, 63, 100};
  const std::array<uint32_t, kLanes> paths{0b1, 0b10001101, 0b1000001000100001, ops::chainPath(kRows)};
  std::array<std::vector<uint32_t>, kLanes> pages;
  uint32_t total = 0;
  for (uint32_t lane = 0; lane < kLanes; ++lane) total += (committed[lane] + kRows + kv::kPageTokens - 1) / kv::kPageTokens;
  const HostKvExtents::Geometry geometry = HostKvExtents::spread(total);
  HostKvExtents pool(backend, layout, geometry.extentPages, geometry.extents);
  const std::vector<uint32_t> ids = HostKvExtents::mixedPages(geometry, total, 7);
  std::array<metal::MetalBuffer, kLanes> tables;
  for (uint32_t lane = 0, next = 0; lane < kLanes; ++lane) {
    const uint32_t count = (committed[lane] + kRows + kv::kPageTokens - 1) / kv::kPageTokens;
    pages[lane].assign(ids.begin() + next, ids.begin() + next + count);
    next += count;
    tables[lane] = test::sharedBuffer(backend, count * sizeof(SplashKvPage));
    pool.writeTable(pages[lane], tables[lane]);
  }
  std::mt19937 random(layout.kvHeads * 131 + layout.headDimension + layout.valuesTokenMajor);
  for (uint32_t extent = 0; extent < pool.extentCount(); ++extent)
    for (std::byte &byte : pool.bytes(extent)) byte = std::byte(random());
  std::vector<std::vector<std::byte>> expected;
  for (uint32_t extent = 0; extent < pool.extentCount(); ++extent)
    expected.emplace_back(pool.bytes(extent).begin(), pool.bytes(extent).end());
  // The host's moves, from the bytes before them, into `expected`.
  const uint64_t element = layout.format == kv::Format::Int8 ? 1 : 2;
  const auto at = [&](std::vector<std::vector<std::byte>> &bytes, uint32_t lane, uint32_t layer, uint32_t tensor,
                      uint32_t token, uint64_t offset) -> std::byte * {
    const uint32_t page = pages[lane][token / kv::kPageTokens];
    const std::byte *slab = pool.slab<std::byte>(layer, tensor, page);
    const uint32_t extent = page / pool.extentPages();
    return bytes[extent].data() + (slab - pool.bytes(extent).data()) + offset;
  };
  const std::vector<std::vector<std::byte>> before = expected;
  auto source = before;
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    for (uint32_t depth = 0, rows = paths[lane]; rows; ++depth, rows &= rows - 1) {
      const uint32_t row = static_cast<uint32_t>(std::countr_zero(rows));
      if (row == depth) continue;
      const uint32_t from = committed[lane] + row, to = committed[lane] + depth;
      for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer)
        for (uint32_t head = 0; head < layout.kvHeads; ++head) {
          for (uint32_t dimension = 0; dimension < layout.headDimension; ++dimension) {
            const uint64_t key = splash_kv_key_element(head, 0, dimension, layout.headDimension) * element;
            std::copy_n(at(source, lane, layer, SPLASH_KV_KEYS, from,
                           key + uint64_t{from % kv::kPageTokens} * layout.headDimension * element),
                        element,
                        at(expected, lane, layer, SPLASH_KV_KEYS, to,
                           key + uint64_t{to % kv::kPageTokens} * layout.headDimension * element));
            std::copy_n(at(source, lane, layer, SPLASH_KV_VALUES, from,
                           pool.valueElement(head, from % kv::kPageTokens, dimension) * element),
                        element,
                        at(expected, lane, layer, SPLASH_KV_VALUES, to,
                           pool.valueElement(head, to % kv::kPageTokens, dimension) * element));
          }
          if (layout.format == kv::Format::Int8)
            for (const uint32_t tensor : {SPLASH_KV_KEY_SCALES, SPLASH_KV_VALUE_SCALES})
              std::copy_n(at(source, lane, layer, tensor, from,
                             splash_kv_scale_element(head, from % kv::kPageTokens) * sizeof(float)),
                          sizeof(float),
                          at(expected, lane, layer, tensor, to,
                             splash_kv_scale_element(head, to % kv::kPageTokens) * sizeof(float)));
        }
    }
  const metal::MetalBuffer pathBuffer = test::sharedBuffer(backend, sizeof(paths));
  std::memcpy(pathBuffer.contents(), paths.data(), sizeof(paths));
  metal::CommandGraph graph;
  ops::PagedAttention::addPathCommit(graph, layout, pool.extentPages(), tables, committed, kRows, pathBuffer);
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  for (uint32_t extent = 0; extent < pool.extentCount(); ++extent)
    require(std::equal(expected[extent].begin(), expected[extent].end(), pool.bytes(extent).begin()),
            what + ": the pool is not the paths' rows moved to their depths and every other byte as it was");
  require(expected != before, what + ": no path moved a row");
  // The tables hold the slots of a block's rows from each lane's position,
  // a lane's path holds its anchor's and rows of a block.
  rejects(
      [&] {
        metal::CommandGraph refused;
        const std::array<metal::MetalBuffer, 1> shorter{test::sharedBuffer(backend, sizeof(SplashKvPage))};
        ops::PagedAttention::addPathCommit(refused, layout, pool.extentPages(), shorter, std::array{31U}, 2,
                                           pathBuffer);
      },
      "page table buffer holds", what + ": a table short of a block's slots was accepted");
  std::cout << what << ": PASS\n";
}

int main(int argc, char **argv) {
  try {
    require(argc >= 1 && argc <= 3, "usage: paged-attention-plan [METALLIB [--long]]");
    for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
      for (const Geometry geometry : kGeometries) {
        checkSplitRule(geometry.queryHeads, geometry.layout(format));
        checkGrouping(geometry.queryHeads, geometry.layout(format));
        checkInt8Queries(geometry.queryHeads, geometry.layout(format));
      }
    if (argc == 1) {
      std::cout << "paged attention plans: CPU PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    checkBf16StoreEdges(backend);
    for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
      for (const kv::Layout layout :
           {kv::Layout{.attentionLayers = 1, .kvHeads = 4, .headDimension = 256},
            kv::Layout{.attentionLayers = 1, .kvHeads = 2, .headDimension = 256},
            kv::Layout{.attentionLayers = 1, .kvHeads = 4, .headDimension = 128},
            kv::Layout{.attentionLayers = 1, .kvHeads = 2, .headDimension = 256, .valuesTokenMajor = true,
                       .reads = kv::Reads::SelectedBlocks}}) {
        kv::Layout formatted = layout;
        formatted.format = format;
        checkPathCommit(backend, formatted);
      }
    ropeExtents(backend);
    ropeRangeMatchesPositions(backend);
    ropeTargetScaling(backend);
    for (const Geometry geometry : kGeometries)
      bufferExtents(backend, geometry.queryHeads, geometry.layout(kv::Format::Int8));
    attentionBlocks(backend);
    stageEncoding(backend);
    blockListsMatchMasks(backend);
    // The prepare kernels store each KV format; their norms are of each type
    // the geometry's families store.
    const std::array lanes{ops::PagedAttention::chunkParams(0, 8, 32, 1), ops::PagedAttention::chunkParams(9, 8, 32, 1),
                           ops::PagedAttention::chunkParams(31, 8, 32, 2)};
    const std::array chunk{ops::PagedAttention::chunkParams(100, 37, 64, 5)};
    const std::array mixed{ops::PagedAttention::chunkParams(0, 5, 32, 1), ops::PagedAttention::chunkParams(3, 13, 32, 1),
                           ops::PagedAttention::chunkParams(40, 8, 32, 2)};
    // A decode step's lanes of a cap of 16 rows, holding 1, 16, 9 and 3.
    const std::array capped{ops::PagedAttention::chunkParams(0, 16, 32, 1), ops::PagedAttention::chunkParams(9, 16, 32, 1),
                            ops::PagedAttention::chunkParams(31, 16, 32, 2), ops::PagedAttention::chunkParams(70, 16, 32, 3)};
    constexpr std::array<uint32_t, 4> cappedRows{1, 16, 9, 3};
    for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
      for (const Geometry geometry : kGeometries)
        for (bool float32 : splash::test::familyNorms(geometry)) {
          for (const std::span<const kv::ChunkedPrefillParams> sequences :
               {std::span<const kv::ChunkedPrefillParams>(lanes), std::span<const kv::ChunkedPrefillParams>(chunk),
                std::span<const kv::ChunkedPrefillParams>(mixed)})
            checkProjection(backend, geometry.queryHeads, geometry.layout(format), float32, sequences);
          checkProjection(backend, geometry.queryHeads, geometry.layout(format), float32, capped, cappedRows);
        }
    if (argc == 3 && std::string_view(argv[2]) == "--long") {
      for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
        for (const Geometry geometry : kGeometries)
          for (uint32_t history : {131072U, 260096U}) {
            const kv::Layout layout = geometry.layout(format);
            const std::array lane{Lane{2048, history, 0}};
            auto chunkCase = makeCase(backend, geometry.queryHeads, layout, lane);
            checkReference(chunkCase, run(backend, chunkCase, {.testBounds = true}));
            auto lanesCase = makeCase(backend, geometry.queryHeads, layout, lanesOf(4, kLaneRows, history));
            checkReference(lanesCase, run(backend, lanesCase, {.testBounds = true}));
            std::cout << "long attention: format=" << kv::formatName(format) << " q=" << geometry.queryHeads
                      << " kv=" << geometry.kvHeads << " history=" << history << " PASS\n"
                      << std::flush;
          }
      std::cout << "long paged attention plans: PASS\n";
      return 0;
    }
    require(argc == 2, "usage: paged-attention-plan [METALLIB [--long]]");
    for (auto format : {kv::Format::Int8, kv::Format::BFloat16})
      for (const Geometry geometry : kGeometries) {
        const uint32_t heads = geometry.queryHeads;
        const kv::Layout layout = geometry.layout(format);
        for (const auto [history, rows] : std::array<std::array<uint32_t, 2>, 10>{
                 {{0, 1}, {33, 7}, {255, 17}, {1023, 8}, {0, 2048}, {4093, 1057}, {16383, 257}, {100, 77}, {8064, 130},
                  {6145, 2048}}})
          checkPrefill(backend, heads, layout, history, rows);
        for (const std::vector<Lane> &lanesOfTails :
             {std::vector<Lane>{{1, 0, 0}}, std::vector<Lane>{{17, 255, 0}}, std::vector<Lane>{{1057, 4093, 0}},
              std::vector<Lane>{{256, 16383, 0}}, std::vector<Lane>{{2048, 6145, 0}},
              std::vector<Lane>{{13, 4093, 0}, {1, 0, 1}, {40, 1023, 2}, {77, 6145, 3}}})
          checkLastTiles(backend, heads, layout, lanesOfTails);
        for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
          checkLanes(backend, heads, layout, 0, lanes);
          checkLanes(backend, heads, layout, 1023, lanes);
        }
        checkLanes(backend, heads, layout, 16384, 2);
        checkMixed(backend, heads, layout);
        checkPackedLanes(backend, heads, layout);
        if (format == kv::Format::Int8) checkPackedGate(backend, heads, layout);
        if (selectsBlocks(layout)) {
          // The sparse kernels through the block lists of QSA masks: a chunk
          // whose last tile holds fewer rows, its last tile, and lanes of
          // their own histories.
          const std::array chunkLane{Lane{77, 6145, 0}};
          auto prefill = makeCase(backend, heads, layout, chunkLane);
          setQsaMasks(prefill);
          checkReference(prefill, run(backend, prefill));
          checkLastTiles(backend, heads, layout, chunkLane, true);
          auto lanesCase = makeCase(backend, heads, layout, lanesOf(3, kLaneRows, 1023));
          setQsaMasks(lanesCase);
          checkReference(lanesCase, run(backend, lanesCase));
          std::cout << "paged sparse attention: format=" << kv::formatName(format) << " PASS\n";
          checkLocations(backend, heads, layout);
        }
        for (const auto [history, rows] : std::array<std::array<uint32_t, 2>, 2>{{{6145, 77}, {4096, 130}}}) {
          checkStagedPrefill(backend, heads, layout, history, rows, false);
          if (selectsBlocks(layout)) checkStagedPrefill(backend, heads, layout, history, rows, true);
        }
      }
    std::cout << "paged attention plans: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "paged attention plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}
