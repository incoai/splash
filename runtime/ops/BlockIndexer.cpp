#include "ops/BlockIndexer.hpp"

#include "metal/abi/BlockIndexer.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace splash::ops {

namespace {

constexpr uint64_t kKeyBytes = BLOCK_INDEXER_DIMENSION * sizeof(uint16_t);
constexpr uint64_t kPendingBytes = BLOCK_INDEXER_PENDING * kKeyBytes;

// The parameters every indexer dispatch of `rows`' lanes shares. The
// kernels index a lane's rows from its first row, and the score scratch's
// from the first lane's, so the lanes' rows are in row order.
BlockIndexerParams indexerParams(const IndexerRows &rows) {
  if (rows.lanes.empty() || rows.firstPositions.size() != rows.lanes.size() ||
      rows.index.tables.size() != rows.lanes.size())
    throw std::invalid_argument("invalid indexer rows");
  requireLaneRows(rows.lanes);
  requireRowOrder(rows.lanes);
  BlockIndexerParams params{.lanes = static_cast<uint32_t>(rows.lanes.size()),
                            .first_position = {},
                            .first_row = {},
                            .rows = {},
                            .counts = {},
                            .lane_records = 0,
                            .row_offset = 0,
                            .top_blocks = rows.topBlocks,
                            .mask_words = rows.maskWords,
                            .score_stride = rows.scoreStride,
                            .index_offset = rows.index.offset,
                            .rope_scale = 0.0F};
  std::ranges::copy(rows.firstPositions, std::begin(params.first_position));
  copyLaneRows(rows.lanes, params.first_row, params.rows);
  return params;
}

} // namespace

void BlockIndexer::addProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                              const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                              metal::MetalBuffer queries, metal::MetalBuffer keys, const IndexerRows &rows) {
  const BlockIndexerParams params = indexerParams(rows);
  if (!queryNorm.float32) throw std::invalid_argument("the indexer query norm is not F32");
  const uint64_t laneRows = rowsEnd(rows.lanes);
  requireBytes(query, laneRows * BLOCK_INDEXER_HEADS * BLOCK_INDEXER_DIMENSION * sizeof(float), "indexer query");
  requireBytes(key, laneRows * BLOCK_INDEXER_DIMENSION * sizeof(float), "indexer key");
  requireBytes(queryNorm.buffer, BLOCK_INDEXER_DIMENSION * sizeof(float), "indexer query norm");
  requireBytes(ropeCos, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "indexer RoPE cosine");
  requireBytes(ropeSin, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "indexer RoPE sine");
  requireBytes(queries, laneRows * BLOCK_INDEXER_HEADS * BLOCK_INDEXER_DIMENSION * sizeof(uint16_t), "indexer queries");
  requireBytes(keys, laneRows * kKeyBytes, "indexer keys");
  graph.add("block_indexer_project",
            {std::move(query), std::move(key), queryNorm.buffer, std::move(ropeCos), std::move(ropeSin),
             std::move(queries), std::move(keys)},
            params, {mostRows(rows.lanes), BLOCK_INDEXER_HEADS + 1, params.lanes}, {BLOCK_INDEXER_DIMENSION, 1, 1});
}

void BlockIndexer::addPool(metal::CommandGraph &graph, const IndexerBlocks &blocks, const NormWeights &keyNorm,
                           metal::MetalBuffer inverseFrequencies, float ropeScale, const IndexerRows &rows) {
  BlockIndexerParams params = indexerParams(rows);
  if (blocks.counts.size() != params.lanes || blocks.pending.size() != params.lanes ||
      (params.lanes > 1 && !blocks.laneRecords))
    throw std::invalid_argument("invalid indexer blocks");
  if (!(ropeScale > 0.0F)) throw std::invalid_argument("invalid indexer RoPE scale");
  params.lane_records = blocks.laneRecords;
  params.rope_scale = ropeScale;
  uint32_t most = 0;
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    const uint32_t count = blocks.counts[lane];
    if (count && params.lanes > 1 && count > blocks.laneRecords)
      throw std::invalid_argument("an indexer lane's blocks overrun the next lane's");
    // A lane's rows complete at most the blocks their last row ends, the
    // first of them reaching back into its pending keys.
    if (count > completeBlocks(rows.firstPositions[lane], rows.lanes[lane].rows) -
                    rows.firstPositions[lane] / BLOCK_INDEXER_BLOCK)
      throw std::invalid_argument("an indexer lane's blocks are not its rows'");
    requireBytes(blocks.records, (uint64_t{blocks.laneRecords} * lane + count) * sizeof(BlockIndexerBlock),
                 "indexer block");
    requireBytes(blocks.pending[lane], kPendingBytes, "indexer pending keys");
    params.counts[lane] = count;
    most = std::max(most, count);
  }
  if (!most) return;
  requireBytes(blocks.keys, rowsEnd(rows.lanes) * kKeyBytes, "indexer keys");
  const auto pending = laneBindings(blocks.pending);
  const auto tables = laneBindings(rows.index.tables);
  graph.add("block_indexer_pool",
            {blocks.records, blocks.keys, pending[0], pending[1], pending[2], pending[3], keyNorm.buffer,
             std::move(inverseFrequencies), tables[0], tables[1], tables[2], tables[3]},
            params, {most, params.lanes, 1}, {BLOCK_INDEXER_DIMENSION, 1, 1});
}

void BlockIndexer::addSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer scores,
                             uint32_t scoreRows, metal::MetalBuffer mask, const IndexerRows &rows) {
  BlockIndexerParams params = indexerParams(rows);
  // The scratch holds the dispatch's rows from its first lane's first row.
  const uint64_t laneRows = rowsEnd(rows.lanes), scratchRows = laneRows - rows.lanes.front().first;
  const uint32_t most = mostRows(rows.lanes);
  if (!scoreRows || !rows.topBlocks || !rows.maskWords || (params.lanes > 1 && scoreRows < scratchRows))
    throw std::invalid_argument("invalid indexer selection");
  const auto tables = laneBindings(rows.index.tables);
  uint32_t blocks = 0;
  for (size_t lane = 0; lane < rows.lanes.size(); ++lane)
    blocks = std::max(blocks,
                      static_cast<uint32_t>(completeBlocks(rows.firstPositions[lane], rows.lanes[lane].rows)));
  // A row's scores hold each complete block before it, and its bitmap those
  // and its tail block: the kernels write no further.
  if (rows.scoreStride < blocks || uint64_t{rows.maskWords} * 32 <= blocks)
    throw std::invalid_argument("indexer scores or bitmaps narrower than the rows' history");
  requireBytes(queries, laneRows * BLOCK_INDEXER_HEADS * BLOCK_INDEXER_DIMENSION * sizeof(uint16_t), "indexer queries");
  requireBytes(scores, (params.lanes > 1 ? scratchRows : std::min(scoreRows, most)) * uint64_t{rows.scoreStride} *
                           sizeof(float),
               "indexer score");
  requireBytes(mask, laneRows * rows.maskWords * sizeof(uint32_t), "indexer bitmap");
  // One lane's rows in batches of the scratch's rows; several lanes' at once.
  const uint32_t batchRows = params.lanes > 1 ? most : scoreRows;
  for (uint32_t offset = 0; offset < most; offset += batchRows) {
    params.row_offset = offset;
    for (size_t lane = 0; lane < rows.lanes.size(); ++lane)
      params.rows[lane] = std::min(batchRows, rows.lanes[lane].rows - offset);
    const uint32_t batch = std::min(batchRows, most - offset);
    graph.add("block_indexer_score", {queries, tables[0], tables[1], tables[2], tables[3], scores}, params,
              {(blocks + BLOCK_INDEXER_SCORE_BLOCKS - 1) / BLOCK_INDEXER_SCORE_BLOCKS,
               (batch + BLOCK_INDEXER_SCORE_ROWS - 1) / BLOCK_INDEXER_SCORE_ROWS, params.lanes},
              {BLOCK_INDEXER_SCORE_THREADS, 1, 1});
    graph.add("block_indexer_select", {scores, mask}, params, {batch, params.lanes, 1},
              {BLOCK_INDEXER_SELECT_THREADS, 1, 1});
  }
}

void BlockIndexer::addPending(metal::CommandGraph &graph, metal::MetalBuffer keys, metal::MetalBuffer pendingIn,
                              metal::MetalBuffer pendingOut, uint32_t firstPosition, uint32_t retained) {
  requireBytes(keys, uint64_t{retained} * kKeyBytes, "indexer keys");
  requireBytes(pendingIn, kPendingBytes, "indexer pending keys");
  requireBytes(pendingOut, kPendingBytes, "indexer pending keys");
  const BlockIndexerPendingParams params{.lanes = 1,
                                         .layer_rows = 0,
                                         .first_position = {firstPosition},
                                         .first_row = {0},
                                         .rows = {retained},
                                         .retained = {retained}};
  graph.add("block_indexer_pending", {std::move(keys), std::move(pendingIn), std::move(pendingOut)}, params, {1, 1, 1},
            {BLOCK_INDEXER_DIMENSION, 1, 1});
}

void BlockIndexer::addCommit(metal::CommandGraph &graph, metal::MetalBuffer keys, uint32_t layers, uint32_t layerRows,
                             std::span<const LaneRows> laneRows, std::span<const uint32_t> firstPositions,
                             metal::MetalBuffer retained, std::span<const metal::MetalBuffer> pendingIn,
                             std::span<const metal::MetalBuffer> pendingOut) {
  const auto lanes = static_cast<uint32_t>(firstPositions.size());
  if (!layers || laneRows.size() != lanes || pendingIn.size() != lanes || pendingOut.size() != lanes)
    throw std::invalid_argument("invalid indexer commit");
  requireLaneRows(laneRows);
  if (rowsEnd(laneRows) > layerRows) throw std::invalid_argument("invalid indexer commit");
  requireBytes(keys, uint64_t{layers} * layerRows * kKeyBytes, "indexer keys");
  requireBytes(retained, uint64_t{lanes} * sizeof(uint32_t), "indexer retained rows");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(pendingIn[lane], layers * kPendingBytes, "indexer pending keys");
    requireBytes(pendingOut[lane], layers * kPendingBytes, "indexer pending keys");
  }
  BlockIndexerPendingParams params{
      .lanes = lanes, .layer_rows = layerRows, .first_position = {}, .first_row = {}, .rows = {}, .retained = {}};
  std::ranges::copy(firstPositions, std::begin(params.first_position));
  copyLaneRows(laneRows, params.first_row, params.rows);
  const auto in = laneBindings(pendingIn), out = laneBindings(pendingOut);
  graph.add("block_indexer_commit",
            {std::move(keys), std::move(retained), in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3]},
            params, {layers, lanes, 1}, {BLOCK_INDEXER_DIMENSION, 1, 1});
}

} // namespace splash::ops
