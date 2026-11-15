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

// The parameters a verify step's dispatches share: its lanes' anchors, and
// its rows' lanes and places on the device.
BlockIndexerParams treeParams(const IndexerTreeRows &rows) {
  const auto lanes = static_cast<uint32_t>(rows.anchors.size());
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || rows.index.tables.size() != lanes || !rows.laneCap ||
      rows.laneCap > SPLASH_TARGET_VERIFY_ROWS || rows.rows < lanes || rows.rows > uint64_t{lanes} * rows.laneCap)
    throw std::invalid_argument("invalid indexer tree rows");
  requireBytes(rows.laneRows, uint64_t{lanes} * sizeof(SplashLaneRows), "indexer lane rows");
  requireBytes(rows.tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "indexer tree rows");
  BlockIndexerParams params{.lanes = lanes,
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
  std::ranges::copy(rows.anchors, std::begin(params.first_position));
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

void BlockIndexer::addTreePool(metal::CommandGraph &graph, const IndexerBlocks &blocks, metal::MetalBuffer blockKeys,
                               const NormWeights &keyNorm, metal::MetalBuffer inverseFrequencies, float ropeScale,
                               const IndexerTreeRows &rows) {
  BlockIndexerParams params = treeParams(rows);
  if (blocks.counts.size() != params.lanes || blocks.pending.size() != params.lanes || !blocks.laneRecords)
    throw std::invalid_argument("invalid indexer blocks");
  if (!(ropeScale > 0.0F)) throw std::invalid_argument("invalid indexer RoPE scale");
  params.lane_records = blocks.laneRecords;
  params.rope_scale = ropeScale;
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    // A lane's records hold every block a path of its most rows may
    // complete, from its anchor's block on.
    const uint32_t anchor = rows.anchors[lane], count = blocks.counts[lane];
    if (count > blocks.laneRecords || count < completeBlocks(anchor, rows.laneCap) - anchor / BLOCK_INDEXER_BLOCK)
      throw std::invalid_argument("an indexer lane's blocks are not its rows'");
    requireBytes(blocks.records, (uint64_t{blocks.laneRecords} * lane + count) * sizeof(BlockIndexerBlock),
                 "indexer block");
    requireBytes(blocks.pending[lane], kPendingBytes, "indexer pending keys");
    params.counts[lane] = count;
  }
  requireBytes(blocks.keys, uint64_t{rows.rows} * kKeyBytes, "indexer keys");
  requireBytes(blockKeys, uint64_t{rows.rows} * kKeyBytes, "indexer block keys");
  const auto pending = laneBindings(blocks.pending);
  graph.add("block_indexer_pool_tree",
            {blocks.records, blocks.keys, rows.tree, rows.laneRows, pending[0], pending[1], pending[2], pending[3],
             keyNorm.buffer, std::move(inverseFrequencies), std::move(blockKeys)},
            params, {rows.laneCap, params.lanes, 1}, {BLOCK_INDEXER_DIMENSION, 1, 1});
}

void BlockIndexer::addTreeSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer blockKeys,
                                 metal::MetalBuffer scores, metal::MetalBuffer mask, metal::MetalBuffer rowMasks,
                                 const IndexerTreeRows &rows) {
  const BlockIndexerParams params = treeParams(rows);
  if (!rows.topBlocks || !rows.maskWords) throw std::invalid_argument("invalid indexer selection");
  // A row's scores hold each complete block before it, and its bitmap those,
  // its tail block and the blocks of its lane's slots up to its own: the
  // kernels write no further.
  uint32_t blocks = 0, history = 0;
  for (const uint32_t anchor : rows.anchors) {
    blocks = std::max(blocks, static_cast<uint32_t>(completeBlocks(anchor, rows.laneCap)));
    history = std::max(history, anchor / BLOCK_INDEXER_BLOCK);
  }
  if (rows.scoreStride < blocks || uint64_t{rows.maskWords} * 32 <= blocks)
    throw std::invalid_argument("indexer scores or bitmaps narrower than the rows' history");
  requireBytes(queries, uint64_t{rows.rows} * BLOCK_INDEXER_HEADS * BLOCK_INDEXER_DIMENSION * sizeof(uint16_t),
               "indexer queries");
  requireBytes(blockKeys, uint64_t{rows.rows} * kKeyBytes, "indexer block keys");
  requireBytes(scores, uint64_t{rows.rows} * rows.scoreStride * sizeof(float), "indexer score");
  requireBytes(mask, uint64_t{rows.rows} * rows.maskWords * sizeof(uint32_t), "indexer bitmap");
  requireBytes(rowMasks, uint64_t{rows.rows} * sizeof(uint32_t), "indexer row masks");
  const auto tables = laneBindings(rows.index.tables);
  // The history's tiles, then each row of tiles' last, the step's own blocks.
  graph.add("block_indexer_score_tree",
            {std::move(queries), std::move(blockKeys), rows.tree, rows.laneRows, tables[0], tables[1], tables[2],
             tables[3], scores},
            params,
            {(history + BLOCK_INDEXER_SCORE_BLOCKS - 1) / BLOCK_INDEXER_SCORE_BLOCKS + 1,
             (rows.laneCap + BLOCK_INDEXER_SCORE_ROWS - 1) / BLOCK_INDEXER_SCORE_ROWS, params.lanes},
            {BLOCK_INDEXER_SCORE_THREADS, 1, 1});
  graph.add("block_indexer_select_tree", {scores, rows.tree, rows.laneRows, std::move(mask), std::move(rowMasks)},
            params, {rows.laneCap, params.lanes, 1}, {BLOCK_INDEXER_SELECT_THREADS, 1, 1});
}

void BlockIndexer::addPending(metal::CommandGraph &graph, metal::MetalBuffer keys, metal::MetalBuffer pendingIn,
                              metal::MetalBuffer pendingOut, uint32_t firstPosition, uint32_t retained) {
  requireBytes(keys, uint64_t{retained} * kKeyBytes, "indexer keys");
  requireBytes(pendingIn, kPendingBytes, "indexer pending keys");
  requireBytes(pendingOut, kPendingBytes, "indexer pending keys");
  const BlockIndexerPendingParams params{.first_position = firstPosition, .retained = retained};
  graph.add("block_indexer_pending", {std::move(keys), std::move(pendingIn), std::move(pendingOut)}, params, {1, 1, 1},
            {BLOCK_INDEXER_DIMENSION, 1, 1});
}

void BlockIndexer::addCommit(metal::CommandGraph &graph, const IndexerCommit &commit) {
  const auto lanes = static_cast<uint32_t>(commit.anchors.size());
  if (!commit.layers || !commit.layerRows || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH ||
      commit.pendingIn.size() != lanes || commit.pendingOut.size() != lanes || commit.indexTables.size() != lanes)
    throw std::invalid_argument("invalid indexer commit");
  const uint64_t keyBytes = uint64_t{commit.layers} * commit.layerRows * kKeyBytes;
  requireBytes(commit.keys, keyBytes, "indexer keys");
  requireBytes(commit.blockKeys, keyBytes, "indexer block keys");
  requireBytes(commit.paths, uint64_t{lanes} * sizeof(uint32_t), "indexer accepted paths");
  requireBytes(commit.laneRows, uint64_t{lanes} * sizeof(SplashLaneRows), "indexer lane rows");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(commit.pendingIn[lane], commit.layers * kPendingBytes, "indexer pending keys");
    requireBytes(commit.pendingOut[lane], commit.layers * kPendingBytes, "indexer pending keys");
  }
  BlockIndexerCommitParams params{
      .lanes = lanes, .layer_rows = commit.layerRows, .index_layer_bytes = commit.indexLayerBytes, .first_position = {}};
  std::ranges::copy(commit.anchors, std::begin(params.first_position));
  const auto in = laneBindings(commit.pendingIn), out = laneBindings(commit.pendingOut);
  const auto tables = laneBindings(commit.indexTables);
  graph.add("block_indexer_commit",
            {commit.keys, commit.blockKeys, commit.paths, commit.laneRows, in[0], in[1], in[2], in[3], out[0], out[1],
             out[2], out[3], tables[0], tables[1], tables[2], tables[3]},
            params, {commit.layers, lanes, 1}, {BLOCK_INDEXER_DIMENSION, 1, 1});
}

} // namespace splash::ops
