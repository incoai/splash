#pragma once

// The block indexer of a block-sparse attention (Qwen3.8-Flash-Next's QSA,
// for one), which picks the blocks a full-attention layer's rows attend
// (kernels/shared/block_indexer.metal). Its projections run through
// ops::Linear; a target composes them (model/Qwen4ExpTarget.cpp).

#include "metal/CommandGraph.hpp"
#include "metal/abi/BlockIndexer.h"
#include "ops/LaneRows.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Where a layer's block index sits for a dispatch's lanes: lane l's pooled
// keys through index table tables[l], an entry per logical page
// (SplashKvPage), `offset` bytes into the index plane each entry names
// (metal/abi/BlockIndexer.h). Every KV page keeps its index in its extent's
// index plane, whatever its form, so an index table has entries of its own
// (kv::PageStorage::writeIndexEntries and indexOffset).
struct BlockIndex final {
  std::span<const metal::MetalBuffer> tables;
  uint32_t offset = 0;
};

// The indexer's rows of a prompt sequence, committed as they run: `rows`
// rows (its queries', keys' and bitmaps' from their first) from logical
// position firstPosition, its index through `indexTable`, `indexOffset`
// bytes into each entry's plane, which attend sparsely past topBlocks
// complete blocks; bitmaps of maskWords words and scores of scoreStride per
// row, which cover each row's complete blocks and its tail.
struct IndexerSequence final {
  uint32_t rows = 0;
  uint32_t firstPosition = 0;
  metal::MetalBuffer indexTable;
  uint32_t indexOffset = 0;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
};

// The blocks a prompt sequence's rows complete, which the pool pools into
// its index: `count` BlockIndexerBlock records of `records`, from the rows'
// raw keys (`keys`, bf16 rows of BLOCK_INDEXER_DIMENSION as the project
// writes them) and the tokens before its first row from `pending`, its
// pending keys (BLOCK_INDEXER_PENDING bf16 rows).
struct IndexerSequenceBlocks final {
  metal::MetalBuffer records;
  uint32_t count = 0;
  metal::MetalBuffer keys;
  metal::MetalBuffer pending;
};

// The blocks a verify step's lanes' rows may complete, which the tree pool
// pools: lane l's counts[l] BlockIndexerBlock records from record
// l * laneRecords of `records`, of the blocks from its anchor's on that a
// path of its most rows may complete, from the rows' raw keys (`keys`) and
// the tokens before lane l's anchor from pending[l], its pending keys.
struct IndexerBlocks final {
  metal::MetalBuffer records;
  std::span<const uint32_t> counts;
  uint32_t laneRecords = 0;
  metal::MetalBuffer keys;
  std::span<const metal::MetalBuffer> pending;
};

// The indexer's rows of a verify step, whose lanes are blocks of rows from
// their anchors as the step's proposer lays them out on the device
// (metal/abi/VerifyTree.h): each lane's rows of the step's `rows` rows from
// row 0 (SplashLaneRows, in lane order, 1 to laneCap a lane,
// ops::verifyLaneCap) in `laneRows` and every row's place in its lane's tree
// (SplashTreeRow) in `tree`; lane l's anchor at logical position anchors[l];
// and as a sequence's (IndexerSequence) the lanes' index, the budget, and the
// words of a row's bitmap and its scores, a row of each per row of the step.
struct IndexerTreeRows final {
  metal::MetalBuffer laneRows;
  metal::MetalBuffer tree;
  uint32_t rows = 0;
  uint32_t laneCap = 0;
  std::span<const uint32_t> anchors;
  BlockIndex index;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
};

// The commit after a verify accepted each lane's path, in every attention
// layer of the step: `layers` layers whose raw keys (`keys`) and block keys
// (`blockKeys`, BlockIndexer::addTreePool's) are layerRows rows apart from
// the step's row 0; each lane's accepted path (`paths`, a row mask of its
// rows a lane, metal/abi/VerifyTree.h), its rows on the device
// (`laneRows`), its anchor and its pending keys of every layer before and
// after the step; and each lane's index table, the index of attention layer
// a a * indexLayerBytes into every entry's plane
// (kv::PageStorage::indexLayerBytes).
struct IndexerCommit final {
  metal::MetalBuffer keys;
  metal::MetalBuffer blockKeys;
  uint32_t layers = 0;
  uint32_t layerRows = 0;
  metal::MetalBuffer paths;
  metal::MetalBuffer laneRows;
  std::span<const uint32_t> anchors;
  std::span<const metal::MetalBuffer> pendingIn;
  std::span<const metal::MetalBuffer> pendingOut;
  std::span<const metal::MetalBuffer> indexTables;
  uint32_t indexLayerBytes = 0;
};

struct BlockIndexer final {
  // The complete blocks of BLOCK_INDEXER_BLOCK tokens before the last of
  // `rows` rows from `firstPosition`, its own included.
  [[nodiscard]] static constexpr uint64_t completeBlocks(uint64_t firstPosition, uint32_t rows) noexcept {
    return (firstPosition + rows) / BLOCK_INDEXER_BLOCK;
  }
  // Whether any row of `rows` from `firstPosition` attends sparsely: has
  // more than topBlocks complete blocks.
  [[nodiscard]] static bool sparse(uint32_t firstPosition, uint32_t rows, uint32_t topBlocks) noexcept {
    return completeBlocks(firstPosition, rows) > topBlocks;
  }
  // The indexer's query heads of `lanes`' rows (a prompt sequence's as one
  // lane, or a verify step's lanes, in row order), normalized and rotated by
  // each row's RoPE table (bf16 rows of BLOCK_INDEXER_HEADS heads), and the
  // rows' raw keys (bf16).
  static void addProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                         const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                         metal::MetalBuffer queries, metal::MetalBuffer keys, std::span<const LaneRows> lanes);
  // The pooled keys of the blocks a prompt sequence's rows complete, into its
  // index, rotated by the target's RoPE of `inverseFrequencies` scaled by
  // ropeScale, as the queries' tables are: YaRN's attention factor, 1
  // without YaRN.
  static void addPool(metal::CommandGraph &graph, const IndexerSequenceBlocks &blocks, const NormWeights &keyNorm,
                      metal::MetalBuffer inverseFrequencies, float ropeScale, const IndexerSequence &sequence);
  // A prompt sequence's rows' scores of their complete blocks' pooled keys
  // and, from them, their bitmaps, in batches of the score scratch's
  // scoreRows rows.
  static void addSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer scores,
                        uint32_t scoreRows, metal::MetalBuffer mask, const IndexerSequence &sequence);
  // A verify step's pool: each of its rows that completes a block pools it
  // from its path into its block key (bf16 rows of BLOCK_INDEXER_DIMENSION,
  // a row per row of the step); none goes into the index.
  static void addTreePool(metal::CommandGraph &graph, const IndexerBlocks &blocks, metal::MetalBuffer blockKeys,
                          const NormWeights &keyNorm, metal::MetalBuffer inverseFrequencies, float ropeScale,
                          const IndexerTreeRows &rows);
  // A verify step's scores and bitmaps: each row's of the blocks complete
  // before its anchor from the index and of those its path completes from
  // the step's block keys (addTreePool's), and its bitmap of the blocks of
  // K/V slots it reads with its row mask (a word per row of the step) of
  // its block's rows it attends (ops::AttentionBlocks::rowMasks).
  static void addTreeSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer blockKeys,
                            metal::MetalBuffer scores, metal::MetalBuffer mask, metal::MetalBuffer rowMasks,
                            const IndexerTreeRows &rows);
  // A sequence's pending keys of one layer after its first `retained` rows
  // from `firstPosition`, of which `keys` holds the raw keys: those of the
  // block they leave open, from `keys` and `pendingIn`, its pending keys
  // before them.
  static void addPending(metal::CommandGraph &graph, metal::MetalBuffer keys, metal::MetalBuffer pendingIn,
                         metal::MetalBuffer pendingOut, uint32_t firstPosition, uint32_t retained);
  // The commit after a verify accepted each lane's path: the block keys of
  // the blocks the path's rows complete into each layer's index, and the
  // lane's pending keys of each layer after the path.
  static void addCommit(metal::CommandGraph &graph, const IndexerCommit &commit);
};

} // namespace splash::ops
