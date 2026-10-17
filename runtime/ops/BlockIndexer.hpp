#pragma once

// The block indexer of a block-sparse attention (Qwen3.8-Flash-Next's QSA,
// for one), which picks the blocks a full-attention layer's rows attend
// (kernels/shared/block_indexer.metal). Its projections run through
// ops::Linear; a target composes them (model/Qwen4ExpTarget.cpp).

#include "metal/CommandGraph.hpp"
#include "metal/abi/BlockIndexer.h"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// Where a layer's block index sits for a dispatch's lanes: lane l's pooled
// keys through index table tables[l], an entry per logical page
// (SplashKvPage), `offset` bytes into each entry's extent (metal/abi/
// BlockIndexer.h). The index sits in the KV pages' extents
// (kv::PageStorage::indexOffset), so a lane's index table is its page table.
struct BlockIndex final {
  std::span<const metal::MetalBuffer> tables;
  uint32_t offset = 0;
};

// The indexer's rows of a dispatch's lanes (a verify step's, or a prefill
// sequence as one): `rows` query rows of each lane, lane l's from logical
// position firstPositions[l], which attend sparsely past topBlocks complete
// blocks of the index; bitmaps of maskWords words and scores of scoreStride
// per row, which cover each row's complete blocks and its tail.
struct IndexerRows final {
  uint32_t rows = 0;
  std::span<const uint32_t> firstPositions;
  BlockIndex index;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
};

// The blocks a dispatch's lanes' rows complete, which the pool pools: lane
// l's counts[l] BlockIndexerBlock records from record l * laneRecords of
// `records`, from the rows' raw keys (`keys`, bf16 rows of
// BLOCK_INDEXER_DIMENSION as the project writes them) and the tokens before
// lane l's rows from pending[l], its pending keys (BLOCK_INDEXER_PENDING bf16
// rows).
struct IndexerBlocks final {
  metal::MetalBuffer records;
  std::span<const uint32_t> counts;
  uint32_t laneRecords = 0;
  metal::MetalBuffer keys;
  std::span<const metal::MetalBuffer> pending;
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
  // The indexer's query heads, normalized and rotated (bf16 rows of
  // BLOCK_INDEXER_HEADS heads), and the rows' raw keys (bf16).
  static void addProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                         const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                         metal::MetalBuffer queries, metal::MetalBuffer keys, const IndexerRows &rows);
  // The pooled keys of the blocks the lanes' rows complete, into the index.
  static void addPool(metal::CommandGraph &graph, const IndexerBlocks &blocks, const NormWeights &keyNorm,
                      metal::MetalBuffer inverseFrequencies, const IndexerRows &rows);
  // The rows' scores of their complete blocks' pooled keys and, from them,
  // their bitmaps. A dispatch of one lane scores its rows in batches of the
  // score scratch's scoreRows rows; one of several lanes scores every lane's
  // rows at once, which the scratch must hold.
  static void addSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer scores,
                        uint32_t scoreRows, metal::MetalBuffer mask, const IndexerRows &rows);
  // A sequence's pending keys of one layer after its first `retained` rows
  // from `firstPosition`, of which `keys` holds the raw keys: those of the
  // block they leave open, from `keys` and `pendingIn`, its pending keys
  // before them.
  static void addPending(metal::CommandGraph &graph, metal::MetalBuffer keys, metal::MetalBuffer pendingIn,
                         metal::MetalBuffer pendingOut, uint32_t firstPosition, uint32_t retained);
  // Each verify lane's pending keys of each of `layers` layers after the rows
  // it retained (`retained`, a count per lane the GPU wrote), from the step's
  // keys of every layer (`layerRows` rows apart, each lane's `laneRows` rows
  // from firstPositions[l] after the lanes before it) and pendingIn[l], the
  // lane's pending keys before the step; a lane's layers follow each other in
  // its pending keys.
  static void addCommit(metal::CommandGraph &graph, metal::MetalBuffer keys, uint32_t layers, uint32_t layerRows,
                        uint32_t laneRows, std::span<const uint32_t> firstPositions, metal::MetalBuffer retained,
                        std::span<const metal::MetalBuffer> pendingIn,
                        std::span<const metal::MetalBuffer> pendingOut);
};

} // namespace splash::ops
