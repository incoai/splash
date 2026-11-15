#pragma once

// Parameters of a block indexer (kernels/shared/block_indexer.metal,
// ops/BlockIndexer.cpp), whose block bitmaps are the block masks of a
// full-attention layer's sparse attention (Qwen3.8-Flash-Next's QSA, for one).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"
#include "metal/abi/VerifyTree.h"

// The block indexer of a full-attention layer: BLOCK_INDEXER_HEADS query
// heads and one key head of BLOCK_INDEXER_DIMENSION values, the shape the
// kernels are compiled for, which a model's layout must match; blocks of
// BLOCK_INDEXER_BLOCK tokens, those of the sparse attention's block
// masks (metal/abi/PagedAttention.h); `top_blocks` blocks per query row past
// which the row attends sparsely. The layer's index holds the pooled key of
// every complete block (bf16 [dimension]): a lane's index table holds an
// entry per logical page (SplashKvPage), the page's blocks' keys
// `index_offset` bytes into the index plane the entry names, after those
// of the pages before it (ops::BlockIndex). A block pools the raw keys of
// its tokens: a step's rows' (bf16 [row][dimension], which the project
// writes) and, before a lane's first row, the lane's pending keys, those of
// its open block (bf16 [slot][dimension] of BLOCK_INDEXER_PENDING slots, the
// token at position b * BLOCK_INDEXER_BLOCK + slot of the block b that is not
// complete), which its state keeps.
//
// A prompt's rows are committed as they run: the blocks they complete go into
// the index, from which every row scores them. A verify step's lane is a
// block of rows from its anchor, a tree (metal/abi/VerifyTree.h) whose rows'
// lanes and places the step's proposer writes on the device (SplashLaneRows,
// SplashTreeRow): a row's position is the anchor's plus its depth, so
// siblings share positions, and a block one of its rows completes holds that
// row's path's tokens. Such a row keeps the block's pooled key as its own,
// its block key (bf16 [row][dimension], block_indexer_pool_tree); a row
// scores the blocks complete before the anchor from the index and those its
// path completes from its path's block keys; and once the verify accepted a
// path, the commit writes the block keys of the path's rows into the index
// (block_indexer_commit).
#define BLOCK_INDEXER_DIMENSION 128u
#define BLOCK_INDEXER_HEADS 4u
#define BLOCK_INDEXER_BLOCK SPLASH_ATTENTION_BLOCK_TOKENS
#define BLOCK_INDEXER_PENDING (BLOCK_INDEXER_BLOCK - 1u)
// The score kernel's tiles: BLOCK_INDEXER_SCORE_ROWS rows by
// BLOCK_INDEXER_SCORE_BLOCKS blocks, a threadgroup of
// BLOCK_INDEXER_SCORE_THREADS threads each.
#define BLOCK_INDEXER_SCORE_ROWS 8u
#define BLOCK_INDEXER_SCORE_BLOCKS 32u
#define BLOCK_INDEXER_SCORE_THREADS 256u
// The select kernel's threads, a threadgroup per row.
#define BLOCK_INDEXER_SELECT_THREADS 1024u
// How far past a row's own tail block the block lists of the tile that holds
// it read its bitmap (attention_block_lists, metal/abi/PagedAttention.h): a
// tile of at most SPLASH_ATTENTION_BLOCK_ROW_BITS rows lists up to its last
// row's tail, at most this many blocks past any of its rows'. The select
// writes a row's bitmap that far.
#define BLOCK_INDEXER_TILE_SLACK \
  ((SPLASH_ATTENTION_BLOCK_ROW_BITS - 1u + BLOCK_INDEXER_BLOCK - 1u) / BLOCK_INDEXER_BLOCK)
// The projection of `lanes` lanes' rows (block_indexer_project): lane l's
// rows[l] rows from row first_row[l] of the queries and keys, the lanes in
// row order: a prompt sequence's rows as one, or a verify step's lanes.
struct BlockIndexerProjectParams {
  uint32_t lanes;
  uint32_t first_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t rows[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(BlockIndexerProjectParams) == 4 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "block indexer projection parameters are the same size on both sides");

// A prompt sequence's dispatch (block_indexer_pool, block_indexer_score,
// block_indexer_select), its rows committed as they run: `rows` rows of the
// queries, keys and bitmaps from their first, the first at logical position
// first_position + row_offset, through the sequence's index table. The pool
// pools its `blocks` blocks into the index, rotating their keys by the
// target's RoPE scaled by rope_scale, as the queries' tables are
// (RopeTableParams::target_scale). The selection scores a long sequence's
// rows in batches of the score scratch's rows: the batch from row
// `row_offset` of its queries and bitmaps, scored into the scratch's first
// rows.
struct BlockIndexerSequenceParams {
  uint32_t first_position;
  uint32_t rows;
  uint32_t blocks;
  uint32_t row_offset;
  uint32_t top_blocks;
  uint32_t mask_words;   // 32-bit words of a row's block bitmap
  uint32_t score_stride; // scores per row of the score scratch
  uint32_t index_offset; // the layer's index in every index entry's plane, in bytes
  float rope_scale;
};

static_assert(sizeof(BlockIndexerSequenceParams) == 36,
              "block indexer sequence parameters are the same size on both sides");

// A verify step's dispatch (block_indexer_pool_tree, block_indexer_score_tree,
// block_indexer_select_tree) over `lanes` lanes, each through its own index
// table, whose rows the step's proposer writes on the device, of the step's
// rows from row 0 (scores and bitmaps alike): lane l's anchor at
// first_position[l]; its pool's counts[l] records, lane_records records after
// the lane's before it, of the blocks from the anchor's on, one a block a path
// of the lane's most rows may complete; and as a sequence's, its budget,
// bitmap and score widths, index and RoPE scale.
struct BlockIndexerTreeParams {
  uint32_t lanes;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t counts[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lane_records;
  uint32_t top_blocks;
  uint32_t mask_words;
  uint32_t score_stride;
  uint32_t index_offset;
  float rope_scale;
};

static_assert(sizeof(BlockIndexerTreeParams) == 28 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "block indexer tree parameters are the same size on both sides");

// A sequence's pending keys after its first `retained` rows from
// first_position (block_indexer_pending): the keys of the rows' last block
// that they do not complete, from the rows' keys and, before the first row,
// its pending keys before them.
struct BlockIndexerPendingParams {
  uint32_t first_position;
  uint32_t retained;
};

static_assert(sizeof(BlockIndexerPendingParams) == 8,
              "block indexer pending parameters are the same size on both sides");

// The commit after a verify accepted a path of each of `lanes` lanes' rows
// (block_indexer_commit), in every attention layer: lane l's anchor at
// first_position[l]; a layer's keys and block keys `layer_rows` rows after
// the layer's before it, from the step's row 0; attention layer a's index
// a * index_layer_bytes bytes into every index entry's plane
// (kv::PageStorage::indexLayerBytes).
struct BlockIndexerCommitParams {
  uint32_t lanes;
  uint32_t layer_rows;
  uint32_t index_layer_bytes;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(BlockIndexerCommitParams) == 12 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "block indexer commit parameters are the same size on both sides");

// One block to pool: its first token's logical position and RoPE position
// (t, h, w; text repeats one), to which its pooled key is rotated.
struct BlockIndexerBlock {
  uint32_t position;
  uint32_t rope[3];
};

static_assert(sizeof(BlockIndexerBlock) == 16,
              "block indexer blocks are 16 bytes on both sides");
