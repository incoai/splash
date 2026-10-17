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

// The block indexer of a full-attention layer: BLOCK_INDEXER_HEADS query
// heads and one key head of BLOCK_INDEXER_DIMENSION values, the shape the
// kernels are compiled for, which a model's layout must match; blocks of
// BLOCK_INDEXER_BLOCK tokens, those of the sparse attention's block
// masks (metal/abi/PagedAttention.h); `top_blocks` blocks per query row past
// which the row attends sparsely. The layer's index holds the pooled key of
// every complete block (bf16 [dimension]): a lane's index table holds an
// entry per logical page (SplashKvPage), the page's blocks' keys
// `index_offset` bytes into the entry's extent, after those of the pages
// before it in the extent (ops::BlockIndex). A block pools the raw keys of
// its tokens: a step's rows' (bf16 [row][dimension], which the project
// writes) and, before a lane's first row, the lane's pending keys, those of
// its open block (bf16 [slot][dimension] of BLOCK_INDEXER_PENDING slots, the
// token at position b * BLOCK_INDEXER_BLOCK + slot of the block b that is not
// complete), which its state keeps.
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
// An indexer dispatch serves `lanes` lanes, each through its own index table:
// a verify step's lanes, or a prefill sequence as one. Lane l's `rows` rows
// follow the rows of the lanes before it in the queries, the keys, the scores
// and the bitmaps, the first at logical position first_position[l] +
// row_offset. A dispatch of one lane selects a long sequence's rows in
// batches of the score scratch's rows: the batch from row `row_offset` of its
// queries and bitmaps, scored into the scratch's first rows. The pool pools
// lane l's counts[l] blocks, its records lane_records records apart.
struct BlockIndexerParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t counts[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lane_records;
  uint32_t row_offset;
  uint32_t top_blocks;
  uint32_t mask_words;   // 32-bit words of a row's block bitmap
  uint32_t score_stride; // scores per row of the score scratch
  uint32_t index_offset; // the layer's index in every index entry's extent, in bytes
};

static_assert(sizeof(BlockIndexerParams) == 32 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "block indexer parameters are the same size on both sides");

// The pending keys of each of `lanes` lanes after the first retained[l] of
// its `rows` rows of a step from first_position[l], in `layers` layers: the
// keys of the rows' last block that they do not complete, from the step's
// keys (`layer_rows` rows between two layers') and, before the lane's first
// row, its pending keys before the step. block_indexer_pending takes the
// retained rows from these parameters (a prefill sequence's, or those before
// a state it captures), block_indexer_commit from the counts a verify wrote.
struct BlockIndexerPendingParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t layer_rows;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t retained[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(BlockIndexerPendingParams) == 12 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "block indexer pending parameters are the same size on both sides");

// One block to pool: its first token's logical position and RoPE position
// (t, h, w; text repeats one), to which its pooled key is rotated.
struct BlockIndexerBlock {
  uint32_t position;
  uint32_t rope[3];
};

static_assert(sizeof(BlockIndexerBlock) == 16,
              "block indexer blocks are 16 bytes on both sides");
