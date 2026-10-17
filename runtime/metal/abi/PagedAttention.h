#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"

// stride: rows of one KV head's (and one query group's) chunk staging;
// equals the store's chunk_stride.
struct FullPrefillParams {
  uint32_t tokens;
  uint32_t stride;
};

static_assert(sizeof(FullPrefillParams) == 8,
              "Full attention prefill parameters are 8 bytes on both sides");

// Each lane holds SPLASH_TARGET_VERIFY_ROWS rows in a chunk staging of
// SPLASH_VERIFY_CHUNK_STRIDE rows.
struct FullDecodeBatchParams {
  uint32_t lanes;
};

static_assert(sizeof(FullDecodeBatchParams) == 4,
              "Full attention verify parameters are 4 bytes on both sides");

// One full-attention layer's paged KV. Every current row is written directly
// into its final page slot before attention. Prefill and verify both read
// all visible history from the same paged representation. Decode accepts rows
// only by advancing committed_tokens; the next command overwrites rejected
// slots. Page tables hold one SplashKvPage per logical page, and kv places the
// layer in the pool's extents; the host sets it for each layer it encodes.
struct SplashChunkedPrefillParams {
  uint32_t committed_tokens;
  uint32_t chunk_tokens;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  SplashKvLayer kv;
};

static_assert(sizeof(SplashChunkedPrefillParams) == 24,
              "Chunked store parameters are 24 bytes on both sides");

// Sparse attention (kernels/shared/sparse_attention.metal,
// ops::AttentionBlocks): a query row attends only to the blocks of
// SPLASH_ATTENTION_BLOCK_TOKENS tokens its block mask names, within its causal
// history; the mask is per row a bitmap over the history's blocks, block b at
// bit b % 32 of word b / 32. A tile's rows' masks become its block list: in
// ascending order each block of the tile's history that some row attends, as
// block << SPLASH_ATTENTION_BLOCK_ROW_BITS | a bit per row of the tile that
// attends it, in lists of block_stride entries with their counts apart. A
// sparse split reads list block_list (verify: its lane's), or a prefill tile
// list block_list + tile; without a stride a tile attends to every block of
// its history.
#define SPLASH_ATTENTION_BLOCK_TOKENS 4u
#define SPLASH_ATTENTION_BLOCK_ROW_BITS 8u

static_assert(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS / SPLASH_ATTENTION_BLOCK_TOKENS <
                  1u << (32 - SPLASH_ATTENTION_BLOCK_ROW_BITS),
              "A list entry's bits above its rows' hold any block");

// The block lists of up to SPLASH_MAXIMUM_BATCH_WIDTH sequences of rows (a
// verify step's lanes, or a prefill chunk's sequences), in tiles of
// `tile_rows` rows: sequence s's `rows` rows, whose block masks are rows
// `mask_row` on of the masks, `mask_words` words each, the first at position
// `first_position`, its last tile holding the rest; its tile t's entries go
// to list `list` + t, `stride` entries each. A threadgroup of
// SPLASH_ATTENTION_BLOCK_LIST_THREADS threads lists one tile.
#define SPLASH_ATTENTION_BLOCK_LIST_THREADS 256u
struct SplashAttentionBlockListSequence {
  uint32_t mask_row;
  uint32_t rows;
  uint32_t first_position;
  uint32_t list;
};

struct SplashAttentionBlockListParams {
  uint32_t tile_rows;
  uint32_t mask_words;
  uint32_t stride;
  uint32_t sequences;
  SplashAttentionBlockListSequence sequence[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashAttentionBlockListParams) == 16 + 16 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Block list parameters are the same size on both sides");

// Prefill divides each query tile's history into balanced splits: the page
// loop its visible Page32 pages, a sparse split its block list in whole
// partition units. Split and reduce use the same count and partition rule.
struct SplashPrefillAttentionParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  SplashKvLayer kv;
  uint32_t split_count;
  uint32_t block_stride;
  uint32_t block_list;
};

static_assert(sizeof(SplashPrefillAttentionParams) == 36,
              "Prefill attention parameters are 36 bytes on both sides");

// A verify lane attends all its SPLASH_TARGET_VERIFY_ROWS rows, which its
// queries and output hold in SPLASH_VERIFY_CHUNK_STRIDE rows per KV head.
struct SplashVerifyAttentionParams {
  uint32_t committed_tokens;
  uint32_t page_table_entries;
  SplashKvLayer kv;
  // Filled from the plan: this lane's history-scaled split count and the
  // plan-wide slot stride that every lane's partials use.
  uint32_t split_count;
  uint32_t slot_splits;
  // The lane's block list, which its sparse split and reduce read.
  uint32_t block_stride;
  uint32_t block_list;
};

static_assert(sizeof(SplashVerifyAttentionParams) == 32,
              "Verify attention parameters are 32 bytes on both sides");
