#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"
#include "metal/abi/VerifyTree.h"

// One full-attention layer's paged KV. Every current row is written directly
// into a page slot before attention, and every row reads its visible history
// from the same paged representation. A decode step's rows sit at the slots
// of their rows of the lane's block; acceptance moves the accepted path's rows
// to the slots of their depths (SplashKvPathCommitParams), nothing for a
// chain's, and advances committed_tokens; the next command overwrites the
// slots past them.
// Page tables hold one SplashKvPage per logical page, and kv places the layer
// in the pool's extents; the host sets it for each layer it encodes.

// The parts a dense paged-attention tile of a GQA group of `group` query heads
// per KV head runs in (kernels/common/paged_attention_tile.h
// splash_attention_tile_part): the fewest that divide its eight rows and leave
// each at most 64 fused rows, the 256-thread tile's. A group of up to eight
// is one part, sixteen two, thirty-two four. Two 64-row parts of a group of
// sixteen run long prefill and verify 20-40% faster than one 128-row tile of
// 512 threads: both parts read the same pages, the second mostly from cache,
// and the smaller threadgroups keep more of the GPU busy. More parts are
// slower again at long histories.
static_assert(SPLASH_ATTENTION_TILE_ROWS == 8, "a tile's parts divide its eight rows");
#define SPLASH_ATTENTION_TILE_PARTS(group) \
  ((group) <= 8u ? 1u : (group) <= 16u ? 2u : (group) <= 32u ? 4u : 8u)

// Sparse attention (kernels/shared/sparse_attention.metal,
// ops::AttentionBlocks): a query row attends only to the blocks of
// SPLASH_ATTENTION_BLOCK_TOKENS tokens its block mask names, within its causal
// history; the mask is per row a bitmap over the history's blocks, block b at
// bit b % 32 of word b / 32. A tile's rows' masks become its block list: in
// ascending order each block of the tile's history that some row attends, as
// block << SPLASH_ATTENTION_BLOCK_ROW_BITS | a bit per row of the tile that
// attends it, in lists of block_stride entries with their counts apart. A
// sequence's sparse tile t reads list block_list + t; without a stride a tile
// attends to every block of its history.
#define SPLASH_ATTENTION_BLOCK_TOKENS 4u
#define SPLASH_ATTENTION_BLOCK_ROW_BITS SPLASH_ATTENTION_TILE_ROWS
// A sparse split takes at least SPLASH_SPARSE_SPLIT_BLOCKS entries of its
// tile's list, and a list spreads over at least SPLASH_SPARSE_MINIMUM_SPLITS
// splits while it gives each a partition unit, so that a lane's split count
// follows its own list, not its history.
#define SPLASH_SPARSE_SPLIT_BLOCKS 48u
#define SPLASH_SPARSE_MINIMUM_SPLITS 32u

static_assert(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS / SPLASH_ATTENTION_BLOCK_TOKENS <
                  1u << (32 - SPLASH_ATTENTION_BLOCK_ROW_BITS),
              "A list entry's bits above its rows' hold any block");

// A list's blocks may sit outside the tile's page table (ops::AttentionBlocks
// locations, the blocks a spilled request's hot block cache holds), which the
// located split reads: a location per entry, in an array of the lists'
// shape, names a page of the locations' own page table (location >>
// SPLASH_ATTENTION_LOCATION_BLOCK_BITS) and the block's place in it (the low
// bits), or is SPLASH_ATTENTION_PAGED for a block whose page the page table
// holds, at its own place. The locations' pages hold the layer where the
// dispatch's `kv` places it, as the pool's do.
#define SPLASH_ATTENTION_PAGED 0xffffffffu
#define SPLASH_ATTENTION_LOCATION_BLOCK_BITS 3u

static_assert(SPLASH_TARGET_KV_BLOCK_TOKENS / SPLASH_ATTENTION_BLOCK_TOKENS ==
                  1u << SPLASH_ATTENTION_LOCATION_BLOCK_BITS,
              "A location's low bits hold a block's place in its page");

// A command's rows (ops::AttentionRows): `rows` packed rows, sequence s's
// rows entry s of the command's layout (SplashLaneRows: its first row and its
// rows), which a prompt's host writes and a decode step's selector writes on
// the device. Every attention kernel reads a sequence's rows there, and an
// entry past its sequence's row bound or the command's rows holds none, so
// that its sequence runs nothing, as the host checks a layout it writes.

// The block lists of up to SPLASH_MAXIMUM_BATCH_WIDTH sequences of a
// command's `rows` rows, in tiles of `tile_rows` rows: sequence s's rows,
// entry `sequence` of the command's layout, at most row_bound of them, whose
// block masks are the masks' rows of its packed rows, `mask_words` words
// each, the first at position `first_position`, its last tile holding the
// rest; its tile t's entries go to list `list` + t, `stride` entries each,
// and a tile past its rows lists nothing. A threadgroup of
// SPLASH_ATTENTION_BLOCK_LIST_THREADS threads lists one tile.
#define SPLASH_ATTENTION_BLOCK_LIST_THREADS 256u
struct SplashAttentionBlockListSequence {
  uint32_t sequence;
  uint32_t row_bound;
  uint32_t first_position;
  uint32_t list;
};

struct SplashAttentionBlockListParams {
  uint32_t tile_rows;
  uint32_t mask_words;
  uint32_t stride;
  uint32_t sequences;
  uint32_t rows;
  SplashAttentionBlockListSequence sequence[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashAttentionBlockListParams) == 20 + 16 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Block list parameters are the same size on both sides");

// One sequence of an attention layer's dispatches (ops::PagedAttention): at
// most row_bound rows, at positions from committed_tokens, its entry of the
// command's layout from row layout_row of it on (0, or the first row of its
// last tile, ops::PagedAttention::lastTiles); staged from row stage_row on of
// each staging, a KV head's rows chunk_stride apart (queries and attention
// rows [KV head][row][query head in group][dimension], keys [KV head][row]
// [dimension], values [KV head][dimension][row]), and reading its history
// through the dispatch's page table `s` (its place among the dispatch's
// sequences), page_table_entries long. Its bound's tiles of
// SPLASH_ATTENTION_TILE_ROWS rows are the dispatch's from first_tile on, and
// a tile past its rows writes nothing; each splits its pages into
// split_count balanced runs, whose partials take the slots from first_slot
// (in a tile's parts) on, and a sparse one, tile t of the sequence, attends
// through list block_list + t of block_stride entries. Split and reduce use
// the same count and partition rule. The bound sizes everything the host
// lays out for a sequence (its staging, page table, tiles, partial slots and
// lists), so that rows the device chooses within it need nothing else.
struct SplashAttentionSequence {
  uint32_t committed_tokens;
  uint32_t row_bound;
  uint32_t layout_row;
  uint32_t stage_row;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t split_count;
  uint32_t first_tile;
  uint32_t first_slot;
  uint32_t block_stride;
  uint32_t block_list;
};

static_assert(sizeof(SplashAttentionSequence) == 44,
              "Attention sequences are 44 bytes on both sides");

// A dispatch's sequences of a command of `rows` packed rows, in tile order,
// the command's from first_sequence on (its sequence s is entry
// first_sequence + s of the command's layout): `tiles`, their tiles' sum,
// sizes the split and reduce grids, and the per-row kernels (q/k projection,
// store, gate) take a grid of (head or element group, row in sequence,
// sequence), each threadgroup reading its sequence's layout entry once. A
// threadgroup of a row past its sequence's rows returns. Their pages hold
// the layer where `kv` places it in their
// extents, the same for every sequence of a dispatch, which the GPU reads once
// for the whole dispatch: with a placement per sequence, the prompt tiles of
// two KV heads ran up to 7% slower (M5 Max).
// With `trees` nonzero each sequence is a decode lane's block of rows from
// its anchor, at committed_tokens, a tree (metal/abi/VerifyTree.h) whose
// row r's place is the tree rows' entry of its packed row: its K/V sit at the
// slot of its row, as a chain's, and its position is the anchor's plus its
// depth, so it attends the history of that position's window and, of the
// block's rows, its ancestors and itself; a tree's tiles read their pages
// from the anchor's window. Without, rows attend causally.
// With `last_rows` nonzero the gate writes one row a sequence, its first
// (the sequence's last row: ops::AttentionOutput::Last), sequence s's as
// output row s; without, every row as its packed row. A gate group of
// SPLASH_ATTENTION_GATE_THREADS threads takes a thread's element (two, for
// the out-projection's table) of one row.
#define SPLASH_ATTENTION_GATE_THREADS 256u
struct SplashAttentionParams {
  uint32_t sequences;
  uint32_t first_sequence;
  uint32_t rows;
  uint32_t tiles;
  uint32_t trees;
  uint32_t last_rows;
  SplashKvLayer kv;
  SplashAttentionSequence sequence[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashAttentionParams) == 32 + 44 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Attention parameters are the same size on both sides");

// The commit of each lane's accepted path after a decode step
// (ops::PagedAttention::addPathCommit): a step stores a lane's rows at the
// slots of their rows of its block, committed_tokens[lane] plus the row, so
// that siblings never collide, and the path's row at depth k then moves to
// the slot of its depth, committed_tokens[lane] plus k, in each of the
// pool's `regions` layer regions in extents of extent_pages pages.
struct SplashKvPathCommitParams {
  uint32_t extent_pages;
  uint32_t regions;
  uint32_t committed_tokens[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SplashKvPathCommitParams) == 8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Path commit parameters are the same size on both sides");

// A prefill whose history has pages that hold only their index reads one
// layer's history staged (kernels/shared/kv_stage.metal, ops::PagedAttention::
// addStageLayer): pages of the pool's layer from first_page on, a threadgroup
// each, reached through a page table, copied to the same pages of a staging
// table, whose storage holds that one layer; a pool entry that is
// SPLASH_KV_PAGE_ABSENT is left to the reads that fill it. data_bytes and
// scale_bytes are a page's bytes of one tensor of the layer (zero scale bytes
// for BF16).
struct SplashKvStageParams {
  SplashKvLayer pool;
  SplashKvLayer staged;
  uint32_t first_page;
  uint32_t data_bytes;
  uint32_t scale_bytes;
};

static_assert(sizeof(SplashKvStageParams) == 28,
              "KV stage parameters are 28 bytes on both sides");
