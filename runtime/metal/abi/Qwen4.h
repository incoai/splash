#pragma once

// Parameters of Qwen3.8-Flash-Next's own kernels (kernels/shared/qwen4.metal,
// kernels/shared/qwen4_qsa.metal, ops/Qwen4.cpp): its hyper-connections, its
// PLE n-gram embedding, and the QSA indexer, whose block bitmaps are the
// block masks of its full-attention layers' sparse attention.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"

// Threads of the per-row kernels: one threadgroup per row.
#define QWEN4_ROW_THREADS 256u
// The most streams a mix reads.
#define QWEN4_MAXIMUM_STREAMS 4u
// The mix norm's kernels, compiled for Flash-Next's shape of
// QWEN4_MAXIMUM_STREAMS streams of QWEN4_NORM_HIDDEN values
// (ops::Qwen4::addNorm): qwen4_hyper_norm_quarters, a threadgroup of
// QWEN4_NORM_QUARTER_THREADS threads per row, for an injecting mix of at
// most QWEN4_NORM_QUARTER_ROWS rows, and qwen4_hyper_norm_loaded, a
// threadgroup of QWEN4_ROW_THREADS per row, for the rest.
#define QWEN4_NORM_HIDDEN 2560u
#define QWEN4_NORM_QUARTER_THREADS (QWEN4_MAXIMUM_STREAMS * QWEN4_ROW_THREADS)
#define QWEN4_NORM_QUARTER_ROWS 64u

// A mix over `streams` streams of `hidden` values per row: its norm and
// injection (hyper_norm), its low-rank gate (hyper_low, `rank` values) and
// its average (hyper_mix); or the injection of a block's output (combine) and
// the streams' start (init).
struct Qwen4HyperParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t rank;
  uint32_t inject; // hyper_norm: whether to compute injection weights
};

static_assert(sizeof(Qwen4HyperParams) == 20,
              "Qwen4 hyper-connection parameters are 20 bytes on both sides");

// The PLE gate (ple_gate) and dilated convolution (ple_conv) of `rows` rows
// of one sequence: `taps` taps `dilation` rows apart, whose history is the
// (taps - 1) * dilation rows before the first. ple_conv writes the history
// after its rows when write_history is set; ple_history writes the history
// after a sequence's first `rows` rows, and ple_commit the history after a
// lane's retained rows of a verify step, a count the GPU wrote.
struct Qwen4PleParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t taps;
  uint32_t dilation;
  uint32_t write_history;
};

static_assert(sizeof(Qwen4PleParams) == 24,
              "Qwen4 PLE parameters are 24 bytes on both sides");

// QSA (Qwen Sparse Attention) of a full-attention layer: the indexer's
// QWEN4_QSA_HEADS query heads and one key head of QWEN4_QSA_DIMENSION values;
// blocks of QWEN4_QSA_BLOCK tokens, those of the sparse attention's block
// masks (metal/abi/PagedAttention.h); `top_blocks` blocks per query row past
// which the row attends sparsely. Each KV page also holds, per layer, the raw
// indexer keys of its SPLASH_TARGET_KV_BLOCK_TOKENS tokens (bf16 [token]
// [dimension]) and the pooled keys of its blocks (bf16 [block][dimension]) in
// its extent's index regions (ops/PageStorage indexPlacement), which kernels
// reach through the page's table entry and the regions' byte offsets.
#define QWEN4_QSA_DIMENSION 128u
#define QWEN4_QSA_HEADS 4u
#define QWEN4_QSA_BLOCK SPLASH_ATTENTION_BLOCK_TOKENS
// The score kernel's tiles: QWEN4_QSA_SCORE_ROWS rows by
// QWEN4_QSA_SCORE_BLOCKS blocks, a threadgroup of QWEN4_QSA_SCORE_THREADS
// threads each.
#define QWEN4_QSA_SCORE_ROWS 8u
#define QWEN4_QSA_SCORE_BLOCKS 32u
#define QWEN4_QSA_SCORE_THREADS 256u
// The select kernel's threads, a threadgroup per row.
#define QWEN4_QSA_SELECT_THREADS 1024u
// A QSA dispatch serves `lanes` lanes, each through its own page table: a
// verify step's lanes, or a prefill sequence as one. Lane l's `rows` rows
// follow the rows of the lanes before it in the queries, the scores and the
// bitmaps, the first at logical position first_position[l] + row_offset. A
// dispatch of one lane selects a long sequence's rows in batches of the score
// scratch's rows: the batch from row `row_offset` of its queries and bitmaps,
// scored into the scratch's first rows. The pool pools lane l's counts[l]
// blocks, its records lane_records records apart.
struct Qwen4QsaParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t counts[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lane_records;
  uint32_t row_offset;
  uint32_t top_blocks;
  uint32_t mask_words;    // 32-bit words of a row's block bitmap
  uint32_t score_stride;  // scores per row of the score scratch
  uint32_t keys_offset;   // the layer's raw-key region in every extent, in bytes
  uint32_t pooled_offset; // the layer's pooled-key region in every extent, in bytes
};

static_assert(sizeof(Qwen4QsaParams) == 36 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Qwen4 QSA parameters are the same size on both sides");

// One block to pool: its first token's logical position and RoPE position
// (t, h, w; text repeats one), to which its pooled key is rotated.
struct Qwen4QsaBlock {
  uint32_t position;
  uint32_t rope[3];
};

static_assert(sizeof(Qwen4QsaBlock) == 16,
              "Qwen4 QSA blocks are 16 bytes on both sides");

// The mean of each of `rows` rows of `streams` streams of `hidden` fp32
// values (qwen4_stream_mean), written as bf16 at column `column` of row
// `row` + r of rows of `stride` values.
struct Qwen4StreamMeanParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t row;
  uint32_t stride;
  uint32_t column;
};

static_assert(sizeof(Qwen4StreamMeanParams) == 24,
              "Qwen4 stream mean parameters are 24 bytes on both sides");

// The PLE rows (the n-gram hash, model/Qwen4Exp.hpp) of each lane's
// SPLASH_TARGET_VERIFY_ROWS verify rows on the GPU, from the rows' tokens and
// the ngram - 1 tokens before a lane's first row, nearest first
// (QWEN4_PLE_NONE for none). The hash mixes the n-grams of 2 to `ngram`
// tokens, at most QWEN4_PLE_MAXIMUM_NGRAM (Flash-Next's are 2- and 3-grams),
// with a multiplier per n-gram position.
#define QWEN4_PLE_NONE 0xffffffffu
#define QWEN4_PLE_MAXIMUM_NGRAM 3u
#define QWEN4_PLE_MAXIMUM_HEADS 16u
struct Qwen4PleHashParams {
  uint32_t rows;
  uint32_t ngram;
  uint32_t heads_per_ngram;
  uint32_t eos;
  uint64_t multipliers[QWEN4_PLE_MAXIMUM_NGRAM];
  uint32_t vocabularies[QWEN4_PLE_MAXIMUM_HEADS];
  uint32_t offsets[QWEN4_PLE_MAXIMUM_HEADS];
};

static_assert(sizeof(Qwen4PleHashParams) == 168,
              "Qwen4 PLE hash parameters are 168 bytes on both sides");
