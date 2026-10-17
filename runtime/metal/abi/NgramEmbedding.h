#pragma once

// Parameters of the n-gram embedding kernels (kernels/shared/ngram_embedding.metal,
// ops/NgramEmbedding.cpp), Qwen3.8-Flash-Next's PLE: the hashes of each
// token's 2- to n-gram select rows of a table, whose key and value
// projections gate a value per stream of a hyper-connected residual
// (metal/abi/HyperConnection.h), added to the streams through a dilated
// convolution.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/HyperConnection.h"

// The gate (ngram_embedding_gate) and dilated convolution
// (ngram_embedding_conv) of `rows` rows of one sequence over `streams`
// streams of `hidden` values: `taps` taps `dilation` rows apart, whose
// history is the (taps - 1) * dilation rows before the first.
// ngram_embedding_conv writes the history after its rows when write_history
// is set; ngram_embedding_history writes the history after a sequence's first
// `rows` rows, and ngram_embedding_commit the history after a lane's retained
// rows of a verify step, a count the GPU wrote.
struct NgramEmbeddingParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t taps;
  uint32_t dilation;
  uint32_t write_history;
};

static_assert(sizeof(NgramEmbeddingParams) == 24,
              "n-gram embedding parameters are 24 bytes on both sides");

// The table rows (the n-gram hash) of each lane's SPLASH_TARGET_VERIFY_ROWS
// verify rows on the GPU (ngram_embedding_hash), from the rows' tokens and
// the ngram - 1 tokens before a lane's first row, nearest first
// (NGRAM_EMBEDDING_NONE for none). The hash mixes the n-grams
// of 2 to `ngram` tokens, at most NGRAM_EMBEDDING_MAXIMUM_NGRAM, with a
// multiplier per n-gram position; each n-gram's heads_per_ngram heads take
// its hash modulo their row counts (`vocabularies`) past their `offsets`.
#define NGRAM_EMBEDDING_NONE 0xffffffffu
#define NGRAM_EMBEDDING_MAXIMUM_NGRAM 3u
#define NGRAM_EMBEDDING_MAXIMUM_HEADS 16u
struct NgramHashParams {
  uint32_t rows;
  uint32_t ngram;
  uint32_t heads_per_ngram;
  uint32_t eos;
  uint64_t multipliers[NGRAM_EMBEDDING_MAXIMUM_NGRAM];
  uint32_t vocabularies[NGRAM_EMBEDDING_MAXIMUM_HEADS];
  uint32_t offsets[NGRAM_EMBEDDING_MAXIMUM_HEADS];
};

static_assert(sizeof(NgramHashParams) == 168,
              "n-gram hash parameters are 168 bytes on both sides");
