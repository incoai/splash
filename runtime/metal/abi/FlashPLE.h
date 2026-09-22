#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FlashPLEHashParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t heads_per_ngram;
  uint32_t eos_token;
  uint32_t vocabulary_size;
  uint32_t reserved0;
  uint64_t table_rows;
};

// Eight original, homogeneous affine shards fit the ordinary Metal buffer
// argument limit. Indices remain global; only the matching shard reads a row.
struct FlashPLEGatherParams {
  uint32_t rows;
  uint32_t heads;
  uint32_t head_width;
  uint32_t shard_count;
  uint64_t first_row;
  uint64_t shard_rows;
  uint64_t weight_row_stride_bytes;
  uint64_t parameter_row_stride_bytes;
  uint64_t table_rows;
};

struct FlashPLEPostParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t width;
  uint32_t streams;
  uint32_t flags;
  uint32_t state_rows;
  uint32_t taps;
  uint32_t dilation;
};

struct FlashPLEPrefixParams {
  FlashPLEPostParams geometry;
  uint32_t vocabulary_size;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

static_assert(sizeof(FlashPLEHashParams) == 32,
              "Flash PLE hash parameters are 32 bytes");
static_assert(sizeof(FlashPLEGatherParams) == 56,
              "Flash PLE gather parameters are 56 bytes");
static_assert(sizeof(FlashPLEPostParams) == 32,
              "Flash PLE post parameters are 32 bytes");
static_assert(sizeof(FlashPLEPrefixParams) == 48,
              "Flash PLE prefix parameters are 48 bytes");
