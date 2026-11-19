#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Row r embeds tokens[r * token_stride] (token_stride 0: contiguous).
struct Q4EmbeddingParams {
  uint32_t rows;
  uint32_t vocabulary_size;
  uint32_t token_stride;
  uint32_t padding;
};

static_assert(sizeof(Q4EmbeddingParams) == 16,
              "Q4 embedding parameters are 16 bytes on both sides");

struct VerifyInputBatchParams {
  uint32_t vocabulary;
};

static_assert(sizeof(VerifyInputBatchParams) == 4,
              "Verify input parameters are 4 bytes on both sides");
