#pragma once
#include "FlashDenseCache.h"

// Only checked immutable BF16[640,2560] gate/up operands are accepted. Main
// rows are complete M32 tiles, N128, four SIMD groups, direct device BF16 A.
// This explicit ABI keeps the dense-cache field ordering for qualification,
// while naming the producer that preserves both BF16 dot/SwiGLU boundaries.
struct FlashSharedExpertFusedParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t output_begin;
  uint32_t output_count;
  uint32_t tile_rows;
  uint32_t tile_outputs;
  uint32_t reserved;
};
static_assert(sizeof(FlashSharedExpertFusedParams) == 32,
              "Flash shared expert parameters must match host/shader layouts");
