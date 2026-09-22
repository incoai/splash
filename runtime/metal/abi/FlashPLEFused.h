#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// All 128 immutable source shards have the same logical Q4/G32 row layout.
// Their pointers live in one Tier 2 argument buffer: W ids 0..127, S ids
// 128..255, and B ids 256..383. No table payload is copied or transformed.
struct FlashPLEFusedParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t eos_token;
  uint32_t vocabulary_size;
  uint32_t shard_count;
  uint32_t reserved0;
  uint64_t shard_rows;
  uint64_t weight_row_stride_bytes;
  uint64_t parameter_row_stride_bytes;
  uint64_t table_rows;
};

static_assert(sizeof(FlashPLEFusedParams) == 56,
              "Flash PLE fused parameters are 56 bytes");
