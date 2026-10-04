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

// Prefill divides each query tile's visible Page32 history into balanced
// splits. The same count and partition rule are used by split and reduce.
struct SplashPrefillAttentionParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  SplashKvLayer kv;
  uint32_t split_count;
};

static_assert(sizeof(SplashPrefillAttentionParams) == 28,
              "Prefill attention parameters are 28 bytes on both sides");

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
};

static_assert(sizeof(SplashVerifyAttentionParams) == 24,
              "Verify attention parameters are 24 bytes on both sides");
