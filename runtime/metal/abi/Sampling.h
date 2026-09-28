#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t row_offset;
  uint32_t top_k;
  float temperature;
  float top_p;
  uint32_t mask_words;
  uint32_t mask_row_offset;
  uint32_t constrained;
};

static_assert(sizeof(TargetSamplingParams) == 32,
              "Target sampling parameters are 32 bytes on both sides");

struct TargetSamplingBatchParams {
  uint32_t vocabulary;
  uint32_t rows_per_lane;
  uint32_t lanes;
  uint32_t mask_words;
  uint32_t top_k[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  float top_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t constrained_mask;
};

static_assert(sizeof(TargetSamplingBatchParams) == 68,
              "Batched target sampling parameters are 68 bytes on both sides");

// Rows [row_offset, row_offset + rows) of a [rows][vocabulary] logit buffer,
// rows_per_lane rows per lane. Lanes set in lane_mask ignore end-of-sequence:
// both stop token logits become -inf before any target sampling reads them.
struct SuppressStopParams {
  uint32_t vocabulary;
  uint32_t row_offset;
  uint32_t rows;
  uint32_t rows_per_lane;
  uint32_t lane_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
};

static_assert(sizeof(SuppressStopParams) == 28,
              "Stop suppression parameters are 28 bytes on both sides");

struct SelectorBatchParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t sampling_mask;
  uint32_t vocabulary;
};

static_assert(sizeof(SelectorBatchParams) == 44,
              "Draft selector parameters are 44 bytes on both sides");

struct VerifyInputBatchParams {
  uint32_t lanes;
  uint32_t vocabulary;
};

static_assert(sizeof(VerifyInputBatchParams) == 8,
              "Verify input parameters are 8 bytes on both sides");

// The lane view of AcceptBatchParams. No host bytes flow through it: the
// acceptance kernel builds one per lane from the batched struct.
struct AcceptParams {
  uint32_t remaining;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
};

static_assert(sizeof(AcceptParams) == 12,
              "Acceptance lane parameters are 12 bytes on both sides");

struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t lanes;
  uint32_t sampling_mask;
};

static_assert(sizeof(AcceptBatchParams) == 32,
              "Batched acceptance parameters are 32 bytes on both sides");
