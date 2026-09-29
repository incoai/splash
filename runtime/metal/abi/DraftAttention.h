#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct DraftConvBatchParams {
  uint32_t groups;
  uint32_t finish;
  uint32_t lanes;
};

static_assert(sizeof(DraftConvBatchParams) == 12,
              "Draft convolution parameters are 12 bytes on both sides");

struct DraftQkvBatchParams {
  uint32_t groups;
  uint32_t lanes;
};

static_assert(sizeof(DraftQkvBatchParams) == 8,
              "Draft QKV parameters are 8 bytes on both sides");

// Draft and target cache pages share token boundaries. Compute tiles are
// independent of storage geometry. Each address points to one layer's
// contiguous K/V pair, with the key tensor followed by the value tensor.
#define SPLASH_DRAFT_PAGE_TOKENS SPLASH_TARGET_KV_BLOCK_TOKENS
#define SPLASH_DRAFT_PAGE_COUNT                                                \
  (SPLASH_DRAFT_SLIDING_WINDOW / SPLASH_DRAFT_PAGE_TOKENS)
struct DraftKvAddresses {
  uint64_t keys[SPLASH_DRAFT_PAGE_COUNT];
};

struct DraftAttentionBatchParams {
  uint32_t cache_stride;
  uint32_t splits;
  uint32_t lanes;
  uint32_t cache_length[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t paged;
  DraftKvAddresses pages[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftAttentionBatchParams) == 2080,
              "Draft attention parameters are 2080 bytes on both sides");

struct DraftContextParams {
  uint32_t tokens;
  uint32_t cache_stride;
  uint32_t start_position;
  uint32_t paged;
  DraftKvAddresses pages;
};

static_assert(sizeof(DraftContextParams) == 528,
              "Draft context prefill parameters are 528 bytes on both sides");

struct DraftContextBatchParams {
  uint32_t cache_stride;
  uint32_t lanes;
  uint32_t start_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t paged;
  uint32_t reserved;
  DraftKvAddresses pages[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextBatchParams) == 2080,
              "Draft context commit parameters are 2080 bytes on both sides");

struct CaptureParams {
  uint32_t rows;
  uint32_t slot;
  uint32_t source_start;
  uint32_t destination_start;
  uint32_t hidden_width;
  uint32_t target_width;
};

static_assert(sizeof(CaptureParams) == 24,
              "Target hidden capture parameters are 24 bytes on both sides");

struct LastHiddenRowsParams {
  uint32_t rows;
  uint32_t width;
};

static_assert(sizeof(LastHiddenRowsParams) == 8,
              "Last hidden row parameters are 8 bytes on both sides");
