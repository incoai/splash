#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The draft attention the kernels are compiled for, the only one
// ops::DraftAttention accepts: 32 query heads over 8 KV heads of 128
// dimensions, so a q|k|v row holds 6144 values and an attention row 4096.
// The attention core runs a KV head's four query heads over the eight query
// rows together, as 32 attention rows.
#define SPLASH_DRAFT_QUERY_HEADS 32u
#define SPLASH_DRAFT_KV_HEADS 8u
#define SPLASH_DRAFT_HEAD_DIMENSION 128u
#define SPLASH_DRAFT_ATTENTION_WIDTH                                           \
  (SPLASH_DRAFT_QUERY_HEADS * SPLASH_DRAFT_HEAD_DIMENSION)
#define SPLASH_DRAFT_QKV_WIDTH                                                 \
  (SPLASH_DRAFT_ATTENTION_WIDTH +                                              \
   2u * SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_HEAD_DIMENSION)
#define SPLASH_DRAFT_ATTENTION_ROWS                                            \
  (SPLASH_DRAFT_QUERY_HEADS / SPLASH_DRAFT_KV_HEADS * SPLASH_DRAFT_QUERY_ROWS)

// The dynamic convolutions of a draft layer (draft_conv), each run in two
// stages, before its attention or MLP and after it with the residual: a
// stage weighs a channel's row and the rows before it, its taps, each by a
// base weight plus the dynamic weight of the channel's group of 16
// channels. The draft projects every row's dynamic weights, for each stage,
// tap and group of a `hidden`-wide layer.
#define SPLASH_DRAFT_CONVOLUTION_STAGES 2u
#define SPLASH_DRAFT_CONVOLUTION_TAPS 2u
#define SPLASH_DRAFT_CONVOLUTION_GROUP 16u
inline constexpr uint32_t draft_dynamic_width(uint32_t hidden, uint32_t taps);
inline constexpr uint32_t draft_dynamic_width(uint32_t hidden) {
  return draft_dynamic_width(hidden, SPLASH_DRAFT_CONVOLUTION_TAPS);
}
inline constexpr uint32_t draft_dynamic_width(uint32_t hidden, uint32_t taps) {
  return SPLASH_DRAFT_CONVOLUTION_STAGES * taps *
         (hidden / SPLASH_DRAFT_CONVOLUTION_GROUP);
}

#ifndef __METAL_VERSION__
// The draft geometries the kernels are compiled for (decode/draft.metal and
// prefill/draft_context.metal), which a draft's configuration must match in
// full: a draft of another geometry is refused, as one the kernels cannot
// serve. A variant's kernels carry its suffix after the shipped kernels'
// names, and a KV head's query heads are shared among whole head groups of
// the rows one threadgroup attends, as the target's attention kernels share
// a KV head's query heads.
struct SplashDraftVariant {
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t head_dimension;
  uint32_t window;
  uint32_t convolution_taps;
  uint32_t splits;
  uint32_t head_groups;
  // The proposal the draft was trained for: the block's rows a cycle drafts
  // (the target verifies as many) and the selector's kept candidates, which
  // the variant's kernels are instantiated with.
  uint32_t rows;
  uint32_t candidates;
  const char *suffix;
};
inline constexpr SplashDraftVariant SPLASH_DRAFT_VARIANTS[] = {
    // The shipped drafts: 32 query heads over 8 KV heads of 128 dimensions,
    // a 2048-slot ring, two-tap convolutions, four split threadgroups, one
    // head group per KV head.
    {SPLASH_DRAFT_QUERY_HEADS, SPLASH_DRAFT_KV_HEADS, SPLASH_DRAFT_HEAD_DIMENSION,
     SPLASH_DRAFT_SLIDING_WINDOW, SPLASH_DRAFT_CONVOLUTION_TAPS,
     SPLASH_DRAFT_ATTENTION_SPLITS, 1, SPLASH_DRAFT_QUERY_ROWS,
     SPLASH_DRAFT_CANDIDATES, ""},
    // Ripple's draft: 80 query heads over the 8 KV heads, a 4096-slot ring,
    // four-tap convolutions, 16-row proposals of 32 candidates (the draft
    // was trained so), head groups of two query heads (five 32-row
    // threadgroups per KV head), eight split threadgroups.
    {80, 8, 128, 4096, 4, 8, 5, 16, 32, "_kv8_h2_d128_w4096_s8"},
};
#endif

// The batched draft kernels' grids cover exactly the dispatch's lanes:
// per-lane arrays hold those lanes, and entries past them are zero and unread.

// A convolution stage (finish: the second, with the residual) of a draft
// `hidden` wide, a whole number of convolution groups.
struct DraftConvBatchParams {
  uint32_t finish;
  uint32_t hidden;
};

static_assert(sizeof(DraftConvBatchParams) == 8,
              "Draft convolution parameters are 8 bytes on both sides");

// The attention core's split count (SPLASH_DRAFT_ATTENTION_SPLITS) and its
// rings' slots per KV head (SPLASH_DRAFT_SLIDING_WINDOW) are compiled in.
// value_stride, the values ring's stride between head dimensions, is the
// window too (the op passes nothing else) but stays a run-time value: with it
// compiled into the value tiles, the split kernel returns wrong rows at
// random, and reports no error, on an M5 Max under full
// MTL_SHADER_VALIDATION; without validation its output is bitwise that of
// the run-time stride.
struct DraftAttentionBatchParams {
  uint32_t value_stride;
  uint32_t lanes;
  uint32_t cache_length[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftAttentionBatchParams) == 24,
              "Draft attention parameters are 24 bytes on both sides");

struct DraftContextParams {
  uint32_t tokens;
  uint32_t start_position;
};

static_assert(sizeof(DraftContextParams) == 8,
              "Draft context prefill parameters are 8 bytes on both sides");

struct DraftContextBatchParams {
  uint32_t start_position[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DraftContextBatchParams) == 16,
              "Draft context commit parameters are 16 bytes on both sides");

// A context window holds, per ring slot, the draft's context row, from which
// every layer's ring keys and values are computed: 4-bit codes over groups of
// SPLASH_DRAFT_CONTEXT_GROUP values, all slots' codes first, then each
// group's fp16 scale and minimum. Its kernels run one simdgroup per row and
// group, SPLASH_DRAFT_CONTEXT_SIMDGROUPS to a threadgroup.
#define SPLASH_DRAFT_CONTEXT_GROUP 64u
#define SPLASH_DRAFT_CONTEXT_SIMDGROUPS 8u
inline constexpr uint64_t draft_context_codes_bytes(uint32_t width) {
  return uint64_t(SPLASH_DRAFT_SLIDING_WINDOW) * (width / 2);
}
inline constexpr uint64_t draft_context_window_bytes(uint32_t width) {
  return draft_context_codes_bytes(width) +
         uint64_t(SPLASH_DRAFT_SLIDING_WINDOW) * (width / SPLASH_DRAFT_CONTEXT_GROUP) * 4;
}

struct DraftContextWindowParams {
  uint32_t rows;
  uint32_t start_position;
  uint32_t width;
};

static_assert(sizeof(DraftContextWindowParams) == 12,
              "Draft context window parameters are 12 bytes on both sides");
