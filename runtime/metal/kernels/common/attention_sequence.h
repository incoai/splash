#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// A dispatch's sequences (SplashAttentionParams) are in row order: the one
// that holds the dispatch's row `row` or tile `tile` is the last whose first
// row or tile is at most it. A sequence's place among them names its page
// table among the dispatch's bindings (SPLASH_LANE_BINDING). The lookup is
// three selects over every entry, not a loop to params.sequences: a tile
// whose sequence such a loop found ran its page loop up to a quarter slower
// (the 64-fused-row tiles of two KV heads).
inline uint splash_attention_sequence_of_row(constant SplashAttentionParams &params, uint row) {
  uint sequence = 0;
#pragma unroll
  for (uint s = 1; s < SPLASH_MAXIMUM_BATCH_WIDTH; ++s)
    sequence = s < params.sequences && row >= params.sequence[s].first_row ? s : sequence;
  return sequence;
}

inline uint splash_attention_sequence_of_tile(constant SplashAttentionParams &params, uint tile) {
  uint sequence = 0;
#pragma unroll
  for (uint s = 1; s < SPLASH_MAXIMUM_BATCH_WIDTH; ++s)
    sequence = s < params.sequences && tile >= params.sequence[s].first_tile ? s : sequence;
  return sequence;
}

// Whether a sequence's rows, staging and pages, and the layer the
// dispatch's pages hold, are within the kernels' bounds, as the host checks
// them (ops::PagedAttention): the store's contract, and with its split count
// the page loop's.
inline bool splash_attention_rows_valid(constant SplashAttentionParams &params, constant SplashAttentionSequence &s) {
  return s.rows > 0 && s.rows <= SPLASH_PREFILL_TOKEN_BUDGET && s.chunk_stride >= s.rows &&
         s.chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET && s.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         ulong(s.committed_tokens) + s.rows <= ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         s.page_table_entries >= (s.committed_tokens + s.rows + SplashKvPageTokens - 1) / SplashKvPageTokens &&
         params.kv.extent_pages > 0;
}
