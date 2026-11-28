#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// A dispatch's sequences (SplashAttentionParams) are in tile order: the one
// that holds the dispatch's tile `tile` is the last whose first tile is at
// most it. A sequence's place among them names its page table among the
// dispatch's bindings (SPLASH_LANE_BINDING) and its entry of the command's
// layout. The lookup is seven selects over every entry, not a loop to
// params.sequences: a tile whose sequence such a loop found ran its page loop
// up to a quarter slower (the 64-fused-row tiles of two KV heads).
inline uint splash_attention_sequence_of_tile(constant SplashAttentionParams &params, uint tile) {
  uint sequence = 0;
#pragma unroll
  for (uint s = 1; s < SPLASH_MAXIMUM_BATCH_WIDTH; ++s)
    sequence = s < params.sequences && tile >= params.sequence[s].first_tile ? s : sequence;
  return sequence;
}

// Whether a sequence's row bound, staging and pages, and the layer the
// dispatch's pages hold, are within the kernels' bounds, as the host checks
// them (ops::PagedAttention): the store's contract, and with its split count
// the page loop's.
inline bool splash_attention_rows_valid(constant SplashAttentionParams &params, constant SplashAttentionSequence &s) {
  return s.row_bound > 0 && s.row_bound <= SPLASH_PREFILL_TOKEN_BUDGET && s.chunk_stride >= s.row_bound &&
         s.chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET && s.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         ulong(s.committed_tokens) + s.row_bound <= ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         s.page_table_entries >= (s.committed_tokens + s.row_bound + SplashKvPageTokens - 1) / SplashKvPageTokens &&
         params.kv.extent_pages > 0;
}

// Dispatch sequence s's rows of the command (ops::AttentionRows): its first
// packed row and its rows, from its layout entry's row layout_row on. An
// entry past the sequence's row bound or the command's rows holds none.
struct SplashAttentionRows {
  uint first;
  uint rows;
};

inline SplashAttentionRows splash_attention_rows(device const SplashLaneRows *layout,
                                                 constant SplashAttentionParams &params, uint s) {
  const SplashLaneRows entry = layout[params.first_sequence + s];
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const uint rows = entry.rows - min(entry.rows, sequence.layout_row);
  if (rows > sequence.row_bound || ulong(entry.first) + entry.rows > params.rows)
    return {0, 0};
  return {entry.first + sequence.layout_row, rows};
}

// Where a reduce writes row `row` of dispatch sequence s (counted from its
// rows' first, SplashAttentionRows): its packed row, the output row it
// writes, its packed row or, with last_rows (SplashAttentionParams), its
// sequence's place among the command's for the sequence's last row alone,
// and whether it writes: a row the sequence holds, and with last_rows its
// last.
struct SplashAttentionOutput {
  ulong packed;
  ulong output;
  bool writes;
};

inline SplashAttentionOutput splash_attention_output(constant SplashAttentionParams &params, uint s,
                                                     SplashAttentionRows rows, uint row) {
  const ulong packed = ulong(rows.first) + row;
  return {packed, params.last_rows ? ulong(params.first_sequence) + s : packed,
          row < rows.rows && (!params.last_rows || row + 1 == rows.rows)};
}
