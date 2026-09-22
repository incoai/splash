#pragma once

// C/Metal ABI constants shared by host and shader compilation. Startup
// static assertions check the corresponding model and operator contracts.
#define SPLASH_DRAFT_QUERY_ROWS 8u
#define SPLASH_DRAFT_PROPOSAL_TOKENS 7u
#define SPLASH_TARGET_VERIFY_ROWS 8u
#define SPLASH_MAXIMUM_CONTEXT_TOKENS 262144u
#define SPLASH_SPECULATIVE_SCRATCH_TOKENS                                  \
  (SPLASH_TARGET_VERIFY_ROWS - 1u)
#define SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS                                  \
  (SPLASH_MAXIMUM_CONTEXT_TOKENS + SPLASH_SPECULATIVE_SCRATCH_TOKENS)
#define SPLASH_MAXIMUM_BATCH_WIDTH 4u
#define SPLASH_PREFILL_TOKEN_BUDGET 2048u
#define SPLASH_DRAFT_SLIDING_WINDOW 2048u
#define SPLASH_TARGET_KV_BLOCK_TOKENS 32u
#define SPLASH_PREFILL_ATTENTION_TILE_ROWS 8u
#define SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
#define SPLASH_VERIFY_ATTENTION_SPLITS 32u
// Verify attention starts from the configured split count and adds one split
// per this many visible Page32 blocks, capped at the maximum that sizes the
// partial workspace.
#define SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT 16u
#define SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#define SPLASH_TARGET_SAMPLING_SHARDS 16u
#define SPLASH_DRAFT_SAMPLING_SHARDS 8u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define SPLASH_GDN_SCAN_STATE_ROWS 16u
#define SPLASH_GDN_SCAN_THREADS 128u
// Widest row norm_rms_staged holds in threadgroup memory, and the thread count
// its load and store loops stride by. shared/normalization.metal sizes the
// staging array from the width and ops/Normalization.cpp selects the kernel
// and dispatch width from the same two literals.
#define SPLASH_STAGED_NORM_WIDTH 5120u
#define SPLASH_STAGED_NORM_THREADS 1024u
#define SPLASH_ALLOCATION_EXTENT_TARGET_BYTES (128ull * 1024ull * 1024ull)
