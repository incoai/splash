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
// Rows per KV head (and per query group) of one lane's verify chunk staging:
// one KV block, which holds the lane's SPLASH_TARGET_VERIFY_ROWS rows.
#define SPLASH_VERIFY_CHUNK_STRIDE SPLASH_TARGET_KV_BLOCK_TOKENS
#define SPLASH_PREFILL_ATTENTION_TILE_ROWS 8u
#define SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
// Draft attention deals the live ring tiles of one (lane, KV head)
// round-robin to this many groups, the last of which also attends the eight
// current rows. Fixed rather than derived from the GPU so the combine order,
// and with it the rounding, is the same on every machine and lane count.
#define SPLASH_DRAFT_ATTENTION_SPLITS 4u
#define SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#define SPLASH_TARGET_SAMPLING_SHARDS 16u
// Threads of each group that selects a sampled row over the whole
// vocabulary (decode_sample_vocabulary*); a bracket of at most this many
// tokens is ordered in threadgroup memory, one token per thread.
#define SPLASH_TARGET_VOCABULARY_THREADS 1024u
// Groups that share such a row's draw, each over its own slice of the
// vocabulary, so the eight rows of a lane spread over the GPU's cores.
#define SPLASH_TARGET_VOCABULARY_GROUPS 8u
// The ranges of the vocabulary such a row's draw sums: one per simdgroup of
// the row's groups.
#define SPLASH_TARGET_VOCABULARY_RANGES                                    \
  (SPLASH_TARGET_VOCABULARY_GROUPS * (SPLASH_TARGET_VOCABULARY_THREADS / 32u))
#define SPLASH_DRAFT_SAMPLING_SHARDS 8u
// Candidates the draft selector keeps per proposal position; acceptance and
// the sampled draw read them.
#define SPLASH_DRAFT_CANDIDATES 16u
// The rank of the draft selector's codebooks.
#define SPLASH_DRAFT_SELECTOR_RANK 256u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define SPLASH_GDN_SCAN_STATE_ROWS 16u
#define SPLASH_GDN_SCAN_THREADS 128u
// The epsilon of every RMS norm the kernels compute, which a target's and a
// draft's configurations must state. Unsuffixed, so the host compares a
// configuration's double with it exactly and the kernels take it as a float.
#define SPLASH_RMS_EPSILON 1e-6
// Plain norms of at most SPLASH_STAGED_NORM_ROWS rows of at most
// SPLASH_STAGED_NORM_WIDTH columns run norm_rms_staged, whose 1024-thread
// groups hold a row in threadgroup memory: the region where it measured
// faster than norm_rms (shared/normalization.metal), which covers every
// decode norm of a 2048-wide model and its short prefill chunks.
#define SPLASH_STAGED_NORM_WIDTH 2048u
#define SPLASH_STAGED_NORM_ROWS 64u
#define SPLASH_STAGED_NORM_THREADS 1024u
