#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): row r of a
// lane's selection draws with its uniform r, the first token after a prompt
// (a selection of one row) with uniform 0. A sampled verify row draws its
// token from the target's whole distribution, and acceptance follows a row's
// child only when the child's token is that draw, so every token a lane keeps
// is a draw from the target and the rows of one path draw with uniforms of
// their own. Lane l's uniforms start at l * SPLASH_SAMPLING_UNIFORMS.
#define SPLASH_SAMPLING_UNIFORMS SPLASH_TARGET_VERIFY_ROWS

// The stop tokens a target's selections and acceptance compare
// (ops::StopTokens): up to SPLASH_STOP_TOKENS ids, every place past them
// SPLASH_NO_TOKEN, which no vocabulary holds. It is no stop token: a row with
// no selectable logit selects it, and the runtime fails that lane
// (Runtime.cpp, invalidSelection) whatever its family's list.
#define SPLASH_STOP_TOKENS 4u
#define SPLASH_NO_TOKEN 0xFFFFFFFFu

// One target-policy dispatch over the selected rows of its lanes, a grid of a
// lane's rows by lanes: lane l selects rows[l] rows of the logits from row
// first_row[l] (ops::LaneRows), at most SPLASH_TARGET_VERIFY_ROWS. Its row
// i, the logits row s = first_row[l] + i, reads mask row
// l * (SPLASH_TARGET_VERIFY_ROWS + 1) + mask_row + i of the lane's
// SPLASH_TARGET_VERIFY_ROWS + 1 constraint-mask rows and draws with uniform
// l * SPLASH_SAMPLING_UNIFORMS + i; workspaces, vocabulary rows and output
// tokens are indexed by s.
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t mask_row;
  uint32_t first_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t rows[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t top_k[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  float top_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  float min_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_tokens[SPLASH_STOP_TOKENS];
};

static_assert(sizeof(TargetSamplingParams) == 136,
              "Target sampling parameters are 136 bytes on both sides");

// One shard's share of a sampled row's softmax denominator: the largest
// logit it admits, the sum of exp((logit - maximum) / temperature) over its
// admitted tokens, and how many it admits.
struct TargetShardMass {
  float maximum;
  float sum;
  uint32_t admitted;
};

static_assert(sizeof(TargetShardMass) == 12,
              "Target shard masses are 12 bytes on both sides");

// A sampled row's selection over the whole vocabulary. The search records
// the row's largest admitted logit and where its min-p/top-k/top-p
// distribution ends in the order of the logits (the key and id of its last
// token, metal/kernels/decode/sampling.metal); the token the draw takes from
// that distribution goes to the output tokens.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
};

static_assert(sizeof(TargetVocabularyRow) == 12,
              "Target vocabulary rows are 12 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens, and one past the last
// of those with weight.
struct TargetVocabularyRange {
  float weight;
  uint32_t after;
};

static_assert(sizeof(TargetVocabularyRange) == 8,
              "Target vocabulary ranges are 8 bytes on both sides");

// A penalized request's word for each vocabulary token, in its state lane's
// row of the penalty table (ops::Sampling::rebuildPenaltyWords): the
// prompt bit marks a prompt token, and the count is how often the target
// selected it.
#define SPLASH_PENALTY_PROMPT_BIT 0x80000000u
#define SPLASH_PENALTY_COUNT_MASK 0x7fffffffu

// The penalized lanes of one penalty dispatch. Entry e penalizes its lane's
// selected rows, rows[e] rows of the logits from first_row[e]
// (TargetSamplingParams), from the penalty table row table_row[e]; the
// rows' verify input rows hold its draft tokens. repetition_inverse is
// 1 / repetition, saturated to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t entries;
  uint32_t first_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t rows[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[SPLASH_MAXIMUM_BATCH_WIDTH];
  float presence[SPLASH_MAXIMUM_BATCH_WIDTH];
  float frequency[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 120,
              "Sampling penalty parameters are 120 bytes on both sides");

// One selection's thinking cutoffs (ops::SamplingPolicy::thinkingCutoff), its
// lanes' rows laid out as in TargetSamplingParams. A lane whose row is below
// its rows selects its end token tokens[i] at row + i for each i below its
// count and its rows less row, unless a draft token before that row is close
// or also_ends, which ended thinking first. The end of a lane in
// waiting_mask, a constrained one, waits for its mask: its row is instead the
// first from there whose constraint mask row (mask_row on, of mask_words
// words, as in TargetSamplingParams) admits tokens[0]. The rows before it keep
// their own selections, as do all the rows of a selection without one. A row
// whose selection is the sentinel of a row with no selectable logit (a token
// outside the vocabulary) keeps it, so the lane still fails. A lane whose row
// is 0 and whose end does not wait drafts its end tokens before the verify
// (decode_draft_thinking_cutoff), so acceptance keeps them all.
struct ThinkingCutoffParams {
  uint32_t vocabulary;
  uint32_t waiting_mask;
  uint32_t mask_words;
  uint32_t mask_row;
  uint32_t first_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t rows[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t count[SPLASH_MAXIMUM_BATCH_WIDTH];
  // The token that closes thinking, the last of the lane's end.
  uint32_t close[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t also_ends[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t tokens[SPLASH_MAXIMUM_BATCH_WIDTH][SPLASH_TARGET_VERIFY_ROWS];
};

static_assert(sizeof(ThinkingCutoffParams) == 240,
              "Thinking cutoff parameters are 240 bytes on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread.
struct SelectorBatchParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t vocabulary;
};

static_assert(sizeof(SelectorBatchParams) == 24,
              "Draft selector parameters are 24 bytes on both sides");

// The rows of its block each lane of a decode cycle verifies (its verify
// rows, draft_verify_rows), chosen for the whole batch by expected accepted
// tokens per unit of the cycle's cost. The selector writes each lane's
// confidence at each proposal position: the largest probability among the
// position's candidates (the softmax at temperature 1 of their scores), the
// probability of the candidate it proposes there. Once the proposals
// before it are accepted, a proposal is accepted with probability p, where
// logit(p) = slope * logit(confidence) + intercept[position]; a lane's
// survival at a position, the product of p up to it, is the probability that
// its proposals up to it are all accepted, and verifying one more row adds
// that much to the lane's expected tokens. Every lane verifies its anchor;
// the rows past it go to the largest survivals over all lanes, as many as
// maximize the expected tokens over cost[the rows verified in all], none past
// a lane's maximum. With every_row, each lane verifies its maximum. The
// verify's MoE steps route the verify rows alone, and acceptance retains at
// most that many tokens.
struct VerifyRowsParams {
  float slope;
  float intercept[SPLASH_DRAFT_PROPOSAL_TOKENS];
  float cost[SPLASH_DRAFT_QUERY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH + 1];
  uint32_t lanes;
  uint32_t every_row;
  uint32_t maximum[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(VerifyRowsParams) == 188,
              "Verify rows parameters are 188 bytes on both sides");

// Acceptance of each lane's verify block, rows[l] rows from first_row[l] of
// the command's packed rows (ops::LaneRows), as a tree
// (metal/abi/VerifyTree.h): from its anchor's row the lane follows the child,
// among its verify rows, whose token (its verify input row) is the current
// row's selection, until no child is. It retains the selections of that
// path's rows in order, the last the target's token after the path, at most
// remaining[l] of them and none after a stop token, and the mask of the rows
// it retains (bit j for row j of its block). A greedy row selects its argmax
// and a sampled row draws from the target's whole distribution, so a lane
// retains the target's own tokens under either policy.
struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t first_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t rows[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_tokens[SPLASH_STOP_TOKENS];
};

static_assert(sizeof(AcceptBatchParams) == 64,
              "Batched acceptance parameters are 64 bytes on both sides");
