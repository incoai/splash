#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The uniforms one lane draws a decode cycle with, in [0, 1): the first
// token after a prompt draws SPLASH_UNIFORM_INITIAL; the draft's sampled
// proposal at position p draws SPLASH_UNIFORM_PROPOSALS + p; acceptance tests
// draft token p against SPLASH_UNIFORM_ACCEPTANCE + p; and a sampled verify
// row draws its correction, or the bonus token after the whole draft, with
// SPLASH_UNIFORM_CORRECTION. Lane l's uniforms start at
// l * SPLASH_SAMPLING_UNIFORMS.
#define SPLASH_UNIFORM_INITIAL 0u
#define SPLASH_UNIFORM_PROPOSALS 1u
#define SPLASH_UNIFORM_ACCEPTANCE                                          \
  (SPLASH_UNIFORM_PROPOSALS + SPLASH_DRAFT_PROPOSAL_TOKENS)
#define SPLASH_UNIFORM_CORRECTION                                          \
  (SPLASH_UNIFORM_ACCEPTANCE + SPLASH_DRAFT_PROPOSAL_TOKENS)
#define SPLASH_SAMPLING_UNIFORMS (SPLASH_UNIFORM_CORRECTION + 1u)

// One target-policy dispatch over lanes of SPLASH_TARGET_VERIFY_ROWS logits
// rows, SPLASH_TARGET_VERIFY_ROWS + 1 constraint-mask rows and
// SPLASH_SAMPLING_UNIFORMS uniforms each. Selected row s is row s % rows of
// lane s / rows: its logits row is lane * SPLASH_TARGET_VERIFY_ROWS +
// logits_row + s % rows, its mask row lane * (SPLASH_TARGET_VERIFY_ROWS + 1) +
// mask_row + s % rows, and its draw takes uniform
// lane * SPLASH_SAMPLING_UNIFORMS + uniform; workspaces and output tokens are
// indexed by s. Rows below drafted_rows follow draft token s % rows (verify
// input row s % rows + 1).
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t rows;
  uint32_t logits_row;
  uint32_t mask_row;
  uint32_t uniform;
  uint32_t drafted_rows;
  uint32_t top_k[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  float top_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  float min_p[SPLASH_MAXIMUM_BATCH_WIDTH];
  // Lanes that sample; the others take the argmax.
  uint32_t sampling_mask;
  uint32_t constrained_mask;
  // Lanes that ignore end-of-sequence: they never select a stop token.
  uint32_t exclude_stop_mask;
  uint32_t stop_token_0;
  uint32_t stop_token_1;
};

static_assert(sizeof(TargetSamplingParams) == 112,
              "Target sampling parameters are 112 bytes on both sides");

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
// token, metal/kernels/decode/sampling.metal). The draw writes a drafted
// row's target probability of its draft token; the token it draws goes to
// the output tokens: for a verify row with a draft token, the correction
// acceptance takes if it rejects that token (a draw from the residual
// distribution); otherwise a draw from the row's distribution.
struct TargetVocabularyRow {
  float maximum;
  uint32_t end_key;
  uint32_t end_last;
  float draft_probability;
};

static_assert(sizeof(TargetVocabularyRow) == 16,
              "Target vocabulary rows are 16 bytes on both sides");

// One range of the vocabulary in such a row's draw, which one simdgroup of
// the row's groups sums: the kept weight of its tokens other than the
// draft's candidates, and one past the last of those with weight.
struct TargetVocabularyRange {
  float rest;
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

// The penalized lanes of one penalty dispatch. Each entry names the lane of
// its logits and the penalty table row it reads; rows penalizes that many
// rows of the lane's SPLASH_TARGET_VERIFY_ROWS, from row_offset.
// repetition_inverse is 1 / repetition, saturated to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t rows;
  uint32_t row_offset;
  uint32_t entries;
  uint32_t logits_lane[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[SPLASH_MAXIMUM_BATCH_WIDTH];
  float presence[SPLASH_MAXIMUM_BATCH_WIDTH];
  float frequency[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 112,
              "Sampling penalty parameters are 112 bytes on both sides");

// One selection's thinking cutoffs (ops::SamplingPolicy::thinkingCutoff), its
// rows laid out as in TargetSamplingParams. A lane whose row is below rows
// selects its end token tokens[i] at row + i for each i below its count and
// rows - row, unless a draft token before that row is close or also_ends,
// which ended thinking first; a sampled lane's drafted row then gives its
// draft token the target probability 1 when it is the row's end token, else
// 0. The end of a lane in waiting_mask, a constrained one, waits for its
// mask: its row is instead the first from there whose constraint mask row
// (mask_row on, of mask_words words, as in TargetSamplingParams) admits
// tokens[0]. The rows before it keep their own selections, as do all the rows
// of a selection without one. A row whose selection is the sentinel of a row
// with no selectable logit (a token outside the vocabulary) keeps it, so the
// lane still fails. A lane whose row is 0 and whose end does not wait drafts
// its end tokens before the verify (decode_draft_thinking_cutoff), so
// acceptance keeps them all.
struct ThinkingCutoffParams {
  uint32_t vocabulary;
  uint32_t rows;
  uint32_t drafted_rows;
  uint32_t sampling_mask;
  uint32_t waiting_mask;
  uint32_t mask_words;
  uint32_t mask_row;
  uint32_t row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t count[SPLASH_MAXIMUM_BATCH_WIDTH];
  // The token that closes thinking, the last of the lane's end.
  uint32_t close[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t also_ends[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t tokens[SPLASH_MAXIMUM_BATCH_WIDTH][SPLASH_TARGET_VERIFY_ROWS];
};

static_assert(sizeof(ThinkingCutoffParams) == 220,
              "Thinking cutoff parameters are 220 bytes on both sides");

// The batched selector and acceptance kernels' grids cover exactly the
// dispatch's lanes: per-lane arrays hold those lanes, and entries past them
// are zero and unread.
struct SelectorBatchParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  uint32_t sampling_mask;
  uint32_t vocabulary;
};

static_assert(sizeof(SelectorBatchParams) == 44,
              "Draft selector parameters are 44 bytes on both sides");

// The rows of its block each lane of a decode cycle verifies (its verify
// rows, draft_verify_rows), chosen for the whole batch by expected accepted
// tokens per unit of the cycle's cost. The selector writes each lane's
// confidence at each proposal position: the largest probability among the
// position's candidates (the softmax at temperature 1 of their scores), which
// does not depend on the candidate the lane draws there. Once the proposals
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

struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_token_0;
  uint32_t stop_token_1;
  uint32_t sampling_mask;
};

static_assert(sizeof(AcceptBatchParams) == 28,
              "Batched acceptance parameters are 28 bytes on both sides");
