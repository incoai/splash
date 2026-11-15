#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/VerifyTree.h"
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
// lane's rows by lanes: lane l selects its verify rows, the first
// verify_rows[l] of its lane rows (SplashLaneRows, metal/abi/VerifyTree.h),
// which a decode cycle's selector writes (VerifyRowsParams), at most
// SPLASH_TARGET_VERIFY_ROWS. Its row i, the logits row s = first + i, reads
// mask row l * (SPLASH_TARGET_VERIFY_ROWS + 1) + mask_row + i of the lane's
// SPLASH_TARGET_VERIFY_ROWS + 1 constraint-mask rows and draws with uniform
// l * SPLASH_SAMPLING_UNIFORMS + i; workspaces, vocabulary rows and output
// tokens are indexed by s.
struct TargetSamplingParams {
  uint32_t vocabulary;
  uint32_t mask_words;
  uint32_t mask_row;
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

static_assert(sizeof(TargetSamplingParams) == 104,
              "Target sampling parameters are 104 bytes on both sides");

// The selection of each lane's first token after its prompt from its one row
// of the logits, row[l]: decode_sample_first_rows writes the lane rows and
// verify rows (TargetSamplingParams) that the selection's kernels read.
struct SampleFirstRowsParams {
  uint32_t row[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SampleFirstRowsParams) == 16,
              "First-token row parameters are 16 bytes on both sides");

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

// The penalized lanes of one penalty dispatch. Entry e penalizes the selected
// rows of the dispatch's lane lane[e] (TargetSamplingParams), from the
// penalty table row table_row[e]; the rows' verify input rows hold its draft
// tokens, of which a row's context adds its ancestors'
// (metal/abi/VerifyTree.h). repetition_inverse is 1 / repetition, saturated
// to the largest float.
struct SamplingPenaltyParams {
  uint32_t vocabulary;
  uint32_t entries;
  uint32_t lane[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t table_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition[SPLASH_MAXIMUM_BATCH_WIDTH];
  float repetition_inverse[SPLASH_MAXIMUM_BATCH_WIDTH];
  float presence[SPLASH_MAXIMUM_BATCH_WIDTH];
  float frequency[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(SamplingPenaltyParams) == 104,
              "Sampling penalty parameters are 104 bytes on both sides");

// One selection's thinking cutoffs (ops::SamplingPolicy::thinkingCutoff) over
// its lanes' selected rows (TargetSamplingParams), each a row of its block's
// tree (metal/abi/VerifyTree.h). A lane whose row is below its rows
// selects its end token tokens[i] at each row of depth row + i, for each i
// below its count, unless a draft token on the row's path up to depth row is
// close or also_ends, which ended thinking first. The end of a lane in
// waiting_mask, a constrained one, whose block is a chain, waits for its mask:
// its row is instead the first from there whose constraint mask row (mask_row
// on, of mask_words words, as in TargetSamplingParams) admits tokens[0]. The
// rows before it keep their own selections, as do all the rows of a selection
// without one. A row whose selection is the sentinel of a row with no
// selectable logit (a token outside the vocabulary) keeps it, so the lane
// still fails. A lane whose row is 0 and whose end does not wait proposes its
// end tokens as a host chain of its tree (DraftTreeParams), so acceptance keeps
// them all.
struct ThinkingCutoffParams {
  uint32_t vocabulary;
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

static_assert(sizeof(ThinkingCutoffParams) == 80 + 16 * SPLASH_TARGET_VERIFY_ROWS,
              "Thinking cutoff parameters are the same size on both sides");

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

// The most tokens a host chain proposes: a verify block's rows after its
// anchor.
#define SPLASH_TREE_CHAIN_TOKENS (SPLASH_TARGET_VERIFY_ROWS - 1u)
// A host chain token's estimates, by its agreement with the draft at its
// depth (model::LookupAgreement): the draft's most likely candidate there,
// another of its candidates, none of them, or past the draft's positions.
#define SPLASH_TREE_AGREEMENTS 4u
// The most children a tree's draft row takes, its most likely candidates.
#define SPLASH_TREE_CHILDREN 4u

// The tree each lane grows (draft_select_tree), up to a verify block's rows
// from row l * SPLASH_TARGET_VERIFY_ROWS of the selector's trees, its
// anchor's first, in the order it takes them. A row's estimate is the
// probability that the target accepts the path to it:
// the product over the path of its steps' estimates. A step to a draft
// candidate at proposal position p, of probability q among the position's
// candidates after its parent's (the softmax at temperature 1 of their unary
// scores plus the edge from the parent's candidate), is estimated at
// sigmoid(slope * logit(q) + intercept[p]); a step to host chain token d of
// the lane, chain[l] tokens from the anchor, at its rate for its agreement
// with the draft at depth d. The tree grows best first by estimate, a draft
// row taking its children[l] most likely candidates and a chain token the
// next, one row for each token path (a step to a token its parent already has
// a child of joins that child), until the lane's rows or none past depth
// depth[l] are left.
struct DraftTreeParams {
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t depth[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t chain[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t children[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  float slope;
  float intercept[SPLASH_DRAFT_PROPOSAL_TOKENS];
};

static_assert(sizeof(DraftTreeParams) == 100,
              "Draft tree parameters are 100 bytes on both sides");

// The rows of a decode cycle (draft_verify_rows): `rows` rows over its lanes,
// 1 to lane_cap a lane, laid out from the lanes' trees (DraftTreeParams).
// Every lane verifies its anchor; the cycle's other rows go to the largest
// estimates of the trees' rows, a lane's in the order its tree took them
// (each at most its parent's, which comes before it), the earlier lane first
// among equal ones, none past a lane's cap. With every_row the cycle verifies as many as its rows hold;
// otherwise as many of those as maximize the expected tokens, its lanes plus
// the estimates it took, over cost[the rows it verifies in all]. A lane's
// verify rows are its anchor and the rows it took, numbered in preorder, its
// likeliest path on its first rows. Lanes' rows are packed in lane order
// (SplashLaneRows), and the cycle's rows past every lane's verify rows go to
// lanes under the cap, the first lane first, each a row of its anchor's token
// that extends the path of the row before it: no kernel selects, retains or
// routes through the MoE experts a row past its lane's verify rows.
struct VerifyRowsParams {
  float cost[SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH + 1];
  uint32_t lanes;
  uint32_t rows;
  uint32_t lane_cap;
  uint32_t every_row;
};

static_assert(sizeof(VerifyRowsParams) == 4 * (SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH + 5),
              "Verify rows parameters are the same size on both sides");

// Acceptance of each lane's verify rows, the first verify_rows[l] of its lane
// rows (SplashLaneRows) of the command's packed rows, as a tree
// (metal/abi/VerifyTree.h): from its anchor's row the lane follows the child
// whose token (its verify input row) is the current row's selection, until no
// child is. It retains the selections of that path's rows in order, the last
// the target's token after the path, at most remaining[l] of them and none
// after a stop token; the mask of the rows it retains (bit j for row j of its
// block); and the path's proposals among them, all of them where the cap or
// a stop token cuts the path. A greedy row selects its argmax and a sampled
// row draws from the target's whole distribution, so a lane retains the
// target's own tokens under either policy.
struct AcceptBatchParams {
  uint32_t remaining[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t stop_tokens[SPLASH_STOP_TOKENS];
};

static_assert(sizeof(AcceptBatchParams) == 32,
              "Batched acceptance parameters are 32 bytes on both sides");
