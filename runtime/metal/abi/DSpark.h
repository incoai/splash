#pragma once

// Parameters of DSpark's proposal kernels (kernels/decode/dspark.metal,
// model/DSparkDraft.cpp): each lane's block of blockSize rows drafts a token
// a row, its Markov head's chain from the lane's anchor: the token of row r
// is the argmax of the row's logits plus the bias the Markov head projects
// from the embedding of the token before it, and its confidence the sigmoid
// of the confidence head's dot with the row's pre-sum of the streams and that
// embedding. The chain is the lane's tree (dspark_tree).
#include "metal/abi/ExecutionGeometry.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Threads of a lane's threadgroup of a step.
#define DSPARK_STEP_THREADS 1024u

// Step `position` of the chain of `lanes` lanes: rows of `lane_rows` a lane
// of the logits (fp32 [rows][vocabulary]) and of the streams and mixes
// (fp32 [rows][MHC_STREAMS][hidden], [rows][MHC_MIXES]); the bias (fp32
// [lanes][vocabulary]) and Markov embedding (bf16 [lanes][rank]) of each
// lane's token before the step's; the confidence head's weights, fp32
// [hidden + rank]. Its proposal and confidence go to place `position` of the
// lane's SPLASH_DRAFT_PROPOSAL_TOKENS.
struct DSparkStepParams {
  uint32_t lanes;
  uint32_t vocabulary;
  uint32_t hidden;
  uint32_t rank;
  uint32_t block;
  uint32_t position;
  uint32_t lane_rows;
};

static_assert(sizeof(DSparkStepParams) == 28, "DSpark step parameters are 28 bytes on both sides");

// Each lane's tree of a decode cycle (dspark_tree), in the selector's tree
// buffers (ops::DraftSelectorBuffers: from row l * SPLASH_TARGET_VERIFY_ROWS,
// metal/abi/VerifyTree.h): a chain from its anchor, of no more rows past it
// than depth[l]. Of a lane of a host chain of chain[l] tokens, which a
// grammar or a thinking end forces or a lookup proposes in place of DSpark's
// (model::Runtime's proposeChain), those tokens, each of no draft probability
// (-1) and estimated at the product of the chain's rates (each token's of an
// agreement with the draft not known, SPLASH_TREE_AGREEMENTS - 1, a forced
// token's 1) to it; of any other, its `block`
// proposals, each of draft probability its confidence q and a chance
// sigmoid(slope[l] * logit(q) + intercept[l][p]) at proposal position p,
// estimated at the product of the chances along its path, as the lane's
// acceptance model estimates a draft row's (DraftTreeParams). A row draws no
// children: its tokens are deterministic, so a sampled lane's verify row
// draws from the target's distribution alone, and acceptance follows the
// row's child where its token is that draw.
struct DSparkTreeParams {
  uint32_t lanes;
  uint32_t block;
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t depth[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t chain[SPLASH_MAXIMUM_BATCH_WIDTH];
  float slope[SPLASH_MAXIMUM_BATCH_WIDTH];
  float intercept[SPLASH_MAXIMUM_BATCH_WIDTH][SPLASH_DRAFT_PROPOSAL_TOKENS];
};

static_assert(sizeof(DSparkTreeParams) == 8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH * (4 + SPLASH_DRAFT_PROPOSAL_TOKENS),
              "DSpark tree parameters are the same size on both sides");
