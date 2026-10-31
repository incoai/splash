#pragma once

// Parameters of DSpark's proposal kernels (kernels/decode/dspark.metal,
// model/DSparkDraft.cpp): each lane's block of blockSize rows drafts a token
// a row, its Markov head's chain from the lane's anchor: the token of row r
// is the argmax of the row's logits plus the bias the Markov head projects
// from the embedding of the token before it, and its confidence the sigmoid
// of the confidence head's dot with the row's pre-sum of the streams and that
// embedding (DeepSeek's inference/model.py DSparkBlock).
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
// [hidden + rank]. The last step (position block - 1) also fills the
// proposal positions past the block with its token, at confidence zero.
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
