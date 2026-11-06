#pragma once

// DeepSeek-V4.1's DSpark drafter's proposals (kernels/decode/dspark.metal,
// metal/abi/DSpark.h): its Markov head's chain over each lane's block.

#include "metal/CommandGraph.hpp"
#include "metal/abi/DSpark.h"

namespace splash::ops {

// A step's inputs: the block rows' logits (fp32 [rows][vocabulary]), the
// bias the Markov head projects from each lane's token before the step's
// (fp32 [lanes][vocabulary]), the rows' streams and mixes (fp32
// [rows][MHC_STREAMS][hidden], [rows][MHC_MIXES]), that token's Markov
// embedding (bf16 [lanes][rank]) and the confidence head (fp32 [hidden +
// rank]). Its outputs: the proposals as DraftSelectorBuffers holds them, and
// each lane's token before the step's (uint32 [lanes]), which becomes the
// step's.
struct DSparkStepBuffers final {
  metal::MetalBuffer logits;
  metal::MetalBuffer bias;
  metal::MetalBuffer streams;
  metal::MetalBuffer mixes;
  metal::MetalBuffer markov;
  metal::MetalBuffer confidenceHead;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer confidence;
  metal::MetalBuffer candidates;
  metal::MetalBuffer proposalProbabilities;
  metal::MetalBuffer previous;
};

struct DSparkMarkov final {
  // Step params.position of the chains of params.lanes lanes.
  static void addStep(metal::CommandGraph &graph, const DSparkStepBuffers &buffers, const DSparkStepParams &params);
};

} // namespace splash::ops
