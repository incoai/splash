#include "ops/DSparkMarkov.hpp"

namespace splash::ops {

void DSparkMarkov::addStep(metal::CommandGraph &graph, const DSparkStepBuffers &buffers,
                           const DSparkStepParams &params) {
  // A threadgroup per lane.
  graph.add("dspark_markov_step",
            {buffers.logits, buffers.bias, buffers.streams, buffers.mixes, buffers.markov, buffers.confidenceHead,
             buffers.proposedTokens, buffers.confidence, buffers.candidates, buffers.proposalProbabilities,
             buffers.previous},
            params, {params.lanes, 1, 1}, {DSPARK_STEP_THREADS, 1, 1});
}

} // namespace splash::ops
