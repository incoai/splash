#include "ops/DSparkMarkov.hpp"

namespace splash::ops {

void DSparkMarkov::addStep(metal::CommandGraph &graph, const DSparkStepBuffers &buffers,
                           const DSparkStepParams &params) {
  // A threadgroup per lane.
  graph.add("dspark_markov_step",
            {buffers.logits, buffers.bias, buffers.streams, buffers.mixes, buffers.markov, buffers.confidenceHead,
             buffers.proposedTokens, buffers.confidence, buffers.previous},
            params, {params.lanes, 1, 1}, {DSPARK_STEP_THREADS, 1, 1});
}

void DSparkMarkov::addTrees(metal::CommandGraph &graph, const DSparkTreeBuffers &buffers,
                            const DSparkTreeParams &params) {
  graph.add("dspark_tree",
            {buffers.proposedTokens, buffers.confidence, buffers.chainTokens, buffers.chainRates, buffers.treeTokens,
             buffers.trees,
             buffers.treeScores, buffers.treeSizes, buffers.treeDraws, buffers.treeCaps},
            params, {1, 1, 1}, {32, 1, 1});
}

} // namespace splash::ops
