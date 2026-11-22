#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/MoE.hpp"

namespace splash::test {

// One step of a MoE block as the runtime composes it (QwenTarget's layers):
// its routing, a streamed block's load beside its resident experts' passes,
// then its passes over the rest and its combine.
inline void addMoeStep(metal::CommandGraph &graph, const ops::MoeBuffers &buffers,
                       const ops::BlockMoeWeights &weights, const ops::MoePlan &plan) {
  ops::MoE::addRouting(graph, buffers, weights, plan);
  if (weights.stream) {
    const ops::MoeLoadPart part{&buffers, &plan};
    weights.stream->addLoad(graph, {&part, 1}, weights.streamBlock,
                            [&] { ops::MoE::addResident(graph, buffers, weights, plan); });
  }
  ops::MoE::addRouted(graph, buffers, weights, plan);
}

} // namespace splash::test
