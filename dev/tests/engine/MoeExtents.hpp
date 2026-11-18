#pragma once

#include "MoeStep.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/MoE.hpp"

#include <array>
#include <tuple>
#include <cstdint>
#include <utility>

namespace splash::test {

// Each buffer a MoE plan reaches, at its extent and one element short: the
// rows' bf16 input, residual, output and router input (those the step has),
// and every scratch field the plan's workspace sizes, of expert ids, fp32
// routing weights, tile descriptors, the tile count, routes, route rows and
// expert slots, and bf16 grouped rows, intermediates and outputs.
inline void requireMoeExtents(metal::MetalBackend &backend, const ops::MoeBuffers &buffers,
                              const ops::BlockMoeWeights &weights, const ops::MoePlan &plan) {
  constexpr std::array<uint64_t, ops::kMoeScratchFields.size()> elements{4, 4, sizeof(MoeTileDescriptor), 4, 4,
                                                                          4, 4, 2, 2, 2};
  const auto encode = [&](metal::CommandGraph &graph, const ops::MoeBuffers &changed) {
    addMoeStep(graph, changed, weights, plan);
  };
  for (size_t index = 0; index < ops::kMoeScratchFields.size(); ++index) {
    const ops::MoeScratchField &field = ops::kMoeScratchFields[index];
    if (const uint64_t bytes = plan.workspace().*field.bytes)
      requireExtent(backend, buffers.scratch.*field.buffer, bytes, elements[index], field.name,
                    [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                      ops::MoeBuffers changed = buffers;
                      changed.scratch.*field.buffer = view;
                      encode(graph, changed);
                    });
  }
  for (const auto &[member, name, width] :
       {std::tuple{&ops::MoeBuffers::input, "MoE input", plan.shape().hiddenSize},
        std::tuple{&ops::MoeBuffers::residual, "MoE residual", plan.shape().hiddenSize},
        std::tuple{&ops::MoeBuffers::output, "MoE output", plan.shape().hiddenSize},
        std::tuple{&ops::MoeBuffers::routerInput, "MoE router input", plan.shape().routerInput()}}) {
    if (!(buffers.*member)) continue;   // a buffer the step goes without
    requireExtent(backend, buffers.*member, uint64_t{plan.rows()} * width * 2, 2, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    ops::MoeBuffers changed = buffers;
                    changed.*member = view;
                    encode(graph, changed);
                  });
  }
}

} // namespace splash::test
