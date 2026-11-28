#pragma once

#include "MoeStep.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/MoE.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>

namespace splash::test {

// Each buffer a MoE plan reaches, at its extent and one element short: the
// rows' bf16 input (a narrow step's in-place matmul's, MoePlan::inputRows),
// residual and output, and every scratch field the plan's
// workspace sizes, of expert ids, fp32 routing weights, tile descriptors, the
// tile count, routes, route rows and expert slots, bf16 grouped rows,
// intermediates and outputs, and the int8 passes' codes and scales.
inline void requireMoeExtents(metal::MetalBackend &backend, const ops::MoeBuffers &buffers,
                              const ops::BlockMoeWeights &weights, const ops::MoePlan &plan) {
  constexpr std::array<uint64_t, ops::kMoeScratchFields.size()> elements{4, 4, sizeof(MoeTileDescriptor), 4, 4,
                                                                          4, 4, 2, 2, 2, 1, 8, 1, 8};
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
  for (const auto &[member, name, rows] : {std::tuple{&ops::MoeBuffers::input, "MoE input", plan.inputRows()},
                                           std::tuple{&ops::MoeBuffers::residual, "MoE residual", plan.rows()},
                                           std::tuple{&ops::MoeBuffers::output, "MoE output", plan.rows()}})
    requireExtent(backend, buffers.*member, uint64_t{rows} * plan.shape().hiddenSize * 2, 2, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    ops::MoeBuffers changed = buffers;
                    changed.*member = view;
                    encode(graph, changed);
                  });
}

} // namespace splash::test
