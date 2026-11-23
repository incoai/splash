#include "ops/BlockResidual.hpp"

#include "ops/BufferExtent.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops {

void BlockResidual::add(metal::CommandGraph &graph, std::span<const StridedRows> states, metal::MetalBuffer queries,
                        std::span<const metal::MetalBuffer> outputs, uint32_t outputStride, uint32_t hidden,
                        uint32_t rows) {
  if (states.empty() || states.size() > BLOCK_RESIDUAL_MAXIMUM_STATES || outputs.empty() ||
      outputs.size() > BLOCK_RESIDUAL_MAXIMUM_QUERIES || !hidden || hidden > BLOCK_RESIDUAL_MAXIMUM_HIDDEN || !rows ||
      outputStride < hidden)
    throw std::invalid_argument("invalid block residual");
  BlockResidualParams params{
      rows, hidden,       static_cast<uint32_t>(states.size()), static_cast<uint32_t>(outputs.size()),
      {},   outputStride, 1.0f / std::sqrt(float(hidden))};
  for (size_t k = 0; k < states.size(); ++k) {
    if (states[k].stride < hidden) throw std::invalid_argument("block residual state stride");
    requireBytes(states[k].buffer, rowBytes(rows, states[k].stride, hidden, sizeof(float)), "block residual state");
    params.state_stride[k] = states[k].stride;
  }
  requireBytes(queries, uint64_t{outputs.size()} * hidden * sizeof(float), "block residual queries");
  for (const metal::MetalBuffer &output : outputs)
    requireBytes(output, rowBytes(rows, outputStride, hidden, sizeof(float)), "block residual output");
  // Slots past the states and outputs bind the first again, which the
  // kernel does not read or write.
  std::vector<metal::MetalBuffer> bindings;
  for (uint32_t k = 0; k < BLOCK_RESIDUAL_MAXIMUM_STATES; ++k)
    bindings.push_back(states[k < states.size() ? k : 0].buffer);
  bindings.push_back(std::move(queries));
  for (uint32_t j = 0; j < BLOCK_RESIDUAL_MAXIMUM_QUERIES; ++j) bindings.push_back(outputs[j < outputs.size() ? j : 0]);
  graph.add("block_residual", std::move(bindings), params, {rows, 1, 1}, {BLOCK_RESIDUAL_THREADS, 1, 1});
}

} // namespace splash::ops
