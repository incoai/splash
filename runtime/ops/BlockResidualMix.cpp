#include "ops/BlockResidualMix.hpp"

#include "metal/abi/BlockResidualMix.h"
#include "ops/BufferExtent.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

// A call's kernel and grid (metal/abi/BlockResidualMix.h): a threadgroup a
// row and query for at most BLOCK_RESIDUAL_MIX_QUERY_ROWS rows, a threadgroup
// a row for more.
struct MixLaunch final {
  const char *suffix;
  metal::DispatchSize groups;
  metal::DispatchSize threads;
};
MixLaunch launch(BlockResidualShape shape, uint32_t rows) {
  if (rows <= BLOCK_RESIDUAL_MIX_QUERY_ROWS)
    return {"", {rows, shape.queries, 1}, {BLOCK_RESIDUAL_MIX_THREADS, 1, 1}};
  return {"_rows", {rows, 1, 1}, {BLOCK_RESIDUAL_MIX_ROW_THREADS, 1, 1}};
}

// States and queries the kernels hold, and planes that hold the rows.
void requireShape(BlockResidualShape shape, const BlockResidualStates &states, uint32_t rows) {
  if (!shape.hidden || !rows || shape.stack + 1 > BLOCK_RESIDUAL_MIX_MAXIMUM_STATES || !shape.queries ||
      shape.queries > BLOCK_RESIDUAL_MIX_MAXIMUM_QUERIES || (shape.stack && states.planeRows < rows))
    throw std::invalid_argument("invalid block residual mix shape");
}

void requireStates(const BlockResidualStates &states, BlockResidualShape shape, uint32_t rows) {
  requireBytes(states.current, uint64_t{rows} * shape.hidden * (states.currentFloat ? 4 : 2),
               "block residual mix current rows");
  if (shape.stack)
    requireBytes(states.stack, (uint64_t{shape.stack - 1} * states.planeRows + rows) * shape.hidden * 2,
                 "block residual mix stack");
}

// The scores' scale, hidden^-1/2, as the source scales its dot products.
BlockResidualMixParams mixParams(const BlockResidualStates &states, BlockResidualShape shape, uint32_t rows) {
  return {rows, shape.hidden, shape.stack, shape.queries, states.planeRows * shape.hidden,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.hidden)))};
}

// Without stack states the kernel reads no stack, which binds the current
// rows.
metal::MetalBuffer stackBinding(const BlockResidualStates &states, BlockResidualShape shape) {
  return shape.stack ? states.stack : states.current;
}

// The instance of the launch's shape for the current rows' type.
std::string instance(const char *name, const MixLaunch &launch, const BlockResidualStates &states) {
  return std::string(name) + launch.suffix + (states.currentFloat ? "_f32" : "");
}

} // namespace

void BlockResidualMix::addMix(metal::CommandGraph &graph, const BlockResidualStates &states,
                              metal::MetalBuffer queries, metal::MetalBuffer output, BlockResidualShape shape,
                              uint32_t rows) {
  requireShape(shape, states, rows);
  requireStates(states, shape, rows);
  requireBytes(queries, uint64_t{shape.queries} * shape.hidden * sizeof(float), "block residual mix queries");
  requireBytes(output, uint64_t{rows} * shape.queries * shape.hidden * sizeof(float), "block residual mix output");
  const MixLaunch mix = launch(shape, rows);
  graph.add(instance("block_residual_mix", mix, states),
            {stackBinding(states, shape), states.current, std::move(queries), std::move(output)},
            mixParams(states, shape, rows), mix.groups, mix.threads);
}

PreparedInput BlockResidualMix::addMixNorm(metal::CommandGraph &graph, const BlockResidualStates &states,
                                           metal::MetalBuffer queries, std::span<const NormWeights> norms,
                                           std::span<const metal::MetalBuffer> outputs, BlockResidualShape shape,
                                           uint32_t rows) {
  requireShape(shape, states, rows);
  if (shape.queries > BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS || norms.size() != shape.queries ||
      outputs.size() != shape.queries)
    throw std::invalid_argument("invalid block residual mix norms");
  requireStates(states, shape, rows);
  requireBytes(queries, uint64_t{shape.queries} * shape.hidden * sizeof(float), "block residual mix queries");
  std::vector<metal::MetalBuffer> bindings{stackBinding(states, shape), states.current, std::move(queries)};
  // Unused norm and output slots bind the first query's, which they never
  // read or write.
  for (uint32_t q = 0; q < BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS; ++q) {
    const NormWeights &norm = norms[q < shape.queries ? q : 0];
    if (!norm.float32) throw std::invalid_argument("a block residual mix norm is not F32");
    requireBytes(norm.buffer, norm.bytes(shape.hidden), "block residual mix norm");
    bindings.push_back(norm.buffer);
  }
  for (uint32_t q = 0; q < BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS; ++q) {
    const metal::MetalBuffer &out = outputs[q < shape.queries ? q : 0];
    requireBytes(out, uint64_t{rows} * shape.hidden * sizeof(uint16_t), "block residual mix normalized");
    bindings.push_back(out);
  }
  const MixLaunch mix = launch(shape, rows);
  graph.add(instance("block_residual_mix_norm", mix, states), std::move(bindings), mixParams(states, shape, rows),
            mix.groups, mix.threads);
  return {};
}

} // namespace splash::ops
