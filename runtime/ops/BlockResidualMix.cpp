#include "ops/BlockResidualMix.hpp"

#include "metal/abi/BlockResidualMix.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
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
  if (!BlockResidualMix::rowKernels(rows))
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

// A call's one mix of one query whose fp32 current rows are its
// convolution's (BlockResidualStates::convolution), in the convolved kernel,
// a threadgroup a row and query: Norm through `norm` into its bf16
// `normalized` rows, or not into its fp32 `output` and, where it names them,
// `normalized` rows. Slots a kernel does not touch bind the current rows or
// the queries. A call of more rows than that threadgroup's mix takes is
// refused (BlockResidualMix::convolves).
void addConvolved(metal::CommandGraph &graph, const char *name, const BlockResidualStates &states,
                  BlockResidualShape shape, uint32_t rows, const metal::MetalBuffer &queries,
                  const metal::MetalBuffer &output, const metal::MetalBuffer &norm,
                  const metal::MetalBuffer &normalized) {
  const ShortConvVerify &c = *states.convolution;
  if (shape.queries != 1 || !states.currentFloat || c.shape.channels != shape.hidden || c.rows.rows != rows ||
      !c.buffers.output.sameView(states.current))
    throw std::invalid_argument("invalid block residual mix of a convolution's rows");
  if (!BlockResidualMix::convolves(rows))
    throw std::invalid_argument("a block residual mix of more than " + std::to_string(BLOCK_RESIDUAL_MIX_QUERY_ROWS) +
                                " rows runs its convolution's dispatch, not the convolution");
  const BlockResidualMixParams mix = mixParams(states, shape, rows);
  const BlockResidualMixConvolutionParams p{
      {{mix, mix}, 1, output && normalized ? 1u : 0u},
      ShortConvolution::verifyParams(c.buffers, c.shape, c.stateIn, c.state, c.layout, c.rows, c.tree, c.tape,
                                     ShortConvOutput::Sum)};
  const auto in = laneBindings(c.stateIn);
  const MixLaunch convolved = launch(shape, rows);
  graph.add(std::string(name) + "_convolved",
            {stackBinding(states, shape), states.current, queries, output ? output : states.current,
             norm ? norm : queries, normalized ? normalized : states.current, c.buffers.input, in[0], in[1], in[2],
             in[3], in[4], in[5], in[6], in[7], c.buffers.taps, c.buffers.residual ? c.buffers.residual : states.current,
             c.layout, c.tree, c.tape},
            p, convolved.groups, convolved.threads);
}

} // namespace

void BlockResidualMix::addMix(metal::CommandGraph &graph, std::span<const BlockResidualMixInput> mixes,
                              uint32_t rows) {
  if (mixes.empty() || mixes.size() > BLOCK_RESIDUAL_MIX_MAXIMUM_MIXES)
    throw std::invalid_argument("a block residual mix call takes 1 to 2 mixes");
  const BlockResidualMixInput &first = mixes.front();
  const bool normalized = static_cast<bool>(first.normalized);
  for (const BlockResidualMixInput &mix : mixes) {
    requireShape(mix.shape, mix.states, rows);
    requireStates(mix.states, mix.shape, rows);
    const uint64_t values = uint64_t{rows} * mix.shape.queries * mix.shape.hidden;
    requireBytes(mix.queries, uint64_t{mix.shape.queries} * mix.shape.hidden * sizeof(float),
                 "block residual mix queries");
    requireBytes(mix.output, values * sizeof(float), "block residual mix output");
    if (normalized) requireBytes(mix.normalized, values * sizeof(uint16_t), "block residual mix normalized output");
    if (mix.shape.hidden != first.shape.hidden || mix.states.currentFloat != first.states.currentFloat ||
        static_cast<bool>(mix.normalized) != normalized)
      throw std::invalid_argument("a block residual mix call's mixes are of one hidden size, current type and "
                                  "normalized outputs or none");
  }
  if (normalized && first.shape.hidden % 4) throw std::invalid_argument("normalized mix outputs of whole float4s");
  if (std::ranges::any_of(mixes, [](const BlockResidualMixInput &mix) { return mix.states.convolution; })) {
    if (mixes.size() != 1) throw std::invalid_argument("a block residual mix of a convolution's rows is its call's one");
    addConvolved(graph, "block_residual_mix", first.states, first.shape, rows, first.queries, first.output, {},
                 first.normalized);
    return;
  }
  // The second mix's slots bind the first's where the call has one, and the
  // normalized outputs' the outputs where it writes none.
  BlockResidualMixesParams p{{}, static_cast<uint32_t>(mixes.size()), normalized ? 1u : 0u};
  metal::BufferList bindings;
  uint32_t queries = 0;
  for (uint32_t m = 0; m < BLOCK_RESIDUAL_MIX_MAXIMUM_MIXES; ++m) {
    const BlockResidualMixInput &mix = mixes[m < mixes.size() ? m : 0];
    p.mix[m] = mixParams(mix.states, mix.shape, rows);
    if (m < mixes.size()) queries += mix.shape.queries;
    bindings.insert(bindings.end(), {stackBinding(mix.states, mix.shape), mix.states.current, mix.queries, mix.output,
                                     normalized ? mix.normalized : mix.output});
  }
  // A threadgroup a row and query of each mix, or of the _rows kernels a row
  // of each mix.
  MixLaunch mix = launch(first.shape, rows);
  mix.groups.y = BlockResidualMix::rowKernels(rows) ? p.mixes : queries;
  graph.add(instance("block_residual_mix", mix, first.states), std::move(bindings), p, mix.groups, mix.threads);
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
  metal::BufferList bindings{stackBinding(states, shape), states.current, std::move(queries)};
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
  if (states.convolution) {
    addConvolved(graph, "block_residual_mix_norm", states, shape, rows, bindings[2], {}, norms[0].buffer,
                 outputs[0]);
    return {};
  }
  const BlockResidualMixParams mix = mixParams(states, shape, rows);
  const MixLaunch norm = launch(shape, rows);
  graph.add(instance("block_residual_mix_norm", norm, states), std::move(bindings),
            BlockResidualMixesParams{{mix, mix}, 1, 0}, norm.groups, norm.threads);
  return {};
}

} // namespace splash::ops
