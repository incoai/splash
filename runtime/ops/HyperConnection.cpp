#include "ops/HyperConnection.hpp"

#include "metal/abi/HyperConnection.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = HYPER_CONNECTION_ROW_THREADS;

void requireShape(HyperShape shape) {
  if (!shape.hidden || !shape.streams || shape.streams > HYPER_CONNECTION_MAXIMUM_STREAMS)
    throw std::invalid_argument("invalid hyper-connection shape");
}

uint32_t groups(uint32_t width) { return (width + kThreads - 1) / kThreads; }

HyperConnectionParams hyperParams(HyperShape shape, uint32_t rows, bool inject = false) {
  return {rows, shape.hidden, shape.streams, shape.rank, inject ? 1u : 0u};
}

// The bytes of `rows` rows of the fp32 streams, of `rows` bf16 rows of
// `width` values, and of `rows` rows' injection weights, an fp32 per stream.
uint64_t streamBytes(HyperShape shape, uint32_t rows) { return uint64_t{rows} * shape.width() * sizeof(float); }
uint64_t bfloat16Bytes(uint32_t rows, uint32_t width) { return uint64_t{rows} * width * sizeof(uint16_t); }
uint64_t weightBytes(HyperShape shape, uint32_t rows) { return uint64_t{rows} * shape.streams * sizeof(float); }

} // namespace

void HyperConnection::addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                              HyperShape shape, uint32_t rows) {
  requireShape(shape);
  requireBytes(embedding, bfloat16Bytes(rows, shape.hidden), "hyper-connection embedding");
  requireBytes(streams, streamBytes(shape, rows), "hyper-connection streams");
  graph.add("hyper_connection_init", {std::move(embedding), std::move(streams)}, hyperParams(shape, rows),
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

void HyperConnection::addNorm(metal::CommandGraph &graph, metal::MetalBuffer streams, const NormWeights &norm,
                              metal::MetalBuffer inject, metal::MetalBuffer normalized, metal::MetalBuffer weights,
                              HyperShape shape, uint32_t rows) {
  requireShape(shape);
  if (shape.streams != HYPER_CONNECTION_MAXIMUM_STREAMS || shape.hidden != HYPER_CONNECTION_NORM_HIDDEN)
    throw std::invalid_argument("no hyper-connection norm kernel for the shape");
  if (!norm.float32) throw std::invalid_argument("the hyper-connection norm is not F32");
  const bool injects = static_cast<bool>(inject);
  requireBytes(streams, streamBytes(shape, rows), "hyper-connection streams");
  requireBytes(norm.buffer, norm.bytes(shape.width()), "hyper-connection norm");
  requireBytes(normalized, bfloat16Bytes(rows, shape.width()), "hyper-connection normalized");
  if (injects) {
    requireBytes(inject, uint64_t{shape.streams} * shape.width() * sizeof(float), "hyper-connection injection");
    requireBytes(weights, weightBytes(shape, rows), "hyper-connection weight");
  }
  // A mix without injection binds its norm in the unused slots.
  std::vector<metal::MetalBuffer> bindings{std::move(streams), norm.buffer, injects ? inject : norm.buffer,
                                           std::move(normalized), injects ? weights : norm.buffer};
  // An injecting mix of decode's few rows runs a threadgroup of a quarter per
  // stream each, the rest one of kThreads (kernels/shared/hyper_connection.metal).
  const bool quarters = injects && rows <= HYPER_CONNECTION_NORM_QUARTER_ROWS;
  graph.add(quarters ? "hyper_connection_norm_quarters" : "hyper_connection_norm_loaded", std::move(bindings),
            hyperParams(shape, rows, injects), {rows, 1, 1},
            {quarters ? HYPER_CONNECTION_NORM_QUARTER_THREADS : kThreads, 1, 1});
}

PreparedInput HyperConnection::addLow(metal::CommandGraph &graph, metal::MetalBuffer low, HyperShape shape,
                                      uint32_t rows, LinearScratch scratch, LinearInput layout) {
  requireShape(shape);
  requireBytes(low, bfloat16Bytes(rows, shape.rank), "hyper-connection low");
  if (layout == LinearInput::Plain) {
    graph.add("hyper_connection_low", {std::move(low)}, hyperParams(shape, rows), {groups(rows * shape.rank), 1, 1},
              {kThreads, 1, 1});
    return {};
  }
  requireTableScratch(scratch, layout, shape.rank, rows);
  graph.add(std::string("hyper_connection_low") + tableSuffix(layout), {low, scratch.input, scratch.sums},
            hyperParams(shape, rows), {rows, 1, 1}, {kThreads, 1, 1});
  return {std::move(low), layout};
}

PreparedInput HyperConnection::addMix(metal::CommandGraph &graph, metal::MetalBuffer normalized,
                                      metal::MetalBuffer gate, metal::MetalBuffer mixed, HyperShape shape,
                                      uint32_t rows, LinearScratch scratch, LinearInput layout) {
  requireShape(shape);
  requireBytes(normalized, bfloat16Bytes(rows, shape.width()), "hyper-connection normalized");
  requireBytes(gate, bfloat16Bytes(rows, shape.width()), "hyper-connection gate");
  requireBytes(mixed, bfloat16Bytes(rows, shape.hidden), "hyper-connection mix");
  if (layout == LinearInput::Plain) {
    graph.add("hyper_connection_mix", {std::move(normalized), std::move(gate), std::move(mixed)},
              hyperParams(shape, rows), {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
    return {};
  }
  // Each threadgroup writes whole spans of its 256 values.
  if (shape.hidden % kThreads) throw std::invalid_argument("a mix table takes whole threadgroups of values");
  requireTableScratch(scratch, layout, shape.hidden, rows);
  graph.add(std::string("hyper_connection_mix") + tableSuffix(layout),
            {std::move(normalized), std::move(gate), mixed, scratch.input, scratch.sums}, hyperParams(shape, rows),
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
  return {std::move(mixed), layout};
}

void HyperConnection::addCombine(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer branch,
                                 metal::MetalBuffer weights, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  requireBytes(streams, streamBytes(shape, rows), "hyper-connection streams");
  requireBytes(branch, bfloat16Bytes(rows, shape.hidden), "hyper-connection branch");
  requireBytes(weights, weightBytes(shape, rows), "hyper-connection weight");
  graph.add("hyper_connection_combine", {std::move(streams), std::move(branch), std::move(weights)},
            hyperParams(shape, rows), {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

void HyperConnection::addStreamMean(metal::CommandGraph &graph, metal::MetalBuffer streams,
                                    metal::MetalBuffer destination, RowRegion to, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  if (!rows || uint64_t{to.column} + shape.hidden > to.stride) throw std::invalid_argument("invalid stream mean");
  requireBytes(streams, streamBytes(shape, rows), "stream mean input");
  requireBytes(destination, (uint64_t{to.row} + rows) * to.stride * sizeof(uint16_t), "stream mean output");
  const HyperStreamMeanParams mean{rows, shape.hidden, shape.streams, to.row, to.stride, to.column};
  graph.add("hyper_connection_stream_mean", {std::move(streams), std::move(destination)}, mean,
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

} // namespace splash::ops
