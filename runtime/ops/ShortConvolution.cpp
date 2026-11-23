#include "ops/ShortConvolution.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = SHORT_CONVOLUTION_THREADS;
constexpr uint32_t kHistory = SHORT_CONVOLUTION_HISTORY;

uint32_t groups(uint32_t channels) { return (channels + kThreads - 1) / kThreads; }

// A dynamic convolution's heads split its channels evenly.
void requireShape(const ShortConvolutionShape &shape) {
  if (!shape.channels || (shape.dynamic() && shape.channels % shape.headChannels))
    throw std::invalid_argument("invalid short convolution shape");
}

// The bytes `rows` rows of `width` elements reach, `stride` elements apart,
// a stride that holds them.
uint64_t stridedBytes(const StridedRows &rows, uint32_t count, uint32_t width, FloatElement element, const char *what) {
  if (rows.stride < width) throw std::invalid_argument(std::string("short convolution ") + what + " stride");
  return rowBytes(count, rows.stride, width, elementBytes(element));
}

std::string instance(const char *kernel, const ShortConvolutionShape &shape) {
  return std::string(kernel) + "_" + elementToken(shape.input) + "_" + elementToken(shape.state);
}

} // namespace

float ShortConvolutionShape::scale() const noexcept {
  return 1.0f / std::sqrt(float((dynamic() ? 2 : 1) * SHORT_CONVOLUTION_TAPS + 1));
}

uint64_t shortConvolutionHistoryBytes(FloatElement state, uint32_t channels, uint32_t stride) {
  return rowBytes(kHistory, stride ? stride : channels, channels, elementBytes(state));
}

void ShortConvolution::addConvolution(metal::CommandGraph &graph, const ShortConvolutionBuffers &buffers,
                                      ShortConvolutionHistories histories, ShortConvolutionShape shape,
                                      metal::MetalBuffer layout, const LayoutRows &rows,
                                      const ShortConvolutionTree &tree) {
  requireShape(shape);
  const bool writeHistory = !histories.out.empty(), inTree = static_cast<bool>(tree);
  // A verify block's paths are row masks of 32 rows, and its history follows
  // the path it accepts (addCommit), not its rows.
  requireLayoutRows(rows, inTree ? 32 : std::numeric_limits<uint32_t>::max());
  if (histories.in.size() != rows.sequences || (writeHistory && histories.out.size() != rows.sequences) ||
      (inTree && writeHistory))
    throw std::invalid_argument("invalid short convolution");
  if (histories.stride < shape.channels) throw std::invalid_argument("short convolution history stride");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  requireBytes(buffers.input.buffer, stridedBytes(buffers.input, rows.rows, shape.channels, shape.input, "input"),
               "short convolution input");
  if (shape.dynamic())
    requireBytes(
        buffers.dynamic.buffer,
        stridedBytes(buffers.dynamic, rows.rows, shape.heads() * SHORT_CONVOLUTION_TAPS, shape.input, "dynamic taps"),
        "short convolution dynamic taps");
  requireBytes(buffers.taps, uint64_t{shape.channels} * SHORT_CONVOLUTION_TAPS * sizeof(float),
               "short convolution taps");
  requireBytes(buffers.output.buffer,
               stridedBytes(buffers.output, rows.rows, shape.channels, FloatElement::Float32, "output"),
               "short convolution output");
  const uint64_t historyBytes = shortConvolutionHistoryBytes(shape.state, shape.channels, histories.stride);
  for (const metal::MetalBuffer &history : histories.in)
    requireBytes(history, historyBytes, "short convolution history");
  for (const metal::MetalBuffer &history : histories.out)
    requireBytes(history, historyBytes, "short convolution next history");
  if (inTree) {
    requireBytes(tree.rows, uint64_t{rows.rows} * sizeof(SplashTreeRow), "short convolution tree rows");
    requireBytes(tree.tape.buffer, stridedBytes(tree.tape, rows.rows, shape.channels, shape.state, "tape"),
                 "short convolution tape");
  }
  const auto in = laneBindings(histories.in), out = laneBindings(writeHistory ? histories.out : histories.in);
  // A static convolution binds the input for the dynamic taps, and a prompt
  // the taps for the tree and the output for the tape, which they do not
  // touch.
  const ShortConvolutionParams params{rows.rows,
                                      shape.channels,
                                      rows.sequences,
                                      buffers.input.stride,
                                      buffers.output.stride,
                                      shape.dynamic() ? buffers.dynamic.stride : 0,
                                      histories.stride,
                                      inTree ? tree.tape.stride : 0,
                                      shape.headChannels,
                                      writeHistory ? 1u : 0u,
                                      inTree ? 1u : 0u,
                                      buffers.accumulate ? 1u : 0u,
                                      shape.scale()};
  std::vector<metal::MetalBuffer> bindings{buffers.input.buffer,
                                           shape.dynamic() ? buffers.dynamic.buffer : buffers.input.buffer,
                                           buffers.taps, buffers.output.buffer};
  appendLaneBindings(bindings, in, out);
  bindings.insert(bindings.end(), {std::move(layout), inTree ? tree.rows : buffers.taps,
                                   inTree ? tree.tape.buffer : buffers.output.buffer});
  graph.add(instance("short_convolution", shape), std::move(bindings), params,
            {rows.bound + (writeHistory ? kHistory : 0), groups(shape.channels), rows.sequences}, {kThreads, 1, 1});
}

void ShortConvolution::addHistory(metal::CommandGraph &graph, const StridedRows &input, metal::MetalBuffer historyIn,
                                  metal::MetalBuffer historyOut, uint32_t historyStride, ShortConvolutionShape shape,
                                  uint32_t rows) {
  requireShape(shape);
  if (!rows || historyStride < shape.channels) throw std::invalid_argument("invalid short convolution history");
  requireBytes(input.buffer, stridedBytes(input, rows, shape.channels, shape.input, "input"),
               "short convolution input");
  const uint64_t historyBytes = shortConvolutionHistoryBytes(shape.state, shape.channels, historyStride);
  requireBytes(historyIn, historyBytes, "short convolution history");
  requireBytes(historyOut, historyBytes, "short convolution next history");
  graph.add(instance("short_convolution_history", shape), {input.buffer, std::move(historyIn), std::move(historyOut)},
            ShortConvolutionHistoryParams{rows, shape.channels, input.stride, historyStride},
            {kHistory, groups(shape.channels), 1}, {kThreads, 1, 1});
}

void ShortConvolution::addCommit(metal::CommandGraph &graph, metal::MetalBuffer tape,
                                 std::span<const metal::MetalBuffer> historyIn,
                                 std::span<const metal::MetalBuffer> historyOut, metal::MetalBuffer paths,
                                 const ShortConvolutionUnits &units, metal::MetalBuffer layout,
                                 const LayoutRows &rows) {
  // A lane's path is a row mask of its rows.
  requireLayoutRows(rows, 32);
  if (!units.width || !units.units || historyIn.size() != rows.sequences || historyOut.size() != rows.sequences ||
      units.historyStride < units.width || units.tapeStride < units.width ||
      (units.units > 1 && (units.unitStride < uint64_t{kHistory - 1} * units.historyStride + units.width ||
                           units.tapeUnitStride < uint64_t{rows.rows - 1} * units.tapeStride + units.width)))
    throw std::invalid_argument("invalid short convolution commit");
  const uint64_t element = elementBytes(units.state);
  const uint64_t historyBytes =
      (uint64_t{units.units - 1} * units.unitStride + uint64_t{kHistory - 1} * units.historyStride + units.width) *
      element;
  for (const metal::MetalBuffer &history : historyIn) requireBytes(history, historyBytes, "short convolution history");
  for (const metal::MetalBuffer &history : historyOut)
    requireBytes(history, historyBytes, "short convolution next history");
  // The GPU's paths are of the lanes' verify rows, any of which they may hold.
  requireBytes(
      tape,
      (uint64_t{units.units - 1} * units.tapeUnitStride + uint64_t{rows.rows - 1} * units.tapeStride + units.width) *
          element,
      "short convolution tape");
  requireBytes(paths, uint64_t{rows.sequences} * sizeof(uint32_t), "short convolution accepted path");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  const auto in = laneBindings(historyIn), out = laneBindings(historyOut);
  std::vector<metal::MetalBuffer> bindings{std::move(tape)};
  appendLaneBindings(bindings, in, out);
  bindings.insert(bindings.end(), {std::move(paths), std::move(layout)});
  graph.add(std::string("short_convolution_commit_") + elementToken(units.state), std::move(bindings),
            ShortConvolutionCommitParams{rows.rows, rows.sequences, units.width, units.units, units.historyStride,
                                         units.unitStride, units.tapeStride, units.tapeUnitStride},
            {groups(units.width), kHistory, rows.sequences * units.units}, {kThreads, 1, 1});
}

} // namespace splash::ops
