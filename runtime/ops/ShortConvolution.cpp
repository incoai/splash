#include "ops/ShortConvolution.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/ShortConvolution.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::ops {

// The kernels bind each lane's state, eight (metal/kernels/common/lane_bindings.h).
static_assert(SPLASH_MAXIMUM_BATCH_WIDTH == 8, "the lane bindings name eight lanes");
namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kTaps = SHORT_CONVOLUTION_TAPS;
constexpr uint32_t kStateRows = SHORT_CONVOLUTION_STATE_ROWS;

uint32_t groups(uint32_t width) { return (width + kThreads - 1) / kThreads; }

// A dynamic convolution's heads split its channels evenly.
void requireShape(ShortConvShape shape) {
  if (!shape.channels || (shape.dynamic() && shape.channels % shape.dynamicHeads))
    throw std::invalid_argument("invalid short convolution shape");
}

// The convolution's columns lie in its state matrix's.
void requireState(ShortConvShape shape, ShortConvState state) {
  if (uint64_t{state.column} + shape.channels > state.stride)
    throw std::invalid_argument("invalid short convolution state");
}

// The source divides the sum of the taps' products and the input by
// sqrt(taps + 1), or for dynamic taps, whose products count twice as many
// weights, sqrt(2 taps + 1): 3 at four taps.
float divisor(ShortConvShape shape) {
  return static_cast<float>(std::sqrt(static_cast<double>((shape.dynamic() ? 2 : 1) * kTaps + 1)));
}

// The values a row of the convolution's input holds: its own stride, or its
// channels.
uint32_t inputStride(const ShortConvBuffers &buffers, ShortConvShape shape) {
  return buffers.inputStride ? buffers.inputStride : shape.channels;
}

ShortConvolutionParams convolutionParams(ShortConvShape shape, uint32_t rows, ShortConvState state) {
  ShortConvolutionParams p{};
  p.rows = rows;
  p.channels = shape.channels;
  p.heads = shape.dynamicHeads;
  p.state_stride = state.stride;
  p.state_column = state.column;
  p.divisor = divisor(shape);
  p.input_stride = shape.channels;
  return p;
}

// The bytes of `rows` bf16 or fp32 rows of `width` values, and of a state
// matrix whose last row reaches the convolution's columns.
uint64_t bfloat16Rows(uint32_t rows, uint32_t width) { return uint64_t{rows} * width * sizeof(uint16_t); }
uint64_t floatRows(uint32_t rows, uint32_t width) { return uint64_t{rows} * width * sizeof(float); }
uint64_t stateExtent(ShortConvShape shape, ShortConvState state) {
  return rowBytes(kStateRows, state.stride, uint64_t{state.column} + shape.channels, sizeof(uint16_t));
}

// Throws unless the buffers hold a convolution's `rows` rows written as
// `output` asks: its input's channels within its rows, a residual where the
// output adds one, the dynamic taps of a dynamic shape.
void requireBuffers(const ShortConvBuffers &buffers, ShortConvShape shape, ShortConvOutput output, uint32_t rows) {
  const bool sum = output == ShortConvOutput::Sum, residual = static_cast<bool>(buffers.residual);
  const uint32_t stride = inputStride(buffers, shape);
  if (uint64_t{buffers.inputColumn} + shape.channels > stride || (output == ShortConvOutput::Plain && residual) ||
      (output == ShortConvOutput::RoundedSum && !residual) ||
      (shape.dynamic() && buffers.dynamicStride < shape.dynamicHeads * kTaps))
    throw std::invalid_argument("invalid short convolution");
  requireBytes(buffers.input, rowBytes(rows, stride, uint64_t{buffers.inputColumn} + shape.channels, sizeof(uint16_t)),
               "short convolution input");
  requireBytes(buffers.taps, uint64_t{shape.channels} * kTaps * sizeof(float), "short convolution taps");
  if (shape.dynamic())
    requireBytes(buffers.dynamicTaps,
                 rowBytes(rows, buffers.dynamicStride, shape.dynamicHeads * kTaps, sizeof(uint16_t)),
                 "short convolution dynamic taps");
  if (residual)
    requireBytes(buffers.residual, sum ? bfloat16Rows(rows, shape.channels) : floatRows(rows, shape.channels),
                 "short convolution residual");
  requireBytes(buffers.output, sum ? floatRows(rows, shape.channels) : bfloat16Rows(rows, shape.channels),
               "short convolution output");
}

// Throws unless a verify step's tree rows and tape hold its rows, the tape
// the state's columns.
void requireTape(metal::MetalBuffer tree, metal::MetalBuffer tape, ShortConvShape shape, ShortConvState state,
                 uint32_t rows) {
  requireBytes(tree, uint64_t{rows} * sizeof(SplashTreeRow), "short convolution tree rows");
  requireBytes(tape, rowBytes(rows, state.stride, uint64_t{state.column} + shape.channels, sizeof(uint16_t)),
               "short convolution tape");
}

} // namespace

void ShortConvolution::addConvolution(metal::CommandGraph &graph, const ShortConvBuffers &buffers,
                                      ShortConvShape shape, ShortConvOutput output,
                                      std::span<const metal::MetalBuffer> stateIn,
                                      std::span<const metal::MetalBuffer> stateOut, ShortConvState state,
                                      metal::MetalBuffer layout, const LayoutRows &rows,
                                      std::span<const ShortConvCapture> captures, metal::MetalBuffer tree,
                                      metal::MetalBuffer tape) {
  requireShape(shape);
  requireState(shape, state);
  const bool writeState = !stateOut.empty(), inTree = static_cast<bool>(tree);
  const bool sum = output == ShortConvOutput::Sum, residual = static_cast<bool>(buffers.residual);
  // A verify block's paths are row masks of 32 rows, and its state follows
  // the path it accepts (addCommit), from the inputs its rows copy into the
  // tape; captures are a prompt's.
  requireLayoutRows(rows, inTree ? 32 : std::numeric_limits<uint32_t>::max());
  if (stateIn.size() != rows.sequences || (writeState && stateOut.size() != rows.sequences) ||
      (inTree && (writeState || !captures.empty())) || inTree != static_cast<bool>(tape))
    throw std::invalid_argument("invalid short convolution");
  for (const ShortConvCapture &capture : captures)
    if (capture.sequence >= rows.sequences || !capture.row)
      throw std::invalid_argument("invalid short convolution capture");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  requireBuffers(buffers, shape, output, rows.rows);
  for (const metal::MetalBuffer &s : stateIn) requireBytes(s, stateExtent(shape, state), "short convolution state");
  for (const metal::MetalBuffer &s : stateOut)
    requireBytes(s, stateExtent(shape, state), "short convolution next state");
  for (const ShortConvCapture &capture : captures)
    requireBytes(capture.state, stateExtent(shape, state), "short convolution captured state");
  if (inTree) requireTape(tree, tape, shape, state, rows.rows);
  ShortConvolutionParams p = convolutionParams(shape, rows.rows, state);
  p.output = static_cast<uint32_t>(output);
  p.residual = residual ? 1u : 0u;
  p.write_state = writeState ? 1u : 0u;
  p.tree = inTree ? 1u : 0u;
  p.sequences = rows.sequences;
  p.dynamic_stride = buffers.dynamicStride;
  p.input_stride = inputStride(buffers, shape);
  p.input_column = buffers.inputColumn;
  const auto in = laneBindings(stateIn), out = laneBindings(writeState ? stateOut : stateIn);
  // The buffers a dispatch does not touch bind another of its buffers: the
  // dynamic taps and the tree the static taps, the residual, the tape and
  // the unused captures the output.
  const metal::MetalBuffer &taps = buffers.taps, &result = buffers.output;
  // The first SHORT_CONVOLUTION_CAPTURES captures with the convolution, each
  // later group in a dispatch of its own that writes them alone.
  size_t first = 0;
  do {
    const size_t count = std::min<size_t>(captures.size() - first, SHORT_CONVOLUTION_CAPTURES);
    p.captures = static_cast<uint32_t>(count);
    p.captures_only = first ? 1u : 0u;
    std::array<metal::MetalBuffer, SHORT_CONVOLUTION_CAPTURES> captured;
    for (size_t q = 0; q < captured.size(); ++q) {
      captured[q] = q < count ? captures[first + q].state : result;
      p.capture_sequences[q] = q < count ? captures[first + q].sequence : 0;
      p.capture_rows[q] = q < count ? captures[first + q].row : 0;
    }
    // Rows past a sequence's: its state after them, then each capture's.
    const uint32_t stateRows = kStateRows * (count ? 1 + uint32_t(count) : writeState ? 1 : 0);
    graph.add(sum ? "short_convolution_sum" : "short_convolution",
              {buffers.input, in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], out[0], out[1], out[2], out[3],
               out[4], out[5], out[6], out[7], taps, shape.dynamic() ? buffers.dynamicTaps : taps,
               residual ? buffers.residual : result, result, layout, inTree ? tree : taps, inTree ? tape : result,
               captured[0], captured[1], captured[2], captured[3]},
              p, {rows.bound + stateRows, groups(shape.channels), rows.sequences}, {kThreads, 1, 1});
    first += count;
  } while (first < captures.size());
}

void ShortConvolution::addConvolutionPair(metal::CommandGraph &graph, std::span<const ShortConvBuffers, 2> buffers,
                                          ShortConvShape shape, std::span<const ShortConvState, 2> states,
                                          std::span<const metal::MetalBuffer> stateIn, metal::MetalBuffer layout,
                                          const LayoutRows &rows, metal::MetalBuffer tree, metal::MetalBuffer tape) {
  requireShape(shape);
  // A verify block's rows, along its paths, into the tape (addConvolution).
  requireLayoutRows(rows, 32);
  const ShortConvBuffers &first = buffers[0], &second = buffers[1];
  if (stateIn.size() != rows.sequences || !tree || !tape || states[0].stride != states[1].stride ||
      inputStride(first, shape) != inputStride(second, shape) ||
      (shape.dynamic() && first.dynamicStride != second.dynamicStride))
    throw std::invalid_argument("invalid short convolution pair");
  // In one dispatch each writes its state's columns of the tape and its
  // output: the two's columns lie apart, and their outputs are not one view.
  const uint32_t apart = states[0].column > states[1].column ? states[0].column - states[1].column
                                                             : states[1].column - states[0].column;
  if (apart < shape.channels) throw std::invalid_argument("a short convolution pair's state columns overlap");
  if (first.output.sameView(second.output)) throw std::invalid_argument("a short convolution pair writes one output");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  for (size_t index = 0; index < buffers.size(); ++index) {
    requireState(shape, states[index]);
    requireBuffers(buffers[index], shape, ShortConvOutput::Plain, rows.rows);
    for (const metal::MetalBuffer &s : stateIn)
      requireBytes(s, stateExtent(shape, states[index]), "short convolution state");
    requireTape(tree, tape, shape, states[index], rows.rows);
  }
  ShortConvolutionParams p = convolutionParams(shape, rows.rows, states[0]);
  p.output = SHORT_CONVOLUTION_OUTPUT_PLAIN;
  p.tree = 1;
  p.sequences = rows.sequences;
  p.dynamic_stride = first.dynamicStride;
  p.input_stride = inputStride(first, shape);
  p.input_column = first.inputColumn;
  p.pair_input_column = second.inputColumn;
  p.pair_state_column = states[1].column;
  const auto in = laneBindings(stateIn);
  // A static shape binds each convolution's taps in its dynamic taps' slot,
  // which it does not read.
  graph.add("short_convolution_pair",
            {first.input, second.input, in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], first.taps,
             second.taps, shape.dynamic() ? first.dynamicTaps : first.taps,
             shape.dynamic() ? second.dynamicTaps : second.taps, first.output, second.output, std::move(layout),
             std::move(tree), std::move(tape)},
            p, {rows.bound, 2 * groups(shape.channels), rows.sequences}, {kThreads, 1, 1});
}

ShortConvolutionParams ShortConvolution::verifyParams(const ShortConvBuffers &buffers, ShortConvShape shape,
                                                      std::span<const metal::MetalBuffer> stateIn,
                                                      ShortConvState state, metal::MetalBuffer layout,
                                                      const LayoutRows &rows, metal::MetalBuffer tree,
                                                      metal::MetalBuffer tape) {
  requireShape(shape);
  requireState(shape, state);
  // A verify block's paths are row masks of 32 rows.
  requireLayoutRows(rows, 32);
  // It convolves the rows it computes: no view of wider input rows.
  if (shape.dynamic() || stateIn.size() != rows.sequences || !buffers.residual || !tree || !tape ||
      buffers.inputStride || buffers.inputColumn)
    throw std::invalid_argument("invalid short convolution of a verify step's computed rows");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  requireBytes(buffers.taps, uint64_t{shape.channels} * kTaps * sizeof(float), "short convolution taps");
  requireBytes(buffers.residual, floatRows(rows.rows, shape.channels), "short convolution residual");
  requireBytes(buffers.output, bfloat16Rows(rows.rows, shape.channels), "short convolution output");
  for (const metal::MetalBuffer &s : stateIn) requireBytes(s, stateExtent(shape, state), "short convolution state");
  requireTape(tree, tape, shape, state, rows.rows);
  ShortConvolutionParams p = convolutionParams(shape, rows.rows, state);
  p.output = SHORT_CONVOLUTION_OUTPUT_ROUNDED_SUM;
  p.residual = 1;
  p.tree = 1;
  p.sequences = rows.sequences;
  return p;
}

namespace {

// The commit's dispatch over `sequences` lanes of a step of `stepRows` rows,
// each lane's state after its whole path, or with pathRows a fold's after
// its path's first pathRows rows (short_convolution_commit).
void addPathState(metal::CommandGraph &graph, metal::MetalBuffer tape, std::span<const metal::MetalBuffer> stateIn,
                  std::span<const metal::MetalBuffer> stateOut, metal::MetalBuffer paths, uint32_t columns,
                  metal::MetalBuffer layout, uint32_t sequences, uint32_t stepRows, uint32_t pathRows) {
  if (!columns || stateIn.size() != sequences || stateOut.size() != sequences)
    throw std::invalid_argument("invalid short convolution commit");
  // The GPU's paths are of the lanes' verify rows, any of which they may hold.
  requireBytes(tape, bfloat16Rows(stepRows, columns), "short convolution tape");
  for (const metal::MetalBuffer &s : stateIn)
    requireBytes(s, ShortConvolution::stateBytes(columns), "short convolution state");
  for (const metal::MetalBuffer &s : stateOut)
    requireBytes(s, ShortConvolution::stateBytes(columns), "short convolution next state");
  requireBytes(paths, uint64_t{sequences} * sizeof(uint32_t), "short convolution accepted path");
  requireBytes(layout, uint64_t{sequences} * sizeof(SplashLaneRows), "short convolution layout");
  ShortConvolutionParams p{};
  p.rows = stepRows;
  p.channels = columns;
  p.sequences = sequences;
  p.state_stride = columns;
  p.captures = pathRows ? 1u : 0u;
  p.capture_rows[0] = pathRows;
  const auto in = laneBindings(stateIn), out = laneBindings(stateOut);
  graph.add("short_convolution_commit",
            {std::move(tape), in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], out[0], out[1], out[2], out[3],
             out[4], out[5], out[6], out[7], std::move(paths), std::move(layout)},
            p, {kStateRows, groups(columns), sequences}, {kThreads, 1, 1});
}

} // namespace

void ShortConvolution::addCommit(metal::CommandGraph &graph, metal::MetalBuffer tape,
                                 std::span<const metal::MetalBuffer> stateIn,
                                 std::span<const metal::MetalBuffer> stateOut, metal::MetalBuffer paths,
                                 uint32_t columns, metal::MetalBuffer layout, const LayoutRows &rows) {
  // A lane's path is a row mask of its rows.
  requireLayoutRows(rows, 32);
  addPathState(graph, std::move(tape), stateIn, stateOut, std::move(paths), columns, std::move(layout),
               rows.sequences, rows.rows, 0);
}

void ShortConvolution::addFold(metal::CommandGraph &graph, metal::MetalBuffer tape, metal::MetalBuffer stateIn,
                               metal::MetalBuffer fold, metal::MetalBuffer path, uint32_t columns,
                               metal::MetalBuffer layout, uint32_t stepRows, uint32_t pathRows) {
  // A path is a row mask of a lane's rows, at most 32 of the step's.
  if (!stepRows || !pathRows || pathRows > 32)
    throw std::invalid_argument("invalid short convolution fold");
  addPathState(graph, std::move(tape), std::span(&stateIn, 1), std::span(&fold, 1), std::move(path), columns,
               std::move(layout), 1, stepRows, pathRows);
}

} // namespace splash::ops
