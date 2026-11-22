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

ShortConvolutionParams convolutionParams(ShortConvShape shape, uint32_t rows, ShortConvState state) {
  ShortConvolutionParams p{};
  p.rows = rows;
  p.channels = shape.channels;
  p.heads = shape.dynamicHeads;
  p.state_stride = state.stride;
  p.state_column = state.column;
  p.divisor = divisor(shape);
  return p;
}

// The bytes of `rows` bf16 or fp32 rows of `width` values, and of a state
// matrix whose last row reaches the convolution's columns.
uint64_t bfloat16Rows(uint32_t rows, uint32_t width) { return uint64_t{rows} * width * sizeof(uint16_t); }
uint64_t floatRows(uint32_t rows, uint32_t width) { return uint64_t{rows} * width * sizeof(float); }
uint64_t stateExtent(ShortConvShape shape, ShortConvState state) {
  return rowBytes(kStateRows, state.stride, uint64_t{state.column} + shape.channels, sizeof(uint16_t));
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
      (inTree && (writeState || !captures.empty())) || inTree != static_cast<bool>(tape) ||
      (output == ShortConvOutput::Plain && residual) || (output == ShortConvOutput::RoundedSum && !residual) ||
      (shape.dynamic() && buffers.dynamicStride < shape.dynamicHeads * kTaps))
    throw std::invalid_argument("invalid short convolution");
  for (const ShortConvCapture &capture : captures)
    if (capture.sequence >= rows.sequences || !capture.row)
      throw std::invalid_argument("invalid short convolution capture");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  requireBytes(buffers.input, bfloat16Rows(rows.rows, shape.channels), "short convolution input");
  requireBytes(buffers.taps, uint64_t{shape.channels} * kTaps * sizeof(float), "short convolution taps");
  if (shape.dynamic())
    requireBytes(buffers.dynamicTaps,
                 rowBytes(rows.rows, buffers.dynamicStride, shape.dynamicHeads * kTaps, sizeof(uint16_t)),
                 "short convolution dynamic taps");
  if (residual)
    requireBytes(buffers.residual, sum ? bfloat16Rows(rows.rows, shape.channels) : floatRows(rows.rows, shape.channels),
                 "short convolution residual");
  requireBytes(buffers.output, sum ? floatRows(rows.rows, shape.channels) : bfloat16Rows(rows.rows, shape.channels),
               "short convolution output");
  for (const metal::MetalBuffer &s : stateIn) requireBytes(s, stateExtent(shape, state), "short convolution state");
  for (const metal::MetalBuffer &s : stateOut)
    requireBytes(s, stateExtent(shape, state), "short convolution next state");
  for (const ShortConvCapture &capture : captures)
    requireBytes(capture.state, stateExtent(shape, state), "short convolution captured state");
  if (inTree) {
    requireBytes(tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "short convolution tree rows");
    requireBytes(tape, rowBytes(rows.rows, state.stride, uint64_t{state.column} + shape.channels, sizeof(uint16_t)),
                 "short convolution tape");
  }
  ShortConvolutionParams p = convolutionParams(shape, rows.rows, state);
  p.output = static_cast<uint32_t>(output);
  p.residual = residual ? 1u : 0u;
  p.write_state = writeState ? 1u : 0u;
  p.tree = inTree ? 1u : 0u;
  p.sequences = rows.sequences;
  p.dynamic_stride = buffers.dynamicStride;
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

void ShortConvolution::addCommit(metal::CommandGraph &graph, metal::MetalBuffer tape,
                                 std::span<const metal::MetalBuffer> stateIn,
                                 std::span<const metal::MetalBuffer> stateOut, metal::MetalBuffer paths,
                                 uint32_t columns, metal::MetalBuffer layout, const LayoutRows &rows) {
  // A lane's path is a row mask of its rows.
  requireLayoutRows(rows, 32);
  if (!columns || stateIn.size() != rows.sequences || stateOut.size() != rows.sequences)
    throw std::invalid_argument("invalid short convolution commit");
  // The GPU's paths are of the lanes' verify rows, any of which they may hold.
  requireBytes(tape, bfloat16Rows(rows.rows, columns), "short convolution tape");
  for (const metal::MetalBuffer &s : stateIn)
    requireBytes(s, ShortConvolution::stateBytes(columns), "short convolution state");
  for (const metal::MetalBuffer &s : stateOut)
    requireBytes(s, ShortConvolution::stateBytes(columns), "short convolution next state");
  requireBytes(paths, uint64_t{rows.sequences} * sizeof(uint32_t), "short convolution accepted path");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "short convolution layout");
  ShortConvolutionParams p{};
  p.rows = rows.rows;
  p.channels = columns;
  p.sequences = rows.sequences;
  p.state_stride = columns;
  const auto in = laneBindings(stateIn), out = laneBindings(stateOut);
  graph.add("short_convolution_commit",
            {std::move(tape), in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], out[0], out[1], out[2], out[3],
             out[4], out[5], out[6], out[7], std::move(paths), std::move(layout)},
            p, {kStateRows, groups(columns), rows.sequences}, {kThreads, 1, 1});
}

} // namespace splash::ops
