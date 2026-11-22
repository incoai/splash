#pragma once

// Inkling 2's short convolution (kernels/shared/short_convolution.metal,
// metal/abi/ShortConvolution.h): a causal depthwise convolution of
// SHORT_CONVOLUTION_TAPS taps added to its input and scaled, over the keys and
// values a layer projects (with a global layer's dynamic per-head taps), its
// attention's output and its MoE output. Each sequence keeps the last
// SHORT_CONVOLUTION_HISTORY inputs of each of its layer applications'
// convolutions as its state, its history, which a prompt's chunks carry
// from one to the next (addConvolution), a prefix cache's state captures
// inside a chunk (addHistory), and a verify's commit advances along the path
// it accepted (addCommit), as ops::NgramEmbedding does for its own dilated
// convolution. The math is fp32; a history keeps its inputs as `state`
// elements, and a row reads its past inputs as a history keeps them, so a
// prompt's output is the same however its chunks and steps split it.

#include "metal/CommandGraph.hpp"
#include "metal/abi/ShortConvolution.h"
#include "metal/abi/VerifyTree.h"
#include "ops/FloatRows.hpp"
#include "ops/LaneRows.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A convolution of `channels` channels whose inputs are `input` elements and
// whose history keeps `state` elements; with headChannels, a dynamic
// convolution whose rows add their own per-head taps, heads of headChannels
// channels each.
struct ShortConvolutionShape final {
  uint32_t channels = 0;
  uint32_t headChannels = 0;
  FloatElement input = FloatElement::Float32;
  FloatElement state = FloatElement::BFloat16;
  [[nodiscard]] constexpr bool dynamic() const noexcept { return headChannels != 0; }
  [[nodiscard]] constexpr uint32_t heads() const noexcept { return dynamic() ? channels / headChannels : 0; }
  // The output's scale: 1 / sqrt(taps + 1), with dynamic taps 1 / sqrt(2 taps + 1).
  [[nodiscard]] float scale() const noexcept;
};

// The bytes of a history of `channels` channels of `state` elements, its rows
// `stride` elements apart (channels when 0).
[[nodiscard]] uint64_t shortConvolutionHistoryBytes(FloatElement state, uint32_t channels, uint32_t stride = 0);

// The rows of a convolution: their inputs, the dynamic taps of each (a
// dynamic convolution's: heads x SHORT_CONVOLUTION_TAPS values of the input's
// type per row, head h's lag l at h * taps + l, lag 0 the row's own input),
// the static taps (the GGUF tensor as stored: F32, each channel's taps
// together, its last the current row's), and the fp32 output, which, with
// accumulate, adds to the destination's values. The output may not overlap
// the inputs.
struct ShortConvolutionBuffers final {
  StridedRows input;
  StridedRows dynamic;
  metal::MetalBuffer taps;
  StridedRows output;
  bool accumulate = false;
};

// The histories of a command's sequences, one buffer each (a view of the
// sequence's state at the convolution's first channel), rows `stride`
// elements apart: `in` before each sequence's rows and, when `out` holds
// them, the history after them, which may not be the same memory.
struct ShortConvolutionHistories final {
  std::span<const metal::MetalBuffer> in;
  std::span<const metal::MetalBuffer> out;
  uint32_t stride = 0;
};

// A verify step's rows as trees (a SplashTreeRow each, metal/abi/
// VerifyTree.h), and the tape its convolution writes each row's input to,
// at its packed row, as a history keeps it, from which the commit writes the
// history after the path the lane accepts.
struct ShortConvolutionTree final {
  metal::MetalBuffer rows;
  StridedRows tape;
  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(rows); }
};

// A commit's units: `units` convolutions of `width` channels each (a target's
// convolutions of each of its layer applications, side by side), unit u's
// history at u x unitStride elements of a lane's state (its rows
// historyStride apart) and its tape at u x tapeUnitStride elements of the
// tape (the step's packed rows tapeStride apart).
struct ShortConvolutionUnits final {
  uint32_t width = 0;
  uint32_t units = 0;
  uint32_t historyStride = 0;
  uint32_t unitStride = 0;
  uint32_t tapeStride = 0;
  uint32_t tapeUnitStride = 0;
  FloatElement state = FloatElement::BFloat16;
};

struct ShortConvolution final {
  // The convolution of a command's sequences' rows (a prompt chunk's
  // sequences, or a verify step's lanes) as its layout holds them (`layout`,
  // a SplashLaneRows a sequence, of `rows`), in one dispatch: sequence s's
  // past inputs before its first row are those of histories.in[s], and with
  // histories.out its history after its rows goes to histories.out[s]. With
  // `tree`, the rows of verify blocks of at most 32 rows, whose past inputs
  // follow each row's path, writing each row's input to the tape; their
  // commit then writes their history (addCommit).
  static void addConvolution(metal::CommandGraph &graph, const ShortConvolutionBuffers &buffers,
                             ShortConvolutionHistories histories, ShortConvolutionShape shape,
                             metal::MetalBuffer layout, const LayoutRows &rows, const ShortConvolutionTree &tree = {});
  // The history after the first `rows` rows of one sequence's prefill (a
  // state captured inside its command), from its history before them
  // (historyIn) and its inputs, into historyOut; both histories' rows are
  // historyStride elements apart.
  static void addHistory(metal::CommandGraph &graph, const StridedRows &input, metal::MetalBuffer historyIn,
                         metal::MetalBuffer historyOut, uint32_t historyStride, ShortConvolutionShape shape,
                         uint32_t rows);
  // The history after each lane's accepted path (`paths`, a row mask the GPU
  // wrote a lane, metal/abi/VerifyTree.h) of its verify block's rows on the
  // tape, as the step's layout holds them (`layout`, a SplashLaneRows a lane,
  // of `rows`, at most 32 rows a lane), for every unit in one dispatch: lane
  // l's histories before it historyIn[l], after it historyOut[l], which may
  // not be the same memory.
  static void addCommit(metal::CommandGraph &graph, metal::MetalBuffer tape,
                        std::span<const metal::MetalBuffer> historyIn, std::span<const metal::MetalBuffer> historyOut,
                        metal::MetalBuffer paths, const ShortConvolutionUnits &units, metal::MetalBuffer layout,
                        const LayoutRows &rows);
};

} // namespace splash::ops
