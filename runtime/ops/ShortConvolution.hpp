#pragma once

// The short convolution (kernels/shared/short_convolution.metal): a causal
// depthwise convolution of SHORT_CONVOLUTION_TAPS taps over a projection's
// output rows, in fp32, whose state is the input rows before a sequence's
// first (metal/abi/ShortConvolution.h). It runs over a command's sequences
// as the n-gram embedding's convolution does (ops/NgramEmbedding.hpp): a
// prompt chunk's rows in order, writing each sequence's state after them, or
// a decode step's rows along their paths of each lane's tree, whose commit
// writes the state after the path the lane accepts.

#include "metal/CommandGraph.hpp"
#include "metal/abi/ShortConvolution.h"
#include "ops/LaneRows.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A convolution of `channels` channels, its taps static, or with
// dynamicHeads heads the static taps plus each row's own taps of each head
// (channels / dynamicHeads channels each), which a projection of the row
// writes.
struct ShortConvShape final {
  uint32_t channels = 0;
  uint32_t dynamicHeads = 0;
  [[nodiscard]] constexpr bool dynamic() const noexcept { return dynamicHeads != 0; }
};

// What a convolution writes of bf16(y): bf16 rows of it (Plain); fp32 rows
// of a bf16 residual plus it, or of it alone without a residual (Sum: the
// residual inside a layer); bf16 rows of an fp32 residual plus it, rounded
// (RoundedSum: a layer's output, as the source rounds it).
enum class ShortConvOutput : uint8_t {
  Plain = SHORT_CONVOLUTION_OUTPUT_PLAIN,
  Sum = SHORT_CONVOLUTION_OUTPUT_SUM,
  RoundedSum = SHORT_CONVOLUTION_OUTPUT_ROUNDED_SUM,
};

// A convolution's rows: its input, bf16 [rows][channels], as the projection
// writes it; its static taps, F32 [channels][SHORT_CONVOLUTION_TAPS] as a
// GGUF stores them, a conv1d weight's order (column TAPS - 1 - k weighs the
// input k rows back, the last the row itself); a dynamic convolution's
// per-row taps by lag (tap k of a head the input k rows back), bf16
// rows of dynamicStride values (its projection's output, whole tiles wide)
// of which the first dynamicHeads x SHORT_CONVOLUTION_TAPS are used; a sum's
// residual [rows][channels], bf16 for Sum (which may have none) or fp32 for
// RoundedSum; and its output [rows][channels], fp32 for Sum, else bf16.
struct ShortConvBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer taps;
  metal::MetalBuffer dynamicTaps;
  uint32_t dynamicStride = 0;
  metal::MetalBuffer residual;
  metal::MetalBuffer output;
};

// A convolution's place in a lane's state matrix, SHORT_CONVOLUTION_STATE_ROWS
// bf16 rows of `stride` columns, oldest row first: its `channels` columns from
// `column`. A target lays all its convolutions side by side in one matrix,
// so that every convolution binds the lanes' own buffers and one commit
// writes them all; a decode step's tape has the matrix's columns.
struct ShortConvState final {
  uint32_t column = 0;
  uint32_t stride = 0;
};

// A state captured inside a prompt sequence (a cached state's boundary):
// the state after the first `row` rows of sequence `sequence` of the
// command's layout, into `state`, a state matrix as the sequence's own.
struct ShortConvCapture final {
  uint32_t sequence = 0;
  uint32_t row = 0;
  metal::MetalBuffer state;
};

struct ShortConvolution final {
  // The bytes of a state matrix of `columns` columns.
  [[nodiscard]] static constexpr uint64_t stateBytes(uint32_t columns) noexcept {
    return uint64_t{SHORT_CONVOLUTION_STATE_ROWS} * columns * sizeof(uint16_t);
  }
  // The convolution of a command's sequences' rows as its layout holds them
  // (`layout`, a SplashLaneRows a sequence, of `rows`), in one dispatch
  // whose grid spans the sequences' bound: sequence s's taps before its
  // first row read stateIn[s], and with stateOut (a prompt's) its state after
  // its rows goes to stateOut[s], another buffer than stateIn[s], and each
  // capture's state to its own buffer (SHORT_CONVOLUTION_CAPTURES of them in
  // the convolution's dispatch, the rest in dispatches of their own). With
  // `tree`, the rows' places (a SplashTreeRow a row) of verify blocks of at
  // most 32 rows, whose taps follow each row's path, and `tape`, rows of the
  // state's stride for the command's rows, into which each row's input goes
  // at the state's columns; the commit then writes the lanes' states
  // (addCommit).
  static void addConvolution(metal::CommandGraph &graph, const ShortConvBuffers &buffers, ShortConvShape shape,
                             ShortConvOutput output, std::span<const metal::MetalBuffer> stateIn,
                             std::span<const metal::MetalBuffer> stateOut, ShortConvState state,
                             metal::MetalBuffer layout, const LayoutRows &rows,
                             std::span<const ShortConvCapture> captures = {}, metal::MetalBuffer tree = {},
                             metal::MetalBuffer tape = {});
  // The state after each lane's accepted path (`paths`, a row mask the GPU
  // wrote a lane, metal/abi/VerifyTree.h) of its verify block's rows in the
  // tape the step's convolutions wrote, as the step's layout holds them
  // (`layout`, a SplashLaneRows a lane, of `rows`, at most 32 rows a lane),
  // of every column of the lanes' state matrices of `columns` columns, in one
  // dispatch: lane l's state before it stateIn[l], after it stateOut[l].
  static void addCommit(metal::CommandGraph &graph, metal::MetalBuffer tape,
                        std::span<const metal::MetalBuffer> stateIn, std::span<const metal::MetalBuffer> stateOut,
                        metal::MetalBuffer paths, uint32_t columns, metal::MetalBuffer layout,
                        const LayoutRows &rows);
};

} // namespace splash::ops
