#pragma once

// The block residual mix (kernels/shared/block_residual_mix.metal, a call of
// at most BLOCK_RESIDUAL_MIX_QUERY_ROWS rows; block_residual_mix_rows.metal,
// a call of more): learned queries attend over a row's states, the inputs of
// the blocks before it (its block stack) and its current rows, each
// RMS-normalized without weights; a query's output is the softmax-weighted
// sum of the normalized states, in fp32. What reads a mix's output is a
// weighted RMS norm writing a projection's input (addMixNorm), as after the
// hyper-connection mix (ops/HyperConnection.hpp), or a message gate
// (ops/MessageGate.hpp), which reads the fp32 outputs (addMix), and its
// fused kernels the outputs normalized, which the mix writes beside them.

#include "metal/CommandGraph.hpp"
#include "metal/abi/BlockResidualMix.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/ShortConvolution.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A mix of `queries` queries over `stack` stack states and the current rows,
// of `hidden` values each.
struct BlockResidualShape final {
  uint32_t hidden = 0;
  uint32_t stack = 0;
  uint32_t queries = 0;
};

// A mix's states: `stack` bf16 planes of rows of `hidden` values, plane s
// planeRows rows from plane s - 1 (a layer's inputs, as the step's hidden
// rows hold them), then the current rows, rows of `hidden` values, bf16 (a
// layer's input) or fp32 (currentFloat: the residual inside a layer).
// Without stack planes the stack binds nothing. With `convolution` the
// current rows, fp32, are that verify step's convolution of its rows (Sum,
// into its buffers.output, which `current` views), which each row's
// threadgroup of the mix, a call's one mix of one query, runs before it
// mixes the row (the _convolved kernels): the convolution's dispatch fewer,
// the same bits. Only a call of rows a threadgroup mixes a row and query
// (BlockResidualMix::convolves) takes one.
struct BlockResidualStates final {
  metal::MetalBuffer stack;
  uint32_t planeRows = 0;
  metal::MetalBuffer current;
  bool currentFloat = false;
  const ShortConvVerify *convolution = nullptr;
};

// One mix of a call of addMix: its states, its F32 queries
// [queries][hidden], its shape and its output, fp32 rows
// [rows][queries][hidden]; and, for the message gates' fused kernels that
// read its rows normalized (MessageGate::readsNormalized), each query's
// output through their input normalization (kernels/common/
// message_gate_normalize.h) into bf16 rows [rows][queries][hidden]
// (`normalized`), in the mix's dispatch where a dispatch of its own ran it:
// all of a call's mixes or none.
struct BlockResidualMixInput final {
  BlockResidualStates states;
  metal::MetalBuffer queries;
  BlockResidualShape shape;
  metal::MetalBuffer output;
  metal::MetalBuffer normalized;
};

struct BlockResidualMix final {
  // Whether a call of `rows` rows runs the _rows kernels, a threadgroup a
  // row, past BLOCK_RESIDUAL_MIX_QUERY_ROWS rows; else a threadgroup a row
  // and query (ExecutionPlans::decodeRowClasses splits a target's decode
  // classes there).
  [[nodiscard]] static constexpr bool rowKernels(uint32_t rows) noexcept {
    return rows > BLOCK_RESIDUAL_MIX_QUERY_ROWS;
  }
  // Whether a mix of `rows` rows may run its current rows' convolution itself
  // (BlockResidualStates::convolution): where a threadgroup mixes a row and
  // query. A call of more rows runs the convolution's dispatch, then the
  // _rows mix, as a prompt chunk does: a many-row mix that convolved its rows
  // (1.7.5's block_residual_mix_rows_convolved) wrote its gate's fp32 rows
  // 3.6e-07 off the two dispatches' on Apple11 (a 12-core M6: 45,448 of
  // 165,888 values in 37 of 72 rows; none on Apple9 or Apple10), so a row's
  // bits depended on how many rows shared its step. The mix kernels leave
  // contraction to the compiler, and Apple11's contracted the convolved
  // kernel's sums otherwise than the mix's own: with contraction forbidden in
  // block_residual_mix_rows.metal the two agreed on the M6 (a scratch build).
  // That pragma also changes how every family compiles a prompt chunk's
  // many-row mix, whose bits were not compared, so it is not what this does.
  [[nodiscard]] static constexpr bool convolves(uint32_t rows) noexcept { return !rowKernels(rows); }
  // Each query's output of each of `mixes` (1 to
  // BLOCK_RESIDUAL_MIX_MAXIMUM_MIXES, of one hidden size and one type of
  // current rows) of `rows` rows into its output, in one dispatch: mixes that
  // read nothing another writes (a gated application's messages and its
  // mixer's inputs) run side by side.
  static void addMix(metal::CommandGraph &graph, std::span<const BlockResidualMixInput> mixes, uint32_t rows);
  // Each query's output through its own weighted RMS norm (F32, as a GGUF
  // stores norms) into its own bf16 rows [rows][hidden], one norm and output
  // a query (at most BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS). The projections
  // reading them prepare their own tables, so the scratch describes nothing.
  static PreparedInput addMixNorm(metal::CommandGraph &graph, const BlockResidualStates &states,
                                  metal::MetalBuffer queries, std::span<const NormWeights> norms,
                                  std::span<const metal::MetalBuffer> outputs, BlockResidualShape shape,
                                  uint32_t rows);
};

} // namespace splash::ops
