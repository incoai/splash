#pragma once

// The block residual mix (kernels/shared/block_residual_mix.metal, a decode
// step's rows; block_residual_mix_rows.metal, a prompt chunk's): learned
// queries attend over a row's states, the inputs of the blocks before it (its
// block stack) and its current rows, each RMS-normalized without weights; a
// query's output is the softmax-weighted sum of the normalized states, in
// fp32. What reads a mix's output is a weighted RMS norm writing a
// projection's input (addMixNorm), as after the hyper-connection mix
// (ops/HyperConnection.hpp), or a message gate (ops/MessageGate.hpp), which
// reads the fp32 outputs (addMix).

#include "metal/CommandGraph.hpp"
#include "metal/abi/BlockResidualMix.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

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
// Without stack planes the stack binds nothing.
struct BlockResidualStates final {
  metal::MetalBuffer stack;
  uint32_t planeRows = 0;
  metal::MetalBuffer current;
  bool currentFloat = false;
};

struct BlockResidualMix final {
  // Each query's output of `rows` rows as fp32 rows [rows][queries][hidden];
  // `queries` the F32 queries [queries][hidden].
  static void addMix(metal::CommandGraph &graph, const BlockResidualStates &states, metal::MetalBuffer queries,
                     metal::MetalBuffer output, BlockResidualShape shape, uint32_t rows);
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
