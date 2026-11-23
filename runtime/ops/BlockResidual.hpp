#pragma once

// Inkling 2's block attention residuals (kernels/shared/block_residual.metal,
// metal/abi/BlockResidual.h): per row, learned queries attend over the depth,
// the states of a row's completed blocks and its current partial one, each
// RMS normalized without a gain, and return the attention's mix of the
// normalized states. A layer's attention and MoE inputs, a loop pass's
// messages and the final norm's input are such mixes (1, 3 or 4 queries over
// 2 to 5 states). Rows are independent, so the mix keeps no state between
// steps: a target holds its states in its step scratch.

#include "metal/CommandGraph.hpp"
#include "metal/abi/BlockResidual.h"
#include "ops/FloatRows.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A normalized output: its multipliers and its bf16 rows.
struct BlockResidualNorm final {
  NormWeights weights;
  metal::MetalBuffer output;
};

struct BlockResidual final {
  // For each of `rows` rows and each query j (rows of `hidden` fp32 values in
  // `queries`, the GGUF tensor as stored), outputs[j]'s row: the mix of the
  // fp32 rows of `states` (each its own view and stride: separate buffers or
  // views of one bank), at most BLOCK_RESIDUAL_MAXIMUM_STATES states and
  // BLOCK_RESIDUAL_MAXIMUM_QUERIES queries of at most
  // BLOCK_RESIDUAL_MAXIMUM_HIDDEN values, the scores scaled by
  // 1 / sqrt(hidden). The outputs' rows are outputStride elements apart; an
  // output may be one of the states.
  // `norms`, at most one per output and all of one multiplier type, RMS
  // normalize the first norms.size() outputs into bf16 rows of `hidden`
  // values, the bits Normalization::addRmsRows makes of the fp32 outputs.
  static void add(metal::CommandGraph &graph, std::span<const StridedRows> states, metal::MetalBuffer queries,
                  std::span<const metal::MetalBuffer> outputs, uint32_t outputStride, uint32_t hidden, uint32_t rows,
                  std::span<const BlockResidualNorm> norms = {});
};

} // namespace splash::ops
