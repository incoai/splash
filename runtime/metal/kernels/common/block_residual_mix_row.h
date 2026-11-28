#pragma once

#include "metal/abi/BlockResidualMix.h"
#include "metal/kernels/common/short_convolution_row.h"

#include <metal_stdlib>

// A block residual mix's row (metal/abi/BlockResidualMix.h), as both its
// kernels read it (kernels/shared/block_residual_mix.metal, a threadgroup a
// row and query, and block_residual_mix_rows.metal, a threadgroup a row): its
// states, the stack's planes, then the current rows, bf16 (a layer's input)
// or fp32 (a residual inside a layer), Current, each read as fp32.
template <class Current> struct BlockResidualMixRow {
  device const bfloat *stack;
  device const Current *current;
  uint stack_states;
  uint plane;

  float state(uint s, uint d) const {
    return s < stack_states ? float(stack[ulong(s) * plane + d]) : float(current[d]);
  }
  uint states() const { return stack_states + 1; }
};

// The softmax, in place, of each of `queries` queries' scores of `states`
// states (a query's `stride` apart in `weights`), each scaled by p.scale,
// which thread 0 of a mix takes once its scores are parked. Always inlined:
// compiled as a function of its own, then inlined, it reordered the blocks
// of every mix kernel.
__attribute__((always_inline)) inline void block_residual_mix_softmax(threadgroup float *weights, uint queries,
                                                                      uint stride, uint states,
                                                                      constant BlockResidualMixParams &p) {
  for (uint q = 0; q < queries; ++q) {
    threadgroup float *query = weights + q * stride;
    float maximum = -INFINITY;
    for (uint s = 0; s < states; ++s) maximum = max(maximum, query[s] * p.scale);
    float total = 0.0f;
    for (uint s = 0; s < states; ++s) {
      query[s] = exp(query[s] * p.scale - maximum);
      total += query[s];
    }
    for (uint s = 0; s < states; ++s) query[s] /= total;
  }
}

// Query q's output at column d: the weighted sum of the normalized states,
// from the parked norms and weights in `shared` (either kernel's), a
// query's weights BLOCK_RESIDUAL_MIX_MAXIMUM_STATES apart.
template <class Current, class Shared>
inline float block_residual_mix_output(BlockResidualMixRow<Current> mix, uint q, uint d,
                                       threadgroup const Shared &shared) {
  float sum = 0.0f;
  for (uint s = 0; s < mix.states(); ++s)
    sum += shared.weights[q * BLOCK_RESIDUAL_MIX_MAXIMUM_STATES + s] * (mix.state(s, d) * shared.inverse[s]);
  return sum;
}

// The lane of a decode step's row `row` (its entry of the step's layout, of
// `sequences` lanes), or SPLASH_MAXIMUM_BATCH_WIDTH for a row of no lane.
inline uint block_residual_mix_lane(device const SplashLaneRows *layout, uint sequences, uint row) {
  for (uint s = 0; s < min(sequences, SPLASH_MAXIMUM_BATCH_WIDTH); ++s) {
    const SplashLaneRows rows = layout[s];
    if (row >= rows.first && row - rows.first < rows.rows) return s;
  }
  return SPLASH_MAXIMUM_BATCH_WIDTH;
}

// The current row `row` of a decode step's mix whose current rows are its
// mixer's output convolution (BlockResidualMixConvolutionParams): each of
// the row's channels as short_convolution_sum writes it, along the row's
// path in its lane's rows (`rows`, the lane's state before them
// `state_in`), onto the residual where c.residual adds one, into `current`,
// and the row's input into the tape; each thread the columns
// tid + Threads i, which it alone then reads and mixes, so no barrier
// follows in a row and query's mix (one over the device writes took a
// convolved mix about a microsecond). A many-row mix does not convolve
// (ops::BlockResidualMix::convolves).
template <uint Threads>
inline void block_residual_mix_convolve(device const bfloat *input, device const bfloat *state_in,
                                        device const float *taps, device const bfloat *residual,
                                        device float *current, SplashLaneRows rows,
                                        device const SplashTreeRow *tree, device bfloat *tape,
                                        constant ShortConvolutionParams &c, uint row, uint tid) {
  device const bfloat *sequence = input + c.input_column + ulong(rows.first) * c.input_stride;
  const uint path = tree[row].ancestors;
  const int depth = int(tree[row].depth);
  for (uint channel = tid; channel < c.channels; channel += Threads) {
    const ulong column = c.state_column + channel;
    const auto at = [&](int t) -> bfloat {
      return t >= 0 ? sequence[ulong(t) * c.input_stride + channel]
                    : state_in[ulong(int(SHORT_CONVOLUTION_STATE_ROWS) + t) * c.state_stride + column];
    };
    const bfloat own = sequence[ulong(row - rows.first) * c.input_stride + channel];
    // Static taps: the dynamic taps' slot is never read.
    short_convolution_output(at, taps, sequence, residual, current, c, channel, row, path, depth, own);
    tape[ulong(row) * c.state_stride + column] = own;
  }
}
