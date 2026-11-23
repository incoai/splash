#pragma once

#include "metal/abi/BlockResidualMix.h"

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
