#pragma once

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
