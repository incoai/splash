#pragma once

#include <metal_stdlib>

using namespace metal;

// The sums over a threadgroup of Simdgroups simdgroups of the first `count`
// of a thread's `values`, returned in every thread: per simdgroup, then over
// the simdgroups in order, so that a row's sums are the same bits whatever
// the rows beside it. `scratch` holds Simdgroups * N floats.
template <uint Simdgroups, uint N>
inline void threadgroup_row_sums(thread float (&values)[N], uint count, threadgroup float *scratch, uint lane,
                                 uint sg) {
  for (uint i = 0; i < count; ++i) {
    const float sum = simd_sum(values[i]);
    if (lane == 0) scratch[sg * N + i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < count; ++i) {
    float total = 0.0f;
    for (uint g = 0; g < Simdgroups; ++g) total += scratch[g * N + i];
    values[i] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
