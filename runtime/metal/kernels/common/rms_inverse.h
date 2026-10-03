#pragma once

#include "metal/abi/KernelABI.h"

// The epsilon of every RMS norm.
constant constexpr float kRmsEpsilon = 1e-6f;

// The inverse RMS of `width` values from each thread's sum of squares,
// reduced per simdgroup and then over the first Simdgroups simdgroups, in
// order, by thread 0, and returned to every thread. `reductions` holds a
// partial per simdgroup. This and row_squares keep the source order of their
// float additions (reassociate(off), scoped to their bodies), whichever file
// includes them.
template <ushort Simdgroups = 8>
inline float rms_inverse_of_sums(float sum, uint width, threadgroup float *reductions,
                                 uint thread_index, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < Simdgroups; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / width + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return reductions[0];
}

// A thread's squares of columns thread_index + 256 i of a row in device or
// threadgroup memory, in column order.
template <class Row>
inline float row_squares(Row row, uint width, uint thread_index) {
#pragma clang fp reassociate(off)
  float sum = 0.0f;
  for (uint column = thread_index; column < width; column += 256) {
    float value = float(row[column]);
    sum += value * value;
  }
  return sum;
}

// Inverse RMS of one row, reduced across the 256-thread group and returned to
// every thread.
inline float rms_inverse(device const bfloat *row_input, uint width,
                         threadgroup float *reductions, uint thread_index,
                         uint lane, uint simd_group) {
  return rms_inverse_of_sums(row_squares(row_input, width, thread_index), width, reductions,
                             thread_index, lane, simd_group);
}
