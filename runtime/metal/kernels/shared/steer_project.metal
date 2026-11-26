#include "metal/abi/KernelABI.h"

// A layer's residual rows projected off one unit-norm direction
// (metal/abi/Steering.h, ops::Steering): h <- h - alpha (h . d) d, per row.
// A threadgroup a row; the row's dot product is reduced in threadgroup memory,
// then every thread subtracts its columns' share. fp32 throughout, as the
// residual stream is.
kernel void steer_project_f32(device float *rows [[buffer(0)]],
                              device const float *direction [[buffer(1)]],
                              constant SteerParams &p [[buffer(2)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
  if (row >= p.rows)
    return;
  device float *h = rows + ulong(row) * p.hidden;
  float dot = 0.0f;
  for (uint column = tid; column < p.hidden; column += STEER_THREADS)
    dot += h[column] * direction[column];
  threadgroup float partial[STEER_THREADS];
  partial[tid] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = STEER_THREADS / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      partial[tid] += partial[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float coefficient = p.alpha * partial[0];
  for (uint column = tid; column < p.hidden; column += STEER_THREADS)
    h[column] -= coefficient * direction[column];
}
