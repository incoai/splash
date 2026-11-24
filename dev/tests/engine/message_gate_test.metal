// The message gate's reference path (message_gate_reference.h): the
// normalization and concatenation before its down projection and the gated
// row's norm after its forget and gain projections, a threadgroup a row,
// built with the production flags, as the engine built them while it ran
// them.
#pragma clang fp reassociate(off)
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"
#include "message_gate_reference.h"

#include <metal_stdlib>

using namespace metal;

// A threadgroup of kThreads threads a row, each thread the columns
// tid + kThreads i.
constant constexpr uint kThreads = MESSAGE_GATE_REFERENCE_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;

// The sums over the threadgroup of the first `count` of a thread's `values`,
// returned in every thread: per simdgroup, then over the simdgroups in
// order, so that a row's sums are the same bits whatever the rows beside it.
// `scratch` holds kSimdgroups * N floats.
template <uint N>
inline void message_gate_test_row_sums(thread float (&values)[N], uint count, threadgroup float *scratch, uint lane,
                                       uint sg) {
  for (uint i = 0; i < count; ++i) {
    const float sum = simd_sum(values[i]);
    if (lane == 0) scratch[sg * N + i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < count; ++i) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[g * N + i];
    values[i] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A message gate's input to its down projection: the row and its message,
// each RMS-normalized without weights, side by side in a bf16 row of
// 2 hidden values (the source's down projection is a bf16 Linear).
kernel void message_gate_test_prepare(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                                      device bfloat *concat [[buffer(2)]],
                                      constant MessageGateReferenceParams &p [[buffer(3)]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * 2];
  if (row >= p.rows) return;
  device const float *xr = x + ulong(row) * p.x_stride + p.x_column;
  device const float *mr = message + ulong(row) * p.message_stride + p.message_column;
  float squares[2] = {0.0f, 0.0f};
  for (uint d = tid; d < p.hidden; d += kThreads) {
    squares[0] += xr[d] * xr[d];
    squares[1] += mr[d] * mr[d];
  }
  message_gate_test_row_sums(squares, 2, scratch, lane, sg);
  const float xs = rsqrt(squares[0] / p.hidden + kRmsEpsilon), ms = rsqrt(squares[1] / p.hidden + kRmsEpsilon);
  device bfloat *out = concat + ulong(row) * 2 * p.hidden;
  for (uint d = tid; d < p.hidden; d += kThreads) {
    out[d] = bfloat(xr[d] * xs);
    out[p.hidden + d] = bfloat(mr[d] * ms);
  }
}

// The gated row, (1 - tanh(forget)) x + tanh(gain) message, in fp32 from the
// forget and gain projections' fp32 rows [rows][hidden], through the
// weighted RMS norm after it into bf16 rows.
inline float message_gate_test_gated(device const float *xr, device const float *mr, device const float *forget,
                                     device const float *gain, uint d) {
  return (1.0f - precise::tanh(forget[d])) * xr[d] + precise::tanh(gain[d]) * mr[d];
}
kernel void message_gate_test_norm(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                                   device const float *forget [[buffer(2)]], device const float *gain [[buffer(3)]],
                                   device const float *norm [[buffer(4)]], device bfloat *output [[buffer(5)]],
                                   constant MessageGateReferenceParams &p [[buffer(6)]],
                                   uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups];
  if (row >= p.rows) return;
  device const float *xr = x + ulong(row) * p.x_stride + p.x_column;
  device const float *mr = message + ulong(row) * p.message_stride + p.message_column;
  device const float *f = forget + ulong(row) * p.hidden, *g = gain + ulong(row) * p.hidden;
  float squares[1] = {0.0f};
  for (uint d = tid; d < p.hidden; d += kThreads) {
    const float v = message_gate_test_gated(xr, mr, f, g, d);
    squares[0] += v * v;
  }
  message_gate_test_row_sums(squares, 1, scratch, lane, sg);
  const float inverse = rsqrt(squares[0] / p.hidden + kRmsEpsilon);
  for (uint d = tid; d < p.hidden; d += kThreads)
    output[ulong(row) * p.hidden + d] = bfloat((message_gate_test_gated(xr, mr, f, g, d) * inverse) * norm[d]);
}
