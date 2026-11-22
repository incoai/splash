#pragma clang fp reassociate(off)
#include "metal/abi/KernelABI.h"
#include "metal/abi/MessageGate.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/row_sums.h"

#include <metal_stdlib>

using namespace metal;

// Message gates (metal/abi/MessageGate.h) over fp32 rows: a threadgroup of
// kGateThreads threads a row, each thread the columns tid + kGateThreads i.
constant constexpr uint kGateThreads = MESSAGE_GATE_THREADS;
constant constexpr uint kGateSimdgroups = kGateThreads / 32;

// A message gate's input to its down projection: the row and its message,
// each RMS-normalized without weights, side by side in a bf16 row of
// 2 hidden values (the source's down projection is a bf16 Linear).
kernel void message_gate_prepare(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                                 device bfloat *concat [[buffer(2)]], constant MessageGateParams &p [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kGateSimdgroups * 2];
  if (row >= p.rows) return;
  device const float *xr = x + ulong(row) * p.x_stride + p.x_column;
  device const float *mr = message + ulong(row) * p.message_stride + p.message_column;
  float squares[2] = {0.0f, 0.0f};
  for (uint d = tid; d < p.hidden; d += kGateThreads) {
    squares[0] += xr[d] * xr[d];
    squares[1] += mr[d] * mr[d];
  }
  threadgroup_row_sums<kGateSimdgroups>(squares, 2, scratch, lane, sg);
  const float xs = rsqrt(squares[0] / p.hidden + kRmsEpsilon), ms = rsqrt(squares[1] / p.hidden + kRmsEpsilon);
  device bfloat *out = concat + ulong(row) * 2 * p.hidden;
  for (uint d = tid; d < p.hidden; d += kGateThreads) {
    out[d] = bfloat(xr[d] * xs);
    out[p.hidden + d] = bfloat(mr[d] * ms);
  }
}

// The gated row, (1 - tanh(forget)) x + tanh(gain) message, in fp32 from the
// forget and gain projections' fp32 rows [rows][hidden], through the
// weighted RMS norm after it into bf16 rows.
inline float message_gated(device const float *xr, device const float *mr, device const float *forget,
                           device const float *gain, uint d) {
  return (1.0f - precise::tanh(forget[d])) * xr[d] + precise::tanh(gain[d]) * mr[d];
}
kernel void message_gate_norm(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                              device const float *forget [[buffer(2)]], device const float *gain [[buffer(3)]],
                              device const float *norm [[buffer(4)]], device bfloat *output [[buffer(5)]],
                              constant MessageGateParams &p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kGateSimdgroups];
  if (row >= p.rows) return;
  device const float *xr = x + ulong(row) * p.x_stride + p.x_column;
  device const float *mr = message + ulong(row) * p.message_stride + p.message_column;
  device const float *f = forget + ulong(row) * p.hidden, *g = gain + ulong(row) * p.hidden;
  float squares[1] = {0.0f};
  for (uint d = tid; d < p.hidden; d += kGateThreads) {
    const float v = message_gated(xr, mr, f, g, d);
    squares[0] += v * v;
  }
  threadgroup_row_sums<kGateSimdgroups>(squares, 1, scratch, lane, sg);
  const float inverse = rsqrt(squares[0] / p.hidden + kRmsEpsilon);
  for (uint d = tid; d < p.hidden; d += kGateThreads)
    output[ulong(row) * p.hidden + d] = bfloat((message_gated(xr, mr, f, g, d) * inverse) * norm[d]);
}
