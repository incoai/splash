// The control vector on a Qwen3.5-family target's residual
// (abi/ResidualSteer.h, ops/ResidualSteer.cpp, model/ControlVector.hpp):
// after a steered layer's FFN, before the DFlash2 capture. The residual is
// bf16; a projection's dot is summed in fp32 over the row's threadgroup, and
// every value is read and written as fp32.

#include "metal/abi/ResidualSteer.h"

#include <metal_stdlib>

using namespace metal;

constant constexpr uint kThreads = RESIDUAL_STEER_THREADS;
constant constexpr ushort kSimdgroups = kThreads / 32;

// A control vector after a steered layer (ResidualSteerParams): a
// threadgroup per row. A projection sums the row's dot with the direction in
// fp32 over the threadgroup, then takes scale times it along the direction
// off every value; an addition adds the scaled direction.
kernel void residual_steer(device bfloat *residual [[buffer(0)]],
                           device const float *directions [[buffer(1)]],
                           constant ResidualSteerParams &p [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float partial[kSimdgroups];
  device bfloat *x = residual + ulong(row) * p.hidden;
  device const float *v = directions + ulong(p.layer) * p.hidden;
  if (p.mode == RESIDUAL_STEER_ADD) {
    for (uint d = tid; d < p.hidden; d += kThreads) x[d] = bfloat(fma(p.scale, v[d], float(x[d])));
    return;
  }
  float dot = 0.0f;
  for (uint d = tid; d < p.hidden; d += kThreads) dot = fma(float(x[d]), v[d], dot);
  dot = simd_sum(dot);
  if (lane == 0) partial[sg] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < kSimdgroups; ++g) total += partial[g];
  const float along = p.scale * total;
  for (uint d = tid; d < p.hidden; d += kThreads) x[d] = bfloat(fma(-along, v[d], float(x[d])));
}
