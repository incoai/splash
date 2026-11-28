#pragma once

#include "metal/abi/MessageGate.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

// The partials and the inverse message_gate_normalize parks in threadgroup
// memory: a partial a simdgroup of its first MESSAGE_GATE_THREADS threads,
// then the inverse.
constant constexpr uint kMessageGateNormalizeSlots = MESSAGE_GATE_THREADS / 32 + 1;

// A message gate's input normalization (metal/abi/MessageGate.h) of an fp32
// row of `hidden` values, whole float4s, into `normalized`, the bf16 row its
// fused down projection reads: its inverse RMS without weights from its
// squares by quads, thread t of the threadgroup's first MESSAGE_GATE_THREADS
// adding quads t + MESSAGE_GATE_THREADS i in order, each quad's squares in
// order, reduced by simd_sum and then over those threads' simdgroups in
// order by thread 0; then each value times it, rounded to bf16, by the
// threadgroup's `Threads` threads. The mixes that write a gate's inputs
// normalize each row they write: a row and query's mix by the same formula
// from its own sums, in its own order (kernels/shared/block_residual_mix.metal),
// a many-row mix by this one, every thread of the threadgroup past a barrier
// over the row's writes (block_residual_mix_rows.metal); the gates' tests
// normalize their inputs by this one (dev/tests/engine/
// message_gate_test.metal). The sums keep their order (reassociate(off),
// scoped to its body), whichever file includes it.
template <uint Threads>
__attribute__((noinline)) void message_gate_normalize(device const float *row, device bfloat *normalized, uint hidden,
                                                      uint tid, uint lane, uint sg, threadgroup float *partials) {
#pragma clang fp reassociate(off)
  constexpr uint kThreads = MESSAGE_GATE_THREADS, kSimdgroups = kThreads / 32;
  static_assert(Threads % kThreads == 0, "the normalization's threads are whole simdgroups of the threadgroup's");
  device const float4 *quads = reinterpret_cast<device const float4 *>(row);
  if (tid < kThreads) {
    float squares = 0.0f;
    for (uint d = tid; d < hidden / 4; d += kThreads) {
      const float4 v = quads[d];
      squares += v.x * v.x;
      squares += v.y * v.y;
      squares += v.z * v.z;
      squares += v.w * v.w;
    }
    squares = simd_sum(squares);
    if (lane == 0) partials[sg] = squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint s = 0; s < kSimdgroups; ++s) total += partials[s];
    partials[kSimdgroups] = rsqrt(total / hidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = partials[kSimdgroups];
  device bfloat4 *out = reinterpret_cast<device bfloat4 *>(normalized);
  for (uint d = tid; d < hidden / 4; d += Threads) out[d] = bfloat4(quads[d] * scale);
}
