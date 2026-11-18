#include "metal/abi/Engram.h"
#include "metal/kernels/common/activation.h"

#include <metal_stdlib>

using namespace metal;

// DeepSeek-V4.1's Engram gate (metal/abi/Engram.h), as engram.py computes it
// (in dev/tools/deepseek41_reference.py Forward.engram): for each stream s of
// a row, dot = sum(h_s * w_s * key_s) / RMS(h_s) / RMS(key_s) / sqrt(hidden),
// gate = sigmoid(sign(dot) sqrt(max(|dot|, floor))), and h_s += gate value.
// A threadgroup per row, which reads the row's streams twice: for the sums,
// then to add the gated value.

constant constexpr uint kStreams = MHC_STREAMS;
constant constexpr uint kThreads = ENGRAM_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;

kernel void engram_gate(device float *streams [[buffer(0)]],
                        device const bfloat *kv [[buffer(1)]],
                        device const float *weights [[buffer(2)]],
                        constant EngramParams &p [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  // Each stream's sums of h^2, key^2 and h w key.
  threadgroup float partials[kSimdgroups][3 * kStreams];
  threadgroup float gates[kStreams];
  if (row >= p.rows) return;
  device float *h = streams + ulong(row) * kStreams * p.hidden;
  device const bfloat *row_kv = kv + ulong(row) * (kStreams + 1) * p.hidden;
  float sums[3 * kStreams];
  for (uint i = 0; i < 3 * kStreams; ++i) sums[i] = 0.0f;
  for (uint s = 0; s < kStreams; ++s)
    for (uint d = tid; d < p.hidden; d += kThreads) {
      const float x = h[s * p.hidden + d], key = float(row_kv[s * p.hidden + d]);
      sums[3 * s] = fma(x, x, sums[3 * s]);
      sums[3 * s + 1] = fma(key, key, sums[3 * s + 1]);
      sums[3 * s + 2] = fma(x * weights[s * p.hidden + d], key, sums[3 * s + 2]);
    }
  for (uint i = 0; i < 3 * kStreams; ++i) {
    const float sum = simd_sum(sums[i]);
    if (lane == 0) partials[sg][i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < kStreams) {
    float total[3] = {0.0f, 0.0f, 0.0f};
    for (uint g = 0; g < kSimdgroups; ++g)
      for (uint i = 0; i < 3; ++i) total[i] += partials[g][3 * tid + i];
    const float width = float(p.hidden);
    const float inverse = precise::divide(1.0f, precise::sqrt(total[0] / width + p.norm_epsilon)) /
                          precise::sqrt(total[1] / width + p.norm_epsilon);
    const float dot = total[2] * inverse * rsqrt(width);
    gates[tid] = splash_sigmoid(sign(dot) * sqrt(max(fabs(dot), p.magnitude_floor)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const bfloat *value = row_kv + kStreams * p.hidden;
  for (uint s = 0; s < kStreams; ++s)
    for (uint d = tid; d < p.hidden; d += kThreads) h[s * p.hidden + d] += gates[s] * float(value[d]);
}

// A verify step's rows' Engram rows (EngramHashParams): a thread per row and
// Engram layer. The products and the XOR wrap as int64 does, and a row is
// the running XOR's modulo of the head's prime as Python's, never negative.
kernel void engram_hash(device const uint *tokens [[buffer(0)]], device const uint *prior [[buffer(1)]],
                        device const int *token_map [[buffer(2)]], device const long *multipliers [[buffer(3)]],
                        device const long *primes [[buffer(4)]], device const long *offsets [[buffer(5)]],
                        device uint *out [[buffer(6)]], constant EngramHashParams &p [[buffer(7)]],
                        uint2 at [[thread_position_in_grid]]) {
  const uint row = at.x, layer = at.y;
  if (row >= p.rows || layer >= p.layers) return;
  const uint lane = row / p.lane_rows, inLane = row % p.lane_rows, per = (p.ngram - 1) * p.heads;
  device uint *rows = out + (ulong(layer) * p.rows + row) * per;
  ulong rolling = 0;
  for (uint n = 0; n < p.ngram; ++n) {
    const uint token = n <= inLane ? tokens[row - n] : prior[lane * (p.ngram - 1) + (n - inLane - 1)];
    const ulong product = ulong(long(token_map[token])) * ulong(multipliers[layer * p.ngram + n]);
    rolling = n == 0 ? product : rolling ^ product;
    if (n == 0) continue;
    for (uint h = 0; h < p.heads; ++h) {
      const uint index = layer * per + (n - 1) * p.heads + h;
      const long prime = primes[index];
      long value = long(rolling) % prime;
      if (value < 0) value += prime;
      rows[(n - 1) * p.heads + h] = uint(value + offsets[index]) + p.first_rows[layer];
    }
  }
}
