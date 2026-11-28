// The block residual mix's scores and the short convolution's output, each
// as 1.7.4 computed it and as this tree does, for their error against fp64
// (convolution-mix-kernels, mixConvPrecisionTest). Compiled with the
// production flags and block_residual_mix.metal's file-scope pragma, as the
// engine compiles the kernels they copy; 1.7.4's convolution, whose file had
// none, turns reassociation back on in its own body.
#pragma clang fp reassociate(off)
#include "metal/abi/BlockResidualMix.h"
#include "metal/abi/KernelABI.h"
#include "metal/abi/ShortConvolution.h"
#include "metal/kernels/common/block_residual_mix_row.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

namespace precision_test {

constant constexpr uint kThreads = BLOCK_RESIDUAL_MIX_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;
constant constexpr uint kStates = BLOCK_RESIDUAL_MIX_MAXIMUM_STATES;
static_assert(kSimdgroups == 32, "a slot's partials are one a lane of a simdgroup");

struct Shared {
  float partials[2 * kStates * kSimdgroups];
  float inverse[kStates];
  float weights[kStates];
};

// Thread i's sum i over the simdgroups' partials, in order.
inline float total_of(threadgroup const float *partials, uint slot) {
  float total = 0.0f;
  for (uint g = 0; g < kSimdgroups; ++g) total += partials[slot * kSimdgroups + g];
  return total;
}

}  // namespace precision_test

// The tests' parameters, as the host's std::pairs lay them out.
struct MixPrecisionParams {
  struct BlockResidualMixParams mix;
  uint32_t one_pass;
};
struct ConvPrecisionParams {
  uint32_t channels;
  uint32_t ordered;
};

// One query's mix of a row's states (fp32 current rows), its weights to
// `weights` [rows][states] and its output to `output` [rows][hidden]: `onePass`
// 0 as 1.7.4 scored the states (each state's inverse RMS, a pass and a
// reduction, then the query's dot with each normalized state, another, each
// total a thread's in simdgroup order, and thread 0's softmax), 1 as this
// tree does (block_residual_mix_scores: both sums in one pass and one
// reduction, the score the dot times the inverse, each total by simd_sum, a
// simdgroup's partial a lane, and the softmax in a simdgroup's lanes: here
// every simdgroup's, the values of the tree's simdgroup 0). Grid: rows.
kernel void mix_precision_test(device const bfloat *stack [[buffer(0)]], device const float *current [[buffer(1)]],
                               device const float *query [[buffer(2)]], device float *weights [[buffer(3)]],
                               device float *output [[buffer(4)]], constant MixPrecisionParams &params [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  using namespace precision_test;
  constant BlockResidualMixParams &p = params.mix;
  const bool onePass = params.one_pass;
  threadgroup Shared shared;
  const BlockResidualMixRow<float> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                       p.stack_plane};
  const uint states = mix.states();
  if (onePass) {
    float sums[2 * kStates];
    for (uint i = 0; i < 2 * kStates; ++i) sums[i] = 0.0f;
    for (uint d = tid; d < p.hidden; d += kThreads) {
      const float w = query[d];
      for (uint s = 0; s < kStates; ++s)
        if (s < states) {
          const float x = mix.state(s, d);
          sums[s] += x * x;
          sums[kStates + s] += w * x;
        }
    }
    for (uint i = 0; i < 2 * kStates; ++i) {
      const float sum = simd_sum(sums[i]);
      if (lane == 0) shared.partials[i * kSimdgroups + sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse[kStates], weight[kStates], maximum = -INFINITY;
    for (uint s = 0; s < kStates; ++s) {
      inverse[s] = rsqrt(simd_sum(shared.partials[s * kSimdgroups + lane]) / p.hidden + kRmsEpsilon);
      weight[s] = simd_sum(shared.partials[(kStates + s) * kSimdgroups + lane]) * inverse[s];
      if (s < states) maximum = max(maximum, weight[s] * p.scale);
    }
    float total = 0.0f;
    for (uint s = 0; s < kStates; ++s) {
      weight[s] = s < states ? exp(weight[s] * p.scale - maximum) : 0.0f;
      total += weight[s];
    }
    if (sg == 0)
      for (uint s = 0; s < kStates; ++s)
        if (lane == s) {
          shared.inverse[s] = inverse[s];
          shared.weights[s] = weight[s] / total;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    float squares[kStates];
    for (uint s = 0; s < kStates; ++s) squares[s] = 0.0f;
    for (uint d = tid; d < p.hidden; d += kThreads)
      for (uint s = 0; s < kStates; ++s)
        if (s < states) {
          const float x = mix.state(s, d);
          squares[s] += x * x;
        }
    for (uint s = 0; s < kStates; ++s) {
      const float sum = simd_sum(squares[s]);
      if (lane == 0) shared.partials[s * kSimdgroups + sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < states) shared.inverse[tid] = rsqrt(total_of(shared.partials, tid) / p.hidden + kRmsEpsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dots[kStates];
    for (uint s = 0; s < kStates; ++s) dots[s] = 0.0f;
    for (uint d = tid; d < p.hidden; d += kThreads) {
      const float w = query[d];
      for (uint s = 0; s < kStates; ++s)
        if (s < states) dots[s] += w * (mix.state(s, d) * shared.inverse[s]);
    }
    for (uint s = 0; s < kStates; ++s) {
      const float sum = simd_sum(dots[s]);
      if (lane == 0) shared.partials[(kStates + s) * kSimdgroups + sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < states) shared.weights[tid] = total_of(shared.partials, kStates + tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) block_residual_mix_softmax(shared.weights, 1, kStates, states, p);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < states) weights[ulong(row) * kStates + tid] = shared.weights[tid];
  for (uint d = tid; d < p.hidden; d += kThreads)
    output[ulong(row) * p.hidden + d] = block_residual_mix_output(mix, 0, d, shared);
}

// One row's short convolution of static taps along a chain (rows t - 3 ..
// t of `input`, the state before the first row zero), Sum onto a bf16
// residual, at each channel: `ordered` 0 its sums as the compiler may order
// them (1.7.4's short_convolution.metal, of no pragma), 1 in source order
// (reassociate(off), this tree's short_convolution_output). Grid: (rows,
// channel groups of 256).
inline float convolve_reassociated(device const bfloat *input, device const float *taps, uint channels, uint row,
                                   uint c, float divisor) {
#pragma clang fp reassociate(on)
  float sum = 0.0f;
  for (uint k = 0; k < SHORT_CONVOLUTION_TAPS; ++k) {
    const int back = int(row) - int(k);
    const float x = back >= 0 ? float(input[ulong(back) * channels + c]) : 0.0f;
    sum += taps[ulong(c) * SHORT_CONVOLUTION_TAPS + (SHORT_CONVOLUTION_TAPS - 1 - k)] * x;
  }
  return (sum + float(input[ulong(row) * channels + c])) / divisor;
}
inline float convolve_ordered(device const bfloat *input, device const float *taps, uint channels, uint row, uint c,
                              float divisor) {
  float sum = 0.0f;
  for (uint k = 0; k < SHORT_CONVOLUTION_TAPS; ++k) {
    const int back = int(row) - int(k);
    const float x = back >= 0 ? float(input[ulong(back) * channels + c]) : 0.0f;
    sum += taps[ulong(c) * SHORT_CONVOLUTION_TAPS + (SHORT_CONVOLUTION_TAPS - 1 - k)] * x;
  }
  return (sum + float(input[ulong(row) * channels + c])) / divisor;
}
kernel void conv_precision_test(device const bfloat *input [[buffer(0)]], device const float *taps [[buffer(1)]],
                                device const bfloat *residual [[buffer(2)]], device float *output [[buffer(3)]],
                                constant ConvPrecisionParams &params [[buffer(4)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  const uint channels = params.channels, ordered = params.ordered;
  const uint row = group.x, c = group.y * 256 + tid;
  if (c >= channels) return;
  const float divisor = sqrt(float(SHORT_CONVOLUTION_TAPS + 1));
  const float y = ordered ? convolve_ordered(input, taps, channels, row, c, divisor)
                          : convolve_reassociated(input, taps, channels, row, c, divisor);
  const ulong i = ulong(row) * channels + c;
  output[i] = float(residual[i]) + float(bfloat(y));
}
