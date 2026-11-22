#include "metal/abi/BlockResidual.h"
#include "metal/abi/ExecutionGeometry.h"

#include <metal_stdlib>

using namespace metal;

// Inkling 2's depth attention (metal/abi/BlockResidual.h) in fp32: a
// threadgroup per row, each thread holding its channels of every state and
// query in registers from one round of loads, so that a decode step's few
// rows wait on memory once rather than once per channel a thread takes.

constant constexpr uint kStates = BLOCK_RESIDUAL_MAXIMUM_STATES;
constant constexpr uint kQueries = BLOCK_RESIDUAL_MAXIMUM_QUERIES;
constant constexpr uint kThreads = BLOCK_RESIDUAL_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;
constant constexpr uint kPerThread = BLOCK_RESIDUAL_MAXIMUM_HIDDEN / kThreads;
// A row's sums: each state's squares, then each query's dot product with
// each state (query j's with state k at kStates + j * kStates + k).
constant constexpr uint kSums = kStates + kQueries * kStates;
constant constexpr float kEpsilon = SPLASH_RMS_EPSILON;
static_assert(kSums <= kThreads, "a thread totals each sum");

// Threadgroup `row`; thread t takes channels t + kThreads i. Each query's
// logits are its dot products with the states times the states' inverse RMS
// and the scale (q . s^_k = (q . s_k) / rms(s_k)); thread j takes query j's
// softmax and folds the inverse RMS into each state's weight, b_jk = a_jk /
// rms(s_k), so that out_j = sum_k b_jk s_k over the values the threads hold.
// A thread writes its channels of the outputs only after reading them of
// every state, so an output may be one of the states' rows.
kernel void block_residual(device const float *state0 [[buffer(0)]], device const float *state1 [[buffer(1)]],
                           device const float *state2 [[buffer(2)]], device const float *state3 [[buffer(3)]],
                           device const float *state4 [[buffer(4)]], device const float *state5 [[buffer(5)]],
                           device const float *queries [[buffer(6)]], device float *output0 [[buffer(7)]],
                           device float *output1 [[buffer(8)]], device float *output2 [[buffer(9)]],
                           device float *output3 [[buffer(10)]], constant BlockResidualParams &p [[buffer(11)]],
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[kSimdgroups * kSums];
  threadgroup float totals[kSums];
  threadgroup float weights[kQueries * kStates];
  if (row >= p.rows) return;
  device const float *const bound[kStates] = {state0, state1, state2, state3, state4, state5};
  float values[kPerThread][kStates];
  float sums[kSums];
#pragma unroll
  for (uint i = 0; i < kSums; ++i) sums[i] = 0.0f;
#pragma unroll
  for (uint i = 0; i < kPerThread; ++i) {
    const uint c = tid + i * kThreads;
    const bool inside = c < p.hidden;
    float q[kQueries];
#pragma unroll
    for (uint j = 0; j < kQueries; ++j) q[j] = inside && j < p.queries ? queries[ulong(j) * p.hidden + c] : 0.0f;
#pragma unroll
    for (uint k = 0; k < kStates; ++k) {
      const float v = inside && k < p.states ? bound[k][ulong(row) * p.state_stride[k] + c] : 0.0f;
      values[i][k] = v;
      sums[k] += v * v;
#pragma unroll
      for (uint j = 0; j < kQueries; ++j) sums[kStates + j * kStates + k] += q[j] * v;
    }
  }
#pragma unroll
  for (uint i = 0; i < kSums; ++i) {
    const float sum = simd_sum(sums[i]);
    if (lane == 0) partials[sg * kSums + i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < kSums) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += partials[g * kSums + tid];
    totals[tid] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < kQueries) {
    const uint j = tid;
    float inverse[kStates], logits[kStates], most = -INFINITY;
    for (uint k = 0; k < kStates; ++k) {
      inverse[k] = rsqrt(totals[k] / float(p.hidden) + kEpsilon);
      logits[k] = totals[kStates + j * kStates + k] * inverse[k] * p.scale;
      if (k < p.states) most = max(most, logits[k]);
    }
    float total = 0.0f;
    for (uint k = 0; k < kStates; ++k) {
      logits[k] = k < p.states ? precise::exp(logits[k] - most) : 0.0f;
      total += logits[k];
    }
    // Queries past p.queries and states past p.states weigh nothing.
    for (uint k = 0; k < kStates; ++k) weights[j * kStates + k] = j < p.queries ? logits[k] / total * inverse[k] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float *const outputs[kQueries] = {output0, output1, output2, output3};
#pragma unroll
  for (uint i = 0; i < kPerThread; ++i) {
    const uint c = tid + i * kThreads;
    if (c >= p.hidden) continue;
#pragma unroll
    for (uint j = 0; j < kQueries; ++j) {
      if (j >= p.queries) continue;
      float out = 0.0f;
#pragma unroll
      for (uint k = 0; k < kStates; ++k) out += weights[j * kStates + k] * values[i][k];
      outputs[j][ulong(row) * p.output_stride + c] = out;
    }
  }
}
