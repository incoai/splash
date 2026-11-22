#pragma clang fp reassociate(off)
#include "metal/abi/BlockResidualMix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// Block residual mixes (metal/abi/BlockResidualMix.h): a threadgroup of
// kMixThreads threads a row, each thread the columns tid + kMixThreads i,
// in fp32 from the bf16 stack planes and the current rows, bf16 (a layer's
// input) or fp32 (a residual inside a layer): Current, the instance's
// element type, the fp32 one suffixed _f32. A row runs in two phases, each a
// function out of line, which hand over through threadgroup memory alone:
// the row's states' norms and its queries' weights, each sum reduced by
// thread 0 into its own parked value, then the outputs from those and the
// states. No thread keeps an array of them across a barrier: an inlined
// mix whose threads each held a row's norms and weights through its
// reductions computed one simdgroup's outputs wrongly, now and then, under
// shader validation (as moe_gate_up_staged in moe_gguf.metal).
constant constexpr uint kMixThreads = BLOCK_RESIDUAL_MIX_THREADS;
constant constexpr uint kMixSimdgroups = kMixThreads / 32;
constant constexpr uint kStates = BLOCK_RESIDUAL_MIX_MAXIMUM_STATES;
constant constexpr uint kQueries = BLOCK_RESIDUAL_MIX_MAXIMUM_QUERIES;
constant constexpr uint kNorms = BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS;

// What a row's phases share in threadgroup memory: a partial per simdgroup
// of each reduction, its own slots each, and the parked values: each state's
// inverse RMS without weights, each query's softmax weight of each state,
// and the norm variant's each output's inverse RMS.
struct BlockResidualMixShared {
  float partials[(kStates + kQueries * kStates + kNorms) * kMixSimdgroups];
  float inverse[kStates];
  float weights[kQueries * kStates];
  float output_inverse[kNorms];
};

// A row's states: the stack's planes, then the current rows.
template <class Current> struct BlockResidualMixRow {
  device const bfloat *stack;
  device const Current *current;
  uint stack_states;
  uint plane;

  float state(uint s, uint d) const {
    return s < stack_states ? float(stack[ulong(s) * plane + d]) : float(current[d]);
  }
  uint states() const { return stack_states + 1; }
};

// The sum over the threadgroup of each thread's `sum`: per simdgroup, then
// over the simdgroups in order by thread 0, which hands the total to
// `finish` and parks what it returns at `parked`; every thread past the
// second barrier. `partials` holds this reduction's slot a simdgroup.
template <class Finish>
inline void block_residual_mix_reduce(float sum, threadgroup float *partials, threadgroup float *parked, uint tid,
                                      uint lane, uint sg, Finish finish) {
  sum = simd_sum(sum);
  if (lane == 0) partials[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < kMixSimdgroups; ++g) total += partials[g];
    *parked = finish(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase one: each state's inverse RMS, then each query's score of each
// state (query . normalized state, the sums of the threads' columns in
// order) and, by thread 0, the softmax of each query's scaled scores.
template <class Current>
__attribute__((noinline)) void block_residual_mix_weights(BlockResidualMixRow<Current> mix,
                                                          device const float *queries,
                                                          constant BlockResidualMixParams &p, uint tid, uint lane,
                                                          uint sg, threadgroup BlockResidualMixShared &shared) {
  const uint states = mix.states();
  for (uint s = 0; s < states; ++s) {
    float squares = 0.0f;
    for (uint d = tid; d < p.hidden; d += kMixThreads) {
      const float x = mix.state(s, d);
      squares += x * x;
    }
    block_residual_mix_reduce(squares, shared.partials + s * kMixSimdgroups, &shared.inverse[s], tid, lane, sg,
                              [&](float total) { return rsqrt(total / p.hidden + kRmsEpsilon); });
  }
  for (uint q = 0; q < p.queries; ++q)
    for (uint s = 0; s < states; ++s) {
      const float inverse = shared.inverse[s];
      float dot = 0.0f;
      for (uint d = tid; d < p.hidden; d += kMixThreads)
        dot += queries[ulong(q) * p.hidden + d] * (mix.state(s, d) * inverse);
      block_residual_mix_reduce(dot, shared.partials + (kStates + q * kStates + s) * kMixSimdgroups,
                                &shared.weights[q * kStates + s], tid, lane, sg, [](float total) { return total; });
    }
  if (tid == 0)
    for (uint q = 0; q < p.queries; ++q) {
      threadgroup float *weights = shared.weights + q * kStates;
      float maximum = -INFINITY;
      for (uint s = 0; s < states; ++s) maximum = max(maximum, weights[s] * p.scale);
      float total = 0.0f;
      for (uint s = 0; s < states; ++s) {
        weights[s] = exp(weights[s] * p.scale - maximum);
        total += weights[s];
      }
      for (uint s = 0; s < states; ++s) weights[s] /= total;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Query q's output at column d: the weighted sum of the normalized states,
// from the parked norms and weights.
template <class Current>
inline float block_residual_mix_output(BlockResidualMixRow<Current> mix, uint q, uint d,
                                       threadgroup const BlockResidualMixShared &shared) {
  float sum = 0.0f;
  for (uint s = 0; s < mix.states(); ++s)
    sum += shared.weights[q * kStates + s] * (mix.state(s, d) * shared.inverse[s]);
  return sum;
}

// Phase two of block_residual_mix: each query's output as fp32 rows.
template <class Current>
__attribute__((noinline)) void block_residual_mix_outputs(BlockResidualMixRow<Current> mix, device float *output,
                                                          constant BlockResidualMixParams &p, uint row, uint tid,
                                                          threadgroup const BlockResidualMixShared &shared) {
  for (uint q = 0; q < p.queries; ++q)
    for (uint d = tid; d < p.hidden; d += kMixThreads)
      output[(ulong(row) * p.queries + q) * p.hidden + d] = block_residual_mix_output(mix, q, d, shared);
}

// Phase two of block_residual_mix_norm: each query's output's inverse RMS,
// then its normalized and weighted bf16 rows.
template <class Current>
__attribute__((noinline)) void block_residual_mix_norm_outputs(BlockResidualMixRow<Current> mix,
                                                               device const float *norm0, device const float *norm1,
                                                               device const float *norm2, device bfloat *output0,
                                                               device bfloat *output1, device bfloat *output2,
                                                               constant BlockResidualMixParams &p, uint row, uint tid,
                                                               uint lane, uint sg,
                                                               threadgroup BlockResidualMixShared &shared) {
  for (uint q = 0; q < p.queries; ++q) {
    float squares = 0.0f;
    for (uint d = tid; d < p.hidden; d += kMixThreads) {
      const float o = block_residual_mix_output(mix, q, d, shared);
      squares += o * o;
    }
    block_residual_mix_reduce(squares, shared.partials + (kStates + kQueries * kStates + q) * kMixSimdgroups,
                              &shared.output_inverse[q], tid, lane, sg,
                              [&](float total) { return rsqrt(total / p.hidden + kRmsEpsilon); });
  }
  device const float *norms[kNorms] = {norm0, norm1, norm2};
  device bfloat *outputs[kNorms] = {output0, output1, output2};
  for (uint q = 0; q < p.queries; ++q) {
    const float inverse = shared.output_inverse[q];
    for (uint d = tid; d < p.hidden; d += kMixThreads)
      outputs[q][ulong(row) * p.hidden + d] =
          bfloat((block_residual_mix_output(mix, q, d, shared) * inverse) * norms[q][d]);
  }
}

// Each query's output as fp32 rows [rows][queries][hidden]: the message
// gate's input and the loop's messages.
template <class Current>
kernel void block_residual_mix(device const bfloat *stack [[buffer(0)]], device const Current *current [[buffer(1)]],
                               device const float *queries [[buffer(2)]], device float *output [[buffer(3)]],
                               constant BlockResidualMixParams &p [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  if (row >= p.rows) return;
  const BlockResidualMixRow<Current> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                         p.stack_plane};
  block_residual_mix_weights(mix, queries, p, tid, lane, sg, shared);
  block_residual_mix_outputs(mix, output, p, row, tid, shared);
}

// Each query's output through its weighted RMS norm (F32 weights, as a GGUF
// stores its norms) into its own bf16 rows: the input of the projections
// after the mix, which prepare their own tables, as after the
// hyper-connection mix (shared/hyper_connection.metal). Unused norm and
// output slots bind the first.
template <class Current>
kernel void block_residual_mix_norm(device const bfloat *stack [[buffer(0)]],
                                    device const Current *current [[buffer(1)]],
                                    device const float *queries [[buffer(2)]], device const float *norm0 [[buffer(3)]],
                                    device const float *norm1 [[buffer(4)]], device const float *norm2 [[buffer(5)]],
                                    device bfloat *output0 [[buffer(6)]], device bfloat *output1 [[buffer(7)]],
                                    device bfloat *output2 [[buffer(8)]],
                                    constant BlockResidualMixParams &p [[buffer(9)]],
                                    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  if (row >= p.rows) return;
  const BlockResidualMixRow<Current> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                         p.stack_plane};
  block_residual_mix_weights(mix, queries, p, tid, lane, sg, shared);
  block_residual_mix_norm_outputs(mix, norm0, norm1, norm2, output0, output1, output2, p, row, tid, lane, sg, shared);
}

// The instances: bf16 current rows (a layer's input), and fp32 (_f32, the
// residual inside a layer).
#define BLOCK_RESIDUAL_MIX_INSTANCES(Current, suffix)                                                          \
  using BlockResidualMixKernel##Current = void(device const bfloat *, device const Current *, device const float *, \
                                               device float *, constant BlockResidualMixParams &, uint, uint, uint,  \
                                               uint);                                                                 \
  using BlockResidualMixNormKernel##Current =                                                                         \
      void(device const bfloat *, device const Current *, device const float *, device const float *,               \
           device const float *, device const float *, device bfloat *, device bfloat *, device bfloat *,            \
           constant BlockResidualMixParams &, uint, uint, uint, uint);                                               \
  template [[host_name("block_residual_mix" suffix)]] kernel BlockResidualMixKernel##Current                         \
      block_residual_mix<Current>;                                                                                    \
  template [[host_name("block_residual_mix_norm" suffix)]] kernel BlockResidualMixNormKernel##Current               \
      block_residual_mix_norm<Current>;
BLOCK_RESIDUAL_MIX_INSTANCES(bfloat, "")
BLOCK_RESIDUAL_MIX_INSTANCES(float, "_f32")
#undef BLOCK_RESIDUAL_MIX_INSTANCES
