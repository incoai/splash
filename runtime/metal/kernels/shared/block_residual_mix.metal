#pragma clang fp reassociate(off)
#include "metal/abi/BlockResidualMix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/block_residual_mix_row.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// Block residual mixes (metal/abi/BlockResidualMix.h): a threadgroup of
// kMixThreads threads a row and query (grid y), each thread the columns
// tid + kMixThreads i, in fp32 from the bf16 stack planes and the current
// rows, bf16 (a layer's input) or fp32 (a residual inside a layer): Current,
// the instance's element type, the fp32 one suffixed _f32. A threadgroup
// runs in phases, each a function out of line, which hand over through
// threadgroup memory alone: the row's states' norms, its query's weights of
// the normalized states, then its query's output from those and the states.
// A phase takes all its sums in one pass over the columns, each thread's of
// its columns in order, and reduces them together: each by simd_sum into
// its own slot a simdgroup, then by thread i sum i over the simdgroups in
// order, which parks its value between two barriers. Every query's
// threadgroup computes the row's norms alike, so each query's output is the
// one a threadgroup of every query computes (metal/abi/BlockResidualMix.h
// has what a threadgroup a row took). A thread's sums live in an array of
// the most states, every slot past the row's zero, and reach threadgroup
// memory before the barrier: no thread keeps an array of them across a
// barrier (an inlined mix whose threads each held a row's norms and weights
// through its reductions computed one simdgroup's outputs wrongly, now and
// then, under shader validation, as moe_gate_up_staged in moe_gguf.metal).
constant constexpr uint kMixThreads = BLOCK_RESIDUAL_MIX_THREADS;
constant constexpr uint kMixSimdgroups = kMixThreads / 32;
constant constexpr uint kStates = BLOCK_RESIDUAL_MIX_MAXIMUM_STATES;
constant constexpr uint kNorms = BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS;

// What a threadgroup's phases share in threadgroup memory: a partial per
// simdgroup of each sum (each state's squares, the query's score of each
// state, the norm variant's output's squares), and the parked values: each
// state's inverse RMS without weights, the query's softmax weight of each
// state, and the output's inverse RMS.
struct BlockResidualMixShared {
  float partials[(2 * kStates + 1) * kMixSimdgroups];
  float inverse[kStates];
  float weights[kStates];
  float output_inverse;
};

// Each thread's `count` sums (of an array of Count) reduced together: per
// simdgroup into slot i of `partials` (kMixSimdgroups floats a slot), then,
// past a barrier, sum i over the simdgroups in order by thread i, which hands
// the total to `finish` and parks what it returns at parked[i]; every thread
// past the second barrier.
template <uint Count, class Finish>
inline void block_residual_mix_reduce(thread const float (&sums)[Count], uint count, threadgroup float *partials,
                                      threadgroup float *parked, uint tid, uint lane, uint sg, Finish finish) {
  for (uint i = 0; i < Count; ++i) {
    const float sum = simd_sum(sums[i]);
    if (lane == 0 && i < count) partials[i * kMixSimdgroups + sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < count) {
    float total = 0.0f;
    for (uint g = 0; g < kMixSimdgroups; ++g) total += partials[tid * kMixSimdgroups + g];
    parked[tid] = finish(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase one: each state's inverse RMS.
template <class Current>
__attribute__((noinline)) void block_residual_mix_norms(BlockResidualMixRow<Current> mix,
                                                        constant BlockResidualMixParams &p, uint tid, uint lane,
                                                        uint sg, threadgroup BlockResidualMixShared &shared) {
  const uint states = mix.states();
  float squares[kStates];
  for (uint s = 0; s < kStates; ++s) squares[s] = 0.0f;
  for (uint d = tid; d < p.hidden; d += kMixThreads)
    for (uint s = 0; s < kStates; ++s)
      if (s < states) {
        const float x = mix.state(s, d);
        squares[s] += x * x;
      }
  block_residual_mix_reduce(squares, states, shared.partials, shared.inverse, tid, lane, sg,
                            [&](float total) { return rsqrt(total / p.hidden + kRmsEpsilon); });
}

// Phase two: the query's score of each state (query . normalized state),
// then, by thread 0, the softmax of its scaled scores.
template <class Current>
__attribute__((noinline)) void block_residual_mix_weights(BlockResidualMixRow<Current> mix,
                                                          device const float *query,
                                                          constant BlockResidualMixParams &p, uint tid, uint lane,
                                                          uint sg, threadgroup BlockResidualMixShared &shared) {
  const uint states = mix.states();
  float dots[kStates];
  for (uint s = 0; s < kStates; ++s) dots[s] = 0.0f;
  for (uint d = tid; d < p.hidden; d += kMixThreads) {
    const float weight = query[d];
    for (uint s = 0; s < kStates; ++s)
      if (s < states) dots[s] += weight * (mix.state(s, d) * shared.inverse[s]);
  }
  block_residual_mix_reduce(dots, states, shared.partials + kStates * kMixSimdgroups, shared.weights, tid, lane, sg,
                            [](float total) { return total; });
  if (tid == 0) block_residual_mix_softmax(shared.weights, 1, kStates, states, p);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase three of block_residual_mix: the query's output as fp32 rows.
template <class Current>
__attribute__((noinline)) void block_residual_mix_outputs(BlockResidualMixRow<Current> mix, device float *output,
                                                          constant BlockResidualMixParams &p, uint tid,
                                                          threadgroup const BlockResidualMixShared &shared) {
  for (uint d = tid; d < p.hidden; d += kMixThreads) output[d] = block_residual_mix_output(mix, 0, d, shared);
}

// Phase three of block_residual_mix_norm: the query's output's inverse RMS,
// then its normalized and weighted bf16 row.
template <class Current>
__attribute__((noinline)) void block_residual_mix_norm_outputs(BlockResidualMixRow<Current> mix,
                                                               device const float *norm, device bfloat *output,
                                                               constant BlockResidualMixParams &p, uint tid,
                                                               uint lane, uint sg,
                                                               threadgroup BlockResidualMixShared &shared) {
  float squares[1] = {0.0f};
  for (uint d = tid; d < p.hidden; d += kMixThreads) {
    const float o = block_residual_mix_output(mix, 0, d, shared);
    squares[0] += o * o;
  }
  block_residual_mix_reduce(squares, 1, shared.partials + 2 * kStates * kMixSimdgroups, &shared.output_inverse, tid,
                            lane, sg, [&](float total) { return rsqrt(total / p.hidden + kRmsEpsilon); });
  const float inverse = shared.output_inverse;
  for (uint d = tid; d < p.hidden; d += kMixThreads)
    output[d] = bfloat((block_residual_mix_output(mix, 0, d, shared) * inverse) * norm[d]);
}

// Each query's output as fp32 rows [rows][queries][hidden]: the message
// gate's input and the loop's messages. Grid: (rows, queries).
template <class Current>
kernel void block_residual_mix(device const bfloat *stack [[buffer(0)]], device const Current *current [[buffer(1)]],
                               device const float *queries [[buffer(2)]], device float *output [[buffer(3)]],
                               constant BlockResidualMixParams &p [[buffer(4)]],
                               uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  const uint row = group.x, q = group.y;
  if (row >= p.rows || q >= p.queries) return;
  const BlockResidualMixRow<Current> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                         p.stack_plane};
  block_residual_mix_norms(mix, p, tid, lane, sg, shared);
  block_residual_mix_weights(mix, queries + ulong(q) * p.hidden, p, tid, lane, sg, shared);
  block_residual_mix_outputs(mix, output + (ulong(row) * p.queries + q) * p.hidden, p, tid, shared);
}

// Each query's output through its weighted RMS norm (F32 weights, as a GGUF
// stores its norms) into its own bf16 rows: the input of the projections
// after the mix, which prepare their own tables, as after the
// hyper-connection mix (shared/hyper_connection.metal). Unused norm and
// output slots bind the first. Grid: (rows, queries).
template <class Current>
kernel void block_residual_mix_norm(device const bfloat *stack [[buffer(0)]],
                                    device const Current *current [[buffer(1)]],
                                    device const float *queries [[buffer(2)]], device const float *norm0 [[buffer(3)]],
                                    device const float *norm1 [[buffer(4)]], device const float *norm2 [[buffer(5)]],
                                    device bfloat *output0 [[buffer(6)]], device bfloat *output1 [[buffer(7)]],
                                    device bfloat *output2 [[buffer(8)]],
                                    constant BlockResidualMixParams &p [[buffer(9)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  const uint row = group.x, q = group.y;
  if (row >= p.rows || q >= p.queries || q >= kNorms) return;
  const BlockResidualMixRow<Current> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                         p.stack_plane};
  block_residual_mix_norms(mix, p, tid, lane, sg, shared);
  block_residual_mix_weights(mix, queries + ulong(q) * p.hidden, p, tid, lane, sg, shared);
  device const float *norm = q == 0 ? norm0 : q == 1 ? norm1 : norm2;
  device bfloat *output = q == 0 ? output0 : q == 1 ? output1 : output2;
  block_residual_mix_norm_outputs(mix, norm, output + ulong(row) * p.hidden, p, tid, lane, sg, shared);
}

// The instances: bf16 current rows (a layer's input), and fp32 (_f32, the
// residual inside a layer).
#define BLOCK_RESIDUAL_MIX_INSTANCES(Current, suffix)                                                          \
  using BlockResidualMixKernel##Current = void(device const bfloat *, device const Current *, device const float *, \
                                               device float *, constant BlockResidualMixParams &, uint2, uint, uint, \
                                               uint);                                                                 \
  using BlockResidualMixNormKernel##Current =                                                                         \
      void(device const bfloat *, device const Current *, device const float *, device const float *,               \
           device const float *, device const float *, device bfloat *, device bfloat *, device bfloat *,            \
           constant BlockResidualMixParams &, uint2, uint, uint, uint);                                              \
  template [[host_name("block_residual_mix" suffix)]] kernel BlockResidualMixKernel##Current                         \
      block_residual_mix<Current>;                                                                                    \
  template [[host_name("block_residual_mix_norm" suffix)]] kernel BlockResidualMixNormKernel##Current               \
      block_residual_mix_norm<Current>;
BLOCK_RESIDUAL_MIX_INSTANCES(bfloat, "")
BLOCK_RESIDUAL_MIX_INSTANCES(float, "_f32")
#undef BLOCK_RESIDUAL_MIX_INSTANCES
