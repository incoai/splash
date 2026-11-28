#pragma clang fp reassociate(off)
#include "metal/abi/BlockResidualMix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/block_residual_mix_row.h"
#include "metal/kernels/common/message_gate_normalize.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// Block residual mixes of more than BLOCK_RESIDUAL_MIX_QUERY_ROWS rows
// (metal/abi/BlockResidualMix.h), a prompt chunk's or a decode step's alike:
// a threadgroup of kRowThreads threads a row (and mix of a dispatch's, grid
// y), each thread the columns
// tid + kRowThreads i, in fp32 from the bf16 stack planes and the current
// rows, bf16 (a layer's input) or fp32 (a residual inside a layer): Current,
// the instance's element type, the fp32 one suffixed _f32. A row runs in two
// phases, each a function out of line, which hand over through threadgroup
// memory alone: the row's states' norms and its queries' weights, their
// sums taken in one pass and reduced together, then the outputs from those
// and the states. No thread keeps an array of them across a barrier: an
// inlined mix whose threads each held a row's norms and weights through its
// reductions computed one simdgroup's outputs wrongly, now and then, under
// shader validation (as moe_gate_up_staged in moe_gguf.metal). A call of at
// most that many rows, every one-lane decode step's among them, runs
// block_residual_mix.metal's threadgroup a row and query instead.
constant constexpr uint kRowThreads = BLOCK_RESIDUAL_MIX_ROW_THREADS;
constant constexpr uint kRowSimdgroups = kRowThreads / 32;
constant constexpr uint kRowStates = BLOCK_RESIDUAL_MIX_MAXIMUM_STATES;
constant constexpr uint kRowQueries = BLOCK_RESIDUAL_MIX_MAXIMUM_QUERIES;
constant constexpr uint kRowNorms = BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS;

// What a row's phases share in threadgroup memory: a partial per simdgroup
// of each reduction, its own slots each, and the parked values: each state's
// inverse RMS without weights, each query's softmax weight of each state,
// and the norm variant's each output's inverse RMS.
struct MixRowShared {
  float partials[(kRowStates + kRowQueries * kRowStates + kRowNorms) * kRowSimdgroups];
  float inverse[kRowStates];
  float weights[kRowQueries * kRowStates];
  float output_inverse[kRowNorms];
};

// The sum over the threadgroup of each thread's `sum`: per simdgroup, then
// over the simdgroups in order by thread 0, which hands the total to
// `finish` and parks what it returns at `parked`; every thread past the
// second barrier. `partials` holds this reduction's slot a simdgroup.
template <class Finish>
inline void mix_row_reduce(float sum, threadgroup float *partials, threadgroup float *parked, uint tid, uint lane,
                           uint sg, Finish finish) {
  sum = simd_sum(sum);
  if (lane == 0) partials[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < kRowSimdgroups; ++g) total += partials[g];
    *parked = finish(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase one: each state's inverse RMS and each query's score of it, from
// one pass over the row's states that takes each state's squares and each
// query's dot with it (the threads' columns in order), reduced together, as
// block_residual_mix_scores takes a query's: each sum by simd_sum into its
// own slot a simdgroup, then by thread i sum i over the simdgroups; past a
// barrier the score of a normalized state is the dot times the state's
// inverse RMS. Then, by thread 0, the softmax of each query's scaled scores.
// Norms and scores apart took a pass and two barriers a state and a score.
template <class Current>
__attribute__((noinline)) void mix_row_weights(BlockResidualMixRow<Current> mix, device const float *queries,
                                               constant BlockResidualMixParams &p, uint tid, uint lane, uint sg,
                                               threadgroup MixRowShared &shared) {
  const uint states = mix.states();
  // Each state's squares, then query q's dot with state s at
  // kRowStates + q kRowStates + s.
  constexpr uint kSums = kRowStates + kRowQueries * kRowStates;
  static_assert(kSums <= kRowThreads, "a thread a sum");
  float sums[kSums];
  for (uint i = 0; i < kSums; ++i) sums[i] = 0.0f;
  for (uint d = tid; d < p.hidden; d += kRowThreads) {
    float weights[kRowQueries];
    for (uint q = 0; q < kRowQueries; ++q) weights[q] = q < p.queries ? queries[ulong(q) * p.hidden + d] : 0.0f;
    for (uint s = 0; s < kRowStates; ++s)
      if (s < states) {
        const float x = mix.state(s, d);
        sums[s] += x * x;
        for (uint q = 0; q < kRowQueries; ++q) sums[kRowStates + q * kRowStates + s] += weights[q] * x;
      }
  }
  for (uint i = 0; i < kSums; ++i) {
    const float sum = simd_sum(sums[i]);
    if (lane == 0) shared.partials[i * kRowSimdgroups + sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  if (tid < kSums)
    for (uint g = 0; g < kRowSimdgroups; ++g) total += shared.partials[tid * kRowSimdgroups + g];
  if (tid < states) shared.inverse[tid] = rsqrt(total / p.hidden + kRmsEpsilon);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid >= kRowStates && tid < kSums) {
    const uint s = (tid - kRowStates) % kRowStates;
    if ((tid - kRowStates) / kRowStates < p.queries && s < states)
      shared.weights[tid - kRowStates] = total * shared.inverse[s];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) block_residual_mix_softmax(shared.weights, p.queries, kRowStates, states, p);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase two of block_residual_mix_rows: each query's output as fp32 rows.
template <class Current>
__attribute__((noinline)) void mix_row_outputs(BlockResidualMixRow<Current> mix, device float *output,
                                               constant BlockResidualMixParams &p, uint row, uint tid,
                                               threadgroup const MixRowShared &shared) {
  for (uint q = 0; q < p.queries; ++q)
    for (uint d = tid; d < p.hidden; d += kRowThreads)
      output[(ulong(row) * p.queries + q) * p.hidden + d] = block_residual_mix_output(mix, q, d, shared);
}

// Phase two of block_residual_mix_norm_rows: each query's output's inverse
// RMS, then its normalized and weighted bf16 rows.
template <class Current>
__attribute__((noinline)) void mix_row_norm_outputs(BlockResidualMixRow<Current> mix, device const float *norm0,
                                                    device const float *norm1, device const float *norm2,
                                                    device bfloat *output0, device bfloat *output1,
                                                    device bfloat *output2, constant BlockResidualMixParams &p,
                                                    uint row, uint tid, uint lane, uint sg,
                                                    threadgroup MixRowShared &shared) {
  for (uint q = 0; q < p.queries; ++q) {
    float squares = 0.0f;
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const float o = block_residual_mix_output(mix, q, d, shared);
      squares += o * o;
    }
    mix_row_reduce(squares, shared.partials + (kRowStates + kRowQueries * kRowStates + q) * kRowSimdgroups,
                   &shared.output_inverse[q], tid, lane, sg,
                   [&](float total) { return rsqrt(total / p.hidden + kRmsEpsilon); });
  }
  device const float *norms[kRowNorms] = {norm0, norm1, norm2};
  device bfloat *outputs[kRowNorms] = {output0, output1, output2};
  for (uint q = 0; q < p.queries; ++q) {
    const float inverse = shared.output_inverse[q];
    for (uint d = tid; d < p.hidden; d += kRowThreads)
      outputs[q][ulong(row) * p.hidden + d] =
          bfloat((block_residual_mix_output(mix, q, d, shared) * inverse) * norms[q][d]);
  }
}

// Phase three of block_residual_mix_rows with normalized outputs
// (p.normalized): each query's fp32 row, once every thread has written its
// columns, through the message gates' input normalization into its bf16 row.
inline void mix_row_normalized(device const float *output, device bfloat *normalized,
                               constant BlockResidualMixParams &p, uint row, uint tid, uint lane, uint sg,
                               threadgroup float *partials) {
  threadgroup_barrier(mem_flags::mem_device);
  for (uint q = 0; q < p.queries; ++q) {
    const ulong out = (ulong(row) * p.queries + q) * p.hidden;
    message_gate_normalize<kRowThreads>(output + out, normalized + out, p.hidden, tid, lane, sg, partials);
  }
}

// Each query's output of each of the dispatch's mixes as fp32 rows
// [rows][queries][hidden], mix m's into output m, and with mixes.normalized
// each through the message gates' input normalization into normalized m: the
// message gates' inputs and the loop's messages. Grid: (rows, mixes).
// Without a second mix its slots bind the first's, and without normalized
// outputs those slots the outputs.
template <class Current>
kernel void block_residual_mix_rows(device const bfloat *stack0 [[buffer(0)]],
                                    device const Current *current0 [[buffer(1)]],
                                    device const float *queries0 [[buffer(2)]], device float *output0 [[buffer(3)]],
                                    device bfloat *normalized0 [[buffer(4)]], device const bfloat *stack1 [[buffer(5)]],
                                    device const Current *current1 [[buffer(6)]],
                                    device const float *queries1 [[buffer(7)]], device float *output1 [[buffer(8)]],
                                    device bfloat *normalized1 [[buffer(9)]],
                                    constant BlockResidualMixesParams &mixes [[buffer(10)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MixRowShared shared;
  threadgroup float partials[kMessageGateNormalizeSlots];
  const uint row = group.x, m = group.y;
  if (m >= mixes.mixes) return;
  constant BlockResidualMixParams &p = mixes.mix[m];
  if (row >= p.rows) return;
  const BlockResidualMixRow<Current> mix{(m ? stack1 : stack0) + ulong(row) * p.hidden,
                                         (m ? current1 : current0) + ulong(row) * p.hidden, p.stack, p.stack_plane};
  mix_row_weights(mix, m ? queries1 : queries0, p, tid, lane, sg, shared);
  mix_row_outputs(mix, m ? output1 : output0, p, row, tid, shared);
  if (mixes.normalized)
    mix_row_normalized(m ? output1 : output0, m ? normalized1 : normalized0, p, row, tid, lane, sg, partials);
}

// Each query's output through its weighted RMS norm (F32 weights, as a GGUF
// stores its norms) into its own bf16 rows: the input of the projections
// after the mix, which prepare their own tables, as after the
// hyper-connection mix (shared/hyper_connection.metal). Unused norm and
// output slots bind the first. Grid: rows.
template <class Current>
kernel void block_residual_mix_norm_rows(device const bfloat *stack [[buffer(0)]],
                                         device const Current *current [[buffer(1)]],
                                         device const float *queries [[buffer(2)]],
                                         device const float *norm0 [[buffer(3)]],
                                         device const float *norm1 [[buffer(4)]],
                                         device const float *norm2 [[buffer(5)]],
                                         device bfloat *output0 [[buffer(6)]], device bfloat *output1 [[buffer(7)]],
                                         device bfloat *output2 [[buffer(8)]],
                                         constant BlockResidualMixesParams &mixes [[buffer(9)]],
                                         uint row [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MixRowShared shared;
  constant BlockResidualMixParams &p = mixes.mix[0];
  if (row >= p.rows) return;
  const BlockResidualMixRow<Current> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                  p.stack_plane};
  mix_row_weights(mix, queries, p, tid, lane, sg, shared);
  mix_row_norm_outputs(mix, norm0, norm1, norm2, output0, output1, output2, p, row, tid, lane, sg, shared);
}

// The instances: bf16 current rows (a layer's input), and fp32 (_f32, the
// residual inside a layer).
#define BLOCK_RESIDUAL_MIX_ROW_INSTANCES(Current, suffix)                                                      \
  using MixRowKernel##Current =                                                                                 \
      void(device const bfloat *, device const Current *, device const float *, device float *, device bfloat *, \
           device const bfloat *, device const Current *, device const float *, device float *, device bfloat *, \
           constant BlockResidualMixesParams &, uint2, uint, uint, uint);                                      \
  using MixNormRowKernel##Current =                                                                             \
      void(device const bfloat *, device const Current *, device const float *, device const float *,         \
           device const float *, device const float *, device bfloat *, device bfloat *, device bfloat *,      \
           constant BlockResidualMixesParams &, uint, uint, uint, uint);                                       \
  template [[host_name("block_residual_mix_rows" suffix)]] kernel MixRowKernel##Current                        \
      block_residual_mix_rows<Current>;                                                                         \
  template [[host_name("block_residual_mix_norm_rows" suffix)]] kernel MixNormRowKernel##Current               \
      block_residual_mix_norm_rows<Current>;
BLOCK_RESIDUAL_MIX_ROW_INSTANCES(bfloat, "")
BLOCK_RESIDUAL_MIX_ROW_INSTANCES(float, "_f32")
#undef BLOCK_RESIDUAL_MIX_ROW_INSTANCES
