#pragma clang fp reassociate(off)
#include "metal/abi/BlockResidualMix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/block_residual_mix_row.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// Block residual mixes (metal/abi/BlockResidualMix.h): a threadgroup of
// kMixThreads threads a row and query (grid y, a dispatch's mixes' queries
// one after another), each thread the columns
// tid + kMixThreads i, in fp32 from the bf16 stack planes and the current
// rows, bf16 (a layer's input) or fp32 (a residual inside a layer): Current,
// the instance's element type, the fp32 one suffixed _f32. A threadgroup
// runs in phases, each a function out of line, which hand over through
// threadgroup memory alone: the row's states' norms and its query's weights
// of the normalized states, then its query's output from those and the
// states.
// A phase takes all its sums in one pass over the columns, each thread's of
// its columns in order, and reduces them together: each by simd_sum into
// its own slot a simdgroup, then, past one barrier, a slot's partials added
// by simd_sum, a simdgroup's partial a lane (block_residual_mix_total): by
// simdgroup 0 alone for the norms and scores, whose softmax it parks past a
// second barrier, and by every simdgroup itself for an output's squares, so
// that every thread holds that total with no second barrier. Every query's
// threadgroup computes the row's norms alike, so each query's output is the
// one a threadgroup of every query computes (metal/abi/BlockResidualMix.h has
// what a threadgroup a row took). A thread's sums live in an array of the
// most states, every slot past the row's zero, and reach threadgroup memory
// before the barrier, and the norms and weights live in threadgroup memory:
// no thread keeps an array of them across a barrier (an inlined mix whose
// threads each held a row's norms and weights through its reductions
// computed one simdgroup's outputs wrongly, now and then, under shader
// validation, as moe_gate_up_staged in moe_gguf.metal).
constant constexpr uint kMixThreads = BLOCK_RESIDUAL_MIX_THREADS;
constant constexpr uint kMixSimdgroups = kMixThreads / 32;
constant constexpr uint kStates = BLOCK_RESIDUAL_MIX_MAXIMUM_STATES;
constant constexpr uint kNorms = BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS;

// What a threadgroup's phases share in threadgroup memory: a partial per
// simdgroup of each sum (each state's squares, the query's dot with each
// state, the output's squares), and each state's inverse RMS without weights
// and the query's softmax weight of each state.
struct BlockResidualMixShared {
  float partials[(2 * kStates + 1) * kMixSimdgroups];
  float inverse[kStates];
  float weights[kStates];
};
static_assert(kMixSimdgroups == 32, "a slot's partials are one a lane of a simdgroup");

// A slot's total over the threadgroup's simdgroups, past the barrier after
// their partials: a simdgroup's partial a lane, added by simd_sum, alike in
// every simdgroup.
inline float block_residual_mix_total(threadgroup const float *slot, uint lane) { return simd_sum(slot[lane]); }

// Query q's output at column d from the weights and norms: the weighted sum
// of the normalized states (block_residual_mix_output's sum).
template <class Current>
inline float block_residual_mix_query_output(BlockResidualMixRow<Current> mix, uint d,
                                             threadgroup const float *weights, threadgroup const float *inverse) {
  float sum = 0.0f;
  for (uint s = 0; s < mix.states(); ++s) sum += weights[s] * (mix.state(s, d) * inverse[s]);
  return sum;
}

// Phase one: each state's inverse RMS and the query's score of it, from one
// pass over the row's states that takes each state's squares and its dot with
// the query, reduced together: the score of the normalized state is the dot
// times the state's inverse RMS. Then, by simdgroup 0, the softmax of the
// scaled scores, its lanes 0 to states - 1 parking the norms and weights
// past a second barrier. Norms and scores apart took a pass and a reduction
// more, and a thread's totals parked between two barriers, and thread 0's
// softmax past a third; every simdgroup taking the softmax itself, into its
// own copy with no second barrier, took longer on Apple10 and Apple11 (its
// 32 copies on one core cost more than the barrier they save).
template <class Current>
__attribute__((noinline)) void block_residual_mix_scores(BlockResidualMixRow<Current> mix,
                                                         device const float *query,
                                                         constant BlockResidualMixParams &p, uint tid, uint lane,
                                                         uint sg, threadgroup BlockResidualMixShared &shared) {
  const uint states = mix.states();
  // Each state's squares, then each state's dot with the query.
  float sums[2 * kStates];
  for (uint s = 0; s < 2 * kStates; ++s) sums[s] = 0.0f;
  for (uint d = tid; d < p.hidden; d += kMixThreads) {
    const float weight = query[d];
    for (uint s = 0; s < kStates; ++s)
      if (s < states) {
        const float x = mix.state(s, d);
        sums[s] += x * x;
        sums[kStates + s] += weight * x;
      }
  }
  for (uint i = 0; i < 2 * kStates; ++i) {
    const float sum = simd_sum(sums[i]);
    if (lane == 0) shared.partials[i * kMixSimdgroups + sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Simdgroup 0: the softmax of the scaled scores, block_residual_mix_softmax's.
  if (sg == 0) {
    float inverse[kStates], weight[kStates], maximum = -INFINITY;
    for (uint s = 0; s < kStates; ++s) {
      const float squares = block_residual_mix_total(shared.partials + s * kMixSimdgroups, lane);
      const float dot = block_residual_mix_total(shared.partials + (kStates + s) * kMixSimdgroups, lane);
      inverse[s] = rsqrt(squares / p.hidden + kRmsEpsilon);
      weight[s] = dot * inverse[s];
      if (s < states) maximum = max(maximum, weight[s] * p.scale);
    }
    float total = 0.0f;
    for (uint s = 0; s < kStates; ++s) {
      weight[s] = s < states ? exp(weight[s] * p.scale - maximum) : 0.0f;
      total += weight[s];
    }
    for (uint s = 0; s < kStates; ++s)
      if (lane == s) {
        shared.inverse[s] = inverse[s];
        shared.weights[s] = weight[s] / total;
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase three of block_residual_mix: the query's output as fp32 rows; with
// `normalized`, each thread's columns' squares, their total past a barrier,
// and the output, as the thread wrote it, times its inverse RMS without
// weights into bf16 (the message gates' input normalization,
// kernels/common/message_gate_normalize.h, from the mix's own sums).
template <class Current>
__attribute__((noinline)) void block_residual_mix_outputs(BlockResidualMixRow<Current> mix, device float *output,
                                                          device bfloat *normalized,
                                                          constant BlockResidualMixParams &p, uint tid, uint lane,
                                                          uint sg, threadgroup BlockResidualMixShared &shared) {
  float squares = 0.0f;
  for (uint d = tid; d < p.hidden; d += kMixThreads) {
    const float o = block_residual_mix_query_output(mix, d, shared.weights, shared.inverse);
    output[d] = o;
    squares += o * o;
  }
  if (!normalized) return;
  threadgroup float *slot = shared.partials + 2 * kStates * kMixSimdgroups;
  squares = simd_sum(squares);
  if (lane == 0) slot[sg] = squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = rsqrt(block_residual_mix_total(slot, lane) / p.hidden + kRmsEpsilon);
  for (uint d = tid; d < p.hidden; d += kMixThreads) normalized[d] = bfloat(output[d] * inverse);
}

// Phase three of block_residual_mix_norm: the query's output's inverse RMS,
// then its normalized and weighted bf16 row.
template <class Current>
__attribute__((noinline)) void block_residual_mix_norm_outputs(BlockResidualMixRow<Current> mix,
                                                               device const float *norm, device bfloat *output,
                                                               constant BlockResidualMixParams &p, uint tid,
                                                               uint lane, uint sg,
                                                               threadgroup BlockResidualMixShared &shared) {
  threadgroup float *slot = shared.partials + 2 * kStates * kMixSimdgroups;
  float squares = 0.0f;
  for (uint d = tid; d < p.hidden; d += kMixThreads) {
    const float o = block_residual_mix_query_output(mix, d, shared.weights, shared.inverse);
    squares += o * o;
  }
  squares = simd_sum(squares);
  if (lane == 0) slot[sg] = squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = rsqrt(block_residual_mix_total(slot, lane) / p.hidden + kRmsEpsilon);
  for (uint d = tid; d < p.hidden; d += kMixThreads)
    output[d] = bfloat((block_residual_mix_query_output(mix, d, shared.weights, shared.inverse) * inverse) *
                       norm[d]);
}

// Each query's output of each of the dispatch's mixes as fp32 rows
// [rows][queries][hidden], mix m's into output m, and with mixes.normalized
// each through the message gates' input normalization into normalized m: the
// message gates' inputs and the loop's messages. Grid: (rows, the first
// mix's queries then the second's). Without a second mix its slots bind the
// first's, and without normalized outputs those slots the outputs.
template <class Current>
kernel void block_residual_mix(device const bfloat *stack0 [[buffer(0)]], device const Current *current0 [[buffer(1)]],
                               device const float *queries0 [[buffer(2)]], device float *output0 [[buffer(3)]],
                               device bfloat *normalized0 [[buffer(4)]], device const bfloat *stack1 [[buffer(5)]],
                               device const Current *current1 [[buffer(6)]], device const float *queries1 [[buffer(7)]],
                               device float *output1 [[buffer(8)]], device bfloat *normalized1 [[buffer(9)]],
                               constant BlockResidualMixesParams &mixes [[buffer(10)]],
                               uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  const uint m = group.y < mixes.mix[0].queries ? 0 : 1, row = group.x, q = m ? group.y - mixes.mix[0].queries : group.y;
  if (m >= mixes.mixes) return;
  constant BlockResidualMixParams &p = mixes.mix[m];
  if (row >= p.rows || q >= p.queries) return;
  const BlockResidualMixRow<Current> mix{(m ? stack1 : stack0) + ulong(row) * p.hidden,
                                         (m ? current1 : current0) + ulong(row) * p.hidden, p.stack, p.stack_plane};
  const ulong out = (ulong(row) * p.queries + q) * p.hidden;
  block_residual_mix_scores(mix, (m ? queries1 : queries0) + ulong(q) * p.hidden, p, tid, lane, sg, shared);
  block_residual_mix_outputs(mix, (m ? output1 : output0) + out,
                             mixes.normalized ? (m ? normalized1 : normalized0) + out : nullptr, p, tid, lane, sg,
                             shared);
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
                                    constant BlockResidualMixesParams &mixes [[buffer(9)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  constant BlockResidualMixParams &p = mixes.mix[0];
  const uint row = group.x, q = group.y;
  if (row >= p.rows || q >= p.queries || q >= kNorms) return;
  const BlockResidualMixRow<Current> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                         p.stack_plane};
  block_residual_mix_scores(mix, queries + ulong(q) * p.hidden, p, tid, lane, sg, shared);
  device const float *norm = q == 0 ? norm0 : q == 1 ? norm1 : norm2;
  device bfloat *output = q == 0 ? output0 : q == 1 ? output1 : output2;
  block_residual_mix_norm_outputs(mix, norm, output + ulong(row) * p.hidden, p, tid, lane, sg, shared);
}

// A decode step's mix of one query whose fp32 current rows are its mixer's
// output convolution (metal/abi/BlockResidualMix.h,
// BlockResidualMixConvolutionParams): each row's threadgroup first convolves
// its row into `current` (block_residual_mix_convolve; a row of no lane is
// not convolved), then mixes it, Norm false as block_residual_mix (`out` its
// fp32 output, and with normalized outputs `normalized` its bf16 rows the
// message gate reads), Norm true as block_residual_mix_norm (`norm` its
// norm, `normalized` its bf16 rows). Grid: (rows, 1).
template <bool Norm>
kernel void block_residual_mix_convolved(
    device const bfloat *stack [[buffer(0)]], device float *current [[buffer(1)]],
    device const float *queries [[buffer(2)]], device float *out [[buffer(3)]], device const float *norm [[buffer(4)]],
    device bfloat *normalized [[buffer(5)]], device const bfloat *input [[buffer(6)]],
    device const bfloat *state_in0 [[buffer(7)]], device const bfloat *state_in1 [[buffer(8)]],
    device const bfloat *state_in2 [[buffer(9)]], device const bfloat *state_in3 [[buffer(10)]],
    device const bfloat *state_in4 [[buffer(11)]], device const bfloat *state_in5 [[buffer(12)]],
    device const bfloat *state_in6 [[buffer(13)]], device const bfloat *state_in7 [[buffer(14)]],
    device const float *taps [[buffer(15)]], device const bfloat *residual [[buffer(16)]],
    device const SplashLaneRows *layout [[buffer(17)]], device const SplashTreeRow *tree [[buffer(18)]],
    device bfloat *tape [[buffer(19)]], constant BlockResidualMixConvolutionParams &params [[buffer(20)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup BlockResidualMixShared shared;
  constant BlockResidualMixParams &p = params.mixes.mix[0];
  constant ShortConvolutionParams &c = params.convolution;
  const uint row = group.x;
  if (row >= p.rows || group.y) return;
  const uint s = block_residual_mix_lane(layout, c.sequences, row);
  if (s < SPLASH_MAXIMUM_BATCH_WIDTH)
    block_residual_mix_convolve<kMixThreads>(
        input, SPLASH_LANE_BINDING(s, state_in0, state_in1, state_in2, state_in3, state_in4, state_in5, state_in6,
                                   state_in7),
        taps, residual, current, layout[s], tree, tape, c, row, tid);
  const BlockResidualMixRow<float> mix{stack + ulong(row) * p.hidden, current + ulong(row) * p.hidden, p.stack,
                                       p.stack_plane};
  block_residual_mix_scores(mix, queries, p, tid, lane, sg, shared);
  const ulong at = ulong(row) * p.hidden;
  if (Norm) {
    block_residual_mix_norm_outputs(mix, norm, normalized + at, p, tid, lane, sg, shared);
    return;
  }
  block_residual_mix_outputs(mix, out + at, params.mixes.normalized ? normalized + at : nullptr, p, tid, lane, sg,
                             shared);
}

// The instances: bf16 current rows (a layer's input), and fp32 (_f32, the
// residual inside a layer).
#define BLOCK_RESIDUAL_MIX_INSTANCES(Current, suffix)                                                          \
  using BlockResidualMixKernel##Current =                                                                             \
      void(device const bfloat *, device const Current *, device const float *, device float *, device bfloat *,    \
           device const bfloat *, device const Current *, device const float *, device float *, device bfloat *,    \
           constant BlockResidualMixesParams &, uint2, uint, uint, uint);                                            \
  using BlockResidualMixNormKernel##Current =                                                                         \
      void(device const bfloat *, device const Current *, device const float *, device const float *,               \
           device const float *, device const float *, device bfloat *, device bfloat *, device bfloat *,            \
           constant BlockResidualMixesParams &, uint2, uint, uint, uint);                                            \
  template [[host_name("block_residual_mix" suffix)]] kernel BlockResidualMixKernel##Current                         \
      block_residual_mix<Current>;                                                                                    \
  template [[host_name("block_residual_mix_norm" suffix)]] kernel BlockResidualMixNormKernel##Current               \
      block_residual_mix_norm<Current>;
BLOCK_RESIDUAL_MIX_INSTANCES(bfloat, "")
BLOCK_RESIDUAL_MIX_INSTANCES(float, "_f32")
#undef BLOCK_RESIDUAL_MIX_INSTANCES

// The convolved instances (fp32 current rows): block_residual_mix's and
// block_residual_mix_norm's.
using BlockResidualMixConvolvedKernel =
    void(device const bfloat *, device float *, device const float *, device float *, device const float *,
         device bfloat *, device const bfloat *, device const bfloat *, device const bfloat *, device const bfloat *,
         device const bfloat *, device const bfloat *, device const bfloat *, device const bfloat *,
         device const bfloat *, device const float *, device const bfloat *, device const SplashLaneRows *,
         device const SplashTreeRow *, device bfloat *, constant BlockResidualMixConvolutionParams &, uint2, uint,
         uint, uint);
template [[host_name("block_residual_mix_convolved")]] kernel BlockResidualMixConvolvedKernel
    block_residual_mix_convolved<false>;
template [[host_name("block_residual_mix_norm_convolved")]] kernel BlockResidualMixConvolvedKernel
    block_residual_mix_convolved<true>;
