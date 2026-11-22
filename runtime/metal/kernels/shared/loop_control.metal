#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/LoopControl.h"
#include "metal/kernels/common/sgmatrix.h"

#include <metal_stdlib>

using namespace metal;

// Inkling 2's loop control (metal/abi/LoopControl.h): the gate of a loop
// pass's message into a stream of a layer's input, in fp32 throughout on
// simdgroup 8x8 MMA with exact operands (the fp32 rows, the weights as stored,
// F32 or BF16 widened), so a result differs from the fp64 one by fp32
// accumulation alone. The rank-64 projections are too small for the
// projection tiles' plans and run here: the down projection with the
// inputs' and messages' RMS (two passes would read each row twice), the
// forget and gain projections with the tanh gates and the mix, so that the
// rows of neither pass reach memory between them.

constant constexpr uint kFragments = LOOP_CONTROL_TILE_ROWS / 8; // 8-row fragments of a tile
constant constexpr uint kRows = LOOP_CONTROL_TILE_ROWS;
constant constexpr uint kStep = 64;                              // inputs of a step: eight runs of 8
constant constexpr uint kSimdgroups = LOOP_CONTROL_THREADS / 32;
constant constexpr uint kParts = kSimdgroups / 2;                // K parts of each half of the down input
constant constexpr float kEpsilon = SPLASH_RMS_EPSILON;
static_assert(kParts >= kFragments, "down simdgroup f reduces fragment f");
static_assert(kSimdgroups * 8 == LOOP_CONTROL_GATE_COLUMNS, "a gate simdgroup takes 8 columns");
static_assert(LOOP_CONTROL_DOWN_COLUMNS == 8, "a down threadgroup takes one 8-column MMA tile");

// Stream `s`'s binding of the four (LOOP_CONTROL_MAXIMUM_STREAMS). A macro,
// as SPLASH_LANE_BINDING is, so the chain stays selects.
static_assert(LOOP_CONTROL_MAXIMUM_STREAMS == 4, "one binding per stream");
#define LOOP_CONTROL_STREAM(s, b0, b1, b2, b3) ((s) == 0 ? (b0) : ((s) == 1 ? (b1) : ((s) == 2 ? (b2) : (b3))))

// Eight consecutive values from `values` (16-byte aligned: a row of a multiple
// of 8 values from an aligned start), widened to fp32; zeros outside.
struct Eight {
  float4 low, high;
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
};
inline Eight eight(device const float *values, bool inside) {
  device const float4 *quads = reinterpret_cast<device const float4 *>(values);
  return inside ? Eight{quads[0], quads[1]} : Eight{float4(0.0f), float4(0.0f)};
}
inline Eight eight(device const bfloat *values, bool inside) {
  if (!inside) return {float4(0.0f), float4(0.0f)};
  const bfloat4 low = reinterpret_cast<device const bfloat4 *>(values)[0],
                high = reinterpret_cast<device const bfloat4 *>(values)[1];
  return {float4(low), float4(high)};
}

// d = W_down . concat(rms(u), rms(m)) for output columns [8 tile.x, + 8) of
// rows [32 tile.y, + 32). Simdgroups 0-3 take the input's half of the K
// inputs (u), 4-7 the message's (m), each a part of the half's steps of 64
// inputs for every fragment, with each lane's sum of squares of the inputs
// it reads; simdgroup f then adds fragment f's parts in order and scales
// each half by its rows' inverse RMS: d = (W_u . u) / rms(u) + (W_m . m) /
// rms(m). As in kernels/shared/gguf_float.metal, MMA j of a step from input k
// multiplies inputs k + 8 i + j over its index i, lane (fm, fn) loading
// W[column fn, fn + 1][k + 8 fm, + 8) and x[row fm][k + 8 fn, + 16), so the
// four lanes of a row read each of its inputs once (their sums of squares
// add over lane bits 0 and 3), and even and odd MMAs accumulate apart.
template <typename W>
inline void loop_control_down(device const float *input, device const float *message, device const W *weights,
                              device float *low, constant LoopControlParams &p, uint2 tile, uint sg, uint lane,
                              threadgroup float2 *parts, threadgroup float *squares) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint hidden = p.hidden, width = 2 * hidden, column = tile.x * 8 + l.fn, row0 = tile.y * kRows;
  const uint live = min(kRows, p.rows - row0), side = sg / kParts, part = sg % kParts;
  device const float *x = side ? message : input;
  const uint stride = side ? p.message_stride : p.input_stride;
  const uint steps = hidden / kStep, first = part * steps / kParts, last = (part + 1) * steps / kParts;
  device const W *w0 = weights + ulong(column) * width + side * hidden, *w1 = w0 + width;
  float2 acc[kFragments][2];
  float sq[kFragments];
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) {
    acc[f][0] = acc[f][1] = float2(0.0f);
    sq[f] = 0.0f;
  }
  // The weights and the first fragment's inputs load a step ahead, so that a
  // decode step's one fragment waits on memory once, not once a step.
  const bool inside0 = l.fm < live;
  device const float *x0 = x + ulong(row0 + (inside0 ? l.fm : 0)) * stride + 8 * l.fn;
  Eight next[4];
  const auto load = [&](uint step) {
    const uint k = step * kStep;
    next[0] = eight(w0 + k + 8 * l.fm, true);
    next[1] = eight(w1 + k + 8 * l.fm, true);
    next[2] = eight(x0 + k, inside0);
    next[3] = eight(x0 + k + 8, inside0);
  };
  if (first < last) load(first);
  for (uint step = first; step < last; ++step) {
    const uint k = step * kStep;
    const Eight b0 = next[0], b1 = next[1], first0 = next[2], first1 = next[3];
    if (step + 1 < last) load(step + 1);
    // Fragments past the live rows are skipped with continue, not ended with
    // break, so the later fragments' MMAs stay under uniform control flow
    // (kernels/shared/gguf_float.metal).
#pragma unroll
    for (uint f = 0; f < kFragments; ++f) {
      if (f * 8 >= live) continue;
      const uint row = f * 8 + l.fm;
      const bool inside = row < live;
      device const float *values = x + ulong(row0 + (inside ? row : 0)) * stride + k + 8 * l.fn;
      const Eight a0 = f ? eight(values, inside) : first0, a1 = f ? eight(values + 8, inside) : first1;
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        sq[f] += a0[j] * a0[j] + a1[j] * a1[j];
        sgmatrix::mma_acc<float>(acc[f][j % 2], float2(a0[j], a1[j]), float2(b0[j], b1[j]));
      }
    }
  }
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) {
    sq[f] += simd_shuffle_xor(sq[f], ushort(1));
    sq[f] += simd_shuffle_xor(sq[f], ushort(8));
    parts[(sg * kFragments + f) * 32 + lane] = acc[f][0] + acc[f][1];
    squares[(sg * kFragments + f) * 32 + lane] = sq[f];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint f = sg, row = f * 8 + l.fm;
  if (f >= kFragments || row >= live) return;
  float2 sums[2] = {float2(0.0f), float2(0.0f)};
  float squared[2] = {0.0f, 0.0f};
  for (uint s = 0; s < kSimdgroups; ++s) {
    sums[s / kParts] += parts[(s * kFragments + f) * 32 + lane];
    squared[s / kParts] += squares[(s * kFragments + f) * 32 + lane];
  }
  const float2 d =
      rsqrt(squared[0] / float(hidden) + kEpsilon) * sums[0] + rsqrt(squared[1] / float(hidden) + kEpsilon) * sums[1];
  device float *out = low + ulong(row0 + row) * p.rank + column;
  out[0] = d.x;
  out[1] = d.y;
}

// Grid (rank / 8, ceil(rows / 32), streams), LOOP_CONTROL_THREADS threads.
#define LOOP_CONTROL_DOWN_KERNEL(Name, W)                                                                              \
  kernel void Name(device const float *input0 [[buffer(0)]], device const float *input1 [[buffer(1)]],                 \
                   device const float *input2 [[buffer(2)]], device const float *input3 [[buffer(3)]],                 \
                   device const float *message0 [[buffer(4)]], device const float *message1 [[buffer(5)]],             \
                   device const float *message2 [[buffer(6)]], device const float *message3 [[buffer(7)]],             \
                   device const W *weights0 [[buffer(8)]], device const W *weights1 [[buffer(9)]],                     \
                   device const W *weights2 [[buffer(10)]], device const W *weights3 [[buffer(11)]],                   \
                   device float *low0 [[buffer(12)]], device float *low1 [[buffer(13)]],                               \
                   device float *low2 [[buffer(14)]], device float *low3 [[buffer(15)]],                               \
                   constant LoopControlParams &p [[buffer(16)]], uint3 tg [[threadgroup_position_in_grid]],            \
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {              \
    threadgroup float2 parts[kSimdgroups * kFragments * 32];                                                       \
    threadgroup float squares[kSimdgroups * kFragments * 32];                                                      \
    const uint s = tg.z;                                                                                               \
    if (s >= min(p.streams, LOOP_CONTROL_MAXIMUM_STREAMS)) return;                                                     \
    loop_control_down<W>(LOOP_CONTROL_STREAM(s, input0, input1, input2, input3),                                       \
                         LOOP_CONTROL_STREAM(s, message0, message1, message2, message3),                               \
                         LOOP_CONTROL_STREAM(s, weights0, weights1, weights2, weights3),                               \
                         LOOP_CONTROL_STREAM(s, low0, low1, low2, low3), p, tg.xy, sg, lane, parts, squares);          \
  }
LOOP_CONTROL_DOWN_KERNEL(loop_control_down_f32, float)
LOOP_CONTROL_DOWN_KERNEL(loop_control_down_bf16, bfloat)
#undef LOOP_CONTROL_DOWN_KERNEL

// The forget and gain projections of d for output columns [64 tile.x + 8 sg,
// + 8) of rows [32 tile.y, + 32), and the gate's mix of each of their
// elements: out = (1 - tanh(W_forget . d)) u + tanh(W_gain . d) m. Lane (fm,
// fn) holds the projections' columns fn, fn + 1 of fragment row fm, the
// elements it then mixes, so an output may be its input's rows.
template <typename W>
inline void loop_control_gate(device const float *low, device const W *forget, device const W *gain,
                              device const float *input, device const float *message, device float *output,
                              constant LoopControlParams &p, uint2 tile, uint sg, uint lane) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint rank = p.rank, column = tile.x * LOOP_CONTROL_GATE_COLUMNS + sg * 8 + l.fn, row0 = tile.y * kRows;
  const uint live = min(kRows, p.rows - row0);
  device const W *f0 = forget + ulong(column) * rank, *f1 = f0 + rank;
  device const W *g0 = gain + ulong(column) * rank, *g1 = g0 + rank;
  // Each lane's inputs and messages load before the projections, whose
  // weights and rows they wait on with them.
  float2 forgets[kFragments][2], gains[kFragments][2], inputs[kFragments], messages[kFragments];
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) {
    forgets[f][0] = forgets[f][1] = gains[f][0] = gains[f][1] = float2(0.0f);
    const uint row = f * 8 + l.fm;
    if (row >= live) continue;
    device const float *u = input + ulong(row0 + row) * p.input_stride + column;
    device const float *m = message + ulong(row0 + row) * p.message_stride + column;
    inputs[f] = float2(u[0], u[1]);
    messages[f] = float2(m[0], m[1]);
  }
  for (uint k = 0; k < rank; k += kStep) {
    const Eight bf0 = eight(f0 + k + 8 * l.fm, true), bf1 = eight(f1 + k + 8 * l.fm, true);
    const Eight bg0 = eight(g0 + k + 8 * l.fm, true), bg1 = eight(g1 + k + 8 * l.fm, true);
#pragma unroll
    for (uint f = 0; f < kFragments; ++f) {
      if (f * 8 >= live) continue;
      const uint row = f * 8 + l.fm;
      const bool inside = row < live;
      device const float *values = low + ulong(row0 + (inside ? row : 0)) * rank + k + 8 * l.fn;
      const Eight a0 = eight(values, inside), a1 = eight(values + 8, inside);
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        const float2 a = float2(a0[j], a1[j]);
        sgmatrix::mma_acc<float>(forgets[f][j % 2], a, float2(bf0[j], bf1[j]));
        sgmatrix::mma_acc<float>(gains[f][j % 2], a, float2(bg0[j], bg1[j]));
      }
    }
  }
#pragma unroll
  for (uint f = 0; f < kFragments; ++f) {
    const uint row = f * 8 + l.fm;
    if (row >= live) continue;
    const float2 keep = 1.0f - precise::tanh(forgets[f][0] + forgets[f][1]);
    const float2 take = precise::tanh(gains[f][0] + gains[f][1]);
    const float2 mixed = keep * inputs[f] + take * messages[f];
    device float *out = output + ulong(row0 + row) * p.output_stride + column;
    out[0] = mixed.x;
    out[1] = mixed.y;
  }
}

// Grid (hidden / 64, ceil(rows / 32), streams), LOOP_CONTROL_THREADS threads.
#define LOOP_CONTROL_GATE_KERNEL(Name, W)                                                                              \
  kernel void Name(device const float *low0 [[buffer(0)]], device const float *low1 [[buffer(1)]],                     \
                   device const float *low2 [[buffer(2)]], device const float *low3 [[buffer(3)]],                     \
                   device const W *forget0 [[buffer(4)]], device const W *forget1 [[buffer(5)]],                       \
                   device const W *forget2 [[buffer(6)]], device const W *forget3 [[buffer(7)]],                       \
                   device const W *gain0 [[buffer(8)]], device const W *gain1 [[buffer(9)]],                           \
                   device const W *gain2 [[buffer(10)]], device const W *gain3 [[buffer(11)]],                         \
                   device const float *input0 [[buffer(12)]], device const float *input1 [[buffer(13)]],               \
                   device const float *input2 [[buffer(14)]], device const float *input3 [[buffer(15)]],               \
                   device const float *message0 [[buffer(16)]], device const float *message1 [[buffer(17)]],           \
                   device const float *message2 [[buffer(18)]], device const float *message3 [[buffer(19)]],           \
                   device float *output0 [[buffer(20)]], device float *output1 [[buffer(21)]],                         \
                   device float *output2 [[buffer(22)]], device float *output3 [[buffer(23)]],                         \
                   constant LoopControlParams &p [[buffer(24)]], uint3 tg [[threadgroup_position_in_grid]],            \
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {              \
    const uint s = tg.z;                                                                                               \
    if (s >= min(p.streams, LOOP_CONTROL_MAXIMUM_STREAMS)) return;                                                     \
    loop_control_gate<W>(                                                                                              \
        LOOP_CONTROL_STREAM(s, low0, low1, low2, low3), LOOP_CONTROL_STREAM(s, forget0, forget1, forget2, forget3),    \
        LOOP_CONTROL_STREAM(s, gain0, gain1, gain2, gain3), LOOP_CONTROL_STREAM(s, input0, input1, input2, input3),    \
        LOOP_CONTROL_STREAM(s, message0, message1, message2, message3),                                                \
        LOOP_CONTROL_STREAM(s, output0, output1, output2, output3), p, tg.xy, sg, lane);                               \
  }
LOOP_CONTROL_GATE_KERNEL(loop_control_gate_f32, float)
LOOP_CONTROL_GATE_KERNEL(loop_control_gate_bf16, bfloat)
#undef LOOP_CONTROL_GATE_KERNEL
