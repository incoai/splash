#include "metal/abi/KernelABI.h"
#include "metal/abi/ShortConvolution.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/short_convolution_row.h"
#include "metal/kernels/common/verify_tree.h"

#include <metal_stdlib>

using namespace metal;

// The short convolution (metal/abi/ShortConvolution.h): a threadgroup of
// kConvolutionThreads channels of one row, as ngram_embedding_conv runs its
// dilated convolution.
constant constexpr uint kConvolutionThreads = 256;
constant constexpr uint kTaps = SHORT_CONVOLUTION_TAPS;
constant constexpr int kStateRows = int(SHORT_CONVOLUTION_STATE_ROWS);
constant constexpr uint kCaptures = SHORT_CONVOLUTION_CAPTURES;

// The convolution over each of a command's sequences' rows (a prompt
// chunk's sequences, or a verify step's lanes), threadgroup (row, channel
// group, sequence) of sequence s's rows its entry of the command's layout:
// tap k of channel c reads the input k rows back (`input` its channel 0 of the
// command's first row, rows p.input_stride values apart), before the
// sequence's first row from its state (state_in s, its channels from
// `state_column`, the tape's too). The sum of the taps' products and
// the row's input, over the divisor, is rounded to bf16 and written as the
// output asks: Out bf16 (SHORT_CONVOLUTION_OUTPUT_PLAIN, or ROUNDED_SUM of an
// fp32 residual) or fp32 (SUM of a bf16 residual or none); Residual is the
// other type. Rows past a sequence's write states: the first kStateRows its
// state after its rows (write_state, state_out s), the next kStateRows each
// capture's of the sequence after its first capture_rows rows. A verify
// block's rows (p.tree, a row's place in `tree`) count back along their
// paths and copy their input into the tape, which the commit reads; a
// prompt's bind another buffer for the tree and the tape, which they do not
// touch. A row past its sequence's rows and states, or a sequence whose rows
// run past the command's, writes nothing.
// The buffers of a dispatch's captures.
struct ShortConvolutionCaptured {
  device bfloat *state[kCaptures];
};

template <class Out, class Residual>
inline void short_convolution_row(device const bfloat *input, uint state_column, device const bfloat *state_in,
                                  device bfloat *state_out, device const float *taps,
                                  device const bfloat *dynamic_taps, device const Residual *residual,
                                  device Out *output, device const SplashLaneRows *layout,
                                  device const SplashTreeRow *tree, device bfloat *tape,
                                  ShortConvolutionCaptured captured, constant ShortConvolutionParams &p,
                                  uint3 group, uint tid) {
  const uint c = group.y * kConvolutionThreads + tid, row = group.x, s = group.z;
  if (c >= p.channels || s >= min(p.sequences, SPLASH_MAXIMUM_BATCH_WIDTH)) return;
  const SplashLaneRows rows = layout[s];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  device const bfloat *sequence = input + ulong(rows.first) * p.input_stride;
  const ulong column = state_column + c;
  const auto at = [&](int t) -> bfloat {
    return t >= 0 ? sequence[ulong(t) * p.input_stride + c]
                  : state_in[ulong(kStateRows + t) * p.state_stride + column];
  };
  if (row < rows.rows) {
    if (p.captures_only) return;
    // The row's path and depth: a prompt's rows are a chain.
    const uint packed = rows.first + row;
    const uint path = p.tree ? tree[packed].ancestors : 0u;
    const int depth = p.tree ? int(tree[packed].depth) : int(row);
    const bfloat own = sequence[ulong(row) * p.input_stride + c];
    // Every channel of a dynamic head shares the row's taps of the head.
    device const bfloat *dynamic =
        dynamic_taps + ulong(packed) * p.dynamic_stride + (p.heads ? c / (p.channels / p.heads) : 0u) * kTaps;
    short_convolution_output(at, taps, dynamic, residual, output, p, c, packed, path, depth, own);
    if (p.tree) tape[ulong(packed) * p.state_stride + column] = own;
    return;
  }
  // Rows past the sequence's write its states: row rows + j its state after
  // its rows' row j, row rows + kStateRows (q + 1) + j capture q's.
  const uint past = row - rows.rows, j = past % uint(kStateRows), slot = past / uint(kStateRows);
  if (slot == 0) {
    if (p.write_state && !p.captures_only)
      state_out[ulong(j) * p.state_stride + column] = at(int(rows.rows) - kStateRows + int(j));
    return;
  }
  const uint q = slot - 1;
  if (q < p.captures && p.capture_sequences[q] == s)
    captured.state[q][ulong(j) * p.state_stride + column] = at(int(p.capture_rows[q]) - kStateRows + int(j));
}

#define SHORT_CONVOLUTION(Name, Out, Residual)                                                                 \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const bfloat *state_in0 [[buffer(1)]],     \
                   device const bfloat *state_in1 [[buffer(2)]], device const bfloat *state_in2 [[buffer(3)]], \
                   device const bfloat *state_in3 [[buffer(4)]], device const bfloat *state_in4 [[buffer(5)]], \
                   device const bfloat *state_in5 [[buffer(6)]], device const bfloat *state_in6 [[buffer(7)]], \
                   device const bfloat *state_in7 [[buffer(8)]], device bfloat *state_out0 [[buffer(9)]],      \
                   device bfloat *state_out1 [[buffer(10)]], device bfloat *state_out2 [[buffer(11)]],         \
                   device bfloat *state_out3 [[buffer(12)]], device bfloat *state_out4 [[buffer(13)]],         \
                   device bfloat *state_out5 [[buffer(14)]], device bfloat *state_out6 [[buffer(15)]],         \
                   device bfloat *state_out7 [[buffer(16)]], device const float *taps [[buffer(17)]],          \
                   device const bfloat *dynamic_taps [[buffer(18)]],                                           \
                   device const Residual *residual [[buffer(19)]], device Out *output [[buffer(20)]],          \
                   device const SplashLaneRows *layout [[buffer(21)]],                                         \
                   device const SplashTreeRow *tree [[buffer(22)]], device bfloat *tape [[buffer(23)]],        \
                   device bfloat *capture0 [[buffer(24)]], device bfloat *capture1 [[buffer(25)]],             \
                   device bfloat *capture2 [[buffer(26)]], device bfloat *capture3 [[buffer(27)]],             \
                   constant ShortConvolutionParams &p [[buffer(28)]],                                          \
                   uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {    \
    const uint s = group.z;                                                                                    \
    const ShortConvolutionCaptured captured{{capture0, capture1, capture2, capture3}};                        \
    short_convolution_row(input + p.input_column, p.state_column,                                              \
                          SPLASH_LANE_BINDING(s, state_in0, state_in1, state_in2, state_in3, state_in4,        \
                                              state_in5, state_in6, state_in7),                                \
                          SPLASH_LANE_BINDING(s, state_out0, state_out1, state_out2, state_out3, state_out4,   \
                                              state_out5, state_out6, state_out7),                             \
                          taps, dynamic_taps, residual, output, layout, tree, tape, captured, p, group, tid);  \
  }
static_assert(SHORT_CONVOLUTION_CAPTURES == 4, "the kernels bind four captures");
SHORT_CONVOLUTION(short_convolution, bfloat, float)
SHORT_CONVOLUTION(short_convolution_sum, float, bfloat)
#undef SHORT_CONVOLUTION

// A verify step's two convolutions of one shape in one dispatch, plain
// (ops::ShortConvolution::addConvolutionPair): grid (row, channel group,
// lane), the first convolution's channel groups, then the second's, each a
// threadgroup of short_convolution's over its own input (from column
// input_column of input0's rows, pair_input_column of input1's), taps,
// dynamic taps and output and its own state and tape columns (state_column,
// pair_state_column), so each writes the bits of its own dispatch. It writes
// no state: the commit writes the lanes' from the tape.
kernel void short_convolution_pair(device const bfloat *input0 [[buffer(0)]], device const bfloat *input1 [[buffer(1)]],
                                   device const bfloat *state_in0 [[buffer(2)]],
                                   device const bfloat *state_in1 [[buffer(3)]],
                                   device const bfloat *state_in2 [[buffer(4)]],
                                   device const bfloat *state_in3 [[buffer(5)]],
                                   device const bfloat *state_in4 [[buffer(6)]],
                                   device const bfloat *state_in5 [[buffer(7)]],
                                   device const bfloat *state_in6 [[buffer(8)]],
                                   device const bfloat *state_in7 [[buffer(9)]], device const float *taps0 [[buffer(10)]],
                                   device const float *taps1 [[buffer(11)]],
                                   device const bfloat *dynamic_taps0 [[buffer(12)]],
                                   device const bfloat *dynamic_taps1 [[buffer(13)]],
                                   device bfloat *output0 [[buffer(14)]], device bfloat *output1 [[buffer(15)]],
                                   device const SplashLaneRows *layout [[buffer(16)]],
                                   device const SplashTreeRow *tree [[buffer(17)]], device bfloat *tape [[buffer(18)]],
                                   constant ShortConvolutionParams &p [[buffer(19)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  const uint groups = (p.channels + kConvolutionThreads - 1) / kConvolutionThreads;
  const bool second = group.y >= groups;
  device const float *taps = second ? taps1 : taps0;
  device bfloat *output = second ? output1 : output0;
  // No state rows run (the grid's rows are the lanes'), so the state and
  // capture buffers bind the output, which they never write, and the residual
  // (plain) the taps, which it never reads.
  const ShortConvolutionCaptured captured{{output, output, output, output}};
  short_convolution_row(second ? input1 + p.pair_input_column : input0 + p.input_column,
                        second ? p.pair_state_column : p.state_column,
                        SPLASH_LANE_BINDING(group.z, state_in0, state_in1, state_in2, state_in3, state_in4, state_in5,
                                            state_in6, state_in7),
                        output, taps, second ? dynamic_taps1 : dynamic_taps0, taps, output, layout, tree, tape,
                        captured, p, uint3(group.x, second ? group.y - groups : group.y, group.z), tid);
}

// The state after each lane's accepted path of its verify block, a row mask
// the GPU wrote (acceptance's retained rows, paths[l]), of every column of
// the lanes' state matrices (p.channels of them, rows of p.state_stride):
// its rows on the path, which the step's convolutions copied into the tape,
// then the state before (state_in l); threadgroup (j, y, lane) writes state
// row j's columns from y x kConvolutionThreads on (state_out l). A fold
// (p.captures) takes the path's first p.capture_rows[0] rows alone, the
// state the commit writes where the lane retained exactly those, and a path
// of fewer leaves its state_out as it was. A lane whose rows run past the
// step's writes nothing.
kernel void short_convolution_commit(device const bfloat *tape [[buffer(0)]],
                                     device const bfloat *state_in0 [[buffer(1)]],
                                     device const bfloat *state_in1 [[buffer(2)]],
                                     device const bfloat *state_in2 [[buffer(3)]],
                                     device const bfloat *state_in3 [[buffer(4)]],
                                     device const bfloat *state_in4 [[buffer(5)]],
                                     device const bfloat *state_in5 [[buffer(6)]],
                                     device const bfloat *state_in6 [[buffer(7)]],
                                     device const bfloat *state_in7 [[buffer(8)]],
                                     device bfloat *state_out0 [[buffer(9)]], device bfloat *state_out1 [[buffer(10)]],
                                     device bfloat *state_out2 [[buffer(11)]], device bfloat *state_out3 [[buffer(12)]],
                                     device bfloat *state_out4 [[buffer(13)]], device bfloat *state_out5 [[buffer(14)]],
                                     device bfloat *state_out6 [[buffer(15)]], device bfloat *state_out7 [[buffer(16)]],
                                     device const uint *paths [[buffer(17)]],
                                     device const SplashLaneRows *layout [[buffer(18)]],
                                     constant ShortConvolutionParams &p [[buffer(19)]],
                                     uint3 group [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  const uint lane = group.z, c = group.y * kConvolutionThreads + tid, j = group.x;
  if (lane >= min(p.sequences, SPLASH_MAXIMUM_BATCH_WIDTH) || c >= p.channels || j >= uint(kStateRows)) return;
  const SplashLaneRows rows = layout[lane];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  const uint path = paths[lane], retained = popcount(path);
  const uint taken = p.captures ? p.capture_rows[0] : retained;
  if (taken > retained) return;
  const int source = int(taken) - kStateRows + int(j);
  device const bfloat *state_in = SPLASH_LANE_BINDING(lane, state_in0, state_in1, state_in2, state_in3, state_in4,
                                                      state_in5, state_in6, state_in7);
  device bfloat *state_out = SPLASH_LANE_BINDING(lane, state_out0, state_out1, state_out2, state_out3, state_out4,
                                                 state_out5, state_out6, state_out7);
  state_out[ulong(j) * p.state_stride + c] =
      source < 0 ? state_in[ulong(kStateRows + source) * p.state_stride + c]
                 : tape[ulong(rows.first + splash_path_row(path, uint(source))) * p.state_stride + c];
}
