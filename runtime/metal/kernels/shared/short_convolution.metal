#include "metal/abi/KernelABI.h"
#include "metal/abi/ShortConvolution.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/verify_tree.h"

#include <metal_stdlib>

using namespace metal;

// Inkling 2's short convolution (metal/abi/ShortConvolution.h): causal and
// depthwise, SHORT_CONVOLUTION_TAPS taps, in fp32 over inputs of type T (fp32
// or bf16 projection rows) and histories of type S (the state's elements).
// The taps are the GGUF tensor as stored, a channel's SHORT_CONVOLUTION_TAPS
// fp32 taps together (ggml's [taps, channels]), its last the current row's.

constant constexpr uint kTaps = SHORT_CONVOLUTION_TAPS;
constant constexpr int kHistory = int(SHORT_CONVOLUTION_HISTORY);
constant constexpr uint kThreads = SHORT_CONVOLUTION_THREADS;

// The convolution over each of a command's sequences' rows (a prompt chunk's
// sequences, or a verify step's lanes), threadgroup (row, channel group,
// sequence) of sequence s's rows its entry of the command's layout: a
// thread's row and channel. A row's input `lag` rows back (lag 1 to 3) is
// read as a history keeps it, in S: from its sequence's rows, before its
// first from its history (history_in s); a verify block's rows (p.tree, a
// row's place in `tree`) count back along their paths, and write their own
// input to the tape at their packed row, which the commit reads. Its
// current input, the residual and the sum are fp32. With write_history, rows
// past the sequence's write its history after its rows (history_out s). A
// prompt binds another buffer for the tree and the tape, which it does not
// touch, and a static convolution another for the dynamic taps. A row past
// its sequence's rows and history, or a sequence whose rows run past the
// command's, writes nothing.
template <typename T, typename S>
inline void short_convolution(device const T *input, device const T *lambda, device const float *taps,
                              device float *output, device const S *history_in, device S *history_out,
                              device const SplashLaneRows *layout, device const SplashTreeRow *tree, device S *tape,
                              constant ShortConvolutionParams &p, uint3 group, uint tid) {
  const uint c = group.y * kThreads + tid, row = group.x, s = group.z;
  if (c >= p.channels) return;
  const SplashLaneRows rows = layout[s];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  // Input `t` of the sequence's rows, before its first the history's.
  const auto past = [&](int t) -> float {
    return t >= 0 ? float(S(input[ulong(rows.first + uint(t)) * p.input_stride + c]))
                  : float(history_in[ulong(kHistory + t) * p.history_stride + c]);
  };
  if (row < rows.rows) {
    // The row's path and depth: a prompt's rows are a chain.
    const uint packed = rows.first + row;
    const uint path = p.tree ? tree[packed].ancestors : 0u;
    const int depth = p.tree ? int(tree[packed].depth) : int(row);
    const float x = float(input[ulong(packed) * p.input_stride + c]);
    device const T *dynamic =
        lambda + ulong(packed) * p.lambda_stride + (p.head_channels ? c / p.head_channels : 0) * kTaps;
    float sum = 0.0f;
    for (uint j = 0; j < kTaps; ++j) {
      const uint lag = kTaps - 1 - j;
      const float weight = taps[ulong(c) * kTaps + j] + (p.head_channels ? float(dynamic[lag]) : 0.0f);
      const int back = depth - int(lag);
      const float value = lag == 0 ? x : past(back >= 0 && p.tree ? int(splash_path_row(path, uint(back))) : back);
      sum += weight * value;
    }
    const float result = (x + sum) * p.scale;
    device float *out = output + ulong(packed) * p.output_stride + c;
    *out = p.accumulate ? *out + result : result;
    if (p.tree) tape[ulong(packed) * p.tape_stride + c] = S(x);
  } else if (p.write_history && row - rows.rows < uint(kHistory)) {
    // Rows past the sequence's write its history: row rows + j holds input
    // rows - 3 + j.
    const int j = int(row - rows.rows);
    history_out[ulong(j) * p.history_stride + c] = S(past(int(rows.rows) - kHistory + j));
  }
}

#define SHORT_CONVOLUTION_KERNEL(Name, T, S)                                                                           \
  kernel void Name(device const T *input [[buffer(0)]], device const T *lambda [[buffer(1)]],                          \
                   device const float *taps [[buffer(2)]], device float *output [[buffer(3)]],                         \
                   device const S *history_in0 [[buffer(4)]], device const S *history_in1 [[buffer(5)]],               \
                   device const S *history_in2 [[buffer(6)]], device const S *history_in3 [[buffer(7)]],               \
                   device const S *history_in4 [[buffer(8)]], device const S *history_in5 [[buffer(9)]],               \
                   device const S *history_in6 [[buffer(10)]], device const S *history_in7 [[buffer(11)]],             \
                   device S *history_out0 [[buffer(12)]], device S *history_out1 [[buffer(13)]],                       \
                   device S *history_out2 [[buffer(14)]], device S *history_out3 [[buffer(15)]],                       \
                   device S *history_out4 [[buffer(16)]], device S *history_out5 [[buffer(17)]],                       \
                   device S *history_out6 [[buffer(18)]], device S *history_out7 [[buffer(19)]],                       \
                   device const SplashLaneRows *layout [[buffer(20)]],                                                 \
                   device const SplashTreeRow *tree [[buffer(21)]], device S *tape [[buffer(22)]],                     \
                   constant ShortConvolutionParams &p [[buffer(23)]], uint3 group [[threadgroup_position_in_grid]],    \
                   uint tid [[thread_index_in_threadgroup]]) {                                                         \
    const uint s = group.z;                                                                                            \
    if (s >= min(p.sequences, SPLASH_MAXIMUM_BATCH_WIDTH)) return;                                                     \
    short_convolution<T, S>(input, lambda, taps, output,                                                               \
                            SPLASH_LANE_BINDING(s, history_in0, history_in1, history_in2, history_in3, history_in4,    \
                                                history_in5, history_in6, history_in7),                                \
                            SPLASH_LANE_BINDING(s, history_out0, history_out1, history_out2, history_out3,             \
                                                history_out4, history_out5, history_out6, history_out7),               \
                            layout, tree, tape, p, group, tid);                                                        \
  }
SHORT_CONVOLUTION_KERNEL(short_convolution_f32_f32, float, float)
SHORT_CONVOLUTION_KERNEL(short_convolution_f32_bf16, float, bfloat)
SHORT_CONVOLUTION_KERNEL(short_convolution_bf16_f32, bfloat, float)
SHORT_CONVOLUTION_KERNEL(short_convolution_bf16_bf16, bfloat, bfloat)
#undef SHORT_CONVOLUTION_KERNEL

// The history after a prefill sequence's first p.rows rows (a state captured
// inside its command): its history before them, then its inputs as a
// history keeps them. Threadgroup (j, channel group) writes history row j.
template <typename T, typename S>
inline void short_convolution_history(device const T *input, device const S *history_in, device S *history_out,
                                      constant ShortConvolutionHistoryParams &p, uint2 group, uint tid) {
  const uint c = group.y * kThreads + tid, j = group.x;
  if (c >= p.channels || j >= uint(kHistory)) return;
  const int source = int(p.rows) - kHistory + int(j);
  history_out[ulong(j) * p.history_stride + c] = source < 0
                                                     ? history_in[ulong(kHistory + source) * p.history_stride + c]
                                                     : S(input[ulong(source) * p.input_stride + c]);
}

#define SHORT_CONVOLUTION_HISTORY_KERNEL(Name, T, S)                                                                   \
  kernel void Name(device const T *input [[buffer(0)]], device const S *history_in [[buffer(1)]],                      \
                   device S *history_out [[buffer(2)]], constant ShortConvolutionHistoryParams &p [[buffer(3)]],       \
                   uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {           \
    short_convolution_history<T, S>(input, history_in, history_out, p, group, tid);                                    \
  }
SHORT_CONVOLUTION_HISTORY_KERNEL(short_convolution_history_f32_f32, float, float)
SHORT_CONVOLUTION_HISTORY_KERNEL(short_convolution_history_f32_bf16, float, bfloat)
SHORT_CONVOLUTION_HISTORY_KERNEL(short_convolution_history_bf16_f32, bfloat, float)
SHORT_CONVOLUTION_HISTORY_KERNEL(short_convolution_history_bf16_bf16, bfloat, bfloat)
#undef SHORT_CONVOLUTION_HISTORY_KERNEL

// The history after each lane's accepted path of its verify block (a row
// mask the GPU wrote, paths[l], of the lane's rows its entry of the step's
// layout holds), for every unit (a layer's convolutions side by side) in one
// dispatch: threadgroup (channel group, history row j, lane x units + unit),
// from the lane's history before (history_in l) and its rows' tape to its
// history after (history_out l). A lane whose rows run past the step's
// writes nothing.
template <typename S>
inline void short_convolution_commit(device const S *tape, device const S *history_in, device S *history_out,
                                     device const uint *paths, device const SplashLaneRows *layout,
                                     constant ShortConvolutionCommitParams &p, uint3 group, uint tid) {
  const uint c = group.x * kThreads + tid, j = group.y, lane = group.z / p.units, unit = group.z % p.units;
  if (c >= p.width || j >= uint(kHistory)) return;
  const SplashLaneRows rows = layout[lane];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  const uint path = paths[lane];
  const int source = int(popcount(path)) - kHistory + int(j);
  const ulong state = ulong(unit) * p.unit_stride;
  history_out[state + ulong(j) * p.history_stride + c] =
      source < 0 ? history_in[state + ulong(kHistory + source) * p.history_stride + c]
                 : tape[ulong(unit) * p.tape_unit_stride +
                        ulong(rows.first + splash_path_row(path, uint(source))) * p.tape_stride + c];
}

#define SHORT_CONVOLUTION_COMMIT_KERNEL(Name, S)                                                                       \
  kernel void Name(device const S *tape [[buffer(0)]], device const S *history_in0 [[buffer(1)]],                      \
                   device const S *history_in1 [[buffer(2)]], device const S *history_in2 [[buffer(3)]],               \
                   device const S *history_in3 [[buffer(4)]], device const S *history_in4 [[buffer(5)]],               \
                   device const S *history_in5 [[buffer(6)]], device const S *history_in6 [[buffer(7)]],               \
                   device const S *history_in7 [[buffer(8)]], device S *history_out0 [[buffer(9)]],                    \
                   device S *history_out1 [[buffer(10)]], device S *history_out2 [[buffer(11)]],                       \
                   device S *history_out3 [[buffer(12)]], device S *history_out4 [[buffer(13)]],                       \
                   device S *history_out5 [[buffer(14)]], device S *history_out6 [[buffer(15)]],                       \
                   device S *history_out7 [[buffer(16)]], device const uint *paths [[buffer(17)]],                     \
                   device const SplashLaneRows *layout [[buffer(18)]],                                                 \
                   constant ShortConvolutionCommitParams &p [[buffer(19)]],                                            \
                   uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {           \
    const uint lane = group.z / p.units;                                                                               \
    if (lane >= min(p.lanes, SPLASH_MAXIMUM_BATCH_WIDTH)) return;                                                      \
    short_convolution_commit<S>(tape,                                                                                  \
                                SPLASH_LANE_BINDING(lane, history_in0, history_in1, history_in2, history_in3,          \
                                                    history_in4, history_in5, history_in6, history_in7),               \
                                SPLASH_LANE_BINDING(lane, history_out0, history_out1, history_out2, history_out3,      \
                                                    history_out4, history_out5, history_out6, history_out7),           \
                                paths, layout, p, group, tid);                                                         \
  }
SHORT_CONVOLUTION_COMMIT_KERNEL(short_convolution_commit_f32, float)
SHORT_CONVOLUTION_COMMIT_KERNEL(short_convolution_commit_bf16, bfloat)
#undef SHORT_CONVOLUTION_COMMIT_KERNEL
