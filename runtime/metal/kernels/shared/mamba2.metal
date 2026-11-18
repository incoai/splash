#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/lane_bindings.h"

// Mamba-2 (Nemotron-H's mixer, ops/Mamba2.hpp), after llama.cpp's
// build_mamba2_layer and the CPU ssm_scan: a packed row of the in projection
// is z | x B C | dt; x B C go through the 4-tap causal convolution, its bias
// and SiLU; each head h of P state rows over the N columns of its group's
// B and C advances
//   s = s * exp(softplus(dt + dt_bias) A) + B (x softplus(dt + dt_bias)),
//   y = s . C + D x,
// and its output row is y silu(z). The grouped RMS norm of the rows
// (mamba2_group_norm) follows, since a norm group spans many heads.
//
// Threadgroups are 256 threads, one value head of one lane: one simdgroup per
// verify row in the prologue, and in the scan the head's P state rows strided
// over the eight simdgroups, each lane holding four of the N columns.
//
// As GDN's decode does (decode/gdn.metal), a step writes the state before its
// own rows and leaves the rows on the lane's tape; the lane's next step folds
// the rows it retained into the state first. x, B, C, decay and dt stay fp32
// throughout, on the tape too, so a folded row advances the state exactly as
// the step that produced it did.
constant uint kMamba2Simdgroups = 8;

template <uint P, uint N> struct Mamba2DecodeShared {
  float c[SPLASH_TARGET_VERIFY_ROWS * N];
  float b[SPLASH_TARGET_VERIFY_ROWS * N];
  float x[SPLASH_TARGET_VERIFY_ROWS * P];
  float decay[SPLASH_TARGET_VERIFY_ROWS];
  float dt[SPLASH_TARGET_VERIFY_ROWS];
};

// One layer's tape of one lane's step (metal/abi/Mamba2.h).
struct Mamba2Tape {
  device bfloat *inputs;
  device float *bx;
  device float *decay;
  device float *dt;
};

template <uint Groups, uint Heads, uint P, uint N, uint ConvDim>
inline Mamba2Tape mamba2_tape(device uchar *tape, ulong offset) {
  constexpr uint BxWidth = Groups * N + Heads * P;
  device uchar *layer = tape + offset;
  return {reinterpret_cast<device bfloat *>(layer),
          reinterpret_cast<device float *>(layer + mamba2_tape_bx_offset(ConvDim)),
          reinterpret_cast<device float *>(layer + mamba2_tape_decay_offset(ConvDim, BxWidth)),
          reinterpret_cast<device float *>(layer + mamba2_tape_dt_offset(ConvDim, BxWidth, Heads))};
}

// The causal convolution of conv channel `channel` at one token, then its bias
// and SiLU, in ggml's order (ssm_conv's tap sum, the bias, silu). Conv
// channels are the packed row's from `packed_offset` on; the history (the
// carried rows and the pending rows' tape inputs) is in conv channels.
inline float mamba2_conv_silu(device const bfloat *packed, uint packed_width, uint packed_offset,
                              GdnConvHistory history, device const float *weights,
                              device const float *bias, uint token, uint channel) {
  constexpr uint Taps = SPLASH_GDN_CONVOLUTION_TAPS, Carried = Taps - 1;
  float value = 0.0f;
  for (uint tap = 0; tap < Taps; ++tap) {
    const uint position = token + tap;
    const bfloat input = position < Carried
                             ? history.at(position, channel)
                             : packed[(position - Carried) * packed_width + packed_offset + channel];
    value += float(input) * weights[channel * Taps + tap];
  }
  value += bias[channel];
  return value / (1.0f + precise::exp(-value));
}

// Each verify row's convolution, its decay and softplus(dt) for this head,
// one simdgroup a row (channels 32 g + lane), into threadgroup memory beside
// the pending rows' from their tape; the rows' convolution inputs, B, x,
// decay and dt go to this step's tape, and the carried convolution rows the
// pending rows leave to conv_state_out. A group's B and C, and their tape and
// carry, are shared by its heads; its first head writes them.
template <uint Groups, uint Heads, uint P, uint N, uint ConvDim, uint PackedWidth, uint XbcOffset,
          uint DtOffset>
inline void mamba2_decode_prologue(device const bfloat *packed, device const float *conv_weights,
                                   device const float *conv_bias, GdnConvHistory history,
                                   device bfloat *conv_state_out, uint rows, Mamba2Tape step_tape,
                                   device const float *a, device const float *dt_bias,
                                   threadgroup Mamba2DecodeShared<P, N> &shared, uint head, uint lane,
                                   uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint XWidth = Heads * P, BxWidth = Groups * N + XWidth;
  constexpr uint XGroups = P / 32, NGroups = N / 32;
  static_assert(P % 32 == 0 && N == 128, "whole simdgroups of x, four state columns per lane");
  static_assert(ConvDim == XWidth + 2 * Groups * N, "x | B | C");
  const uint group = head / HeadsPerGroup;
  const bool shared_writer = head % HeadsPerGroup == 0;
  // The carried convolution rows the pending rows leave, simdgroups 0-2 a
  // row each.
  if (simd_group < SPLASH_GDN_CONVOLUTION_TAPS - 1) {
    for (uint g = 0; g < XGroups; ++g) {
      const uint channel = head * P + 32 * g + lane;
      conv_state_out[simd_group * ConvDim + channel] = history.at(simd_group, channel);
    }
    if (shared_writer)
      for (uint g = 0; g < NGroups; ++g) {
        const uint b_channel = XWidth + group * N + 32 * g + lane, c_channel = b_channel + Groups * N;
        conv_state_out[simd_group * ConvDim + b_channel] = history.at(simd_group, b_channel);
        conv_state_out[simd_group * ConvDim + c_channel] = history.at(simd_group, c_channel);
      }
  }
  // The step's rows, a simdgroup's every eighth.
  for (uint token = simd_group; token < rows; token += kMamba2Simdgroups) {
  device const bfloat *row = packed + token * PackedWidth;
  for (uint g = 0; g < XGroups; ++g) {
    const uint channel = head * P + 32 * g + lane;
    const float x = mamba2_conv_silu(packed, PackedWidth, XbcOffset, history, conv_weights, conv_bias, token,
                                     channel);
    shared.x[token * P + 32 * g + lane] = x;
    step_tape.bx[token * BxWidth + Groups * N + head * P + 32 * g + lane] = x;
    step_tape.inputs[token * ConvDim + channel] = row[XbcOffset + channel];
  }
  for (uint g = 0; g < NGroups; ++g) {
    const uint b_channel = XWidth + group * N + 32 * g + lane;
    const uint c_channel = b_channel + Groups * N;
    const float b = mamba2_conv_silu(packed, PackedWidth, XbcOffset, history, conv_weights, conv_bias, token,
                                     b_channel);
    const float c = mamba2_conv_silu(packed, PackedWidth, XbcOffset, history, conv_weights, conv_bias, token,
                                     c_channel);
    shared.b[token * N + 32 * g + lane] = b;
    shared.c[token * N + 32 * g + lane] = c;
    if (shared_writer) {
      step_tape.bx[token * BxWidth + group * N + 32 * g + lane] = b;
      step_tape.inputs[token * ConvDim + b_channel] = row[XbcOffset + b_channel];
      step_tape.inputs[token * ConvDim + c_channel] = row[XbcOffset + c_channel];
    }
  }
  if (lane == 0) {
    // ggml_compute_softplus_f32: the input itself above 20.
    const float raw = float(row[DtOffset + head]) + dt_bias[head];
    const float dt = raw > 20.0f ? raw : precise::log(1.0f + precise::exp(raw));
    const float decay = precise::exp(dt * a[head]);
    shared.dt[token] = dt;
    shared.decay[token] = decay;
    step_tape.dt[token * Heads + head] = dt;
    step_tape.decay[token * Heads + head] = decay;
  }
  }
}

// The scan over the head's P state rows, four fp32 columns a lane,
// RowsInFlight rows together. The pending rows advance the incoming state to
// the committed one, which goes to state_out; the step's rows then advance it
// for their outputs, y = s . C + D x gated by silu(z), which go to `y` in
// fp32. The next rows' state is loaded before these rows' is stored.
template <uint Groups, uint Heads, uint P, uint N, uint PackedWidth, uint ZOffset, uint RowsInFlight>
inline void mamba2_decode_scan(device const float *state_in, device float *state_out,
                               threadgroup Mamba2DecodeShared<P, N> &shared, Mamba2Tape pending_tape,
                               uint pending, uint rows, device const bfloat *packed, device const float *d,
                               device float *y, uint head, uint lane, uint simd_group) {
  constexpr uint BxWidth = Groups * N + Heads * P, HeadsPerGroup = Heads / Groups;
  device const float *pending_b = pending_tape.bx + (head / HeadsPerGroup) * N + lane * 4;
  device const float *pending_x = pending_tape.bx + Groups * N + head * P;
  constexpr uint Batches = P / kMamba2Simdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  const auto base = [&](uint batch, uint r) {
    const uint state_row = (batch + r) * kMamba2Simdgroups + simd_group;
    return (ulong(head) * P + state_row) * N + lane * 4;
  };
  const float skip = d[head];
  float state[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i) state[r][i] = state_in[base(0, r) + i];
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint state_row[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) state_row[r] = (batch + r) * kMamba2Simdgroups + simd_group;
    // The pending rows, from their tape, where the step that left them put
    // their fp32 x, B, decay and dt.
    for (uint token = 0; token < pending; ++token) {
      const float decay = pending_tape.decay[token * Heads + head], dt = pending_tape.dt[token * Heads + head];
      device const float *b = pending_b + token * BxWidth;
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float x_dt = pending_x[token * BxWidth + state_row[r]] * dt;
        for (uint i = 0; i < 4; ++i) state[r][i] = state[r][i] * decay + b[i] * x_dt;
      }
    }
    float upcoming[RowsInFlight][4] = {};
    if (batch + RowsInFlight < Batches) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i) upcoming[r][i] = state_in[base(batch + RowsInFlight, r) + i];
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i) state_out[base(batch, r) + i] = state[r][i];
    for (uint token = 0; token < rows; ++token) {
      const float decay = shared.decay[token], dt = shared.dt[token];
      threadgroup const float *b = shared.b + token * N + lane * 4;
      threadgroup const float *c = shared.c + token * N + lane * 4;
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float x_dt = shared.x[token * P + state_row[r]] * dt;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] = state[r][i] * decay + b[i] * x_dt;
          result[r] += state[r][i] * c[i];
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r) {
          const uint column = head * P + state_row[r];
          const float z = float(packed[token * PackedWidth + ZOffset + column]);
          const float out = result[r] + shared.x[token * P + state_row[r]] * skip;
          y[token * Heads * P + column] = out * (z / (1.0f + precise::exp(-z)));
        }
      }
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i) state[r][i] = upcoming[r][i];
  }
}

// One lane's step of one layer from its state (the convolution's carried rows
// and the recurrent state) to the state before its own rows; the pending rows
// come from the tape at `pending_tape`, the step's rows go to `step_tape`.
template <uint Groups, uint Heads, uint P, uint N, uint ConvDim, uint PackedWidth, uint ZOffset, uint XbcOffset,
          uint DtOffset>
inline void mamba2_decode_step(device const bfloat *packed, device const float *conv_weights,
                               device const float *conv_bias, device const bfloat *conv_state_in,
                               device bfloat *conv_state_out, device const float *state_in, device float *state_out,
                               device uchar *tape, ulong pending_offset, ulong step_offset, uint pending,
                               uint rows, device const float *a, device const float *dt_bias, device const float *d,
                               device float *y, uint head, uint lane, uint simd_group,
                               threadgroup Mamba2DecodeShared<P, N> &shared) {
  const Mamba2Tape pending_tape = mamba2_tape<Groups, Heads, P, N, ConvDim>(tape, pending_offset);
  const Mamba2Tape step_tape = mamba2_tape<Groups, Heads, P, N, ConvDim>(tape, step_offset);
  const GdnConvHistory history = gdn_conv_history(conv_state_in, pending_tape.inputs, ConvDim, ConvDim, pending);

  mamba2_decode_prologue<Groups, Heads, P, N, ConvDim, PackedWidth, XbcOffset, DtOffset>(
      packed, conv_weights, conv_bias, history, conv_state_out, rows, step_tape, a, dt_bias, shared, head, lane,
      simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mamba2_decode_scan<Groups, Heads, P, N, PackedWidth, ZOffset, 2>(state_in, state_out, shared, pending_tape,
                                                                   pending, rows, packed, d, y, head, lane,
                                                                   simd_group);
}

// Grid {heads, lanes}.
template <uint Groups, uint Heads, uint P, uint N, uint ConvDim, uint PackedWidth, uint ZOffset, uint XbcOffset,
          uint DtOffset>
inline void mamba2_decode_batch_phase(
    device const bfloat *packed, device const float *conv_weights, device const float *conv_bias,
    device const uchar *current0, device const uchar *current1, device const uchar *current2,
    device const uchar *current3, device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device uchar *tape, device const float *a, device const float *dt_bias,
    device const float *d, device float *y, device const SplashLaneRows *lane_rows,
    constant GDNDecodeBatchParams &params, uint2 group, uint lane, uint simd_group,
    threadgroup Mamba2DecodeShared<P, N> &shared) {
  const uint batch = group.y;
  const SplashLaneRows rows = lane_rows[batch];
  device const uchar *current = SPLASH_LANE_BINDING(batch, current0, current1, current2, current3);
  device uchar *next = SPLASH_LANE_BINDING(batch, next0, next1, next2, next3);
  mamba2_decode_step<Groups, Heads, P, N, ConvDim, PackedWidth, ZOffset, XbcOffset, DtOffset>(
      packed + ulong(rows.first) * PackedWidth, conv_weights, conv_bias,
      reinterpret_cast<device const bfloat *>(current + ulong(params.layer) * params.conv_layer_bytes),
      reinterpret_cast<device bfloat *>(next + ulong(params.layer) * params.conv_layer_bytes),
      reinterpret_cast<device const float *>(current + params.convolution_state_bytes +
                                             ulong(params.layer) * params.recurrent_layer_bytes),
      reinterpret_cast<device float *>(next + params.convolution_state_bytes +
                                       ulong(params.layer) * params.recurrent_layer_bytes),
      tape, params.pending_tape[batch], params.step_tape[batch], uint(popcount(params.pending_path[batch])),
      min(rows.rows, uint(SPLASH_TARGET_VERIFY_ROWS)), a, dt_bias, d, y + ulong(rows.first) * Heads * P, group.x,
      lane, simd_group, shared);
}

#define MAMBA2_DECODE_KERNEL(Name, Groups, Heads, P, N, ConvDim, PackedWidth, ZOffset, XbcOffset, DtOffset)   \
  kernel void Name(device const bfloat *packed [[buffer(0)]], device const float *conv_weights [[buffer(1)]],  \
                   device const float *conv_bias [[buffer(2)]], device const uchar *current0 [[buffer(3)]],    \
                   device const uchar *current1 [[buffer(4)]], device const uchar *current2 [[buffer(5)]],     \
                   device const uchar *current3 [[buffer(6)]], device uchar *next0 [[buffer(7)]],              \
                   device uchar *next1 [[buffer(8)]], device uchar *next2 [[buffer(9)]],                       \
                   device uchar *next3 [[buffer(10)]], device uchar *tape [[buffer(11)]],                      \
                   device const float *a [[buffer(12)]], device const float *dt_bias [[buffer(13)]],           \
                   device const float *d [[buffer(14)]], device float *y [[buffer(15)]],                       \
                   device const SplashLaneRows *lane_rows [[buffer(16)]],                                      \
                   constant GDNDecodeBatchParams &params [[buffer(17)]],                                       \
                   uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],      \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                       \
    threadgroup Mamba2DecodeShared<P, N> shared;                                                               \
    mamba2_decode_batch_phase<Groups, Heads, P, N, ConvDim, PackedWidth, ZOffset, XbcOffset, DtOffset>(        \
        packed, conv_weights, conv_bias, current0, current1, current2, current3, next0, next1, next2, next3,   \
        tape, a, dt_bias, d, y, lane_rows, params, group, lane, simd_group, shared);                                      \
  }

// The compiled geometries (ops/Mamba2.cpp, kCompiledMamba2), named for their
// groups and heads: 8 groups of 128-column B and C and 128 heads of 64 rows
// (Nemotron 3.5 Bolt's), packed rows z (8192) | x B C (10240) | dt (128).
MAMBA2_DECODE_KERNEL(verify_mamba2_g8_h128, 8, 128, 64, 128, 10240, 18560, 0, 8192, 18432)
#undef MAMBA2_DECODE_KERNEL

// A prefill's step of up to eight rows of one sequence's layer, its states in
// their own buffers (a cell's layer views): lane 0 of the parameters names
// the tapes and pending rows; their layer and strides are unused. Grid {heads}.
#define MAMBA2_PREFILL_KERNEL(Name, Groups, Heads, P, N, ConvDim, PackedWidth, ZOffset, XbcOffset, DtOffset)  \
  kernel void Name(device const bfloat *packed [[buffer(0)]], device const float *conv_weights [[buffer(1)]],  \
                   device const float *conv_bias [[buffer(2)]], device const bfloat *conv_state_in [[buffer(3)]], \
                   device const float *state_in [[buffer(4)]], device bfloat *conv_state_out [[buffer(5)]],    \
                   device float *state_out [[buffer(6)]], device uchar *tape [[buffer(7)]],                    \
                   device const float *a [[buffer(8)]], device const float *dt_bias [[buffer(9)]],             \
                   device const float *d [[buffer(10)]], device float *y [[buffer(11)]],                       \
                   constant GDNDecodeBatchParams &params [[buffer(12)]],                                       \
                   uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],      \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                       \
    threadgroup Mamba2DecodeShared<P, N> shared;                                                               \
    mamba2_decode_step<Groups, Heads, P, N, ConvDim, PackedWidth, ZOffset, XbcOffset, DtOffset>(               \
        packed, conv_weights, conv_bias, conv_state_in, conv_state_out, state_in, state_out, tape,             \
        params.pending_tape[0], params.step_tape[0], uint(popcount(params.pending_path[0])),                   \
        SPLASH_TARGET_VERIFY_ROWS, a, dt_bias, d, y, group.x, lane, simd_group, shared);                                                                                   \
  }
MAMBA2_PREFILL_KERNEL(prefill_mamba2_g8_h128, 8, 128, 64, 128, 10240, 18560, 0, 8192, 18432)
#undef MAMBA2_PREFILL_KERNEL

// The grouped RMS norm of the gated rows: one threadgroup of 256 threads per
// (group, row), each thread four of the group's Width values; ggml's
// rms_norm (the mean of squares, 1 / sqrt(mean + epsilon)) times the
// weights, rounded to bf16 for the out projection. Grid {groups, rows}.
template <uint Width, uint Groups>
inline void mamba2_group_norm_phase(device const float *y, device const float *weights, device bfloat *hidden,
                                    constant Mamba2NormParams &params, uint2 group, uint thread_index,
                                    uint lane, uint simd_group, threadgroup float *partials) {
  constexpr uint Threads = 256, PerThread = Width / Threads;
  static_assert(Width % Threads == 0, "whole values per thread");
  const ulong base = ulong(group.y) * Width * Groups + ulong(group.x) * Width;
  float values[PerThread];
  float sum = 0.0f;
  for (uint k = 0; k < PerThread; ++k) {
    values[k] = y[base + thread_index + Threads * k];
    sum += values[k] * values[k];
  }
  sum = simd_sum(sum);
  if (lane == 0) partials[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < Threads / 32; ++s) total += partials[s];
  const float inverse = 1.0f / precise::sqrt(total / Width + params.epsilon);
  for (uint k = 0; k < PerThread; ++k) {
    const uint index = thread_index + Threads * k;
    hidden[base + index] = bfloat(values[k] * inverse * weights[group.x * Width + index]);
  }
}

#define MAMBA2_GROUP_NORM_KERNEL(Name, Width, Groups)                                                      \
  kernel void Name(device const float *y [[buffer(0)]], device const float *weights [[buffer(1)]],             \
                   device bfloat *hidden [[buffer(2)]], constant Mamba2NormParams &params [[buffer(3)]],       \
                   uint2 group [[threadgroup_position_in_grid]],                                               \
                   uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                       \
    threadgroup float partials[8];                                                                             \
    mamba2_group_norm_phase<Width, Groups>(y, weights, hidden, params, group, thread_index, lane, simd_group,   \
                                           partials);                                                          \
  }
// The groups of 1024 of the geometry of 8 groups and 128 heads of 64.
MAMBA2_GROUP_NORM_KERNEL(mamba2_group_norm_g8_h128, 1024, 8)
#undef MAMBA2_GROUP_NORM_KERNEL
