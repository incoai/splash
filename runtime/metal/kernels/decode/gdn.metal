#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/gguf_sgmatrix.h"

// Decode threadgroups are 256 threads: one simdgroup per verify row in the
// prologue and the gate, and in the scan the head's 128 state rows strided
// over the eight simdgroups (each advances two of its sixteen rows at a time).
//
// A threadgroup is one value head of one lane, so a layer is only 48 of them
// per lane and each one's chain of memory round trips sets the layer's time
// (below the state traffic's bandwidth bound at one lane on both families).
// The phases therefore hand each other their operands in threadgroup memory
// instead of device memory, issue their device loads before their stores
// (the scan loads the next rows' state before storing the current rows'),
// and the gate runs one simdgroup per row without barriers. Over the 27B's 48
// layers with DRAM-cold states this took one to four lanes from 2.23 / 3.00 /
// 4.36 / 4.86 ms to 1.81 / 2.50 / 3.52 / 4.30 ms on a 40-core M3 Max and from
// 1.93 / 4.02 / 5.43 / 7.03 ms to 1.62 / 2.90 / 4.31 / 5.55 ms on a 16-core
// M5 Pro, bitwise unchanged.
constant uint kDecodeSimdgroups = 8;

inline void grid_completion(device atomic_uint &arrived,
                            device atomic_uint &generation, uint group_count,
                            uint thread_index) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    uint ticket = atomic_fetch_add_explicit(&arrived, 1, memory_order_relaxed);
    if (ticket + 1 == group_count) {
      atomic_store_explicit(&arrived, 0, memory_order_relaxed);
      atomic_fetch_add_explicit(&generation, 1, memory_order_relaxed);
    }
  }
}

// The threadgroup operands of one value head: the rows' prepared q/k, the
// head's v, the gates and the recurrent output rows.
template <uint HeadDim> struct GdnDecodeShared {
  bfloat queries[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat keys[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat values[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat rows[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  float decay[SPLASH_TARGET_VERIFY_ROWS];
  bfloat beta[SPLASH_TARGET_VERIFY_ROWS];
};

// Eight verify rows' conv+SiLU, q/k RMS norms and gates for one value head.
// One simdgroup per row holds channels 32g + lane (g = 0..3). RMS reduction
// sums each 32-channel group, then adds the four partials in channel order.
// q/k/v and the gates go to threadgroup memory for the scan, and to device
// memory for the commit.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_decode_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *mixed_qkv, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    uint packed_width, threadgroup GdnDecodeShared<HeadDim> &shared,
    uint value_head, uint lane, uint simd_group) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(Tokens == kDecodeSimdgroups && HeadDim == 128,
                "one simdgroup per verify row, four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  // The key head's q/k rows and conv carry are shared by HeadsPerKey value
  // heads; the first of them writes the shared copies.
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const bool carrier = simd_group < 3;
  const uint token = simd_group;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;

  float q[Groups], k[Groups];
  bfloat v[Groups], carry_v[Groups], carry_q[Groups], carry_k[Groups];
  for (uint g = 0; g < Groups; ++g) {
    q[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               packed_width, ConvDim, token,
                               q_channel + 32 * g));
    k[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               packed_width, ConvDim, token,
                               k_channel + 32 * g));
    v[g] = gdn_conv_silu(packed, conv_state_in, conv_weights, packed_width,
                         ConvDim, token, v_channel + 32 * g);
    if (carrier) {
      carry_v[g] = gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim,
                                  Tokens, simd_group, v_channel + 32 * g);
      carry_q[g] = shared_writer
          ? gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                           simd_group, q_channel + 32 * g)
          : bfloat(0.0f);
      carry_k[g] = shared_writer
          ? gdn_conv_carry(packed, conv_state_in, packed_width, ConvDim, Tokens,
                           simd_group, k_channel + 32 * g)
          : bfloat(0.0f);
    }
  }
  GdnGates gates{};
  if (lane == 0)
    gates = gdn_gates(packed + token * packed_width, dt_bias, a_scale, BOffset,
                      AOffset, value_head);
  float q_sum = 0.0f, k_sum = 0.0f;
  for (uint g = 0; g < Groups; ++g) {
    q_sum += simd_sum(q[g] * q[g]);
    k_sum += simd_sum(k[g] * k[g]);
  }
  const float q_scale = rsqrt(q_sum / HeadDim + 1e-6f);
  const float k_scale = rsqrt(k_sum / HeadDim + 1e-6f);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
    const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
    shared.queries[token * HeadDim + dim] = query;
    shared.keys[token * HeadDim + dim] = key;
    shared.values[token * HeadDim + dim] = v[g];
    if (shared_writer) {
      mixed_qkv[token * ConvDim + q_channel + 32 * g] = query;
      mixed_qkv[token * ConvDim + k_channel + 32 * g] = key;
    }
    mixed_qkv[token * ConvDim + v_channel + 32 * g] = v[g];
  }
  if (lane == 0) {
    const uint gate_index = token * ValueHeads + value_head;
    beta[gate_index] = gates.beta;
    decay[gate_index] = gates.decay;
    shared.beta[token] = gates.beta;
    shared.decay[token] = gates.decay;
  }
  if (carrier) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] = carry_v[g];
      if (shared_writer) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] = carry_q[g];
        conv_state_out[row * ConvDim + k_channel + 32 * g] = carry_k[g];
      }
    }
  }
}

// Delta-rule recurrence over 128 state rows, four fp32 columns per lane.
// RowsInFlight rows advance together to overlap their reductions and arithmetic.
// Each row preserves the decay, memory, delta, update, output operation order.
// The next rows' state is loaded before these rows' state is stored, and the
// output rows go to threadgroup memory for the gate.
template <uint HeadDim, uint RowsInFlight>
inline void gdn_decode_scan(device const float *state_in,
                            device float *state_out,
                            threadgroup GdnDecodeShared<HeadDim> &shared,
                            uint value_head, uint lane, uint simd_group) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  const auto base = [&](uint batch, uint r) {
    const uint value_dim = (batch + r) * kDecodeSimdgroups + simd_group;
    return (ulong(value_head) * HeadDim + value_dim) * HeadDim + lane * 4;
  };
  float state[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      state[r][i] = state_in[base(0, r) + i];
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint value_dim[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r)
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
    for (uint token = 0; token < Tokens; ++token) {
      const float d = shared.decay[token];
      const float b = float(shared.beta[token]);
      threadgroup const bfloat *key = shared.keys + token * HeadDim + lane * 4;
      threadgroup const bfloat *query =
          shared.queries + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(shared.values[token * HeadDim + value_dim[r]]) -
             memory[r]) *
            b;
        result[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] += float(key[i]) * delta;
          result[r] += state[r][i] * float(query[i]);
        }
        result[r] = simd_sum(result[r]);
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r)
          shared.rows[token * HeadDim + value_dim[r]] = bfloat(result[r]);
      }
    }
    float upcoming[RowsInFlight][4] = {};
    if (batch + RowsInFlight < Batches) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          upcoming[r][i] = state_in[base(batch + RowsInFlight, r) + i];
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i) {
        state_out[base(batch, r) + i] = state[r][i];
        state[r][i] = upcoming[r][i];
      }
  }
}

// Gated RMSNorm of one row's recurrent output for this value head, one
// simdgroup per row with dimensions 32g + lane: the lane assignment and the
// channel-order sum of the four simd_sum partials reproduce gdn_gate_phase
// (whose zero partials of simdgroups 4..7 add nothing to a non-negative sum).
// Reassociation is off and the operations are written in the order the
// compiler emits for gdn_gate_phase, so the rows are bitwise the same: with
// fast-math reassociation this shape rounded about one output in 10^5
// differently. Leaves the gated row in shared.rows for the out-projection
// table.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_decode_gate(threadgroup GdnDecodeShared<HeadDim> &shared,
                            device const bfloat *packed,
                            device const bfloat *norm_weight,
                            device bfloat *recurrent, device bfloat *hidden,
                            uint packed_width, bool tiled, uint value_head,
                            uint lane, uint token) {
#pragma clang fp reassociate(off)
  constexpr uint Groups = HeadDim / 32;
  constexpr uint ZOffset = ConvDim;
  const ulong row = ulong(token) * ValueHeads;
  const ulong hidden_base =
      (row + gdn_output_head<KeyHeads, ValueHeads>(value_head, tiled)) *
      HeadDim;
  bfloat gate[Groups], weight[Groups];
  float value[Groups];
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    gate[g] = packed[token * packed_width + ZOffset + value_head * HeadDim + dim];
    weight[g] = norm_weight[dim];
    value[g] = float(shared.rows[token * HeadDim + dim]);
  }
  float total = 0.0f;
  for (uint g = 0; g < Groups; ++g)
    total += simd_sum(value[g] * value[g]);
  const float inverse = rsqrt(total / HeadDim + 1e-6f);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    recurrent[(row + value_head) * HeadDim + dim] = bfloat(value[g]);
    const bfloat normalized = bfloat((value[g] * inverse) * float(weight[g]));
    const float z = float(gate[g]);
    const bfloat gated = bfloat((float(normalized) * z) /
                                (1.0f + fast::exp2(-1.44269504089f * z)));
    hidden[hidden_base + dim] = gated;
    shared.rows[token * HeadDim + dim] = gated;
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void
gdn_commit_phase(device const bfloat *packed, device const bfloat *mixed_qkv,
                 device const float *decay, device const bfloat *beta,
                 device const bfloat *conv_state_in,
                 device bfloat *conv_state_out, device const float *state_in,
                 device float *state_out, uint retained, uint groups,
                 uint packed_width, uint group, uint thread_index, uint lane,
                 uint simd_group) {
  constexpr uint KeyDim = HeadDim, ValueDim = HeadDim;
  constexpr uint ValueBatches = ValueDim / 8;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;

  uint count = retained;
  if (count == SPLASH_TARGET_VERIFY_ROWS)
    return;
  for (uint element = group * 256 + thread_index; element < 3 * ConvDim;
       element += groups * 256) {
    uint row = element / ConvDim;
    uint channel = element % ConvDim;
    conv_state_out[element] = gdn_conv_carry(
        packed, conv_state_in, packed_width, ConvDim, count, row, channel);
  }

  for (uint task = group; task < ValueHeads * ValueBatches; task += groups) {
    uint value_head = task / ValueBatches;
    uint value_dim = (task % ValueBatches) * 8 + simd_group;
    uint key_head = value_head / HeadsPerKey;
    ulong state_base = (ulong(value_head) * ValueDim + value_dim) * KeyDim;
    float local_state[4];
    for (uint i = 0; i < 4; ++i) {
      local_state[i] = state_in[state_base + lane * 4 + i];
    }
    for (uint token = 0; token < count; ++token) {
      ulong key_base = ulong(token) * ConvDim + key_head * KeyDim;
      float memory = 0.0f;
      float d = decay[token * ValueHeads + value_head];
      for (uint i = 0; i < 4; ++i) {
        uint dim = lane * 4 + i;
        local_state[i] *= d;
        memory +=
            local_state[i] * float(mixed_qkv[key_base + dim + KeyWidth]);
      }
      memory = simd_sum(memory);
      ulong value_index =
          ulong(token) * ConvDim + value_head * ValueDim + value_dim +
          2 * KeyWidth;
      float delta = (float(mixed_qkv[value_index]) - memory) *
                    float(beta[token * ValueHeads + value_head]);
      for (uint i = 0; i < 4; ++i) {
        uint dim = lane * 4 + i;
        local_state[i] += float(mixed_qkv[key_base + dim + KeyWidth]) * delta;
      }
    }
    for (uint i = 0; i < 4; ++i) {
      state_out[state_base + lane * 4 + i] = local_state[i];
    }
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_commit_prefix_batch_phase(
    device const bfloat *packed, device const bfloat *mixed_qkv,
    device const float *decay, device const bfloat *beta,
    device const uchar *current_0, device const uchar *current_1,
    device const uchar *current_2, device const uchar *current_3,
    device uchar *next_0, device uchar *next_1, device uchar *next_2,
    device uchar *next_3, device const uint *retained,
    constant GDNBatchCommitParams &params, uint3 group, uint thread_index,
    uint simd_lane, uint simd_group) {
  if (group.z >= params.lanes)
    return;
  uint batch = group.z;
  uint layer = group.y;
  device const uchar *current = batch == 0   ? current_0
                                : batch == 1 ? current_1
                                : batch == 2 ? current_2
                                             : current_3;
  device uchar *next = batch == 0   ? next_0
                       : batch == 1 ? next_1
                       : batch == 2 ? next_2
                                    : next_3;
  packed += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.packed_stride;
  mixed_qkv += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.mixed_stride;
  decay += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.decay_stride;
  beta += (ulong(layer) * SPLASH_MAXIMUM_BATCH_WIDTH + batch) * params.beta_stride;
  device const bfloat *conv_state_in = reinterpret_cast<device const bfloat *>(
      current + ulong(layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  gdn_commit_phase<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      packed, mixed_qkv, decay, beta, conv_state_in, conv_state_out, state_in,
      state_out, retained[batch], params.groups, params.packed_width, group.x,
      thread_index, simd_lane, simd_group);
}

#define GDN_COMMIT_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim)         \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *mixed_qkv [[buffer(1)]],                           \
      device const float *decay [[buffer(2)]],                                \
      device const bfloat *beta [[buffer(3)]],                                \
      device const uchar *current_0 [[buffer(4)]],                            \
      device const uchar *current_1 [[buffer(5)]],                            \
      device const uchar *current_2 [[buffer(6)]],                            \
      device const uchar *current_3 [[buffer(7)]],                            \
      device uchar *next_0 [[buffer(8)]], device uchar *next_1 [[buffer(9)]], \
      device uchar *next_2 [[buffer(10)]],                                    \
      device uchar *next_3 [[buffer(11)]],                                    \
      device const uint *retained [[buffer(12)]],                             \
      constant GDNBatchCommitParams &params [[buffer(13)]],                   \
      uint3 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]],                           \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    gdn_commit_prefix_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim>(    \
        packed, mixed_qkv, decay, beta, current_0, current_1, current_2,       \
        current_3, next_0, next_1, next_2, next_3, retained, params, group,    \
        thread_index, simd_lane, simd_group);                                 \
  }

GDN_COMMIT_ENTRY(verify_gdn_commit, 16, 48, 128, 10240)
GDN_COMMIT_ENTRY(verify_gdn_commit_vh32, 16, 32, 128, 8192)
#undef GDN_COMMIT_ENTRY

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight, class Table>
inline void gdn_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device bfloat *mixed, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    device bfloat *recurrent, device const bfloat *gdn_norm_weight,
    device bfloat *gdn_hidden, device atomic_uint *arrived,
    device atomic_uint *generation, constant GDNDecodeBatchParams &params,
    uint2 group, uint thread_index, uint lane, uint simd_group,
    threadgroup GdnDecodeShared<HeadDim> &shared,
    device bfloat *q4_table = nullptr, device float *q4_sums = nullptr) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  uint batch = group.y;
  if (batch >= params.lanes || group.x >= ValueHeads)
    return;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  packed += ulong(batch) * Rows * params.packed_width;
  mixed += ulong(batch) * Rows * ConvDim;
  decay += ulong(batch) * Rows * ValueHeads;
  beta += ulong(batch) * Rows * ValueHeads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);

  device bfloat *lane_recurrent =
      recurrent + ulong(batch) * Rows * ValueWidth;
  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      packed, conv_weights, conv_state_in, conv_state_out, mixed, a_scale,
      dt_bias, decay, beta, params.packed_width, shared, group.x, lane,
      simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gdn_decode_scan<HeadDim, RowsInFlight>(state_in, state_out, shared, group.x,
                                         lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool tiled = params.tiled_heads != 0;
  gdn_decode_gate<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      shared, packed, gdn_norm_weight, lane_recurrent, lane_hidden,
      params.packed_width, tiled, group.x, lane, simd_group);
  if (q4_table) {
    // Each group owns this head for all eight rows; each simdgroup writes
    // the table of the row it just gated.
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint head = gdn_output_head<KeyHeads, ValueHeads>(group.x, tiled);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = head * HeadDim + g * 64 + 2 * lane;
      const uint local = simd_group * HeadDim + g * 64 + 2 * lane;
      Table::write(q4_table + ulong(batch) * ValueWidth * Rows,
                   q4_sums + ulong(batch) * Table::sums_per_tile(ValueWidth), ValueWidth,
                   column / 64, simd_group, lane, shared.rows[local], shared.rows[local + 1]);
    }
  }
  grid_completion(arrived[batch], generation[batch], ValueHeads,
                  thread_index);
}

#define GDN_DECODE_BUFFERS \
    device const bfloat *packed [[buffer(0)]], \
    device const bfloat *conv_weights [[buffer(1)]], \
    device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]], \
    device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]], \
    device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]], \
    device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]], \
    device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]], \
    device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]], \
    device bfloat *beta [[buffer(14)]], device bfloat *recurrent [[buffer(15)]], \
    device const bfloat *gdn_norm_weight [[buffer(16)]], device bfloat *gdn_hidden [[buffer(17)]], \
    device atomic_uint *arrived [[buffer(18)]], device atomic_uint *generation [[buffer(19)]]
#define GDN_DECODE_THREADS \
    uint2 group [[threadgroup_position_in_grid]], \
    uint thread_index [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, Table, Sums, Layout) \
    threadgroup GdnDecodeShared<HeadDim> shared; \
    gdn_decode_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, 2, Layout>( \
        packed, conv_weights, current0, current1, current2, current3, next0, \
        next1, next2, next3, mixed, a_scale, dt_bias, decay, beta, recurrent, \
        gdn_norm_weight, gdn_hidden, arrived, generation, params, group, \
        thread_index, lane, simd_group, shared, Table, Sums);
#define GDN_DECODE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim) \
  kernel void Name(GDN_DECODE_BUFFERS, \
      constant GDNDecodeBatchParams &params [[buffer(20)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, nullptr, nullptr, q4sg::Table64) \
  }
// The out-projection's table (Layout: q4sg::Table64 affine, q16sg::Table16 GGUF).
#define GDN_DECODE_TABLE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, Layout) \
  kernel void Name(GDN_DECODE_BUFFERS, \
      device bfloat *q4_table [[buffer(20)]], device float *q4_sums [[buffer(21)]], \
      constant GDNDecodeBatchParams &params [[buffer(22)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, q4_table, q4_sums, Layout) \
  }

// Two rows overlap reductions and arithmetic without the register cost of four.
GDN_DECODE_ENTRY(verify_gdn_fused, 16, 48, 128, 10240)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32, 16, 32, 128, 8192)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_q4, 16, 48, 128, 10240, q4sg::Table64)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_q4_vh32, 16, 32, 128, 8192, q4sg::Table64)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_q16, 16, 48, 128, 10240, q16sg::Table16)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_q16_vh32, 16, 32, 128, 8192, q16sg::Table16)
#undef GDN_DECODE_ENTRY
#undef GDN_DECODE_TABLE_ENTRY
#undef GDN_DECODE_BODY
#undef GDN_DECODE_THREADS
#undef GDN_DECODE_BUFFERS
