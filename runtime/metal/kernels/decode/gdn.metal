#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

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
// and the gate runs one simdgroup per row without barriers: over the 27B's 48
// layers with DRAM-cold states that takes 11-28% less time at one to four
// lanes than the same phases through device memory (40-core M3 Max, 16-core
// M5 Pro), with the same bits.
//
// Which of a step's rows a lane retains is known only after the step, so the
// state a step writes is the one before its rows: it leaves the rows on the
// lane's tape, and the lane's next step folds the retained ones into the
// state before its own rows. A step reads and writes the state once.
constant uint kDecodeSimdgroups = 8;

// The threadgroup operands of one value head: the rows' prepared q/k, the
// head's v, the gates and the recurrent output rows, and the k, v and gates
// of the pending rows, which the head folds into its state first.
template <uint HeadDim> struct GdnDecodeShared {
  bfloat queries[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat keys[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat values[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat rows[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  float decay[SPLASH_TARGET_VERIFY_ROWS];
  bfloat beta[SPLASH_TARGET_VERIFY_ROWS];
  bfloat pending_keys[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  bfloat pending_values[SPLASH_TARGET_VERIFY_ROWS * HeadDim];
  float pending_decay[SPLASH_TARGET_VERIFY_ROWS];
  bfloat pending_beta[SPLASH_TARGET_VERIFY_ROWS];
};

// One layer's tape of one lane's step (metal/abi/GDN.h): the rows'
// convolution inputs, their k rows (each key head's) then v rows, decay and
// beta.
struct GdnTape {
  device bfloat *inputs;
  device bfloat *kv;
  device float *decay;
  device bfloat *beta;
};

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline GdnTape gdn_tape(device uchar *tape, ulong offset) {
  constexpr uint KvWidth = (KeyHeads + ValueHeads) * HeadDim;
  device uchar *layer = tape + offset;
  return {reinterpret_cast<device bfloat *>(layer),
          reinterpret_cast<device bfloat *>(layer + gdn_tape_kv_offset(ConvDim)),
          reinterpret_cast<device float *>(layer + gdn_tape_decay_offset(ConvDim, KvWidth)),
          reinterpret_cast<device bfloat *>(
              layer + gdn_tape_beta_offset(ConvDim, KvWidth, ValueHeads))};
}

// Eight verify rows' conv+SiLU, q/k RMS norms and gates for one value head.
// One simdgroup per row holds channels 32g + lane (g = 0..3). RMS reduction
// sums each 32-channel group, then adds the four partials in channel order.
// The convolution reads the inputs before the rows from `history`: the
// carried rows of the state the lane's pending rows leave, which also go to
// conv_state_out. q/k/v and the gates go to threadgroup memory for the scan,
// beside the pending rows' k, v and gates from their tape; the rows'
// convolution inputs, k, v and gates go to this step's tape, from which the
// lane's next step folds in the rows the lane retains.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth>
inline void gdn_decode_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    GdnConvHistory history, device bfloat *conv_state_out,
    GdnTape pending_tape, uint pending, GdnTape step_tape,
    device const float *a_scale, device const bfloat *dt_bias,
    threadgroup GdnDecodeShared<HeadDim> &shared, uint value_head, uint lane,
    uint simd_group) {
  constexpr uint Tokens = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint KvWidth = KeyWidth + ValueWidth;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(Tokens == kDecodeSimdgroups && HeadDim == 128,
                "one simdgroup per verify row, four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  // The key head's q/k rows and conv carry are shared by HeadsPerKey value
  // heads; the first of them writes the shared copies.
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const bool carrier = simd_group < SPLASH_GDN_CONVOLUTION_TAPS - 1;
  const uint token = simd_group;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
  // The head's k and v columns of a tape row.
  const uint k_column = key_head * HeadDim + lane;
  const uint v_column = KeyWidth + value_head * HeadDim + lane;

  if (token < pending) {
    for (uint g = 0; g < Groups; ++g) {
      shared.pending_keys[token * HeadDim + 32 * g + lane] =
          pending_tape.kv[token * KvWidth + k_column + 32 * g];
      shared.pending_values[token * HeadDim + 32 * g + lane] =
          pending_tape.kv[token * KvWidth + v_column + 32 * g];
    }
    if (lane == 0) {
      shared.pending_decay[token] =
          pending_tape.decay[token * ValueHeads + value_head];
      shared.pending_beta[token] =
          pending_tape.beta[token * ValueHeads + value_head];
    }
  }

  float q[Groups], k[Groups];
  bfloat v[Groups], carry_v[Groups], carry_q[Groups], carry_k[Groups];
  for (uint g = 0; g < Groups; ++g) {
    q[g] = float(gdn_conv_silu(packed, history, conv_weights, PackedWidth,
                               token, q_channel + 32 * g));
    k[g] = float(gdn_conv_silu(packed, history, conv_weights, PackedWidth,
                               token, k_channel + 32 * g));
    v[g] = gdn_conv_silu(packed, history, conv_weights, PackedWidth, token,
                         v_channel + 32 * g);
    if (carrier) {
      carry_v[g] = history.at(simd_group, v_channel + 32 * g);
      carry_q[g] = shared_writer ? history.at(simd_group, q_channel + 32 * g)
                                 : bfloat(0.0f);
      carry_k[g] = shared_writer ? history.at(simd_group, k_channel + 32 * g)
                                 : bfloat(0.0f);
    }
  }
  GdnGates gates{};
  if (lane == 0)
    gates = gdn_gates(packed + token * PackedWidth, dt_bias, a_scale, BOffset,
                      AOffset, value_head);
  float q_sum = 0.0f, k_sum = 0.0f;
  for (uint g = 0; g < Groups; ++g) {
    q_sum += simd_sum(q[g] * q[g]);
    k_sum += simd_sum(k[g] * k[g]);
  }
  const float q_scale = rsqrt(q_sum / HeadDim + kRmsEpsilon);
  const float k_scale = rsqrt(k_sum / HeadDim + kRmsEpsilon);
  device const bfloat *inputs = packed + token * PackedWidth;
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
    const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
    shared.queries[token * HeadDim + dim] = query;
    shared.keys[token * HeadDim + dim] = key;
    shared.values[token * HeadDim + dim] = v[g];
    step_tape.inputs[token * ConvDim + v_channel + 32 * g] =
        inputs[v_channel + 32 * g];
    step_tape.kv[token * KvWidth + v_column + 32 * g] = v[g];
    if (shared_writer) {
      step_tape.inputs[token * ConvDim + q_channel + 32 * g] =
          inputs[q_channel + 32 * g];
      step_tape.inputs[token * ConvDim + k_channel + 32 * g] =
          inputs[k_channel + 32 * g];
      step_tape.kv[token * KvWidth + k_column + 32 * g] = key;
    }
  }
  if (lane == 0) {
    const uint gate_index = token * ValueHeads + value_head;
    step_tape.beta[gate_index] = gates.beta;
    step_tape.decay[gate_index] = gates.decay;
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
// RowsInFlight rows advance together to overlap their reductions and
// arithmetic. The pending rows advance the incoming state to the committed
// one, which goes to state_out; the step's own rows then advance it only for
// their outputs, which go to threadgroup memory for the gate. Each row
// preserves the decay, memory, delta, update, output operation order. The
// next rows' state is loaded before these rows' state is stored.
template <uint HeadDim, uint RowsInFlight>
inline void gdn_decode_scan(device const float *state_in,
                            device float *state_out,
                            threadgroup GdnDecodeShared<HeadDim> &shared,
                            uint pending, uint value_head, uint lane,
                            uint simd_group) {
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
    for (uint token = 0; token < pending; ++token) {
      const float d = shared.pending_decay[token];
      const float b = float(shared.pending_beta[token]);
      threadgroup const bfloat *key =
          shared.pending_keys + token * HeadDim + lane * 4;
      float memory[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        memory[r] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] *= d;
          memory[r] += state[r][i] * float(key[i]);
        }
        memory[r] = simd_sum(memory[r]);
      }
      for (uint r = 0; r < RowsInFlight; ++r) {
        const float delta =
            (float(shared.pending_values[token * HeadDim + value_dim[r]]) -
             memory[r]) *
            b;
        for (uint i = 0; i < 4; ++i)
          state[r][i] += float(key[i]) * delta;
      }
    }
    float upcoming[RowsInFlight][4] = {};
    if (batch + RowsInFlight < Batches) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          upcoming[r][i] = state_in[base(batch + RowsInFlight, r) + i];
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[base(batch, r) + i] = state[r][i];
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
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state[r][i] = upcoming[r][i];
  }
}

// Gated RMSNorm of one row's recurrent output for this value head, one
// simdgroup per row with dimensions 32g + lane: the lane assignment and the
// channel-order sum of the four simd_sum partials reproduce the prefill gate,
// gdn_gate_phase, whose four simdgroups add their partials in that order.
// Reassociation is off and the operations are written in the order the
// compiler emits for gdn_gate_phase, so the rows are bitwise the same: with
// fast-math reassociation this shape rounded about one output in 10^5
// differently. Leaves the gated row in shared.rows for the out-projection
// table.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, class W>
inline void gdn_decode_gate(threadgroup GdnDecodeShared<HeadDim> &shared,
                            device const bfloat *packed,
                            device const W *norm_weight, device bfloat *hidden,
                            bool tiled, uint value_head, uint lane,
                            uint token) {
#pragma clang fp reassociate(off)
  constexpr uint Groups = HeadDim / 32;
  constexpr uint ZOffset = ConvDim;
  const ulong row = ulong(token) * ValueHeads;
  const ulong hidden_base =
      (row + gdn_output_head<KeyHeads, ValueHeads>(value_head, tiled)) *
      HeadDim;
  bfloat gate[Groups];
  W weight[Groups];
  float value[Groups];
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    gate[g] = packed[token * PackedWidth + ZOffset + value_head * HeadDim + dim];
    weight[g] = norm_weight[dim];
    value[g] = float(shared.rows[token * HeadDim + dim]);
  }
  float total = 0.0f;
  for (uint g = 0; g < Groups; ++g)
    total += simd_sum(value[g] * value[g]);
  const float inverse = rsqrt(total / HeadDim + kRmsEpsilon);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat normalized = bfloat((value[g] * inverse) * float(weight[g]));
    const float z = float(gate[g]);
    const bfloat gated = bfloat((float(normalized) * z) /
                                (1.0f + fast::exp2(-1.44269504089f * z)));
    hidden[hidden_base + dim] = gated;
    shared.rows[token * HeadDim + dim] = gated;
  }
}

// Grid {value heads, lanes}.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, uint RowsInFlight, class Table, class W>
inline void gdn_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device uchar *tape, device const float *a_scale,
    device const bfloat *dt_bias,
    device const W *gdn_norm_weight, device bfloat *gdn_hidden,
    constant GDNDecodeBatchParams &params,
    uint2 group, uint lane, uint simd_group,
    threadgroup GdnDecodeShared<HeadDim> &shared,
    device bfloat *table, device float *sums) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  uint batch = group.y;
  device const uchar *current = SPLASH_LANE_BINDING(batch, current0, current1, current2, current3);
  device uchar *next = SPLASH_LANE_BINDING(batch, next0, next1, next2, next3);
  packed += ulong(batch) * Rows * PackedWidth;
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

  const uint pending = params.pending_rows[batch];
  const GdnTape pending_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(tape, params.pending_tape[batch]);
  const GdnTape step_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(tape, params.step_tape[batch]);
  const GdnConvHistory history{conv_state_in, pending_tape.inputs, ConvDim,
                               ConvDim, pending};

  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      packed, conv_weights, history, conv_state_out, pending_tape, pending,
      step_tape, a_scale, dt_bias, shared, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gdn_decode_scan<HeadDim, RowsInFlight>(state_in, state_out, shared, pending,
                                         group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool tiled = params.tiled_heads != 0;
  gdn_decode_gate<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      shared, packed, gdn_norm_weight, lane_hidden, tiled, group.x, lane,
      simd_group);
  if (table) {
    // Each group owns this head for all eight rows; each simdgroup writes
    // the table of the row it just gated.
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint head = gdn_output_head<KeyHeads, ValueHeads>(group.x, tiled);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = head * HeadDim + g * 64 + 2 * lane;
      const uint local = simd_group * HeadDim + g * 64 + 2 * lane;
      Table::write(table + ulong(batch) * ValueWidth * Rows,
                   sums + ulong(batch) * Table::sums_per_tile(ValueWidth), ValueWidth,
                   column / 64, simd_group, lane, shared.rows[local], shared.rows[local + 1]);
    }
  }
}

// W: the norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define GDN_DECODE_BUFFERS(W) \
    device const bfloat *packed [[buffer(0)]], \
    device const bfloat *conv_weights [[buffer(1)]], \
    device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]], \
    device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]], \
    device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]], \
    device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]], \
    device uchar *tape [[buffer(10)]], device const float *a_scale [[buffer(11)]], \
    device const bfloat *dt_bias [[buffer(12)]], device const W *gdn_norm_weight [[buffer(13)]], \
    device bfloat *gdn_hidden [[buffer(14)]]
#define GDN_DECODE_THREADS \
    uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, table, sums, Layout) \
    threadgroup GdnDecodeShared<HeadDim> shared; \
    gdn_decode_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, 2, Layout>( \
        packed, conv_weights, current0, current1, current2, current3, next0, \
        next1, next2, next3, tape, a_scale, dt_bias, \
        gdn_norm_weight, gdn_hidden, params, group, lane, simd_group, shared, \
        table, sums);
// Entries without a table pass null pointers, which skip the write; their Layout only completes the template.
#define GDN_DECODE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, W) \
  kernel void Name(GDN_DECODE_BUFFERS(W), \
      constant GDNDecodeBatchParams &params [[buffer(15)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, nullptr, nullptr, gguf_sg::Table16) \
  }
// The out-projection's table (Layout: gguf_sg::Table16).
#define GDN_DECODE_TABLE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Layout, W) \
  kernel void Name(GDN_DECODE_BUFFERS(W), \
      device bfloat *table [[buffer(15)]], device float *sums [[buffer(16)]], \
      constant GDNDecodeBatchParams &params [[buffer(17)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, table, sums, Layout) \
  }

// Two rows overlap reductions and arithmetic without the register cost of four.
GDN_DECODE_ENTRY(verify_gdn_fused, 16, 48, 128, 10240, 16640, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32, 16, 32, 128, 8192, 12544, bfloat)
// Table16 feeds the register tile, after norms that are F32 (a GGUF's) or bf16 (an MLX target's).
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16, 16, 48, 128, 10240, 16640, gguf_sg::Table16, bfloat)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh32, 16, 32, 128, 8192, 12544, gguf_sg::Table16, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_f32, 16, 48, 128, 10240, 16640, float)
GDN_DECODE_ENTRY(verify_gdn_fused_vh32_f32, 16, 32, 128, 8192, 12544, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_f32, 16, 48, 128, 10240, 16640, gguf_sg::Table16, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh32_f32, 16, 32, 128, 8192, 12544, gguf_sg::Table16, float)
#undef GDN_DECODE_ENTRY
#undef GDN_DECODE_TABLE_ENTRY
#undef GDN_DECODE_BODY
#undef GDN_DECODE_THREADS
#undef GDN_DECODE_BUFFERS
