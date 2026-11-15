#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

// Decode threadgroups are 256 threads: one simdgroup per row of the lane's
// block (its SplashLaneRows: its first row and up to Rows rows), two rows
// each in a 16-row kernel, in the prologue and the gate, and in the
// recurrence the head's 128 state rows strided over the eight simdgroups
// (each advances two of its sixteen rows at a time). A kernel holds up to
// Rows rows of each lane's block and of its pending path: eight, the
// threadgroup memory of all but a tree's operands (gdn_decode_transform) a
// layer's eight-row chains took before 16-row blocks, or sixteen, which took
// 13-19% longer over eight-row chains of the 27B's and 35B's GDN layers
// (DRAM-cold, an M6) for holding twice the memory.
//
// A threadgroup is one value head of one lane, so a layer is only 48 of them
// per lane and each one's chain of memory round trips sets the layer's time
// (below the state traffic's bandwidth bound at one lane on both families).
// The phases therefore hand each other their operands in threadgroup memory
// instead of device memory, issue their device loads before their stores
// (the scan loads the next rows' state before storing the current rows'),
// and the gate runs one simdgroup per row without barriers: over the 27B's 48
// layers with DRAM-cold states that takes 11-28% less time than the same
// phases through device memory, with the same bits.
//
// The block's rows are a tree (metal/abi/VerifyTree.h): a row's token follows
// the tokens of its path from the anchor, which its convolution reads and
// which its recurrent output advances the state through. A chain's rows run
// the sequential scan; a tree's rows each read the state before the block
// along their own path (gdn_decode_transform) without a state per row.
//
// Which of a step's rows a lane accepts is known only after the step, so the
// state a step writes is the one before its rows: it leaves every row on the
// lane's tape, and the lane's next step folds the accepted path's rows into
// the state before its own rows. A step reads and writes the state once. Its
// rows past the lane's verify rows are never accepted or read: the
// recurrence skips them and their recurrent output is zero. A fold alone
// (gdn_fold_phase) folds the first rows of a step's accepted path as that
// next step would, for the state of a page the path reaches.
constant uint kDecodeSimdgroups = 8;

// The k, v and gates of one value head's rows a lane folds into its state,
// up to Rows of them: a step's pending rows, or the rows a fold alone folds.
template <uint HeadDim, uint Rows> struct GdnPendingShared {
  bfloat keys[Rows * HeadDim];
  bfloat values[Rows * HeadDim];
  float decay[Rows];
  bfloat beta[Rows];
};

// The threadgroup operands of one value head: the rows' prepared q/k, the
// head's v, the gates, the recurrent output rows and the rows' paths; a
// tree's coefficients of its rows (gdn_decode_transform), whose transformed
// queries then take the q/k rows' place; and the pending rows, which the
// head folds into its state first. The v rows are held dimension-major, each
// dimension's rows in turn (values[dim * Rows + row]): a tree's lanes, a lane
// a row, read one dimension of every row from consecutive addresses, and a
// chain one row's at a time. Measured on an M5 Pro over the 27B's and 35B's
// GDN layers, the 8- and the 16-row kernels: 8-row chains ran 0.4-2.5%
// faster than the base, 16-row trees 0.4-1.5% faster than with rows of
// HeadDim values and up to 1.4% faster than with rows padded to HeadDim + 2
// (0.3% slower in one case of eight), 8-row trees within 1.3% of either.
template <uint HeadDim, uint Rows> struct GdnDecodeShared {
  static_assert(Rows % kDecodeSimdgroups == 0 && Rows <= SPLASH_TARGET_VERIFY_ROWS && Rows <= 32,
                "whole rows a simdgroup, and a lane of the simdgroup a row");
  // The rows each simdgroup holds.
  static constexpr constant uint Passes = Rows / kDecodeSimdgroups;
  bfloat queries[Rows * HeadDim];
  bfloat keys[Rows * HeadDim];
  bfloat values[HeadDim * Rows];
  bfloat rows[Rows * HeadDim];
  float decay[Rows];
  bfloat beta[Rows];
  uint paths[Rows];
  float coefficients[Rows * Rows];
  GdnPendingShared<HeadDim, Rows> pending;
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

// Tape row `row`, one value head's k, v and gates of it, to row `token` of
// threadgroup memory: one simdgroup per row, channels 32g + lane.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint Rows>
inline void gdn_load_pending(GdnTape tape, threadgroup GdnPendingShared<HeadDim, Rows> &pending, uint token,
                             uint row, uint value_head, uint lane) {
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint KvWidth = KeyWidth + ValueHeads * HeadDim;
  // The head's k and v columns of a tape row.
  const uint k_column = value_head / (ValueHeads / KeyHeads) * HeadDim + lane;
  const uint v_column = KeyWidth + value_head * HeadDim + lane;
  for (uint g = 0; g < HeadDim / 32; ++g) {
    pending.keys[token * HeadDim + 32 * g + lane] = tape.kv[row * KvWidth + k_column + 32 * g];
    pending.values[token * HeadDim + 32 * g + lane] = tape.kv[row * KvWidth + v_column + 32 * g];
  }
  if (lane == 0) {
    pending.decay[token] = tape.decay[row * ValueHeads + value_head];
    pending.beta[token] = tape.beta[row * ValueHeads + value_head];
  }
}

// The path of a block row from the anchor (metal/abi/VerifyTree.h): its
// ancestors, of the rows before it.
inline uint gdn_decode_path(device const SplashTreeRow *tree, uint row) {
  return tree[row].ancestors & ((2u << row) - 1u);
}

// The `rows` rows' conv+SiLU, q/k RMS norms and gates for one value head.
// One simdgroup per row holds channels 32g + lane (g = 0..3). RMS reduction
// sums each 32-channel group, then adds the four partials in channel order.
// A row's convolution reads the inputs of its path's tokens before it: those
// in the block from its packed rows, those before it from `history`: the
// carried rows of the state the lane's pending path leaves, which also go to
// conv_state_out. q/k/v and the gates go to threadgroup memory for the
// recurrence, beside the pending rows' k, v and gates from their tape; the
// rows' convolution inputs, k, v and gates go to this step's tape, from which
// the lane's next step folds in the path it accepts, and the rows' paths to
// threadgroup memory for the recurrence. Rows from `verified` on get zero
// recurrent output. A simdgroup's rows past the block's write none of
// theirs, but the pending rows and the state's carried rows.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, uint Rows>
inline void gdn_decode_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    GdnConvHistory history, uint pending_path, device bfloat *conv_state_out,
    GdnTape pending_tape, GdnTape step_tape, device const SplashTreeRow *tree,
    uint rows, uint verified, device const float *a_scale, device const bfloat *dt_bias,
    threadgroup GdnDecodeShared<HeadDim, Rows> &shared, uint value_head, uint lane,
    uint simd_group) {
  constexpr uint Taps = SPLASH_GDN_CONVOLUTION_TAPS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint KvWidth = KeyWidth + ValueWidth;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(HeadDim == 128, "four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  // The key head's q/k rows and conv carry are shared by HeadsPerKey value
  // heads; the first of them writes the shared copies.
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const bool carrier = simd_group < Taps - 1;
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
  // The head's k and v columns of a tape row.
  const uint k_column = key_head * HeadDim + lane;
  const uint v_column = KeyWidth + value_head * HeadDim + lane;

  // The carried rows are loaded before the rows' passes and go to
  // conv_state_out after them: Apple9 miscompiles a carry loaded and stored
  // within the first pass (gdn-decode's "convolution carry mismatch" on the
  // 16-row kernels, an M4 under macOS 26.6), and stored before the passes too
  // the 35B's 8-row chains took 3.3% longer at one lane and the 27B's 1.0% at
  // four (an M5 Pro).
  bfloat carry_v[Groups], carry_q[Groups], carry_k[Groups];
  if (carrier) {
    for (uint g = 0; g < Groups; ++g) {
      carry_v[g] = history.at(simd_group, v_channel + 32 * g);
      carry_q[g] = shared_writer ? history.at(simd_group, q_channel + 32 * g) : bfloat(0.0f);
      carry_k[g] = shared_writer ? history.at(simd_group, k_channel + 32 * g) : bfloat(0.0f);
    }
  }
  for (uint pass = 0; pass < GdnDecodeShared<HeadDim, Rows>::Passes; ++pass) {
    const uint token = simd_group + pass * kDecodeSimdgroups;
    // Pending row `token` is the pending path's row at that depth.
    if (token < uint(popcount(pending_path)))
      gdn_load_pending<KeyHeads, ValueHeads, HeadDim, Rows>(
          pending_tape, shared.pending, token, splash_path_row(pending_path, token), value_head, lane);

    // A simdgroup past the block's rows computes the block's last row again
    // and writes none of it: it reads no packed row of the next block's, or
    // past the last block's end. Every simdgroup computes a row because the
    // prologue that skipped those rows' work around its reductions computed
    // wrong tape rows and gates on Apple9 (an M4 under macOS 26.6, at eight
    // rows a block, which skip nothing), though not on Apple10.
    const uint computed = min(token, rows - 1);
    const bool writes = token < rows;
    // The block rows of the row's last Taps path tokens, its own last, and
    // its depth, at which its convolution taps the history.
    const uint path = gdn_decode_path(tree, computed);
    uint path_rows[Taps];
    for (uint tap = Taps, up = path; tap-- > 0;) {
      path_rows[tap] = up ? 31 - clz(up) : 0;
      up &= ~(1u << path_rows[tap]);
    }
    const GdnConvTaps taps = gdn_conv_taps(packed, PackedWidth, history, popcount(path) - 1, path_rows);
    float q[Groups], k[Groups];
    bfloat v[Groups];
    for (uint g = 0; g < Groups; ++g) {
      q[g] = float(gdn_conv_silu(taps, conv_weights, q_channel + 32 * g));
      k[g] = float(gdn_conv_silu(taps, conv_weights, k_channel + 32 * g));
      v[g] = gdn_conv_silu(taps, conv_weights, v_channel + 32 * g);
    }
    GdnGates gates{};
    if (lane == 0)
      gates = gdn_gates(packed + computed * PackedWidth, dt_bias, a_scale,
                        BOffset, AOffset, value_head);
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint g = 0; g < Groups; ++g) {
      q_sum += simd_sum(q[g] * q[g]);
      k_sum += simd_sum(k[g] * k[g]);
    }
    const float q_scale = rsqrt(q_sum / HeadDim + kRmsEpsilon);
    const float k_scale = rsqrt(k_sum / HeadDim + kRmsEpsilon);
    device const bfloat *inputs = packed + computed * PackedWidth;
    for (uint g = 0; g < Groups; ++g) {
      const uint dim = 32 * g + lane;
      const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
      const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
      if (writes) {
        shared.queries[token * HeadDim + dim] = query;
        shared.keys[token * HeadDim + dim] = key;
        shared.values[dim * Rows + token] = v[g];
        if (token >= verified)
          shared.rows[token * HeadDim + dim] = bfloat(0.0f);
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
    }
    if (lane == 0 && writes) {
      const uint gate_index = token * ValueHeads + value_head;
      step_tape.beta[gate_index] = gates.beta;
      step_tape.decay[gate_index] = gates.decay;
      shared.beta[token] = gates.beta;
      shared.decay[token] = gates.decay;
      shared.paths[token] = path;
    }
  }
  if (carrier) {
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[simd_group * ConvDim + v_channel + 32 * g] = carry_v[g];
      if (shared_writer) {
        conv_state_out[simd_group * ConvDim + q_channel + 32 * g] = carry_q[g];
        conv_state_out[simd_group * ConvDim + k_channel + 32 * g] = carry_k[g];
      }
    }
  }
}

// Folds the first `rows` rows of `pending` into RowsInFlight state rows of a
// value head (rows value_dim, four fp32 columns per lane), each in the decay,
// memory, delta, update operation order: the same arithmetic wherever a
// lane's rows are folded, a step's pending ones or a fold's alone.
template <uint HeadDim, uint Rows, uint RowsInFlight>
inline void gdn_fold_rows(thread float (&state)[RowsInFlight][4],
                          threadgroup const GdnPendingShared<HeadDim, Rows> &pending,
                          uint rows, thread const uint (&value_dim)[RowsInFlight],
                          uint lane) {
  for (uint token = 0; token < rows; ++token) {
    const float d = pending.decay[token];
    const float b = float(pending.beta[token]);
    threadgroup const bfloat *key =
        pending.keys + token * HeadDim + lane * 4;
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
          (float(pending.values[token * HeadDim + value_dim[r]]) -
           memory[r]) *
          b;
      for (uint i = 0; i < 4; ++i)
        state[r][i] += float(key[i]) * delta;
    }
  }
}

// Delta-rule recurrence over 128 state rows, four fp32 columns per lane.
// RowsInFlight rows advance together to overlap their reductions and
// arithmetic. The pending rows advance the incoming state to the committed
// one, which goes to state_out; the block's first `verified` rows then
// advance it only for their outputs, which go to threadgroup memory for the
// gate: in turn, a chain's (Tree false), or each tree row from the committed
// state along its path, through its transformed query and its path's
// coefficients (gdn_decode_transform). Each row of a chain preserves the
// decay, memory, delta, update, output operation order. The next rows' state
// is loaded before these rows' state is stored. State is the stored element:
// float, or bfloat, widened as it is loaded and rounded to nearest even as
// it is stored; the arithmetic is fp32 with either.
template <uint HeadDim, uint Rows, uint RowsInFlight, bool Tree, class State>
inline void gdn_decode_scan(device const State *state_in,
                            device State *state_out,
                            threadgroup GdnDecodeShared<HeadDim, Rows> &shared,
                            uint pending, uint verified, uint value_head,
                            uint lane, uint simd_group) {
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  const auto base = [&](uint batch, uint r) {
    const uint value_dim = (batch + r) * kDecodeSimdgroups + simd_group;
    return (ulong(value_head) * HeadDim + value_dim) * HeadDim + lane * 4;
  };
  threadgroup const float *transformed =
      reinterpret_cast<threadgroup const float *>(shared.queries);
  float state[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      state[r][i] = float(state_in[base(0, r) + i]);
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint value_dim[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r)
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
    gdn_fold_rows<HeadDim, Rows, RowsInFlight>(state, shared.pending, pending, value_dim, lane);
    float upcoming[RowsInFlight][4] = {};
    if (batch + RowsInFlight < Batches) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          upcoming[r][i] = float(state_in[base(batch + RowsInFlight, r) + i]);
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[base(batch, r) + i] = State(state[r][i]);
    // A tree lane's value of each of the batch's state rows, which the path
    // term of every row reads: the 16-row kernel reads it once a batch, which
    // the compiler leaves in the rows' loop beside the stores of their
    // outputs, and so runs 16-row trees 12-22% faster (an M5 Pro, the 27B's
    // and 35B's GDN layers, one and four lanes). The 8-row kernel reads it
    // once a row and only on the row's path. Read outside the path's
    // condition (once a batch or once a row), its trees ran 10-20% faster on
    // an M5 Max and 10-13% faster on an M5 Pro, but 13% slower on the M5 Pro
    // at one lane of the 27B, and its chains 0.5% slower.
    constexpr bool kValuesOnce = Rows == 16;
    float lane_values[RowsInFlight];
    if constexpr (Tree && kValuesOnce)
      for (uint r = 0; r < RowsInFlight; ++r)
        lane_values[r] = float(shared.values[value_dim[r] * Rows + lane % Rows]);
    for (uint token = 0; token < Rows; ++token) {
      if (token == verified)
        break;
      float result[RowsInFlight];
      if constexpr (Tree) {
        // The path's terms, a lane per row of the block, beside the
        // committed state's product with the transformed query.
        threadgroup const float *query = transformed + token * HeadDim + lane * 4;
        const uint row = lane % Rows;
        const bool on_path = lane < Rows && (shared.paths[token] >> row & 1);
        const float coefficient = shared.coefficients[token * Rows + row];
        for (uint r = 0; r < RowsInFlight; ++r) {
          if constexpr (kValuesOnce)
            result[r] = on_path ? coefficient * lane_values[r] : 0.0f;
          else
            result[r] = on_path ? coefficient * float(shared.values[value_dim[r] * Rows + row]) : 0.0f;
          for (uint i = 0; i < 4; ++i)
            result[r] += state[r][i] * query[i];
          result[r] = simd_sum(result[r]);
        }
      } else {
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
        for (uint r = 0; r < RowsInFlight; ++r) {
          const float delta =
              (float(shared.values[value_dim[r] * Rows + token]) -
               memory[r]) *
              b;
          result[r] = 0.0f;
          for (uint i = 0; i < 4; ++i) {
            state[r][i] += float(key[i]) * delta;
            result[r] += state[r][i] * float(query[i]);
          }
          result[r] = simd_sum(result[r]);
        }
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

// Whether the block's first `verified` rows are a chain, each row's parent
// the row before it, from their paths the prologue left. Every simdgroup
// reads the same paths, so every thread takes the same branch on it.
template <uint HeadDim, uint Rows>
inline bool gdn_decode_chain(threadgroup const GdnDecodeShared<HeadDim, Rows> &shared, uint verified, uint lane) {
  return simd_all(lane >= verified || shared.paths[lane] == (2u << lane) - 1u);
}

// A tree row's output from the state before the block's rows: with S_j the
// state after the j-th token of its path (S_-1 the committed state), each
// token's update S_j = d_j S_j-1 (I - b_j k_j k_j^T) + b_j v_j k_j^T gives
// S_j z = S_j-1 (d_j (z - c k_j)) + c v_j for c = b_j (k_j . z), so the
// row's output S_D q = S_-1 z_-1 + sum_j c_j v_j, where z_D = q and z_j-1 =
// d_j (z_j - c_j k_j), going up the path from the row to the anchor. One
// simdgroup per row transforms its query: its path, the coefficient of each
// of its rows (zero for the others) and its transformed query z_-1, which
// takes the q/k rows' place once every row read them, go to threadgroup
// memory for gdn_decode_scan.
template <uint HeadDim, uint Rows>
inline void gdn_decode_transform(threadgroup GdnDecodeShared<HeadDim, Rows> &shared, uint verified, uint lane,
                                 uint simd_group) {
  constexpr uint Passes = GdnDecodeShared<HeadDim, Rows>::Passes;
  float z[Passes][4], coefficient[Passes];
  for (uint pass = 0; pass < Passes; ++pass) {
    const uint token = simd_group + pass * kDecodeSimdgroups;
    coefficient[pass] = 0.0f;
    if (token >= verified)
      continue;
    const uint path = shared.paths[token];
    threadgroup const bfloat *query = shared.queries + token * HeadDim + lane * 4;
    for (uint i = 0; i < 4; ++i)
      z[pass][i] = float(query[i]);
    for (uint up = path; up;) {
      const uint row = 31 - clz(up);
      up &= ~(1u << row);
      threadgroup const bfloat *key = shared.keys + row * HeadDim + lane * 4;
      float product = 0.0f;
      for (uint i = 0; i < 4; ++i)
        product += float(key[i]) * z[pass][i];
      const float c = float(shared.beta[row]) * simd_sum(product);
      const float d = shared.decay[row];
      for (uint i = 0; i < 4; ++i)
        z[pass][i] = d * (z[pass][i] - c * float(key[i]));
      if (lane == row)
        coefficient[pass] = c;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float *transformed = reinterpret_cast<threadgroup float *>(shared.queries);
  static_assert(sizeof(shared.queries) + sizeof(shared.keys) == Rows * HeadDim * sizeof(float),
                "the transformed queries take the q/k rows' place");
  for (uint pass = 0; pass < Passes; ++pass) {
    const uint token = simd_group + pass * kDecodeSimdgroups;
    if (token >= verified)
      continue;
    for (uint i = 0; i < 4; ++i)
      transformed[token * HeadDim + lane * 4 + i] = z[pass][i];
    if (lane < Rows)
      shared.coefficients[token * Rows + lane] = coefficient[pass];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
          uint PackedWidth, uint Rows, class W>
inline void gdn_decode_gate(threadgroup GdnDecodeShared<HeadDim, Rows> &shared,
                            device const bfloat *packed,
                            device const W *norm_weight, device bfloat *hidden,
                            bool tiled, bool sigmoid_gate, uint value_head, uint lane,
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
    // Qwen3.5 gates by silu(z), Qwen3.8-Flash-Next by sigmoid(z).
    const bfloat gated = bfloat((float(normalized) * (sigmoid_gate ? 1.0f : z)) /
                                (1.0f + fast::exp2(-1.44269504089f * z)));
    hidden[hidden_base + dim] = gated;
    shared.rows[token * HeadDim + dim] = gated;
  }
}

// Grid {value heads, lanes}.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, uint Rows, uint RowsInFlight, class Table, class W, class State>
inline void gdn_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device uchar *tape, device const float *a_scale,
    device const bfloat *dt_bias,
    device const W *gdn_norm_weight, device bfloat *gdn_hidden,
    device const uint *verify_rows, device const SplashTreeRow *tree,
    device const SplashLaneRows *lane_rows, constant GDNDecodeBatchParams &params,
    uint2 group, uint lane, uint simd_group,
    threadgroup GdnDecodeShared<HeadDim, Rows> &shared,
    device bfloat *table, device float *sums) {
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  uint batch = group.y;
  device const uchar *current = SPLASH_LANE_BINDING(batch, current0, current1, current2, current3);
  device uchar *next = SPLASH_LANE_BINDING(batch, next0, next1, next2, next3);
  // The lane's block of the step's rows.
  const uint first_row = lane_rows[batch].first;
  const uint rows = lane_rows[batch].rows;
  packed += ulong(first_row) * PackedWidth;
  tree += first_row;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const State *state_in = reinterpret_cast<device const State *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device State *state_out = reinterpret_cast<device State *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);

  const uint pending_path = params.pending_path[batch];
  const uint pending = popcount(pending_path);
  // The recurrence runs the block's rows up to the lane's verify rows.
  const uint verified = min(verify_rows[batch], rows);
  const GdnTape pending_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(tape, params.pending_tape[batch]);
  const GdnTape step_tape =
      gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(tape, params.step_tape[batch]);
  const GdnConvHistory history =
      gdn_conv_history(conv_state_in, pending_tape.inputs, ConvDim, ConvDim, pending, pending_path);

  device bfloat *lane_hidden = gdn_hidden + ulong(first_row) * ValueWidth;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows>(
      packed, conv_weights, history, pending_path, conv_state_out, pending_tape, step_tape, tree,
      rows, verified, a_scale, dt_bias, shared, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (gdn_decode_chain(shared, verified, lane)) {
    gdn_decode_scan<HeadDim, Rows, RowsInFlight, false, State>(state_in, state_out, shared, pending, verified,
                                                               group.x, lane, simd_group);
  } else {
    gdn_decode_transform<HeadDim, Rows>(shared, verified, lane, simd_group);
    gdn_decode_scan<HeadDim, Rows, RowsInFlight, true, State>(state_in, state_out, shared, pending, verified,
                                                              group.x, lane, simd_group);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The gate and the table need no barrier past here: a simdgroup's rows
  // past the block's are done.
  const bool tiled = (params.flags & GDN_TILED_HEADS) != 0;
  for (uint pass = 0; pass < GdnDecodeShared<HeadDim, Rows>::Passes; ++pass) {
    const uint token = simd_group + pass * kDecodeSimdgroups;
    if (token >= rows)
      return;
    gdn_decode_gate<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows>(
        shared, packed, gdn_norm_weight, lane_hidden, tiled,
        (params.flags & GDN_SIGMOID_GATE) != 0, group.x, lane, token);
    if (table) {
      // Each group owns this head for the block's rows; each simdgroup
      // writes the table of the row it just gated, in the table tile that
      // holds the row (blocks may share a tile, each writing its own rows of
      // it).
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint head = gdn_output_head<KeyHeads, ValueHeads>(group.x, tiled);
      const uint row = first_row + token;
      const uint tile = row / gguf_sg::kRows;
      for (uint g = 0; g < HeadDim / 64; ++g) {
        const uint column = head * HeadDim + g * 64 + 2 * lane;
        const uint local = token * HeadDim + g * 64 + 2 * lane;
        Table::write(table + ulong(tile) * ValueWidth * gguf_sg::kRows,
                     sums + ulong(tile) * Table::sums_per_tile(ValueWidth), ValueWidth,
                     column / 64, row % gguf_sg::kRows, lane, shared.rows[local], shared.rows[local + 1]);
      }
    }
  }
}

// A lane's state after the first `rows` rows of the path it accepted of its
// step (`retained_rows`, the row mask its acceptance wrote, metal/abi/
// VerifyTree.h), the one its next step folds when the lane retained exactly
// those, of one value head of one layer: those rows of the step's tape folded
// into the state before them (the cell the step wrote, `state`), and the
// convolution rows they leave, as the next step's prologue carries them, in
// `fold`, a cell of the same strides. Nothing when the path holds fewer rows.
// Grid {value heads, layers}.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint RowsInFlight, class State>
inline void gdn_fold_phase(device const uchar *state, device uchar *fold,
                           device uchar *tape, device const uint *retained_rows,
                           constant GDNFoldParams &params, uint2 group,
                           uint lane, uint simd_group,
                           threadgroup GdnPendingShared<HeadDim, SPLASH_TARGET_VERIFY_ROWS> &pending) {
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint KvWidth = KeyWidth + ValueHeads * HeadDim;
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  const uint rows = params.rows, path = retained_rows[0];
  if (uint(popcount(path)) < rows)
    return;
  const uint value_head = group.x, layer = group.y;
  const GdnTape step = gdn_tape<KeyHeads, ValueHeads, HeadDim, ConvDim>(
      tape, params.step_tape +
                ulong(layer) * gdn_tape_layer_bytes(ConvDim, KvWidth, ValueHeads));
  for (uint token = simd_group; token < rows; token += kDecodeSimdgroups)
    gdn_load_pending<KeyHeads, ValueHeads, HeadDim, SPLASH_TARGET_VERIFY_ROWS>(
        step, pending, token, splash_path_row(path, token), value_head, lane);
  // The head's v channels, and its key head's q and k channels from the first
  // of the key head's value heads.
  if (simd_group < SPLASH_GDN_CONVOLUTION_TAPS - 1) {
    const GdnConvHistory history = gdn_conv_history(
        reinterpret_cast<device const bfloat *>(state + ulong(layer) * params.conv_layer_bytes), step.inputs,
        ConvDim, ConvDim, rows, path);
    device bfloat *carried =
        reinterpret_cast<device bfloat *>(fold + ulong(layer) * params.conv_layer_bytes) +
        simd_group * ConvDim;
    const uint q_channel = value_head / HeadsPerKey * HeadDim + lane;
    const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
    for (uint g = 0; g < HeadDim / 32; ++g) {
      carried[v_channel + 32 * g] = history.at(simd_group, v_channel + 32 * g);
      if (value_head % HeadsPerKey == 0) {
        carried[q_channel + 32 * g] = history.at(simd_group, q_channel + 32 * g);
        carried[KeyWidth + q_channel + 32 * g] =
            history.at(simd_group, KeyWidth + q_channel + 32 * g);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong recurrent = params.convolution_state_bytes + ulong(layer) * params.recurrent_layer_bytes;
  device const State *state_in = reinterpret_cast<device const State *>(state + recurrent);
  device State *state_out = reinterpret_cast<device State *>(fold + recurrent);
  const auto base = [&](uint value_dim) {
    return (ulong(value_head) * HeadDim + value_dim) * HeadDim + lane * 4;
  };
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint value_dim[RowsInFlight];
    float values[RowsInFlight][4];
    for (uint r = 0; r < RowsInFlight; ++r) {
      value_dim[r] = (batch + r) * kDecodeSimdgroups + simd_group;
      for (uint i = 0; i < 4; ++i)
        values[r][i] = float(state_in[base(value_dim[r]) + i]);
    }
    gdn_fold_rows<HeadDim, SPLASH_TARGET_VERIFY_ROWS, RowsInFlight>(values, pending, rows, value_dim, lane);
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[base(value_dim[r]) + i] = State(values[r][i]);
  }
}

// W: the norm weights' stored type (float: a GGUF's F32 norms, _f32).
// verify_rows: each lane's verify rows; tree: the step's tree rows, a lane's
// block's from its first row; lane_rows: each lane's block.
#define GDN_DECODE_BUFFERS(W) \
    device const bfloat *packed [[buffer(0)]], \
    device const bfloat *conv_weights [[buffer(1)]], \
    device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]], \
    device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]], \
    device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]], \
    device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]], \
    device uchar *tape [[buffer(10)]], device const float *a_scale [[buffer(11)]], \
    device const bfloat *dt_bias [[buffer(12)]], device const W *gdn_norm_weight [[buffer(13)]], \
    device bfloat *gdn_hidden [[buffer(14)]], device const uint *verify_rows [[buffer(15)]], \
    device const SplashTreeRow *tree [[buffer(16)]], device const SplashLaneRows *lane_rows [[buffer(17)]]
#define GDN_DECODE_THREADS \
    uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows, table, sums, Layout, W, State) \
    threadgroup GdnDecodeShared<HeadDim, Rows> shared; \
    gdn_decode_batch_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows, 2, Layout, W, State>( \
        packed, conv_weights, current0, current1, current2, current3, next0, \
        next1, next2, next3, tape, a_scale, dt_bias, \
        gdn_norm_weight, gdn_hidden, verify_rows, tree, lane_rows, params, group, lane, simd_group, shared, \
        table, sums);
// Entries without a table pass null pointers, which skip the write; their Layout only completes the template.
#define GDN_DECODE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows, W, State) \
  kernel void Name(GDN_DECODE_BUFFERS(W), \
      constant GDNDecodeBatchParams &params [[buffer(18)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows, nullptr, nullptr, gguf_sg::Table16, W, \
                    State) \
  }
// The out-projection's table (Layout: gguf_sg::Table16).
#define GDN_DECODE_TABLE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows, Layout, W, State) \
  kernel void Name(GDN_DECODE_BUFFERS(W), \
      device bfloat *table [[buffer(18)]], device float *sums [[buffer(19)]], \
      constant GDNDecodeBatchParams &params [[buffer(20)]], GDN_DECODE_THREADS) { \
    GDN_DECODE_BODY(KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Rows, table, sums, Layout, W, State) \
  }
// Each entry over an fp32 state and, as Name_bf16_state, a bf16 one, of the
// narrow rows (SPLASH_GDN_NARROW_ROWS) and, as ..._r16, of a verify block's.
#define GDN_DECODE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, W) \
  GDN_DECODE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, SPLASH_GDN_NARROW_ROWS, W, float) \
  GDN_DECODE_KERNEL(Name##_bf16_state, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, SPLASH_GDN_NARROW_ROWS, W, \
                    bfloat) \
  GDN_DECODE_KERNEL(Name##_r16, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, SPLASH_TARGET_VERIFY_ROWS, W, \
                    float) \
  GDN_DECODE_KERNEL(Name##_bf16_state_r16, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, \
                    SPLASH_TARGET_VERIFY_ROWS, W, bfloat)
#define GDN_DECODE_TABLE_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, Layout, W) \
  GDN_DECODE_TABLE_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, SPLASH_GDN_NARROW_ROWS, Layout, \
                          W, float) \
  GDN_DECODE_TABLE_KERNEL(Name##_bf16_state, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, \
                          SPLASH_GDN_NARROW_ROWS, Layout, W, bfloat) \
  GDN_DECODE_TABLE_KERNEL(Name##_r16, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, SPLASH_TARGET_VERIFY_ROWS, \
                          Layout, W, float) \
  GDN_DECODE_TABLE_KERNEL(Name##_bf16_state_r16, KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth, \
                          SPLASH_TARGET_VERIFY_ROWS, Layout, W, bfloat)

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
// Qwen3.5-122B-A10B (64 value heads).
GDN_DECODE_ENTRY(verify_gdn_fused_vh64, 16, 64, 128, 12288, 20736, bfloat)
GDN_DECODE_ENTRY(verify_gdn_fused_vh64_f32, 16, 64, 128, 12288, 20736, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh64, 16, 64, 128, 12288, 20736, gguf_sg::Table16, bfloat)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_vh64_f32, 16, 64, 128, 12288, 20736, gguf_sg::Table16, float)
// Qwen3.8-Flash-Next (16512-column rows), which loads from GGUFs only, whose
// norms are F32.
GDN_DECODE_ENTRY(verify_gdn_fused_p16512_f32, 16, 48, 128, 10240, 16512, float)
GDN_DECODE_TABLE_ENTRY(verify_gdn_fused_table16_p16512_f32, 16, 48, 128, 10240, 16512, gguf_sg::Table16, float)

// The fold of a step's rows alone (GDN::addFold), over an fp32 state and, as
// Name_bf16_state, a bf16 one. It reads no packed rows, so shapes of one head
// layout share it.
#define GDN_FOLD_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, State) \
  kernel void Name(device const uchar *state [[buffer(0)]], device uchar *fold [[buffer(1)]], \
      device uchar *tape [[buffer(2)]], device const uint *retained_rows [[buffer(3)]], \
      constant GDNFoldParams &params [[buffer(4)]], GDN_DECODE_THREADS) { \
    threadgroup GdnPendingShared<HeadDim, SPLASH_TARGET_VERIFY_ROWS> pending; \
    gdn_fold_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, 2, State>( \
        state, fold, tape, retained_rows, params, group, lane, simd_group, pending); \
  }
#define GDN_FOLD_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim) \
  GDN_FOLD_KERNEL(Name, KeyHeads, ValueHeads, HeadDim, ConvDim, float) \
  GDN_FOLD_KERNEL(Name##_bf16_state, KeyHeads, ValueHeads, HeadDim, ConvDim, bfloat)

GDN_FOLD_ENTRY(verify_gdn_fold, 16, 48, 128, 10240)
GDN_FOLD_ENTRY(verify_gdn_fold_vh32, 16, 32, 128, 8192)
GDN_FOLD_ENTRY(verify_gdn_fold_vh64, 16, 64, 128, 12288)
#undef GDN_FOLD_ENTRY
#undef GDN_FOLD_KERNEL
#undef GDN_DECODE_ENTRY
#undef GDN_DECODE_TABLE_ENTRY
#undef GDN_DECODE_KERNEL
#undef GDN_DECODE_TABLE_KERNEL
#undef GDN_DECODE_BODY
#undef GDN_DECODE_THREADS
#undef GDN_DECODE_BUFFERS
