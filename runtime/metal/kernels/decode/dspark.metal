#include "metal/abi/DSpark.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/lane_bindings.h"

// A DSpark draft's decode kernels (ops::DSpark): the commit of each lane's
// f_rnn row, the Markov head's steps over the whole vocabulary, and the fold
// of its output projection into the vocabulary at load.

constant constexpr uint kRank = SPLASH_DSPARK_RANK;
constant constexpr uint kJoint = SPLASH_DSPARK_JOINT;
constant constexpr uint kCandidates = SPLASH_DRAFT_CANDIDATES;
constant constexpr uint kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constant constexpr uint kBlockRows = SPLASH_DRAFT_QUERY_ROWS;
constant constexpr uint kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;

// One group per lane (DSparkCommitParams): the lane's last retained row of
// its rows of the step, the path's last (its retained rows' highest set bit
// of those its rows hold), into its f_rnn row. Acceptance retains a lane's
// anchor at least.
kernel void dspark_frnn_commit(device const bfloat *rows [[buffer(0)]],
                               device bfloat *frnn0 [[buffer(1)]],
                               device bfloat *frnn1 [[buffer(2)]],
                               device bfloat *frnn2 [[buffer(3)]],
                               device bfloat *frnn3 [[buffer(4)]],
                               device bfloat *frnn4 [[buffer(5)]],
                               device bfloat *frnn5 [[buffer(6)]],
                               device bfloat *frnn6 [[buffer(7)]],
                               device bfloat *frnn7 [[buffer(8)]],
                               device const uint *retained_rows [[buffer(9)]],
                               device const SplashLaneRows *lane_rows [[buffer(10)]],
                               constant DSparkCommitParams &params [[buffer(11)]],
                               uint lane [[threadgroup_position_in_grid]],
                               uint thread_index [[thread_index_in_threadgroup]]) {
  if (lane >= params.lanes)
    return;
  uint path = retained_rows[lane];
  const uint count = min(uint(popcount(path)), lane_rows[lane].rows);
  if (!count)
    return;
  for (uint earlier = 0; earlier + 1 < count; ++earlier)
    path &= path - 1;
  device const bfloat *source = rows + (ulong(lane_rows[lane].first) + ctz(path)) * params.width;
  device bfloat *destination = SPLASH_LANE_BINDING(lane, frnn0, frnn1, frnn2, frnn3, frnn4, frnn5, frnn6, frnn7);
  for (uint column = thread_index; column < params.width; column += 256)
    destination[column] = source[column];
}

// One simdgroup per joint output, eight a group, over every lane: z = W
// [state; w1[prev]; h; f_rnn] + b, the state zero at step 0 and prev the
// lane's anchor there. h is the lane's draft hidden row step + 1, f_rnn the
// lane's context row (rows of params.hidden bf16 values).
kernel void dspark_markov_joint(device const bfloat *joint [[buffer(0)]],
                                device const float *bias [[buffer(1)]],
                                device const float *state [[buffer(2)]],
                                device const bfloat *embedding [[buffer(3)]],
                                device const uint *previous [[buffer(4)]],
                                device const bfloat *hidden [[buffer(5)]],
                                device const bfloat *frnn [[buffer(6)]],
                                device float *z [[buffer(7)]],
                                constant DSparkMarkovParams &params [[buffer(8)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint output = group * 8 + simd_group;
  if (output >= kJoint)
    return;
  const uint width = 2 * kRank + 2 * params.hidden;
  device const bfloat *weights = joint + ulong(output) * width;
  float sums[kLanes] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (params.step > 0)
    for (uint k = lane; k < kRank; k += 32) {
      const float w = float(weights[k]);
      for (uint l = 0; l < kLanes; ++l)
        if (l < params.lanes)
          sums[l] += w * state[l * kRank + k];
    }
  for (uint k = lane; k < kRank; k += 32) {
    const float w = float(weights[kRank + k]);
    for (uint l = 0; l < kLanes; ++l)
      if (l < params.lanes) {
        const uint token = params.step > 0 ? previous[l] : params.anchor[l];
        sums[l] += w * float(embedding[ulong(token) * kRank + k]);
      }
  }
  device const bfloat *h = weights + 2 * kRank, *f = h + params.hidden;
  for (uint k = lane; k < params.hidden; k += 32) {
    const float wh = float(h[k]), wf = float(f[k]);
    for (uint l = 0; l < kLanes; ++l)
      if (l < params.lanes) {
        const ulong row = ulong(l) * kBlockRows + params.step + 1;
        sums[l] += wh * float(hidden[row * params.hidden + k]) + wf * float(frnn[ulong(l) * params.hidden + k]);
      }
  }
  for (uint l = 0; l < kLanes; ++l) {
    const float total = simd_sum(sums[l]);
    if (l < params.lanes && lane == 0)
      z[l * kJoint + output] = total + bias[output];
  }
}

// The step's corrected logits of every lane over the tokens it may propose:
// cap * tanh((base + W2 o + b2) / cap), o = tanh(z's output part), the
// lane's base logits its draft row step + 1's. A group's simdgroups take a
// token each in turn, its lanes eight of the token's W2 row each.
kernel void dspark_markov_logits(device const float *z [[buffer(0)]],
                                 device const bfloat *fold_weights [[buffer(1)]],
                                 device const float *fold_bias [[buffer(2)]],
                                 device const float *base [[buffer(3)]],
                                 device float *logits [[buffer(4)]],
                                 constant DSparkMarkovParams &params [[buffer(5)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint groups [[threadgroups_per_grid]],
                                 uint thread_index [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float outputs[kLanes][kRank];
  for (uint l = 0; l < kLanes; ++l)
    outputs[l][thread_index] = l < params.lanes ? precise::tanh(z[l * kJoint + 2 * kRank + thread_index]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float o[kLanes][8];
  for (uint l = 0; l < kLanes; ++l)
    for (uint k = 0; k < 8; ++k)
      o[l][k] = outputs[l][lane * 8 + k];
  const uint share = (params.proposed + groups - 1) / groups;
  const uint begin = min(group * share, params.proposed), end = min(begin + share, params.proposed);
  for (uint token = begin + simd_group; token < end; token += 8) {
    device const bfloat *row = fold_weights + ulong(token) * kRank + lane * 8;
    float w[8];
    for (uint k = 0; k < 8; ++k)
      w[k] = float(row[k]);
    float sums[kLanes];
    for (uint l = 0; l < kLanes; ++l) {
      float sum = 0.0f;
      for (uint k = 0; k < 8; ++k)
        sum += w[k] * o[l][k];
      sums[l] = simd_sum(sum);
    }
    if (lane < params.lanes) {
      const ulong baseRow = ulong(lane) * kBlockRows + params.step + 1;
      const float value = base[baseRow * params.vocabulary + token] + sums[lane] + fold_bias[token];
      logits[ulong(lane) * params.vocabulary + token] = params.cap * precise::tanh(value / params.cap);
    }
  }
}

inline bool dspark_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// A sorted top-16 in registers, best first ((value desc, token asc), so ties
// resolve to the lowest token), for an entry that beats its last.
inline void dspark_insert(thread float (&values)[kCandidates], thread uint (&ids)[kCandidates], float value,
                          uint token) {
#pragma clang loop unroll(full)
  for (uint slot = kCandidates - 1; slot > 0; --slot) {
    const bool here = dspark_beats(value, token, values[slot], ids[slot]);
    const bool above = dspark_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  const bool top = dspark_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// The group's best 16 of every thread's sorted list, in order, to `values`
// and `ids` (thread 0's): sixteen rounds, each taking the best head of all
// threads, whose thread pops it.
inline void dspark_merge(thread float (&values)[kCandidates], thread uint (&ids)[kCandidates],
                         threadgroup float *best_values, threadgroup uint *best_ids, threadgroup float *out_values,
                         threadgroup uint *out_ids, uint thread_index, uint lane, uint simd_group) {
  for (uint rank = 0; rank < kCandidates; ++rank) {
    const float head = simd_max(values[0]);
    const uint token = simd_min(values[0] == head ? ids[0] : 0xffffffffu);
    if (lane == 0) {
      best_values[simd_group] = head;
      best_ids[simd_group] = token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float value = best_values[0];
      uint id = best_ids[0];
      for (uint other = 1; other < 8; ++other)
        if (dspark_beats(best_values[other], best_ids[other], value, id)) {
          value = best_values[other];
          id = best_ids[other];
        }
      out_values[rank] = value;
      out_ids[rank] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (values[0] == out_values[rank] && ids[0] == out_ids[rank]) {
#pragma clang loop unroll(full)
      for (uint slot = 0; slot + 1 < kCandidates; ++slot) {
        values[slot] = values[slot + 1];
        ids[slot] = ids[slot + 1];
      }
      values[kCandidates - 1] = -INFINITY;
      ids[kCandidates - 1] = 0xffffffffu;
    }
  }
}

// One group per (lane, shard): the shard's top 16 of the lane's step
// logits, into its partial (values and tokens, [lane][shard][16]).
kernel void dspark_markov_top16(device const float *logits [[buffer(0)]],
                                device float *partial_values [[buffer(1)]],
                                device uint *partial_ids [[buffer(2)]],
                                constant DSparkMarkovParams &params [[buffer(3)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint thread_index [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_DSPARK_SHARDS;
  const uint batch = group / Shards, shard = group % Shards;
  if (batch >= params.lanes)
    return;
  const uint share = (params.proposed + Shards - 1) / Shards;
  const uint begin = min(shard * share, params.proposed), end = min(begin + share, params.proposed);
  device const float *row = logits + ulong(batch) * params.vocabulary;
  float values[kCandidates];
  uint ids[kCandidates];
  for (uint i = 0; i < kCandidates; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  for (uint token = begin + thread_index; token < end; token += 256) {
    const float value = row[token];
    if (dspark_beats(value, token, values[kCandidates - 1], ids[kCandidates - 1]))
      dspark_insert(values, ids, value, token);
  }
  threadgroup float best_values[8], out_values[kCandidates];
  threadgroup uint best_ids[8], out_ids[kCandidates];
  dspark_merge(values, ids, best_values, best_ids, out_values, out_ids, thread_index, lane, simd_group);
  if (thread_index < kCandidates) {
    const ulong at = (ulong(batch) * Shards + shard) * kCandidates + thread_index;
    partial_values[at] = out_values[thread_index];
    partial_ids[at] = out_ids[thread_index];
  }
}

// The splitmix64 finalizer.
inline ulong dspark_hash(ulong value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
  return value ^ (value >> 31);
}

// A standard Gumbel of `token` at step `step` under seed `seed`, of a uniform
// in (0, 1) from the top 23 bits of their hash.
inline float dspark_gumbel(ulong seed, uint step, uint token) {
  const ulong value = dspark_hash(dspark_hash(seed + 0x9e3779b97f4a7c15ul * (ulong(step) + 1)) ^
                                  0xd1b54a32d192ed03ul * (ulong(token) + 1));
  const float uniform = (float(value >> 41) + 0.5f) * 0x1p-23f;
  return -log(-log(uniform));
}

// One group per lane: the step's 16 candidates, the shards' best, and the
// one the lane chooses (a greedy lane its most likely, a sampled lane a draw
// from their softmax at its temperature, by the Gumbel-max of its seed),
// first, the others after it in order: the lane's candidates (tokens) and
// their logits (unary) at position `step` ([lane][position][16], the
// selector's lattice, DraftSelector::addChain), the token the next step
// follows, and the state after the step: s = g s + (1 - g) c, g = sigmoid(z's
// gate part), c = tanh(z's candidate part).
kernel void dspark_markov_select(device const float *partial_values [[buffer(0)]],
                                 device const uint *partial_ids [[buffer(1)]],
                                 device const float *z [[buffer(2)]],
                                 device float *state [[buffer(3)]],
                                 device uint *previous [[buffer(4)]],
                                 device uint *candidates [[buffer(5)]],
                                 device float *unary [[buffer(6)]],
                                 constant DSparkMarkovParams &params [[buffer(7)]],
                                 uint batch [[threadgroup_position_in_grid]],
                                 uint thread_index [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Entries = SPLASH_DSPARK_SHARDS * kCandidates;
  if (batch >= params.lanes)
    return;
  float values[kCandidates];
  uint ids[kCandidates];
  for (uint i = 0; i < kCandidates; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  for (uint entry = thread_index; entry < Entries; entry += 256) {
    const ulong at = ulong(batch) * Entries + entry;
    const float value = partial_values[at];
    const uint token = partial_ids[at];
    if (dspark_beats(value, token, values[kCandidates - 1], ids[kCandidates - 1]))
      dspark_insert(values, ids, value, token);
  }
  threadgroup float best_values[8], out_values[kCandidates];
  threadgroup uint best_ids[8], out_ids[kCandidates];
  threadgroup uint chosen;
  dspark_merge(values, ids, best_values, best_ids, out_values, out_ids, thread_index, lane, simd_group);
  if (thread_index == 0) {
    uint pick = 0;
    const float temperature = params.temperature[batch];
    if (temperature > 0.0f) {
      const ulong seed = ulong(params.noise[batch][1]) << 32 | params.noise[batch][0];
      float best = -INFINITY;
      for (uint i = 0; i < kCandidates; ++i) {
        if (out_ids[i] == 0xffffffffu)
          continue;
        const float key = (out_values[i] - out_values[0]) / temperature + dspark_gumbel(seed, params.step, out_ids[i]);
        if (key > best) {
          best = key;
          pick = i;
        }
      }
    }
    chosen = pick;
    previous[batch] = out_ids[pick];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < kCandidates) {
    // The chosen candidate first, then the others in their order.
    const uint source = thread_index == 0 ? chosen : thread_index - (thread_index <= chosen ? 1 : 0);
    const ulong at = (ulong(batch) * kPositions + params.step) * kCandidates + thread_index;
    candidates[at] = out_ids[source];
    unary[at] = out_values[source];
  }
  device const float *joint = z + ulong(batch) * kJoint;
  device float *s = state + ulong(batch) * kRank;
  const float gate = 1.0f / (1.0f + exp(-joint[thread_index]));
  const float candidate = precise::tanh(joint[kRank + thread_index]);
  const float previous_state = params.step > 0 ? s[thread_index] : 0.0f;
  s[thread_index] = gate * previous_state + (1.0f - gate) * candidate;
}

// One thread per vocabulary row (DSparkFoldParams): of the head's projection
// of `rows` rows from row `first` on, row r < rank is column r of the folded
// weights, row rank the bias's high part and row rank + 1 its low part, which
// adds to it.
kernel void dspark_fold_rows(device const float *projected [[buffer(0)]],
                             device bfloat *fold_weights [[buffer(1)]],
                             device float *fold_bias [[buffer(2)]],
                             constant DSparkFoldParams &params [[buffer(3)]],
                             uint token [[thread_position_in_grid]]) {
  if (token >= params.vocabulary)
    return;
  for (uint row = 0; row < params.rows; ++row) {
    const uint column = params.first + row;
    const float value = projected[ulong(row) * params.vocabulary + token];
    if (column < params.rank)
      fold_weights[ulong(token) * params.rank + column] = bfloat(value);
    else if (column == params.rank)
      fold_bias[token] = value;
    else
      fold_bias[token] += value;
  }
}
