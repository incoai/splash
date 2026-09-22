#include "metal/abi/KernelABI.h"

// Sorted top-K lists: descending value, ascending token id on ties.
template <uint K>
inline void top_insert(thread float *values, thread uint *ids, float value,
                       uint token) {
  if (!(value > values[K - 1] ||
        (value == values[K - 1] && token < ids[K - 1])))
    return;
  uint slot = K - 1;
  while (slot > 0 && (value > values[slot - 1] ||
                      (value == values[slot - 1] && token < ids[slot - 1]))) {
    values[slot] = values[slot - 1];
    ids[slot] = ids[slot - 1];
    --slot;
  }
  values[slot] = value;
  ids[slot] = token;
}

// Each simdgroup drains its sorted top-K lists into threadgroup memory;
// thread 0 merges the eight lists into the shard's partial.
template <uint K>
__attribute__((always_inline)) inline void top_shard_store(
    thread float (&local_values)[K], thread uint (&local_ids)[K],
    threadgroup float *group_values, threadgroup uint *group_ids,
    device uint *partial_ids, device float *partial_values,
    uint group, uint thread_index, uint lane,
    uint simd_group) {
  uint cursor = 0;
  for (uint rank = 0; rank < K; ++rank) {
    float value = cursor < K ? local_values[cursor] : -INFINITY;
    uint token = cursor < K ? local_ids[cursor] : 0xffffffffu;
    float simd_best = simd_max(value);
    uint simd_id = simd_min(value == simd_best ? token : 0xffffffffu);
    uint winner =
        simd_min(value == simd_best && token == simd_id ? lane : 0xffffffffu);
    if (lane == 0) {
      group_values[simd_group * K + rank] = simd_best;
      group_ids[simd_group * K + rank] = simd_id;
    }
    if (lane == winner)
      ++cursor;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_index == 0) {
    float final_values[K];
    uint final_ids[K];
    for (uint i = 0; i < K; ++i) {
      final_values[i] = -INFINITY;
      final_ids[i] = 0xffffffffu;
    }
    for (uint item = 0; item < 8 * K; ++item)
      top_insert<K>(final_values, final_ids, group_values[item],
                    group_ids[item]);
    for (uint rank = 0; rank < K; ++rank) {
      partial_ids[group * K + rank] = final_ids[rank];
      partial_values[group * K + rank] = final_values[rank];
    }
  }
}

// Keep the existing sparse-buffer layout for both top-1 and top-32. The
// choice is uniform across a threadgroup; greedy lanes need only one winner.
template <uint K>
__attribute__((always_inline)) inline void target_top_shard(
    device const bfloat *source, uint vocabulary,
    device const uint *token_mask, bool constrained, ulong mask_origin,
    device uint *partial_ids, device float *partial_values,
    uint group, uint thread_index, uint lane, uint simd_group,
    threadgroup float *group_values, threadgroup uint *group_ids) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  for (uint token = (group % Shards) * 256 + thread_index; token < vocabulary;
       token += Shards * 256) {
    if (constrained &&
        (token_mask[mask_origin + token / 32] & (1u << (token % 32))) == 0)
      continue;
    top_insert<K>(values, ids, float(source[token]), token);
  }
  top_shard_store<K>(values, ids, group_values, group_ids,
                     partial_ids + ulong(group) * 32,
                     partial_values + ulong(group) * 32,
                     0, thread_index, lane, simd_group);
}

// Merges the Shards partials of one row into its sorted top-K.
template <uint K, uint Shards>
__attribute__((always_inline)) inline void top_partials_reduce(
    device const uint *partial_ids, device const float *partial_values,
    uint row, thread float (&values)[K], thread uint (&ids)[K]) {
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  uint origin = row * Shards * K;
  for (uint item = 0; item < Shards * K; ++item)
    top_insert<K>(values, ids, partial_values[origin + item],
                  partial_ids[origin + item]);
}

// One row's sparse target distribution: the merged top-32 is softmaxed at
// temperature over its first top_k entries, truncated by top_p, renormalized
// and written in ascending token-id order; slots past the valid count carry
// ~0u. Top-k=1 stores only its unit-probability winner. The constant
// references keep the divisions inside the loops.
__attribute__((always_inline)) inline void top32_probs_row(
    device const uint *partial_ids, device const float *partial_values,
    device uint *top_ids, device float *top_probs, uint row,
    constant uint &top_k, constant float &temperature, constant float &top_p) {
  if (top_k == 1) {
    float best = -INFINITY;
    uint token = 0xffffffffu;
    for (uint shard = 0; shard < SPLASH_TARGET_SAMPLING_SHARDS; ++shard) {
      ulong index = (ulong(row) * SPLASH_TARGET_SAMPLING_SHARDS + shard) * 32;
      float value = partial_values[index];
      uint id = partial_ids[index];
      if (value > best || (value == best && id < token)) {
        best = value;
        token = id;
      }
    }
    for (uint rank = 0; rank < 32; ++rank) {
      top_ids[ulong(row) * 32 + rank] = rank == 0 ? token : 0xffffffffu;
      top_probs[ulong(row) * 32 + rank] =
          rank == 0 && token != 0xffffffffu ? 1.0f : 0.0f;
    }
    return;
  }
  float values[32];
  uint ids[32];
  top_partials_reduce<32, SPLASH_TARGET_SAMPLING_SHARDS>(partial_ids, partial_values, row, values, ids);

  uint valid_count = 0;
  while (valid_count < 32 && ids[valid_count] != 0xffffffffu)
    ++valid_count;
  uint selected_count = min(top_k, valid_count);

  float probabilities[32];
  float sum = 0.0f;
  for (uint rank = 0; rank < 32; ++rank) {
    float probability = rank < selected_count
                            ? exp((values[rank] - values[0]) / temperature)
                            : 0.0f;
    probabilities[rank] = probability;
    sum += probability;
  }
  float prefix = 0.0f;
  float kept_sum = 0.0f;
  for (uint rank = 0; rank < 32; ++rank) {
    float probability = probabilities[rank] / sum;
    bool keep = rank < selected_count && prefix <= top_p;
    probabilities[rank] = keep ? probability : 0.0f;
    prefix += probability;
    kept_sum += probabilities[rank];
  }
  uint used = 0;
  for (uint output = 0; output < 32; ++output) {
    ulong destination = ulong(row) * 32 + output;
    if (output >= valid_count) {
      top_ids[destination] = 0xffffffffu;
      top_probs[destination] = 0.0f;
      continue;
    }
    uint best = 32;
    uint best_id = 0xffffffffu;
    for (uint rank = 0; rank < valid_count; ++rank) {
      if ((used & (1u << rank)) == 0 && ids[rank] < best_id) {
        best = rank;
        best_id = ids[rank];
      }
    }
    used |= 1u << best;
    top_ids[destination] = best_id;
    top_probs[destination] = probabilities[best] / kept_sum;
  }
}

kernel void
decode_sample_top32_sharded(device const bfloat *logits [[buffer(0)]],
                     device uint *partial_ids [[buffer(1)]],
                     device float *partial_values [[buffer(2)]],
                     device const uint *token_mask [[buffer(3)]],
                     constant TargetSamplingParams &params [[buffer(4)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint thread_index [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float group_values[8 * 32];
  threadgroup uint group_ids[8 * 32];
  uint row = group / SPLASH_TARGET_SAMPLING_SHARDS;
  device const bfloat *source =
      logits + ulong(params.row_offset + row) * params.vocabulary;
  ulong mask_origin = ulong(params.mask_row_offset + row) * params.mask_words;
  if (params.top_k == 1)
    target_top_shard<1>(source, params.vocabulary, token_mask, params.constrained,
                        mask_origin, partial_ids, partial_values,
                        group, thread_index, lane, simd_group,
                        group_values, group_ids);
  else
    target_top_shard<32>(source, params.vocabulary, token_mask, params.constrained,
                         mask_origin, partial_ids, partial_values,
                         group, thread_index, lane, simd_group,
                         group_values, group_ids);
}

kernel void decode_sample_sparse_top1(device const uint *top_ids [[buffer(0)]],
                        device const float *top_probs [[buffer(1)]],
                        device uint *tokens [[buffer(2)]],
                        uint row [[thread_position_in_grid]]) {
  uint best_id = 0xffffffffu;
  float best_probability = -1.0f;
  for (uint rank = 0; rank < 32; ++rank) {
    uint index = row * 32 + rank;
    float probability = top_probs[index];
    uint token = top_ids[index];
    if (probability > best_probability ||
        (probability == best_probability && token < best_id)) {
      best_probability = probability;
      best_id = token;
    }
  }
  tokens[row] = best_id;
}

kernel void decode_sample_top32_probs(device const uint *partial_ids [[buffer(0)]],
                               device const float *partial_values [[buffer(1)]],
                               device uint *top_ids [[buffer(2)]],
                               device float *top_probs [[buffer(3)]],
                               constant TargetSamplingParams &params
                               [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint thread_index
                               [[thread_index_in_threadgroup]]) {
  if (thread_index != 0)
    return;
  top32_probs_row(partial_ids, partial_values, top_ids, top_probs, row,
                  params.top_k, params.temperature, params.top_p);
}

kernel void decode_sample_top32_sharded_batch(
    device const bfloat *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    device const uint *token_mask [[buffer(3)]],
    constant TargetSamplingBatchParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float group_values[8 * 32];
  threadgroup uint group_ids[8 * 32];
  uint global_row = group / SPLASH_TARGET_SAMPLING_SHARDS;
  uint batch = global_row / params.rows_per_lane;
  uint row = global_row % params.rows_per_lane;
  if (batch >= params.lanes)
    return;
  device const bfloat *source = logits + ulong(global_row) * params.vocabulary;
  bool constrained = (params.constrained_mask & (1u << batch)) != 0;
  ulong mask_origin =
      ulong(batch) * (SPLASH_TARGET_VERIFY_ROWS + 1) * params.mask_words +
      ulong(row + 1) * params.mask_words;
  if (params.top_k[batch] == 1)
    target_top_shard<1>(source, params.vocabulary, token_mask, constrained,
                        mask_origin, partial_ids, partial_values,
                        group, thread_index, lane, simd_group,
                        group_values, group_ids);
  else
    target_top_shard<32>(source, params.vocabulary, token_mask, constrained,
                         mask_origin, partial_ids, partial_values,
                         group, thread_index, lane, simd_group,
                         group_values, group_ids);
}

kernel void decode_sample_top32_probs_batch(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device uint *top_ids [[buffer(2)]], device float *top_probs [[buffer(3)]],
    constant TargetSamplingBatchParams &params [[buffer(4)]],
    uint global_row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (thread_index != 0)
    return;
  uint batch = global_row / params.rows_per_lane;
  if (batch >= params.lanes)
    return;
  top32_probs_row(partial_ids, partial_values, top_ids, top_probs, global_row,
                  params.top_k[batch], params.temperature[batch],
                  params.top_p[batch]);
}

inline bool top_beats(float value, uint token, float other, uint other_token) {
  return value > other || (value == other && token < other_token);
}

// Register-resident sorted top-16 insert for an entry the caller has already
// checked against the last slot; the unrolled shift keeps every index static.
inline void top16_insert(thread float (&values)[16], thread uint (&ids)[16],
                         float value, uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
    bool here = top_beats(value, token, values[slot], ids[slot]);
    bool above = top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

// Pops the head of a register-resident sorted list of Count entries.
template <uint Count>
inline void top_pop(thread float (&values)[Count], thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

// The simdgroup's best list head: (value desc, id asc), so ties and the
// empty sentinel (-inf, ~0u) resolve the same way everywhere.
inline void simd_best_head(float value, uint token, thread float &best,
                           thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

// One shard of one proposal row. Threads stream their share of the shard in
// 16-byte vectors, keep the chunk in registers, and first find the 16th
// largest of the per-thread maxima: at least sixteen tokens are that large,
// so nothing below it can be in the row's top-16 and the exact sorted insert
// only runs for the few survivors. The group then pops its best sixteen in
// rank order. The (value desc, id asc) order is total, so the partial is the
// same set in the same order whatever the thread partition.
kernel void draft_select_top16_sharded(
    device const bfloat *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant uint &vocabulary [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = 16, VectorTokens = 8, ChunkVectors = 16;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start = (ulong(batch) * Rows + position + 1) * vocabulary;
  uint shard_tokens = (vocabulary + Shards - 1) / Shards;
  uint begin = min(shard * shard_tokens, vocabulary);
  uint end = min(begin + shard_tokens, vocabulary);
  // Vector loads require 16-byte alignment. Handle the shard's unaligned head
  // and tail with scalar inserts; the logits binding must also be aligned.
  uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) %
                   VectorTokens);
  head = min(head, end - begin);
  uint vectors = (end - begin - head) / VectorTokens;
  uint vector_begin = begin + head;
  uint tail_begin = vector_begin + vectors * VectorTokens;
  device const bfloat *row = logits + row_start;
  device const uint4 *vector_row =
      reinterpret_cast<device const uint4 *>(row + vector_begin);

  float values[K];
  uint ids[K];
  for (uint i = 0; i < K; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  if (thread_index < head) {
    uint token = begin + thread_index;
    float value = float(row[token]);
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }
  if (thread_index < end - tail_begin) {
    uint token = tail_begin + thread_index;
    float value = float(row[token]);
    if (top_beats(value, token, values[K - 1], ids[K - 1]))
      top16_insert(values, ids, value, token);
  }

  threadgroup float maxima[256];
  threadgroup float thresholds[8];
  for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
    uint4 loaded[ChunkVectors];
    float best = -INFINITY;
    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      loaded[i] = index < vectors ? vector_row[index] : uint4(0u);
      if (index < vectors) {
        for (uint word = 0; word < 4; ++word) {
          float low = as_type<float>(loaded[i][word] << 16);
          float high = as_type<float>(loaded[i][word] & 0xffff0000u);
          if (low > best)
            best = low;
          if (high > best)
            best = high;
        }
      }
    }
    // The 16th largest thread maximum: the minimum over the maxima that
    // fewer than sixteen others exceed.
    maxima[thread_index] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint other = 0; other < 256; ++other)
      above += maxima[other] > best ? 1u : 0u;
    float candidate = simd_min(above < K ? best : INFINITY);
    if (lane == 0)
      thresholds[simd_group] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float threshold = thresholds[0];
    for (uint other = 1; other < 8; ++other)
      threshold = min(threshold, thresholds[other]);

    for (uint i = 0; i < ChunkVectors; ++i) {
      uint index = chunk + thread_index + i * 256;
      if (index >= vectors)
        continue;
      uint token = vector_begin + index * VectorTokens;
      for (uint word = 0; word < 4; ++word) {
        float low = as_type<float>(loaded[i][word] << 16);
        float high = as_type<float>(loaded[i][word] & 0xffff0000u);
        if (low >= threshold &&
            top_beats(low, token + 2 * word, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, low, token + 2 * word);
        if (high >= threshold &&
            top_beats(high, token + 2 * word + 1, values[K - 1], ids[K - 1]))
          top16_insert(values, ids, high, token + 2 * word + 1);
      }
    }
  }

  // Sixteen rounds pop the group-wide best head; the simdgroup bests
  // alternate between two slots so one barrier per round suffices.
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  for (uint rank = 0; rank < K; ++rank) {
    float head_value = values[0];
    uint head_id = ids[0];
    float simd_value;
    uint simd_id;
    simd_best_head(head_value, head_id, simd_value, simd_id);
    uint slot = rank & 1;
    if (lane == 0) {
      round_values[slot][simd_group] = simd_value;
      round_ids[slot][simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = round_values[slot][0];
    uint best_id = round_ids[slot][0];
    for (uint other = 1; other < 8; ++other) {
      float value = round_values[slot][other];
      uint token = round_ids[slot][other];
      if (top_beats(value, token, best, best_id)) {
        best = value;
        best_id = token;
      }
    }
    if (thread_index == rank) {
      partial_ids[group * K + rank] = best_id;
      partial_values[group * K + rank] = best;
    }
    if (head_value == best && head_id == best_id)
      top_pop<K>(values, ids);
  }
}

// Merges the eight shard partials of one row into the simdgroup's lanes: each
// lane holds four consecutive entries of one shard's sorted list and sixteen
// rounds pop the simdgroup-wide best into rank order, lane `rank` keeping it.
inline void top16_merge_shards(device const uint *partial_ids,
                               device const float *partial_values,
                               uint row, uint lane, thread float &value,
                               thread uint &token) {
  constexpr uint K = 16, Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      top_pop<Entries>(values, ids);
  }
}

// One group per proposal row. Simdgroup 0 merges its candidates and unary
// scores; simdgroup 1 merges the preceding row (the anchor for row 0). The
// remaining simdgroups score the 16 x 16 predecessor/candidate edge table in
// fixed per-lane reduction order. The table follows the shard partials in
// partial-values scratch.
kernel void draft_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device float *partial_values [[buffer(1)]],
    device uint *candidates [[buffer(2)]],
    device bfloat *unary [[buffer(3)]],
    device const bfloat *hidden [[buffer(4)]],
    device const bfloat *predecessor_codebook [[buffer(5)]],
    device const bfloat *successor_codebook [[buffer(6)]],
    constant SelectorBatchParams &params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr uint Candidates = 16;
  constexpr uint Rank = 256;
  uint batch = row / Positions;
  uint position = row % Positions;
  if (batch >= params.lanes)
    return;
  threadgroup uint successors[Candidates];
  threadgroup uint predecessors[Candidates];
  if (simd_group == 0) {
    float value;
    uint token;
    top16_merge_shards(partial_ids, partial_values, row, lane, value, token);
    if (lane < Candidates) {
      candidates[row * Candidates + lane] = token;
      unary[row * Candidates + lane] = bfloat(value);
      successors[lane] = token;
    }
  } else if (simd_group == 1) {
    uint token = params.anchor[batch];
    if (position > 0) {
      float value;
      top16_merge_shards(partial_ids, partial_values, row - 1, lane, value,
                         token);
    }
    if (lane < Candidates)
      predecessors[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One task is one predecessor and eight of the sixteen candidates, so a
  // simdgroup issues all its codebook loads at once instead of one cold row
  // per edge.
  constexpr uint TaskCandidates = 8, Dims = Rank / 32;
  uint tasks = (position > 0 ? Candidates : 1) * (Candidates / TaskCandidates);
  device float *table = partial_values +
                        ulong(params.lanes) * Positions * Shards * Candidates +
                        ulong(row) * Candidates * Candidates;
  device const bfloat *row_hidden =
      hidden + (ulong(batch) * Rows + position + 1) * Rank;
  for (uint task = simd_group; task < tasks; task += threads / 32) {
    uint predecessor_index = task / (Candidates / TaskCandidates);
    uint first_candidate =
        task % (Candidates / TaskCandidates) * TaskCandidates;
    // Ids are produced by the top-k selection and are always in range; the
    // clamp only keeps a corrupted id inside the codebooks.
    uint safe_predecessor =
        min(predecessors[predecessor_index], params.vocabulary - 1u);
    float context[Dims];
    float successor[TaskCandidates][Dims];
    for (uint i = 0; i < Dims; ++i) {
      uint dim = lane + i * 32;
      context[i] = float(predecessor_codebook[safe_predecessor * Rank + dim]) *
                   float(row_hidden[dim]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      uint safe_candidate =
          min(successors[first_candidate + j], params.vocabulary - 1u);
      for (uint i = 0; i < Dims; ++i)
        successor[j][i] =
            float(successor_codebook[safe_candidate * Rank + lane + i * 32]);
    }
    for (uint j = 0; j < TaskCandidates; ++j) {
      float score = 0.0f;
      for (uint i = 0; i < Dims; ++i)
        score += context[i] * successor[j][i];
      score = simd_sum(score);
      if (lane == 0)
        table[predecessor_index * Candidates + first_candidate + j] = score;
    }
  }
}

// One thread per lane walks the seven positions: the score of a candidate is
// its unary score plus the edge from the previously chosen candidate, read
// from the table draft_select_edges left in the partial-values scratch.
kernel void draft_select_dflash(
    device const uint *candidates [[buffer(0)]],
    device const bfloat *unary [[buffer(1)]],
    device const float *partial_values [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]], device float *q_probs [[buffer(5)]],
    constant SelectorBatchParams &params [[buffer(6)]],
    uint batch [[thread_position_in_grid]]) {
  if (batch >= params.lanes)
    return;
  constexpr ulong Rows = SPLASH_DRAFT_QUERY_ROWS;
  constexpr ulong Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr ulong Shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  constexpr ulong Candidates = 16;
  candidates += batch * Positions * Candidates;
  unary += batch * Positions * Candidates;
  device const float *tables = partial_values +
                               params.lanes * Positions * Shards * Candidates +
                               batch * Positions * Candidates * Candidates;
  uniforms += batch * 2 * Rows;
  tokens += batch * Positions;
  q_probs += batch * Positions * Candidates;

  const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
  uint predecessor_index = 0;
  for (uint position = 0; position < Positions; ++position) {
    device const float *edges =
        tables + (position * Candidates + predecessor_index) * Candidates;
    float scores[Candidates];
    for (uint i = 0; i < Candidates; ++i)
      scores[i] = float(unary[position * Candidates + i]) + edges[i];
    uint selected = 0;
    if (sampling) {
      float maximum = scores[0];
      for (uint i = 1; i < Candidates; ++i)
        maximum = max(maximum, scores[i]);
      float sum = 0.0f;
      for (uint i = 0; i < Candidates; ++i) {
        float probability =
            exp((scores[i] - maximum) / params.temperature[batch]);
        q_probs[position * Candidates + i] = probability;
        sum += probability;
      }
      float cumulative = 0.0f;
      selected = Candidates - 1;
      for (uint i = 0; i < Candidates; ++i) {
        float probability = q_probs[position * Candidates + i] / sum;
        q_probs[position * Candidates + i] = probability;
        cumulative += probability;
        if (selected == Candidates - 1 &&
            cumulative > uniforms[position + 1]) {
          selected = i;
        }
      }
    } else {
      for (uint i = 1; i < Candidates; ++i) {
        if (scores[i] > scores[selected])
          selected = i;
      }
    }
    predecessor_index = selected;
    tokens[position] = candidates[position * Candidates + selected];
  }
}

inline float sparse_lookup(device const uint *ids,
                           device const float *probabilities, uint count,
                           uint token) {
  for (uint i = 0; i < count; ++i) {
    if (ids[i] == token)
      return probabilities[i];
  }
  return 0.0f;
}

inline uint sparse_sample(device const uint *ids,
                          device const float *probabilities, float uniform) {
  float total = 0.0f;
  for (uint i = 0; i < 32; ++i)
    total += probabilities[i];
  float threshold = uniform * total;
  float cumulative = 0.0f;
  uint fallback = ids[0];
  for (uint i = 0; i < 32; ++i) {
    if (!(probabilities[i] > 0.0f))
      continue;
    fallback = ids[i];
    cumulative += probabilities[i];
    if (cumulative > threshold)
      return ids[i];
  }
  return fallback;
}

inline uint sparse_residual_sample(device const uint *target_ids,
                                   device const float *target_probs,
                                   device const uint *draft_ids,
                                   device const float *draft_probs,
                                   float uniform) {
  float total = 0.0f;
  for (uint i = 0; i < 32; ++i) {
    float q = sparse_lookup(draft_ids, draft_probs, 16, target_ids[i]);
    total += max(target_probs[i] - q, 0.0f);
  }
  if (!(total > 0.0f)) {
    return sparse_sample(target_ids, target_probs, uniform);
  }
  float threshold = uniform * total;
  float cumulative = 0.0f;
  uint fallback = target_ids[0];
  for (uint i = 0; i < 32; ++i) {
    float q = sparse_lookup(draft_ids, draft_probs, 16, target_ids[i]);
    float residual = max(target_probs[i] - q, 0.0f);
    if (!(residual > 0.0f))
      continue;
    fallback = target_ids[i];
    cumulative += residual;
    if (cumulative > threshold)
      return target_ids[i];
  }
  return fallback;
}

kernel void decode_sample_sparse_draw(device const uint *top_ids [[buffer(0)]],
                                 device const float *top_probs [[buffer(1)]],
                                 device const float *uniforms [[buffer(2)]],
                                 device uint *tokens [[buffer(3)]]) {
  tokens[0] = sparse_sample(top_ids, top_probs, uniforms[0]);
}

// Keeps at most params.remaining of the accepted tokens plus the correction,
// cut after the first stop token, and records the count and the next anchor.
inline void finish_acceptance(device const uint *tokens, uint accepted,
                              AcceptParams params, device uint &retained,
                              device uint &next_anchor,
                              device uint &accepted_count) {
  accepted_count = accepted;
  retained = min(accepted + 1, params.remaining);
  for (uint i = 0; i < retained; ++i) {
    if (tokens[i] == params.stop_token_0 || tokens[i] == params.stop_token_1) {
      retained = i + 1;
      break;
    }
  }
  next_anchor = tokens[retained - 1];
}

inline void accept_sampled_lane(device const uint *draft_tokens,
                                device const uint *draft_ids,
                                device const float *draft_probs,
                                device const uint *target_ids,
                                device const float *target_probs,
                                device const float *uniforms,
                                device uint *output_tokens,
                                device uint &retained,
                                device uint &next_anchor,
                                device uint &accepted_count,
                                AcceptParams params) {
  uint accepted = 0;
  while (accepted < SPLASH_DRAFT_PROPOSAL_TOKENS) {
    uint token = draft_tokens[accepted];
    float q = sparse_lookup(draft_ids + accepted * 16,
                            draft_probs + accepted * 16, 16, token);
    float p = sparse_lookup(target_ids + accepted * 32,
                            target_probs + accepted * 32, 32, token);
    if (!(uniforms[accepted + SPLASH_TARGET_VERIFY_ROWS] * q < p))
      break;
    output_tokens[accepted] = token;
    ++accepted;
  }
  if (accepted == SPLASH_DRAFT_PROPOSAL_TOKENS) {
    output_tokens[accepted] =
        sparse_sample(target_ids + SPLASH_DRAFT_PROPOSAL_TOKENS * 32,
                      target_probs + SPLASH_DRAFT_PROPOSAL_TOKENS * 32,
                      uniforms[2 * SPLASH_TARGET_VERIFY_ROWS - 1]);
  } else {
    output_tokens[accepted] = sparse_residual_sample(
        target_ids + accepted * 32, target_probs + accepted * 32,
        draft_ids + accepted * 16, draft_probs + accepted * 16,
        uniforms[2 * SPLASH_TARGET_VERIFY_ROWS - 1]);
  }
  finish_acceptance(output_tokens, accepted, params, retained, next_anchor,
                    accepted_count);
}

kernel void decode_sample_argmax_sharded(device const bfloat *logits [[buffer(0)]],
                              device float *partial_values [[buffer(1)]],
                              device uint *partial_indices [[buffer(2)]],
                              constant uint &vocabulary [[buffer(3)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint thread_index
                              [[thread_index_in_threadgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  uint row = group / Shards;
  uint shard = group % Shards;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  device const bfloat *source = logits + ulong(row) * vocabulary;
  for (uint token = shard * 256 + thread_index; token < vocabulary;
       token += Shards * 256) {
    float value = float(source[token]);
    if (value > best || (value == best && token < best_index)) {
      best = value;
      best_index = token;
    }
  }
  uint lane = thread_index & 31;
  uint simd_group = thread_index >> 5;
  float simd_best = simd_max(best);
  uint simd_index = simd_min(best == simd_best ? best_index : 0xffffffffu);
  if (lane == 0) {
    group_values[simd_group] = simd_best;
    group_indices[simd_group] = simd_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float value = lane < 8 ? group_values[lane] : -INFINITY;
    float group_best = simd_max(value);
    uint index =
        lane < 8 && value == group_best ? group_indices[lane] : 0xffffffffu;
    index = simd_min(index);
    if (lane == 0) {
      partial_values[group] = group_best;
      partial_indices[group] = index;
    }
  }
}

kernel void decode_sample_argmax_reduce(device const float *partial_values [[buffer(0)]],
                             device const uint *partial_indices [[buffer(1)]],
                             device uint *tokens [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint Shards = SPLASH_TARGET_SAMPLING_SHARDS;
  float value = lane < Shards ? partial_values[row * Shards + lane] : -INFINITY;
  float best = simd_max(value);
  uint index = lane < Shards && value == best
                   ? partial_indices[row * Shards + lane]
                   : 0xffffffffu;
  index = simd_min(index);
  if (lane == 0)
    tokens[row] = index;
}

kernel void verify_input_tokens(
    device const uint *draft_input [[buffer(0)]],
    device const uint *draft_tokens [[buffer(1)]],
    device uint *verify_input [[buffer(2)]],
    constant VerifyInputBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  uint count = params.lanes * SPLASH_TARGET_VERIFY_ROWS;
  if (index < count) {
    uint batch = index / SPLASH_TARGET_VERIFY_ROWS;
    uint row = index % SPLASH_TARGET_VERIFY_ROWS;
    uint token = row == 0
                     ? draft_input[batch * SPLASH_TARGET_VERIFY_ROWS]
                     : draft_tokens[batch * SPLASH_DRAFT_PROPOSAL_TOKENS +
                                    row - 1];
    verify_input[index] = min(token, params.vocabulary - 1u);
  }
}

inline void accept_greedy_lane(device const uint *draft_tokens,
                               device uint *target_tokens,
                               device uint &retained,
                               device uint &next_anchor,
                               device uint &accepted_count,
                               AcceptParams params) {
  uint accepted = 0;
  while (accepted < SPLASH_DRAFT_PROPOSAL_TOKENS &&
         draft_tokens[accepted] == target_tokens[accepted]) {
    ++accepted;
  }
  finish_acceptance(target_tokens, accepted, params, retained, next_anchor,
                    accepted_count);
}

kernel void decode_accept_dflash(
    device const uint *draft_tokens [[buffer(0)]],
    device const uint *draft_ids [[buffer(1)]],
    device const float *draft_probs [[buffer(2)]],
    device const uint *target_ids [[buffer(3)]],
    device const float *target_probs [[buffer(4)]],
    device const float *uniforms [[buffer(5)]],
    device uint *target_tokens [[buffer(6)]],
    device uint *retained [[buffer(7)]],
    device uint *next_anchor [[buffer(8)]],
    device uint *accepted_count [[buffer(9)]],
    constant AcceptBatchParams &params [[buffer(10)]],
    uint batch [[threadgroup_position_in_grid]]) {
  if (batch >= params.lanes)
    return;
  uint remaining = params.remaining[batch];
  AcceptParams lane_params{remaining, params.stop_token_0,
                           params.stop_token_1};
  device const uint *lane_draft =
      draft_tokens + batch * SPLASH_DRAFT_PROPOSAL_TOKENS;
  device uint *lane_target =
      target_tokens + batch * SPLASH_TARGET_VERIFY_ROWS;
  if (params.sampling_mask & (1u << batch)) {
    accept_sampled_lane(
        lane_draft,
        draft_ids + batch * SPLASH_DRAFT_PROPOSAL_TOKENS * 16,
        draft_probs + batch * SPLASH_DRAFT_PROPOSAL_TOKENS * 16,
        target_ids + batch * SPLASH_TARGET_VERIFY_ROWS * 32,
        target_probs + batch * SPLASH_TARGET_VERIFY_ROWS * 32,
        uniforms + batch * 2 * SPLASH_TARGET_VERIFY_ROWS, lane_target,
        retained[batch], next_anchor[batch], accepted_count[batch],
        lane_params);
  } else {
    accept_greedy_lane(lane_draft, lane_target, retained[batch],
                       next_anchor[batch], accepted_count[batch], lane_params);
  }
}
