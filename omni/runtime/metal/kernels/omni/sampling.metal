// Token selection of the Omni models (ops::OmniSampling). omni_sample_codes
// draws one codec code per row from fp32 logits of at most 4096 ids: the
// repetition penalty over the row's history (logit < 0 ? logit * p : logit /
// p), the suppression of ids from suppress_from on except EOS, the optional
// per-row EOS mask, then greedy argmax at temperature 0 or temperature,
// top-k (at most 256, by radix select, ties to the lowest id, then ranked
// into descending order), top-p over the kept ids and a draw from the row's
// uniform. omni_argmax takes the greedy token of wide rows (the ASR
// vocabulary).

#include "metal/abi/Omni.h"
#include <metal_stdlib>

using namespace metal;

// The maximum of 256 threads' (value, index) pairs, lowest index on ties.
inline void omni_block_argmax(thread float &value, thread uint &index, threadgroup float *values,
                              threadgroup uint *indices, uint tid, uint lane, uint sg) {
  float best = value;
  uint best_index = index;
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_index = simd_shuffle_down(best_index, offset);
    if (other > best || (other == best && other_index < best_index)) {
      best = other;
      best_index = other_index;
    }
  }
  if (lane == 0) {
    values[sg] = best;
    indices[sg] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  best = values[0];
  best_index = indices[0];
  for (uint g = 1; g < 8; ++g)
    if (values[g] > best || (values[g] == best && indices[g] < best_index)) {
      best = values[g];
      best_index = indices[g];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  value = best;
  index = best_index;
  (void)tid;
}

// The order key of a logit: a monotonic unsigned image of the float, so the
// k largest values are the k largest keys.
inline uint omni_order_key(float value) {
  const uint bits = as_type<uint>(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// A row's code: the policy applied to its logits, then greedy, or top-k by
// radix select (exact: the kept ids are the k largest (value, lowest index)
// pairs, as repeated argmax would pick them), the kept ids ranked into
// descending order, softmax at the temperature, top-p and the draw.
kernel void omni_sample_codes(device const float *logits [[buffer(0)]], device const uint *history [[buffer(1)]],
                              device const uint *history_counts [[buffer(2)]], device const float *uniforms [[buffer(3)]],
                              device uint *output [[buffer(4)]], constant OmniSampleParams &p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[OMNI_SAMPLE_MAXIMUM_VOCABULARY];
  threadgroup atomic_uint marks[OMNI_SAMPLE_MAXIMUM_VOCABULARY / 32];
  threadgroup atomic_uint bins[256];
  threadgroup uint kept_indices[OMNI_SAMPLE_MAXIMUM_TOP_K];
  threadgroup uint sorted_indices[OMNI_SAMPLE_MAXIMUM_TOP_K];
  threadgroup float probabilities[OMNI_SAMPLE_MAXIMUM_TOP_K];
  threadgroup float reduce_values[8];
  threadgroup uint reduce_indices[8];
  threadgroup uint selected_bin, remaining_shared, counters[2], stripe_totals[8];
  if (row >= p.rows) return;
  const uint V = p.vocabulary;
  for (uint i = tid; i < OMNI_SAMPLE_MAXIMUM_VOCABULARY / 32; i += 256)
    atomic_store_explicit(&marks[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool penalized = p.repetition_penalty != 1.0f;
  if (penalized) {
    const uint count = min(history_counts[row], p.history_stride);
    device const uint *past = history + ulong(row) * p.history_stride;
    for (uint i = tid; i < count; i += 256) {
      const uint token = past[i];
      if (token < V) atomic_fetch_or_explicit(&marks[token / 32], 1u << (token % 32), memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const float *row_logits = logits + ulong(row) * V;
  for (uint i = tid; i < V; i += 256) {
    float value = row_logits[i];
    if (penalized && (atomic_load_explicit(&marks[i / 32], memory_order_relaxed) >> (i % 32)) & 1u)
      value = value < 0.0f ? value * p.repetition_penalty : value / p.repetition_penalty;
    if (p.suppress_from && i >= p.suppress_from && i != p.eos) value = -INFINITY;
    if (row < 32 && ((p.mask_eos >> row) & 1u) && i == p.eos) value = -INFINITY;
    values[i] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool greedy = p.temperature <= 0.0f;
  const uint stride = max(p.output_stride, 1u);
  if (greedy) {
    float best = -INFINITY;
    uint best_index = 0xffffffffu;
    for (uint i = tid; i < V; i += 256)
      if (values[i] > best || (values[i] == best && i < best_index)) {
        best = values[i];
        best_index = i;
      }
    omni_block_argmax(best, best_index, reduce_values, reduce_indices, tid, lane, sg);
    if (tid == 0) output[row * stride] = best_index < V ? best_index : 0;
    return;
  }
  const uint k = min(max(p.top_k, 1u), min(OMNI_SAMPLE_MAXIMUM_TOP_K, V));
  // Radix select of the k-th largest key, 8 bits a pass from the top: the
  // selected bin holds the k-th largest among the keys sharing the prefix;
  // `remaining` is how many of the keys equal to the final threshold are
  // among the k largest.
  uint prefix = 0, remaining = k;
  for (uint pass = 0; pass < 4; ++pass) {
    const uint shift = 24 - pass * 8;
    const uint mask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
    for (uint i = tid; i < 256; i += 256) atomic_store_explicit(&bins[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < V; i += 256) {
      const uint key = omni_order_key(values[i]);
      if ((key & mask) == prefix) atomic_fetch_add_explicit(&bins[(key >> shift) & 0xffu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Counts of the bins above each bin (descending suffix sums): bin b
    // is the one where the keys at or above it first reach `remaining`.

    // Suffix over bins tid..255: a prefix sum over the reversed order.
    const uint reversed = 255 - tid;
    const uint reversed_count = atomic_load_explicit(&bins[reversed], memory_order_relaxed);
    uint inclusive = simd_prefix_inclusive_sum(reversed_count);
    if (lane == 31) stripe_totals[sg] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < sg; ++g) inclusive += stripe_totals[g];
    // Thread tid holds the count of keys in bins >= reversed, with
    // `inclusive`; bins above it hold inclusive - reversed_count.
    const uint above = inclusive - reversed_count;
    if (above < remaining && inclusive >= remaining) {
      selected_bin = reversed;
      remaining_shared = remaining - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= selected_bin << shift;
    remaining = remaining_shared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint threshold = prefix;
  // The keys above the threshold are all kept; of the keys equal to it the
  // `remaining` lowest ids are.
  if (tid == 0) {
    counters[0] = 0;
    counters[1] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < V; i += 256)
    if (omni_order_key(values[i]) > threshold) {
      const uint slot = atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&counters[0]), 1u, memory_order_relaxed);
      kept_indices[slot] = i;
    } else if (omni_order_key(values[i]) == threshold) {
      atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&counters[1]), 1u, memory_order_relaxed);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint above_count = counters[0], tie_count = counters[1];
  if (tie_count == remaining) {
    // Every tie is kept: order among them does not matter yet.
    for (uint i = tid; i < V; i += 256)
      if (omni_order_key(values[i]) == threshold) {
        const uint slot = above_count + atomic_fetch_add_explicit(reinterpret_cast<threadgroup atomic_uint *>(&counters[1]), 1u, memory_order_relaxed) - tie_count;
        kept_indices[slot] = i;
      }
  } else {
    // The `remaining` lowest tied ids, by their rank in id order: stripes
    // of 256 ids, a prefix sum per stripe.
    uint base = 0;
    for (uint begin = 0; begin < V; begin += 256) {
      const uint i = begin + tid;
      const uint tie = i < V && omni_order_key(values[i]) == threshold ? 1u : 0u;
      uint rank = simd_prefix_exclusive_sum(tie);
      const uint simd_total = simd_sum(tie);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) stripe_totals[sg] = simd_total;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint total = 0;
      for (uint g = 0; g < 8; ++g) {
        if (g < sg) rank += stripe_totals[g];
        total += stripe_totals[g];
      }
      if (tie && base + rank < remaining) kept_indices[above_count + base + rank] = i;
      base += total;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Rank the kept ids into descending (value, lowest id) order.
  if (tid < k) {
    const uint mine = kept_indices[tid];
    const float value = values[mine];
    uint rank = 0;
    for (uint j = 0; j < k; ++j) {
      const uint other = kept_indices[j];
      const float other_value = values[other];
      rank += (other_value > value || (other_value == value && other < mine)) ? 1u : 0u;
    }
    sorted_indices[rank] = mine;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Softmax over the kept ids at the temperature.
  const float top = values[sorted_indices[0]] / p.temperature;
  if (tid < k) {
    const float value = values[sorted_indices[tid]];
    probabilities[tid] = value == -INFINITY ? 0.0f : exp(value / p.temperature - top);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float mass = 0.0f;
  for (uint i = 0; i < k; ++i) mass += probabilities[i];
  // The top-p nucleus: the smallest prefix whose mass reaches top_p.
  uint kept = k;
  if (p.top_p < 1.0f) {
    float cumulative = 0.0f;
    for (uint i = 0; i < k; ++i) {
      cumulative += probabilities[i] / mass;
      if (cumulative >= p.top_p) {
        kept = i + 1;
        break;
      }
    }
  }
  float nucleus = 0.0f;
  for (uint i = 0; i < kept; ++i) nucleus += probabilities[i];
  const float draw = uniforms[row * max(p.uniform_stride, 1u)] * nucleus;
  float cumulative = 0.0f;
  uint selected = sorted_indices[kept - 1];
  for (uint i = 0; i < kept; ++i) {
    cumulative += probabilities[i];
    if (draw < cumulative) {
      selected = sorted_indices[i];
      break;
    }
  }
  output[row * stride] = selected < V ? selected : 0;
}

// The greedy token of a wide row (the ASR vocabulary): a 1024-thread group
// per row, four logits a load, then the group's (value, lowest index)
// maximum over the simdgroups.
// The first index of a row's largest logit (0 for a row without a finite
// one), known to thread 0 of the row's threadgroup.
inline uint omni_argmax_row(device const float *logits, constant OmniArgmaxParams &p, uint row, uint tid, uint lane,
                            uint sg, threadgroup float *reduce_values, threadgroup uint *reduce_indices) {
  device const float *row_logits = logits + ulong(row) * p.stride;
  float best = -INFINITY;
  uint best_index = 0xffffffffu;
  const uint vectors = p.vocabulary / 4;
  for (uint v = tid; v < vectors; v += 1024) {
    const float4 value = *reinterpret_cast<device const float4 *>(row_logits + v * 4);
    const uint i = v * 4;
    if (value.x > best || (value.x == best && i < best_index)) { best = value.x; best_index = i; }
    if (value.y > best || (value.y == best && i + 1 < best_index)) { best = value.y; best_index = i + 1; }
    if (value.z > best || (value.z == best && i + 2 < best_index)) { best = value.z; best_index = i + 2; }
    if (value.w > best || (value.w == best && i + 3 < best_index)) { best = value.w; best_index = i + 3; }
  }
  for (uint i = vectors * 4 + tid; i < p.vocabulary; i += 1024) {
    const float value = row_logits[i];
    if (value > best || (value == best && i < best_index)) {
      best = value;
      best_index = i;
    }
  }
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_index = simd_shuffle_down(best_index, offset);
    if (other > best || (other == best && other_index < best_index)) {
      best = other;
      best_index = other_index;
    }
  }
  if (lane == 0) {
    reduce_values[sg] = best;
    reduce_indices[sg] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return 0;
  best = reduce_values[0];
  best_index = reduce_indices[0];
  for (uint g = 1; g < 32; ++g)
    if (reduce_values[g] > best || (reduce_values[g] == best && reduce_indices[g] < best_index)) {
      best = reduce_values[g];
      best_index = reduce_indices[g];
    }
  return best_index < p.vocabulary ? best_index : 0;
}

kernel void omni_argmax(device const float *logits [[buffer(0)]], device uint *output [[buffer(1)]],
                        constant OmniArgmaxParams &p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float reduce_values[32];
  threadgroup uint reduce_indices[32];
  const uint best = omni_argmax_row(logits, p, row, tid, lane, sg, reduce_values, reduce_indices);
  if (tid == 0) output[row * max(p.output_stride, 1u)] = best;
}

// The argmax through a token map: a draft's reduced vocabulary to the
// target's tokens (a DFlash draft's d2t, as target ids).
kernel void omni_argmax_mapped(device const float *logits [[buffer(0)]], device uint *output [[buffer(1)]],
                               device const uint *map [[buffer(2)]], constant OmniArgmaxParams &p [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  threadgroup float reduce_values[32];
  threadgroup uint reduce_indices[32];
  const uint best = omni_argmax_row(logits, p, row, tid, lane, sg, reduce_values, reduce_indices);
  if (tid == 0) output[row * max(p.output_stride, 1u)] = map[best];
}
