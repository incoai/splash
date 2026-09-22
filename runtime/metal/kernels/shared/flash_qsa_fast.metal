#include "metal/abi/FlashQSAFast.h"
#include <metal_stdlib>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint flash_qsa_fast_budget = 512;
constant uint flash_qsa_fast_width = 2051;
constant uint flash_qsa_fast_numeric_failure = 1u << 8;
constant uint flash_qsa_fast_position_failure = 1u << 9;

inline void flash_qsa_fast_failure(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}

inline uint flash_qsa_fast_visible(FlashQSAParams params, uint row) {
  const uint count = params.begin + row + 1;
  return metal::min(count / 4, flash_qsa_fast_budget) * 4 + count % 4;
}

inline uint flash_qsa_fast_token(device const uint *selected,
                                  FlashQSAParams params, uint row, uint slot) {
  const uint completed = (params.begin + row + 1) / 4;
  const uint picked = metal::min(completed, flash_qsa_fast_budget);
  return slot < picked * 4
      ? selected[ulong(row) * flash_qsa_fast_budget + slot / 4] * 4 + slot % 4
      : completed * 4 + slot - picked * 4;
}

inline bfloat flash_qsa_fast_gate(bfloat attention, bfloat source) {
  const bfloat exponential = bfloat(precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  const bfloat sigmoid = source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
  return attention * sigmoid;
}

inline float flash_qsa_fast_inverse(threadgroup float *partial, float square,
                                      uint dimension, float epsilon, uint tid,
                                      uint lane, uint simd) {
  const float sum = simd_sum(square);
  if (lane == 0)
    partial[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += partial[i];
    partial[0] = precise::rsqrt(total / float(dimension) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partial[0];
}

inline float flash_qsa_fast_weight(device const uchar *weight, uint dimension,
                                    bool is_float) {
  return is_float ? ((device const float *)weight)[dimension]
                  : float(((device const bfloat *)weight)[dimension]);
}

inline float flash_qsa_fast_rotate(threadgroup float *normalized, uint column,
                                     long position, float theta) {
  if (column >= 64)
    return normalized[column];
  const uint pair = column & 31u;
  const float frequency = 1.0f / metal::pow(theta, float(pair) / 32.0f);
  const float angle = float(position) * frequency;
  const float c = metal::cos(angle), s = metal::sin(angle);
  return column < 32 ? normalized[column] * c - normalized[column + 32] * s
                     : normalized[column] * c + normalized[column - 32] * s;
}

// Fuses three normalization/RoPE launches plus append while retaining the
// qualified per-head FP32 reduction tree and BF16 storage boundaries. Each
// output/cache row has exactly one owner; complete-block pooling stays ordered
// after this dispatch in the same asynchronous command.
kernel void flash_qsa_fast_prepare(
    device const bfloat *q_projection [[buffer(0)]],
    device const bfloat *k_projection [[buffer(1)]],
    device const bfloat *v_projection [[buffer(2)]],
    device const bfloat *index_projection [[buffer(3)]],
    device const uchar *q_weight [[buffer(4)]],
    device const uchar *k_weight [[buffer(5)]],
    device const uchar *index_weight [[buffer(6)]],
    device const long *positions [[buffer(7)]],
    device bfloat *queries [[buffer(8)]],
    device bfloat *key_cache [[buffer(9)]],
    device bfloat *index_queries [[buffer(10)]],
    device bfloat *value_cache [[buffer(11)]],
    device bfloat *raw_index_cache [[buffer(12)]],
    device long *index_positions [[buffer(13)]],
    device atomic_uint *diagnostics [[buffer(14)]],
    constant FlashQSAFastParams &params [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, fused_head = group.y;
  if (row >= params.common.rows || fused_head >= 30)
    return;
  threadgroup float normalized[256];
  threadgroup float partial[8];
  const uint plane = fused_head < 24 ? 0 : (fused_head < 26 ? 1 : 2);
  const uint head = fused_head - (plane == 0 ? 0 : (plane == 1 ? 24 : 26));
  const uint dimension = plane == 2 ? 128 : 256;
  const uint heads = plane == 0 ? 24 : (plane == 1 ? 2 : 4);
  const uint stride = plane == 0 ? 12288 : (plane == 1 ? 512 : 640);
  const uint head_stride = plane == 0 ? 512 : dimension;
  const device bfloat *input = plane == 0 ? q_projection
                               : (plane == 1 ? k_projection : index_projection);
  const device uchar *weight = plane == 0 ? q_weight
                               : (plane == 1 ? k_weight : index_weight);
  device bfloat *output = plane == 0 ? queries
                          : (plane == 1 ? key_cache : index_queries);
  const ulong physical_row = ulong(params.common.begin) + row;
  const long position = params.common.positions_supplied ? positions[row] : long(physical_row);
  if (plane == 1) {
    const bfloat value = v_projection[ulong(row) * 512 + head * 256 + tid];
    value_cache[physical_row * 512 + head * 256 + tid] = value;
    if (!isfinite(float(value)))
      flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
  }
  if (plane == 2 && head == 0) {
    if (tid < 128) {
      const bfloat raw = index_projection[ulong(row) * 640 + 512 + tid];
      raw_index_cache[physical_row * 128 + tid] = raw;
      if (!isfinite(float(raw)))
        flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
    }
    if (tid == 0)
      index_positions[physical_row] = position;
  }
  const ulong source = ulong(row) * stride + head * head_stride;
  const float value = tid < dimension ? float(input[source + tid]) : 0.0f;
  if (!isfinite(value))
    flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
  const float inverse = flash_qsa_fast_inverse(partial, value * value, dimension,
      params.common.epsilon, tid, lane, simd);
  if (tid < dimension) {
    const float raw_weight = flash_qsa_fast_weight(weight, tid,
                                                   (params.norm_dtype_mask & (1u << plane)) != 0);
    const uint convention = (params.norm_convention_bits >> (2 * plane)) & 3u;
    const float scale = convention == 0 ? 1.0f + raw_weight : raw_weight;
    normalized[tid] = float(bfloat((value * inverse) * scale));
    if (!isfinite(normalized[tid]))
      flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
  }
  if (tid == 0 && (position < 0 || position > 2147483647L))
    flash_qsa_fast_failure(diagnostics, flash_qsa_fast_position_failure);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < dimension) {
    const ulong output_row = plane == 1 ? physical_row : row;
    const float rotated = flash_qsa_fast_rotate(normalized, tid, position,
                                                 params.common.theta);
    output[(output_row * heads + head) * dimension + tid] = bfloat(rotated);
    if (!isfinite(rotated))
      flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
  }
}

// Each SIMD group owns one score at a time. This is the qualified dot-product
// order, with contiguous cache loads; only intermediate storage is moved to
// threadgroup memory. Selected K/V remain direct reads from their real cache.
inline void flash_qsa_fast_scores(
    threadgroup float *query, threadgroup float *scores,
    device const bfloat *key_cache, device const uint *selected,
    device atomic_uint *diagnostics, FlashQSAParams params,
    uint row, uint head, uint start, uint stop, uint lane, uint simd) {
  for (uint tile = start; tile < stop; tile += 8) {
    const uint slot = tile + simd;
    if (slot >= stop)
      continue;
    const uint token = flash_qsa_fast_token(selected, params, row, slot);
    float score = -INFINITY;
    if (token >= params.capacity || token > params.begin + row) {
      if (lane == 0)
        flash_qsa_fast_failure(diagnostics, flash_qsa_fast_position_failure);
    } else {
      float dot = 0.0f;
      const ulong base = (ulong(token) * 2 + head / 12) * 256;
      for (uint dimension = lane; dimension < 256; dimension += 32)
        dot += query[dimension] * float(key_cache[base + dimension]);
      score = simd_sum(dot) * 0.0625f;
      if (lane == 0 && !isfinite(score))
        flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
    }
    if (lane == 0)
      scores[slot - start] = score;
  }
}

inline void flash_qsa_fast_max_sum(
    threadgroup float *scores, uint count, threadgroup float *partial,
    threadgroup float *state, uint tid, uint lane, uint simd) {
  float maximum = -INFINITY;
  for (uint i = tid; i < count; i += 256)
    maximum = metal::max(maximum, scores[i]);
  maximum = simd_max(maximum);
  if (lane == 0)
    partial[simd] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    maximum = partial[0];
    for (uint i = 1; i < 8; ++i)
      maximum = metal::max(maximum, partial[i]);
    state[0] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  for (uint i = tid; i < count; i += 256)
    sum += metal::exp(scores[i] - state[0]);
  sum = simd_sum(sum);
  if (lane == 0)
    partial[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    sum = 0.0f;
    for (uint i = 0; i < 8; ++i)
      sum += partial[i];
    state[1] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void flash_qsa_fast_canonical(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *key_cache [[buffer(1)]],
    device const bfloat *value_cache [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device const bfloat *q_projection [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAFastParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= params.common.rows || head >= 24)
    return;
  threadgroup float query[256];
  threadgroup float scores[flash_qsa_fast_width];
  threadgroup bfloat probabilities[flash_qsa_fast_width];
  threadgroup float partial[8];
  threadgroup float state[2];
  const uint count = flash_qsa_fast_visible(params.common, row);
  query[tid] = float(queries[(ulong(row) * 24 + head) * 256 + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  flash_qsa_fast_scores(query, scores, key_cache, selected, diagnostics,
                         params.common, row, head, 0, count, lane, simd);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  flash_qsa_fast_max_sum(scores, count, partial, state, tid, lane, simd);
  if (tid == 0 && (!(state[1] > 0.0f) || !isfinite(state[1]) || !isfinite(state[0])))
    flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
  for (uint i = tid; i < count; i += 256)
    probabilities[i] = bfloat(metal::exp(scores[i] - state[0]) / state[1]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float accumulated = 0.0f;
  for (uint slot = 0; slot < count; ++slot) {
    const uint token = flash_qsa_fast_token(selected, params.common, row, slot);
    if (token >= params.common.capacity || token > params.common.begin + row) {
      accumulated = NAN;
      flash_qsa_fast_failure(diagnostics, flash_qsa_fast_position_failure);
      break;
    }
    accumulated += float(probabilities[slot]) *
        float(value_cache[(ulong(token) * 2 + head / 12) * 256 + tid]);
  }
  const bfloat gate = q_projection[ulong(row) * 12288 + head * 512 + 256 + tid];
  const bfloat gated = flash_qsa_fast_gate(bfloat(accumulated), gate);
  output[(ulong(row) * 24 + head) * 256 + tid] = gated;
  if (!isfinite(float(gated)) || !isfinite(float(gate)))
    flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
}

// Explicit F32-probability alternative, partitioned to expose enough groups
// for the 80-core M5 Ultra singleton. Partial numerators and denominators are
// kept F32; no BF16 probability cast occurs in this kernel.
inline void flash_qsa_fast_f32_partition_impl(
    device const bfloat *queries, device const bfloat *key_cache,
    device const bfloat *value_cache, device const uint *selected,
    device float *statistics, device float *numerators,
    device atomic_uint *diagnostics, FlashQSAFastParams params,
    uint3 group, uint tid, uint lane, uint simd,
    threadgroup float *query, threadgroup float *scores,
    threadgroup float *weights, threadgroup float *partial,
    threadgroup float *state) {
  const uint row = group.x, head = group.y, partition = group.z;
  if (row >= params.common.rows || head >= 24 || partition >= params.partitions)
    return;
  const uint visible = flash_qsa_fast_visible(params.common, row);
  const uint chunk = (visible + params.partitions - 1) / params.partitions;
  const uint start = metal::min(partition * chunk, visible);
  const uint stop = metal::min(start + chunk, visible);
  const uint count = stop - start;
  const ulong group_index = (ulong(row) * 24 + head) * params.maximum_partitions + partition;
  if (!count) {
    if (tid == 0) {
      statistics[group_index * 2] = -INFINITY;
      statistics[group_index * 2 + 1] = 0.0f;
    }
    numerators[group_index * 256 + tid] = 0.0f;
    return;
  }
  query[tid] = float(queries[(ulong(row) * 24 + head) * 256 + tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  flash_qsa_fast_scores(query, scores, key_cache, selected, diagnostics,
                         params.common, row, head, start, stop, lane, simd);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  flash_qsa_fast_max_sum(scores, count, partial, state, tid, lane, simd);
  for (uint i = tid; i < count; i += 256)
    weights[i] = metal::exp(scores[i] - state[0]);
  if (tid == 0) {
    statistics[group_index * 2] = state[0];
    statistics[group_index * 2 + 1] = state[1];
    if (!(state[1] > 0.0f) || !isfinite(state[1]) || !isfinite(state[0]))
      flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f;
  for (uint slot = start; slot < stop; ++slot) {
    const uint token = flash_qsa_fast_token(selected, params.common, row, slot);
    if (token >= params.common.capacity || token > params.common.begin + row) {
      numerator = NAN;
      flash_qsa_fast_failure(diagnostics, flash_qsa_fast_position_failure);
      break;
    }
    numerator += weights[slot - start] *
        float(value_cache[(ulong(token) * 2 + head / 12) * 256 + tid]);
  }
  numerators[group_index * 256 + tid] = numerator;
}

#define FLASH_QSA_F32_PARTITION_ENTRY(NAME, CAPACITY)                        \
  kernel void NAME(                                                        \
      device const bfloat *queries [[buffer(0)]],                          \
      device const bfloat *key_cache [[buffer(1)]],                         \
      device const bfloat *value_cache [[buffer(2)]],                       \
      device const uint *selected [[buffer(3)]],                           \
      device float *statistics [[buffer(4)]],                              \
      device float *numerators [[buffer(5)]],                              \
      device atomic_uint *diagnostics [[buffer(6)]],                       \
      constant FlashQSAFastParams &params [[buffer(7)]],                    \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]],                             \
      uint simd [[simdgroup_index_in_threadgroup]]) {                       \
    threadgroup float query[256];                                         \
    threadgroup float scores[CAPACITY];                                   \
    threadgroup float weights[CAPACITY];                                  \
    threadgroup float partial[8];                                         \
    threadgroup float state[2];                                           \
    flash_qsa_fast_f32_partition_impl(queries, key_cache, value_cache,       \
        selected, statistics, numerators, diagnostics, params, group, tid, \
        lane, simd, query, scores, weights, partial, state);                \
  }

// Preserve the original generic symbol for the already compiled v1 oracle.
FLASH_QSA_F32_PARTITION_ENTRY(flash_qsa_fast_f32_partition, 2051)
FLASH_QSA_F32_PARTITION_ENTRY(flash_qsa_fast_f32_partition_c2051, 2051)
FLASH_QSA_F32_PARTITION_ENTRY(flash_qsa_fast_f32_partition_c1026, 1026)
FLASH_QSA_F32_PARTITION_ENTRY(flash_qsa_fast_f32_partition_c513, 513)
FLASH_QSA_F32_PARTITION_ENTRY(flash_qsa_fast_f32_partition_c257, 257)

kernel void flash_qsa_fast_f32_reduce(
    device const float *statistics [[buffer(0)]],
    device const float *numerators [[buffer(1)]],
    device const bfloat *q_projection [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashQSAFastParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= params.common.rows || head >= 24)
    return;
  const ulong base = (ulong(row) * 24 + head) * params.maximum_partitions;
  float maximum = -INFINITY;
  for (uint p = 0; p < params.partitions; ++p)
    maximum = metal::max(maximum, statistics[(base + p) * 2]);
  float sum = 0.0f, value = 0.0f;
  for (uint p = 0; p < params.partitions; ++p) {
    const float local_sum = statistics[(base + p) * 2 + 1];
    if (local_sum == 0.0f)
      continue;
    const float factor = metal::exp(statistics[(base + p) * 2] - maximum);
    sum += local_sum * factor;
    value += numerators[(base + p) * 256 + tid] * factor;
  }
  const bfloat attention = bfloat(value / sum);
  const bfloat gate = q_projection[ulong(row) * 12288 + head * 512 + 256 + tid];
  const bfloat gated = flash_qsa_fast_gate(attention, gate);
  output[(ulong(row) * 24 + head) * 256 + tid] = gated;
  if (!(sum > 0.0f) || !isfinite(sum) || !isfinite(float(gated)) || !isfinite(float(gate)))
    flash_qsa_fast_failure(diagnostics, flash_qsa_fast_numeric_failure);
}
