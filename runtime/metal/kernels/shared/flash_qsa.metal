#include "metal/abi/FlashQSA.h"
#include <metal_stdlib>

using namespace metal;

// Preserve BF16 array boundaries in the inspected MLX implementation.
// Qwen4's Metal RoPE itself consumes BF16 normalized values but performs the
// rotary products in F32 before one BF16 storage cast.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint flash_qsa_budget = 512;
constant uint flash_qsa_token_width = 2051;
constant uint flash_qsa_numeric_failure = 1u << 8;
constant uint flash_qsa_position_failure = 1u << 9;

inline void flash_qsa_failure(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}

inline float flash_qsa_inverse(threadgroup float *partial, float square_sum,
                                uint dimension, float epsilon, uint tid,
                                uint lane, uint simd) {
  const float total = simd_sum(square_sum);
  if (lane == 0)
    partial[simd] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < 8; ++i)
      sum += partial[i];
    partial[0] = precise::rsqrt(sum / float(dimension) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partial[0];
}

inline float flash_qsa_rotate(threadgroup float *normalized, uint column,
                               long position, float theta) {
  if (column >= 64)
    return normalized[column];
  const uint pair = column & 31u;
  const float inverse_frequency = 1.0f / metal::pow(theta, float(pair) / 32.0f);
  const float angle = float(position) * inverse_frequency;
  const float c = metal::cos(angle), s = metal::sin(angle);
  return column < 32 ? normalized[column] * c - normalized[column + 32] * s
                     : normalized[column] * c + normalized[column - 32] * s;
}

template <class Weight>
inline void flash_qsa_norm_rope_impl(
    device const bfloat *input, device const Weight *weight,
    device const long *positions, device bfloat *output,
    device atomic_uint *diagnostics, FlashQSAParams params, uint2 group,
    uint tid, uint lane, uint simd, threadgroup float *normalized,
    threadgroup float *partial) {
  const uint row = group.x, head = group.y;
  const uint dimension = params.mode == 2 ? 128 : 256;
  const uint heads = params.mode == 0 ? 24 : (params.mode == 1 ? 2 : 4);
  if (row >= params.rows || head >= heads)
    return;
  const uint input_stride = params.mode == 0 ? 12288 : (params.mode == 1 ? 512 : 640);
  const uint head_stride = params.mode == 0 ? 512 : dimension;
  const ulong source = ulong(row) * input_stride + head * head_stride;
  const long position = params.positions_supplied ? positions[row]
                                                 : long(params.begin) + row;
  float value = tid < dimension ? float(input[source + tid]) : 0.0f;
  if (!isfinite(value))
    flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  const float inverse = flash_qsa_inverse(partial, value * value, dimension,
                                           params.epsilon, tid, lane, simd);
  if (tid < dimension) {
    const float raw_weight = float(weight[tid]);
    const float scale = params.norm_convention == 0 ? 1.0f + raw_weight : raw_weight;
    normalized[tid] = float(bfloat((value * inverse) * scale));
    if (!isfinite(normalized[tid]))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
  if (tid == 0 && (position < 0 || position > 2147483647L))
    flash_qsa_failure(diagnostics, flash_qsa_position_failure);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < dimension) {
    const ulong destination_row = params.mode == 1 ? ulong(params.begin) + row : row;
    const ulong destination = (destination_row * heads + head) * dimension + tid;
    const float rotated = flash_qsa_rotate(normalized, tid, position, params.theta);
    output[destination] = bfloat(rotated);
    if (!isfinite(rotated))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
}

#define FLASH_QSA_NORM_ENTRY(NAME, WEIGHT)                                   \
  kernel void NAME(                                                        \
      device const bfloat *input [[buffer(0)]],                            \
      device const WEIGHT *weight [[buffer(1)]],                           \
      device const long *positions [[buffer(2)]],                          \
      device bfloat *output [[buffer(3)]],                                \
      device atomic_uint *diagnostics [[buffer(4)]],                       \
      constant FlashQSAParams &params [[buffer(5)]],                       \
      uint2 group [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]],                             \
      uint simd [[simdgroup_index_in_threadgroup]]) {                       \
    threadgroup float normalized[256];                                    \
    threadgroup float partial[8];                                         \
    flash_qsa_norm_rope_impl(input, weight, positions, output, diagnostics, \
                             params, group, tid, lane, simd, normalized,   \
                             partial);                                    \
  }

FLASH_QSA_NORM_ENTRY(flash_qsa_norm_rope_bf16, bfloat)
FLASH_QSA_NORM_ENTRY(flash_qsa_norm_rope_f32, float)

kernel void flash_qsa_append_aux(
    device const bfloat *values [[buffer(0)]],
    device const bfloat *index_projection [[buffer(1)]],
    device const long *positions [[buffer(2)]],
    device bfloat *value_cache [[buffer(3)]],
    device bfloat *raw_index_cache [[buffer(4)]],
    device long *index_positions [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAParams &params [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (row >= params.rows)
    return;
  const ulong position = ulong(params.begin) + row;
  for (uint dimension = tid; dimension < 512; dimension += 256) {
    const bfloat value = values[ulong(row) * 512 + dimension];
    value_cache[position * 512 + dimension] = value;
    if (!isfinite(float(value)))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
  if (tid < 128) {
    const bfloat value = index_projection[ulong(row) * 640 + 512 + tid];
    raw_index_cache[position * 128 + tid] = value;
    if (!isfinite(float(value)))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
  if (tid == 0) {
    const long rotary_position = params.positions_supplied ? positions[row] : long(position);
    index_positions[position] = rotary_position;
    if (rotary_position < 0 || rotary_position > 2147483647L)
      flash_qsa_failure(diagnostics, flash_qsa_position_failure);
  }
}

template <class Weight>
inline void flash_qsa_pool_rope_impl(
    device const bfloat *raw_keys, device const Weight *weight,
    device const long *positions, device bfloat *pooled,
    device atomic_uint *diagnostics, FlashQSAParams params, uint group,
    uint tid, uint lane, uint simd, threadgroup float *normalized,
    threadgroup float *partial) {
  if (group >= params.new_blocks)
    return;
  const uint block = params.first_block + group;
  const ulong start = ulong(block) * 4;
  float value = 0.0f;
  if (tid < 128) {
    for (uint token = 0; token < 4; ++token)
      value += float(raw_keys[(start + token) * 128 + tid]);
    value = float(bfloat(value / 4.0f));
  }
  const float inverse = flash_qsa_inverse(partial, value * value, 128,
                                           params.epsilon, tid, lane, simd);
  if (tid < 128) {
    const float raw_weight = float(weight[tid]);
    const float scale = params.norm_convention == 0 ? 1.0f + raw_weight : raw_weight;
    normalized[tid] = float(bfloat((value * inverse) * scale));
    if (!isfinite(normalized[tid]))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128) {
    const float rotated = flash_qsa_rotate(normalized, tid, positions[start], params.theta);
    pooled[ulong(block) * 128 + tid] = bfloat(rotated);
    if (!isfinite(rotated))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
}

#define FLASH_QSA_POOL_ENTRY(NAME, WEIGHT)                                  \
  kernel void NAME(                                                       \
      device const bfloat *raw_keys [[buffer(0)]],                         \
      device const WEIGHT *weight [[buffer(1)]],                           \
      device const long *positions [[buffer(2)]],                          \
      device bfloat *pooled [[buffer(3)]],                                 \
      device atomic_uint *diagnostics [[buffer(4)]],                       \
      constant FlashQSAParams &params [[buffer(5)]],                       \
      uint group [[threadgroup_position_in_grid]],                        \
      uint tid [[thread_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]],                             \
      uint simd [[simdgroup_index_in_threadgroup]]) {                       \
    threadgroup float normalized[128];                                    \
    threadgroup float partial[8];                                         \
    flash_qsa_pool_rope_impl(raw_keys, weight, positions, pooled,           \
                             diagnostics, params, group, tid, lane, simd, \
                             normalized, partial);                        \
  }

FLASH_QSA_POOL_ENTRY(flash_qsa_pool_rope_bf16, bfloat)
FLASH_QSA_POOL_ENTRY(flash_qsa_pool_rope_f32, float)

kernel void flash_qsa_index_scores(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *pooled [[buffer(1)]],
    device float *scores [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant FlashQSAParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.y, block = group.x * 8 + simd;
  const uint completed = (params.begin + row + 1) / 4;
  if (row >= params.rows || completed <= flash_qsa_budget || block >= completed)
    return;
  float score = 0.0f;
  for (uint head = 0; head < 4; ++head) {
    float dot = 0.0f;
    for (uint dimension = lane; dimension < 128; dimension += 32)
      dot += float(queries[(ulong(row) * 4 + head) * 128 + dimension]) *
             float(pooled[ulong(block) * 128 + dimension]);
    score += metal::max(simd_sum(dot), 0.0f);
  }
  if (lane == 0) {
    score *= 0.08838834764831845f; // 1 / sqrt(128)
    scores[ulong(row) * params.block_capacity + block] = score;
    if (!isfinite(score))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
}

// Exact FP32 top512. Four radix-byte histograms identify the cutoff without
// rounding scores or allocating an index sheet. A segmented chronological
// scan retains the highest block IDs when exact ties cross the cutoff, the
// same policy as the checkpoint's argpartition implementation.
kernel void flash_qsa_select_blocks(
    device const float *scores [[buffer(0)]],
    device uint *selected [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashQSAParams &params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (row >= params.rows)
    return;
  const uint completed = (params.begin + row + 1) / 4;
  device uint *out = selected + ulong(row) * flash_qsa_budget;
  if (completed <= flash_qsa_budget) {
    for (uint i = tid; i < flash_qsa_budget; i += 256)
      out[i] = i < completed ? i : 0xffffffffu;
    return;
  }
  threadgroup atomic_uint histogram[256];
  threadgroup uint cutoff[2];
  threadgroup uint greater[256];
  threadgroup uint ties[256];
  threadgroup uint offsets[256];
  threadgroup uint totals[2];
  if (tid == 0) {
    cutoff[0] = 0;
    cutoff[1] = 0;
  }
  const uint segment = (completed + 255) / 256;
  const uint start = metal::min(tid * segment, completed);
  const uint stop = metal::min(start + segment, completed);
  const device float *input = scores + ulong(row) * params.block_capacity;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int shift = 24; shift >= 0; shift -= 8) {
    atomic_store_explicit(&histogram[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint prefix = cutoff[0];
    for (uint i = start; i < stop; ++i) {
      const float source = input[i];
      const uint key = isfinite(source) && source >= 0.0f ? as_type<uint>(source) : 0;
      if (shift == 24 || (key >> uint(shift + 8)) == (prefix >> uint(shift + 8)))
        atomic_fetch_add_explicit(&histogram[(key >> uint(shift)) & 255u],
                                  1, memory_order_relaxed);
      if (!isfinite(source) || source < 0.0f)
        flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint already_greater = cutoff[1];
      for (int byte = 255; byte >= 0; --byte) {
        const uint count = atomic_load_explicit(&histogram[byte], memory_order_relaxed);
        if (already_greater + count >= flash_qsa_budget) {
          cutoff[0] |= uint(byte) << uint(shift);
          cutoff[1] = already_greater;
          break;
        }
        already_greater += count;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint local_greater = 0, local_ties = 0;
  for (uint i = start; i < stop; ++i) {
    const uint key = as_type<uint>(input[i]);
    local_greater += key > cutoff[0];
    local_ties += key == cutoff[0];
  }
  greater[tid] = local_greater;
  ties[tid] = local_ties;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint greater_count = 0, tie_count = 0;
    for (uint i = 0; i < 256; ++i) {
      greater_count += greater[i];
      const uint count = ties[i];
      ties[i] = tie_count;
      tie_count += count;
    }
    totals[0] = tie_count - (flash_qsa_budget - greater_count); // ties to skip
    totals[1] = greater_count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint skipped_here = metal::min(local_ties,
      totals[0] > ties[tid] ? totals[0] - ties[tid] : 0u);
  offsets[tid] = local_greater + local_ties - skipped_here;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint offset = 0;
    for (uint i = 0; i < 256; ++i) {
      const uint count = offsets[i];
      offsets[i] = offset;
      offset += count;
    }
    if (offset != flash_qsa_budget)
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint ordinal = ties[tid], destination = offsets[tid];
  for (uint i = start; i < stop; ++i) {
    const uint key = as_type<uint>(input[i]);
    bool take = key > cutoff[0];
    if (key == cutoff[0])
      take = ordinal++ >= totals[0];
    if (take && destination < flash_qsa_budget)
      out[destination++] = i;
  }
}

inline uint flash_qsa_visible_width(FlashQSAParams params, uint row) {
  const uint visible = params.begin + row + 1;
  return metal::min(visible / 4, flash_qsa_budget) * 4 + visible % 4;
}

inline uint flash_qsa_token(device const uint *selected, FlashQSAParams params,
                             uint row, uint candidate) {
  const uint completed = (params.begin + row + 1) / 4;
  const uint picked = metal::min(completed, flash_qsa_budget);
  return candidate < picked * 4
      ? selected[ulong(row) * flash_qsa_budget + candidate / 4] * 4 + candidate % 4
      : completed * 4 + candidate - picked * 4;
}

kernel void flash_qsa_main_scores(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *key_cache [[buffer(1)]],
    device const uint *selected [[buffer(2)]],
    device float *scores [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashQSAParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.y, head = group.z, candidate = group.x * 8 + simd;
  if (row >= params.rows || head >= 24 || candidate >= flash_qsa_token_width)
    return;
  float score = -INFINITY;
  if (candidate < flash_qsa_visible_width(params, row)) {
    const uint token = flash_qsa_token(selected, params, row, candidate);
    if (token >= params.capacity || token > params.begin + row) {
      if (lane == 0)
        flash_qsa_failure(diagnostics, flash_qsa_position_failure);
    } else {
      float dot = 0.0f;
      const ulong q_base = (ulong(row) * 24 + head) * 256;
      const ulong k_base = (ulong(token) * 2 + head / 12) * 256;
      for (uint dimension = lane; dimension < 256; dimension += 32)
        dot += float(queries[q_base + dimension]) * float(key_cache[k_base + dimension]);
      score = simd_sum(dot) * 0.0625f; // 1 / sqrt(256)
      if (lane == 0 && !isfinite(score))
        flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
    }
  }
  if (lane == 0)
    scores[(ulong(row) * 24 + head) * flash_qsa_token_width + candidate] = score;
}

kernel void flash_qsa_probabilities(
    device const float *scores [[buffer(0)]],
    device bfloat *probabilities [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashQSAParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= params.rows || head >= 24)
    return;
  threadgroup float partial[8];
  threadgroup float state[2];
  const ulong base = (ulong(row) * 24 + head) * flash_qsa_token_width;
  const uint width = flash_qsa_visible_width(params, row);
  float maximum = -INFINITY;
  for (uint i = tid; i < width; i += 256)
    maximum = metal::max(maximum, scores[base + i]);
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
  for (uint i = tid; i < width; i += 256)
    sum += metal::exp(scores[base + i] - state[0]);
  sum = simd_sum(sum);
  if (lane == 0)
    partial[simd] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    sum = 0.0f;
    for (uint i = 0; i < 8; ++i)
      sum += partial[i];
    state[1] = sum;
    if (!(sum > 0.0f) || !isfinite(sum) || !isfinite(state[0]))
      flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < flash_qsa_token_width; i += 256)
    probabilities[base + i] = i < width
        ? bfloat(metal::exp(scores[base + i] - state[0]) / state[1]) : bfloat(0.0f);
}

kernel void flash_qsa_values_gate(
    device const bfloat *probabilities [[buffer(0)]],
    device const bfloat *value_cache [[buffer(1)]],
    device const uint *selected [[buffer(2)]],
    device const bfloat *q_projection [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashQSAParams &params [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint dimension [[thread_index_in_threadgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= params.rows || head >= 24 || dimension >= 256)
    return;
  const ulong p_base = (ulong(row) * 24 + head) * flash_qsa_token_width;
  const uint width = flash_qsa_visible_width(params, row);
  float accumulated = 0.0f;
  for (uint candidate = 0; candidate < width; ++candidate) {
    const uint token = flash_qsa_token(selected, params, row, candidate);
    if (token >= params.capacity || token > params.begin + row) {
      flash_qsa_failure(diagnostics, flash_qsa_position_failure);
      accumulated = NAN;
      break;
    }
    accumulated += float(probabilities[p_base + candidate]) *
                   float(value_cache[(ulong(token) * 2 + head / 12) * 256 + dimension]);
  }
  const bfloat attention = bfloat(accumulated);
  const float gate = float(q_projection[ulong(row) * 12288 + head * 512 + 256 + dimension]);
  // QSA calls standalone mx.sigmoid(BF16), whose GPU unary contract retains
  // BF16 exp/add/reciprocal/subtraction stages and uses precise exp. Forming
  // the sigmoid in F32 and casting only its result changes the query gate.
  const bfloat exponential = bfloat(metal::precise::exp(metal::abs(gate)));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  const bfloat activated_gate = bfloat(gate) < bfloat(0.0f)
                                    ? tail : bfloat(1.0f) - tail;
  const bfloat gated = attention * activated_gate;
  output[(ulong(row) * 24 + head) * 256 + dimension] = gated;
  if (!isfinite(float(gated)) || !isfinite(gate))
    flash_qsa_failure(diagnostics, flash_qsa_numeric_failure);
}
