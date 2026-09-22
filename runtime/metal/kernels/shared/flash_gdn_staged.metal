#include "metal/abi/FlashGDN.h"
#include <metal_stdlib>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline void gds_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

inline bool gds_valid(FlashGDNParams p, device atomic_uint &diagnostics) {
  const bool valid = p.rows && p.rows <= 2048 && p.lanes && p.lanes <= 32 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * 10240 * 2 &&
      p.convolution_lane_stride_bytes % 2 == 0 &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      p.recurrent_lane_stride_bytes % 4 == 0;
  if (!valid) gds_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

template <uint Values, uint Time>
inline void gds_recurrence(
    device const bfloat *mixed, device const float *decay,
    device const bfloat *beta, device float *recurrent,
    device bfloat *output, device atomic_uint &diagnostics,
    constant FlashGDNParams &p, uint3 group, uint thread_index,
    uint lane, uint simd_group, threadgroup bfloat *queries,
    threadgroup bfloat *keys, threadgroup bfloat *values,
    threadgroup float *decays, threadgroup bfloat *betas) {
  if (!gds_valid(p, diagnostics) || group.x >= 48 || group.y >= 128 / Values ||
      group.z >= p.lanes) return;
  constexpr uint Threads = Values * 32;
  constexpr uint KeyStride = 136;
  constexpr uint ValueStride = Values + 8;
  const uint head = group.x, batch = group.z;
  const uint key_head = head / 3;
  const uint value_begin = group.y * Values;
  const uint dimension = value_begin + simd_group;
  const ulong state_base = ulong(batch) * p.recurrent_lane_stride_bytes / 4 +
                           (head * 128 + dimension) * 128 + 4 * lane;
  float state[4];
  for (uint i = 0; i < 4; ++i) state[i] = recurrent[state_base + i];
  for (uint begin = 0; begin < p.rows; begin += Time) {
    const uint count = min(Time, p.rows - begin);
    // Every key/query element is read once per value block, rather than once
    // per value dimension. Cooperative loads stay coalesced along K.
    for (uint item = thread_index; item < count * 128; item += Threads) {
      const uint token = item / 128, key_dimension = item % 128;
      const ulong source = (ulong(batch) * p.rows + begin + token) * 10240 +
                            key_head * 128 + key_dimension;
      queries[token * KeyStride + key_dimension] = mixed[source];
      keys[token * KeyStride + key_dimension] = mixed[source + 2048];
    }
    for (uint item = thread_index; item < count * Values; item += Threads) {
      const uint token = item / Values, value_dimension = item % Values;
      const ulong source = (ulong(batch) * p.rows + begin + token) * 10240 +
                            4096 + head * 128 + value_begin + value_dimension;
      values[token * ValueStride + value_dimension] = mixed[source];
    }
    for (uint token = thread_index; token < count; token += Threads) {
      const ulong source = (ulong(batch) * p.rows + begin + token) * 48 + head;
      decays[token] = decay[source]; betas[token] = beta[source];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Token order and scalar accumulation order deliberately match the
    // qualified recurrence. Only the address space of q/k/v/gates changes.
    for (uint token = 0; token < count; ++token) {
      float q[4], k[4];
      float memory = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        q[i] = float(queries[token * KeyStride + 4 * lane + i]);
        k[i] = float(keys[token * KeyStride + 4 * lane + i]);
        state[i] = state[i] * decays[token];
        memory += state[i] * k[i];
      }
      memory = simd_sum(memory);
      const float value = float(values[token * ValueStride + simd_group]);
      const float delta = (value - memory) * float(betas[token]);
      float result = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[i] = state[i] + k[i] * delta;
        result += state[i] * q[i];
      }
      result = simd_sum(result);
      if (lane == 0) {
        output[(ulong(batch) * p.rows + begin + token) * 6144 +
                head * 128 + dimension] = bfloat(result);
        if (!isfinite(result)) gds_flag(diagnostics, FlashGDNNonFinite);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 4; ++i) {
    recurrent[state_base + i] = state[i];
    if (!isfinite(state[i])) gds_flag(diagnostics, FlashGDNNonFinite);
  }
}

#define GDS_ENTRY(NAME, V, T)                                               \
  kernel void NAME(                                                        \
      device const bfloat *mixed [[buffer(0)]],                             \
      device const float *decay [[buffer(1)]],                              \
      device const bfloat *beta [[buffer(2)]],                              \
      device float *recurrent [[buffer(3)]],                               \
      device bfloat *output [[buffer(4)]],                                  \
      device atomic_uint &diagnostics [[buffer(5)]],                        \
      constant FlashGDNParams &p [[buffer(6)]],                              \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint thread_index [[thread_index_in_threadgroup]],                    \
      uint lane [[thread_index_in_simdgroup]],                              \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup bfloat queries[T * 136], keys[T * 136];                     \
    threadgroup bfloat values[T * (V + 8)], betas[T];                       \
    threadgroup float decays[T];                                           \
    gds_recurrence<V, T>(mixed, decay, beta, recurrent, output, diagnostics, \
        p, group, thread_index, lane, simd_group, queries, keys, values,     \
        decays, betas);                                                     \
  }

GDS_ENTRY(flash_gdn_staged_v8_t16, 8, 16)
GDS_ENTRY(flash_gdn_staged_v8_t32, 8, 32)
GDS_ENTRY(flash_gdn_staged_v16_t16, 16, 16)
GDS_ENTRY(flash_gdn_staged_v16_t32, 16, 32)

#undef GDS_ENTRY
