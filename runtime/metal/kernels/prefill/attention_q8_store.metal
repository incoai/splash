#include "metal/kernels/common/paged_store_row.h"

// Writes every current row directly into its final Q8 page slot. Decode writes
// all speculative rows; acceptance is represented solely by the host-visible
// committed length. A rejected suffix remains unreachable and is overwritten
// by the next command starting at the same logical position.
template <uint KVHeads, typename CacheElement>
inline void splash_q8_store_chunk_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device CacheElement *cache_keys, device float *key_scales_buffer,
    device CacheElement *cache_values, device float *value_scales_buffer,
    device const uint *page_table,
    constant SplashChunkedPrefillParams &params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  uint rows = params.chunk_tokens * KVHeads;
  if (!splash_chunk_contract_valid(params) || group >= 2 * rows ||
      thread_index >= SplashQ8HeadDimension)
    return;

  bool value_tensor = group >= rows;
  uint row = value_tensor ? group - rows : group;
  uint chunk_token = row % params.chunk_tokens;
  uint head = row / params.chunk_tokens;
  splash_store_kv_row<KVHeads>(chunk_keys, chunk_values, cache_keys,
                                 key_scales_buffer, cache_values, value_scales_buffer,
                                 page_table, params, maxima, value_tensor, head,
                                 chunk_token, thread_index, simd_lane,
                                 simd_group);
}

kernel void
prefill_attention_q8_store(device const bfloat *chunk_keys [[buffer(0)]],
                        device const bfloat *chunk_values [[buffer(1)]],
                        device char *cache_keys [[buffer(2)]],
                        device float *key_scales_buffer [[buffer(3)]],
                        device char *cache_values [[buffer(4)]],
                        device float *value_scales_buffer [[buffer(5)]],
                        device const uint *page_table [[buffer(6)]],
                        constant SplashChunkedPrefillParams &params
                        [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_q8_store_chunk_phase<4>(
      chunk_keys, chunk_values, cache_keys, key_scales_buffer, cache_values,
      value_scales_buffer, page_table, params, maxima, group, thread_index,
      simd_lane, simd_group);
}

kernel void prefill_attention_q8_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device char *cache_keys [[buffer(2)]],
    device float *key_scales_buffer [[buffer(3)]],
    device char *cache_values [[buffer(4)]],
    device float *value_scales_buffer [[buffer(5)]],
    device const uint *page_table [[buffer(6)]],
    constant SplashChunkedPrefillParams &params [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_q8_store_chunk_phase<2>(
      chunk_keys, chunk_values, cache_keys, key_scales_buffer, cache_values,
      value_scales_buffer, page_table, params, maxima, group, thread_index,
      simd_lane, simd_group);
}

// BF16 entries have no scale storage or scale arguments.

kernel void
prefill_attention_bf16_store(device const bfloat *chunk_keys [[buffer(0)]],
                        device const bfloat *chunk_values [[buffer(1)]],
                        device bfloat *cache_keys [[buffer(2)]],
                        device bfloat *cache_values [[buffer(3)]],
                        device const uint *page_table [[buffer(4)]],
                        constant SplashChunkedPrefillParams &params
                        [[buffer(5)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  splash_q8_store_chunk_phase<4>(
      chunk_keys, chunk_values, cache_keys, nullptr, cache_values,
      nullptr, page_table, params, nullptr, group, thread_index,
      simd_lane, simd_group);
}

kernel void prefill_attention_bf16_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device bfloat *cache_keys [[buffer(2)]],
    device bfloat *cache_values [[buffer(3)]],
    device const uint *page_table [[buffer(4)]],
    constant SplashChunkedPrefillParams &params [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  splash_q8_store_chunk_phase<2>(
      chunk_keys, chunk_values, cache_keys, nullptr, cache_values,
      nullptr, page_table, params, nullptr, group, thread_index,
      simd_lane, simd_group);
}
