#include "metal/kernels/common/paged_store_row.h"

// Writes every current row directly into its final page slot. Decode writes
// all speculative rows; acceptance is represented solely by the host-visible
// committed length. A rejected suffix remains unreachable and is overwritten
// by the next command starting at the same logical position.
template <uint KVHeads, typename CacheElement>
inline void splash_store_chunk_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  uint rows = params.chunk_tokens * KVHeads;
  if (!splash_chunk_contract_valid(params) || group >= 2 * rows ||
      thread_index >= SplashKvHeadDimension)
    return;

  bool value_tensor = group >= rows;
  uint row = value_tensor ? group - rows : group;
  uint chunk_token = row % params.chunk_tokens;
  uint head = row / params.chunk_tokens;
  splash_store_kv_row<KVHeads, CacheElement>(
      chunk_keys, chunk_values, page_table, params, maxima, value_tensor, head,
      chunk_token, thread_index, simd_lane, simd_group);
}

#define PAGED_PREFILL_STORE(Name, Heads, CacheElement)                         \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const SplashKvPage *page_table [[buffer(2)]],                     \
      constant SplashChunkedPrefillParams &params [[buffer(3)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    SPLASH_STORE_MAXIMA_##CacheElement                                         \
    splash_store_chunk_phase<Heads, CacheElement>(                             \
        chunk_keys, chunk_values, page_table, params, maxima, group,           \
        thread_index, simd_lane, simd_group);                                  \
  }
PAGED_PREFILL_STORE(prefill_attention_q8_store, 4, int8_t)
PAGED_PREFILL_STORE(prefill_attention_q8_store_kv2_g8, 2, int8_t)
PAGED_PREFILL_STORE(prefill_attention_bf16_store, 4, bfloat)
PAGED_PREFILL_STORE(prefill_attention_bf16_store_kv2_g8, 2, bfloat)
#undef PAGED_PREFILL_STORE
