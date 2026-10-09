#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_store_row.h"

template <uint KVHeads, typename CacheElement>
inline void splash_store_verify_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
    device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
    constant SplashChunkedPrefillParams *params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint GroupsPerLane = 2 * Rows * KVHeads;
  uint batch = group / GroupsPerLane;
  uint local_group = group % GroupsPerLane;
  constant SplashChunkedPrefillParams &lane_params = params[batch];
  if (!splash_chunk_contract_valid(lane_params) ||
      lane_params.chunk_tokens != Rows ||
      lane_params.chunk_stride != SPLASH_VERIFY_CHUNK_STRIDE ||
      thread_index >= SplashKvHeadDimension)
    return;
  device const SplashKvPage *page_table =
      SPLASH_LANE_BINDING(batch, page_table0, page_table1, page_table2, page_table3);
  constexpr ulong lane_tensor_stride =
      ulong(KVHeads) * SPLASH_VERIFY_CHUNK_STRIDE * SplashKvHeadDimension;
  chunk_keys += batch * lane_tensor_stride;
  chunk_values += batch * lane_tensor_stride;

  uint rows = Rows * KVHeads;
  bool value_tensor = local_group >= rows;
  uint row = value_tensor ? local_group - rows : local_group;
  uint chunk_token = row % Rows;
  uint head = row / Rows;
  splash_store_kv_row<KVHeads, CacheElement>(
      chunk_keys, chunk_values, page_table, lane_params, maxima, value_tensor,
      head, chunk_token, thread_index, simd_lane, simd_group);
}

#define PAGED_VERIFY_STORE(Name, Heads, CacheElement)                          \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const SplashKvPage *page_table0 [[buffer(2)]],                    \
      device const SplashKvPage *page_table1 [[buffer(3)]],                    \
      device const SplashKvPage *page_table2 [[buffer(4)]],                    \
      device const SplashKvPage *page_table3 [[buffer(5)]],                    \
      constant SplashChunkedPrefillParams *params [[buffer(6)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    SPLASH_STORE_MAXIMA_##CacheElement                                         \
    splash_store_verify_phase<Heads, CacheElement>(                            \
        chunk_keys, chunk_values, page_table0, page_table1, page_table2,       \
        page_table3, params, maxima, group, thread_index, simd_lane,           \
        simd_group);                                                           \
  }
PAGED_VERIFY_STORE(verify_attention_q8_store, 4, int8_t)
PAGED_VERIFY_STORE(verify_attention_q8_store_kv2_g8, 2, int8_t)
PAGED_VERIFY_STORE(verify_attention_bf16_store, 4, bfloat)
PAGED_VERIFY_STORE(verify_attention_bf16_store_kv2_g8, 2, bfloat)
#undef PAGED_VERIFY_STORE
