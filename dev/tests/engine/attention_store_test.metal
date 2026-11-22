#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/attention_store.h"
#include "metal/kernels/common/lane_bindings.h"

// The attention tests' store of K/V they stage themselves, for attention
// tests whose K/V are no q/k projection's: each row of a dispatch's sequences
// (SplashAttentionParams) from staging, keys [KV head][row][dimension] and
// values [KV head][dimension][row] at each sequence's staging rows, into its
// page slot through the production store routine (splash_store_kv_slot), as
// the q/k projection stores the rows it makes. Group (KV head, row, sequence)
// stores both of the head's tensors; a row past its sequence's rows (its
// layout entry's) stores nothing. Bindings: the command's layout, the staged
// keys and values, a page table per sequence (SPLASH_LANE_BINDING), then the
// parameters.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
inline void splash_test_staged_store(device const SplashLaneRows *layout, device const bfloat *keys,
                                     device const bfloat *values, device const SplashKvPage *page_table,
                                     constant SplashAttentionParams &params, threadgroup float *maxima, uint3 group,
                                     uint thread_index, uint simd_lane, uint simd_group) {
  const uint head = group.x, row = group.y, s = group.z;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  if (head >= KVHeads || thread_index >= HeadDim || !splash_attention_rows_valid(params, sequence) ||
      row >= sequence.row_bound || row >= splash_attention_rows(layout, params, s).rows)
    return;
  const ulong staged = ulong(sequence.stage_row) * HeadDim;
  const bfloat key = keys[staged + (ulong(head) * sequence.chunk_stride + row) * HeadDim + thread_index];
  const bfloat value = values[staged + (ulong(head) * HeadDim + thread_index) * sequence.chunk_stride + row];
  splash_store_kv_slot<KVHeads, HeadDim, CacheElement, ValuesTokenMajor>(
      splash_store_page(page_table, sequence, row), sequence, params.kv, maxima, head, row, key, value, thread_index,
      simd_lane, simd_group);
}

#define TEST_STAGED_KV_STORE(Name, Heads, HeadDim, CacheElement, ValuesTokenMajor)                              \
  kernel void Name(device const SplashLaneRows *layout [[buffer(0)]], device const bfloat *keys [[buffer(1)]],  \
                   device const bfloat *values [[buffer(2)]],                                                    \
                   device const SplashKvPage *page_table0 [[buffer(3)]],                                         \
                   device const SplashKvPage *page_table1 [[buffer(4)]],                                         \
                   device const SplashKvPage *page_table2 [[buffer(5)]],                                         \
                   device const SplashKvPage *page_table3 [[buffer(6)]], device const SplashKvPage *page_table4 [[buffer(7)]], device const SplashKvPage *page_table5 [[buffer(8)]], device const SplashKvPage *page_table6 [[buffer(9)]], device const SplashKvPage *page_table7 [[buffer(10)]],                                         \
                   constant SplashAttentionParams &params [[buffer(11)]],                                         \
                   uint3 group [[threadgroup_position_in_grid]],                                                 \
                   uint thread_index [[thread_index_in_threadgroup]],                                            \
                   uint simd_lane [[thread_index_in_simdgroup]],                                                 \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                         \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                                  \
    splash_test_staged_store<Heads, HeadDim, CacheElement, ValuesTokenMajor>(                                    \
        layout, keys, values, SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7), \
        params, maxima, group, thread_index, simd_lane, simd_group);                                             \
  }
// By page format and the pages' KV heads, head dimension and value order.
TEST_STAGED_KV_STORE(test_staged_kv_store_q8, 4, 256, int8_t, false)
TEST_STAGED_KV_STORE(test_staged_kv_store_q8_kv2_g8, 2, 256, int8_t, false)
TEST_STAGED_KV_STORE(test_staged_kv_store_bf16, 4, 256, bfloat, false)
TEST_STAGED_KV_STORE(test_staged_kv_store_bf16_kv2_g8, 2, 256, bfloat, false)
TEST_STAGED_KV_STORE(test_staged_kv_store_q8_kv4_d128, 4, 128, int8_t, false)
TEST_STAGED_KV_STORE(test_staged_kv_store_bf16_kv4_d128, 4, 128, bfloat, false)
TEST_STAGED_KV_STORE(test_staged_kv_store_q8_kv2_vt, 2, 256, int8_t, true)
TEST_STAGED_KV_STORE(test_staged_kv_store_bf16_kv2_vt, 2, 256, bfloat, true)
#undef TEST_STAGED_KV_STORE
