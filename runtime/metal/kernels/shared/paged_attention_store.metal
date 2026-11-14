#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"

// Writes every current row of a dispatch's sequences (SplashAttentionParams)
// directly into its final page slot. A decode step writes all its rows;
// acceptance is represented solely by the host-visible committed length, and
// a rejected suffix stays unreachable until the next command starting at the
// same logical position overwrites it.

// The scratch of a store entry's scale reduction, by CacheElement: a maximum
// per simdgroup of the HeadDim threads for INT8; BF16 copies the source bits
// and has none.
#define SPLASH_STORE_MAXIMA_int8_t(HeadDim) threadgroup float maxima[(HeadDim) / 32];
#define SPLASH_STORE_MAXIMA_bfloat(HeadDim) threadgroup float *maxima = nullptr;

// One lane per dimension stores row `row` of a sequence's staged keys or
// values, KV head `head`, in its final page slot. INT8 derives a per-row
// scale; BF16 copies the original bits. Slots are addressed inside the head's
// slab of the page: the page index functions at page zero and head zero.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values, device const SplashKvPage *page_table,
    constant SplashAttentionSequence &sequence, SplashKvLayer kv, threadgroup float *maxima, bool value_tensor,
    uint head, uint row, uint dimension, uint simd_lane, uint simd_group) {
  const uint logical_token = sequence.committed_tokens + row;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, HeadDim, CacheElement>(kv, head).page(page);
  const uint page_token = logical_token % SplashKvPageTokens;
  // Keys are [KV head][row][dimension], values [KV head][dimension][row].
  const ulong staged = ulong(sequence.stage_row) * HeadDim;
  const bfloat source =
      value_tensor
          ? chunk_values[staged + (ulong(head) * HeadDim + dimension) * sequence.chunk_stride + row]
          : chunk_keys[staged + (ulong(head) * sequence.chunk_stride + row) * HeadDim + dimension];

  if constexpr (is_same<CacheElement, bfloat>::value) {
    if (value_tensor)
      slab.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, page_token, dimension)] = source;
    else
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = source;
    return;
  } else {

    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = HeadDim / 64; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * 127.0f / maximum)), -127, 127);

    if (value_tensor) {
      slab.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else {
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  }
}

// A threadgroup of HeadDim threads stores one row of one KV head: the keys'
// rows first, then the values', each KV head by KV head over the dispatch's
// rows. A row's sequence names its page table among the bindings.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
inline void splash_store_phase(device const bfloat *chunk_keys, device const bfloat *chunk_values,
                               device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
                               device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
                               constant SplashAttentionParams &params, threadgroup float *maxima, uint group,
                               uint thread_index, uint simd_lane, uint simd_group) {
  const uint rows = params.rows * KVHeads;
  if (group >= 2 * rows || thread_index >= HeadDim)
    return;
  const bool value_tensor = group >= rows;
  const uint head_row = value_tensor ? group - rows : group;
  const uint row = head_row % params.rows;
  const uint s = splash_attention_sequence_of_row(params, row);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  if (!splash_attention_rows_valid(params, sequence) || row - sequence.first_row >= sequence.rows)
    return;
  splash_store_kv_row<KVHeads, HeadDim, CacheElement, ValuesTokenMajor>(
      chunk_keys, chunk_values, SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3),
      sequence, params.kv, maxima, value_tensor, head_row / params.rows, row - sequence.first_row, thread_index,
      simd_lane, simd_group);
}

#define PAGED_ATTENTION_STORE(Name, Heads, HeadDim, CacheElement, ValuesTokenMajor)                              \
  kernel void Name(device const bfloat *chunk_keys [[buffer(0)]], device const bfloat *chunk_values [[buffer(1)]], \
                   device const SplashKvPage *page_table0 [[buffer(2)]],                                         \
                   device const SplashKvPage *page_table1 [[buffer(3)]],                                         \
                   device const SplashKvPage *page_table2 [[buffer(4)]],                                         \
                   device const SplashKvPage *page_table3 [[buffer(5)]],                                         \
                   constant SplashAttentionParams &params [[buffer(6)]],                                         \
                   uint group [[threadgroup_position_in_grid]],                                                  \
                   uint thread_index [[thread_index_in_threadgroup]],                                            \
                   uint simd_lane [[thread_index_in_simdgroup]],                                                 \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                         \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                                  \
    splash_store_phase<Heads, HeadDim, CacheElement, ValuesTokenMajor>(                                          \
        chunk_keys, chunk_values, page_table0, page_table1, page_table2, page_table3, params, maxima, group,     \
        thread_index, simd_lane, simd_group);                                                                    \
  }
PAGED_ATTENTION_STORE(attention_q8_store, 4, 256, int8_t, false)
PAGED_ATTENTION_STORE(attention_q8_store_kv2_g8, 2, 256, int8_t, false)
PAGED_ATTENTION_STORE(attention_bf16_store, 4, 256, bfloat, false)
PAGED_ATTENTION_STORE(attention_bf16_store_kv2_g8, 2, 256, bfloat, false)
// Four KV heads of 128 dimensions.
PAGED_ATTENTION_STORE(attention_q8_store_kv4_d128, 4, 128, int8_t, false)
PAGED_ATTENTION_STORE(attention_bf16_store_kv4_d128, 4, 128, bfloat, false)
// Two KV heads whose values are token-major (sparse attention).
PAGED_ATTENTION_STORE(attention_q8_store_kv2_vt, 2, 256, int8_t, true)
PAGED_ATTENTION_STORE(attention_bf16_store_kv2_vt, 2, 256, bfloat, true)
#undef PAGED_ATTENTION_STORE
