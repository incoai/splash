#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"

// Writes every current row of a dispatch's sequences (SplashAttentionParams)
// directly into its page slot. A decode step writes all its rows, a lane's at
// the slots of their rows of its block; its commit moves the accepted path's
// rows to the slots of their depths (splash_path_commit_phase), and the rows
// past them stay unreachable until the next command starting at the same
// logical position overwrites them.

// The scratch of a store entry's scale reduction, by CacheElement: a maximum
// per simdgroup of the HeadDim threads for INT8; BF16 copies the source bits
// and has none.
#define SPLASH_STORE_MAXIMA_int8_t(HeadDim) threadgroup float maxima[(HeadDim) / 32];
#define SPLASH_STORE_MAXIMA_bfloat(HeadDim) threadgroup float *maxima = nullptr;

// One lane per dimension stores row `row` of a sequence's staged keys or
// values, KV head `head`, in its page slot, if the sequence holds the row
// (`live`, of its layout entry, which the loads within its bound do not wait
// for). INT8 derives a per-row scale; BF16 copies the original bits. Slots
// are addressed inside the head's slab of the page: the page index functions
// at page zero and head zero.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values, device const SplashKvPage *page_table,
    constant SplashAttentionSequence &sequence, SplashKvLayer kv, threadgroup float *maxima, bool value_tensor,
    uint head, uint row, bool live, uint dimension, uint simd_lane, uint simd_group) {
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
  if (!live)
    return;

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

// A threadgroup of HeadDim threads stores one row of one KV head: group
// (tensor and KV head, row, sequence) takes the keys' heads first, then the
// values', of row `row` of the dispatch's sequence, which names its page
// table among the bindings; a row past the sequence's rows (its layout
// entry's) writes nothing.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
inline void splash_store_phase(device const bfloat *chunk_keys, device const bfloat *chunk_values,
                               device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
                               device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
                               device const SplashLaneRows *layout, constant SplashAttentionParams &params,
                               threadgroup float *maxima, uint3 group, uint thread_index, uint simd_lane,
                               uint simd_group) {
  const uint row = group.y, s = group.z;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  if (group.x >= 2 * KVHeads || thread_index >= HeadDim || !splash_attention_rows_valid(params, sequence) ||
      row >= sequence.row_bound)
    return;
  splash_store_kv_row<KVHeads, HeadDim, CacheElement, ValuesTokenMajor>(
      chunk_keys, chunk_values, SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3),
      sequence, params.kv, maxima, group.x >= KVHeads, group.x % KVHeads, row,
      row < splash_attention_rows(layout, params, s).rows, thread_index, simd_lane, simd_group);
}

#define PAGED_ATTENTION_STORE(Name, Heads, HeadDim, CacheElement, ValuesTokenMajor)                              \
  kernel void Name(device const bfloat *chunk_keys [[buffer(0)]], device const bfloat *chunk_values [[buffer(1)]], \
                   device const SplashKvPage *page_table0 [[buffer(2)]],                                         \
                   device const SplashKvPage *page_table1 [[buffer(3)]],                                         \
                   device const SplashKvPage *page_table2 [[buffer(4)]],                                         \
                   device const SplashKvPage *page_table3 [[buffer(5)]],                                         \
                   device const SplashLaneRows *layout [[buffer(6)]],                                            \
                   constant SplashAttentionParams &params [[buffer(7)]],                                         \
                   uint3 group [[threadgroup_position_in_grid]],                                                 \
                   uint thread_index [[thread_index_in_threadgroup]],                                            \
                   uint simd_lane [[thread_index_in_simdgroup]],                                                 \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                         \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                                  \
    splash_store_phase<Heads, HeadDim, CacheElement, ValuesTokenMajor>(                                          \
        chunk_keys, chunk_values, page_table0, page_table1, page_table2, page_table3, layout, params, maxima,    \
        group, thread_index, simd_lane, simd_group);                                                             \
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

// Moves a lane's accepted path's K/V of one KV head of one layer region
// (SplashKvPathCommitParams): its row at depth k from the slot of its row to
// that of its depth, each thread one dimension of every row it moves, a
// threadgroup of HeadDim threads per (region and KV head, lane), and thread 0
// an INT8 page's scales. Rows move in order of depth: a row's target is a
// slot only an earlier row read (rows 0, 2, 3 move 2 to 1, then 3 to 2), as
// every later row sits past it. A row at its depth stays: a chain's path
// moves nothing.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
inline void splash_path_commit_phase(device const SplashKvPage *page_table, uint path, uint committed,
                                     SplashKvLayer kv, uint head, uint dimension) {
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, head);
  for (uint depth = 0, rows = path; rows; ++depth, rows &= rows - 1) {
    const uint row = ctz(rows);
    if (row == depth)
      continue;
    const uint from = committed + row, to = committed + depth;
    const uint source_place = from % SplashKvPageTokens, target_place = to % SplashKvPageTokens;
    const SplashKvPageTensors<CacheElement> source = addressing.page(page_table[from / SplashKvPageTokens]);
    const SplashKvPageTensors<CacheElement> target = addressing.page(page_table[to / SplashKvPageTokens]);
    target.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, target_place, dimension)] =
        source.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, source_place, dimension)];
    target.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, target_place, dimension)] =
        source.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, source_place, dimension)];
    if constexpr (is_same<CacheElement, int8_t>::value) {
      if (dimension == 0) {
        target.key_scales[splash_q8_scale_index<KVHeads>(0, 0, target_place)] =
            source.key_scales[splash_q8_scale_index<KVHeads>(0, 0, source_place)];
        target.value_scales[splash_q8_scale_index<KVHeads>(0, 0, target_place)] =
            source.value_scales[splash_q8_scale_index<KVHeads>(0, 0, source_place)];
      }
    }
  }
}

// Grid {regions x KV heads, lanes}; paths: each lane's accepted path, as
// acceptance wrote it.
#define PAGED_ATTENTION_PATH_COMMIT(Name, Heads, HeadDim, CacheElement, ValuesTokenMajor)                         \
  kernel void Name(device const SplashKvPage *page_table0 [[buffer(0)]],                                         \
                   device const SplashKvPage *page_table1 [[buffer(1)]],                                         \
                   device const SplashKvPage *page_table2 [[buffer(2)]],                                         \
                   device const SplashKvPage *page_table3 [[buffer(3)]],                                         \
                   device const uint *paths [[buffer(4)]],                                                       \
                   constant SplashKvPathCommitParams &params [[buffer(5)]],                                      \
                   uint2 group [[threadgroup_position_in_grid]],                                                 \
                   uint thread_index [[thread_index_in_threadgroup]]) {                                          \
    using Bytes = SplashKvPageBytes<Heads, HeadDim, CacheElement>;                                               \
    const uint lane = group.y;                                                                                   \
    splash_path_commit_phase<Heads, HeadDim, CacheElement, ValuesTokenMajor>(                                    \
        SPLASH_LANE_BINDING(lane, page_table0, page_table1, page_table2, page_table3), paths[lane],              \
        params.committed_tokens[lane],                                                                           \
        splash_kv_layer(params.extent_pages, Bytes::Data, Bytes::Scale, group.x / Heads), group.x % Heads,       \
        thread_index);                                                                                           \
  }
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit, 4, 256, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv2_g8, 2, 256, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit, 4, 256, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv2_g8, 2, 256, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv4_d128, 4, 128, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv4_d128, 4, 128, bfloat, false)
PAGED_ATTENTION_PATH_COMMIT(attention_q8_commit_kv2_vt, 2, 256, int8_t, true)
PAGED_ATTENTION_PATH_COMMIT(attention_bf16_commit_kv2_vt, 2, 256, bfloat, true)
#undef PAGED_ATTENTION_PATH_COMMIT
