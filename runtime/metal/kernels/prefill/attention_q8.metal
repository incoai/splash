#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (splash_paged_attention_tile) over eight query rows of one KV head.

// Prefill split: KV heads vary first, then query tiles, then history splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, bool ScaleInSoftmax, typename CacheElement>
inline void splash_prefill_attention_split_phase(
    device bfloat *queries, device CacheElement *cache_keys,
    device const float *key_scales_buffer, device CacheElement *cache_values,
    device const float *value_scales_buffer, device float *partials,
    device float *statistics, device const uint *page_table,
    constant SplashQ8PrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group,
    uint thread_index) {
  constexpr uint D = SplashQ8HeadDimension;
  uint kv_head = group.x;
  uint split = group.z;
  uint tile = group.y;
  uint tile_start = tile * SplashPrefillTileRows;
  if (!splash_q8_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || split >= params.split_count ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  ulong slot = (ulong(tile) * KVHeads + kv_head) * params.split_count + split;
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                                    SPLASH_PREFILL_ATTENTION_TILE_ROWS,
                                    ScaleInSoftmax>(
      queries + tile_offset, cache_keys, key_scales_buffer, cache_values,
      value_scales_buffer, page_table, kv_head,
      params.committed_tokens + tile_start, active_rows, params.split_count, split,
      partials, statistics, slot, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, thread_index);
}

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void splash_q8_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashQ8PrefillAttentionParams &params, uint3 group,
    uint thread_index) {
  constexpr ushort M = SPLASH_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = SplashQ8HeadDimension;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint tile = group.z;
  uint tile_start = tile * SplashPrefillTileRows;
  if (!splash_q8_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);
  if (fused_row / QueryHeadsPerKVHead >= active_rows)
    return;
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  splash_q8_attention_reduce_row<QueryHeadsPerKVHead,
                                   SPLASH_PREFILL_ATTENTION_TILE_ROWS>(
      partials, statistics, output + tile_offset,
      params.committed_tokens + tile_start, active_rows, params.split_count,
      (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row,
      thread_index);
}

#define Q8_PREFILL_SPLIT(Name, Heads, Group, ScaleInSoftmax)                   \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device int8_t *cache_keys [[buffer(1)]],                                 \
      device const float *key_scales_buffer [[buffer(2)]],                     \
      device int8_t *cache_values [[buffer(3)]],                               \
      device const float *value_scales_buffer [[buffer(4)]],                   \
      device float *partials [[buffer(5)]],                                    \
      device float *statistics [[buffer(6)]],                                  \
      device const uint *page_table [[buffer(7)]],                             \
      constant SplashQ8PrefillAttentionParams &params [[buffer(8)]],           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * SPLASH_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = SplashQ8PageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    splash_prefill_attention_split_phase<Heads, Group, ScaleInSoftmax>(        \
        queries, cache_keys, key_scales_buffer, cache_values, value_scales_buffer, partials,\
        statistics, page_table, params, scores, probabilities, row_max, row_sum,\
        previous_scale, &rescale, group, thread_index);                        \
  }

Q8_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, true)
Q8_PREFILL_SPLIT(prefill_attention_q8_split_cooperative_scale,
                4, 6, false)
Q8_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, true)
Q8_PREFILL_SPLIT(prefill_attention_q8_split_cooperative_scale_kv2_g8,
                2, 8, false)
#undef Q8_PREFILL_SPLIT

kernel void prefill_attention_q8_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8PrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  splash_q8_prefill_attention_reduce_phase<4, 6>(partials, statistics,
                                                   output, params, group,
                                                   thread_index);
}

kernel void prefill_attention_q8_reduce_kv2_g8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8PrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  splash_q8_prefill_attention_reduce_phase<2, 8>(partials, statistics,
                                                   output, params, group,
                                                   thread_index);
}

// BF16 shares the page loop and reduction, without quantization scales.
#define BF16_PREFILL_SPLIT(Name, Heads, Group)                                 \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device bfloat *cache_keys [[buffer(1)]],                                 \
      device bfloat *cache_values [[buffer(2)]],                               \
      device float *partials [[buffer(3)]],                                    \
      device float *statistics [[buffer(4)]],                                  \
      device const uint *page_table [[buffer(5)]],                             \
      constant SplashQ8PrefillAttentionParams &params [[buffer(6)]],           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * SPLASH_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = SplashQ8PageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    splash_prefill_attention_split_phase<Heads, Group, true>(                  \
        queries, cache_keys, nullptr, cache_values, nullptr, partials,         \
        statistics, page_table, params, scores, probabilities, row_max, row_sum,\
        previous_scale, &rescale, group, thread_index);                        \
  }

BF16_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6)
BF16_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8)
#undef BF16_PREFILL_SPLIT
