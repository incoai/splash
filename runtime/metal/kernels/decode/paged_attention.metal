#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_attention_tile.h"

// Verify tiles process one lane's eight rows per KV head and history split.
// Verify and prefill share the device-operand page loop in paged_attention_tile.h.

// Verify entries: one lane per group.z, eight rows, one configured history
// partition, with scratch for scores, probabilities and row statistics.
// INT8 and BF16 entries share one signature: pages are reached through the
// tables, so no entry binds KV storage.
#define PAGED_VERIFY_SPLIT_SIGNATURE(Name)                                     \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]],                    \
      constant SplashVerifyAttentionParams *params [[buffer(7)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]])

#define PAGED_VERIFY_SCRATCH(Group)                                            \
  constexpr uint M = Group * SPLASH_TARGET_VERIFY_ROWS;                        \
  constexpr uint N = SplashKvPageTokens;                                       \
  alignas(16) threadgroup float scores[M * N];                                 \
  alignas(16) threadgroup bfloat probabilities[M * N];                         \
  threadgroup float row_max[M];                                                \
  threadgroup float row_sum[M];                                                \
  threadgroup float previous_scale[M];                                         \
  threadgroup atomic_uint rescale;

// group.x is the KV head, group.y the history split and group.z the lane,
// whose parameters select the page table and the query tile.
#define PAGED_VERIFY_TILE_AT(Heads, Group)                                     \
  const SplashAttentionTile tile =                                             \
      splash_verify_attention_tile<Heads, Group>(params[group.z], group.x,     \
                                                 group.z);                     \
  if (!tile.active || group.y >= tile.splits)                                  \
    return;

#define PAGED_VERIFY_SPLIT(Name, Heads, Group, CacheElement)                   \
  PAGED_VERIFY_SPLIT_SIGNATURE(Name) {                                         \
    PAGED_VERIFY_SCRATCH(Group)                                                \
    PAGED_VERIFY_TILE_AT(Heads, Group)                                         \
    splash_paged_attention_tile<Heads, Group, SPLASH_TARGET_VERIFY_ROWS,       \
                                CacheElement>(                                 \
        queries + tile.offset,                                                 \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2,    \
                            page_table3),                                      \
        params[group.z].kv, group.x, tile.first_position, tile.active_rows,    \
        tile.splits, group.y, partials, statistics, tile.first_slot + group.y, \
        scores, probabilities, row_max, row_sum, previous_scale, &rescale,     \
        thread_index);                                                         \
  }

PAGED_VERIFY_SPLIT(verify_attention_q8_split, 4, 6, int8_t)
PAGED_VERIFY_SPLIT(verify_attention_q8_split_kv2_g8, 2, 8, int8_t)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_VERIFY_SPLIT(verify_attention_bf16_split, 4, 6, bfloat)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_kv2_g8, 2, 8, bfloat)
#undef PAGED_VERIFY_SPLIT
#undef PAGED_VERIFY_TILE_AT
#undef PAGED_VERIFY_SCRATCH
#undef PAGED_VERIFY_SPLIT_SIGNATURE

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void splash_verify_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashVerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr ushort M = SPLASH_TARGET_VERIFY_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = SplashKvHeadDimension;
  const uint fused_row = group.y;
  const SplashAttentionTile tile =
      splash_verify_attention_tile<KVHeads, QueryHeadsPerKVHead>(params[group.z],
                                                                group.x, group.z);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  splash_attention_reduce_row<QueryHeadsPerKVHead, SPLASH_TARGET_VERIFY_ROWS>(
      partials, statistics, output + tile.offset, tile.first_position,
      tile.active_rows, tile.splits, tile.first_slot, fused_row, thread_index,
      weights, group_values);
}

#define PAGED_VERIFY_REDUCE(Name, Heads, Group)                                \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant SplashVerifyAttentionParams *params [[buffer(3)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashVerifyMaximumSplits];                      \
    threadgroup float group_values[8];                                         \
    splash_verify_attention_reduce_phase<Heads, Group>(                        \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values);                                                         \
  }
PAGED_VERIFY_REDUCE(verify_attention_reduce, 4, 6)
PAGED_VERIFY_REDUCE(verify_attention_reduce_kv2_g8, 2, 8)
#undef PAGED_VERIFY_REDUCE
