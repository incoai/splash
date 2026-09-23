#include "metal/kernels/common/paged_attention_tile.h"

// Verify tiles process one lane's eight rows per KV head and history split.
// Verify and prefill share the device-operand page loop in paged_attention_tile.h.

// The tile one verify threadgroup owns: group.x is the KV head, group.y the
// history split and group.z the lane, whose parameters select the page table
// and the query tile. A threadgroup past its lane's split count, or whose
// lane fails the contract, is inactive and does nothing.
struct SplashQ8VerifyTile {
  device bfloat *queries;
  device const uint *page_table;
  ulong slot;
  uint kv_head;
  uint split;
  uint splits;
  uint committed_tokens;
  uint active_rows;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline SplashQ8VerifyTile splash_q8_verify_attention_tile_at(
    device bfloat *queries, device const uint *page_table0,
    device const uint *page_table1, device const uint *page_table2,
    device const uint *page_table3,
    constant SplashQ8VerifyAttentionParams *params, uint3 group) {
  constexpr uint D = SplashQ8HeadDimension;
  SplashQ8VerifyTile tile{};
  uint kv_head = group.x;
  uint split = group.y;
  uint batch = group.z;
  constant SplashQ8VerifyAttentionParams &lane_params = params[batch];
  if (!splash_q8_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || split >= lane_params.split_count)
    return tile;
  ulong group_stride = ulong(lane_params.chunk_stride) * QueryHeadsPerKVHead * D;
  tile.queries = queries + (ulong(batch) * KVHeads + kv_head) * group_stride;
  tile.page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  tile.slot =
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits + split;
  tile.kv_head = kv_head;
  tile.split = split;
  tile.splits = lane_params.split_count;
  tile.committed_tokens = lane_params.committed_tokens;
  tile.active_rows = lane_params.active_rows;
  tile.active = true;
  return tile;
}

// Verify entries: one lane per group.z, eight rows, one configured history
// partition, with scratch for scores, probabilities and row statistics.
#define Q8_VERIFY_SPLIT_SIGNATURE(Name)                                        \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device int8_t *cache_keys [[buffer(1)]],                                 \
      device const float *key_scales_buffer [[buffer(2)]],                     \
      device int8_t *cache_values [[buffer(3)]],                               \
      device const float *value_scales_buffer [[buffer(4)]],                   \
      device float *partials [[buffer(5)]],                                    \
      device float *statistics [[buffer(6)]],                                  \
      device const uint *page_table0 [[buffer(7)]],                            \
      device const uint *page_table1 [[buffer(8)]],                            \
      device const uint *page_table2 [[buffer(9)]],                            \
      device const uint *page_table3 [[buffer(10)]],                           \
      constant SplashQ8VerifyAttentionParams *params [[buffer(11)]],           \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]])

#define Q8_VERIFY_SCRATCH(Group)                                               \
  constexpr uint M = Group * SPLASH_TARGET_VERIFY_ROWS;                        \
  constexpr uint N = SplashQ8PageTokens;                                       \
  alignas(16) threadgroup float scores[M * N];                                 \
  alignas(16) threadgroup bfloat probabilities[M * N];                         \
  threadgroup float row_max[M];                                                \
  threadgroup float row_sum[M];                                                \
  threadgroup float previous_scale[M];                                         \
  threadgroup atomic_uint rescale;

#define Q8_VERIFY_TILE_AT(Heads, Group)                                        \
  const SplashQ8VerifyTile tile =                                              \
      splash_q8_verify_attention_tile_at<Heads, Group>(                        \
          queries, page_table0, page_table1, page_table2, page_table3, params, \
          group);                                                              \
  if (!tile.active)                                                            \
    return;

#define Q8_VERIFY_SPLIT(Name, Heads, Group, ScaleInSoftmax)                    \
  Q8_VERIFY_SPLIT_SIGNATURE(Name) {                                            \
    Q8_VERIFY_SCRATCH(Group)                                                   \
    Q8_VERIFY_TILE_AT(Heads, Group)                                            \
    splash_paged_attention_tile<Heads, Group,                                  \
                                      SPLASH_TARGET_VERIFY_ROWS,               \
                                      ScaleInSoftmax>(                         \
        tile.queries, cache_keys, key_scales_buffer, cache_values, value_scales_buffer,\
        tile.page_table, tile.kv_head, tile.committed_tokens, tile.active_rows,\
        tile.splits, tile.split, partials, statistics, tile.slot, scores,      \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index);                                                         \
  }

Q8_VERIFY_SPLIT(verify_attention_q8_split, 4, 6, true)
Q8_VERIFY_SPLIT(verify_attention_q8_split_cooperative_scale,
                       4, 6, false)
Q8_VERIFY_SPLIT(verify_attention_q8_split_kv2_g8, 2, 8, true)
Q8_VERIFY_SPLIT(
    verify_attention_q8_split_cooperative_scale_kv2_g8, 2, 8, false)
#define BF16_VERIFY_SPLIT_SIGNATURE(Name)                                      \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device bfloat *cache_keys [[buffer(1)]],                                 \
      device bfloat *cache_values [[buffer(2)]],                               \
      device float *partials [[buffer(3)]],                                    \
      device float *statistics [[buffer(4)]],                                  \
      device const uint *page_table0 [[buffer(5)]],                            \
      device const uint *page_table1 [[buffer(6)]],                            \
      device const uint *page_table2 [[buffer(7)]],                            \
      device const uint *page_table3 [[buffer(8)]],                            \
      constant SplashQ8VerifyAttentionParams *params [[buffer(9)]],            \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]])

#define BF16_VERIFY_SPLIT(Name, Heads, Group)                                  \
  BF16_VERIFY_SPLIT_SIGNATURE(Name) {                                          \
    Q8_VERIFY_SCRATCH(Group)                                                   \
    Q8_VERIFY_TILE_AT(Heads, Group)                                            \
    splash_paged_attention_tile<Heads, Group,                                  \
                                      SPLASH_TARGET_VERIFY_ROWS,               \
                                      true>(                                   \
        tile.queries, cache_keys, nullptr, cache_values, nullptr,              \
        tile.page_table, tile.kv_head, tile.committed_tokens, tile.active_rows,\
        tile.splits, tile.split, partials, statistics, tile.slot, scores,      \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index);                                                         \
  }

BF16_VERIFY_SPLIT(verify_attention_bf16_split, 4, 6)
BF16_VERIFY_SPLIT(verify_attention_bf16_split_kv2_g8, 2, 8)
#undef BF16_VERIFY_SPLIT
#undef BF16_VERIFY_SPLIT_SIGNATURE
#undef Q8_VERIFY_SPLIT
#undef Q8_VERIFY_TILE_AT
#undef Q8_VERIFY_SCRATCH
#undef Q8_VERIFY_SPLIT_SIGNATURE

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void splash_q8_verify_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashQ8VerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr ushort M = SPLASH_TARGET_VERIFY_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = SplashQ8HeadDimension;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint batch = group.z;
  constant SplashQ8VerifyAttentionParams &lane_params = params[batch];
  if (!splash_q8_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D)
    return;
  ulong group_stride = ulong(lane_params.chunk_stride) * QueryHeadsPerKVHead * D;
  splash_q8_attention_reduce_row_shared<QueryHeadsPerKVHead,
                                   SPLASH_TARGET_VERIFY_ROWS>(
      partials, statistics,
      output + (ulong(batch) * KVHeads + kv_head) * group_stride,
      lane_params.committed_tokens, lane_params.active_rows,
      lane_params.split_count,
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits, fused_row,
      thread_index, weights, group_values);
}

kernel void verify_attention_q8_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8VerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[SplashVerifyMaximumSplits];
  threadgroup float group_values[8];
  splash_q8_verify_attention_reduce_phase<4, 6>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}

kernel void verify_attention_q8_reduce_kv2_g8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashQ8VerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[SplashVerifyMaximumSplits];
  threadgroup float group_values[8];
  splash_q8_verify_attention_reduce_phase<2, 8>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}
