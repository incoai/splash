#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/sparse_attention_tile.h"

// Sparse attention (metal/abi/PagedAttention.h): the block lists a tile's rows'
// block masks make, and the split and reduce entries that attend through
// them, for the GQA group of twelve over two KV heads on a KV layout that
// keeps values token-major (Qwen3.8-Flash-Next's, for one).

// Threadgroup per tile, group.x the tile and group.y the sequence
// (SplashAttentionBlockListParams): only the blocks of the tile's history are
// listed (the tail bit of a row whose token completes its block names the
// next block), and every thread takes a contiguous run of their words, so the
// entries keep block order. A list keeps to its stride whatever the masks
// hold.
constant constexpr uint kBlockListThreads = SPLASH_ATTENTION_BLOCK_LIST_THREADS;

// The bits of mask word `w` that name one of the first `blocks` blocks.
inline uint splash_history_word_bits(uint blocks, uint w) {
  const uint remaining = blocks - w * 32;
  return remaining >= 32 ? ~0u : (1u << remaining) - 1u;
}

kernel void attention_block_lists(device const uint *mask [[buffer(0)]],
                                  device uint *lists [[buffer(1)]],
                                  device uint *counts [[buffer(2)]],
                                  constant SplashAttentionBlockListParams &p [[buffer(3)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint simd_counts[kBlockListThreads / 32];
  if (group.y >= min(p.sequences, uint(SPLASH_MAXIMUM_BATCH_WIDTH)))
    return;
  const SplashAttentionBlockListSequence s = p.sequence[group.y];
  const uint tile = group.x, first_row = tile * p.tile_rows;
  if (first_row >= s.rows)
    return;
  const uint rows = min(p.tile_rows, s.rows - first_row);
  const uint blocks = splash_sparse_history_blocks(s.first_position + first_row, rows);
  const uint words = min((blocks + 31) / 32, p.mask_words);
  const uint per = (words + kBlockListThreads - 1) / kBlockListThreads;
  const uint w0 = min(tid * per, words), w1 = min(w0 + per, words);
  device const uint *bits = mask + ulong(s.mask_row + first_row) * p.mask_words;
  uint attended = 0;
  for (uint w = w0; w < w1; ++w) {
    uint any = 0;
    for (uint r = 0; r < rows; ++r)
      any |= bits[ulong(r) * p.mask_words + w];
    attended += popcount(any & splash_history_word_bits(blocks, w));
  }
  const uint inclusive = simd_prefix_inclusive_sum(attended);
  if (lane == 31)
    simd_counts[sg] = inclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = inclusive - attended;
  for (uint g = 0; g < sg; ++g)
    before += simd_counts[g];
  device uint *list = lists + ulong(s.list + tile) * p.stride;
  for (uint w = w0; w < w1; ++w) {
    uint row_bits[SPLASH_ATTENTION_BLOCK_ROW_BITS];
    uint any = 0;
    for (uint r = 0; r < rows; ++r) {
      row_bits[r] = bits[ulong(r) * p.mask_words + w];
      any |= row_bits[r];
    }
    any &= splash_history_word_bits(blocks, w);
    while (any) {
      const uint b = ctz(any);
      any &= any - 1;
      uint entry = (w * 32 + b) << SPLASH_ATTENTION_BLOCK_ROW_BITS;
      for (uint r = 0; r < rows; ++r)
        entry |= (row_bits[r] >> b & 1u) << r;
      if (before < p.stride)
        list[before] = entry;
      ++before;
    }
  }
  if (tid == kBlockListThreads - 1)
    counts[s.list + tile] = min(before, p.stride);
}

// The entries of list `index` of the bound lists, at most its stride, or
// without a stride (no lists bound) the blocks of the tile's history.
inline uint splash_sparse_list_count(device const uint *counts, uint stride, uint index,
                                     uint first_position, uint active_rows) {
  return stride ? min(counts[index], stride) : splash_sparse_history_blocks(first_position, active_rows);
}

// A tile's list, null without a stride, and its entries' count.
struct SplashSparseList {
  device const uint *entries;
  uint count;
};

inline SplashSparseList splash_sparse_list(device const uint *lists, device const uint *counts,
                                           uint stride, uint index, uint first_position,
                                           uint active_rows) {
  return {stride ? lists + ulong(index) * stride : nullptr,
          splash_sparse_list_count(counts, stride, index, first_position, active_rows)};
}

// Verify: group.x is the KV head, group.y the split and group.z the lane,
// whose eight rows attend through the lane's list (block_list).
#define SPARSE_VERIFY_SPLIT(Name, Heads, Group, CacheElement)                  \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]],                    \
      device const uint *lists [[buffer(7)]],                                  \
      device const uint *counts [[buffer(8)]],                                 \
      constant SplashVerifyAttentionParams *params [[buffer(9)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;                           \
    threadgroup SplashSparseScratch<CacheElement, Rows * Group> scratch;       \
    constant SplashVerifyAttentionParams &lane = params[group.z];              \
    const SplashAttentionTile tile =                                           \
        splash_verify_attention_tile<Heads, Group>(lane, group.x, group.z);    \
    if (!tile.active || group.y >= tile.splits)                                \
      return;                                                                  \
    const SplashSparseList list =                                              \
        splash_sparse_list(lists, counts, lane.block_stride, lane.block_list,  \
                           tile.first_position, tile.active_rows);             \
    splash_sparse_attention_tile<Heads, Group, Rows, CacheElement>(            \
        queries + tile.offset,                                                 \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2,    \
                            page_table3),                                      \
        lane.kv, group.x, tile.first_position, tile.active_rows, list.entries, \
        list.count, tile.splits, group.y, partials, statistics,                \
        tile.first_slot + group.y, scratch, thread_index);                     \
  }
SPARSE_VERIFY_SPLIT(verify_attention_q8_sparse_kv2_g12, 2, 12, int8_t)
SPARSE_VERIFY_SPLIT(verify_attention_bf16_sparse_kv2_g12, 2, 12, bfloat)
#undef SPARSE_VERIFY_SPLIT

// Combines the splits that took entries of the lane's list.
#define SPARSE_VERIFY_REDUCE(Name, Heads, Group)                               \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device const uint *counts [[buffer(3)]],                                 \
      constant SplashVerifyAttentionParams *params [[buffer(4)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;                           \
    threadgroup float weights[SplashVerifyMaximumSplits];                      \
    threadgroup float group_values[8];                                         \
    constant SplashVerifyAttentionParams &lane = params[group.z];              \
    const uint fused_row = group.y;                                            \
    const SplashAttentionTile tile =                                           \
        splash_verify_attention_tile<Heads, Group>(lane, group.x, group.z);    \
    if (!tile.active || fused_row >= Rows * Group ||                           \
        thread_index >= SplashKvHeadDimension)                                 \
      return;                                                                  \
    const uint count =                                                         \
        splash_sparse_list_count(counts, lane.block_stride, lane.block_list,   \
                                 tile.first_position, tile.active_rows);       \
    splash_attention_combine_splits<Group, Rows>(                              \
        partials, statistics, output + tile.offset, tile.active_rows,          \
        splash_sparse_written_splits(count, tile.splits), tile.first_slot,     \
        fused_row, thread_index, weights, group_values);                       \
  }
SPARSE_VERIFY_REDUCE(verify_attention_sparse_reduce_kv2_g12, 2, 12)
#undef SPARSE_VERIFY_REDUCE

// Prefill: group.x is the KV head, group.y the query tile and group.z the
// split; tile t of the chunk attends through list block_list + t.
#define SPARSE_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                 \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      device const uint *lists [[buffer(4)]],                                  \
      device const uint *counts [[buffer(5)]],                                 \
      constant SplashPrefillAttentionParams &params [[buffer(6)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;                  \
    threadgroup SplashSparseScratch<CacheElement, Rows * Group> scratch;       \
    const SplashAttentionTile tile =                                           \
        splash_prefill_attention_tile<Heads, Group>(params, group.x, group.y); \
    if (!tile.active || group.z >= tile.splits)                                \
      return;                                                                  \
    const SplashSparseList list = splash_sparse_list(                          \
        lists, counts, params.block_stride, params.block_list + group.y,       \
        tile.first_position, tile.active_rows);                                \
    splash_sparse_attention_tile<Heads, Group, Rows, CacheElement>(            \
        queries + tile.offset, page_table, params.kv, group.x,                 \
        tile.first_position, tile.active_rows, list.entries, list.count,       \
        tile.splits, group.z, partials, statistics, tile.first_slot + group.z, \
        scratch, thread_index);                                                \
  }
SPARSE_PREFILL_SPLIT(prefill_attention_q8_sparse_kv2_g12, 2, 12, int8_t)
SPARSE_PREFILL_SPLIT(prefill_attention_bf16_sparse_kv2_g12, 2, 12, bfloat)
#undef SPARSE_PREFILL_SPLIT

#define SPARSE_PREFILL_REDUCE(Name, Heads, Group)                              \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device const uint *counts [[buffer(3)]],                                 \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;                  \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[8];                                         \
    const uint fused_row = group.y;                                            \
    const SplashAttentionTile tile =                                           \
        splash_prefill_attention_tile<Heads, Group>(params, group.x, group.z); \
    if (!tile.active || fused_row >= Rows * Group ||                           \
        thread_index >= SplashKvHeadDimension)                                 \
      return;                                                                  \
    const uint count = splash_sparse_list_count(                               \
        counts, params.block_stride, params.block_list + group.z,              \
        tile.first_position, tile.active_rows);                                \
    splash_attention_combine_splits<Group, Rows>(                              \
        partials, statistics, output + tile.offset, tile.active_rows,          \
        splash_sparse_written_splits(count, tile.splits), tile.first_slot,     \
        fused_row, thread_index, weights, group_values);                       \
  }
SPARSE_PREFILL_REDUCE(prefill_attention_sparse_reduce_kv2_g12, 2, 12)
#undef SPARSE_PREFILL_REDUCE
