#include "metal/kernels/common/kv_path_commit.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/relative_attention_tile.h"

// Relative attention (ops::RelativeAttention, kernels/common/
// relative_attention_tile.h): its prepare, then the page loop over the tiles
// of a dispatch's sequences and the reduce, laid out and bound as the paged
// attention's (kernels/shared/paged_attention.metal), each tile
// SPLASH_ATTENTION_TILE_ROWS query rows of one KV head, every sequence of a
// command in one dispatch, its rows its entry of the command's layout. The
// geometry is Inkling-2-Mini's: 24 query heads over 8 KV heads of 128
// dimensions, three a group, a tile one part of 24 fused rows.

static_assert(SPLASH_ATTENTION_TILE_PARTS(3u) == 1, "a group of three runs whole tiles");

// Prepare: a threadgroup of HeadDim threads per (head, row, sequence),
// relative_attention_prepare_phase. Bindings: the projection rows, the
// convolved K and V rows, the learned query and relative-state scales (F32),
// the stagings of the queries, relative states and sinks, the page tables,
// the command's tree rows (another buffer of the dispatch without), its
// layout and the parameters. _scaled scales the queries and states by their
// rows' log scales (the full layers').
#define RELATIVE_ATTENTION_PREPARE(Name, QHeads, KHeads, HeadDim, PositionScaled, CacheElement)                 \
  kernel void Name(                                                                                          \
      device const bfloat *projection [[buffer(0)]],                                                         \
      device const bfloat *keys [[buffer(1)]],                                                               \
      device const bfloat *values [[buffer(2)]],                                                             \
      device const float *query_scale [[buffer(3)]],                                                         \
      device const float *relative_scale [[buffer(4)]],                                                      \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device float *relative_states [[buffer(6)]],                                                           \
      device float *sinks [[buffer(7)]],                                                                     \
      device const SplashKvPage *page_table0 [[buffer(8)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(9)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(10)]],                                                 \
      device const SplashKvPage *page_table3 [[buffer(11)]],                                                 \
      device const SplashKvPage *page_table4 [[buffer(12)]],                                                 \
      device const SplashKvPage *page_table5 [[buffer(13)]],                                                 \
      device const SplashKvPage *page_table6 [[buffer(14)]],                                                 \
      device const SplashKvPage *page_table7 [[buffer(15)]],                                                 \
      device const SplashTreeRow *tree [[buffer(16)]],                                                       \
      device const SplashLaneRows *layout [[buffer(17)]],                                                    \
      constant SplashAttentionParams &params [[buffer(18)]],                                                 \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[2 * (HeadDim / 32)];                                                        \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    relative_attention_prepare_phase<QHeads, KHeads, HeadDim, PositionScaled, CacheElement>(                 \
        projection, keys, values, query_scale, relative_scale, queries, relative_states, sinks,              \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4,        \
                            page_table5, page_table6, page_table7),                                          \
        tree, layout, params, reductions, maxima, group, thread_index, lane, simd_group);                    \
  }
RELATIVE_ATTENTION_PREPARE(relative_attention_q8_prepare, 24, 8, 128, false, int8_t)
RELATIVE_ATTENTION_PREPARE(relative_attention_bf16_prepare, 24, 8, 128, false, bfloat)
RELATIVE_ATTENTION_PREPARE(relative_attention_q8_prepare_scaled, 24, 8, 128, true, int8_t)
RELATIVE_ATTENTION_PREPARE(relative_attention_bf16_prepare_scaled, 24, 8, 128, true, bfloat)
#undef RELATIVE_ATTENTION_PREPARE

// Split: KV heads vary first, then the dispatch's tiles, then history splits,
// as splash_attention_split_phase's; a tile reads its fused rows' relative
// states from the states' staging at its queries' place.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, typename CacheElement,
          bool Int8Queries>
inline void relative_attention_split_phase(
    device bfloat *queries, device float *partials, device float *statistics,
    device const SplashKvPage *page_table, device const SplashTreeRow *tree, device const SplashLaneRows *layout,
    device const float *relative_states, device const float *relative_bias,
    constant SplashAttentionParams &params, threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, uint3 group, uint thread_index, uint s) {
  const uint split = group.z;
  const uint kv_head = group.x;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, kv_head, group.y - sequence.first_tile);
  if (!tile.active || split >= tile.splits)
    return;
  relative_paged_attention_tile<KVHeads, QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, HeadDim, Window,
                                CacheElement, Int8Queries>(
      queries + tile.offset, relative_states + tile.offset / HeadDim * SPLASH_RELATIVE_ATTENTION_STATES,
      relative_bias, page_table, params.kv, kv_head, tile.first_position, tile.active_rows, tile.splits, split,
      partials, statistics, tile.first_slot + split, scores, probabilities, row_max, row_sum, previous_scale,
      rescale, splash_attention_tree(params, sequence, rows, tree, tile.first_position), thread_index);
}

// The paged attention's split bindings, then the relative states' staging
// and the layer's bias table (F32, [distance][state]) before the parameters.
#define RELATIVE_ATTENTION_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement, Int8Queries)                \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device float *statistics [[buffer(2)]],                                                                 \
      device const SplashKvPage *page_table0 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(6)]],                                                   \
      device const SplashKvPage *page_table4 [[buffer(7)]],                                                   \
      device const SplashKvPage *page_table5 [[buffer(8)]],                                                   \
      device const SplashKvPage *page_table6 [[buffer(9)]],                                                   \
      device const SplashKvPage *page_table7 [[buffer(10)]],                                                  \
      device const SplashTreeRow *tree [[buffer(11)]],                                                        \
      device const SplashLaneRows *layout [[buffer(12)]],                                                     \
      device const float *relative_states [[buffer(13)]],                                                     \
      device const float *relative_bias [[buffer(14)]],                                                       \
      constant SplashAttentionParams &params [[buffer(15)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    constexpr uint M = Group * SPLASH_ATTENTION_TILE_ROWS;                                                    \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    const uint s = splash_attention_sequence_of_tile(params, group.y);                                        \
    relative_attention_split_phase<Heads, Group, HeadDim, Window, CacheElement, Int8Queries>(                 \
        queries, partials, statistics,                                                                        \
        SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5,  \
                            page_table6, page_table7),                                                        \
        tree, layout, relative_states, relative_bias, params, scores, probabilities, row_max, row_sum,        \
        previous_scale, &rescale, group, thread_index, s);                                                    \
  }
// Over the whole history (the full layers) and over a window of the 1023
// keys before each row (the sliding layers' 1024 keys, the row's own
// included); INT8 pages with bf16 or INT8 queries, and BF16 pages.
#define RELATIVE_ATTENTION_SPLITS(Suffix, Window)                                                             \
  RELATIVE_ATTENTION_SPLIT(relative_attention_q8_split##Suffix, 8, 3, 128, Window, int8_t, false)             \
  RELATIVE_ATTENTION_SPLIT(relative_attention_q8q_split##Suffix, 8, 3, 128, Window, int8_t, true)             \
  RELATIVE_ATTENTION_SPLIT(relative_attention_bf16_split##Suffix, 8, 3, 128, Window, bfloat, false)
RELATIVE_ATTENTION_SPLITS(, 0)
RELATIVE_ATTENTION_SPLITS(_w1023, 1023)
#undef RELATIVE_ATTENTION_SPLITS
#undef RELATIVE_ATTENTION_SPLIT

// Reduce: a threadgroup of HeadDim threads per fused row of a tile, group.z
// the dispatch's tile, as splash_attention_reduce_phase's: the row's splits
// with its own sink, staged as its queries, then its gated output row
// (relative_attention_output_row) at its output row. The splits that wrote
// are those splash_attention_reduce_row recomputes: the partition of the same
// Window from the same first page.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, bool Table>
inline void relative_attention_reduce_phase(
    device const float *partials, device const float *statistics, device const bfloat *projection,
    device bfloat *hidden, device bfloat *table, device float *sums, device const SplashLaneRows *layout,
    device const float *sinks, constant SplashAttentionParams &params, uint3 group, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values, threadgroup bfloat *pairs) {
  constexpr ushort M = SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  const uint fused_row = group.y;
  const uint s = splash_attention_sequence_of_tile(params, group.z);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, group.x, group.z - sequence.first_tile);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  uint pages = splash_attention_pages(tile.first_position + tile.active_rows);
  if constexpr (Window != 0)
    pages -= splash_attention_first_page<Window>(
        splash_attention_window_position(params.trees != 0, sequence.committed_tokens, tile.first_position));
  const uint per_split = splash_attention_pages_per_split(pages, tile.splits);
  const uint written = (pages + per_split - 1) / per_split;
  const bfloat value = relative_attention_combine_splits<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, HeadDim>(
      partials, statistics, tile.active_rows, written, tile.first_slot, fused_row, thread_index, weights,
      group_values, sinks[tile.offset / HeadDim + fused_row]);
  const SplashAttentionOutput place = splash_attention_output(
      params, s, rows, tile.first_position - sequence.committed_tokens + fused_row / QueryHeadsPerKVHead);
  relative_attention_output_row<KVHeads * QueryHeadsPerKVHead, HeadDim, Table>(
      value, projection, hidden, table, sums, place.packed, place.output,
      group.x * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead, thread_index, place.writes, pairs);
}

// Bindings: the partials and statistics, the projection rows (their gates),
// the out-projection's input rows, its table and the table's sums (other
// buffers of the dispatch without a table), the command's layout, the rows'
// sinks' staging, then the parameters. Each is compiled plain and,
// _table16, writing the out-projection's Table16 input too.
#define RELATIVE_ATTENTION_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Table)                            \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const bfloat *projection [[buffer(2)]],                                                          \
      device bfloat *hidden [[buffer(3)]],                                                                    \
      device bfloat *table [[buffer(4)]],                                                                     \
      device float *sums [[buffer(5)]],                                                                       \
      device const SplashLaneRows *layout [[buffer(6)]],                                                      \
      device const float *sinks [[buffer(7)]],                                                                \
      constant SplashAttentionParams &params [[buffer(8)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float weights[SplashAttentionMaximumSplits];                                                  \
    threadgroup float group_values[HeadDim / 32];                                                             \
    threadgroup bfloat pairs[HeadDim];                                                                        \
    relative_attention_reduce_phase<Heads, Group, HeadDim, Window, Table>(                                    \
        partials, statistics, projection, hidden, table, sums, layout, sinks, params, group, thread_index,    \
        weights, group_values, pairs);                                                                        \
  }
#define RELATIVE_ATTENTION_REDUCE(Name, Window)                                                               \
  RELATIVE_ATTENTION_REDUCE_KERNEL(Name, 8, 3, 128, Window, false)                                            \
  RELATIVE_ATTENTION_REDUCE_KERNEL(Name##_table16, 8, 3, 128, Window, true)
RELATIVE_ATTENTION_REDUCE(relative_attention_reduce, 0)
RELATIVE_ATTENTION_REDUCE(relative_attention_reduce_w1023, 1023)
#undef RELATIVE_ATTENTION_REDUCE
#undef RELATIVE_ATTENTION_REDUCE_KERNEL

// The path commit (kernels/common/kv_path_commit.h) of the pages relative
// attention stores: 8 KV heads of 128 dimensions, values dimension-major
// (ops::RelativeAttention::addPathCommit).
PAGED_ATTENTION_PATH_COMMIT(relative_attention_q8_commit, 8, 128, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(relative_attention_bf16_commit, 8, 128, bfloat, false)
#undef PAGED_ATTENTION_PATH_COMMIT
