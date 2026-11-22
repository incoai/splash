#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_attention_tile.h"

// The page loop (splash_paged_attention_tile) over the tiles of a dispatch's
// sequences (SplashAttentionParams), each tile SPLASH_ATTENTION_TILE_ROWS
// query rows of one KV head: a decode step's lanes and a prompt chunk's
// sequences alike, every sequence in one dispatch, its rows those of its
// entry of the command's layout (`layout`). A decode step's split binds its
// tree rows (SplashAttentionParams::trees); a prompt's binds another buffer
// of the dispatch in their place, of which it reads nothing.

// Split: KV heads (and each one's tile parts, splash_attention_tile_part)
// vary first, then the dispatch's tiles, then history splits. A tile's
// sequence names its page table among the bindings. A relative geometry's
// (Relative) reads its rows' relative states, a packed row's
// [query head][SPLASH_RELATIVE_WIDTH] F32 each, and the layer's table
// (metal/abi/PagedAttention.h).
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, typename CacheElement,
          bool Int8Queries, bool Relative = false>
inline void splash_attention_split_phase(
    device bfloat *queries, device float *partials, device float *statistics,
    device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
    device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
    device const SplashKvPage *page_table4, device const SplashKvPage *page_table5,
    device const SplashKvPage *page_table6, device const SplashKvPage *page_table7,
    device const SplashTreeRow *tree, device const SplashLaneRows *layout, constant SplashAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, uint3 group, uint thread_index,
    device const float *relative_states = nullptr, device const float4 *relative_table = nullptr) {
  constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead);
  const uint split = group.z;
  const uint kv_head = group.x / Parts;
  const uint s = splash_attention_sequence_of_tile(params, group.y);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile =
      splash_attention_tile_part<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, Parts, HeadDim>(
          splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(params, sequence, rows, kv_head,
                                                                      group.y - sequence.first_tile),
          group.x % Parts, sequence.split_count);
  if (!tile.active || split >= tile.splits)
    return;
  using RelativeTile = conditional_t<Relative, SplashRelativeTile, SplashUnbiasedTile>;
  RelativeTile relative{};
  if constexpr (Relative) {
    // The tile's (part's) first row's states of its KV head's first query head.
    const ulong first_row = ulong(rows.first) + (tile.first_position - sequence.committed_tokens);
    relative = {relative_states + (first_row * KVHeads + kv_head) * QueryHeadsPerKVHead * SPLASH_RELATIVE_WIDTH,
                relative_table};
  }
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS / Parts, HeadDim, Window,
                              CacheElement, Int8Queries, RelativeTile>(
      queries + tile.offset, SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7),
      params.kv, kv_head, tile.first_position, tile.active_rows, tile.splits, split, partials, statistics,
      tile.first_slot + split, scores, probabilities, row_max, row_sum, previous_scale, rescale,
      splash_attention_tree(params, sequence, rows, tree, tile.first_position), thread_index, relative);
}

// Reduce: a threadgroup of HeadDim threads per fused row of a tile, group.z
// the dispatch's tile, which writes the row as the out-projection's input
// (splash_attention_output_row): gated by its query head's gate in its
// projection row (Gates), at its output row (splash_attention_output), and
// with Table into the out-projection's table too. A reduce that adds learned
// sinks passes the layer's, one per query head, last; one whose gates' rows
// hold the rows' sinks (Gates::RowSinks, a relative geometry's query side
// rows) adds each row's own.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim, uint Window, class Gates, bool Table,
          class... Sinks>
inline void splash_attention_reduce_phase(
    device const float *partials, device const float *statistics, device const typename Gates::Element *gates,
    device bfloat *hidden, device bfloat *table, device float *sums, device const SplashLaneRows *layout,
    constant SplashAttentionParams &params, uint3 group, uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, threadgroup bfloat *pairs, Sinks... sinks) {
  constexpr ushort M = SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  constexpr uint Parts = SPLASH_ATTENTION_TILE_PARTS(QueryHeadsPerKVHead);
  const uint fused_row = group.y;
  const uint s = splash_attention_sequence_of_tile(params, group.z);
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, group.x, group.z - sequence.first_tile);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  bfloat value;
  if constexpr (Gates::RowSinks) {
    // The row's sink, of a row the sequence holds.
    const uint row = tile.first_position - sequence.committed_tokens + fused_row / QueryHeadsPerKVHead;
    const uint head = group.x * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead;
    const float sink = row < rows.rows ? gates[Gates::sink(ulong(rows.first) + row, head)] : -INFINITY;
    value = splash_attention_reduce_row<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, Parts, HeadDim, Window>(
        partials, statistics, tile, sequence.split_count, params.trees != 0, sequence.committed_tokens, fused_row,
        thread_index, weights, group_values, sink);
  } else {
    value = splash_attention_reduce_row<QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, Parts, HeadDim, Window>(
        partials, statistics, tile, sequence.split_count, params.trees != 0, sequence.committed_tokens, fused_row,
        thread_index, weights, group_values, splash_group_sinks<QueryHeadsPerKVHead>(group.x, sinks)...);
  }
  const SplashAttentionOutput place = splash_attention_output(
      params, s, rows, tile.first_position - sequence.committed_tokens + fused_row / QueryHeadsPerKVHead);
  splash_attention_output_row<KVHeads * QueryHeadsPerKVHead, KVHeads, HeadDim, Gates, Table>(
      value, gates, hidden, table, sums, place.packed, place.output,
      group.x * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead, thread_index, place.writes, pairs);
}

#define PAGED_ATTENTION_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement, Int8Queries)                   \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device float *statistics [[buffer(2)]],                                                                 \
      device const SplashKvPage *page_table0 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(6)]], device const SplashKvPage *page_table4 [[buffer(7)]], device const SplashKvPage *page_table5 [[buffer(8)]], device const SplashKvPage *page_table6 [[buffer(9)]], device const SplashKvPage *page_table7 [[buffer(10)]],                                                   \
      device const SplashTreeRow *tree [[buffer(11)]],                                                         \
      device const SplashLaneRows *layout [[buffer(12)]],                                                      \
      constant SplashAttentionParams &params [[buffer(13)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    constexpr uint M = Group * SPLASH_ATTENTION_TILE_ROWS / SPLASH_ATTENTION_TILE_PARTS(Group);               \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    splash_attention_split_phase<Heads, Group, HeadDim, Window, CacheElement, Int8Queries>(                   \
        queries, partials, statistics, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7, tree, layout,     \
        params, scores, probabilities, row_max, row_sum, previous_scale, &rescale, group, thread_index);      \
  }

PAGED_ATTENTION_SPLIT(attention_q8_split, 4, 6, 256, 0, int8_t, false)
PAGED_ATTENTION_SPLIT(attention_q8_split_kv2_g8, 2, 8, 256, 0, int8_t, false)
PAGED_ATTENTION_SPLIT(attention_q8_split_kv2_g16, 2, 16, 256, 0, int8_t, false)
// Four KV heads of 128 dimensions over the whole history, and over a window
// of the 512 keys before each row.
PAGED_ATTENTION_SPLIT(attention_q8_split_kv4_g8_d128, 4, 8, 128, 0, int8_t, false)
PAGED_ATTENTION_SPLIT(attention_q8_split_kv4_g8_d128_w512, 4, 8, 128, 512, int8_t, false)
// INT8 query codes against the INT8 keys, where a split reads enough pages
// (AttentionCalibration::int8QueriesFromPages).
PAGED_ATTENTION_SPLIT(attention_q8q_split, 4, 6, 256, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv2_g8, 2, 8, 256, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv2_g16, 2, 16, 256, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv4_g8_d128, 4, 8, 128, 0, int8_t, true)
PAGED_ATTENTION_SPLIT(attention_q8q_split_kv4_g8_d128_w512, 4, 8, 128, 512, int8_t, true)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_ATTENTION_SPLIT(attention_bf16_split, 4, 6, 256, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv2_g8, 2, 8, 256, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv2_g16, 2, 16, 256, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv4_g8_d128, 4, 8, 128, 0, bfloat, false)
PAGED_ATTENTION_SPLIT(attention_bf16_split_kv4_g8_d128_w512, 4, 8, 128, 512, bfloat, false)
#undef PAGED_ATTENTION_SPLIT

// A relative geometry's split binds its rows' relative states and the
// layer's table (F32, SPLASH_RELATIVE_EXTENT rows of SPLASH_RELATIVE_WIDTH)
// after the command's layout.
#define PAGED_ATTENTION_RELATIVE_SPLIT(Name, Heads, Group, HeadDim, Window, CacheElement, Int8Queries)        \
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
      device const float4 *relative_table [[buffer(14)]],                                                     \
      constant SplashAttentionParams &params [[buffer(15)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    constexpr uint M = Group * SPLASH_ATTENTION_TILE_ROWS / SPLASH_ATTENTION_TILE_PARTS(Group);               \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    splash_attention_split_phase<Heads, Group, HeadDim, Window, CacheElement, Int8Queries, true>(             \
        queries, partials, statistics, page_table0, page_table1, page_table2, page_table3, page_table4,      \
        page_table5, page_table6, page_table7, tree, layout, params, scores, probabilities, row_max, row_sum, \
        previous_scale, &rescale, group, thread_index, relative_states, relative_table);                      \
  }
// Eight KV heads of 128 dimensions under 24 query heads (Inkling 2's), over
// the whole history and over a window of the 1023 keys before each row, in
// both page formats and with INT8 queries.
#define PAGED_ATTENTION_RELATIVE_SPLITS(Suffix, Heads, Group, HeadDim, Window)                                \
  PAGED_ATTENTION_RELATIVE_SPLIT(attention_q8_split##Suffix, Heads, Group, HeadDim, Window, int8_t, false)    \
  PAGED_ATTENTION_RELATIVE_SPLIT(attention_q8q_split##Suffix, Heads, Group, HeadDim, Window, int8_t, true)    \
  PAGED_ATTENTION_RELATIVE_SPLIT(attention_bf16_split##Suffix, Heads, Group, HeadDim, Window, bfloat, false)
PAGED_ATTENTION_RELATIVE_SPLITS(_kv8_g3_d128, 8, 3, 128, 0)
PAGED_ATTENTION_RELATIVE_SPLITS(_kv8_g3_d128_w1023, 8, 3, 128, 1023)
#undef PAGED_ATTENTION_RELATIVE_SPLITS
#undef PAGED_ATTENTION_RELATIVE_SPLIT

// Bindings: the partials and statistics, the projection's packed rows (a
// gated geometry's gates, another buffer of the dispatch for an ungated one),
// the out-projection's input rows, its table and the table's sums (other
// buffers of the dispatch without a table), the command's layout, then the
// parameters; a reduce that adds learned sinks binds the layer's, an F32 per
// query head, before them. Each is compiled plain and, _table16, writing
// the out-projection's Table16 input too.
#define PAGED_ATTENTION_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, Table)                       \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const bfloat *packed_qkv [[buffer(2)]],                                                          \
      device bfloat *hidden [[buffer(3)]],                                                                    \
      device bfloat *table [[buffer(4)]],                                                                     \
      device float *sums [[buffer(5)]],                                                                       \
      device const SplashLaneRows *layout [[buffer(6)]],                                                      \
      constant SplashAttentionParams &params [[buffer(7)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float weights[SplashAttentionMaximumSplits];                                                  \
    threadgroup float group_values[HeadDim / 32];                                                             \
    threadgroup bfloat pairs[HeadDim];                                                                        \
    using Gates = SplashGatesOf<Gated, Heads * Group, Heads, HeadDim>;                                        \
    splash_attention_reduce_phase<Heads, Group, HeadDim, Window, Gates, Table>(                               \
        partials, statistics, packed_qkv, hidden, table, sums, layout, params, group, thread_index, weights,   \
        group_values, pairs);                                                                                 \
  }
#define PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, Table)                  \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const bfloat *packed_qkv [[buffer(2)]],                                                          \
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
    using Gates = SplashGatesOf<Gated, Heads * Group, Heads, HeadDim>;                                        \
    splash_attention_reduce_phase<Heads, Group, HeadDim, Window, Gates, Table>(                               \
        partials, statistics, packed_qkv, hidden, table, sums, layout, params, group, thread_index, weights,   \
        group_values, pairs, sinks);                                                                          \
  }
#define PAGED_ATTENTION_REDUCE(Name, Heads, Group, HeadDim, Window, Gated)                                     \
  PAGED_ATTENTION_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, false)                             \
  PAGED_ATTENTION_REDUCE_KERNEL(Name##_table16, Heads, Group, HeadDim, Window, Gated, true)
#define PAGED_ATTENTION_SINK_REDUCE(Name, Heads, Group, HeadDim, Window, Gated)                                \
  PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Gated, false)                        \
  PAGED_ATTENTION_SINK_REDUCE_KERNEL(Name##_table16, Heads, Group, HeadDim, Window, Gated, true)
PAGED_ATTENTION_REDUCE(attention_reduce, 4, 6, 256, 0, true)
PAGED_ATTENTION_REDUCE(attention_reduce_kv2_g8, 2, 8, 256, 0, true)
PAGED_ATTENTION_REDUCE(attention_reduce_kv2_g16, 2, 16, 256, 0, true)
// Flint's layers have no output gate.
PAGED_ATTENTION_SINK_REDUCE(attention_reduce_kv4_g8_d128_sinks, 4, 8, 128, 0, false)
PAGED_ATTENTION_SINK_REDUCE(attention_reduce_kv4_g8_d128_w512_sinks, 4, 8, 128, 512, false)
#undef PAGED_ATTENTION_SINK_REDUCE

// A relative geometry's reduce reads its rows' gates and sinks from their
// query side rows (F32, SplashQuerySideGates) where another's binds its
// packed rows.
#define PAGED_ATTENTION_RELATIVE_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, Table)                    \
  kernel void Name(                                                                                           \
      device const float *partials [[buffer(0)]],                                                             \
      device const float *statistics [[buffer(1)]],                                                           \
      device const float *query_side [[buffer(2)]],                                                           \
      device bfloat *hidden [[buffer(3)]],                                                                    \
      device bfloat *table [[buffer(4)]],                                                                     \
      device float *sums [[buffer(5)]],                                                                       \
      device const SplashLaneRows *layout [[buffer(6)]],                                                      \
      constant SplashAttentionParams &params [[buffer(7)]],                                                   \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]]) {                                                    \
    threadgroup float weights[SplashAttentionMaximumSplits];                                                  \
    threadgroup float group_values[HeadDim / 32];                                                             \
    threadgroup bfloat pairs[HeadDim];                                                                        \
    using Gates = SplashQuerySideGates<Heads * Group, HeadDim>;                                               \
    splash_attention_reduce_phase<Heads, Group, HeadDim, Window, Gates, Table>(                               \
        partials, statistics, query_side, hidden, table, sums, layout, params, group, thread_index, weights,  \
        group_values, pairs);                                                                                 \
  }
#define PAGED_ATTENTION_RELATIVE_REDUCE(Name, Heads, Group, HeadDim, Window)                                  \
  PAGED_ATTENTION_RELATIVE_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Window, false)                          \
  PAGED_ATTENTION_RELATIVE_REDUCE_KERNEL(Name##_table16, Heads, Group, HeadDim, Window, true)
PAGED_ATTENTION_RELATIVE_REDUCE(attention_reduce_kv8_g3_d128, 8, 3, 128, 0)
PAGED_ATTENTION_RELATIVE_REDUCE(attention_reduce_kv8_g3_d128_w1023, 8, 3, 128, 1023)
#undef PAGED_ATTENTION_RELATIVE_REDUCE
#undef PAGED_ATTENTION_RELATIVE_REDUCE_KERNEL
#undef PAGED_ATTENTION_REDUCE
#undef PAGED_ATTENTION_SINK_REDUCE_KERNEL
#undef PAGED_ATTENTION_REDUCE_KERNEL
