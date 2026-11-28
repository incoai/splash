#pragma clang fp reassociate(off)
#include "metal/kernels/common/kv_path_commit.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/relative_attention_tile.h"
#include "metal/kernels/common/split_reduce.h"

// Relative attention (ops::RelativeAttention, kernels/common/
// relative_attention_tile.h): its prepare, then the page loop over the tiles
// of a dispatch's sequences and the reduce, laid out and bound as the paged
// attention's (kernels/shared/paged_attention.metal), or, a dispatch whose
// tiles each read their history in one split, the direct kernels, which
// write their tiles' rows themselves (Direct below); each tile
// SPLASH_ATTENTION_TILE_ROWS query rows of one KV head (a direct kernel's
// its GPU family's rows), every sequence of a command in one
// dispatch, its rows its entry of the command's layout. The geometry is
// Inkling-2-Mini's: 24 query heads over 8 KV heads of 128 dimensions, three a
// group, a tile one part of 24 fused rows. Every kernel here computes in its
// source's order (no reassociation, the pragma above): the split, the direct
// kernels and a decode step's single pass share the page loop, and the direct
// kernels' and the single pass's combines take the reduce's order, so that
// they write the prepare, split and reduce's bits, where with reassociation
// the compiler arranged the page loop of INT8 queries in the split and the
// single pass otherwise.

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
  constexpr uint M = QueryHeadsPerKVHead * SPLASH_ATTENTION_TILE_ROWS;
  const uint split = group.z;
  const uint kv_head = group.x;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const SplashAttentionTile tile = splash_attention_tile<KVHeads, QueryHeadsPerKVHead, HeadDim>(
      params, sequence, rows, kv_head, group.y - sequence.first_tile);
  if (!tile.active || split >= tile.splits)
    return;
  const RelativeAttentionPages run = relative_attention_pages<Window>(
      tile.first_position, tile.active_rows, params.trees != 0, sequence.committed_tokens, tile.splits, split);
  if (run.begin >= run.end)
    return;
  // The INT8 query codes live in the threadgroup's own partials slot, which
  // it writes only after its last page.
  const ulong slot = tile.first_slot + split;
  device float *own = partials + slot * M * HeadDim;
  relative_attention_page_loop<KVHeads, QueryHeadsPerKVHead, SPLASH_ATTENTION_TILE_ROWS, HeadDim, Window,
                               CacheElement, Int8Queries, true>(
      queries + tile.offset, reinterpret_cast<device int8_t *>(own),
      relative_states + tile.offset / HeadDim * SPLASH_RELATIVE_ATTENTION_STATES, relative_bias, page_table,
      params.kv, kv_head, tile.first_position, tile.active_rows, run, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, splash_attention_tree(params, sequence, rows, tree, tile.first_position),
      thread_index,
      [&](thread auto &running, bool) {
        auto target = tensor(own, dextents<int, 2>{HeadDim, M}, array<int, 2>{1, HeadDim});
        running.store(target.slice<HeadDim, M>(0, 0));
        if (thread_index < M) {
          statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
          statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
        }
      });
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

// Direct (ops::RelativeAttention::addAttention, a dispatch whose sequences
// each read their history in one split): a threadgroup of 256 threads per (KV
// head, tile), each tile Rows rows of one KV head (its GPU family's), a
// sequence's tiles the dispatch's from its first_tile on in tiles of that
// many rows. A tile runs the split's page loop over its one run of pages,
// those of its first row's window to its last row's: the union of the runs of
// its two paged-attention tiles of the same rows, whose other pages none of a
// row's keys sit on, so that their probabilities are zero and leave the row's
// statistics and output as they were. It then writes its rows' output as the
// reduce does from that one split: a thread a fused row its weight and
// denominator, each thread the gated outputs of the running output's elements
// it holds. It writes no out-projection table: its chunks, of more than 128
// rows, run the out-projection's prefill tile, which reads plain rows (the
// table, its sums and the simdgroup indices it is given, as the reduce is,
// go unused). The INT8 query codes take the slot of partials at the tile's
// first_slot plus its tile and KV head. Measured on a 2,048-row chunk's layer
// (20-core M5 Pro): the reduce and the partials it reads took a sliding
// layer's attention 10% of its time after 2,048 tokens (4.55 ms against 4.10
// without) and a full layer's 1.7% after 8,192; a tile of twice the rows,
// which reads each page once for both, ran 5-11% faster than two tiles at
// short histories (0 to 2,048 tokens) and 0.6-4% at 8,192 and 30,720. Its
// tiles take the family's rows (ops::AttentionCalibration::directRows: 21,
// whose 63 fused rows it multiplies as 64, or 16) and rescale every row: each
// family's direct tiles are taller than the tiles it bounds.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint Rows, uint HeadDim, uint Window, typename CacheElement,
          bool Int8Queries>
inline void relative_attention_direct_phase(
    device bfloat *queries, device float *partials, device const SplashKvPage *page_table,
    device const SplashTreeRow *tree, device const SplashLaneRows *layout, device const float *relative_states,
    device const float *relative_bias, device const bfloat *projection, device bfloat *hidden, device bfloat *,
    device float *, device const float *sinks, constant SplashAttentionParams &params, threadgroup float *scores,
    threadgroup bfloat *probabilities, threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group, uint thread_index, uint, uint,
    uint s) {
  constexpr uint M = QueryHeadsPerKVHead * Rows, QHeads = KVHeads * QueryHeadsPerKVHead;
  constexpr uint Width = QHeads * HeadDim;
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  const uint kv_head = group.x;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const uint tile = group.y - sequence.first_tile, start = tile * Rows;
  if (!splash_attention_contract_valid(params, sequence) || sequence.split_count != 1 || kv_head >= KVHeads ||
      start >= rows.rows)
    return;
  const uint first_position = sequence.committed_tokens + start, active_rows = min(Rows, rows.rows - start);
  const ulong offset =
      (ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + start) * QueryHeadsPerKVHead * HeadDim;
  const RelativeAttentionPages run =
      relative_attention_pages<Window>(first_position, active_rows, params.trees != 0, sequence.committed_tokens, 1, 0);
  device float *codes = partials + (ulong(sequence.first_slot) + ulong(tile) * KVHeads + kv_head) *
                                       (SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead) * HeadDim;
  static_assert(M * HeadDim <= SPLASH_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead * HeadDim * sizeof(float),
                "a tile's query codes fit a slot of the paged attention's partials");
  relative_attention_page_loop<KVHeads, QueryHeadsPerKVHead, Rows, HeadDim, Window, CacheElement, Int8Queries,
                               false>(
      queries + offset, reinterpret_cast<device int8_t *>(codes),
      relative_states + offset / HeadDim * SPLASH_RELATIVE_ATTENTION_STATES, relative_bias, page_table, params.kv,
      kv_head, first_position, active_rows, run, scores, probabilities, row_max, row_sum, previous_scale, rescale,
      splash_attention_tree(params, sequence, rows, tree, first_position), thread_index,
      [&](thread auto &running, bool running_full) {
        // The combine of the one split (relative_attention_tile.h's
        // Reduce): each active fused row's weight into previous_scale and
        // its denominator into row_sum, from its statistics and its sink,
        // the split's product rounded before the sink's term is added
        // (contract(off)), as the reduce's loop over splits rounds a row's
        // one product: contracted into one fused multiply-add, the two gave
        // other bits.
        if (thread_index < M && thread_index / QueryHeadsPerKVHead < active_rows) {
#pragma clang fp contract(off)
          const float sink = sinks[offset / HeadDim + thread_index];
          const float maximum = row_max[thread_index];
          const float row_maximum = max(maximum, sink);
          const float weight = relative_attention_split_weight(maximum, row_maximum);
          const float denominator =
              relative_attention_denominator(weight * row_sum[thread_index], sink, row_maximum);
          previous_scale[thread_index] = weight;
          row_sum[thread_index] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort index = 0; index < running.get_capacity(); ++index) {
          if (!running_full && !running.is_valid_element(index))
            continue;
          const auto coordinates = running.get_multidimensional_index(index);
          const uint fused_row = uint(coordinates[1]), dimension = uint(coordinates[0]);
          const uint row = fused_row / QueryHeadsPerKVHead;
          if (row >= active_rows)
            continue;
          const SplashAttentionOutput place = splash_attention_output(params, s, rows, start + row);
          if (!place.writes)
            continue;
          float numerator = 0.0f;
          numerator += previous_scale[fused_row] * running[index];
          const float denominator = row_sum[fused_row];
          const bfloat value = bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
          const uint query_head = kv_head * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead;
          const float gate =
              float(projection[place.packed * Row::Stride + Row::GateOffset + query_head * HeadDim + dimension]);
          hidden[place.output * Width + query_head * HeadDim + dimension] =
              relative_attention_gated(float(value), gate);
        }
      });
}

// The split's bindings without its statistics, then the plain reduce's: the
// projection rows (their gates), the out-projection's input rows, a table
// and its sums, which it does not write (other buffers of the dispatch), and
// the rows' sinks' staging, before the parameters. Over the whole history and
// a sliding layer's window, for INT8 pages with bf16 or INT8 queries and BF16
// pages; for tiles of SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS rows and, _r16,
// of SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS_R16.
#define RELATIVE_ATTENTION_DIRECT_KERNEL(Name, Rows, Window, CacheElement, Int8Queries)                          \
  kernel void Name(                                                                                           \
      device bfloat *queries [[buffer(0)]],                                                                   \
      device float *partials [[buffer(1)]],                                                                   \
      device const SplashKvPage *page_table0 [[buffer(2)]],                                                   \
      device const SplashKvPage *page_table1 [[buffer(3)]],                                                   \
      device const SplashKvPage *page_table2 [[buffer(4)]],                                                   \
      device const SplashKvPage *page_table3 [[buffer(5)]],                                                   \
      device const SplashKvPage *page_table4 [[buffer(6)]],                                                   \
      device const SplashKvPage *page_table5 [[buffer(7)]],                                                   \
      device const SplashKvPage *page_table6 [[buffer(8)]],                                                   \
      device const SplashKvPage *page_table7 [[buffer(9)]],                                                   \
      device const SplashTreeRow *tree [[buffer(10)]],                                                        \
      device const SplashLaneRows *layout [[buffer(11)]],                                                     \
      device const float *relative_states [[buffer(12)]],                                                     \
      device const float *relative_bias [[buffer(13)]],                                                       \
      device const bfloat *projection [[buffer(14)]],                                                         \
      device bfloat *hidden [[buffer(15)]],                                                                   \
      device bfloat *table [[buffer(16)]],                                                                    \
      device float *sums [[buffer(17)]],                                                                      \
      device const float *sinks [[buffer(18)]],                                                               \
      constant SplashAttentionParams &params [[buffer(19)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]],                                                      \
      uint lane [[thread_index_in_simdgroup]],                                                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                   \
    constexpr uint M = relative_attention_matmul_rows(3 * Rows);                                              \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    const uint s = splash_attention_sequence_of_tile(params, group.y);                                        \
    relative_attention_direct_phase<8, 3, Rows, 128, Window, CacheElement, Int8Queries>(                      \
        queries, partials,                                                                                    \
        SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5,  \
                            page_table6, page_table7),                                                        \
        tree, layout, relative_states, relative_bias, projection, hidden, table, sums, sinks, params, scores,  \
        probabilities, row_max, row_sum, previous_scale, &rescale, group, thread_index, lane, simd_group, s);   \
  }
#define RELATIVE_ATTENTION_DIRECTS(Suffix, Rows, Window)                                                      \
  RELATIVE_ATTENTION_DIRECT_KERNEL(relative_attention_q8_direct##Suffix, Rows, Window, int8_t, false)         \
  RELATIVE_ATTENTION_DIRECT_KERNEL(relative_attention_q8q_direct##Suffix, Rows, Window, int8_t, true)         \
  RELATIVE_ATTENTION_DIRECT_KERNEL(relative_attention_bf16_direct##Suffix, Rows, Window, bfloat, false)
RELATIVE_ATTENTION_DIRECTS(, SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS, 0)
RELATIVE_ATTENTION_DIRECTS(_w1023, SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS, 1023)
RELATIVE_ATTENTION_DIRECTS(_r16, SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS_R16, 0)
RELATIVE_ATTENTION_DIRECTS(_r16_w1023, SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS_R16, 1023)
#undef RELATIVE_ATTENTION_DIRECTS
#undef RELATIVE_ATTENTION_DIRECT_KERNEL

// Single pass (ops::RelativeAttention::addDecode): a decode step's prepare,
// split and reduce in one dispatch, a threadgroup of 256 threads per (KV
// head, dispatch sequence, history split), each sequence a lane's block of at
// most RowsPerTile rows run as one tile (relative_attention_tile.h, Single
// pass). A split's threadgroup stages its tile's queries and relative states
// in a staging slot of its own, a simdgroup a (row, query head), and stores
// the K/V of the block's rows that sit on its pages, a simdgroup a row, before
// its page loop reads them; no other threadgroup reads those pages' slots of
// them. It then runs the split's page loop and publishes its partials and
// statistics of the tile's rows, and the split that arrives last at its
// tile's counter (kernels/common/split_reduce.h; zero before and after)
// combines them with each row's sink into the gated output rows, its weights
// a thread a fused row and then its numerators a simdgroup a fused row
// (relative_attention_combine_weights, _row). A sequence's slots, from its
// first_slot in tiles of SPLASH_ATTENTION_TILE_ROWS rows (RowsPerTile / tile
// rows of them a slot), hold each KV head's splits' partials, then each KV
// head's splits' staging: the INT8 query codes, the relative states and the
// bf16 queries, in that order.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim, uint Window, bool PositionScaled,
          typename CacheElement, bool Int8Queries, bool Table, bool BoundedRescale>
inline void relative_attention_decode_phase(
    device const bfloat *projection, device const bfloat *keys, device const bfloat *values,
    device const float *query_scale, device const float *relative_scale, device const float *relative_bias,
    device float *partials, device float *statistics, device atomic_uint *counters, device bfloat *hidden,
    device bfloat *table, device float *sums, device const SplashKvPage *page_table, device const SplashTreeRow *tree,
    device const SplashLaneRows *layout, constant SplashAttentionParams &params, threadgroup float *scores,
    threadgroup bfloat *probabilities, threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, threadgroup uint *arrival, uint3 group,
    uint thread_index, uint lane, uint simd_group) {
  using Row = RelativeAttentionRow<KVHeads * QueryHeadsPerKVHead, HeadDim>;
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead, D = HeadDim, States = Row::States;
  constexpr uint Simdgroups = 8;
  // A lane of up to a paged tile's rows takes one of its slots, a verify
  // block's two.
  constexpr uint Tiles = (RowsPerTile + SPLASH_ATTENTION_TILE_ROWS - 1) / SPLASH_ATTENTION_TILE_ROWS;
  static_assert((RowsPerTile <= SPLASH_ATTENTION_TILE_ROWS || Tiles * SPLASH_ATTENTION_TILE_ROWS == RowsPerTile) &&
                    M <= 64,
                "a slot's rows or whole slots, of a 256-thread loop");
  const uint kv_head = group.x, s = group.y, split = group.z;
  constant SplashAttentionSequence &sequence = params.sequence[s];
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  if (!splash_attention_contract_valid(params, sequence) || sequence.row_bound > RowsPerTile ||
      sequence.split_count > SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS || kv_head >= KVHeads || !rows.rows ||
      split >= sequence.split_count)
    return;
  const uint first = sequence.committed_tokens, fused_rows = rows.rows * QueryHeadsPerKVHead;
  const bool in_tree = params.trees != 0;
  const RelativeAttentionPages run =
      relative_attention_pages<Window>(first, rows.rows, in_tree, first, sequence.split_count, split);
  if (run.begin >= run.end)
    return;
  const ulong head_slot = ulong(sequence.first_slot) / Tiles + ulong(kv_head) * sequence.split_count;
  device float *staging =
      partials + (ulong(sequence.first_slot) / Tiles + ulong(KVHeads + kv_head) * sequence.split_count + split) * M * D;
  device float *states = staging + M * D / 4;
  device bfloat *queries = reinterpret_cast<device bfloat *>(staging) + M * D;
  for (uint fused = simd_group; fused < fused_rows; fused += Simdgroups) {
    const uint row = fused / QueryHeadsPerKVHead;
    const ulong packed = ulong(rows.first) + row;
    relative_attention_stage_head<KVHeads * QueryHeadsPerKVHead, D, PositionScaled>(
        projection + packed * Row::Stride, query_scale, relative_scale,
        kv_head * QueryHeadsPerKVHead + fused % QueryHeadsPerKVHead,
        first + (in_tree ? uint(tree[packed].depth) : row), queries + fused * D, states + fused * States, lane);
  }
  for (uint row = simd_group; row < rows.rows; row += Simdgroups) {
    const uint page = (first + row) / SplashKvPageTokens;
    if (page >= run.begin && page < run.end)
      relative_attention_store_head<KVHeads, D, CacheElement>(
          keys, values, (ulong(rows.first) + row) * (KVHeads * D) + kv_head * D, page_table[page], params.kv,
          kv_head, (first + row) % SplashKvPageTokens, lane);
  }
  threadgroup_barrier(mem_flags::mem_device);
  // Publish the tile's rows' partials and statistics, then arrive.
  const ulong slot = head_slot + split;
  relative_attention_page_loop<KVHeads, QueryHeadsPerKVHead, RowsPerTile, D, Window, CacheElement, Int8Queries,
                               BoundedRescale>(
      queries, reinterpret_cast<device int8_t *>(staging), states, relative_bias, page_table, params.kv, kv_head,
      first, rows.rows, run, scores, probabilities, row_max, row_sum, previous_scale, rescale,
      splash_attention_tree(params, sequence, rows, tree, first), thread_index,
      [&](thread auto &running, bool running_full) {
        device coherent(device) float *published = partials + slot * M * D;
#pragma unroll
        for (ushort index = 0; index < running.get_capacity(); ++index) {
          if (!running_full && !running.is_valid_element(index))
            continue;
          const auto coordinates = running.get_multidimensional_index(index);
          if (uint(coordinates[1]) < fused_rows)
            published[uint(coordinates[1]) * D + uint(coordinates[0])] = running[index];
        }
        if (thread_index < fused_rows) {
          device coherent(device) float *row_statistics = statistics + (slot * M + thread_index) * 2;
          row_statistics[0] = row_max[thread_index];
          row_statistics[1] = row_sum[thread_index];
        }
      });
  device atomic_uint *counter = counters + (params.first_sequence + s) * KVHeads + kv_head;
  if (!split_arrive_last(counter, run.written, thread_index, arrival))
    return;
  // The tile's statistics are published: the combine's weights reuse the
  // scores' storage (fused rows x splits, at most M x SplashKvPageTokens)
  // and its denominators the row sums'.
  static_assert(SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS <= SplashKvPageTokens, "weights in the scores' storage");
  if (thread_index < fused_rows) {
    const uint query_head = kv_head * QueryHeadsPerKVHead + thread_index % QueryHeadsPerKVHead;
    const SplashAttentionOutput place = splash_attention_output(params, s, rows, thread_index / QueryHeadsPerKVHead);
    relative_attention_combine_weights<QueryHeadsPerKVHead, RowsPerTile, D>(
        statistics, run.written, head_slot, thread_index,
        float(projection[place.packed * Row::Stride + Row::SinkOffset + query_head]), scores, row_sum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint fused = simd_group; fused < fused_rows; fused += Simdgroups) {
    const uint query_head = kv_head * QueryHeadsPerKVHead + fused % QueryHeadsPerKVHead;
    relative_attention_output_quad<KVHeads * QueryHeadsPerKVHead, D, Table>(
        relative_attention_combine_row<QueryHeadsPerKVHead, RowsPerTile, D>(partials, run.written, head_slot, fused,
                                                                           4 * lane, scores, row_sum[fused]),
        projection, hidden, table, sums, splash_attention_output(params, s, rows, fused / QueryHeadsPerKVHead),
        query_head, 4 * lane, lane);
  }
  split_release(counter, thread_index);
}

// Bindings: the projection rows, the convolved K and V rows, the learned
// query and relative-state scales, the layer's bias table, the partials and
// statistics, the tiles' counters, the out-projection's input rows, its table
// and the table's sums (other buffers of the dispatch without a table), the
// page tables, the command's tree rows (another buffer of the dispatch
// without), its layout and the parameters. Each is compiled for blocks of up
// to 2 rows (_r2, 6 fused rows multiplied as 8), 5 (_r5, 15 as 16), 8 and,
// _r16, a verify block's 16, a narrow cap's lanes the smallest that holds
// their rows (RelativeAttention::plan), whose tiles bound their rescale
// or, _r16_unbounded, rescale every row (the family's choice for the shape,
// ops::AttentionCalibration::boundedRescaleRows); for the layers' two variants
// alone, a full layer's (the whole history, position-scaled: _scaled) and a
// sliding one's (windowed, unscaled: _w1023); plain and writing the
// out-projection's Table16 input too (_table16); for INT8 pages with bf16 or
// INT8 queries, and BF16 pages.
#define RELATIVE_ATTENTION_DECODE_KERNEL(Name, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries, Table, \
                                         BoundedRescale)                                                      \
  kernel void Name(                                                                                           \
      device const bfloat *projection [[buffer(0)]],                                                          \
      device const bfloat *keys [[buffer(1)]],                                                                \
      device const bfloat *values [[buffer(2)]],                                                              \
      device const float *query_scale [[buffer(3)]],                                                          \
      device const float *relative_scale [[buffer(4)]],                                                       \
      device const float *relative_bias [[buffer(5)]],                                                        \
      device float *partials [[buffer(6)]],                                                                   \
      device float *statistics [[buffer(7)]],                                                                 \
      device atomic_uint *counters [[buffer(8)]],                                                             \
      device bfloat *hidden [[buffer(9)]],                                                                    \
      device bfloat *table [[buffer(10)]],                                                                    \
      device float *sums [[buffer(11)]],                                                                      \
      device const SplashKvPage *page_table0 [[buffer(12)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(13)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(14)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(15)]],                                                  \
      device const SplashKvPage *page_table4 [[buffer(16)]],                                                  \
      device const SplashKvPage *page_table5 [[buffer(17)]],                                                  \
      device const SplashKvPage *page_table6 [[buffer(18)]],                                                  \
      device const SplashKvPage *page_table7 [[buffer(19)]],                                                  \
      device const SplashTreeRow *tree [[buffer(20)]],                                                        \
      device const SplashLaneRows *layout [[buffer(21)]],                                                     \
      constant SplashAttentionParams &params [[buffer(22)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                           \
      uint thread_index [[thread_index_in_threadgroup]],                                                      \
      uint lane [[thread_index_in_simdgroup]],                                                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                   \
    constexpr uint M = relative_attention_matmul_rows(3 * RowsPerTile);                                       \
    constexpr uint N = SplashKvPageTokens;                                                                    \
    alignas(16) threadgroup float scores[M * N];                                                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                                                      \
    threadgroup float row_max[M];                                                                             \
    threadgroup float row_sum[M];                                                                             \
    threadgroup float previous_scale[M];                                                                      \
    threadgroup atomic_uint rescale;                                                                          \
    threadgroup uint arrival;                                                                                 \
    relative_attention_decode_phase<8, 3, RowsPerTile, 128, Window, PositionScaled, CacheElement, Int8Queries,  \
                                    Table, BoundedRescale>(                                                   \
        projection, keys, values, query_scale, relative_scale, relative_bias, partials, statistics, counters,  \
        hidden, table, sums,                                                                                  \
        SPLASH_LANE_BINDING(group.y, page_table0, page_table1, page_table2, page_table3, page_table4,          \
                            page_table5, page_table6, page_table7),                                           \
        tree, layout, params, scores, probabilities, row_max, row_sum, previous_scale, &rescale, &arrival,     \
        group, thread_index, lane, simd_group);                                                               \
  }
#define RELATIVE_ATTENTION_DECODE_OUTPUTS(Name, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries,   \
                                          BoundedRescale)                                                     \
  RELATIVE_ATTENTION_DECODE_KERNEL(Name, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries, false, \
                                   BoundedRescale)                                                            \
  RELATIVE_ATTENTION_DECODE_KERNEL(Name##_table16, RowsPerTile, Window, PositionScaled, CacheElement, Int8Queries, \
                                   true, BoundedRescale)
#define RELATIVE_ATTENTION_DECODE_FORMATS(Suffix, RowsPerTile, Window, PositionScaled, BoundedRescale)         \
  RELATIVE_ATTENTION_DECODE_OUTPUTS(relative_attention_q8_decode##Suffix, RowsPerTile, Window, PositionScaled,  \
                                    int8_t, false, BoundedRescale)                                            \
  RELATIVE_ATTENTION_DECODE_OUTPUTS(relative_attention_q8q_decode##Suffix, RowsPerTile, Window, PositionScaled, \
                                    int8_t, true, BoundedRescale)                                             \
  RELATIVE_ATTENTION_DECODE_OUTPUTS(relative_attention_bf16_decode##Suffix, RowsPerTile, Window, PositionScaled, \
                                    bfloat, false, BoundedRescale)
#define RELATIVE_ATTENTION_DECODE_VARIANTS(Suffix, RowsPerTile, BoundedRescale)                                \
  RELATIVE_ATTENTION_DECODE_FORMATS(Suffix##_scaled, RowsPerTile, 0, true, BoundedRescale)                     \
  RELATIVE_ATTENTION_DECODE_FORMATS(Suffix##_w1023, RowsPerTile, 1023, false, BoundedRescale)
RELATIVE_ATTENTION_DECODE_VARIANTS(_r2, 2, true)
RELATIVE_ATTENTION_DECODE_VARIANTS(_r5, 5, true)
RELATIVE_ATTENTION_DECODE_VARIANTS(, 8, true)
RELATIVE_ATTENTION_DECODE_VARIANTS(_r16, 16, true)
RELATIVE_ATTENTION_DECODE_VARIANTS(_r16_unbounded, 16, false)
#undef RELATIVE_ATTENTION_DECODE_VARIANTS
#undef RELATIVE_ATTENTION_DECODE_FORMATS
#undef RELATIVE_ATTENTION_DECODE_OUTPUTS
#undef RELATIVE_ATTENTION_DECODE_KERNEL

// The path commit (kernels/common/kv_path_commit.h) of the pages relative
// attention stores: 8 KV heads of 128 dimensions, values dimension-major
// (ops::RelativeAttention::addPathCommit).
PAGED_ATTENTION_PATH_COMMIT(relative_attention_q8_commit, 8, 128, int8_t, false)
PAGED_ATTENTION_PATH_COMMIT(relative_attention_bf16_commit, 8, 128, bfloat, false)
#undef PAGED_ATTENTION_PATH_COMMIT
