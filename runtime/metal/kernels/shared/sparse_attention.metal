#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/sparse_attention_tile.h"

// Sparse attention (metal/abi/PagedAttention.h): the block lists a tile's rows'
// block masks make, and the split and reduce entries that attend through
// them, for the GQA group of twelve over two KV heads on a KV layout that
// keeps values token-major (Qwen3.8-Flash-Next's, for one).

// Threadgroup per tile, group.x the tile and group.y the sequence
// (SplashAttentionBlockListParams), whose rows are its entry of the
// command's layout (`layout`): only the blocks of the tile's history are
// listed (the tail bit of a row whose token completes its block names the
// next block), and every thread takes a contiguous run of their words, so the
// entries keep block order. A list keeps to its stride whatever the masks
// hold; a tile of the sequence's bound past its rows, or of a layout entry
// past its bound or the command's rows, lists nothing.
constant constexpr uint kBlockListThreads = SPLASH_ATTENTION_BLOCK_LIST_THREADS;

// The bits of mask word `w` that name one of the first `blocks` blocks.
inline uint splash_history_word_bits(uint blocks, uint w) {
  const uint remaining = blocks - w * 32;
  return remaining >= 32 ? ~0u : (1u << remaining) - 1u;
}

kernel void attention_block_lists(device const uint *mask [[buffer(0)]],
                                  device uint *lists [[buffer(1)]],
                                  device uint *counts [[buffer(2)]],
                                  device const SplashLaneRows *layout [[buffer(3)]],
                                  constant SplashAttentionBlockListParams &p [[buffer(4)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint simd_counts[kBlockListThreads / 32];
  if (group.y >= min(p.sequences, uint(SPLASH_MAXIMUM_BATCH_WIDTH)))
    return;
  const SplashAttentionBlockListSequence s = p.sequence[group.y];
  const SplashLaneRows entry = layout[s.sequence];
  const uint sequence_rows =
      entry.rows <= s.row_bound && ulong(entry.first) + entry.rows <= p.rows ? entry.rows : 0;
  const uint tile = group.x, first_row = tile * p.tile_rows;
  if (first_row >= sequence_rows) {
    // A tile of its bound past its rows (a decode lane of fewer rows than
    // the step's lane cap) has an empty list, which the spilled K/V's
    // resolve reads with the others (model::KvBlockCache::addResolve).
    if (tid == 0 && first_row < s.row_bound)
      counts[s.list + tile] = 0;
    return;
  }
  const uint rows = min(p.tile_rows, sequence_rows - first_row);
  const uint blocks = splash_sparse_history_blocks(s.first_position + first_row, rows);
  const uint words = min((blocks + 31) / 32, p.mask_words);
  const uint per = (words + kBlockListThreads - 1) / kBlockListThreads;
  const uint w0 = min(tid * per, words), w1 = min(w0 + per, words);
  device const uint *bits = mask + (ulong(entry.first) + first_row) * p.mask_words;
  // A tile's rows, unrolled over an entry's row bits, so that each row's word
  // stays in a register; a short last tile's missing rows read as zero.
  constexpr uint RowBits = SPLASH_ATTENTION_BLOCK_ROW_BITS;
  uint attended = 0;
  for (uint w = w0; w < w1; ++w) {
    uint any = 0;
#pragma unroll
    for (uint r = 0; r < RowBits; ++r)
      any |= r < rows ? bits[ulong(r) * p.mask_words + w] : 0u;
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
    uint row_bits[RowBits];
    uint any = 0;
#pragma unroll
    for (uint r = 0; r < RowBits; ++r) {
      row_bits[r] = r < rows ? bits[ulong(r) * p.mask_words + w] : 0u;
      any |= row_bits[r];
    }
    any &= splash_history_word_bits(blocks, w);
    while (any) {
      const uint b = ctz(any);
      any &= any - 1;
      uint entry = (w * 32 + b) << SPLASH_ATTENTION_BLOCK_ROW_BITS;
#pragma unroll
      for (uint r = 0; r < RowBits; ++r)
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

// A tile's list, null without a stride, the locations of its entries where
// the split binds them (an array of the lists' shape: a list's locations sit
// where the list does), and its entries' count.
struct SplashSparseList {
  device const uint *entries;
  device const uint *locations;
  uint count;
};

inline SplashSparseList splash_sparse_list(device const uint *lists, device const uint *counts,
                                           uint stride, uint index, uint first_position,
                                           uint active_rows, device const uint *locations = nullptr) {
  const ulong offset = ulong(index) * stride;
  return {stride ? lists + offset : nullptr, locations ? locations + offset : nullptr,
          splash_sparse_list_count(counts, stride, index, first_position, active_rows)};
}

// Split: group.x is the KV head and split, KV head first (x = split * Heads
// + head), and group.y the dispatch's tile, whose sequence's tile t attends
// through list block_list + t; the reduce takes the KV head in group.x and
// the tile in group.z. A sparse sequence's list fills only the first of the
// dispatch's splits: an M5 Max spread this one-dimensional order better than
// a two-dimensional one (x the KV head, y the split), which ran up to 1.7x
// slower on most grids of more than 32 splits. The located split reads each
// entry's block where the list's locations name. Every split binds the
// step's tree rows after the page tables and the command's layout after
// them, as the dense split does, and after the lists' counts the row masks of
// a tree's rows that attend through them (ops::AttentionBlocks::rowMasks, a
// word per row of the command at the tree rows' places); the reduce binds the
// layout after its output.
#define SPARSE_SPLIT_BODY(Heads, Group, HeadDim, CacheElement, Located,       \
                          Locations, LocationPages, Places)                    \
    constexpr uint Rows = SPLASH_ATTENTION_TILE_ROWS;                          \
    threadgroup SplashSparseScratch<CacheElement, Rows * Group, HeadDim>       \
        scratch;                                                               \
    const uint s = splash_attention_sequence_of_tile(params, group.y);         \
    constant SplashAttentionSequence &sequence = params.sequence[s];           \
    const SplashAttentionRows rows = splash_attention_rows(layout, params, s); \
    const uint kv_head = group.x % Heads, split = group.x / Heads;             \
    const uint tile_index = group.y - sequence.first_tile;                     \
    const SplashAttentionTile tile =                                           \
        splash_attention_tile<Heads, Group, HeadDim>(params, sequence, rows,   \
                                                     kv_head, tile_index);     \
    if (!tile.active || split >= tile.splits)                                  \
      return;                                                                  \
    const SplashSparseList list = splash_sparse_list(                          \
        lists, counts, sequence.block_stride, sequence.block_list + tile_index, \
        tile.first_position, tile.active_rows, Locations);                     \
    splash_sparse_attention_tile<Heads, Group, Rows, HeadDim, CacheElement,    \
                                 Located>(                                     \
        queries + tile.offset,                                                 \
        SPLASH_LANE_BINDING(s, page_table0, page_table1, page_table2,          \
                            page_table3, page_table4, page_table5, page_table6, page_table7),                                      \
        params.kv, kv_head, tile.first_position, tile.active_rows,             \
        list.entries, list.count, list.locations, LocationPages, Places,       \
        tile.splits, split, partials, statistics, tile.first_slot + split,     \
        scratch, splash_attention_tree(params, sequence, rows, tree,           \
                                       tile.first_position),                   \
        params.trees && list.entries                                           \
            ? row_masks + rows.first +                                         \
                  (tile.first_position - sequence.committed_tokens)            \
            : nullptr,                                                         \
        thread_index);

#define SPARSE_SPLIT(Name, Heads, Group, HeadDim, CacheElement)                \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]], device const SplashKvPage *page_table4 [[buffer(7)]], device const SplashKvPage *page_table5 [[buffer(8)]], device const SplashKvPage *page_table6 [[buffer(9)]], device const SplashKvPage *page_table7 [[buffer(10)]],                    \
      device const SplashTreeRow *tree [[buffer(11)]],                          \
      device const SplashLaneRows *layout [[buffer(12)]],                       \
      device const uint *lists [[buffer(13)]],                                  \
      device const uint *counts [[buffer(14)]],                                \
      device const uint *row_masks [[buffer(15)]],                             \
      constant SplashAttentionParams &params [[buffer(16)]],                   \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    SPARSE_SPLIT_BODY(Heads, Group, HeadDim, CacheElement, false, nullptr,     \
                      nullptr, nullptr)                                        \
  }

// The located split binds the lists' locations and their pages after the
// row masks. The places of a step's blocks take threadgroup memory of their
// own.
#define SPARSE_LOCATED_SPLIT(Name, Heads, Group, HeadDim, CacheElement)        \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table0 [[buffer(3)]],                    \
      device const SplashKvPage *page_table1 [[buffer(4)]],                    \
      device const SplashKvPage *page_table2 [[buffer(5)]],                    \
      device const SplashKvPage *page_table3 [[buffer(6)]], device const SplashKvPage *page_table4 [[buffer(7)]], device const SplashKvPage *page_table5 [[buffer(8)]], device const SplashKvPage *page_table6 [[buffer(9)]], device const SplashKvPage *page_table7 [[buffer(10)]],                    \
      device const SplashTreeRow *tree [[buffer(11)]],                          \
      device const SplashLaneRows *layout [[buffer(12)]],                       \
      device const uint *lists [[buffer(13)]],                                  \
      device const uint *counts [[buffer(14)]],                                \
      device const uint *row_masks [[buffer(15)]],                             \
      device const uint *locations [[buffer(16)]],                             \
      device const SplashKvPage *location_pages [[buffer(17)]],                \
      constant SplashAttentionParams &params [[buffer(18)]],                   \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup uint places[SplashSparseStep<CacheElement>::Blocks];           \
    SPARSE_SPLIT_BODY(Heads, Group, HeadDim, CacheElement, true, locations,    \
                      location_pages, places)                                  \
  }
SPARSE_SPLIT(attention_q8_sparse_kv2_g12, 2, 12, 256, int8_t)
SPARSE_SPLIT(attention_bf16_sparse_kv2_g12, 2, 12, 256, bfloat)
SPARSE_LOCATED_SPLIT(attention_q8_sparse_located_kv2_g12, 2, 12, 256, int8_t)
SPARSE_LOCATED_SPLIT(attention_bf16_sparse_located_kv2_g12, 2, 12, 256, bfloat)
#undef SPARSE_LOCATED_SPLIT
#undef SPARSE_SPLIT
#undef SPARSE_SPLIT_BODY

// Combines the splits that took entries of the tile's list, and writes the
// row as the out-projection's input (splash_attention_output_row, gated by
// its query head's gate in its packed row), with Table into its Table16
// input too. Bindings: the partials and statistics, the packed rows, the
// out-projection's input rows, its table and the table's sums (other buffers
// of the dispatch without a table), the command's layout, the lists' counts,
// then the parameters.
#define SPARSE_REDUCE_KERNEL(Name, Heads, Group, HeadDim, Table)                 \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device const bfloat *packed_qkv [[buffer(2)]],                           \
      device bfloat *hidden [[buffer(3)]],                                     \
      device bfloat *table [[buffer(4)]],                                      \
      device float *sums [[buffer(5)]],                                        \
      device const SplashLaneRows *layout [[buffer(6)]],                       \
      device const uint *counts [[buffer(7)]],                                 \
      constant SplashAttentionParams &params [[buffer(8)]],                    \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = SPLASH_ATTENTION_TILE_ROWS;                          \
    threadgroup float weights[SplashAttentionMaximumSplits];                   \
    threadgroup float group_values[HeadDim / 32];                              \
    threadgroup bfloat pairs[HeadDim];                                         \
    const uint s = splash_attention_sequence_of_tile(params, group.z);         \
    constant SplashAttentionSequence &sequence = params.sequence[s];           \
    const uint fused_row = group.y, tile_index = group.z - sequence.first_tile; \
    const SplashAttentionRows rows = splash_attention_rows(layout, params, s); \
    const SplashAttentionTile tile = splash_attention_tile<Heads, Group,       \
        HeadDim>(params, sequence, rows, group.x, tile_index);                 \
    if (!tile.active || fused_row >= Rows * Group || thread_index >= HeadDim)  \
      return;                                                                  \
    const uint count = splash_sparse_list_count(                               \
        counts, sequence.block_stride, sequence.block_list + tile_index,       \
        tile.first_position, tile.active_rows);                                \
    const bfloat value = splash_attention_combine_splits<Group, Rows, HeadDim>( \
        partials, statistics, tile.active_rows,                                \
        splash_sparse_written_splits(count, tile.splits), tile.first_slot,     \
        fused_row, thread_index, weights, group_values);                       \
    const SplashAttentionOutput place = splash_attention_output(               \
        params, s, rows,                                                       \
        tile.first_position - sequence.committed_tokens + fused_row / Group);  \
    splash_attention_output_row<Heads * Group, Heads, HeadDim, true, Table>(   \
        value, packed_qkv, hidden, table, sums, place.packed, place.output,    \
        group.x * Group + fused_row % Group, thread_index, place.writes,       \
        pairs);                                                                \
  }
#define SPARSE_REDUCE(Name, Heads, Group, HeadDim)                             \
  SPARSE_REDUCE_KERNEL(Name, Heads, Group, HeadDim, false)                     \
  SPARSE_REDUCE_KERNEL(Name##_table16, Heads, Group, HeadDim, true)
SPARSE_REDUCE(attention_sparse_reduce_kv2_g12, 2, 12, 256)
#undef SPARSE_REDUCE
#undef SPARSE_REDUCE_KERNEL
